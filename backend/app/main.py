from __future__ import annotations

import csv
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field


# ========================
# CONFIG
# ========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
DEFAULT_MODEL_URI = os.getenv(
    "DEFAULT_MODEL_URI",
    "models:/apartment-price-regression@champion",
)

PREDICTION_LOG_PATH = Path(
    os.getenv("PREDICTION_LOG_PATH", "/app/data/predictions/predictions_log.csv")
)

DRIFT_METRICS_PATH = Path(
    os.getenv("DRIFT_METRICS_PATH", "/app/data/monitoring/latest_drift_metrics.json")
)

LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() in {"1", "true", "yes", "y"}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ========================
# MARKET DEFAULTS
# ========================

# Usado apenas quando o payload não envia base_price_m2.
# Mantém compatibilidade com o frontend antigo, que ainda envia city.
CAPITAL_BASE_PRICE_M2 = {
    "Maceió": 9908.0,
    "Maceio": 9908.0,
    "Manaus": 7513.0,
    "Salvador": 8385.0,
    "Fortaleza": 9350.0,
    "Brasília": 10090.0,
    "Brasilia": 10090.0,
    "Vitória": 14818.0,
    "Vitoria": 14818.0,
    "Goiânia": 8226.0,
    "Goiania": 8226.0,
    "São Luís": 8627.0,
    "Sao Luis": 8627.0,
    "Cuiabá": 6931.0,
    "Cuiaba": 6931.0,
    "Campo Grande": 6839.0,
    "Belo Horizonte": 10663.0,
    "Belém": 8882.0,
    "Belem": 8882.0,
    "João Pessoa": 8081.0,
    "Joao Pessoa": 8081.0,
    "Curitiba": 11694.0,
    "Recife": 8615.0,
    "Teresina": 5857.0,
    "Rio de Janeiro": 10939.0,
    "Natal": 6334.0,
    "Porto Alegre": 7579.0,
    "Florianópolis": 13208.0,
    "Florianopolis": 13208.0,
    "São Paulo": 12019.0,
    "Sao Paulo": 12019.0,
    "Aracaju": 5529.0,
    "Rio Branco": 5200.0,
    "Macapá": 5400.0,
    "Macapa": 5400.0,
    "Porto Velho": 5600.0,
    "Boa Vista": 5300.0,
    "Palmas": 5900.0,
}

DEFAULT_CITY = "São Paulo"
DEFAULT_PROPERTY_TYPE = "apartment"
DEFAULT_NEIGHBORHOOD = "Região Residencial"
DEFAULT_BASE_PRICE_M2 = 9769.0

POTENTIAL_INPUT_COLUMNS = [
    "property_type",
    "neighborhood",
    "base_price_m2",
    "city",
    "state",
    "region",
    "price_source",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floor",
    "parking_spaces",
    "neighborhood_score",
    "condo_fee",
    "age_years",
    "distance_to_center_km",
]

NUMERIC_REQUIRED_COLUMNS = [
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floor",
    "parking_spaces",
    "neighborhood_score",
    "condo_fee",
    "age_years",
    "distance_to_center_km",
]

MODEL_NUMERIC_COLUMNS = [
    "base_price_m2",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "floor",
    "parking_spaces",
    "neighborhood_score",
    "condo_fee",
    "age_years",
    "distance_to_center_km",
]


# ========================
# PROMETHEUS METRICS
# ========================

MODEL_PREDICTION_TOTAL = Counter(
    "model_prediction_total",
    "Total de predicoes executadas pelo modelo.",
    ["model_name", "model_version", "model_alias", "endpoint", "status"],
)

MODEL_PREDICTION_LATENCY_SECONDS = Histogram(
    "model_prediction_latency_seconds",
    "Tempo de inferencia do modelo em segundos.",
    ["model_name", "model_version", "model_alias", "endpoint"],
)

MODEL_PREDICTION_BATCH_SIZE = Histogram(
    "model_prediction_batch_size",
    "Quantidade de registros enviados por requisicao de inferencia.",
    ["model_name", "model_version", "model_alias", "endpoint"],
)

MODEL_LAST_PREDICTION_VALUE = Gauge(
    "model_last_prediction_value",
    "Ultimo valor previsto pelo modelo.",
    ["model_name", "model_version", "model_alias", "endpoint"],
)

MODEL_DATA_DRIFT_SCORE = Gauge(
    "model_data_drift_score",
    "Score consolidado de data drift calculado pelo Evidently.",
    ["model_name", "model_version", "model_alias"],
)

MODEL_PREDICTION_DRIFT_SCORE = Gauge(
    "model_prediction_drift_score",
    "Score consolidado de prediction drift calculado pelo Evidently.",
    ["model_name", "model_version", "model_alias"],
)

MODEL_DRIFT_DETECTED = Gauge(
    "model_drift_detected",
    "Indicador binario de drift detectado pelo Evidently: 1 para drift, 0 para sem drift.",
    ["model_name", "model_version", "model_alias"],
)


# ========================
# API SCHEMAS
# ========================

class ApartmentFeatures(BaseModel):
    # Novo schema do modelo candidate v9.
    property_type: str = Field(default=DEFAULT_PROPERTY_TYPE)
    neighborhood: str = Field(default=DEFAULT_NEIGHBORHOOD)
    base_price_m2: Optional[float] = Field(default=None)

    # Campos de auditoria e compatibilidade.
    # city também permite derivar base_price_m2 quando o frontend ainda não envia essa feature.
    city: Optional[str] = Field(default=DEFAULT_CITY)
    state: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    price_source: Optional[str] = Field(default=None)

    # Campos numéricos principais.
    area_m2: float
    bedrooms: float
    bathrooms: float
    floor: float
    parking_spaces: float
    neighborhood_score: float
    condo_fee: float
    age_years: float
    distance_to_center_km: float


class PredictRequest(BaseModel):
    dataframe_records: list[ApartmentFeatures]


class PredictResponse(BaseModel):
    predictions: list[float]
    model_uri: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_alias: Optional[str] = None


class ModelLoadRequest(BaseModel):
    model_name: Optional[str] = Field(default="apartment-price-regression")
    model_uri: str = Field(default="models:/apartment-price-regression@champion")
    model_version: Optional[str] = None
    source_type: Optional[str] = Field(default="mlflow-registry")


class ModelLoadResponse(BaseModel):
    status: str
    model_name: Optional[str]
    model_uri: str
    model_version: Optional[str]
    model_alias: Optional[str] = None
    model_source: Optional[str] = None
    input_columns: list[str] = Field(default_factory=list)


class ModelReloadResponse(BaseModel):
    status: str
    model_name: Optional[str]
    model_uri: str
    model_version: Optional[str]
    model_alias: Optional[str] = None
    model_source: Optional[str] = None
    input_columns: list[str] = Field(default_factory=list)


class DriftRefreshResponse(BaseModel):
    status: str
    drift_metrics_path: str
    data_drift_score: Optional[float] = None
    prediction_drift_score: Optional[float] = None
    drift_detected: Optional[bool] = None


# ========================
# UTILS
# ========================

def resolve_model_registry_info(model_uri: str) -> dict[str, Optional[str]]:
    client = MlflowClient()

    info = {
        "registered_model_name": None,
        "model_version": None,
        "model_alias": None,
        "model_source": None,
    }

    if model_uri.startswith("models:/") and "@" in model_uri:
        model_ref = model_uri.replace("models:/", "")
        model_name, alias = model_ref.split("@", 1)
        version = client.get_model_version_by_alias(model_name, alias)

        info.update(
            {
                "registered_model_name": model_name,
                "model_version": version.version,
                "model_alias": alias,
                "model_source": version.source,
            }
        )

    elif model_uri.startswith("models:/"):
        parts = model_uri.replace("models:/", "").split("/")

        if len(parts) == 2:
            model_name, version_number = parts
            version = client.get_model_version(model_name, version_number)

            info.update(
                {
                    "registered_model_name": model_name,
                    "model_version": version.version,
                    "model_alias": None,
                    "model_source": version.source,
                }
            )

    return info


def get_model_input_columns(model: Any) -> list[str]:
    """
    Lê o schema de entrada salvo pelo MLflow.

    Isso permite que o backend seja compatível com:
    - modelos antigos, que esperam city;
    - modelos novos, que esperam property_type, neighborhood e base_price_m2.
    """
    try:
        schema = model.metadata.get_input_schema()

        if schema is None:
            return []

        columns = []

        for item in schema.inputs:
            name = getattr(item, "name", None)

            if name:
                columns.append(str(name))

        return columns

    except Exception as exc:
        print(f"[model-schema-warning] Não foi possível ler input schema: {exc}")
        return []


def model_to_dict(item: BaseModel) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()

    return item.dict()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_label(value: Optional[str]) -> str:
    return str(value) if value not in {None, ""} else "unknown"


def prediction_metric_labels(
    info: dict[str, Optional[str]],
    endpoint: str,
    status: str,
) -> dict[str, str]:
    return {
        "model_name": safe_label(info.get("model_name")),
        "model_version": safe_label(info.get("model_version")),
        "model_alias": safe_label(info.get("model_alias")),
        "endpoint": endpoint,
        "status": status,
    }


def latency_metric_labels(
    info: dict[str, Optional[str]],
    endpoint: str,
) -> dict[str, str]:
    return {
        "model_name": safe_label(info.get("model_name")),
        "model_version": safe_label(info.get("model_version")),
        "model_alias": safe_label(info.get("model_alias")),
        "endpoint": endpoint,
    }


def drift_metric_labels(info: dict[str, Optional[str]]) -> dict[str, str]:
    return {
        "model_name": safe_label(info.get("model_name")),
        "model_version": safe_label(info.get("model_version")),
        "model_alias": safe_label(info.get("model_alias")),
    }


# ========================
# FEATURE NORMALIZATION
# ========================

def normalize_property_type(value: Any) -> str:
    if value is None:
        return DEFAULT_PROPERTY_TYPE

    normalized = str(value).strip().lower()

    if normalized in {"house", "casa"}:
        return "house"

    if normalized in {"apartment", "apartamento", "apt", "apto"}:
        return "apartment"

    return DEFAULT_PROPERTY_TYPE


def infer_base_price_m2(row: pd.Series) -> float:
    value = row.get("base_price_m2")

    if pd.notna(value):
        return float(value)

    city = row.get("city") or DEFAULT_CITY
    return float(CAPITAL_BASE_PRICE_M2.get(str(city), DEFAULT_BASE_PRICE_M2))


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for column in NUMERIC_REQUIRED_COLUMNS:
        if column not in normalized.columns:
            raise ValueError(f"Campo obrigatório ausente no payload: {column}")

    if "city" not in normalized.columns:
        normalized["city"] = DEFAULT_CITY

    normalized["city"] = normalized["city"].fillna(DEFAULT_CITY).astype(str)

    if "property_type" not in normalized.columns:
        normalized["property_type"] = DEFAULT_PROPERTY_TYPE

    normalized["property_type"] = normalized["property_type"].apply(normalize_property_type)

    if "neighborhood" not in normalized.columns:
        normalized["neighborhood"] = DEFAULT_NEIGHBORHOOD

    normalized["neighborhood"] = (
        normalized["neighborhood"]
        .fillna(DEFAULT_NEIGHBORHOOD)
        .astype(str)
    )

    if "base_price_m2" not in normalized.columns:
        normalized["base_price_m2"] = None

    normalized["base_price_m2"] = normalized.apply(infer_base_price_m2, axis=1)

    optional_string_defaults = {
        "state": "unknown",
        "region": "unknown",
        "price_source": "runtime_default",
    }

    for column, default_value in optional_string_defaults.items():
        if column not in normalized.columns:
            normalized[column] = default_value

        normalized[column] = normalized[column].fillna(default_value).astype(str)

    for column in MODEL_NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("float64")

    ordered_columns = [
        column for column in POTENTIAL_INPUT_COLUMNS if column in normalized.columns
    ]

    return normalized[ordered_columns]


# ========================
# LOGGING
# ========================

def append_prediction_logs(
    endpoint: str,
    info: dict[str, Optional[str]],
    input_records: list[dict[str, Any]],
    predictions: list[float],
    latency_seconds: float,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    if not LOG_PREDICTIONS:
        return

    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = PREDICTION_LOG_PATH.exists()

        fieldnames = [
            "timestamp",
            "endpoint",
            "status",
            "model_uri",
            "model_name",
            "model_version",
            "model_alias",
            "prediction",
            "score",
            "latency_ms",
            "input_features",
            "error_message",
        ]

        with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            if predictions:
                for input_record, prediction in zip(input_records, predictions):
                    writer.writerow(
                        {
                            "timestamp": now_utc_iso(),
                            "endpoint": endpoint,
                            "status": status,
                            "model_uri": info.get("model_uri"),
                            "model_name": info.get("model_name"),
                            "model_version": info.get("model_version"),
                            "model_alias": info.get("model_alias"),
                            "prediction": prediction,
                            "score": prediction,
                            "latency_ms": round(latency_seconds * 1000, 4),
                            "input_features": json.dumps(
                                input_record,
                                ensure_ascii=False,
                                default=str,
                            ),
                            "error_message": error_message,
                        }
                    )
            else:
                writer.writerow(
                    {
                        "timestamp": now_utc_iso(),
                        "endpoint": endpoint,
                        "status": status,
                        "model_uri": info.get("model_uri"),
                        "model_name": info.get("model_name"),
                        "model_version": info.get("model_version"),
                        "model_alias": info.get("model_alias"),
                        "prediction": None,
                        "score": None,
                        "latency_ms": round(latency_seconds * 1000, 4),
                        "input_features": json.dumps(
                            input_records,
                            ensure_ascii=False,
                            default=str,
                        ),
                        "error_message": error_message,
                    }
                )

    except Exception as log_error:
        print(f"[prediction-logging-warning] Falha ao gravar log de predicao: {log_error}")


# ========================
# MODEL STORE
# ========================

class ModelStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self.model: Optional[Any] = None
        self.model_uri: Optional[str] = None
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.model_alias: Optional[str] = None
        self.model_source: Optional[str] = None
        self.input_columns: list[str] = []

    def load(
        self,
        model_uri: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        loaded = mlflow.pyfunc.load_model(model_uri)
        registry_info = resolve_model_registry_info(model_uri)
        input_columns = get_model_input_columns(loaded)

        with self._lock:
            self.model = loaded
            self.model_uri = model_uri
            self.model_name = registry_info.get("registered_model_name") or model_name
            self.model_version = registry_info.get("model_version") or model_version
            self.model_alias = registry_info.get("model_alias")
            self.model_source = registry_info.get("model_source")
            self.input_columns = input_columns

        print(
            "[model-loaded] "
            f"uri={model_uri} version={self.model_version} "
            f"alias={self.model_alias} input_columns={input_columns}"
        )

    def predict(self, df: pd.DataFrame) -> list[float]:
        with self._lock:
            if self.model is None:
                raise RuntimeError("Nenhum modelo foi carregado.")

            input_columns = list(self.input_columns)
            model = self.model

        if input_columns:
            missing_columns = [column for column in input_columns if column not in df.columns]

            if missing_columns:
                raise ValueError(
                    "Payload não contém as colunas exigidas pelo modelo carregado: "
                    f"{missing_columns}. Colunas disponíveis: {list(df.columns)}"
                )

            df_to_predict = df[input_columns].copy()
        else:
            df_to_predict = df.copy()

        preds = model.predict(df_to_predict)

        if hasattr(preds, "tolist"):
            return [float(x) for x in preds.tolist()]

        return [float(x) for x in preds]

    def info(self) -> dict[str, Optional[str]]:
        with self._lock:
            return {
                "model_uri": self.model_uri,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_alias": self.model_alias,
                "model_source": self.model_source,
                "input_columns": self.input_columns,
            }


store = ModelStore()


# ========================
# PREDICTION FLOW
# ========================

def predict_with_observability(
    request: PredictRequest,
    endpoint: str,
) -> tuple[list[float], dict[str, Optional[str]]]:
    start_time = time.perf_counter()
    input_records: list[dict[str, Any]] = []

    try:
        input_records = [model_to_dict(item) for item in request.dataframe_records]
        df = pd.DataFrame(input_records)
        df = normalize_features(df)

        preds = store.predict(df)
        latency_seconds = time.perf_counter() - start_time
        info = store.info()

        MODEL_PREDICTION_TOTAL.labels(
            **prediction_metric_labels(info=info, endpoint=endpoint, status="success")
        ).inc(len(preds))

        MODEL_PREDICTION_LATENCY_SECONDS.labels(
            **latency_metric_labels(info=info, endpoint=endpoint)
        ).observe(latency_seconds)

        MODEL_PREDICTION_BATCH_SIZE.labels(
            **latency_metric_labels(info=info, endpoint=endpoint)
        ).observe(len(preds))

        if preds:
            MODEL_LAST_PREDICTION_VALUE.labels(
                **latency_metric_labels(info=info, endpoint=endpoint)
            ).set(preds[-1])

        append_prediction_logs(
            endpoint=endpoint,
            info=info,
            input_records=input_records,
            predictions=preds,
            latency_seconds=latency_seconds,
            status="success",
        )

        return preds, info

    except Exception as e:
        latency_seconds = time.perf_counter() - start_time
        info = store.info()

        MODEL_PREDICTION_TOTAL.labels(
            **prediction_metric_labels(info=info, endpoint=endpoint, status="error")
        ).inc()

        MODEL_PREDICTION_LATENCY_SECONDS.labels(
            **latency_metric_labels(info=info, endpoint=endpoint)
        ).observe(latency_seconds)

        append_prediction_logs(
            endpoint=endpoint,
            info=info,
            input_records=input_records,
            predictions=[],
            latency_seconds=latency_seconds,
            status="error",
            error_message=str(e),
        )

        raise


# ========================
# DRIFT
# ========================

def refresh_drift_metrics() -> DriftRefreshResponse:
    info = store.info()

    if not DRIFT_METRICS_PATH.exists():
        return DriftRefreshResponse(
            status="not_found",
            drift_metrics_path=str(DRIFT_METRICS_PATH),
        )

    with DRIFT_METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    data_drift_score = metrics.get("data_drift_score")
    prediction_drift_score = metrics.get("prediction_drift_score")
    drift_detected = metrics.get("drift_detected")

    labels = drift_metric_labels(info)

    if data_drift_score is not None:
        MODEL_DATA_DRIFT_SCORE.labels(**labels).set(float(data_drift_score))

    if prediction_drift_score is not None:
        MODEL_PREDICTION_DRIFT_SCORE.labels(**labels).set(float(prediction_drift_score))

    if drift_detected is not None:
        MODEL_DRIFT_DETECTED.labels(**labels).set(1 if bool(drift_detected) else 0)

    return DriftRefreshResponse(
        status="refreshed",
        drift_metrics_path=str(DRIFT_METRICS_PATH),
        data_drift_score=float(data_drift_score) if data_drift_score is not None else None,
        prediction_drift_score=(
            float(prediction_drift_score) if prediction_drift_score is not None else None
        ),
        drift_detected=bool(drift_detected) if drift_detected is not None else None,
    )


# ========================
# FASTAPI
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load(DEFAULT_MODEL_URI, model_name="apartment-price-regression")

    try:
        refresh_drift_metrics()
    except Exception as e:
        print(f"[drift-metrics-warning] Falha ao carregar metricas de drift no startup: {e}")

    yield


app = FastAPI(
    title="Apartment Price API",
    version="1.2.1",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def current_model() -> dict[str, Any]:
    info = store.info()
    return {
        "status": "loaded" if info["model_uri"] else "not_loaded",
        **info,
        "tracking_uri": mlflow.get_tracking_uri(),
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    info = store.info()

    return {
        "status": "ok",
        "model_input_columns": info.get("input_columns") or [],
        "accepted_payload_fields": POTENTIAL_INPUT_COLUMNS,
        "required_numeric_fields": NUMERIC_REQUIRED_COLUMNS,
        "model_numeric_fields": MODEL_NUMERIC_COLUMNS,
        "defaults": {
            "property_type": DEFAULT_PROPERTY_TYPE,
            "neighborhood": DEFAULT_NEIGHBORHOOD,
            "city": DEFAULT_CITY,
            "base_price_m2": "derived from city when omitted",
            "fallback_base_price_m2": DEFAULT_BASE_PRICE_M2,
        },
        "city_base_price_m2_count": len(CAPITAL_BASE_PRICE_M2),
    }


@app.post("/model/load", response_model=ModelLoadResponse)
def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    try:
        store.load(
            model_uri=request.model_uri,
            model_name=request.model_name,
            model_version=request.model_version,
        )

        info = store.info()

        return ModelLoadResponse(
            status="loaded",
            model_name=info["model_name"],
            model_uri=info["model_uri"] or request.model_uri,
            model_version=info["model_version"],
            model_alias=info["model_alias"],
            model_source=info["model_source"],
            input_columns=info.get("input_columns") or [],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao carregar modelo: {e}")


@app.post("/model/reload", response_model=ModelReloadResponse)
def reload_model() -> ModelReloadResponse:
    try:
        current_info = store.info()
        model_uri = current_info["model_uri"] or DEFAULT_MODEL_URI

        store.load(
            model_uri=model_uri,
            model_name=current_info["model_name"] or "apartment-price-regression",
            model_version=current_info["model_version"],
        )

        info = store.info()

        return ModelReloadResponse(
            status="reloaded",
            model_name=info["model_name"],
            model_uri=info["model_uri"] or model_uri,
            model_version=info["model_version"],
            model_alias=info["model_alias"],
            model_source=info["model_source"],
            input_columns=info.get("input_columns") or [],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao recarregar modelo: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        preds, info = predict_with_observability(request=request, endpoint="/predict")

        return PredictResponse(
            predictions=preds,
            model_uri=info["model_uri"] or "",
            model_name=info["model_name"],
            model_version=info["model_version"],
            model_alias=info["model_alias"],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha na predição: {e}")


@app.post("/invocations")
def invocations(request: PredictRequest) -> dict[str, Any]:
    try:
        preds, _ = predict_with_observability(request=request, endpoint="/invocations")
        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha na predição: {e}")


@app.post("/drift/refresh", response_model=DriftRefreshResponse)
def drift_refresh() -> DriftRefreshResponse:
    try:
        return refresh_drift_metrics()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao atualizar métricas de drift: {e}")


@app.get("/drift/status", response_model=DriftRefreshResponse)
def drift_status() -> DriftRefreshResponse:
    return refresh_drift_metrics()
