from __future__ import annotations

import os
from contextlib import asynccontextmanager
from threading import RLock
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
DEFAULT_MODEL_URI = os.getenv(
    "DEFAULT_MODEL_URI",
    "models:/apartment-price-regression@champion",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ApartmentFeatures(BaseModel):
    city: str
    area_m2: int
    bedrooms: int
    bathrooms: int
    floor: int
    parking_spaces: int
    neighborhood_score: float
    condo_fee: float
    age_years: int
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


class ModelReloadResponse(BaseModel):
    status: str
    model_name: Optional[str]
    model_uri: str
    model_version: Optional[str]
    model_alias: Optional[str] = None
    model_source: Optional[str] = None


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


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "city": "string",
            "area_m2": "int64",
            "bedrooms": "int64",
            "bathrooms": "int64",
            "floor": "int64",
            "parking_spaces": "int64",
            "neighborhood_score": "float64",
            "condo_fee": "float64",
            "age_years": "int64",
            "distance_to_center_km": "float64",
        }
    )

class ModelStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self.model: Optional[Any] = None
        self.model_uri: Optional[str] = None
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None
        self.model_alias: Optional[str] = None
        self.model_source: Optional[str] = None

    def load(
        self,
        model_uri: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        loaded = mlflow.pyfunc.load_model(model_uri)
        registry_info = resolve_model_registry_info(model_uri)

        with self._lock:
            self.model = loaded
            self.model_uri = model_uri
            self.model_name = registry_info.get("registered_model_name") or model_name
            self.model_version = registry_info.get("model_version") or model_version
            self.model_alias = registry_info.get("model_alias")
            self.model_source = registry_info.get("model_source")

    def predict(self, df: pd.DataFrame) -> list[float]:
        with self._lock:
            if self.model is None:
                raise RuntimeError("Nenhum modelo foi carregado.")
            preds = self.model.predict(df)

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
            }


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load(DEFAULT_MODEL_URI, model_name="apartment-price-regression")
    yield


app = FastAPI(
    title="Apartment Price API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def current_model() -> dict[str, Optional[str]]:
    info = store.info()
    return {
        "status": "loaded" if info["model_uri"] else "not_loaded",
        **info,
        "tracking_uri": mlflow.get_tracking_uri(),
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
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao recarregar modelo: {e}")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        df = pd.DataFrame([item.model_dump() for item in request.dataframe_records])
        df = normalize_features(df)

        preds = store.predict(df)
        info = store.info()

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
        df = pd.DataFrame([item.model_dump() for item in request.dataframe_records])
        df = normalize_features(df)

        preds = store.predict(df)

        return {"predictions": preds}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha na predição: {e}")