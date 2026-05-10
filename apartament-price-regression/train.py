from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ========================
# CONFIG
# ========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "local_artifacts"

TRAIN_PATH = DATA_DIR / "train_apartments.csv"
VALID_PATH = DATA_DIR / "validation_apartments.csv"
TEST_PATH = DATA_DIR / "test_apartments.csv"

EXPERIMENT_NAME = "apartment-price-regression-minio"
MODEL_NAME = "apartment-price-regression"

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

RANDOM_STATE = 42
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))
GRID_N_JOBS = int(os.getenv("GRID_N_JOBS", "1"))

PRIMARY_METRIC = "rmse"
SERVING_SAFE = True


# ========================
# DATA CONFIG
# ========================

def get_feature_config(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    target_column = "price"

    categorical_features = [
        "property_type",
        "neighborhood",
    ]

    numerical_features = [
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

    audit_features = [
        "state",
        "region",
        "city",
        "price_source",
    ]

    required_columns = categorical_features + numerical_features + [target_column]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "O dataset não possui as colunas esperadas para o treino serving-safe: "
            f"{missing_columns}"
        )

    return categorical_features, numerical_features, audit_features, required_columns


def validate_dataset(name: str, df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Dataset {name} está sem as colunas obrigatórias: {missing_columns}"
        )


def normalize_property_type(value: Any) -> str:
    normalized = str(value).strip().lower()

    if normalized in {"house", "casa"}:
        return "house"

    if normalized in {"apartment", "apartamento", "apartament", "apt", "apto"}:
        return "apartment"

    return "apartment"


def normalize_neighborhood(value: Any) -> str:
    normalized = str(value).strip()

    aliases = {
        "Area Nobre": "Área Nobre",
        "Area Universitaria": "Área Universitária",
        "Regiao Residencial": "Região Residencial",
        "Regiao Comercial": "Região Comercial",
        "Regiao Periferica": "Região Periférica",
    }

    return aliases.get(normalized, normalized)


def load_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
    list[str],
]:
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_column = "price"

    (
        categorical_features,
        numerical_features,
        audit_features,
        required_columns,
    ) = get_feature_config(train_df)

    for name, df in {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }.items():
        validate_dataset(name=name, df=df, required_columns=required_columns)

        df["property_type"] = df["property_type"].apply(normalize_property_type).astype(str)
        df["neighborhood"] = df["neighborhood"].apply(normalize_neighborhood).astype(str)

        df[numerical_features] = df[numerical_features].astype(float)
        df[target_column] = df[target_column].astype(float)

    input_features = categorical_features + numerical_features

    X_train = train_df[input_features]
    y_train = train_df[target_column]

    X_validation = validation_df[input_features]
    y_validation = validation_df[target_column]

    X_test = test_df[input_features]
    y_test = test_df[target_column]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        categorical_features,
        numerical_features,
        audit_features,
    )


# ========================
# PREPROCESSING
# ========================

def build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
                categorical_features,
            ),
            (
                "numeric_scaler",
                StandardScaler(),
                numerical_features,
            ),
        ],
        sparse_threshold=0,
        remainder="drop",
    )


# ========================
# MODEL CANDIDATES
# ========================

@dataclass
class ModelCandidate:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    family: str


def get_model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            name="LinearRegression",
            estimator=LinearRegression(),
            param_grid={},
            family="linear",
        ),
        ModelCandidate(
            name="Ridge",
            estimator=Ridge(),
            param_grid={
                "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
            },
            family="linear_regularized",
        ),
        ModelCandidate(
            name="RandomForestRegressor",
            estimator=RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            param_grid={
                "model__n_estimators": [200],
                "model__max_depth": [16, None],
                "model__min_samples_leaf": [1, 2],
            },
            family="bagging_tree",
        ),
        ModelCandidate(
            name="ExtraTreesRegressor",
            estimator=ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            param_grid={
                "model__n_estimators": [200],
                "model__max_depth": [16, None],
                "model__min_samples_leaf": [1, 2],
            },
            family="bagging_tree",
        ),
        ModelCandidate(
            name="GradientBoostingRegressor",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "model__n_estimators": [200, 250],
                "model__learning_rate": [0.05, 0.08],
                "model__max_depth": [2, 3],
            },
            family="boosting_tree",
        ),
    ]


def build_pipeline(
    candidate: ModelCandidate,
    categorical_features: list[str],
    numerical_features: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", candidate.estimator),
        ]
    )


# ========================
# METRICS
# ========================

def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(X)

    mae = float(mean_absolute_error(y, predictions))
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    r2 = float(r2_score(y, predictions))
    mape = float(np.mean(np.abs((y - predictions) / np.maximum(np.abs(y), 1e-10))) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }, predictions


def save_predictions_preview(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    preview = X.copy()
    preview["y_true"] = y_true.values
    preview["y_pred"] = y_pred
    preview["absolute_error"] = np.abs(preview["y_true"] - preview["y_pred"])
    preview.head(100).to_csv(path, index=False)


# ========================
# MODEL TRAINING
# ========================

def train_candidate(
    candidate: ModelCandidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
    numerical_features: list[str],
) -> dict[str, Any]:
    pipeline = build_pipeline(
        candidate=candidate,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=candidate.param_grid,
        scoring="neg_root_mean_squared_error",
        cv=CV_FOLDS,
        n_jobs=GRID_N_JOBS,
        refit=True,
    )

    started_at = time.perf_counter()
    search.fit(X_train, y_train)
    training_seconds = float(time.perf_counter() - started_at)

    best_model = search.best_estimator_
    validation_metrics, validation_predictions = evaluate(
        model=best_model,
        X=X_validation,
        y=y_validation,
    )

    return {
        "name": candidate.name,
        "family": candidate.family,
        "model": best_model,
        "best_params": search.best_params_,
        "cv_best_rmse": float(abs(search.best_score_)),
        "validation_metrics": validation_metrics,
        "validation_predictions": validation_predictions,
        "training_seconds": training_seconds,
    }


def select_best_model(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Nenhum modelo candidato foi treinado com sucesso.")

    return min(
        results,
        key=lambda item: item["validation_metrics"][PRIMARY_METRIC],
    )


def build_ranking_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in results:
        metrics = result["validation_metrics"]

        rows.append(
            {
                "model_name": result["name"],
                "family": result["family"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
                "validation_mape": metrics["mape"],
                "cv_best_rmse": result["cv_best_rmse"],
                "training_seconds": result["training_seconds"],
                "best_params": json.dumps(result["best_params"], ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows).sort_values("validation_rmse", ascending=True)


# ========================
# SERVING VALIDATION
# ========================

def smoke_test_model(model: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    sample = X.head(1).copy()

    started_at = time.perf_counter()
    prediction = model.predict(sample)
    elapsed_seconds = float(time.perf_counter() - started_at)

    if prediction is None or len(prediction) != 1:
        raise RuntimeError("Smoke test falhou: predict não retornou exatamente uma predição.")

    pipeline_steps = list(model.named_steps.keys())

    if pipeline_steps != ["preprocessor", "model"]:
        raise RuntimeError(
            "Pipeline inválido para serving. Esperado ['preprocessor', 'model'], "
            f"recebido {pipeline_steps}"
        )

    return {
        "smoke_prediction": float(prediction[0]),
        "smoke_predict_seconds": elapsed_seconds,
        "pipeline_steps": pipeline_steps,
    }


# ========================
# MLFLOW LOGGING
# ========================

def log_parent_params(
    categorical_features: list[str],
    numerical_features: list[str],
    audit_features: list[str],
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    candidate_count: int,
) -> None:
    params = {
        "selection_strategy": "GridSearchCV",
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_goal": "minimize",
        "cv_folds": CV_FOLDS,
        "grid_n_jobs": GRID_N_JOBS,
        "candidate_count": candidate_count,
        "categorical_features": ",".join(categorical_features),
        "numerical_features": ",".join(numerical_features),
        "audit_features_excluded_from_training": ",".join(audit_features),
        "serving_safe": "true",
        "custom_transformer_used": "false",
        "derived_features_inside_pipeline": "false",
        "explainability_artifacts_enabled": "false",
        "feature_engineering_policy": "no_custom_transformer_no_derived_features",
        "multicollinearity_policy": "exclude_state_region_city_when_base_price_m2_is_used",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "tracking_uri": TRACKING_URI,
    }

    for key, value in params.items():
        mlflow.log_param(key, value)


def log_candidate_result(candidate: ModelCandidate, result: dict[str, Any]) -> None:
    mlflow.log_param("model_name", candidate.name)
    mlflow.log_param("family", candidate.family)
    mlflow.log_param("param_grid", json.dumps(candidate.param_grid, ensure_ascii=False))
    mlflow.log_param("serving_safe", "true")
    mlflow.log_param("custom_transformer_used", "false")
    mlflow.log_param("best_params", json.dumps(result["best_params"], ensure_ascii=False))

    mlflow.log_metric("cv_best_rmse", result["cv_best_rmse"])
    mlflow.log_metric("training_seconds", result["training_seconds"])

    for key, value in result["validation_metrics"].items():
        mlflow.log_metric(f"validation_{key}", value)


def log_best_model_info(
    best_result: dict[str, Any],
    test_metrics: dict[str, float],
    smoke_info: dict[str, Any],
) -> None:
    mlflow.log_param("best_model_name", best_result["name"])
    mlflow.log_param("best_model_family", best_result["family"])
    mlflow.log_param("best_model_params", json.dumps(best_result["best_params"], ensure_ascii=False))
    mlflow.log_param("pipeline_steps", ",".join(smoke_info["pipeline_steps"]))

    mlflow.log_metric("smoke_predict_seconds", smoke_info["smoke_predict_seconds"])
    mlflow.log_metric("smoke_prediction", smoke_info["smoke_prediction"])

    for key, value in best_result["validation_metrics"].items():
        mlflow.log_metric(f"best_validation_{key}", value)

    for key, value in test_metrics.items():
        mlflow.log_metric(f"test_{key}", value)

    mlflow.set_tag("best_model_name", best_result["name"])
    mlflow.set_tag("model_family", best_result["family"])
    mlflow.set_tag("serving_safe", "true")
    mlflow.set_tag("custom_transformer_used", "false")
    mlflow.set_tag("explainability_artifacts_enabled", "false")
    mlflow.set_tag("pipeline_steps", ",".join(smoke_info["pipeline_steps"]))


# ========================
# REGISTRY
# ========================

def register_model(model_uri: str, run_id: str) -> dict[str, Any]:
    client = MlflowClient()

    try:
        client.get_registered_model(MODEL_NAME)
        print(f"[INFO] Modelo registrado encontrado: {MODEL_NAME}")
    except Exception:
        client.create_registered_model(MODEL_NAME)
        print(f"[INFO] Modelo registrado criado: {MODEL_NAME}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=model_version.version,
    )

    print(f"[OK] Nova versão registrada: v{model_version.version}")
    print(f"[OK] Alias {MODEL_ALIAS} -> v{model_version.version}")

    return {
        "registered_model_name": MODEL_NAME,
        "model_version": model_version.version,
        "model_registry_uri": f"models:/{MODEL_NAME}/{model_version.version}",
        "model_alias_uri": f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "registered_from_model_uri": model_uri,
        "run_id": run_id,
    }


# ========================
# MAIN
# ========================

def main() -> None:
    print("[INFO] Iniciando seleção de modelos de regressão serving-safe...")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        categorical_features,
        numerical_features,
        audit_features,
    ) = load_data()

    candidates = get_model_candidates()
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="model-selection-regression-serving-safe") as parent_run:
        log_parent_params(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            audit_features=audit_features,
            train_rows=len(X_train),
            validation_rows=len(X_validation),
            test_rows=len(X_test),
            candidate_count=len(candidates),
        )

        print("\n[INFO] Features categóricas usadas no treino:", categorical_features)
        print("[INFO] Features numéricas usadas no treino:", numerical_features)
        print("[INFO] Features mantidas apenas para auditoria/drift:", audit_features)
        print("[INFO] Transformer customizado: desativado")
        print("[INFO] Explicabilidade/permutation importance: desativada")

        for candidate in candidates:
            print(f"\n[INFO] Treinando candidato: {candidate.name}")

            with mlflow.start_run(run_name=f"candidate-{candidate.name}", nested=True):
                try:
                    result = train_candidate(
                        candidate=candidate,
                        X_train=X_train,
                        y_train=y_train,
                        X_validation=X_validation,
                        y_validation=y_validation,
                        categorical_features=categorical_features,
                        numerical_features=numerical_features,
                    )

                    results.append(result)
                    log_candidate_result(candidate=candidate, result=result)

                    print(
                        f"[OK] {candidate.name} | "
                        f"validation_rmse={result['validation_metrics']['rmse']:.4f} | "
                        f"validation_r2={result['validation_metrics']['r2']:.4f}"
                    )

                except Exception as exc:
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(exc)[:500])
                    print(f"[ERRO] Falha no candidato {candidate.name}: {exc}")

        best_result = select_best_model(results)
        best_model = best_result["model"]

        smoke_info = smoke_test_model(best_model, X_test)

        print("\n[SMOKE TEST] Modelo passou no teste mínimo de serving")
        print(smoke_info)

        test_metrics, test_predictions = evaluate(
            model=best_model,
            X=X_test,
            y=y_test,
        )

        print("\n[RESULTADO] Melhor modelo encontrado")
        print(f"Modelo: {best_result['name']}")
        print(f"Família: {best_result['family']}")
        print(f"Melhores hiperparâmetros: {best_result['best_params']}")
        print(f"[VALIDATION] {best_result['validation_metrics']}")
        print(f"[TEST] {test_metrics}")

        ranking_df = build_ranking_dataframe(results)

        ranking_path = ARTIFACTS_DIR / "model_selection_ranking.csv"
        validation_preview_path = ARTIFACTS_DIR / "validation_predictions_preview.csv"
        test_preview_path = ARTIFACTS_DIR / "test_predictions_preview.csv"

        ranking_df.to_csv(ranking_path, index=False)

        save_predictions_preview(
            X=X_validation,
            y_true=y_validation,
            y_pred=best_result["validation_predictions"],
            path=validation_preview_path,
        )

        save_predictions_preview(
            X=X_test,
            y_true=y_test,
            y_pred=test_predictions,
            path=test_preview_path,
        )

        mlflow.log_artifact(ranking_path, artifact_path="model_selection")
        mlflow.log_artifact(validation_preview_path, artifact_path="analysis")
        mlflow.log_artifact(test_preview_path, artifact_path="analysis")

        log_best_model_info(
            best_result=best_result,
            test_metrics=test_metrics,
            smoke_info=smoke_info,
        )

        signature = infer_signature(X_train, best_model.predict(X_train.head(5)))
        input_example = X_train.head(3)

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        registry_info = register_model(
            model_uri=model_info.model_uri,
            run_id=parent_run.info.run_id,
        )

        run_summary = {
            "run_id": parent_run.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": TRACKING_URI,
            "selection": {
                "strategy": "GridSearchCV",
                "primary_metric": PRIMARY_METRIC,
                "primary_metric_goal": "minimize",
                "cv_folds": CV_FOLDS,
                "candidate_count": len(candidates),
                "successful_candidates": len(results),
            },
            "features": {
                "categorical_features": categorical_features,
                "numerical_features": numerical_features,
                "audit_features_excluded_from_training": audit_features,
                "derived_features_removed_from_pipeline": True,
                "custom_transformer_used": False,
                "explainability_artifacts_enabled": False,
                "multicollinearity_policy": (
                    "state, region, city and price_source are kept outside training; "
                    "base_price_m2 represents the external market reference."
                ),
            },
            "best_model": {
                "name": best_result["name"],
                "family": best_result["family"],
                "best_params": best_result["best_params"],
                "validation_metrics": best_result["validation_metrics"],
                "test_metrics": test_metrics,
                "smoke_test": smoke_info,
            },
            "ranking": ranking_df.to_dict(orient="records"),
            "model_uri": model_info.model_uri,
            "registry": registry_info,
            "datasets": {
                "train": str(TRAIN_PATH),
                "validation": str(VALID_PATH),
                "test": str(TEST_PATH),
            },
            "artifacts": {
                "ranking": str(ranking_path),
                "validation_preview": str(validation_preview_path),
                "test_preview": str(test_preview_path),
            },
        }

        summary_path = PROJECT_ROOT / "run_summary.json"
        selection_summary_path = PROJECT_ROOT / "model_selection_summary.json"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(run_summary, file, ensure_ascii=False, indent=2)

        with selection_summary_path.open("w", encoding="utf-8") as file:
            json.dump(run_summary, file, ensure_ascii=False, indent=2)

        mlflow.log_artifact(summary_path, artifact_path="summary")
        mlflow.log_artifact(selection_summary_path, artifact_path="summary")

    print("\n[RESUMO FINAL]")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("[FINALIZADO]")


if __name__ == "__main__":
    main()
    