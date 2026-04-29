from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")


# ========================
# DATA
# ========================

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_column = "price"

    categorical_features = ["city"]

    numerical_features = [
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

    required_columns = categorical_features + numerical_features + [target_column]

    for name, df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Dataset {name} está sem as colunas obrigatórias: {missing_columns}"
            )

    X_train = train_df[categorical_features + numerical_features]
    y_train = train_df[target_column]

    X_val = val_df[categorical_features + numerical_features]
    y_val = val_df[target_column]

    X_test = test_df[categorical_features + numerical_features]
    y_test = test_df[target_column]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        categorical_features,
        numerical_features,
    )


# ========================
# MODEL
# ========================

def train_model(X_train, y_train, categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "city_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric_features",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


# ========================
# METRICS
# ========================

def evaluate(model, X, y) -> tuple[dict[str, float], np.ndarray]:
    preds = model.predict(X)

    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2 = float(r2_score(y, preds))
    mape = float(np.mean(np.abs((y - preds) / np.maximum(np.abs(y), 1e-10))) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }, preds


def save_predictions_preview(X, y_true, y_pred, path: Path) -> None:
    preview = X.copy()
    preview["y_true"] = y_true.values
    preview["y_pred"] = y_pred
    preview.head(50).to_csv(path, index=False)


# ========================
# REGISTRY
# ========================

def register_model(model_uri: str, run_id: str) -> dict:
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

def main():
    print("[INFO] Iniciando treinamento...")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        categorical_features,
        numerical_features,
    ) = load_data()

    with mlflow.start_run(run_name="random-forest-prod-city") as run:
        mlflow.log_param("model_name", "RandomForestRegressor")
        mlflow.log_param("uses_city", True)
        mlflow.log_param("categorical_features", ",".join(categorical_features))
        mlflow.log_param("numerical_features", ",".join(numerical_features))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("tracking_uri", TRACKING_URI)

        model = train_model(
            X_train,
            y_train,
            categorical_features,
            numerical_features,
        )

        val_metrics, val_preds = evaluate(model, X_val, y_val)
        test_metrics, test_preds = evaluate(model, X_test, y_test)

        print("[VALIDATION]", val_metrics)
        print("[TEST]", test_metrics)

        for key, value in val_metrics.items():
            mlflow.log_metric(f"validation_{key}", value)

        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        validation_preview_path = ARTIFACTS_DIR / "validation_predictions_preview.csv"
        test_preview_path = ARTIFACTS_DIR / "test_predictions_preview.csv"

        save_predictions_preview(X_val, y_val, val_preds, validation_preview_path)
        save_predictions_preview(X_test, y_test, test_preds, test_preview_path)

        mlflow.log_artifact(validation_preview_path, artifact_path="analysis")
        mlflow.log_artifact(test_preview_path, artifact_path="analysis")

        signature = infer_signature(X_train, model.predict(X_train.head(5)))
        input_example = X_train.head(3)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        registry_info = register_model(
            model_uri=model_info.model_uri,
            run_id=run.info.run_id,
        )

        run_summary = {
            "run_id": run.info.run_id,
            "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": TRACKING_URI,
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
            },
            "model_uri": model_info.model_uri,
            "registry": registry_info,
            "datasets": {
                "train": str(TRAIN_PATH),
                "validation": str(VALID_PATH),
                "test": str(TEST_PATH),
            },
        }

        summary_path = PROJECT_ROOT / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("[FINALIZADO]")


if __name__ == "__main__":
    main()