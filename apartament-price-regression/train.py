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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
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

# Use candidate primeiro para não substituir o champion antes de validar backend/frontend.
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

RANDOM_STATE = 42
CV_FOLDS = int(os.getenv("CV_FOLDS", "3"))

# Em Windows, n_jobs=1 no GridSearchCV costuma ser mais estável.
# Os modelos de árvore ainda usam n_jobs=-1 internamente quando suportado.
GRID_N_JOBS = int(os.getenv("GRID_N_JOBS", "1"))

PRIMARY_METRIC = "rmse"
PERMUTATION_IMPORTANCE_ROWS = int(os.getenv("PERMUTATION_IMPORTANCE_ROWS", "1200"))


# ========================
# FEATURE ENGINEERING
# ========================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Cria atributos derivados simples e explicáveis.

    A versão atual evita multicolinearidade forte:
    - não usa state, region e city no treino quando base_price_m2 já representa
      a referência externa de mercado da capital;
    - não cria rooms_total, pois seria soma direta de bedrooms + bathrooms;
    - mantém interações úteis que não são combinação linear simples.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df["area_per_bedroom"] = df["area_m2"] / np.maximum(df["bedrooms"], 1)
        df["condo_fee_per_m2"] = df["condo_fee"] / np.maximum(df["area_m2"], 1)
        df["bathrooms_per_bedroom"] = df["bathrooms"] / np.maximum(df["bedrooms"], 1)
        df["parking_per_bedroom"] = df["parking_spaces"] / np.maximum(df["bedrooms"], 1)

        df["is_high_floor"] = (df["floor"] >= 15).astype(int)
        df["is_new_property"] = (df["age_years"] <= 5).astype(int)
        df["is_old_property"] = (df["age_years"] >= 25).astype(int)
        df["premium_location"] = (
            (df["neighborhood_score"] >= 8.5)
            & (df["distance_to_center_km"] <= 7.0)
        ).astype(int)

        # Interação econômica: referência externa de mercado * área.
        # É útil para preço de imóvel e não é uma combinação linear simples.
        df["estimated_market_value"] = df["base_price_m2"] * df["area_m2"]

        return df


# ========================
# DATA
# ========================

def get_feature_config(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    target_column = "price"

    # Evitamos state, region e city para reduzir redundância/multicolinearidade:
    # base_price_m2 já codifica a referência externa da cidade/capital.
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

    derived_numerical_features = [
        "area_per_bedroom",
        "condo_fee_per_m2",
        "bathrooms_per_bedroom",
        "parking_per_bedroom",
        "is_high_floor",
        "is_new_property",
        "is_old_property",
        "premium_location",
        "estimated_market_value",
    ]

    # Mantidas fora do treino para auditoria, drift e explicação do dataset.
    audit_features = [
        "state",
        "region",
        "city",
        "price_source",
    ]

    required_columns = (
        categorical_features
        + numerical_features
        + [target_column]
    )

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "O dataset não possui as colunas esperadas para o novo treino: "
            f"{missing}"
        )

    return (
        categorical_features,
        numerical_features,
        derived_numerical_features,
        audit_features,
        required_columns,
    )


def validate_dataset(name: str, df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Dataset {name} está sem as colunas obrigatórias: {missing_columns}"
        )


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VALID_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_column = "price"

    (
        categorical_features,
        numerical_features,
        derived_numerical_features,
        audit_features,
        required_columns,
    ) = get_feature_config(train_df)

    for name, df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        validate_dataset(name, df, required_columns)

        df[numerical_features] = df[numerical_features].astype(float)
        df[target_column] = df[target_column].astype(float)

        for col in categorical_features:
            df[col] = df[col].astype(str)

    input_features = categorical_features + numerical_features

    X_train = train_df[input_features]
    y_train = train_df[target_column]

    X_val = val_df[input_features]
    y_val = val_df[target_column]

    X_test = test_df[input_features]
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
        derived_numerical_features,
        audit_features,
    )


# ========================
# PREPROCESSING
# ========================

def build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
    derived_numerical_features: list[str],
) -> ColumnTransformer:
    all_numerical_features = numerical_features + derived_numerical_features

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
                all_numerical_features,
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
    explainability: str


def get_model_candidates() -> list[ModelCandidate]:
    return [
        ModelCandidate(
            name="LinearRegression",
            estimator=LinearRegression(),
            param_grid={},
            family="linear",
            explainability="high",
        ),
        ModelCandidate(
            name="Ridge",
            estimator=Ridge(),
            param_grid={
                "model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
            },
            family="linear_regularized",
            explainability="high",
        ),
        ModelCandidate(
            name="RandomForestRegressor",
            estimator=RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "model__n_estimators": [300],
                "model__max_depth": [18, None],
                "model__min_samples_leaf": [1, 2],
            },
            family="bagging_tree",
            explainability="medium",
        ),
        ModelCandidate(
            name="ExtraTreesRegressor",
            estimator=ExtraTreesRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "model__n_estimators": [300],
                "model__max_depth": [18, None],
                "model__min_samples_leaf": [1, 2],
            },
            family="bagging_tree",
            explainability="medium",
        ),
        ModelCandidate(
            name="GradientBoostingRegressor",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "model__n_estimators": [250],
                "model__learning_rate": [0.05, 0.08],
                "model__max_depth": [2, 3],
            },
            family="boosting_tree",
            explainability="medium",
        ),
        ModelCandidate(
            name="HistGradientBoostingRegressor",
            estimator=HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "model__max_iter": [250],
                "model__learning_rate": [0.05, 0.08],
                "model__max_leaf_nodes": [31],
                "model__l2_regularization": [0.0, 0.1],
            },
            family="modern_boosting_tree",
            explainability="medium",
        ),
    ]


def build_pipeline(
    candidate: ModelCandidate,
    categorical_features: list[str],
    numerical_features: list[str],
    derived_numerical_features: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        derived_numerical_features=derived_numerical_features,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", candidate.estimator),
        ]
    )


# ========================
# METRICS
# ========================

def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], np.ndarray]:
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
    preview["absolute_error"] = np.abs(preview["y_true"] - preview["y_pred"])
    preview.head(100).to_csv(path, index=False)


# ========================
# MODEL SELECTION
# ========================

def train_candidate(
    candidate: ModelCandidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: list[str],
    numerical_features: list[str],
    derived_numerical_features: list[str],
) -> dict[str, Any]:
    pipeline = build_pipeline(
        candidate=candidate,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        derived_numerical_features=derived_numerical_features,
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
    validation_metrics, validation_preds = evaluate(best_model, X_val, y_val)

    return {
        "name": candidate.name,
        "family": candidate.family,
        "explainability": candidate.explainability,
        "model": best_model,
        "best_params": search.best_params_,
        "cv_best_rmse": float(abs(search.best_score_)),
        "validation_metrics": validation_metrics,
        "validation_preds": validation_preds,
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
                "explainability": result["explainability"],
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
                "validation_mape": metrics["mape"],
                "cv_best_rmse": result["cv_best_rmse"],
                "training_seconds": result["training_seconds"],
                "best_params": json.dumps(result["best_params"], ensure_ascii=False),
            }
        )

    ranking_df = pd.DataFrame(rows)
    ranking_df = ranking_df.sort_values("validation_rmse", ascending=True)

    return ranking_df


# ========================
# EXPLAINABILITY
# ========================

def save_permutation_importance(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    path: Path,
) -> None:
    sample_size = min(len(X_val), PERMUTATION_IMPORTANCE_ROWS)
    X_sample = X_val.head(sample_size)
    y_sample = y_val.head(sample_size)

    result = permutation_importance(
        estimator=model,
        X=X_sample,
        y=y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean_rmse_increase": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_rmse_increase", ascending=False)

    importance_df.to_csv(path, index=False)


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
    print("[INFO] Iniciando seleção de modelos de regressão...")

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
        derived_numerical_features,
        audit_features,
    ) = load_data()

    candidates = get_model_candidates()
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="model-selection-regression-anticollinearity") as parent_run:
        mlflow.log_param("selection_strategy", "GridSearchCV")
        mlflow.log_param("primary_metric", PRIMARY_METRIC)
        mlflow.log_param("primary_metric_goal", "minimize")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("grid_n_jobs", GRID_N_JOBS)
        mlflow.log_param("candidate_count", len(candidates))
        mlflow.log_param("categorical_features", ",".join(categorical_features))
        mlflow.log_param("numerical_features", ",".join(numerical_features))
        mlflow.log_param("derived_numerical_features", ",".join(derived_numerical_features))
        mlflow.log_param("audit_features_excluded_from_training", ",".join(audit_features))
        mlflow.log_param("multicollinearity_policy", "exclude_state_region_city_when_base_price_m2_is_used")
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("tracking_uri", TRACKING_URI)

        print("\n[INFO] Features categóricas usadas no treino:", categorical_features)
        print("[INFO] Features numéricas usadas no treino:", numerical_features)
        print("[INFO] Features derivadas usadas no treino:", derived_numerical_features)
        print("[INFO] Features mantidas apenas para auditoria/drift:", audit_features)

        for candidate in candidates:
            print(f"\n[INFO] Treinando candidato: {candidate.name}")

            with mlflow.start_run(
                run_name=f"candidate-{candidate.name}",
                nested=True,
            ):
                mlflow.log_param("model_name", candidate.name)
                mlflow.log_param("family", candidate.family)
                mlflow.log_param("explainability", candidate.explainability)
                mlflow.log_param(
                    "param_grid",
                    json.dumps(candidate.param_grid, ensure_ascii=False),
                )

                try:
                    result = train_candidate(
                        candidate=candidate,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        categorical_features=categorical_features,
                        numerical_features=numerical_features,
                        derived_numerical_features=derived_numerical_features,
                    )

                    results.append(result)

                    mlflow.log_param(
                        "best_params",
                        json.dumps(result["best_params"], ensure_ascii=False),
                    )
                    mlflow.log_metric("cv_best_rmse", result["cv_best_rmse"])
                    mlflow.log_metric("training_seconds", result["training_seconds"])

                    for key, value in result["validation_metrics"].items():
                        mlflow.log_metric(f"validation_{key}", value)

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

        test_metrics, test_preds = evaluate(best_model, X_test, y_test)

        print("\n[RESULTADO] Melhor modelo encontrado")
        print(f"Modelo: {best_result['name']}")
        print(f"Família: {best_result['family']}")
        print(f"Interpretabilidade: {best_result['explainability']}")
        print(f"Melhores hiperparâmetros: {best_result['best_params']}")
        print(f"[VALIDATION] {best_result['validation_metrics']}")
        print(f"[TEST] {test_metrics}")

        ranking_df = build_ranking_dataframe(results)

        ranking_path = ARTIFACTS_DIR / "model_selection_ranking.csv"
        validation_preview_path = ARTIFACTS_DIR / "validation_predictions_preview.csv"
        test_preview_path = ARTIFACTS_DIR / "test_predictions_preview.csv"
        importance_path = ARTIFACTS_DIR / "best_model_permutation_importance.csv"

        ranking_df.to_csv(ranking_path, index=False)
        save_predictions_preview(
            X_val,
            y_val,
            best_result["validation_preds"],
            validation_preview_path,
        )
        save_predictions_preview(X_test, y_test, test_preds, test_preview_path)
        save_permutation_importance(best_model, X_val, y_val, importance_path)

        mlflow.log_artifact(ranking_path, artifact_path="model_selection")
        mlflow.log_artifact(validation_preview_path, artifact_path="analysis")
        mlflow.log_artifact(test_preview_path, artifact_path="analysis")
        mlflow.log_artifact(importance_path, artifact_path="explainability")

        mlflow.log_param("best_model_name", best_result["name"])
        mlflow.log_param("best_model_family", best_result["family"])
        mlflow.log_param("best_model_explainability", best_result["explainability"])
        mlflow.log_param(
            "best_model_params",
            json.dumps(best_result["best_params"], ensure_ascii=False),
        )

        for key, value in best_result["validation_metrics"].items():
            mlflow.log_metric(f"best_validation_{key}", value)

        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

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
                "derived_numerical_features": derived_numerical_features,
                "audit_features_excluded_from_training": audit_features,
                "multicollinearity_policy": (
                    "state, region, city and price_source are kept outside training; "
                    "base_price_m2 represents the external market reference."
                ),
            },
            "best_model": {
                "name": best_result["name"],
                "family": best_result["family"],
                "explainability": best_result["explainability"],
                "best_params": best_result["best_params"],
                "validation_metrics": best_result["validation_metrics"],
                "test_metrics": test_metrics,
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
                "permutation_importance": str(importance_path),
            },
        }

        summary_path = PROJECT_ROOT / "run_summary.json"
        selection_summary_path = PROJECT_ROOT / "model_selection_summary.json"

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        with selection_summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(summary_path, artifact_path="summary")
        mlflow.log_artifact(selection_summary_path, artifact_path="summary")

    print("\n[RESUMO FINAL]")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    print("[FINALIZADO]")


if __name__ == "__main__":
    main()
