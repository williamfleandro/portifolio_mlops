from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMOTION_RESULT_PATH = REPORTS_DIR / "promotion_result.json"

MODEL_NAME = os.getenv("MODEL_NAME", "apartment-price-regression")
CANDIDATE_ALIAS = os.getenv("CANDIDATE_ALIAS", "candidate")
CHAMPION_ALIAS = os.getenv("CHAMPION_ALIAS", "champion")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

BACKEND_MODEL_LOAD_URL = os.getenv(
    "BACKEND_MODEL_LOAD_URL",
    "http://localhost:30081/model/load",
)

BACKEND_MODEL_URL = os.getenv(
    "BACKEND_MODEL_URL",
    "http://localhost:30081/model",
)

BACKEND_HEALTH_URL = os.getenv(
    "BACKEND_HEALTH_URL",
    "http://localhost:30081/health",
)

MIN_TEST_R2 = float(os.getenv("PROMOTION_MIN_TEST_R2", "0.98"))
MAX_TEST_MAPE = float(os.getenv("PROMOTION_MAX_TEST_MAPE", "6.0"))
MAX_TEST_RMSE = float(os.getenv("PROMOTION_MAX_TEST_RMSE", "100000"))
MAX_SMOKE_SECONDS = float(os.getenv("PROMOTION_MAX_SMOKE_SECONDS", "0.1"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_result(payload: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROMOTION_RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )

    with request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body) if response_body else {}


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    req = request.Request(url=url, method="GET")

    with request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
        return json.loads(response_body) if response_body else {}


def require_metric(metrics: dict[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(
            f"Métrica obrigatória ausente no run do candidate: {name}. "
            f"Métricas disponíveis: {sorted(metrics.keys())}"
        )

    return float(metrics[name])


def evaluate_gates(metrics: dict[str, float]) -> tuple[bool, list[dict[str, Any]]]:
    checks = [
        {
            "metric": "test_r2",
            "value": require_metric(metrics, "test_r2"),
            "rule": f">= {MIN_TEST_R2}",
            "passed": require_metric(metrics, "test_r2") >= MIN_TEST_R2,
        },
        {
            "metric": "test_mape",
            "value": require_metric(metrics, "test_mape"),
            "rule": f"<= {MAX_TEST_MAPE}",
            "passed": require_metric(metrics, "test_mape") <= MAX_TEST_MAPE,
        },
        {
            "metric": "test_rmse",
            "value": require_metric(metrics, "test_rmse"),
            "rule": f"<= {MAX_TEST_RMSE}",
            "passed": require_metric(metrics, "test_rmse") <= MAX_TEST_RMSE,
        },
        {
            "metric": "smoke_predict_seconds",
            "value": require_metric(metrics, "smoke_predict_seconds"),
            "rule": f"<= {MAX_SMOKE_SECONDS}",
            "passed": require_metric(metrics, "smoke_predict_seconds") <= MAX_SMOKE_SECONDS,
        },
    ]

    return all(item["passed"] for item in checks), checks


def print_gate_report(checks: list[dict[str, Any]]) -> None:
    print("\nCritérios de promoção candidate -> champion:")

    for item in checks:
        status = "OK" if item["passed"] else "FALHOU"
        print(
            f"- {status}: {item['metric']} = {item['value']} "
            f"regra {item['rule']}"
        )


def main() -> int:
    print("")
    print("========================================")
    print(" PROMOTION PIPELINE - CANDIDATE -> CHAMPION")
    print("========================================")
    print("")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    result: dict[str, Any] = {
        "timestamp": now_utc(),
        "model_name": MODEL_NAME,
        "candidate_alias": CANDIDATE_ALIAS,
        "champion_alias": CHAMPION_ALIAS,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "status": "started",
    }

    try:
        candidate_version = client.get_model_version_by_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS,
        )

        candidate_version_number = str(candidate_version.version)
        candidate_run_id = candidate_version.run_id

        print(f"Candidate encontrado: {MODEL_NAME}@{CANDIDATE_ALIAS}")
        print(f"Candidate version: v{candidate_version_number}")
        print(f"Candidate run_id: {candidate_run_id}")

        run = client.get_run(candidate_run_id)
        metrics = {key: float(value) for key, value in run.data.metrics.items()}

        passed, checks = evaluate_gates(metrics)

        print_gate_report(checks)

        result.update(
            {
                "candidate_version": candidate_version_number,
                "candidate_run_id": candidate_run_id,
                "checks": checks,
                "metrics": {
                    "test_r2": metrics.get("test_r2"),
                    "test_mape": metrics.get("test_mape"),
                    "test_rmse": metrics.get("test_rmse"),
                    "smoke_predict_seconds": metrics.get("smoke_predict_seconds"),
                },
            }
        )

        if not passed:
            result["status"] = "promotion_blocked"
            result["message"] = (
                "Promoção bloqueada. O candidate não passou em todos os critérios."
            )
            save_result(result)

            print("")
            print("PROMOÇÃO BLOQUEADA")
            print("O modelo candidate NÃO foi promovido para champion.")
            print(f"Resultado salvo em: {PROMOTION_RESULT_PATH}")

            return 2

        print("")
        print("Todos os critérios foram aprovados.")
        print(f"Promovendo {MODEL_NAME} v{candidate_version_number} para alias champion...")

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=candidate_version_number,
        )

        champion_version = client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )

        print(f"Alias champion atualizado para v{champion_version.version}")

        print("")
        print("Carregando champion explicitamente no backend via /model/load...")

        load_payload = {
            "model_name": MODEL_NAME,
            "model_uri": f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}",
            "model_version": str(champion_version.version),
            "source_type": "mlflow-registry",
        }

        backend_load_response = post_json(
            BACKEND_MODEL_LOAD_URL,
            load_payload,
            timeout=120,
        )

        print(json.dumps(backend_load_response, ensure_ascii=False, indent=2))

        print("")
        print("Validando /model...")
        backend_model_response = get_json(BACKEND_MODEL_URL)
        print(json.dumps(backend_model_response, ensure_ascii=False, indent=2))

        print("")
        print("Validando /health...")
        backend_health_response = get_json(BACKEND_HEALTH_URL)
        print(json.dumps(backend_health_response, ensure_ascii=False, indent=2))

        backend_loaded_version = str(backend_model_response.get("model_version"))
        backend_loaded_alias = str(backend_model_response.get("model_alias"))

        if backend_loaded_version != str(champion_version.version):
            raise RuntimeError(
                "Backend não carregou a versão promovida. "
                f"Esperado v{champion_version.version}, recebido v{backend_loaded_version}"
            )

        if backend_loaded_alias != CHAMPION_ALIAS:
            raise RuntimeError(
                "Backend não carregou o alias champion. "
                f"Esperado {CHAMPION_ALIAS}, recebido {backend_loaded_alias}"
            )

        if backend_health_response.get("status") != "ok":
            raise RuntimeError(
                f"Backend /health não retornou ok: {backend_health_response}"
            )

        result.update(
            {
                "status": "promoted",
                "champion_version": str(champion_version.version),
                "backend_load_response": backend_load_response,
                "backend_model_response": backend_model_response,
                "backend_health_response": backend_health_response,
                "message": "Candidate promovido para champion e backend carregado com sucesso.",
            }
        )

        save_result(result)

        print("")
        print("PROMOÇÃO CONCLUÍDA COM SUCESSO")
        print(f"Champion atual: {MODEL_NAME}@{CHAMPION_ALIAS} v{champion_version.version}")
        print(f"Resultado salvo em: {PROMOTION_RESULT_PATH}")

        return 0

    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")

        result.update(
            {
                "status": "error",
                "error": f"HTTPError: {exc}",
                "error_body": error_body,
            }
        )

        save_result(result)

        print("")
        print("ERRO NA PROMOÇÃO")
        print(f"HTTPError: {exc}")
        print(error_body)

        return 1

    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error": str(exc),
            }
        )

        save_result(result)

        print("")
        print("ERRO NA PROMOÇÃO")
        print(str(exc))

        return 1


if __name__ == "__main__":
    raise SystemExit(main())