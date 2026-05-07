from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request, error


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LATEST_METRICS_PATH = PROJECT_ROOT / "reports" / "latest_drift_metrics.json"
DRIFT_RESULT_PATH = PROJECT_ROOT / "reports" / "drift_result.json"
RETRAIN_HISTORY_PATH = PROJECT_ROOT / "reports" / "retrain_history.jsonl"

# Ajuste aqui se o seu script de treino estiver em outro caminho.
TRAIN_SCRIPT_CANDIDATES = [
    PROJECT_ROOT / "apartament-price-regression" / "train.py"
]

DATA_DRIFT_THRESHOLD = float(os.getenv("DATA_DRIFT_THRESHOLD", "0.30"))
PREDICTION_DRIFT_THRESHOLD = float(os.getenv("PREDICTION_DRIFT_THRESHOLD", "0.30"))

BACKEND_RELOAD_URL = os.getenv(
    "BACKEND_RELOAD_URL",
    "http://localhost:30081/model/reload",
)

RUN_MODEL_RELOAD = os.getenv("RUN_MODEL_RELOAD", "true").lower() == "true"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Arquivo JSON inválido: {path}. Erro: {exc}") from exc


def load_drift_metrics() -> dict[str, Any]:
    """
    Fonte principal:
        reports/latest_drift_metrics.json

    Fallback:
        reports/drift_result.json -> latest_metrics
    """

    latest_metrics = load_json(LATEST_METRICS_PATH)

    if latest_metrics is not None:
        return latest_metrics

    drift_result = load_json(DRIFT_RESULT_PATH)

    if drift_result is None:
        raise FileNotFoundError(
            "Nenhum arquivo de drift encontrado. Rode primeiro: "
            "python .\\monitoring\\check_drift.py"
        )

    if "latest_metrics" in drift_result:
        return drift_result["latest_metrics"]

    return {
        "data_drift_score": float(drift_result.get("drift_ratio", 0.0)),
        "prediction_drift_score": 0.0,
        "drift_detected": bool(drift_result.get("retrain_required", False)),
        "retrain_required": bool(drift_result.get("retrain_required", False)),
        "drifted_features_count": int(drift_result.get("drifted_features_count", 0)),
        "total_features_count": int(drift_result.get("total_features", 0)),
    }


def should_retrain(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    data_drift_score = float(metrics.get("data_drift_score", 0.0))
    prediction_drift_score = float(metrics.get("prediction_drift_score", 0.0))
    drift_detected = bool(metrics.get("drift_detected", False))
    retrain_required = bool(metrics.get("retrain_required", False))

    if retrain_required:
        reasons.append("retrain_required=true")

    if drift_detected:
        reasons.append("drift_detected=true")

    if data_drift_score >= DATA_DRIFT_THRESHOLD:
        reasons.append(
            f"data_drift_score={data_drift_score} >= threshold={DATA_DRIFT_THRESHOLD}"
        )

    if prediction_drift_score >= PREDICTION_DRIFT_THRESHOLD:
        reasons.append(
            f"prediction_drift_score={prediction_drift_score} >= threshold={PREDICTION_DRIFT_THRESHOLD}"
        )

    return len(reasons) > 0, reasons


def find_train_script() -> Path:
    for script_path in TRAIN_SCRIPT_CANDIDATES:
        if script_path.exists():
            return script_path

    candidates = "\n".join(str(path) for path in TRAIN_SCRIPT_CANDIDATES)

    raise FileNotFoundError(
        "Script de treinamento não encontrado. Caminhos verificados:\n"
        f"{candidates}\n\n"
        "Ajuste TRAIN_SCRIPT_CANDIDATES no monitoring/retrain_if_needed.py."
    )


def run_training(train_script: Path) -> dict[str, Any]:
    command = [sys.executable, str(train_script)]

    print(f"Executando treinamento: {' '.join(command)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )

    result = {
        "train_script": str(train_script),
        "return_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "success": process.returncode == 0,
    }

    if process.stdout:
        print("\nSTDOUT do treinamento:")
        print(process.stdout)

    if process.stderr:
        print("\nSTDERR do treinamento:")
        print(process.stderr)

    if process.returncode != 0:
        raise RuntimeError(
            f"Falha no treinamento. Código de retorno: {process.returncode}"
        )

    return result


def reload_backend_model() -> dict[str, Any]:
    if not RUN_MODEL_RELOAD:
        return {
            "reload_attempted": False,
            "reason": "RUN_MODEL_RELOAD=false",
        }

    print(f"Chamando reload do backend: {BACKEND_RELOAD_URL}")

    req = request.Request(
        BACKEND_RELOAD_URL,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")

            return {
                "reload_attempted": True,
                "status_code": response.status,
                "response": body,
                "success": 200 <= response.status < 300,
            }

    except error.URLError as exc:
        return {
            "reload_attempted": True,
            "success": False,
            "error": str(exc),
        }


def append_history(event: dict[str, Any]) -> None:
    RETRAIN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RETRAIN_HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    print("Verificando necessidade de retreinamento...")

    try:
        metrics = load_drift_metrics()
        retrain, reasons = should_retrain(metrics)

        event: dict[str, Any] = {
            "timestamp": now_utc(),
            "metrics": metrics,
            "thresholds": {
                "data_drift_threshold": DATA_DRIFT_THRESHOLD,
                "prediction_drift_threshold": PREDICTION_DRIFT_THRESHOLD,
            },
            "retrain_decision": retrain,
            "reasons": reasons,
        }

        print("\nMétricas de drift carregadas:")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        if not retrain:
            print("\nSem drift relevante. Nenhuma ação necessária.")
            event["status"] = "skipped"
            append_history(event)
            return

        print("\nDrift relevante detectado. Iniciando retreinamento.")
        print("Motivos:")
        for reason in reasons:
            print(f"- {reason}")

        train_script = find_train_script()
        training_result = run_training(train_script)

        event["training"] = training_result
        event["status"] = "training_completed"

        reload_result = reload_backend_model()
        event["backend_reload"] = reload_result

        append_history(event)

        print("\nRetreinamento concluído.")
        print(f"Histórico salvo em: {RETRAIN_HISTORY_PATH}")

        if reload_result.get("reload_attempted"):
            print("\nResultado do reload do backend:")
            print(json.dumps(reload_result, ensure_ascii=False, indent=2))

    except Exception as exc:
        error_event = {
            "timestamp": now_utc(),
            "status": "error",
            "error": str(exc),
        }

        append_history(error_event)

        print(f"\nErro no processo de retreinamento: {exc}")
        raise


if __name__ == "__main__":
    main()