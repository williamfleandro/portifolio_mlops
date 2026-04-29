from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference_apartments.csv"
CURRENT_PATH = PROJECT_ROOT / "data" / "current" / "current_apartments.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
HTML_REPORT_PATH = REPORTS_DIR / "data_drift_report.html"
JSON_RESULT_PATH = REPORTS_DIR / "drift_result.json"

FEATURES = [
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


def generate_reference_data(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    return pd.DataFrame(
        {
            "area_m2": rng.integers(35, 240, size=n_samples),
            "bedrooms": rng.integers(1, 5, size=n_samples),
            "bathrooms": rng.integers(1, 4, size=n_samples),
            "floor": rng.integers(0, 30, size=n_samples),
            "parking_spaces": rng.integers(0, 3, size=n_samples),
            "neighborhood_score": rng.uniform(4.0, 10.0, size=n_samples).round(2),
            "condo_fee": rng.uniform(250, 2200, size=n_samples).round(2),
            "age_years": rng.integers(0, 45, size=n_samples),
            "distance_to_center_km": rng.uniform(0.5, 28.0, size=n_samples).round(2),
        }
    )


def generate_current_data_with_drift(n_samples: int = 800, random_state: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    return pd.DataFrame(
        {
            "area_m2": rng.integers(70, 280, size=n_samples),
            "bedrooms": rng.integers(2, 5, size=n_samples),
            "bathrooms": rng.integers(1, 5, size=n_samples),
            "floor": rng.integers(0, 35, size=n_samples),
            "parking_spaces": rng.integers(0, 4, size=n_samples),
            "neighborhood_score": rng.uniform(6.0, 10.0, size=n_samples).round(2),
            "condo_fee": rng.uniform(400, 2800, size=n_samples).round(2),
            "age_years": rng.integers(0, 35, size=n_samples),
            "distance_to_center_km": rng.uniform(0.5, 20.0, size=n_samples).round(2),
        }
    )


def ensure_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not REFERENCE_PATH.exists():
        reference = generate_reference_data()
        reference.to_csv(REFERENCE_PATH, index=False)

    if not CURRENT_PATH.exists():
        current = generate_current_data_with_drift()
        current.to_csv(CURRENT_PATH, index=False)

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    return reference[FEATURES], current[FEATURES]


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(reference_data=reference, current_data=current)

    # Evidently API nova: salvar HTML a partir do snapshot/evaluation result
    if hasattr(snapshot, "save_html"):
        snapshot.save_html(str(HTML_REPORT_PATH))
    elif hasattr(snapshot, "save"):
        snapshot.save(str(HTML_REPORT_PATH))
    else:
        HTML_REPORT_PATH.write_text(
            f"""
            <html>
              <head><title>Data Drift Report</title></head>
              <body>
                <h1>Data Drift Report</h1>
                <p>Relatório visual não disponível nesta versão da API do Evidently.</p>
                <p>Use o arquivo JSON para análise programática.</p>
              </body>
            </html>
            """,
            encoding="utf-8",
        )

    result = {
        "report_path": str(HTML_REPORT_PATH),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "features": FEATURES,
        "note": "Abra o HTML para verificar o relatório visual. Use o JSON para automação.",
    }

    try:
        if hasattr(snapshot, "dict"):
            result["raw_snapshot"] = snapshot.dict()
        elif hasattr(snapshot, "as_dict"):
            result["raw_snapshot"] = snapshot.as_dict()
        elif hasattr(snapshot, "json"):
            result["raw_snapshot"] = snapshot.json()
        else:
            result["raw_snapshot"] = None
    except Exception:
        result["raw_snapshot"] = None

    return result


def decide_retrain(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    drifted_features = []

    for col in FEATURES:
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        ref_std = reference[col].std()

        if ref_std == 0:
            continue

        z_shift = abs(cur_mean - ref_mean) / ref_std

        if z_shift >= 0.5:
            drifted_features.append(
                {
                    "feature": col,
                    "reference_mean": round(float(ref_mean), 4),
                    "current_mean": round(float(cur_mean), 4),
                    "z_shift": round(float(z_shift), 4),
                }
            )

    drift_ratio = len(drifted_features) / len(FEATURES)

    return {
        "drifted_features_count": len(drifted_features),
        "total_features": len(FEATURES),
        "drift_ratio": round(drift_ratio, 4),
        "drifted_features": drifted_features,
        "retrain_required": drift_ratio >= 0.3,
        "rule": "Retreino sugerido quando 30% ou mais das features apresentam z_shift >= 0.5.",
    }


def main() -> None:
    reference, current = ensure_data()

    report_result = run_drift_report(reference, current)
    retrain_decision = decide_retrain(reference, current)

    final_result = {
        **report_result,
        **retrain_decision,
    }

    with JSON_RESULT_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()