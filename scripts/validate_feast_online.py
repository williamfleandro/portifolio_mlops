from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store"
CURRENT_PATH = PROJECT_ROOT / "data" / "current" / "current_predictions.parquet"


ONLINE_FEATURES = [
    "apartment_prediction_features:property_type",
    "apartment_prediction_features:city",
    "apartment_prediction_features:neighborhood",
    "apartment_prediction_features:base_price_m2",
    "apartment_prediction_features:area_m2",
    "apartment_prediction_features:bedrooms",
    "apartment_prediction_features:bathrooms",
    "apartment_prediction_features:floor",
    "apartment_prediction_features:parking_spaces",
    "apartment_prediction_features:neighborhood_score",
    "apartment_prediction_features:condo_fee",
    "apartment_prediction_features:age_years",
    "apartment_prediction_features:distance_to_center_km",
    "apartment_prediction_features:ml_prediction_value",
    "apartment_prediction_features:model_version",
    "apartment_prediction_features:model_alias",
]


def main() -> None:
    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CURRENT_PATH}. "
            "Execute primeiro o consumer Kafka para gerar o Parquet."
        )

    current = pd.read_parquet(CURRENT_PATH)

    if current.empty:
        raise RuntimeError("O Parquet current_predictions.parquet está vazio.")

    latest = (
        current
        .sort_values("event_timestamp")
        .tail(1)
        .iloc[0]
    )

    property_profile_id = str(latest["property_profile_id"])

    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    result = store.get_online_features(
        features=ONLINE_FEATURES,
        entity_rows=[
            {
                "property_profile_id": property_profile_id,
            }
        ],
    ).to_dict()

    print("\n[Feast online retrieval OK]")
    print(f"property_profile_id: {property_profile_id}")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()