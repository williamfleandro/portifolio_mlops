from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO_PATH = PROJECT_ROOT / "feature_store"
CURRENT_PATH = PROJECT_ROOT / "data" / "current" / "current_predictions.parquet"


FEATURES = [
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

    entity_df = (
        current[["property_profile_id", "event_timestamp"]]
        .drop_duplicates()
        .sort_values("event_timestamp")
        .tail(10)
        .reset_index(drop=True)
    )

    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    result = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()

    print("\n[Feast offline retrieval OK]")
    print(result.head(10).to_string(index=False))
    print("\nShape:", result.shape)


if __name__ == "__main__":
    main()