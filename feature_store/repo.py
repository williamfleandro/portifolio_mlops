from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, String


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CURRENT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "current" / "current_predictions.parquet"
)


property_profile = Entity(
    name="property_profile",
    value_type=ValueType.STRING,
    join_keys=["property_profile_id"],
    description=(
        "Perfil determinístico do imóvel criado a partir das features "
        "do evento de predição."
    ),
)


current_predictions_source = FileSource(
    name="current_predictions_source",
    path=str(CURRENT_PREDICTIONS_PATH),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description=(
        "Eventos de predição consumidos do Kafka topic "
        "apartment-price-predictions e persistidos em Parquet."
    ),
)


apartment_prediction_features = FeatureView(
    name="apartment_prediction_features",
    entities=[property_profile],
    ttl=timedelta(days=365),
    online=True,
    source=current_predictions_source,
    schema=[
        Field(name="property_type", dtype=String),
        Field(name="property_type_label", dtype=String),
        Field(name="city", dtype=String),
        Field(name="neighborhood", dtype=String),
        Field(name="base_price_m2", dtype=Float64),
        Field(name="area_m2", dtype=Float64),
        Field(name="bedrooms", dtype=Float64),
        Field(name="bathrooms", dtype=Float64),
        Field(name="floor", dtype=Float64),
        Field(name="parking_spaces", dtype=Float64),
        Field(name="neighborhood_score", dtype=Float64),
        Field(name="condo_fee", dtype=Float64),
        Field(name="age_years", dtype=Float64),
        Field(name="distance_to_center_km", dtype=Float64),
        Field(name="baseline_estimate_value", dtype=Float64),
        Field(name="ml_prediction_value", dtype=Float64),
        Field(name="prediction_difference_value", dtype=Float64),
        Field(name="prediction_difference_percent", dtype=Float64),
        Field(name="model_name", dtype=String),
        Field(name="model_uri", dtype=String),
        Field(name="model_version", dtype=String),
        Field(name="model_alias", dtype=String),
        Field(name="source_system", dtype=String),
        Field(name="endpoint", dtype=String),
    ],
    tags={
        "domain": "real_estate",
        "source": "kafka",
        "topic": "apartment-price-predictions",
        "stage": "current_inference_log",
    },
)