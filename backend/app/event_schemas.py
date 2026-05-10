from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_event_id() -> str:
    return str(uuid4())


class PredictionModelInfo(BaseModel):
    model_name: str = Field(default="apartment-price-regression")
    model_uri: str = Field(default="models:/apartment-price-regression@candidate")
    model_version: Optional[str] = Field(default=None)
    model_alias: Optional[str] = Field(default="candidate")


class PredictionTags(BaseModel):
    baseline_estimate_value: float
    ml_prediction_value: float
    prediction_difference_value: float
    prediction_difference_percent: float


class PredictionFeatures(BaseModel):
    property_type: str
    property_type_label: Optional[str] = None
    city: str
    neighborhood: str

    base_price_m2: float
    area_m2: float
    bedrooms: float
    bathrooms: float
    floor: float
    parking_spaces: float
    neighborhood_score: float
    condo_fee: float
    age_years: float
    distance_to_center_km: float


class PredictionEvent(BaseModel):
    event_id: str = Field(default_factory=new_event_id)
    event_type: str = Field(default="apartment_price_prediction")
    event_version: str = Field(default="1.0")
    event_timestamp: str = Field(default_factory=utc_now_iso)

    source_system: str = Field(default="portfolio-mlops-frontend")
    target_topic: str = Field(default="apartment-price-predictions")
    endpoint: str = Field(default="/predict")

    model: PredictionModelInfo
    prediction_tags: PredictionTags
    features: PredictionFeatures


class KafkaPublishResponse(BaseModel):
    status: str
    topic: str
    partition: Optional[int] = None
    offset: Optional[int] = None
    key: Optional[str] = None
    event_id: str
    message: Optional[str] = None