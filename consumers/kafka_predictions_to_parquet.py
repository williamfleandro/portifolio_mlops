from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Consumer, KafkaException


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "current" / "current_predictions.parquet"

REQUIRED_FEATURES = [
    "property_type",
    "city",
    "neighborhood",
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

NUMERIC_FEATURES = [
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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_timestamp(value: Any) -> pd.Timestamp:
    if value in {None, ""}:
        return pd.Timestamp.now(tz="UTC")

    return pd.to_datetime(value, utc=True)


def build_property_profile_id(row: dict[str, Any]) -> str:
    """
    Cria uma chave determinística para o perfil do imóvel.

    Não é uma chave de imóvel real. É uma chave de perfil operacional
    para permitir versionamento e consulta de features por Entity no Feast.
    """
    key_fields = [
        "property_type",
        "city",
        "neighborhood",
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

    raw_key = "|".join(str(row.get(field, "")) for field in key_fields)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]


def parse_prediction_event(raw_value: bytes) -> dict[str, Any] | None:
    try:
        event = json.loads(raw_value.decode("utf-8"))
    except Exception as exc:
        print(f"[skip] JSON inválido: {exc}")
        return None

    features = event.get("features") or {}
    prediction_tags = event.get("prediction_tags") or {}
    model = event.get("model") or {}

    if not features:
        print("[skip] Evento sem bloco features. Provavelmente é o teste manual antigo.")
        return None

    missing = [field for field in REQUIRED_FEATURES if field not in features]

    if missing:
        print(f"[skip] Evento sem features obrigatórias: {missing}")
        return None

    row: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type", "apartment_price_prediction"),
        "event_version": event.get("event_version", "1.0"),
        "event_timestamp": normalize_timestamp(event.get("event_timestamp")),
        "created_timestamp": pd.Timestamp.now(tz="UTC"),
        "source_system": event.get("source_system"),
        "endpoint": event.get("endpoint"),
        "target_topic": event.get("target_topic"),
        "model_name": model.get("model_name"),
        "model_uri": model.get("model_uri"),
        "model_version": model.get("model_version"),
        "model_alias": model.get("model_alias"),
        "baseline_estimate_value": prediction_tags.get("baseline_estimate_value"),
        "ml_prediction_value": prediction_tags.get("ml_prediction_value"),
        "prediction_difference_value": prediction_tags.get("prediction_difference_value"),
        "prediction_difference_percent": prediction_tags.get("prediction_difference_percent"),
    }

    for field in REQUIRED_FEATURES:
        row[field] = features.get(field)

    row["property_type_label"] = features.get("property_type_label")

    for field in NUMERIC_FEATURES:
        row[field] = float(row[field])

    for field in [
        "baseline_estimate_value",
        "ml_prediction_value",
        "prediction_difference_value",
        "prediction_difference_percent",
    ]:
        if row[field] is not None:
            row[field] = float(row[field])

    row["property_profile_id"] = build_property_profile_id(row)

    return row


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def save_dataset(rows: list[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if new_df.empty:
        return 0

    existing_df = load_existing(output_path)

    full_df = pd.concat([existing_df, new_df], ignore_index=True)

    if "event_id" in full_df.columns:
        full_df = full_df.drop_duplicates(subset=["event_id"], keep="last")

    full_df = full_df.sort_values("event_timestamp").reset_index(drop=True)

    full_df.to_parquet(output_path, index=False)

    return len(full_df)


def build_consumer(args: argparse.Namespace) -> Consumer:
    config = {
        "bootstrap.servers": args.bootstrap_servers,
        "group.id": args.group_id,
        "auto.offset.reset": "earliest" if args.from_beginning else "latest",
        "enable.auto.commit": True,
        "security.protocol": args.security_protocol,
        "client.id": "feast-current-dataset-builder",
    }

    return Consumer(config)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "192.168.56.11:9092,192.168.56.12:9092,192.168.56.13:9092",
        ),
    )
    parser.add_argument(
        "--topic",
        default=os.getenv("KAFKA_PREDICTION_TOPIC", "apartment-price-predictions"),
    )
    parser.add_argument(
        "--group-id",
        default=os.getenv("KAFKA_CONSUMER_GROUP_ID", "feast-current-dataset-builder"),
    )
    parser.add_argument(
        "--security-protocol",
        default=os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_OUTPUT_PATH),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--from-beginning",
        action="store_true",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path).resolve()

    consumer = build_consumer(args)
    consumer.subscribe([args.topic])

    print(f"[consumer] topic={args.topic}")
    print(f"[consumer] bootstrap_servers={args.bootstrap_servers}")
    print(f"[consumer] output_path={output_path}")

    rows: list[dict[str, Any]] = []
    valid_messages = 0
    last_message_time = time.time()

    try:
        while valid_messages < args.max_messages:
            message = consumer.poll(1.0)

            if message is None:
                if time.time() - last_message_time >= args.idle_timeout_seconds:
                    print("[consumer] Timeout sem novas mensagens. Finalizando.")
                    break
                continue

            last_message_time = time.time()

            if message.error():
                raise KafkaException(message.error())

            row = parse_prediction_event(message.value())

            if row is None:
                continue

            rows.append(row)
            valid_messages += 1

            print(
                "[ok] "
                f"event_id={row.get('event_id')} "
                f"city={row.get('city')} "
                f"prediction={row.get('ml_prediction_value')}"
            )

        total_rows = save_dataset(rows, output_path)

        print(f"[done] Novas mensagens válidas: {len(rows)}")
        print(f"[done] Total acumulado no Parquet: {total_rows}")
        print(f"[done] Arquivo: {output_path}")

    finally:
        consumer.close()


if __name__ == "__main__":
    main()