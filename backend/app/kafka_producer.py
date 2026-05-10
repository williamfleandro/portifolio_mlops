from __future__ import annotations

import json
import os
from threading import Event
from typing import Any, Mapping, Optional

from confluent_kafka import KafkaException, Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
KAFKA_PREDICTION_TOPIC = os.getenv(
    "KAFKA_PREDICTION_TOPIC",
    "apartment-price-predictions",
)
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_PRODUCER_CLIENT_ID = os.getenv(
    "KAFKA_PRODUCER_CLIENT_ID",
    "apartment-price-backend-producer",
)
KAFKA_PRODUCER_ENABLED = os.getenv("KAFKA_PRODUCER_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}
KAFKA_PRODUCER_FLUSH_TIMEOUT_SECONDS = float(
    os.getenv("KAFKA_PRODUCER_FLUSH_TIMEOUT_SECONDS", "10")
)


def json_serializer(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def build_event_key(event: Mapping[str, Any]) -> str:
    event_id = event.get("event_id")

    if event_id:
        return str(event_id)

    features = event.get("features", {})

    property_type = features.get("property_type", "unknown")
    city = features.get("city", "unknown")

    return f"{city}:{property_type}"


class KafkaPredictionProducer:
    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        security_protocol: Optional[str] = None,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self.topic = topic or KAFKA_PREDICTION_TOPIC
        self.security_protocol = security_protocol or KAFKA_SECURITY_PROTOCOL
        self.enabled = KAFKA_PRODUCER_ENABLED

        self._producer: Optional[Producer] = None

        if self.enabled:
            self._producer = self._build_producer()

    def _build_producer(self) -> Producer:
        if not self.bootstrap_servers:
            raise RuntimeError(
                "KAFKA_BOOTSTRAP_SERVERS não foi configurado. "
                "Configure os brokers Kafka no backend-rollout.yaml."
            )

        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": KAFKA_PRODUCER_CLIENT_ID,
            "security.protocol": self.security_protocol,
            "acks": "all",
            "enable.idempotence": True,
            "retries": 5,
            "linger.ms": 10,
            "request.timeout.ms": 30000,
            "delivery.timeout.ms": 60000,
            "message.send.max.retries": 5,
        }

        return Producer(config)

    def publish_prediction_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "disabled",
                "topic": self.topic,
                "partition": None,
                "offset": None,
                "key": None,
                "event_id": str(event.get("event_id", "")),
                "message": "Kafka producer desabilitado por KAFKA_PRODUCER_ENABLED=false.",
            }

        if self._producer is None:
            self._producer = self._build_producer()

        delivery_finished = Event()
        delivery_result: dict[str, Any] = {}

        key = build_event_key(event)
        value = json_serializer(event)

        def delivery_callback(error, message) -> None:
            if error is not None:
                delivery_result["error"] = str(error)
            else:
                delivery_result.update(
                    {
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                        "key": key,
                    }
                )

            delivery_finished.set()

        try:
            self._producer.produce(
                topic=self.topic,
                key=key.encode("utf-8"),
                value=value.encode("utf-8"),
                on_delivery=delivery_callback,
            )

            self._producer.poll(0)
            self._producer.flush(KAFKA_PRODUCER_FLUSH_TIMEOUT_SECONDS)

            if not delivery_finished.is_set():
                raise TimeoutError(
                    "Timeout aguardando confirmação de entrega da mensagem Kafka."
                )

            if "error" in delivery_result:
                raise KafkaException(delivery_result["error"])

            return {
                "status": "published",
                "topic": delivery_result.get("topic", self.topic),
                "partition": delivery_result.get("partition"),
                "offset": delivery_result.get("offset"),
                "key": delivery_result.get("key", key),
                "event_id": str(event.get("event_id", "")),
                "message": "Evento de predição publicado no Kafka com sucesso.",
            }

        except Exception as exc:
            raise RuntimeError(f"Falha ao publicar evento no Kafka: {exc}") from exc


_kafka_prediction_producer: KafkaPredictionProducer | None = None


def get_kafka_prediction_producer() -> KafkaPredictionProducer:
    global _kafka_prediction_producer

    if _kafka_prediction_producer is None:
        _kafka_prediction_producer = KafkaPredictionProducer()

    return _kafka_prediction_producer