"""Consumer helpers for authenticating and parsing Hindsight webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import TypeAlias

from pydantic import ValidationError

from hindsight_client_api.models.consolidation_completed_webhook_event import (
    ConsolidationCompletedWebhookEvent,
)
from hindsight_client_api.models.memory_defense_triggered_webhook_event import (
    MemoryDefenseTriggeredWebhookEvent,
)
from hindsight_client_api.models.retain_completed_webhook_event import RetainCompletedWebhookEvent
from hindsight_client_api.models.webhook_event_envelope import WebhookEventEnvelope

SIGNATURE_HEADER = "X-Hindsight-Signature"
_SIGNATURE_PATTERN = re.compile(r"sha256=[0-9a-f]{64}\Z")
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)

ConsumedWebhookEvent: TypeAlias = (
    ConsolidationCompletedWebhookEvent
    | RetainCompletedWebhookEvent
    | MemoryDefenseTriggeredWebhookEvent
    | WebhookEventEnvelope
)

_EVENT_MODELS = {
    "consolidation.completed": ConsolidationCompletedWebhookEvent,
    "retain.completed": RetainCompletedWebhookEvent,
    "memory_defense.triggered": MemoryDefenseTriggeredWebhookEvent,
}


class WebhookError(ValueError):
    """Base error raised by webhook consumer helpers."""


class WebhookSignatureError(WebhookError):
    """Raised when a webhook signature is absent, malformed, or invalid."""


class WebhookPayloadError(WebhookError):
    """Raised when a verified webhook body does not match the event contract."""


def verify_signature(raw_body: bytes, signature: str | None, secret: str | bytes) -> bool:
    """Verify the current ``sha256=<hex>`` signature over exact raw bytes."""
    if not isinstance(raw_body, bytes):
        raise TypeError("raw_body must be bytes")
    if not isinstance(signature, str) or _SIGNATURE_PATTERN.fullmatch(signature) is None:
        return False

    secret_bytes = secret.encode() if isinstance(secret, str) else secret
    if not isinstance(secret_bytes, bytes):
        raise TypeError("secret must be str or bytes")

    expected = "sha256=" + hmac.new(secret_bytes, raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def parse_event(raw_body: bytes) -> ConsumedWebhookEvent:
    """Parse an event without authenticating it.

    Prefer :func:`construct_event` for signed webhooks. This separate entry
    point is intended for webhooks explicitly configured without a secret.
    """
    if not isinstance(raw_body, bytes):
        raise TypeError("raw_body must be bytes")

    try:
        payload = json.loads(raw_body)
        if not isinstance(payload, dict):
            raise TypeError("the payload must be a JSON object")
        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, str) or _RFC3339_PATTERN.fullmatch(timestamp) is None:
            raise ValueError("timestamp must be an RFC 3339 date-time with a timezone")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        event_name = payload.get("event")
        model = _EVENT_MODELS.get(event_name, WebhookEventEnvelope)
        event = model.from_dict(payload)
        if event is None:
            raise TypeError("the payload must be a JSON object")
        return event
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise WebhookPayloadError(f"invalid webhook payload: {exc}") from exc


def construct_event(
    raw_body: bytes,
    headers: Mapping[str, str],
    secret: str | bytes,
) -> ConsumedWebhookEvent:
    """Verify a signed raw request body, then parse it into a generated model."""
    signature = next(
        (value for name, value in headers.items() if name.lower() == SIGNATURE_HEADER.lower()),
        None,
    )
    if not verify_signature(raw_body, signature, secret):
        raise WebhookSignatureError("webhook signature verification failed")
    return parse_event(raw_body)


__all__ = [
    "ConsolidationCompletedWebhookEvent",
    "ConsumedWebhookEvent",
    "MemoryDefenseTriggeredWebhookEvent",
    "RetainCompletedWebhookEvent",
    "SIGNATURE_HEADER",
    "WebhookError",
    "WebhookEventEnvelope",
    "WebhookPayloadError",
    "WebhookSignatureError",
    "construct_event",
    "parse_event",
    "verify_signature",
]
