import hashlib
import hmac
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from hindsight_client.webhooks import (
    RetainCompletedWebhookEvent,
    WebhookEventEnvelope,
    WebhookPayloadError,
    WebhookSignatureError,
    construct_event,
    parse_event,
    verify_signature,
)


class SignatureVector(BaseModel):
    secret: str
    raw_body: str
    signature: str


def _vector() -> SignatureVector:
    path = Path(__file__).parents[2] / "testdata" / "webhook-signatures.json"
    return SignatureVector.model_validate_json(path.read_text())


def _sign(raw_body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def test_verifies_shared_signature_vector_over_exact_bytes():
    vector = _vector()
    raw_body = vector.raw_body.encode()

    assert verify_signature(raw_body, vector.signature, vector.secret)
    assert not verify_signature(raw_body + b"\n", vector.signature, vector.secret)
    assert not verify_signature(raw_body, vector.signature.upper(), vector.secret)


def test_construct_event_uses_case_insensitive_header_and_generated_model():
    vector = _vector()

    event = construct_event(
        vector.raw_body.encode(),
        {"x-hindsight-signature": vector.signature},
        vector.secret,
    )

    assert isinstance(event, RetainCompletedWebhookEvent)
    assert event.data.document_id == "doc-1"


def test_construct_event_verifies_before_attempting_json_parse():
    with pytest.raises(WebhookSignatureError):
        construct_event(b"{", {"X-Hindsight-Signature": "sha256=" + "0" * 64}, "secret")

    raw_body = b"{"
    with pytest.raises(WebhookPayloadError):
        construct_event(raw_body, {"X-Hindsight-Signature": _sign(raw_body, "secret")}, "secret")


def test_unknown_event_uses_forward_compatible_envelope():
    raw_body = json.dumps(
        {
            "event": "future.created",
            "bank_id": "bank-1",
            "operation_id": "op-2",
            "status": "completed",
            "timestamp": "2026-08-11T09:30:00Z",
            "data": {"future": True},
            "future_envelope_field": 42,
        }
    ).encode()

    event = parse_event(raw_body)

    assert isinstance(event, WebhookEventEnvelope)
    assert event.data == {"future": True}
    assert event.additional_properties["future_envelope_field"] == 42


def test_known_event_payload_is_validated():
    raw_body = json.dumps(
        {
            "event": "memory_defense.triggered",
            "bank_id": "bank-1",
            "operation_id": "op-3",
            "status": "block",
            "timestamp": "2026-08-11T09:30:00Z",
            "data": {},
        }
    ).encode()

    with pytest.raises(WebhookPayloadError):
        parse_event(raw_body)


@pytest.mark.parametrize("timestamp", [0, "2026-08-11T09:30:00"])
def test_rejects_non_rfc3339_timestamp(timestamp):
    raw_body = json.dumps(
        {
            "event": "future.created",
            "bank_id": "bank-1",
            "operation_id": "op-2",
            "status": "completed",
            "timestamp": timestamp,
            "data": {},
        }
    ).encode()

    with pytest.raises(WebhookPayloadError, match="RFC 3339"):
        parse_event(raw_body)
