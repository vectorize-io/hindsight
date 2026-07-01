"""Internal request context — signed propagation between internal services.

Verify-once-at-edge model: internal services trust the `X-CM-Context` header 
if it is correctly HMAC-signed with the shared secret.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings

INTERNAL_CONTEXT_HEADER = "x-cm-context"

AuthMethod = Literal["jwt", "api_key", "dev_default", "internal_service"]

class InternalContext(BaseModel):
    tenant_id: str
    actor_id: str
    user_id: str | None = None
    source_app_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    auth_method: AuthMethod


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * (4 - len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_context(ctx: InternalContext) -> str:
    """Sign a context into an X-CM-Context header value."""
    payload = ctx.model_dump_json().encode("utf-8")
    signature = hmac.new(
        settings.internal_context_secret.get_secret_value().encode("utf-8"),
        payload,
        hashlib.sha256
    ).digest()
    return f"{b64url_encode(payload)}.{b64url_encode(signature)}"


def verify_context(header_value: str | None) -> InternalContext | None:
    """Verify + parse an X-CM-Context header value. Returns None if invalid."""
    if not header_value or "." not in header_value:
        return None

    try:
        payload_b64, signature_b64 = header_value.split(".", 1)
        payload_raw = b64url_decode(payload_b64)
        
        expected_sig = hmac.new(
            settings.internal_context_secret.get_secret_value().encode("utf-8"),
            payload_raw,
            hashlib.sha256
        ).digest()
        
        got_sig = b64url_decode(signature_b64)
        
        if not hmac.compare_digest(expected_sig, got_sig):
            return None
            
        data = json.loads(payload_raw.decode("utf-8"))
        return InternalContext(**data)
    except Exception:
        return None
