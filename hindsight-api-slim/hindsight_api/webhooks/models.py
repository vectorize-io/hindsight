"""Pydantic models for the webhook system."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class WebhookEventType(StrEnum):
    CONSOLIDATION_COMPLETED = "consolidation.completed"
    RETAIN_COMPLETED = "retain.completed"
    MEMORY_DEFENSE_TRIGGERED = "memory_defense.triggered"


class ConsolidationEventData(BaseModel):
    observations_created: int | None = None
    observations_updated: int | None = None
    observations_deleted: int | None = None
    error_message: str | None = None


class RetainEventData(BaseModel):
    document_id: str | None = None
    tags: list[str] | None = None


class MemoryDefenseHit(BaseModel):
    """A single secret match inside a non-allow decision.

    ``preview`` is a fingerprinted, redaction-identifiable rendering of the
    matched value (e.g. ``ghp_AAAA...BBBB``) so SIEM operators can correlate
    against their credential inventory WITHOUT the raw secret crossing the
    network. Implementations must never put the raw value here.
    """

    detector: str  # the inner detector that matched (e.g. "GitHub Token")
    preview: str  # fingerprinted value, never the raw secret


class MemoryDefenseEventData(BaseModel):
    """Payload for a memory_defense.triggered event (one item, one non-allow decision).

    The four base fields (``action``/``detector``/``document_id``/``message``)
    plus ``matched_types`` are populated by every implementation including OSS's
    built-in regex defense. The remaining fields are optional SIEM-enrichment
    surfaces that downstream extensions (e.g. hindsight-cloud) populate when
    they have richer per-decision context — severity classification, the API
    key that submitted the retain, fingerprinted hit previews for SIEM
    correlation, and pointers into the audit trail. OSS leaves them ``None``;
    receivers should treat absence as "not provided" rather than "no match".
    """

    action: str  # "redact" or "block"
    detector: str | None = None  # e.g. "sensitive_data"
    document_id: str | None = None
    matched_types: list[str] | None = None  # redaction pattern labels that fired
    message: str | None = None
    # --- Optional SIEM enrichment (populated by extensions, not OSS) ---
    severity: str | None = None  # "low" / "medium" / "high" / "critical"
    api_key_name: str | None = None  # human-readable name of the submitting API key
    hits: list[MemoryDefenseHit] | None = None  # per-match fingerprints for correlation
    memory_unit_id: str | None = None  # drill-down pointer (when the decision was REDACT)
    receipt_uri: str | None = None  # storage pointer for the audit trail entry


class WebhookEvent(BaseModel):
    event: WebhookEventType
    bank_id: str
    operation_id: str
    status: str  # "completed"/"failed" for retain/consolidation; the action ("redact"/"block") for memory_defense
    timestamp: datetime
    data: ConsolidationEventData | RetainEventData | MemoryDefenseEventData


class WebhookHttpConfig(BaseModel):
    """HTTP delivery configuration for a webhook."""

    method: str = Field(default="POST", description="HTTP method: GET or POST")
    timeout_seconds: int = Field(default=30, description="HTTP request timeout in seconds")
    headers: dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    params: dict[str, str] = Field(default_factory=dict, description="Custom HTTP query parameters")


class WebhookConfig(BaseModel):
    id: str
    bank_id: str | None
    url: str
    secret: str | None
    event_types: list[str]
    enabled: bool
    http_config: WebhookHttpConfig = Field(default_factory=WebhookHttpConfig)
