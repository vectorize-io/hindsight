"""Pydantic models for the webhook system."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebhookEventType(StrEnum):
    CONSOLIDATION_COMPLETED = "consolidation.completed"
    RETAIN_COMPLETED = "retain.completed"
    MEMORY_DEFENSE_TRIGGERED = "memory_defense.triggered"


class WebhookModel(BaseModel):
    """Base for webhook wire models that must tolerate additive fields."""

    model_config = ConfigDict(extra="allow")


class ConsolidationEventData(WebhookModel):
    observations_created: int | None = None
    observations_updated: int | None = None
    observations_deleted: int | None = None
    error_message: str | None = None


class RetainEventData(WebhookModel):
    document_id: str | None = None
    tags: list[str] | None = None
    memory_unit_count: int | None = Field(
        default=None,
        description=(
            "Memory units the document owns after this retain (the same number the "
            "Documents API reports). 0 means fact extraction returned nothing, so the "
            "document is stored but unreachable through recall/reflect until it is "
            "reprocessed. Null when the retain carried no document_id, since there is "
            "then no document to count against."
        ),
    )


class MemoryDefenseHit(WebhookModel):
    """A single secret match inside a non-allow decision.

    ``preview`` is a fingerprinted, redaction-identifiable rendering of the
    matched value (e.g. ``ghp_AAAA...BBBB``) so SIEM operators can correlate
    against their credential inventory WITHOUT the raw secret crossing the
    network. Implementations must never put the raw value here.
    """

    detector: str  # the inner detector that matched (e.g. "GitHub Token")
    preview: str  # fingerprinted value, never the raw secret


class MemoryDefenseEventData(WebhookModel):
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


class WebhookEventEnvelope(WebhookModel):
    """Forward-compatible envelope used when an SDK receives an unknown event."""

    event: str
    bank_id: str
    operation_id: str
    status: str
    timestamp: datetime
    data: dict[str, Any]


class WebhookEvent(WebhookEventEnvelope):
    """Internal emitter model retained for callers that select an event dynamically."""

    event: WebhookEventType
    data: ConsolidationEventData | RetainEventData | MemoryDefenseEventData


class ConsolidationCompletedWebhookEvent(WebhookEvent):
    event: Literal["consolidation.completed"]
    data: ConsolidationEventData


class RetainCompletedWebhookEvent(WebhookEvent):
    event: Literal["retain.completed"]
    data: RetainEventData


class MemoryDefenseTriggeredWebhookEvent(WebhookEvent):
    event: Literal["memory_defense.triggered"]
    data: MemoryDefenseEventData


# Keep the discriminator on the wire field so every generated SDK binds an
# event name to the matching data model instead of accepting an ambiguous union.
KnownWebhookEvent = Annotated[
    ConsolidationCompletedWebhookEvent | RetainCompletedWebhookEvent | MemoryDefenseTriggeredWebhookEvent,
    Field(discriminator="event"),
]


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
