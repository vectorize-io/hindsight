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


class MemoryDefenseEventData(BaseModel):
    """Payload for a memory_defense.triggered event (one item, one non-allow decision)."""

    action: str  # "redact" or "block"
    detector: str | None = None  # e.g. "sensitive_data"
    document_id: str | None = None
    matched_types: list[str] | None = None  # redaction pattern labels that fired
    message: str | None = None


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
