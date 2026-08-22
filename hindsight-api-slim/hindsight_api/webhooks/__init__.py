"""Webhook system for Hindsight API event notifications."""

from .manager import WebhookManager
from .models import (
    ConsolidationCompletedWebhookEvent,
    ConsolidationEventData,
    KnownWebhookEvent,
    MemoryDefenseEventData,
    MemoryDefenseHit,
    MemoryDefenseTriggeredWebhookEvent,
    RetainCompletedWebhookEvent,
    RetainEventData,
    WebhookConfig,
    WebhookEvent,
    WebhookEventEnvelope,
    WebhookEventType,
)

__all__ = [
    "WebhookManager",
    "WebhookConfig",
    "WebhookEvent",
    "WebhookEventEnvelope",
    "WebhookEventType",
    "KnownWebhookEvent",
    "ConsolidationCompletedWebhookEvent",
    "RetainCompletedWebhookEvent",
    "MemoryDefenseTriggeredWebhookEvent",
    "ConsolidationEventData",
    "MemoryDefenseEventData",
    "MemoryDefenseHit",
    "RetainEventData",
]
