"""Webhook system for Hindsight API event notifications."""

from .manager import WebhookManager
from .models import (
    ConsolidationEventData,
    MemoryDefenseEventData,
    MemoryDefenseHit,
    RetainEventData,
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
)

__all__ = [
    "WebhookManager",
    "WebhookConfig",
    "WebhookEvent",
    "WebhookEventType",
    "ConsolidationEventData",
    "MemoryDefenseEventData",
    "MemoryDefenseHit",
    "RetainEventData",
]
