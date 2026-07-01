"""Audit logger — in-memory stub.

Target: append to the `audit_events` table (shared Postgres schema). The scaffold
keeps events in process memory so the contract and trace flow are exercisable.
"""

from __future__ import annotations

import time
import uuid

from app.audit.schemas import AuditEvent

_EVENTS: list[AuditEvent] = []


def new_trace_id() -> str:
    return str(uuid.uuid4())


def log_event(
    *,
    tenant_id: str,
    operation: str,
    actor_id: str | None = None,
    source_app_id: str | None = None,
    resource_type: str | None = "memory",
    resource_id: str | None = None,
    outcome: str = "success",
    metadata: dict | None = None,
    trace_id: str | None = None,
) -> AuditEvent:
    event = AuditEvent(
        trace_id=trace_id or new_trace_id(),
        tenant_id=tenant_id,
        actor_id=actor_id,
        source_app_id=source_app_id,
        operation=operation,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        metadata=metadata or {},
        timestamp=int(time.time() * 1000),
    )
    _EVENTS.append(event)
    return event


def get_events(tenant_id: str, limit: int = 100) -> list[AuditEvent]:
    items = [e for e in _EVENTS if e.tenant_id == tenant_id]
    return list(reversed(items))[:limit]


def get_all_events(limit: int = 100) -> list[AuditEvent]:
    """Get events across all tenants (for dashboard)."""
    return list(reversed(_EVENTS))[:limit]
