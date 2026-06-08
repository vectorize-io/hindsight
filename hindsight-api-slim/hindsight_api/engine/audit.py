"""Audit logging for feature usage tracking.

Provides fire-and-forget audit logging of all mutating and core operations
(retain, recall, reflect, bank CRUD, etc.) across HTTP, MCP, and system transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..engine.db_utils import acquire_with_retry

logger = logging.getLogger(__name__)


class AuditLogEntry(BaseModel):
    """A single audit log entry."""

    id: str
    action: str
    transport: str
    bank_id: str | None
    started_at: str | None
    ended_at: str | None
    duration_ms: int | None = Field(
        default=None,
        description="Server-computed duration in milliseconds (started_at → ended_at). Null if not yet completed.",
    )
    request: dict[str, Any] | None
    response: dict[str, Any] | None
    metadata: dict[str, Any]


class AuditLogListResponse(BaseModel):
    """Response model for list audit logs endpoint."""

    bank_id: str
    total: int
    limit: int
    offset: int
    items: list[AuditLogEntry]


class AuditLogStatsBucket(BaseModel):
    """A single time bucket in audit log stats."""

    time: str
    actions: dict[str, int]
    total: int


class AuditLogStatsResponse(BaseModel):
    """Response model for audit log stats endpoint."""

    bank_id: str
    period: str
    trunc: str
    start: str
    buckets: list[AuditLogStatsBucket]


@dataclass
class AuditEntry:
    """A single audit log entry."""

    action: str
    transport: str  # "http", "mcp", "system"
    bank_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_default(obj: Any) -> str:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return "<bytes>"
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _safe_json(data: Any) -> str | None:
    """Serialize data to JSON string, returning None on failure."""
    if data is None:
        return None
    try:
        return json.dumps(data, default=_json_default)
    except Exception:
        logger.debug("Failed to serialize audit data", exc_info=True)
        return None


class AuditLogger:
    """Fire-and-forget audit log writer.

    Retention of old rows is handled by the background :class:`MaintenanceLoop`.
    """

    def __init__(
        self,
        pool_getter: Callable[[], Any],
        schema_getter: Callable[[], str],
        enabled: bool,
        allowed_actions: list[str],
    ) -> None:
        self._pool_getter = pool_getter
        self._schema_getter = schema_getter
        self._enabled = enabled
        self._allowed_actions: frozenset[str] | None = frozenset(allowed_actions) if allowed_actions else None

    def is_enabled(self, action: str) -> bool:
        """Check if audit logging is enabled for this action."""
        if not self._enabled:
            return False
        if self._allowed_actions is not None:
            return action in self._allowed_actions
        return True

    def log_fire_and_forget(self, entry: AuditEntry) -> None:
        """Schedule an audit write as a background task."""
        if not self.is_enabled(entry.action):
            return
        try:
            asyncio.create_task(self._safe_log(entry))
        except RuntimeError:
            # No running event loop (e.g. during shutdown)
            logger.debug("Cannot schedule audit log write: no running event loop")

    async def _safe_log(self, entry: AuditEntry) -> None:
        """Write audit entry to DB. Errors are logged, never raised."""
        pool = self._pool_getter()
        if pool is None:
            logger.debug("Audit log skipped: pool not available")
            return
        try:
            schema = self._schema_getter()
            table = f"{schema}.audit_log"
            async with acquire_with_retry(pool, max_retries=1) as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                        (id, action, transport, bank_id, started_at, ended_at, request, response, metadata)
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb)
                    """,
                    uuid.uuid4(),
                    entry.action,
                    entry.transport,
                    entry.bank_id,
                    entry.started_at,
                    entry.ended_at,
                    _safe_json(entry.request),
                    _safe_json(entry.response),
                    _safe_json(entry.metadata) or "{}",
                )
        except Exception as e:
            logger.warning(f"Audit log write failed for action={entry.action}: {e}")


@asynccontextmanager
async def audit_context(
    audit_logger: AuditLogger | None,
    action: str,
    transport: str,
    bank_id: str | None = None,
    request: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Async context manager that times the operation and writes audit on exit.

    Usage:
        async with audit_context(logger, "retain", "http", bank_id, request_dict) as entry:
            result = await do_work()
            entry.response = result_dict
    """
    if audit_logger is None or not audit_logger.is_enabled(action):
        entry = AuditEntry(action=action, transport=transport, bank_id=bank_id)
        yield entry
        return

    entry = AuditEntry(
        action=action,
        transport=transport,
        bank_id=bank_id,
        started_at=datetime.now(timezone.utc),
        request=request,
        metadata=metadata or {},
    )
    try:
        yield entry
    finally:
        entry.ended_at = datetime.now(timezone.utc)
        audit_logger.log_fire_and_forget(entry)
