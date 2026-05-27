"""
Periodic backstop that re-queues consolidation for banks with unconsolidated
``memory_units``.

The retain pipeline already submits consolidation on every successful retain
(see ``MemoryEngine`` callers of ``submit_async_consolidation``). This scanner
is a backstop for the case where that trigger chain breaks — embeddings
provider outage that fails retains for hours, manual data import that bypasses
retain, etc. Without it, ``memory_units.consolidated_at IS NULL`` rows
accumulate silently because nothing observes the state on its own.

Opt-in via ``HINDSIGHT_API_WORKER_PENDING_CONSOLIDATION_SCAN_INTERVAL_SECONDS``.
Default ``0`` (disabled) preserves existing behavior; set to a positive number
of seconds (e.g. ``300``) to enable.
"""

import asyncio
import logging

from ..engine.memory_engine import MemoryEngine
from ..engine.schema import fq_table_explicit as fq_table
from ..extensions.tenant import TenantExtension
from ..models import RequestContext

logger = logging.getLogger(__name__)

# LIMIT on the per-schema DISTINCT bank_id query. The scanner is a backstop and
# does not need to be strictly complete on every cycle — banks not picked up on
# one cycle will be on the next. The cap protects against a pathological case
# where a tenant has tens of thousands of distinct banks with pending work.
_MAX_BANKS_PER_SCHEMA_PER_SCAN = 100


class PendingConsolidationScanner:
    """Polls ``memory_units`` for unconsolidated rows and queues consolidation.

    Safe to run alongside the existing event-driven triggers: the underlying
    ``submit_async_consolidation`` call passes ``dedupe_by_bank=True``, so if
    a consolidation is already pending for a bank, the scanner's submit is a
    no-op rather than a duplicate enqueue.
    """

    def __init__(
        self,
        memory: MemoryEngine,
        tenant_extension: TenantExtension,
        interval_seconds: int,
    ) -> None:
        self._memory = memory
        self._tenant_extension = tenant_extension
        self._interval_seconds = interval_seconds
        self._shutdown = asyncio.Event()
        # Reuse the same RequestContext shape that other internal/background
        # operations use (e.g. `internal=True` skips extension auth).
        self._internal_ctx = RequestContext(internal=True)

    @property
    def enabled(self) -> bool:
        return self._interval_seconds > 0

    async def run(self) -> None:
        """Loop until ``shutdown()`` is set, scanning every ``interval_seconds``.

        If ``interval_seconds <= 0`` the loop returns immediately without doing
        any work, so it's safe to start unconditionally.
        """
        if not self.enabled:
            logger.info(
                "PendingConsolidationScanner disabled "
                "(HINDSIGHT_API_WORKER_PENDING_CONSOLIDATION_SCAN_INTERVAL_SECONDS=0)"
            )
            return
        logger.info(f"PendingConsolidationScanner started (interval={self._interval_seconds}s)")
        # Run an initial scan immediately so newly-started workers don't wait a
        # full interval to surface backlog left over from a prior process.
        await self._scan_safely()
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=self._interval_seconds)
            except TimeoutError:
                await self._scan_safely()
        logger.info("PendingConsolidationScanner shutting down")

    def shutdown(self) -> None:
        """Request a graceful exit from the scan loop on the next iteration."""
        self._shutdown.set()

    async def _scan_safely(self) -> None:
        """Wrapper around ``_scan_once`` that absorbs scan-level errors so the
        loop survives transient DB hiccups."""
        try:
            queued = await self._scan_once()
            if queued > 0:
                logger.info(f"PendingConsolidationScanner queued {queued} bank(s)")
        except Exception as e:
            logger.warning(f"PendingConsolidationScanner scan failed (will retry next interval): {e}")

    async def _scan_once(self) -> int:
        """Scan every schema once. Returns the number of banks for which a NEW
        consolidation op was queued (banks that hit dedupe are not counted).
        """
        tenants = await self._tenant_extension.list_tenants()
        queued = 0
        for tenant in tenants:
            # Mirror the poller's schema normalization: SQL helpers omit the
            # default-schema prefix, so use None to signal that.
            from ..config import DEFAULT_DATABASE_SCHEMA

            schema = None if tenant.schema == DEFAULT_DATABASE_SCHEMA else tenant.schema
            bank_ids = await self._find_pending_banks(schema)
            for bank_id in bank_ids:
                queued += await self._submit_if_not_dedup(bank_id, schema)
        return queued

    async def _find_pending_banks(self, schema: str | None) -> list[str]:
        """Return distinct ``bank_id``s in the given schema with at least one
        ``memory_units`` row where ``consolidated_at IS NULL`` and consolidation
        hasn't already failed permanently.
        """
        backend = await self._memory._get_backend()
        table = fq_table("memory_units", schema)
        async with backend.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT bank_id
                FROM {table}
                WHERE fact_type IN ('experience', 'world')
                  AND consolidated_at IS NULL
                  AND consolidation_failed_at IS NULL
                LIMIT $1
                """,
                _MAX_BANKS_PER_SCHEMA_PER_SCAN,
            )
        return [r["bank_id"] for r in rows]

    async def _submit_if_not_dedup(self, bank_id: str, schema: str | None) -> int:
        """Submit consolidation for ``bank_id``. Returns 1 if a NEW op was
        created, 0 if dedupe collapsed it or the call failed.

        Errors are logged but not raised so one bad bank doesn't stop the scan.
        """
        # The scanner runs from the worker process which doesn't have a native
        # tenant schema context, so set it explicitly via the contextvar.
        # ALWAYS set (including to None) so a prior tenant's schema doesn't
        # leak into the next iteration.
        from ..engine.memory_engine import _current_schema

        token = _current_schema.set(schema)
        try:
            result = await self._memory.submit_async_consolidation(bank_id=bank_id, request_context=self._internal_ctx)
            return 0 if result.get("deduplicated") else 1
        except Exception as e:
            logger.warning(
                f"PendingConsolidationScanner: submit_async_consolidation failed "
                f"for bank={bank_id} schema={schema}: {e}"
            )
            return 0
        finally:
            _current_schema.reset(token)
