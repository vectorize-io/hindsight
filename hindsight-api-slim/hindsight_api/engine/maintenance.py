"""Background maintenance loop.

A single periodic loop that drives all of Hindsight's recurring housekeeping
from one place, so we don't spawn a separate ``asyncio`` task per concern:

- **Retention sweeps** (hourly): delete ``audit_log`` and ``llm_requests`` rows
  older than their configured retention, across *all* tenant schemas.
- **Consolidation reconcile** (configurable, default 5 min): re-schedule
  consolidation for banks that have eligible-but-unscheduled facts and no
  in-flight consolidation. This recovers facts that were stranded when a
  consolidation operation failed terminally and left them with
  ``consolidated_at IS NULL AND consolidation_failed_at IS NULL`` and nothing to
  re-trigger them.

The loop wakes on a short fixed tick and runs each job when its own
``last_run + interval`` is due (run-at-start, then on interval), so adding jobs
with different cadences doesn't burst CPU. Cross-tenant discovery goes through
server-side PL/pgSQL routines (``public.schemas_with_expired_rows`` and
``public.banks_needing_consolidation``) — one round-trip each — instead of a
per-schema query storm, which matters at thousands of tenants.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ..config import HindsightConfig, get_config
from ..models import RequestContext
from .db_utils import acquire_with_retry
from .schema import _is_oracle

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Short tick so jobs with different cadences share one loop without per-job tasks.
_TICK_SECONDS = 60
# Retention sweeps are not time-sensitive; hourly matches the previous per-sweep cadence.
_RETENTION_INTERVAL_SECONDS = 3600


class MaintenanceLoop:
    """Owns the single periodic maintenance task for a :class:`MemoryEngine`."""

    def __init__(self, engine: "MemoryEngine") -> None:
        self._engine = engine
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Monotonic timestamps of the last run per job, keyed by job name.
        self._last_run: dict[str, float] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the loop if any maintenance job is enabled. Idempotent."""
        if self._task and not self._task.done():
            return
        # PostgreSQL-only: the retention sweeps target PG-only tables (audit_log,
        # llm_requests) and the reconcile relies on PG-only PL/pgSQL routines
        # installed by the maintenance-routines migration. Oracle support is
        # intentionally absent (mirrors that PG-only migration).
        if _is_oracle():
            logger.debug("Maintenance loop not started: PostgreSQL-only")
            return
        if not self._any_job_enabled():
            logger.debug("Maintenance loop not started: no jobs enabled")
            return
        self._stop.clear()
        try:
            self._task = asyncio.create_task(self._run())
        except RuntimeError:
            logger.debug("Cannot start maintenance loop: no running event loop")

    async def stop(self) -> None:
        """Stop the loop and wait for the current tick to finish."""
        self._stop.set()
        if self._task and not self._task.done():
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @staticmethod
    def _any_job_enabled() -> bool:
        cfg = get_config()
        reconcile_on = cfg.consolidation_reconcile_interval_seconds > 0
        audit_on = cfg.audit_log_enabled and cfg.audit_log_retention_days > 0
        llm_on = cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0
        return reconcile_on or audit_on or llm_on

    # ── loop ───────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Maintenance tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=_TICK_SECONDS)
            except asyncio.TimeoutError:
                pass

    def _is_due(self, job: str, interval_seconds: int) -> bool:
        """True if ``job`` has never run or its interval has elapsed; marks it run now."""
        now = time.monotonic()
        last = self._last_run.get(job)
        if last is not None and (now - last) < interval_seconds:
            return False
        self._last_run[job] = now
        return True

    async def _tick(self) -> None:
        cfg = get_config()
        if self._is_due("retention", _RETENTION_INTERVAL_SECONDS):
            await self._run_retention(cfg)
        interval = cfg.consolidation_reconcile_interval_seconds
        if interval > 0 and self._is_due("reconcile", interval):
            await self._run_reconcile()

    # ── retention ──────────────────────────────────────────────────────────

    async def _run_retention(self, cfg: HindsightConfig) -> None:
        # Retention days are static server-level config, so one global cutoff
        # applies to every tenant schema (the routine sweeps them all).
        if cfg.audit_log_enabled and cfg.audit_log_retention_days > 0:
            await self._purge_expired("audit_log", "started_at", cfg.audit_log_retention_days)
        if cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0:
            await self._purge_expired("llm_requests", "started_at", cfg.llm_trace_retention_days)

    async def _purge_expired(self, table: str, ts_col: str, days: int) -> None:
        """Delete rows older than ``days`` from ``table`` across every tenant schema."""
        backend = self._engine._backend
        try:
            async with acquire_with_retry(backend, max_retries=1) as conn:
                rows = await conn.fetch(
                    "SELECT * FROM public.schemas_with_expired_rows($1, $2, $3)", table, ts_col, days
                )
                for row in rows:
                    schema = row[0]
                    # schema names come from pg_class; quote defensively all the same.
                    qschema = '"' + schema.replace('"', '""') + '"'
                    result = await conn.execute(
                        f"DELETE FROM {qschema}.{table} WHERE {ts_col} < NOW() - make_interval(days => $1)",
                        days,
                    )
                    if result and result != "DELETE 0":
                        logger.info(f"Retention sweep {schema}.{table}: {result}")
        except Exception as e:
            logger.warning(f"Retention sweep failed for {table}: {e}")

    # ── consolidation reconcile ──────────────────────────────────────────────

    async def _run_reconcile(self) -> None:
        """Re-schedule consolidation for banks with eligible-but-unscheduled facts."""
        engine = self._engine
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                rows = await conn.fetch("SELECT schema_name, bank_id FROM public.banks_needing_consolidation()")
        except Exception as e:
            logger.warning(f"Consolidation reconcile discovery failed: {e}")
            return
        if not rows:
            return

        # Only enqueue into schemas the worker actually polls (tenant discovery),
        # otherwise the op would never be claimed and would block future reconciles
        # for that bank. The tenant_id (when the extension provides one) lets
        # config resolution honor tenant-level overrides.
        try:
            tenants = await engine._tenant_extension.list_tenants()
        except Exception as e:
            logger.warning(f"Consolidation reconcile tenant discovery failed: {e}")
            return
        tenant_by_schema = {t.schema: t for t in tenants}
        default_schema = get_config().database_schema

        from .memory_engine import _current_schema

        submitted = 0
        skipped_unknown = 0
        for row in rows:
            schema = row["schema_name"]
            bank_id = row["bank_id"]
            tenant = tenant_by_schema.get(schema)
            if tenant is None and schema != default_schema:
                skipped_unknown += 1
                continue
            tenant_id = tenant.tenant_id if tenant else None
            token = _current_schema.set(schema)
            try:
                context = RequestContext(internal=True, tenant_id=tenant_id)
                resolved = await engine._config_resolver.resolve_full_config(bank_id, context)
                # Mirror the retain-time auto-consolidation gate (memory_engine): both
                # observations and auto-consolidation must be enabled for this bank.
                if not (resolved.enable_observations and resolved.enable_auto_consolidation):
                    continue
                await engine.submit_async_consolidation(bank_id=bank_id, request_context=context)
                submitted += 1
            except Exception as e:
                logger.warning(f"Consolidation reconcile failed for bank {bank_id} in {schema}: {e}")
            finally:
                _current_schema.reset(token)

        if submitted or skipped_unknown:
            logger.info(
                f"Consolidation reconcile: scheduled {submitted} bank(s)"
                + (f", skipped {skipped_unknown} in unrecognized schema(s)" if skipped_unknown else "")
            )
