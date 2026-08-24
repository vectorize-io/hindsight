"""Background maintenance loop.

A single periodic loop that drives all of Hindsight's recurring housekeeping
from one place, so we don't spawn a separate ``asyncio`` task per concern:

- **Retention sweeps** (configurable, default hourly): delete ``audit_log`` and
  ``llm_requests`` rows older than their configured retention, across *all*
  tenant schemas.
- **Consolidation reconcile** (configurable, default 5 min): re-schedule
  consolidation for banks that have eligible-but-unscheduled facts and no
  in-flight consolidation. This recovers facts that were stranded when a
  consolidation operation failed terminally and left them with
  ``consolidated_at IS NULL AND consolidation_failed_at IS NULL`` and nothing to
  re-trigger them.
- **Scheduled mental model refresh** (configurable check cadence, default 5 min):
  refresh mental models whose ``trigger.refresh_cron`` schedule is due, but only
  when the model is stale (new memories in its scope since its last refresh), so
  a scheduled tick never burns an LLM call to regenerate identical content. The
  per-model schedule lives in the cron expression; this loop only decides when to
  *check*.

The loop wakes on a short fixed tick and runs each job when its own
``last_run + interval`` is due (run-at-start, then on interval), so adding jobs
with different cadences doesn't burst CPU. Cross-tenant discovery goes through
server-side PL/pgSQL routines (``schemas_with_expired_rows`` and
``banks_needing_consolidation``, in the configured schema — see ``fq_routine``) —
one round-trip each — instead of a per-schema query storm, which matters at
thousands of tenants.

The loop runs in *every* API/worker process with no leader election, so a job that
enqueues work must make that enqueue idempotent or the fleet queues one wave per
process. Operation cleanup deletes a bounded batch per schema; the consolidation
reconcile and the scheduled mental model refresh both dedupe against in-flight
operations inside the inserting transaction (see ``_submit_async_operation``);
retention deletes in bounded chunks claimed with SKIP LOCKED, so concurrent
sweepers split the work instead of colliding (see ``_purge_table_in_batches``).

The *work* is therefore safe to run everywhere. The *discovery* in front of it is
not free: one round-trip on the wire is still one query per tenant schema inside
the routine, every process pays it, and its cost scales with tenant count while
the work it finds does not. Two things keep that proportionate, and both are
load-bearing rather than incidental:

- **Cadence is config, not a constant.** Every job's interval is a server-level
  setting. The jobs that delete rows whose retention is measured in *days*
  (retention, operation cleanup) have no reason to probe every schema every
  minute.
- **The first tick is jittered** (``maintenance_start_jitter_seconds``). Every job
  is due on the first tick, so a fleet started together — a deploy, a rolling
  restart — would fire every cross-tenant probe in every process at the same
  instant. SKIP LOCKED keeps that correct but not cheap: the probes are reads, and
  N of them land at once. Steady state self-staggers; startup does not.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Coroutine
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from ..config import HindsightConfig, get_config
from ..models import RequestContext
from .db_utils import acquire_with_retry
from .schema import _is_oracle, fq_routine, fq_table, fq_table_explicit

if TYPE_CHECKING:
    from .memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# Short tick so jobs with different cadences share one loop without per-job tasks.
_TICK_SECONDS = 60
# ── retention sweep pacing ────────────────────────────────────────────────────
# Retention used to issue one unbounded `DELETE FROM <table> WHERE started_at <
# cutoff` per schema. On a table with a real backlog that is a single statement
# holding row locks for minutes while it reads the whole expired range — and the
# maintenance loop runs in every API/worker process with no leader election, so
# every pod issued it at the same hourly boundary. Observed as two concurrent
# 330s+ deletes pinned on IO.DataFileRead, blocking each other on row locks,
# saturating RDS I/O and tripling recall latency.
#
# The fix is to design the collision out rather than elect one sweeper: each chunk
# claims its rows with FOR UPDATE SKIP LOCKED, so concurrent sweepers take
# *disjoint* chunks instead of waiting on each other, and the total work stays the
# number of expired rows however many pods join in. Chunks are short, index-driven
# transactions with a pause between them, so no statement holds locks for long and
# the deletes never monopolise disk I/O.
_RETENTION_BATCH_SIZE = 2000
# Ceiling on chunks per table per schema per run: a backstop against looping
# forever on a table that is filling faster than it drains. A full run therefore
# removes at most 2M rows per schema, and the next sweep (whose cadence is
# HINDSIGHT_API_RETENTION_SWEEP_INTERVAL_SECONDS, default hourly) continues.
_RETENTION_MAX_BATCHES = 1000
# Breather between chunks. Paces one process at ~8k rows/s worst case; several
# pods sweeping at once multiply that, which is still orders of magnitude gentler
# than the unbounded delete this replaces.
_RETENTION_BATCH_PAUSE_SECONDS = 0.25


class MaintenanceLoop:
    """Owns the single periodic maintenance task for a :class:`MemoryEngine`."""

    def __init__(self, engine: "MemoryEngine | None") -> None:
        """``engine`` is optional so the loop can be exercised for its scheduling alone — the
        cadence tests construct it that way. Every job that needs the engine checks first."""
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
        # Not gated on audit_log_enabled: that is per-bank overridable, so rows
        # can exist even when the deployment default is off. Retention is driven
        # purely by the (server-level) window.
        sweep_on = cfg.retention_sweep_interval_seconds > 0
        audit_on = sweep_on and cfg.audit_log_retention_days > 0
        llm_on = sweep_on and cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0
        mm_refresh_on = cfg.mental_model_refresh_tick_seconds > 0
        op_cleanup_on = cfg.operation_cleanup_interval_seconds > 0 and cfg.operation_retention_days > 0
        return reconcile_on or audit_on or llm_on or mm_refresh_on or op_cleanup_on

    # ── loop ───────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        if not await self._wait_start_jitter():
            return
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Maintenance tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=_TICK_SECONDS)
            except asyncio.TimeoutError:
                pass

    async def _wait_start_jitter(self) -> bool:
        """Delay the first tick by a random offset. Returns False if stopped while waiting.

        Every job is due the first time ``_is_due`` sees it, so N processes started
        together would run all of them at the same instant — the one moment where
        redundant cross-tenant discovery and overlapping DELETEs actually collide.
        Spreading the *first* tick is enough: from then on each process keeps its
        own phase.
        """
        jitter = get_config().maintenance_start_jitter_seconds
        if jitter <= 0:
            return True
        delay = random.uniform(0, jitter)
        logger.debug(f"Maintenance loop: delaying first tick by {delay:.1f}s")
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return True
        return False

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
        retention_interval = cfg.retention_sweep_interval_seconds
        if retention_interval > 0 and self._is_due("retention", retention_interval):
            await self._run_timed("retention", self._run_retention(cfg))
        interval = cfg.consolidation_reconcile_interval_seconds
        if interval > 0 and self._is_due("reconcile", interval):
            await self._run_timed("consolidation reconcile", self._run_reconcile())
        mm_interval = cfg.mental_model_refresh_tick_seconds
        if mm_interval > 0 and self._is_due("mm_refresh", mm_interval):
            await self._run_timed("scheduled mental model refresh", self._run_scheduled_mm_refresh())
        cleanup_interval = cfg.operation_cleanup_interval_seconds
        if (
            cleanup_interval > 0
            and cfg.operation_retention_days > 0
            and self._is_due("operation_cleanup", cleanup_interval)
        ):
            await self._run_timed("operation cleanup", self._run_operation_cleanup(cfg))

    async def _run_timed(self, name: str, coro: Coroutine[Any, Any, None]) -> None:
        """Run a maintenance job and emit one timing line for it.

        Each job keeps its own summary log (counts of work done); this adds a
        single, uniform line per run so the cost of every sweep is observable.
        """
        start = time.monotonic()
        try:
            await coro
        finally:
            logger.info(f"Maintenance: {name} took {time.monotonic() - start:.3f}s")

    # ── retention ──────────────────────────────────────────────────────────

    async def _run_retention(self, cfg: HindsightConfig) -> None:
        # Retention days are static server-level config, so one global cutoff
        # applies to every tenant schema (the routine sweeps them all).
        # Not gated on audit_log_enabled: it is per-bank overridable, so a bank
        # may be writing audit rows while the deployment default is off. Gating
        # the purge on the global flag would let those rows accumulate forever.
        if cfg.audit_log_retention_days > 0:
            await self._purge_expired("audit_log", "started_at", cfg.audit_log_retention_days)
        if cfg.llm_trace_enabled and cfg.llm_trace_retention_days > 0:
            await self._purge_expired("llm_requests", "started_at", cfg.llm_trace_retention_days)

    async def _purge_expired(self, table: str, ts_col: str, days: int) -> int:
        """Delete rows older than ``days`` from ``table`` across every tenant schema.

        Only for the retention tables (``audit_log``, ``llm_requests``): chunking
        deletes by primary key assumes the ``id`` column both of them carry.
        Returns the number of rows deleted by *this* process.
        """
        backend = self._engine._backend
        try:
            async with acquire_with_retry(backend, max_retries=1) as conn:
                rows = await conn.fetch(
                    f"SELECT * FROM {fq_routine('schemas_with_expired_rows')}($1, $2, $3)", table, ts_col, days
                )
        except Exception as e:
            logger.warning(f"Retention sweep discovery failed for {table}: {e}")
            return 0

        # One cutoff for the whole sweep: a per-chunk NOW() would let the window
        # creep forward mid-run, which makes "deleted < batch means done" wrong.
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        total = 0
        for row in rows:
            schema = row[0]
            # schema names come from pg_class; quote defensively all the same.
            qschema = '"' + schema.replace('"', '""') + '"'
            try:
                deleted = await self._purge_table_in_batches(f"{qschema}.{table}", ts_col, cutoff)
            except Exception as e:
                logger.warning(f"Retention sweep failed for {schema}.{table}: {e}")
                continue
            if deleted:
                total += deleted
                logger.info(f"Retention sweep {schema}.{table}: DELETE {deleted}")
        return total

    async def _purge_table_in_batches(self, table: str, ts_col: str, cutoff: datetime) -> int:
        """Delete expired rows from one qualified table in bounded chunks.

        Rows are claimed oldest-first off the ``(started_at)`` index with FOR UPDATE
        SKIP LOCKED, which is what makes a leaderless fleet safe: a chunk never
        waits on another sweeper (or on a writer still finishing its own row), and
        two processes sweeping the same table take disjoint chunks rather than
        redoing each other's work. Each chunk commits on its own, so no transaction
        holds locks longer than one batch.
        """
        backend = self._engine._backend
        deleted = 0
        for batch in range(_RETENTION_MAX_BATCHES):
            if self._stop.is_set():
                break
            if batch:
                await asyncio.sleep(_RETENTION_BATCH_PAUSE_SECONDS)
            async with acquire_with_retry(backend, max_retries=1) as conn, conn.transaction():
                removed = await conn.fetchval(
                    f"""
                    WITH expired AS (
                        SELECT id FROM {table}
                        WHERE {ts_col} < $1
                        ORDER BY {ts_col}
                        LIMIT $2
                        FOR UPDATE SKIP LOCKED
                    ), removed AS (
                        DELETE FROM {table} t USING expired e WHERE t.id = e.id RETURNING 1
                    )
                    SELECT count(*) FROM removed
                    """,
                    cutoff,
                    _RETENTION_BATCH_SIZE,
                )
            deleted += removed
            # A short chunk means the expired range is drained — or that another
            # sweeper holds the rest, which is equally a reason to stop.
            if removed < _RETENTION_BATCH_SIZE:
                break
        else:
            logger.warning(
                f"Retention sweep hit its per-run batch ceiling on {table} after {deleted} row(s); "
                "the remainder is left for the next run"
            )
        return deleted

    # ── terminal operation cleanup ─────────────────────────────────────────

    async def _run_operation_cleanup(self, cfg: HindsightConfig) -> None:
        """Prune one bounded batch of expired terminal operations per tenant schema.

        Previously this rode the worker's task-claiming loop, so it only fired
        when that loop happened to iterate and was interleaved with claiming. It
        is a periodic housekeeping sweep like the retention jobs above, so it
        belongs on the same schedule.

        Discovery is one cross-tenant round-trip (``schemas_with_expired_operations``)
        rather than a connection + prune transaction per tenant; pending and
        processing rows are never prunable, so a schema holding only in-flight
        work is correctly reported as having nothing to do.
        """
        engine = self._engine
        backend = engine._backend
        try:
            async with acquire_with_retry(backend, max_retries=1) as conn:
                rows = await conn.fetch(
                    f"SELECT * FROM {fq_routine('schemas_with_expired_operations')}($1)",
                    cfg.operation_retention_days,
                )
        except Exception as e:
            logger.warning(f"Operation cleanup discovery failed: {e}")
            return
        if not rows:
            return

        # Prune only schemas the deployment actually serves. The routine reports
        # every schema owning an async_operations table, including ones tenant
        # discovery doesn't claim.
        try:
            tenants = await engine._tenant_extension.list_tenants()
        except Exception as e:
            logger.warning(f"Operation cleanup tenant discovery failed: {e}")
            return
        known = {t.schema for t in tenants} | {get_config().database_schema}

        from .memory_engine import _current_schema

        cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.operation_retention_days)
        pruned = 0
        for row in rows:
            schema = row[0]
            if schema not in known:
                continue
            # Oracle resolves unqualified names from a context-bound session
            # schema; on PostgreSQL this is harmless and fq_table stays explicit.
            token = _current_schema.set(schema)
            try:
                table = fq_table_explicit("async_operations", schema)
                async with acquire_with_retry(backend, max_retries=1) as conn:
                    # Delete export archives owned by rows about to be pruned first,
                    # so the file-storage blobs don't outlive their operation row.
                    # Same batch bound as the prune below: the two walk the same
                    # ordered window so a backlog doesn't re-purge already-deleted
                    # archives on every cycle.
                    await engine.purge_expired_export_archives(
                        conn, table, cutoff, batch_size=cfg.operation_cleanup_batch_size
                    )
                    async with conn.transaction():
                        deleted = await backend.ops.prune_terminal_operations(
                            conn, table, cutoff, batch_size=cfg.operation_cleanup_batch_size
                        )
                if deleted:
                    pruned += deleted
                    logger.info(f"Operation cleanup pruned {deleted} expired terminal operations from {schema}")
            except Exception as e:
                logger.warning(f"Operation cleanup failed for schema {schema}: {e}")
            finally:
                _current_schema.reset(token)
        if pruned:
            logger.info(f"Operation cleanup: pruned {pruned} operation(s) total")

    # ── consolidation reconcile ──────────────────────────────────────────────

    async def _run_reconcile(self) -> None:
        """Re-schedule consolidation for banks with eligible-but-unscheduled facts."""
        engine = self._engine
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                rows = await conn.fetch(
                    f"SELECT schema_name, bank_id FROM {fq_routine('banks_needing_consolidation')}()"
                )
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

    # ── scheduled mental model refresh ───────────────────────────────────────

    async def _run_scheduled_mm_refresh(self) -> None:
        """Refresh mental models whose ``trigger.refresh_cron`` is due.

        Discovery (the set of cron-scheduled models, minus any with an in-flight
        refresh) is one cross-tenant round-trip via
        ``mental_models_with_cron()``. Cron *due-ness* is evaluated here in
        Python — a scheduled fire has elapsed when the most recent cron boundary at
        or before now is later than ``last_refreshed_at`` — because cron arithmetic
        isn't expressible in plain SQL. Each due model is refreshed only when it is
        actually stale, so a schedule that fires while nothing changed costs a
        cheap staleness query, not an LLM call.
        """
        engine = self._engine
        try:
            async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                rows = await conn.fetch(
                    "SELECT schema_name, bank_id, mental_model_id, refresh_cron, last_refreshed_at "
                    f"FROM {fq_routine('mental_models_with_cron')}()"
                )
        except Exception as e:
            logger.warning(f"Scheduled mental model refresh discovery failed: {e}")
            return
        if not rows:
            return

        from croniter import croniter

        now = datetime.now(timezone.utc)
        due = []
        for row in rows:
            cron = row["refresh_cron"]
            last = row["last_refreshed_at"]
            try:
                prev_fire = croniter(cron, now).get_prev(datetime)
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Scheduled mental model refresh: skipping invalid cron {cron!r} for "
                    f"{row['schema_name']}/{row['mental_model_id']}: {e}"
                )
                continue
            if last is None or prev_fire > last:
                due.append(row)
        if not due:
            return

        # Only enqueue into schemas the worker actually polls (tenant discovery),
        # otherwise the op would never be claimed. The tenant_id (when provided)
        # lets config resolution honor tenant-level overrides.
        try:
            tenants = await engine._tenant_extension.list_tenants()
        except Exception as e:
            logger.warning(f"Scheduled mental model refresh tenant discovery failed: {e}")
            return
        tenant_by_schema = {t.schema: t for t in tenants}
        default_schema = get_config().database_schema

        from .memory_engine import _current_schema

        submitted = 0
        skipped_unknown = 0
        skipped_fresh = 0
        skipped_in_flight = 0
        for row in due:
            schema = row["schema_name"]
            bank_id = row["bank_id"]
            mm_id = row["mental_model_id"]
            tenant = tenant_by_schema.get(schema)
            if tenant is None and schema != default_schema:
                skipped_unknown += 1
                continue
            tenant_id = tenant.tenant_id if tenant else None
            token = _current_schema.set(schema)
            try:
                context = RequestContext(internal=True, tenant_id=tenant_id)
                # Skip if nothing in the model's scope changed since its last
                # refresh — a scheduled refresh must not regenerate identical
                # content. compute_mental_model_is_stale needs the model's tags +
                # trigger, which the discovery routine doesn't return, so re-read
                # the row under the bank's schema context.
                async with acquire_with_retry(engine._backend, max_retries=1) as conn:
                    mm_row = await conn.fetchrow(
                        f"SELECT id, tags, trigger, last_refreshed_at, last_memory_seen_at "
                        f"FROM {fq_table('mental_models')} "
                        "WHERE bank_id = $1 AND id = $2",
                        bank_id,
                        mm_id,
                    )
                    if mm_row is None:
                        continue
                    is_stale = await engine.compute_mental_model_is_stale(conn, bank_id, mm_row)
                if not is_stale:
                    skipped_fresh += 1
                    continue
                # skip_if_in_flight makes the enqueue itself idempotent. The discovery
                # routine already excludes models with a pending/processing refresh,
                # but that exclusion is a *read*: this loop runs in every process, so
                # every process saw the same "nothing in flight" snapshot and inserted
                # its own operation — one queued wave per process (#3210). The insert
                # now carries the check, so a second one is never created.
                result = await engine.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm_id,
                    request_context=context,
                    skip_if_in_flight=True,
                    automatic=True,
                )
                if result.get("deduplicated"):
                    skipped_in_flight += 1
                else:
                    submitted += 1
            except Exception as e:
                logger.warning(f"Scheduled mental model refresh failed for {mm_id} in {schema}: {e}")
            finally:
                _current_schema.reset(token)

        if submitted or skipped_unknown or skipped_fresh or skipped_in_flight:
            logger.info(
                f"Scheduled mental model refresh: scheduled {submitted} model(s)"
                + (f", {skipped_fresh} up-to-date" if skipped_fresh else "")
                + (f", {skipped_in_flight} already in flight" if skipped_in_flight else "")
                + (f", skipped {skipped_unknown} in unrecognized schema(s)" if skipped_unknown else "")
            )
