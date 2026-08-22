"""
Worker poller for distributed task execution.

Polls the database for pending tasks and executes them using
FOR UPDATE SKIP LOCKED for safe concurrent claiming.

Backend-agnostic: works with any DatabaseBackend implementation
(PostgreSQL via asyncpg, Oracle via oracledb, etc.).
"""

import asyncio
import io
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ..config import get_config
from ..engine.schema import fq_table_explicit as fq_table
from ..metrics import get_metrics_collector
from .exceptions import DeferOperation, RetryTaskAt, format_task_error
from .stage import StageHolder, bind_holder

# Map DB operation_type -> metric `operation` label, collapsing the retain
# variants onto "retain" so async worker completions land on the same
# operation="retain" series the synchronous API path emits. Unknown types
# pass through unchanged.
_RETAIN_OP_TYPES = {"retain", "batch_retain", "file_convert_retain"}


def _current_rss_bytes() -> int | None:
    """Current resident set size in bytes, or None if the platform doesn't expose it cheaply.

    Reads ``/proc/self/statm`` (Linux only), whose second field is the resident
    page count. This is a two-integer parse of a tiny pseudo-file, so it is safe
    to call on every stats tick. macOS has no equivalent without pulling in a
    dependency, so callers fall back to reporting the peak alone.
    """
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


# How long shutdown waits for cancelled tasks to unwind before it reconciles
# their operations. Short: the tasks have already been cancelled and the process
# is on its way out — this only gives a task mid-way through its own terminal
# write the chance to land it, so shutdown does not hand back a row that is
# about to be marked completed.
_CANCEL_DRAIN_TIMEOUT = 5.0


def _metric_operation_label(operation_type: str | None) -> str:
    if operation_type in _RETAIN_OP_TYPES:
        return "retain"
    return operation_type or "unknown"


def _wall_timeout_for(task_type: str) -> float | None:
    """Wall-clock ceiling for one task of this type, or None when unbounded.

    A task that wedges (a lock wait with no deadlock cycle to break it, an LLM
    permit that never frees, a producer blocked on a queue nobody drains) holds
    its worker slot forever: the operation stays 'processing', which the API
    refuses to retry *or* cancel, and once every slot is held the worker stops
    claiming work entirely (#3002). Per-operation timeouts elsewhere bound one
    LLM call or one query, never the whole task — this is the outer backstop
    that turns "wedged until restart" into "failed and retryable".

    Only retain is bounded today; reflect self-bounds inside the engine
    (``reflect_wall_timeout``) and the remaining types have no reported wedge.
    """
    if task_type in _RETAIN_OP_TYPES:
        timeout = get_config().retain_wall_timeout
        return float(timeout) if timeout > 0 else None
    return None


class _WallTimeoutExceeded(Exception):
    """A task was cancelled because it blew through its wall-clock ceiling."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"wall-clock timeout after {timeout:.0f}s")
        self.timeout = timeout


def _updated_row_count(result: Any) -> int:
    """Extract a row count from backend execute() results."""
    if isinstance(result, int):
        return result
    if isinstance(result, str):
        try:
            return int(result.rsplit(" ", 1)[-1])
        except (TypeError, ValueError):
            return 0
    rowcount = getattr(result, "rowcount", None)
    return rowcount if isinstance(rowcount, int) else 0


if TYPE_CHECKING:
    from hindsight_api.engine.db.base import DatabaseBackend, DatabaseConnection
    from hindsight_api.extensions.tenant import TenantExtension

logger = logging.getLogger(__name__)

# Progress logging interval in seconds
PROGRESS_LOG_INTERVAL = 30

# Stuck-task stack-dump thresholds (seconds). Each task gets one stack dump
# per threshold it crosses (5min, 10min, 20min, 40min, 80min...).
STUCK_STACK_INITIAL_THRESHOLD_S = 300
STUCK_STACK_MAX_THRESHOLD_S = 3600 * 6  # cap doubling at 6h


def _summarise_child_error_messages(siblings: "Iterable[Any]") -> str:
    """Pick a representative error message for a parent whose children failed.

    Used when a batch_retain parent transitions to 'failed' because at least
    one child sub-batch failed. Without this, the parent gets a generic
    "One or more sub-batches failed" string and any consumer that reasons
    about errors via error_message (dashboards, alert filters, log
    aggregators) loses the actual cause -- a class of failures that all
    share the same root reason at the child level becomes indistinguishable
    at the parent level.

    Strategy: pick the most common non-empty error_message among failed
    siblings. If they all failed for the same reason (the common case), the
    parent inherits that reason verbatim. If they vary, the most-common one
    is still a useful representative. Falls back to the legacy generic
    string when no failed sibling carries an error_message at all.
    """
    failed_errors: list[str] = []
    for s in siblings:
        if s["status"] != "failed":
            continue
        msg = (s["error_message"] or "").strip()
        if msg:
            failed_errors.append(msg)
    if not failed_errors:
        return "One or more sub-batches failed"
    most_common, _count = Counter(failed_errors).most_common(1)[0]
    return most_common


@dataclass
class ActiveTaskInfo:
    """Tracking info for an in-flight worker task.

    Carries everything the periodic stats / stuck-task logger needs
    so it can render a useful per-task line without touching the DB.
    """

    op_type: str
    bank_id: str
    schema: str | None
    bg_task: "asyncio.Task[Any]"
    started_at: float
    stage_holder: StageHolder
    # Every operation row completed by this execution, including folded retain
    # peers. The lease heartbeat must cover all of them, not only the primary.
    operation_ids: list[str] = field(default_factory=list)
    # Largest stuck-stack threshold (seconds) for which we've already
    # dumped a stack trace; used to suppress repeated dumps.
    last_stack_dump_threshold: int = 0
    task_type: str = ""


@dataclass
class ClaimedTask:
    """A task claimed from the database with its schema context."""

    operation_id: str
    task_dict: dict[str, Any]
    schema: str | None
    folded_operation_ids: list[str] = field(default_factory=list)
    """Peers coalesced into this execution (see ``engine.retain.fold``).

    They were claimed by the same transaction that claimed ``operation_id`` and
    run as part of its execution, so they must reach a terminal state with it —
    never separately, and never not at all.
    """

    @property
    def all_operation_ids(self) -> list[str]:
        """Every operation this execution is responsible for completing."""
        return [self.operation_id, *self.folded_operation_ids]


@dataclass
class SlotAvailability:
    """Available slot capacity across reserved and shared pools.

    Each operation type with a reservation has its own reserved pool.
    The shared pool (max_slots - sum of reservations) is usable by any type.
    """

    reserved: dict[str, int]
    """Per-operation-type remaining reserved capacity."""

    shared: int
    """Remaining shared pool capacity (usable by any operation type)."""


class WorkerPoller:
    """
    Polls the database for pending tasks and executes them.

    Uses FOR UPDATE SKIP LOCKED for safe distributed claiming,
    allowing multiple workers to process tasks without conflicts.

    Supports dynamic multi-tenant discovery via tenant_extension.
    Backend-agnostic via DatabaseBackend abstraction.
    """

    def __init__(
        self,
        backend: "DatabaseBackend",
        worker_id: str,
        executor: Callable[[dict[str, Any]], Awaitable[None]],
        poll_interval_ms: int = 500,
        schema: str | None = None,
        tenant_extension: "TenantExtension | None" = None,
        max_slots: int = 10,
        slot_reservations: dict[str, int] | None = None,
        consolidation_bank_priority: dict[str, int] | None = None,
        max_retries: int = 3,
        heartbeat_interval_seconds: float = 30.0,
        stale_task_timeout_seconds: float = 300.0,
    ):
        """
        Initialize the worker poller.

        Args:
            backend: Database backend (PostgreSQL, Oracle, etc.)
            worker_id: Unique identifier for this worker
            executor: Async function to execute tasks (typically MemoryEngine.execute_task)
            poll_interval_ms: Interval between polls when no tasks found (milliseconds)
            schema: Database schema for single-tenant support (deprecated, use tenant_extension)
            tenant_extension: Extension for dynamic multi-tenant discovery. If None, creates a
                            DefaultTenantExtension with the configured schema.
            max_slots: Maximum concurrent tasks per worker
            slot_reservations: Per-operation-type reserved slot counts (e.g. {"consolidation": 2,
                "retain": 3}). Reserved slots guarantee capacity for that operation type.
                Remaining slots (max_slots - sum of reservations) form a shared pool usable
                by any operation type. Defaults to {"consolidation": 2} if None.
            consolidation_bank_priority: Per-bank priority for consolidation scheduling.
                Maps bank name patterns to integer priorities (higher = claimed first).
                Patterns support ``*`` as wildcard. A bare ``*`` key is the catch-all default.
                When set, consolidation tasks are claimed in priority tiers rather than
                pure created_at order. None or empty dict preserves current behavior.
            max_retries: Maximum retry attempts before a task is marked failed.
                Must be >= 0. Default 3 (matches DEFAULT_WORKER_MAX_RETRIES).
            heartbeat_interval_seconds: How often to refresh the lease for active
                operations. Defaults to 30 seconds.
            stale_task_timeout_seconds: How long a processing operation may go
                without a heartbeat before another worker reclaims it. Defaults
                to 300 seconds.
        """
        self._backend = backend
        self._worker_id = worker_id
        self._executor = executor
        self._poll_interval_ms = poll_interval_ms
        self._schema = schema
        # Always set tenant extension (use DefaultTenantExtension if none provided)
        if tenant_extension is None:
            from ..extensions.builtin.tenant import DefaultTenantExtension

            # Pass schema parameter to DefaultTenantExtension if explicitly provided
            config = {"schema": schema} if schema else {}
            tenant_extension = DefaultTenantExtension(config=config)
        self._tenant_extension = tenant_extension
        self._max_slots = max_slots
        self._slot_reservations: dict[str, int] = (
            slot_reservations if slot_reservations is not None else {"consolidation": 2}
        )
        self._consolidation_bank_priority: dict[str, int] | None = (
            consolidation_bank_priority if consolidation_bank_priority else None
        )
        self._max_retries = max(0, max_retries)  # Never negative
        self._heartbeat_interval_seconds = max(0.0, float(heartbeat_interval_seconds))
        self._stale_task_timeout_seconds = max(0.0, float(stale_task_timeout_seconds))
        # Cache of which optional PG routines are installed on the server
        # (probed once, memoised for the life of the poller).
        from ..engine.db.optional_routines import OptionalRoutines

        self._optional_routines = OptionalRoutines(self._backend)
        self._shutdown = asyncio.Event()
        self._current_tasks: set[asyncio.Task] = set()
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()
        self._last_progress_log = 0.0
        self._tasks_completed_since_log = 0
        # Monotonic stamp of the last completed claim cycle. Reported by the
        # liveness probe so operators can alert on a poller that stopped making
        # progress; None until the first cycle finishes.
        self._last_poll_at: float | None = None
        # Track active tasks locally: operation_id -> ActiveTaskInfo
        self._active_tasks: dict[str, ActiveTaskInfo] = {}
        # Track in-flight tasks by operation type
        self._in_flight_by_type: dict[str, int] = {}
        # Rotation offset for per-tenant fair claiming. Advances past the last
        # schema we serviced so a busy tenant can't monopolize the poll order.
        self._next_schema_idx: int = 0
        # Lease maintenance runs outside the claim loop. Keep one task per
        # poller so heartbeat/recovery cannot overlap with itself or block slot
        # refill. It is created by run() and cancelled during shutdown.
        self._lease_maintenance_task: asyncio.Task[Any] | None = None

    @staticmethod
    def _normalize_poll_schema(schema: str | None) -> str | None:
        """Use None internally for the default schema because SQL helpers omit that prefix."""
        from ..config import DEFAULT_DATABASE_SCHEMA

        return None if schema == DEFAULT_DATABASE_SCHEMA else schema

    async def _get_schemas(self) -> list[str | None]:
        """Get list of schemas to poll. Returns [None] for default schema (no prefix)."""
        tenants = await self._tenant_extension.list_tenants()
        # Convert default schema to None for SQL compatibility (no prefix), keep others as-is
        return [self._normalize_poll_schema(t.schema) for t in tenants]

    async def _scan_active_schemas(self, conn: "DatabaseConnection", schemas: list[str | None]) -> set[str | None]:
        """Find which schemas have pending work.

        Prefers a server-side PL/pgSQL routine (single DB round-trip,
        ~200ms for 1400+ schemas) when ``public.schemas_with_pending_work()``
        is installed. The presence check goes through
        ``OptionalRoutines.is_installed`` which probes ``pg_proc`` once and
        caches the result, so we don't generate a server-side error on
        every poll cycle when the routine isn't installed.

        Falls back to per-schema Python EXISTS queries (~4ms each) on
        non-PostgreSQL backends or when the routine isn't installed. See
        ``hindsight_api.engine.db.optional_routines`` for the canonical
        install SQL.

        Runs on the poll cycle's connection — see ``claim_batch``.
        """
        if await self._optional_routines.is_installed(conn, "schemas_with_pending_work"):
            # The routine IS the authority on where work exists: every schema
            # it returns is claimable, and every schema it does NOT return is
            # treated as having nothing to do this cycle. That is the entire
            # point of installing it — one round-trip replaces N per-schema
            # EXISTS probes. We deliberately do NOT re-verify the omitted
            # schemas with a per-schema scan: that re-runs the exact queries
            # the routine exists to avoid, on every idle poll, silently
            # negating the optimisation.
            #
            # Because the result is trusted wholesale, the routine is only
            # appropriate for multi-tenant deployments. A single-schema
            # (default/public only) install should NOT create it and instead
            # falls through to the per-schema path below — a single cheap
            # EXISTS check that cannot starve. See
            # ``hindsight_api.engine.db.optional_routines``.
            rows = await conn.fetch("SELECT * FROM public.schemas_with_pending_work()")
            return {self._normalize_poll_schema(r[0]) for r in rows}

        return await self._scan_active_schemas_by_exists(conn, schemas)

    async def _scan_active_schemas_by_exists(
        self, conn: "DatabaseConnection", schemas: list[str | None]
    ) -> set[str | None]:
        """Find active schemas using per-schema EXISTS checks."""
        active: set[str | None] = set()
        for schema in schemas:
            table = fq_table("async_operations", schema)
            try:
                has_work = await conn.fetchval(
                    f"SELECT EXISTS(SELECT 1 FROM {table} "
                    f"WHERE status = 'pending' AND task_payload IS NOT NULL LIMIT 1)"
                )
                if has_work:
                    active.add(schema)
            except Exception:
                pass
        return active

    async def _get_available_slots(self) -> SlotAvailability:
        """
        Calculate available slots for claiming tasks.

        Each operation type can have reserved slots (via ``slot_reservations``).
        Reserved slots guarantee capacity for that type — they cannot be used by
        other types. The remaining slots (``max_slots - sum(reservations)``) form
        a shared pool usable by any operation type on a first-come basis.

        When an operation type's in-flight count exceeds its reservation, the
        excess tasks are considered to be using shared pool slots.
        """
        async with self._in_flight_lock:
            total_in_flight = self._in_flight_count
            in_flight_snapshot = dict(self._in_flight_by_type)

        # Per-type reserved availability
        reserved_available: dict[str, int] = {}
        tasks_in_reserved = 0
        for op_type, reserved in self._slot_reservations.items():
            in_flight = in_flight_snapshot.get(op_type, 0)
            reserved_available[op_type] = max(0, reserved - in_flight)
            tasks_in_reserved += min(reserved, in_flight)

        # Shared pool: total slots minus reservations minus tasks using shared slots
        sum_reservations = sum(self._slot_reservations.values())
        shared_pool_size = max(0, self._max_slots - sum_reservations)
        tasks_in_shared = max(0, total_in_flight - tasks_in_reserved)
        shared_available = max(0, shared_pool_size - tasks_in_shared)

        return SlotAvailability(reserved=reserved_available, shared=shared_available)

    async def wait_for_active_tasks(self, timeout: float = 10.0) -> bool:
        """
        Wait for all active background tasks to complete (test helper).

        This is a test-only utility that allows tests to synchronize with
        fire-and-forget background tasks without using sleep().

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout was reached
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            async with self._in_flight_lock:
                if self._in_flight_count == 0:
                    return True

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return False

            # Short sleep to avoid busy-waiting
            await asyncio.sleep(0.01)

    async def claim_batch(self) -> list[ClaimedTask]:
        """
        Claim pending tasks atomically across all tenant schemas,
        respecting per-operation-type slot reservations and shared pool limits.

        Uses FOR UPDATE SKIP LOCKED to ensure no conflicts with other workers.

        Schema iteration is round-robin to prevent one busy tenant from
        starving others. Each poll starts at ``self._next_schema_idx`` and
        wraps around the full list. First pass caps at 1 claim per pool per
        schema so every tenant with pending work gets a fair chance; a second
        pass backfills remaining slots from any schema when there's spare
        capacity. After the call, the offset advances past the last
        schema we serviced (or by 1 if nothing was claimed) so the next
        poll starts at a different position.

        Returns:
            List of ClaimedTask objects containing operation_id, task_dict, and schema
        """
        # Calculate available slots (per-type reserved + shared pool)
        availability = await self._get_available_slots()

        if all(v <= 0 for v in availability.reserved.values()) and availability.shared <= 0:
            return []

        schemas = await self._get_schemas()
        if not schemas:
            return []

        # One pooled connection for the whole cycle — the scan *and* every
        # per-schema claim. Each acquire pays the pool's setup callback (the
        # session GUCs) and each release pays asyncpg's RESET ALL / UNLISTEN /
        # CLOSE ALL; behind a transaction-mode pooler every one of those is its
        # own server-side transaction. Acquiring per schema multiplied that
        # ceremony by the number of active schemas — ~12 statements per
        # schema-visit for 2 useful queries (#3499). The per-schema claims stay
        # sequential and each still runs in its own transaction, so
        # FOR UPDATE SKIP LOCKED semantics are unchanged by sharing the
        # connection.
        async with self._backend.acquire() as conn:
            return await self._claim_batch_on_conn(conn, availability, schemas)

    async def _claim_batch_on_conn(
        self,
        conn: "DatabaseConnection",
        availability: SlotAvailability,
        schemas: list[str | None],
    ) -> list[ClaimedTask]:
        """Run one full claim cycle (scan + per-schema claims) on a single connection."""
        # Scan: find which schemas have pending work using a lightweight
        # EXISTS check (no locks). Then only claim from those schemas
        # using the expensive FOR UPDATE SKIP LOCKED query.
        active_schemas = await self._scan_active_schemas(conn, schemas)

        if not active_schemas:
            self._next_schema_idx = (self._next_schema_idx + 1) % len(schemas)
            return []

        # Build rotation list from active schemas only, preserving their
        # original positions for correct offset advancement.
        all_indexed = list(enumerate(schemas))
        active_indexed = [(i, s) for i, s in all_indexed if s in active_schemas]

        # Rotate so no tenant is always first.
        start = self._next_schema_idx % len(schemas)
        rotated = [x for x in active_indexed if x[0] >= start] + [x for x in active_indexed if x[0] < start]

        all_tasks: list[ClaimedTask] = []
        remaining_reserved = dict(availability.reserved)
        remaining_shared = availability.shared
        last_serviced_idx: int | None = None
        schemas_with_work: list[tuple[int, str | None]] = []

        def _has_capacity() -> bool:
            return any(v > 0 for v in remaining_reserved.values()) or remaining_shared > 0

        def _account_tasks(tasks: list[ClaimedTask]) -> None:
            nonlocal remaining_shared
            for task in tasks:
                op_type = task.task_dict.get("operation_type", "unknown")
                if op_type in remaining_reserved and remaining_reserved[op_type] > 0:
                    remaining_reserved[op_type] -= 1
                else:
                    remaining_shared -= 1

        # Pass 1: fairness pass — iterate only active schemas, cap at
        # 1 claim per pool per schema.
        for orig_idx, schema in rotated:
            if not _has_capacity():
                break

            fair_reserved = {t: min(1, v) for t, v in remaining_reserved.items() if v > 0}
            fair_shared = min(1, remaining_shared) if remaining_shared > 0 else 0
            tasks = await self._claim_batch_for_schema(conn, schema, fair_reserved, fair_shared)

            _account_tasks(tasks)

            if tasks:
                last_serviced_idx = orig_idx
                schemas_with_work.append((orig_idx, schema))

            all_tasks.extend(tasks)

        # Pass 2: capacity pass — fill remaining slots from schemas
        # that had work in pass 1 only.
        if _has_capacity() and schemas_with_work:
            for orig_idx, schema in schemas_with_work:
                if not _has_capacity():
                    break

                tasks = await self._claim_batch_for_schema(
                    conn, schema, {t: v for t, v in remaining_reserved.items() if v > 0}, remaining_shared
                )

                _account_tasks(tasks)

                if tasks:
                    last_serviced_idx = orig_idx

                all_tasks.extend(tasks)

        # Advance offset past the last schema we serviced, or by 1 if
        # nothing was claimed (so we don't keep re-hitting an empty head).
        if last_serviced_idx is not None:
            self._next_schema_idx = (last_serviced_idx + 1) % len(schemas)
        else:
            self._next_schema_idx = (start + 1) % len(schemas)

        return all_tasks

    async def _claim_batch_for_schema(
        self,
        conn: "DatabaseConnection",
        schema: str | None,
        reserved_limits: dict[str, int],
        shared_limit: int,
    ) -> list[ClaimedTask]:
        """Claim tasks from a specific schema respecting per-type and shared slot limits."""
        try:
            return await self._claim_batch_for_schema_inner(conn, schema, reserved_limits, shared_limit)
        except Exception as e:
            # Format schema for logging: custom schemas in quotes, None as-is
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.warning(f"Worker {self._worker_id} failed to claim tasks for schema {schema_display}: {e}")
            return []

    async def _claim_batch_for_schema_inner(
        self,
        conn: "DatabaseConnection",
        schema: str | None,
        reserved_limits: dict[str, int],
        shared_limit: int,
    ) -> list[ClaimedTask]:
        """Inner implementation for claiming tasks from a specific schema.

        Delegates the SQL claiming logic to backend.ops.claim_tasks() which
        handles backend-specific differences (e.g. Oracle's ORA-02014 workaround).

        Runs on the caller's connection (one per poll cycle) but opens its own
        transaction: the claim's row locks are held only until this schema's
        claim commits, not for the whole cycle.
        """
        table = fq_table("async_operations", schema)

        async with conn.transaction():
            all_rows = await self._backend.ops.claim_tasks(
                conn,
                table,
                self._worker_id,
                reserved_limits,
                shared_limit,
                consolidation_bank_priority=self._consolidation_bank_priority,
            )

            if not all_rows:
                return []

            result = []
            for row in all_rows:
                payload = row["task_payload"]
                # Oracle may return JSON columns as dict directly
                task_dict = json.loads(payload) if isinstance(payload, str) else payload
                task_dict["_retry_count"] = row["retry_count"]
                task_dict["_operation_id"] = str(row["operation_id"])
                # The DB column is authoritative for operation_type — inject it
                # into task_dict so in-flight tracking and slot accounting work.
                db_op_type = row["operation_type"]
                if db_op_type:
                    task_dict["operation_type"] = db_op_type
                folded = await self._fold_retain_peers(conn, table, row, task_dict)
                result.append(
                    ClaimedTask(
                        operation_id=str(row["operation_id"]),
                        task_dict=task_dict,
                        schema=schema,
                        folded_operation_ids=folded,
                    )
                )
            return result

    async def _fold_retain_peers(self, conn, table: str, row, task_dict: dict[str, Any]) -> list[str]:
        """Coalesce the retains queued behind ``row`` into its execution.

        Runs inside the claim transaction, so the peers are locked, claimed and
        handed over atomically with the row they join — there is no window in
        which a peer is folded but not claimed, or claimed but not executed.

        Peers this does not take stay pending and are claimed on a later cycle.
        Purely an optimization: on any failure the execution proceeds unfolded,
        one operation at a time, which is exactly the pre-folding behaviour.
        """
        from ..engine.retain.fold import (
            FoldMember,
            FoldMemberRef,
            max_fold_peers_for_retry,
            merge_fold_contents,
            plan_retain_fold,
        )

        serialization_key = row["serialization_key"]
        if not serialization_key or row["operation_type"] != "retain":
            return []
        max_peers = max_fold_peers_for_retry(row["retry_count"] or 0)
        if max_peers <= 0:
            return []

        try:
            from ..config import get_config
            from ..engine.memory_engine import count_tokens

            peer_rows = await self._backend.ops.fetch_foldable_retain_peers(
                conn,
                table,
                row["bank_id"],
                serialization_key,
                max_peers,
            )
            if not peer_rows:
                return []

            def _member(operation_id, payload) -> FoldMember:
                data = json.loads(payload) if isinstance(payload, str) else payload
                return FoldMember(
                    operation_id=str(operation_id),
                    contents=data.get("contents", []),
                    tenant_id=data.get("_tenant_id"),
                    api_key_id=data.get("_api_key_id"),
                    document_tags=data.get("document_tags"),
                    strategy=data.get("strategy"),
                    # A converted upload attaches its storage key to the document
                    # afterwards, and that step only fires for a single-item
                    # retain — so such an operation must never be folded.
                    has_file_metadata=data.get("_file_metadata") is not None,
                )

            primary = _member(row["operation_id"], row["task_payload"])
            peers = [_member(p["operation_id"], p["task_payload"]) for p in peer_rows]
            plan = plan_retain_fold(
                primary,
                peers,
                max_peers=max_peers,
                token_budget=get_config().retain_batch_tokens,
                count_tokens=count_tokens,
            )
            if not plan.peer_ids:
                return []

            # Everything that can fail happens before the peers are claimed, so
            # a failure can only leave them pending — never claimed by an
            # execution that then doesn't carry them.
            folded_ids = set(plan.peer_ids)
            merged_contents = merge_fold_contents(plan.members)
            fold_members = [
                FoldMemberRef(operation_id=m.operation_id, items_count=len(m.contents)).to_payload()
                for m in plan.members
            ]
            await self._backend.ops.mark_operations_processing(
                conn,
                table,
                self._worker_id,
                [p["operation_id"] for p in peer_rows if str(p["operation_id"]) in folded_ids],
            )
            task_dict["contents"] = merged_contents
            task_dict["_fold_members"] = fold_members
            logger.info(
                f"Folded {len(plan.peer_ids)} queued retain(s) for document "
                f"{serialization_key} into operation {row['operation_id']}"
                + (f" ({len(plan.deferred)} left pending)" if plan.deferred else "")
            )
            return plan.peer_ids
        except Exception:
            logger.warning(
                f"Retain fold for document {serialization_key} failed — "
                f"running operation {row['operation_id']} on its own",
                exc_info=True,
            )
            return []

    # -- Fold-aware terminal transitions -----------------------------------
    #
    # A folded execution is responsible for every operation the claim
    # transaction handed it, not just the row it was keyed on. Routing all
    # terminal transitions through these wrappers is what makes that structural:
    # each one fans out over ``task.all_operation_ids``, so no branch can leave
    # a folded peer stuck in 'processing' by only remembering the primary.
    #
    # Every member gets the same outcome, because a fold is one execution:
    # they either all committed or none did. Retry and deferral put the whole
    # group back to pending, so it is re-claimed (and re-folded, more narrowly
    # each time) together.

    async def _mark_all_completed(self, task: ClaimedTask) -> None:
        for operation_id in task.all_operation_ids:
            await self._mark_completed(operation_id, task.schema)

    async def _mark_all_failed(self, task: ClaimedTask, error_message: str) -> None:
        for operation_id in task.all_operation_ids:
            await self._mark_failed(operation_id, error_message, task.schema)

    async def _defer_all(self, task: ClaimedTask, exec_date, reason: str) -> None:
        for operation_id in task.all_operation_ids:
            await self._defer_operation(operation_id, exec_date, reason, task.schema)

    async def _schedule_retry_all(self, task: ClaimedTask, retry_at, reason: str) -> None:
        for operation_id in task.all_operation_ids:
            await self._schedule_retry(operation_id, retry_at, reason, task.schema)

    async def _mark_completed(self, operation_id: str, schema: str | None) -> None:
        """Complete a task only while this worker still owns its processing lease."""
        table = fq_table("async_operations", schema)
        async with self._backend.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'completed', completed_at = now(), updated_at = now()
                    WHERE operation_id = $1 AND status = 'processing' AND worker_id = $2
                    """,
                    operation_id,
                    self._worker_id,
                )
                if _updated_row_count(result):
                    await self._maybe_update_parent_operation(operation_id, schema, conn)

    async def _mark_failed(self, operation_id: str, error_message: str, schema: str | None) -> None:
        """Fail a task only while this worker still owns its processing lease."""
        table = fq_table("async_operations", schema)
        # Truncate error message if too long (max 5000 chars in schema)
        error_message = error_message[:5000] if len(error_message) > 5000 else error_message

        async with self._backend.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'failed', error_message = $2, completed_at = now(), updated_at = now()
                    WHERE operation_id = $1 AND status = 'processing' AND worker_id = $3
                    """,
                    operation_id,
                    error_message,
                    self._worker_id,
                )
                if _updated_row_count(result):
                    await self._maybe_update_parent_operation(operation_id, schema, conn)

    async def _maybe_update_parent_operation(self, child_operation_id: str, schema: str | None, conn) -> None:
        """If this operation is a child of a batch_retain, update the parent status when all siblings are done.

        Must be called within an active transaction that has already updated the child's status.
        The memory engine has an equivalent method that runs inside task execution transactions.
        This poller-level version handles the case where a task fails via an unhandled exception
        that bypasses the memory engine's own failure path (e.g. a DB constraint violation that
        rolls back the engine's transaction before it can update the parent).
        """
        import json
        import uuid

        table = fq_table("async_operations", schema)

        try:
            row = await conn.fetchrow(
                f"SELECT result_metadata, bank_id FROM {table} WHERE operation_id = $1",
                uuid.UUID(child_operation_id),
            )
            if not row:
                return

            result_metadata = row["result_metadata"] or {}
            if isinstance(result_metadata, str):
                result_metadata = json.loads(result_metadata)
            parent_operation_id = result_metadata.get("parent_operation_id")
            if not parent_operation_id:
                return

            bank_id = row["bank_id"]

            # Lock parent to prevent concurrent sibling updates
            parent_row = await conn.fetchrow(
                f"SELECT operation_id FROM {table} WHERE operation_id = $1 AND bank_id = $2 FOR UPDATE",
                uuid.UUID(parent_operation_id),
                bank_id,
            )
            if not parent_row:
                return

            # Check whether all siblings are done. Pull error_message too so a
            # parent that fails can inherit a representative child reason --
            # otherwise the parent's error_message is generic ("One or more
            # sub-batches failed") and downstream consumers (dashboards, alerts,
            # filters) lose the actual cause once a batch has children.
            siblings = await conn.fetch(
                f"""
                SELECT status, error_message FROM {table}
                WHERE bank_id = $1
                  AND result_metadata::jsonb @> $2::jsonb
                """,
                bank_id,
                json.dumps({"parent_operation_id": parent_operation_id}),
            )
            if not siblings or not all(s["status"] in ("completed", "failed") for s in siblings):
                return

            any_failed = any(s["status"] == "failed" for s in siblings)
            if any_failed:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'failed', error_message = $2, updated_at = now()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                    _summarise_child_error_messages(siblings),
                )
            else:
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'completed', updated_at = now(), completed_at = now()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                )
            logger.info(
                f"Poller updated parent operation {parent_operation_id} to "
                f"{'failed' if any_failed else 'completed'} (all siblings done)"
            )
        except Exception as e:
            # Log but don't re-raise — the child has already been marked failed,
            # which is the critical state change. A stuck parent will be caught on
            # the next run or via monitoring.
            logger.error(f"Failed to update parent operation for child {child_operation_id}: {e}")

    async def _schedule_retry(
        self,
        operation_id: str,
        retry_at: "Any",
        error_message: str,
        schema: str | None,
    ) -> None:
        """Retry a task only while this worker still owns its processing lease."""
        table = fq_table("async_operations", schema)
        error_message = error_message[:5000] if len(error_message) > 5000 else error_message
        async with self._backend.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {table}
                SET status = 'pending', next_retry_at = $2, worker_id = NULL, claimed_at = NULL,
                    retry_count = retry_count + 1, error_message = $3, updated_at = now()
                WHERE operation_id = $1 AND status = 'processing' AND worker_id = $4
                """,
                operation_id,
                retry_at,
                error_message,
                self._worker_id,
            )
        logger.warning(f"Task {operation_id} scheduled for retry at {retry_at}: {error_message}")

    async def _defer_operation(
        self,
        operation_id: str,
        exec_date: "Any",
        reason: str,
        schema: str | None,
    ) -> None:
        """Reset task to pending for re-pickup at exec_date without counting as a retry.

        Unlike `_schedule_retry`, this does not bump `retry_count` and does not
        populate `error_message` — defer is intentional backpressure, not a failure.
        """
        table = fq_table("async_operations", schema)
        async with self._backend.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {table}
                SET status = 'pending', next_retry_at = $2, worker_id = NULL, claimed_at = NULL,
                    updated_at = now()
                WHERE operation_id = $1 AND status = 'processing' AND worker_id = $3
                """,
                operation_id,
                exec_date,
                self._worker_id,
            )
        logger.info(f"Task {operation_id} deferred until {exec_date}: {reason}")

    async def execute_task(self, task: ClaimedTask):
        """Execute a single task as a background job (fire-and-forget)."""
        task_type = task.task_dict.get("type", "unknown")
        operation_type = task.task_dict.get("operation_type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        # Stage holder is updated by engine code via stage.set_stage(); the
        # poller reads it during periodic logging to surface what each
        # in-flight task is doing.
        holder = StageHolder(stage=f"queued.{task_type}")

        # Create background task. The holder is passed in and bound to the
        # task's own contextvar scope inside _execute_task_inner so engine
        # code running under that task sees it via stage.set_stage().
        bg_task = asyncio.create_task(self._execute_task_inner(task, holder))

        # Track this task as active
        async with self._in_flight_lock:
            self._active_tasks[task.operation_id] = ActiveTaskInfo(
                op_type=operation_type,
                bank_id=bank_id,
                schema=task.schema,
                bg_task=bg_task,
                started_at=time.monotonic(),
                stage_holder=holder,
                operation_ids=task.all_operation_ids,
                task_type=task_type,
            )
            self._in_flight_count += 1
            self._in_flight_by_type[operation_type] = self._in_flight_by_type.get(operation_type, 0) + 1

        # Add cleanup callback
        bg_task.add_done_callback(lambda _: asyncio.create_task(self._cleanup_task(task.operation_id, operation_type)))

    async def _cleanup_task(self, operation_id: str, operation_type: str):
        """Remove task from tracking after completion."""
        async with self._in_flight_lock:
            if operation_id in self._active_tasks:
                self._active_tasks.pop(operation_id, None)
                self._in_flight_count -= 1
                count = self._in_flight_by_type.get(operation_type, 0)
                if count > 0:
                    self._in_flight_by_type[operation_type] = count - 1
                    if self._in_flight_by_type[operation_type] == 0:
                        del self._in_flight_by_type[operation_type]

    async def _run_executor(self, task: ClaimedTask, task_type: str) -> None:
        """Run the task executor under its type's wall-clock ceiling, if it has one."""
        wall_timeout = _wall_timeout_for(task_type)
        if wall_timeout is None:
            await self._executor(task.task_dict)
            return

        # asyncio.timeout() rather than wait_for(): `expired()` distinguishes our
        # ceiling firing from an inner TimeoutError merely bubbling out (an asyncpg
        # command timeout, say), which wait_for would surface as the same exception.
        # Reporting a task's own timeout as a wedge would send operators hunting for
        # the wrong thing.
        try:
            async with asyncio.timeout(wall_timeout) as cm:
                await self._executor(task.task_dict)
        except asyncio.TimeoutError as e:
            if cm.expired():
                raise _WallTimeoutExceeded(wall_timeout) from e
            raise

    async def _execute_task_inner(self, task: ClaimedTask, holder: StageHolder | None = None):
        """Inner task execution with retry/fail handling.

        Tasks that want to be retried raise RetryTaskAt; the poller sets next_retry_at
        and resets status to 'pending'. All other exceptions are marked as failed immediately.
        Non-retryable failures (e.g., file_convert_retain) are handled by the executor
        internally — it marks the operation as failed and returns normally.
        """
        task_type = task.task_dict.get("type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")
        # Operation metric (source="worker"): record on terminal outcomes only, so
        # async worker throughput and latency (retain, consolidation and the other
        # worker task types) are visible in Prometheus. Prefer the DB-authoritative
        # operation_type.
        #
        # success semantics are deliberately narrow: success=false means the task
        # raised out to the poller (an unexpected error, or retry-exhausted). It does
        # NOT capture deterministic failures that the executor handles itself and
        # returns from normally (file_convert_retain, non-retryable errors via
        # memory_engine.execute_task) — those record success=true here. Treat this as
        # a completion-throughput signal, not a failure-rate one: for authoritative
        # failure visibility use the hindsight_async_operations{status="failed"} gauge,
        # which reads each operation's final DB status.
        op_label = _metric_operation_label(task.task_dict.get("operation_type") or task_type)
        op_start = time.time()
        metrics = get_metrics_collector()
        # None = not a terminal outcome (deferred/retried) → no metric.
        terminal_success: bool | None = None

        # Bind the stage holder in this task's own contextvar scope so engine
        # code running under us can update it via stage.set_stage(). If holder
        # is None (legacy / direct invocation), set_stage becomes a no-op.
        if holder is not None:
            bind_holder(holder)
            holder.stage = f"executor.{task_type}"
            holder.updated_at = time.monotonic()

        try:
            schema_info = f", schema={task.schema}" if task.schema else ""
            logger.debug(f"Executing task {task.operation_id} (type={task_type}, bank={bank_id}{schema_info})")
            if task.schema:
                task.task_dict["_schema"] = task.schema
            await self._run_executor(task, task_type)
            logger.debug(f"Task {task.operation_id} execution finished")
            await self._mark_all_completed(task)
            terminal_success = True
        except _WallTimeoutExceeded as e:
            # The executor has already been cancelled; all that's left is to say so
            # clearly. Handled apart from the generic branch below so the operator
            # sees the wedge for what it is rather than a bare "TimeoutError", and
            # so the stage that was current when the ceiling fired is preserved —
            # that breadcrumb is the only pointer to where the task was stuck.
            stage = holder.stage if holder is not None else "unknown"
            message = (
                f"Task exceeded the {e.timeout:.0f}s wall-clock limit for '{task_type}' "
                f"(stage={stage}) and was cancelled. Raise HINDSIGHT_API_RETAIN_WALL_TIMEOUT "
                f"if this is a legitimately long operation, or set it to 0 to disable the limit."
            )
            logger.error(f"Task {task.operation_id} timed out: {message}")
            await self._mark_all_failed(task, message)
            terminal_success = False
        except DeferOperation as e:
            # Deferral is not a terminal outcome — do not record a completion.
            await self._defer_all(task, e.exec_date, e.reason)
        except RetryTaskAt as e:
            # Retry is not a terminal outcome — do not record a completion.
            await self._schedule_retry_all(task, e.retry_at, str(e))
        except Exception as e:
            # exc_info rather than print_exc(): the stderr copy carries no task id
            # and is the first thing lost to log rotation (issue #3218).
            error_message = format_task_error(e)
            logger.error(f"Task {task.operation_id} failed: {error_message}", exc_info=True)
            try:
                await self._mark_all_failed(task, error_message)
            except Exception:
                # Marking a task failed is itself a DB write, and it can fail
                # (pool exhausted, connection reset, statement timeout). Without
                # this rescue the row stays 'processing' under a worker that has
                # already forgotten it: _cleanup_task drops it from _active_tasks,
                # recover_own_tasks only runs at startup, and no dead-worker logic
                # applies because the worker is alive. See issue #3228.
                logger.exception(f"Could not mark task {task.operation_id} failed; reconciling it for re-claim")
                for operation_id in task.all_operation_ids:
                    try:
                        await self._reclaim_own_processing_tasks(task.schema, operation_id=operation_id)
                    except Exception:
                        logger.exception(f"Could not reconcile task {operation_id}; it stays 'processing'")
            terminal_success = False

        # Record the metric outside the executor's exception scope so a metrics
        # reporting failure can never be mistaken for a task failure and flip terminal state.
        if terminal_success is not None:
            try:
                metrics.record_operation_result(
                    op_label, bank_id, success=terminal_success, duration=time.time() - op_start, source="worker"
                )
            except Exception:
                logger.warning(f"Failed to record worker operation metric for {task.operation_id}", exc_info=True)

    async def _heartbeat_active_tasks(self) -> int:
        """Refresh the lease timestamp for operations executing in this process.

        ``updated_at`` is already the durable progress timestamp for async
        operations. Refreshing it independently of engine progress checkpoints
        keeps a healthy task leased while it is waiting on an external provider
        or another long phase. The worker/status predicates make the update
        harmless if a task completed between the snapshot and this query.

        Returns the number of rows refreshed. Heartbeats are best effort: a
        transient database error must not cancel the underlying task; the next
        tick can refresh it again.
        """
        by_schema: dict[str | None, list[str]] = {}
        async with self._in_flight_lock:
            for operation_id, info in self._active_tasks.items():
                ids = by_schema.setdefault(info.schema, [])
                operation_ids = info.operation_ids or [operation_id]
                for operation_id in operation_ids:
                    if operation_id not in ids:
                        ids.append(operation_id)

        refreshed = 0
        for schema, operation_ids in by_schema.items():
            if not operation_ids:
                continue
            table = fq_table("async_operations", schema)
            try:
                async with self._backend.acquire() as conn:
                    result = await conn.execute(
                        f"""
                        UPDATE {table}
                        SET updated_at = now()
                        WHERE status = 'processing'
                          AND worker_id = $1
                          AND operation_id = ANY($2)
                        """,
                        self._worker_id,
                        operation_ids,
                    )
                    refreshed += _updated_row_count(result)
            except Exception as e:
                logger.debug(
                    f"Worker {self._worker_id} failed to heartbeat "
                    f"{len(operation_ids)} operation(s) in schema {schema or 'default'}: {e}"
                )
        return refreshed

    async def _reclaim_stale_processing_tasks(self, schema: str | None, stale_before: datetime) -> int:
        """Return processing rows whose lease has expired.

        A lease is considered alive when ``updated_at`` (falling back to
        ``claimed_at``/``created_at`` for rows written by older versions) is
        newer than ``stale_before``. Rows are changed atomically under the
        processing predicate, so a concurrent heartbeat wins the race after
        PostgreSQL/Oracle re-checks the row lock. This also covers a worker
        whose own heartbeat was lost during a prolonged database outage;
        filtering by worker id here would leave that row stranded indefinitely.

        Batch-provider operations are deliberately excluded here. Their remote
        job may continue after the Hindsight worker exits and they are recovered
        by ``_recover_batch_operations`` with the same lease cutoff, without
        consuming a worker retry.

        Returns the number of rows returned to ``pending`` (failed rows are
        logged and propagated to any batch parent but are not counted).
        """
        table = fq_table("async_operations", schema)
        max_retries = self._max_retries
        stale_timestamp = "COALESCE(updated_at, claimed_at, created_at) < $1"
        batch_filter = "result_metadata->>'batch_id' IS NULL"
        failed_rows = []

        async with self._backend.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'pending', worker_id = NULL, claimed_at = NULL,
                        next_retry_at = NULL,
                        retry_count = COALESCE(retry_count, 0) + 1,
                        updated_at = now()
                    WHERE status = 'processing'
                      AND {stale_timestamp}
                      AND task_payload IS NOT NULL
                      AND {batch_filter}
                      AND COALESCE(retry_count, 0) < $2
                    """,
                    stale_before,
                    max_retries,
                )

                failed_rows = await conn.fetch(
                    f"""
                    UPDATE {table}
                    SET status = 'failed', worker_id = NULL, claimed_at = NULL,
                        error_message = $3, completed_at = now(), updated_at = now()
                    WHERE status = 'processing'
                      AND {stale_timestamp}
                      AND task_payload IS NOT NULL
                      AND {batch_filter}
                      AND COALESCE(retry_count, 0) >= $2
                    RETURNING operation_id
                    """,
                    stale_before,
                    max_retries,
                    f"stale worker lease expired (retry_count >= {max_retries})",
                )

        # The failed UPDATE has committed before parent propagation, matching
        # _reclaim_own_processing_tasks: a propagation failure cannot undo the
        # terminal child state.
        for failed_row in failed_rows:
            try:
                async with self._backend.acquire() as conn:
                    async with conn.transaction():
                        await self._maybe_update_parent_operation(str(failed_row["operation_id"]), schema, conn)
            except Exception:
                logger.exception(
                    f"Worker {self._worker_id} could not propagate stale failure "
                    f"for operation {failed_row['operation_id']}"
                )

        if failed_rows:
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.warning(
                f"Worker {self._worker_id} moved {len(failed_rows)} stale tasks to 'failed' "
                f"(exceeded {max_retries} recovery attempts in schema {schema_display})"
            )
        return _updated_row_count(result)

    async def recover_stale_tasks(self) -> int:
        """Recover processing rows whose worker lease has expired.

        This is intentionally independent of ``worker_id`` stability: a new
        container can reclaim work left by a hostname-derived id after the old
        worker stops heartbeating. The method is safe to call at startup and on
        every maintenance tick.
        """
        if self._stale_task_timeout_seconds <= 0:
            return 0

        try:
            schemas = await self._get_schemas()
        except Exception as e:
            logger.warning(f"Worker {self._worker_id} could not list schemas for stale-task recovery: {e}")
            return 0

        stale_before = datetime.now(UTC) - timedelta(seconds=self._stale_task_timeout_seconds)
        total_recovered = 0
        for schema in schemas:
            try:
                total_recovered += await self._reclaim_stale_processing_tasks(schema, stale_before)
                # Batch-provider rows use a separate recovery path because a
                # remote batch may outlive the worker and must not spend a retry.
                total_recovered += await self._recover_batch_operations(schema, stale_before=stale_before)
                # Parent reconciliation scans every pending batch aggregator and
                # is intentionally startup-only. Stale child failures propagate
                # their parent above; repeating the full scan on every heartbeat
                # would add avoidable cross-tenant work to lease maintenance.
            except Exception as e:
                schema_display = f'"{schema}"' if schema else str(schema)
                logger.warning(f"Worker {self._worker_id} failed stale-task recovery for schema {schema_display}: {e}")

        if total_recovered:
            logger.info(f"Worker {self._worker_id} recovered {total_recovered} stale tasks")
        return total_recovered

    async def _lease_maintenance_loop(self) -> None:
        """Heartbeat active operations and reclaim expired leases periodically."""
        if self._heartbeat_interval_seconds <= 0:
            return

        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=self._heartbeat_interval_seconds,
                )
                break
            except asyncio.TimeoutError:
                pass

            if self._shutdown.is_set():
                break

            try:
                await self._heartbeat_active_tasks()
                await self.recover_stale_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Worker {self._worker_id} lease maintenance failed: {e}", exc_info=True)

    async def _stop_lease_maintenance(self) -> None:
        """Cancel and await the lease task, if the poller has started one."""
        task = self._lease_maintenance_task
        if task is None or task is asyncio.current_task():
            return
        self._lease_maintenance_task = None
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _reclaim_own_processing_tasks(self, schema: str | None, *, operation_id: str | None = None) -> int:
        """Reconcile rows still claimed by this worker in one schema.

        Rows under the retry budget go back to 'pending'; rows at/over it are
        moved to 'failed' so a task that reliably kills its worker cannot be
        re-claimed forever (claim → grind → die → reclaim → …, see #2675/#2834).
        Batch API operations are excluded — they are long-lived by design and
        `_recover_batch_operations` resets them without spending a retry.

        Every caller that reconciles this worker's own rows goes through here so
        the guards stay in one place: startup recovery (`recover_own_tasks`),
        shutdown release (`release_own_tasks`), and the single-operation rescue
        when a terminal write itself fails (`operation_id` set).

        Returns:
            Number of rows reset to pending (not including those failed).
        """
        table = fq_table("async_operations", schema)
        max_retries = self._max_retries
        # Two separate UPDATEs so their row counts are meaningful:
        #   1. Rows under the limit → increment retry_count, reset to pending
        #   2. Rows at/over the limit → move to failed with a clear reason
        op_filter = "AND operation_id = $3" if operation_id is not None else ""
        op_args = [operation_id] if operation_id is not None else []
        async with self._backend.acquire() as conn:
            # Rows under the limit: increment retry_count and reset to pending
            result = await conn.execute(
                f"""
                UPDATE {table}
                SET status = 'pending', worker_id = NULL, claimed_at = NULL,
                    retry_count = COALESCE(retry_count, 0) + 1, updated_at = now()
                WHERE status = 'processing' AND worker_id = $1
                  AND result_metadata->>'batch_id' IS NULL
                  AND COALESCE(retry_count, 0) < $2
                  {op_filter}
                """,
                self._worker_id,
                max_retries,
                *op_args,
            )
            # Rows that exceeded the limit: move to failed. RETURNING gives us
            # the ids so their parent aggregators can be rolled up below — a
            # batch_retain child sub-batch carries parent_operation_id (not
            # batch_id) in its metadata, so it IS eligible to be failed here,
            # and without propagating that terminal state the parent is stranded
            # in 'processing' forever (the same crash loop this fixes, one level up).
            failed_rows = await conn.fetch(
                f"""
                UPDATE {table}
                SET status = 'failed', worker_id = NULL, claimed_at = NULL,
                    error_message = 'exceeded max recovery attempts (retry_count >= {max_retries})',
                    completed_at = now(), updated_at = now()
                WHERE status = 'processing' AND worker_id = $1
                  AND result_metadata->>'batch_id' IS NULL
                  AND COALESCE(retry_count, 0) >= $2
                  {op_filter}
                RETURNING operation_id
                """,
                self._worker_id,
                max_retries,
                *op_args,
            )

        # Roll each failed child up to its parent aggregator, one transaction
        # per child so a single problematic parent can't undo the others —
        # mirrors the per-task transaction the in-process _mark_failed path uses.
        # The failing UPDATE above already committed, so the children stay failed
        # regardless; _maybe_update_parent_operation no-ops for tasks without a
        # parent_operation_id.
        for failed_row in failed_rows:
            async with self._backend.acquire() as conn:
                async with conn.transaction():
                    await self._maybe_update_parent_operation(str(failed_row["operation_id"]), schema, conn)

        if failed_rows:
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.warning(
                f"Worker {self._worker_id} moved {len(failed_rows)} tasks to 'failed' "
                f"(exceeded {max_retries} recovery attempts in schema {schema_display})"
            )
        return _updated_row_count(result)

    async def recover_own_tasks(self) -> int:
        """
        Recover tasks that were assigned to this worker but not completed.

        This handles the case where a worker crashes while processing tasks.
        On startup, we reset any tasks stuck in 'processing' for this worker_id
        back to 'pending' so they can be picked up again — provided their
        ``retry_count`` has not reached ``max_retries``. Tasks at/over the
        limit are moved to 'failed' with an explanatory error message,
        breaking the infinite loop where a task that kills the worker
        (OOM, infinite loop) is re-claimed forever.

        Also recovers batch API operations that were in-flight.

        If tenant_extension is configured, recovers across all tenant schemas.

        Returns:
            Number of tasks recovered (reset to pending, not including failed)
        """
        schemas = await self._get_schemas()
        total_count = 0

        for schema in schemas:
            try:
                # First, recover batch API operations (before resetting worker tasks)
                total_count += await self._recover_batch_operations(schema)

                total_count += await self._reclaim_own_processing_tasks(schema)

                # Finalize batch_retain parents that the aggregation left behind
                # (crash between a child's terminal commit and the parent update,
                # or children that never committed). These sit 'pending' with a
                # NULL payload — unclaimable and invisible to failed_operations —
                # until reconciled. See issue #2985.
                await self._reconcile_orphaned_parents(schema)
            except Exception as e:
                # Format schema for logging: custom schemas in quotes, None as-is
                schema_display = f'"{schema}"' if schema else str(schema)
                logger.warning(f"Worker {self._worker_id} failed to recover tasks for schema {schema_display}: {e}")

        if total_count > 0:
            logger.info(f"Worker {self._worker_id} recovered {total_count} stale tasks from previous run")
        return total_count

    async def release_own_tasks(self) -> int:
        """Hand back every operation this worker still owns, at shutdown.

        The counterpart to `recover_own_tasks`: the same reconciliation, run when
        the worker stops rather than when it starts. Without it, work cancelled
        past the drain timeout stays 'processing' under a worker id that is never
        coming back (the default id is derived from the hostname, so in a
        container it never recurs) and no client polling that operation ever sees
        it finish. See issue #3228.

        Deliberately skips the batch-operation and orphaned-parent passes:
        those are not scoped to this worker's rows, so they stay a startup
        concern where no other worker can be mid-flight on them.

        Returns:
            Number of operations returned to pending.
        """
        # Nothing here may raise: this runs on the shutdown path, where the DB
        # being unwell is exactly the case that strands rows, and an exception
        # escaping would abort the rest of the teardown.
        try:
            schemas = await self._get_schemas()
        except Exception as e:
            logger.warning(f"Worker {self._worker_id} could not list schemas to release its in-flight tasks: {e}")
            return 0

        total_count = 0
        for schema in schemas:
            try:
                total_count += await self._reclaim_own_processing_tasks(schema)
            except Exception as e:
                schema_display = f'"{schema}"' if schema else str(schema)
                logger.warning(f"Worker {self._worker_id} failed to release tasks for schema {schema_display}: {e}")
        return total_count

    async def _recover_batch_operations(
        self,
        schema: str | None,
        *,
        stale_before: datetime | None = None,
    ) -> int:
        """
        Recover batch API operations that were in-flight when worker crashed.

        Finds operations with batch_id in metadata and re-submits them as tasks
        so polling can resume. Startup recovery passes no cutoff and preserves
        the historical behavior of recovering all in-flight batch operations;
        lease maintenance supplies ``stale_before`` so a healthy worker's batch
        is left alone.

        Args:
            schema: Database schema to recover from

        Returns:
            Number of batch operations recovered
        """
        table = fq_table("async_operations", schema)

        try:
            stale_clause = ""
            fetch_args: list[Any] = []
            if stale_before is not None:
                stale_clause = "AND COALESCE(updated_at, claimed_at, created_at) < $1"
                fetch_args.append(stale_before)

            async with self._backend.acquire() as conn:
                # Find operations with batch_id in metadata (batch API operations)
                rows = await conn.fetch(
                    f"""
                    SELECT operation_id, task_payload, result_metadata
                    FROM {table}
                    WHERE status = 'processing'
                      AND result_metadata ? 'batch_id'
                      AND task_payload IS NOT NULL
                      {stale_clause}
                    """,
                    *fetch_args,
                )

            if not rows:
                return 0

            recovered = 0
            for row in rows:
                operation_id = str(row["operation_id"])
                result_metadata = row["result_metadata"]

                # Parse metadata
                if isinstance(result_metadata, str):
                    result_metadata = json.loads(result_metadata)

                batch_id = result_metadata.get("batch_id")
                batch_provider = result_metadata.get("batch_provider", "openai")

                logger.info(
                    f"Recovering batch operation: operation_id={operation_id}, batch_id={batch_id}, provider={batch_provider}"
                )

                # Mark operation as ready for re-processing
                # Reset to pending with task_payload intact so worker picks it up again
                update_stale_clause = ""
                update_args: list[Any] = [operation_id]
                if stale_before is not None:
                    update_stale_clause = "AND COALESCE(updated_at, claimed_at, created_at) < $2"
                    update_args.append(stale_before)

                # Re-check status and lease age in the UPDATE. A live worker may
                # have heartbeated or completed the row since the SELECT above.
                async with self._backend.acquire() as conn:
                    result = await conn.execute(
                        f"""
                        UPDATE {table}
                        SET status = 'pending', worker_id = NULL, claimed_at = NULL,
                            next_retry_at = NULL, updated_at = now()
                        WHERE operation_id = $1 AND status = 'processing'
                          {update_stale_clause}
                        """,
                        *update_args,
                    )

                if not _updated_row_count(result):
                    continue
                recovered += 1
                logger.info(f"Batch operation {operation_id} reset to pending for re-processing")

            return recovered

        except Exception as e:
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.error(f"Failed to recover batch operations for schema {schema_display}: {e}")
            return 0

    async def _reconcile_orphaned_parents(self, schema: str | None) -> int:
        """Drive stranded batch_retain parent operations to a terminal state.

        A batch_retain parent is a status *aggregator*: it carries no
        task_payload (workers never claim it) and is normally promoted to a
        terminal state by ``_maybe_update_parent_operation`` when its last child
        sub-batch finishes. Two crash windows can strand a parent 'pending'
        forever:

          * every child reached a terminal state but the promotion was skipped
            (the aggregation swallows and logs errors rather than failing the
            child, so a transient error there leaves the parent behind), or
          * the children never committed at all, leaving a parent with zero
            children (older non-atomic create paths, a hard kill mid-submission).

        Either way the parent sits 'pending' with ``task_payload IS NULL``: it is
        unclaimable, never counted in ``failed_operations``, unretryable via the
        API, and its documents are silently absent. See issue #2985.

        On worker startup we reconcile every such parent:

          * children present, all terminal -> completed / failed (mirrors the
            aggregator, inheriting a representative child error on failure),
          * no children at all -> failed, with an explicit reason so an operator
            can see the loss and resubmit the source documents.

        Parents with at least one still-live child are left untouched — the
        normal aggregation path will finish them once their children drain.

        Returns the number of parents driven to a terminal state.
        """
        table = fq_table("async_operations", schema)
        schema_display = f'"{schema}"' if schema else str(schema)
        reconciled = 0
        try:
            async with self._backend.acquire() as conn:
                parents = await conn.fetch(
                    f"""
                    SELECT operation_id, bank_id FROM {table}
                    WHERE operation_type = 'batch_retain'
                      AND status = 'pending'
                      AND task_payload IS NULL
                    """
                )

            for parent in parents:
                parent_id = parent["operation_id"]
                bank_id = parent["bank_id"]
                # One transaction per parent so a single problematic row can't
                # roll back the others (mirrors the per-child rollup above).
                async with self._backend.acquire() as conn:
                    async with conn.transaction():
                        # Re-read under a row lock: skip if the normal aggregation
                        # path (or another worker) already moved it off 'pending'.
                        locked = await conn.fetchrow(
                            f"SELECT status FROM {table} WHERE operation_id = $1 FOR UPDATE",
                            parent_id,
                        )
                        if locked is None or locked["status"] != "pending":
                            continue

                        siblings = await conn.fetch(
                            f"""
                            SELECT status, error_message FROM {table}
                            WHERE bank_id = $1
                              AND result_metadata::jsonb @> $2::jsonb
                            """,
                            bank_id,
                            json.dumps({"parent_operation_id": str(parent_id)}),
                        )

                        # A still-live child will drive the aggregation itself.
                        if any(s["status"] not in ("completed", "failed") for s in siblings):
                            continue

                        if not siblings:
                            await conn.execute(
                                f"""
                                UPDATE {table}
                                SET status = 'failed', error_message = $2,
                                    completed_at = now(), updated_at = now()
                                WHERE operation_id = $1
                                """,
                                parent_id,
                                "orphaned batch_retain parent: no child sub-batches were "
                                "persisted (worker crashed mid-submission); resubmit the "
                                "source documents",
                            )
                        elif any(s["status"] == "failed" for s in siblings):
                            await conn.execute(
                                f"""
                                UPDATE {table}
                                SET status = 'failed', error_message = $2, updated_at = now()
                                WHERE operation_id = $1
                                """,
                                parent_id,
                                _summarise_child_error_messages(siblings),
                            )
                        else:
                            await conn.execute(
                                f"""
                                UPDATE {table}
                                SET status = 'completed', completed_at = now(), updated_at = now()
                                WHERE operation_id = $1
                                """,
                                parent_id,
                            )
                        reconciled += 1

            if reconciled:
                logger.warning(
                    f"Worker {self._worker_id} reconciled {reconciled} stranded "
                    f"batch_retain parent(s) in schema {schema_display}"
                )
        except Exception as e:
            logger.warning(
                f"Worker {self._worker_id} failed to reconcile orphaned parents in schema {schema_display}: {e}"
            )
        return reconciled

    async def run(self):
        """
        Main polling loop with fire-and-forget task execution.

        Continuously polls for pending tasks, spawns them as background tasks,
        and immediately continues polling (up to slot limits).
        """
        await self.recover_own_tasks()
        # The worker id may be hostname-derived and therefore different after a
        # container replacement. Recover leases abandoned by any previous id
        # before beginning normal claims.
        await self.recover_stale_tasks()

        reservations_str = (
            ", ".join(f"{k}={v}" for k, v in self._slot_reservations.items()) if self._slot_reservations else "none"
        )
        shared_pool = max(0, self._max_slots - sum(self._slot_reservations.values()))
        logger.info(
            f"Worker {self._worker_id} starting polling loop "
            f"(max_slots={self._max_slots}, reservations=[{reservations_str}], shared_pool={shared_pool}, "
            f"heartbeat={self._heartbeat_interval_seconds:g}s, "
            f"stale_timeout={self._stale_task_timeout_seconds:g}s)"
        )

        self._lease_maintenance_task = asyncio.create_task(self._lease_maintenance_loop())
        try:
            while not self._shutdown.is_set():
                try:
                    # Claim a batch of tasks (respecting slot limits)
                    tasks = await self.claim_batch()
                    self._last_poll_at = time.monotonic()

                    if tasks:
                        # Log batch info
                        task_types: dict[str, int] = {}
                        schemas_seen: set[str | None] = set()
                        consolidation_count = 0
                        for task in tasks:
                            t = task.task_dict.get("type", "unknown")
                            op_type = task.task_dict.get("operation_type", "unknown")
                            task_types[t] = task_types.get(t, 0) + 1
                            schemas_seen.add(task.schema)
                            if op_type == "consolidation":
                                consolidation_count += 1

                        types_str = ", ".join(f"{k}:{v}" for k, v in task_types.items())
                        # Display None as "default" in logs
                        schemas_str = ", ".join(s if s else "default" for s in schemas_seen)
                        logger.info(
                            f"Worker {self._worker_id} claimed {len(tasks)} tasks "
                            f"({consolidation_count} consolidation): {types_str} (schemas: {schemas_str})"
                        )

                        # Spawn tasks as background jobs (fire-and-forget)
                        for task in tasks:
                            await self.execute_task(task)

                    if tasks:
                        # Continue immediately to claim more tasks (if slots available)
                        continue

                    # No tasks claimed (either no pending tasks or slots full)
                    # Wait before polling again
                    try:
                        await asyncio.wait_for(
                            self._shutdown.wait(),
                            timeout=self._poll_interval_ms / 1000,
                        )
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue polling

                    # Log progress stats periodically
                    await self._log_progress_if_due()

                except asyncio.CancelledError:
                    logger.info(f"Worker {self._worker_id} polling loop cancelled")
                    break
                except Exception as e:
                    logger.error(
                        f"Worker {self._worker_id} error in polling loop: {format_task_error(e)}", exc_info=True
                    )
                    # Backoff on error
                    await asyncio.sleep(1)
        finally:
            self._shutdown.set()
            await self._stop_lease_maintenance()
            logger.info(f"Worker {self._worker_id} polling loop stopped")

    async def shutdown_graceful(self, timeout: float = 30.0):
        """
        Signal shutdown and wait for current tasks to complete.

        Whatever is still claimed by this worker once the drain is over — work
        cancelled past the timeout, or a row stranded earlier by a terminal
        write that never landed — is handed back to 'pending' before returning.

        Args:
            timeout: Maximum time to wait for in-flight tasks (seconds)
        """
        logger.info(f"Worker {self._worker_id} initiating graceful shutdown")
        self._shutdown.set()
        await self._stop_lease_maintenance()

        # Wait for in-flight tasks to complete
        drained = False
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                active_task_objects = [info.bg_task for info in self._active_tasks.values()]

            if in_flight == 0:
                logger.info(f"Worker {self._worker_id} graceful shutdown complete")
                drained = True
                break

            logger.info(f"Worker {self._worker_id} waiting for {in_flight} in-flight tasks")

            # Wait for at least one task to complete
            if active_task_objects:
                done, _ = await asyncio.wait(active_task_objects, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.5)

        if not drained:
            logger.warning(f"Worker {self._worker_id} shutdown timeout after {timeout}s, cancelling remaining tasks")

            # Cancel remaining tasks
            cancelled: list[asyncio.Task] = []
            async with self._in_flight_lock:
                for operation_id, info in list(self._active_tasks.items()):
                    if not info.bg_task.done():
                        info.bg_task.cancel()
                        cancelled.append(info.bg_task)

            # Let the cancellations land before reconciling. asyncio.CancelledError
            # derives from BaseException, so it escapes every `except Exception` in
            # _execute_task_inner and no terminal state is ever written — but a task
            # partway through its own terminal write must be allowed to finish it,
            # or we would hand back a row it is about to complete.
            if cancelled:
                await asyncio.wait(cancelled, timeout=_CANCEL_DRAIN_TIMEOUT)

        # Anything still 'processing' under this worker id is work nobody is
        # running. Hand it back now instead of waiting for a startup recovery
        # that a hostname-derived worker id will never see again (issue #3228).
        released = await self.release_own_tasks()
        if released:
            logger.warning(f"Worker {self._worker_id} returned {released} in-flight operations to 'pending'")

    async def _log_progress_if_due(self):
        """Log progress stats every PROGRESS_LOG_INTERVAL seconds.

        Emits four kinds of lines:
          * [WORKER_STATS]  - aggregate slots / pool / global pending counts
          * [WORKER_TASK]   - one line per in-flight task with age + stage
          * [STUCK_STACK]   - async stack trace for tasks past stuck thresholds
          * [DB_WAITS]      - any non-idle hindsight session waiting on a lock
        """
        now = time.time()
        if now - self._last_progress_log < PROGRESS_LOG_INTERVAL:
            return

        self._last_progress_log = now

        try:
            # Get local active tasks
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                in_flight_by_type = dict(self._in_flight_by_type)
                active_tasks = dict(self._active_tasks)

            # Compute per-type reserved availability and shared pool
            tasks_in_reserved = 0
            reserved_parts = []
            for op_type, reserved in self._slot_reservations.items():
                type_in_flight = in_flight_by_type.get(op_type, 0)
                type_available = max(0, reserved - type_in_flight)
                tasks_in_reserved += min(reserved, type_in_flight)
                reserved_parts.append(f"{op_type}={type_in_flight}/{reserved}(avail={type_available})")
            sum_reservations = sum(self._slot_reservations.values())
            shared_pool_size = max(0, self._max_slots - sum_reservations)
            tasks_in_shared = max(0, in_flight - tasks_in_reserved)
            shared_available = max(0, shared_pool_size - tasks_in_shared)
            reserved_str = ", ".join(reserved_parts) if reserved_parts else "none"

            # Build local processing breakdown (aggregate counts)
            task_groups: dict[tuple[str, str], int] = {}
            for info in active_tasks.values():
                key = (info.op_type, info.bank_id)
                task_groups[key] = task_groups.get(key, 0) + 1

            processing_info = [f"{op}:{bank}({cnt})" for (op, bank), cnt in task_groups.items()]
            processing_str = ", ".join(processing_info[:10]) if processing_info else "none"
            if len(processing_info) > 10:
                processing_str += f" +{len(processing_info) - 10} more"

            # Get global stats from DB — scope the heavy COUNT/GROUP BY
            # queries to schemas that actually have work. With N tenants the
            # full fanout is 2*N queries every PROGRESS_LOG_INTERVAL; scoping
            # via the routine (or per-schema EXISTS fallback) reduces this to
            # 2*active_schemas which is typically << N.
            schemas = await self._get_schemas()
            total_schema_count = len(schemas)

            global_pending = 0
            all_worker_counts: dict[str, int] = {}
            # operation_type -> aggregated bucket counts across schemas
            pending_breakdown: dict[str, dict[str, int]] = {}
            schemas_to_query: set[str | None] = set()

            async with self._backend.acquire() as conn:
                # Schemas with pending async_operations (uses server-side
                # routine when installed, falls back to per-schema EXISTS).
                schemas_with_pending = await self._scan_active_schemas(conn, schemas)

                # Also include schemas that have in-flight tasks on this worker
                # so the "processing" worker_id GROUP BY still reports correctly.
                schemas_with_active_tasks = {info.schema for info in active_tasks.values()}
                schemas_to_query = schemas_with_pending | schemas_with_active_tasks

                for schema in schemas_to_query:
                    table = fq_table("async_operations", schema)

                    # Bucket pending rows by the same predicates the claim query
                    # filters on, so an operator can see why pending > 0 but
                    # nothing is being claimed (orphaned batch_retain parents,
                    # retry backoff, etc.).
                    # Use SUM(CASE WHEN ...) instead of COUNT(*) FILTER (WHERE ...)
                    # for Oracle compatibility — FILTER is PG-specific.
                    try:
                        breakdown_rows = await conn.fetch(
                            f"""
                            SELECT
                                operation_type,
                                COUNT(*) AS total,
                                SUM(CASE WHEN task_payload IS NULL THEN 1 ELSE 0 END) AS payload_null,
                                SUM(CASE WHEN next_retry_at IS NOT NULL AND next_retry_at > now()
                                    THEN 1 ELSE 0 END) AS retry_blocked,
                                SUM(CASE WHEN worker_id IS NOT NULL THEN 1 ELSE 0 END) AS assigned
                            FROM {table}
                            WHERE status = 'pending'
                            GROUP BY operation_type
                            """
                        )
                    except Exception:
                        # Schema may be partially provisioned (table missing).
                        breakdown_rows = []
                    for br in breakdown_rows:
                        op_type = br["operation_type"] or "unknown"
                        bucket = pending_breakdown.setdefault(
                            op_type, {"total": 0, "payload_null": 0, "retry_blocked": 0, "assigned": 0}
                        )
                        bucket["total"] += br["total"]
                        bucket["payload_null"] += br["payload_null"]
                        bucket["retry_blocked"] += br["retry_blocked"]
                        bucket["assigned"] += br["assigned"]
                        global_pending += br["total"]

                    try:
                        worker_rows = await conn.fetch(
                            f"""
                            SELECT worker_id, COUNT(*) as count
                            FROM {table}
                            WHERE status = 'processing'
                            GROUP BY worker_id
                            """
                        )
                    except Exception:
                        worker_rows = []
                    for wr in worker_rows:
                        wid = wr["worker_id"] or "unknown"
                        all_worker_counts[wid] = all_worker_counts.get(wid, 0) + wr["count"]

            other_workers = []
            for wid, cnt in all_worker_counts.items():
                if wid != self._worker_id:
                    other_workers.append(f"{wid}:{cnt}")
            others_str = ", ".join(other_workers) if other_workers else "none"

            # asyncpg pool stats - exhaustion presents as "everything slow",
            # making it invisible without this line.
            pool_str = self._format_pool_stats()
            proc_str = self._format_proc_stats()

            queried_count = len(schemas_to_query)
            # Display queried schemas (cap at 20 for readability)
            queried_list = sorted(s if s else "default" for s in schemas_to_query)
            schemas_str = ", ".join(queried_list[:20])
            if len(queried_list) > 20:
                schemas_str += f" +{len(queried_list) - 20} more"
            logger.info(
                f"[WORKER_STATS] worker={self._worker_id} "
                f"slots={in_flight}/{self._max_slots} | "
                f"reserved: [{reserved_str}] | "
                f"shared={tasks_in_shared}/{shared_pool_size}(avail={shared_available}) | "
                f"global: pending={global_pending} "
                f"(queried={queried_count}/{total_schema_count} schemas: {schemas_str}) | "
                f"others: {others_str} | "
                f"pool: {pool_str} | "
                f"proc: {proc_str} | "
                f"my_active: {processing_str}"
            )

            # Pending breakdown - explains why pending rows aren't being claimed
            # (orphaned batch_retain parents have payload_null > 0, retry storms
            # show up as retry_blocked, etc.). Skip when nothing is pending so
            # the line doesn't add noise on idle deployments.
            if global_pending > 0:
                self._log_pending_breakdown(pending_breakdown)

            # Per-task lines, sorted oldest-first so stuck tasks bubble to the top.
            self._log_per_task_lines(active_tasks, now=time.monotonic())

            # DB lock waits - separate from per-task lines because a single
            # blocking session can wedge many tasks.
            await self._log_db_waits()

        except Exception as e:
            logger.debug(f"Failed to log progress stats: {e}")

    def _format_proc_stats(self) -> str:
        """Render lightweight process memory stats. Returns 'unavailable' if introspection fails.

        ``rss_mb`` is the process's *current* resident set. ``peak_rss_mb`` is the
        high-water mark since start, which only ever climbs.

        Keeping them apart matters: this line used to report ``ru_maxrss`` — the
        peak — under the bare name ``rss_mb``, so a transient allocation left the
        field pinned at its peak for the life of the process and every later
        reading looked like retained memory that had never been freed. That
        misread a reranker burst as a leak while diagnosing issue #3355.
        """
        try:
            import resource
            import sys

            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak = usage.ru_maxrss
            # ru_maxrss is bytes on macOS, kilobytes on Linux.
            if sys.platform != "darwin":
                peak *= 1024
            peak_mb = peak / (1024 * 1024)

            current = _current_rss_bytes()
            if current is None:
                # No cheap current-RSS source (non-Linux). Report only what we
                # actually have rather than passing the peak off as current.
                return f"peak_rss_mb={peak_mb:.0f}"
            return f"rss_mb={current / (1024 * 1024):.0f} peak_rss_mb={peak_mb:.0f}"
        except Exception as e:
            logger.debug(f"Process stats unavailable: {e}")
            return "unavailable"

    def _format_pool_stats(self) -> str:
        """Render connection pool stats. Returns 'unavailable' if pool can't be introspected."""
        try:
            pool = self._backend.get_pool()
            # asyncpg.Pool exposes _holders / _queue internally; fall back gracefully
            # to public methods if the layout ever changes.
            size = pool.get_size() if hasattr(pool, "get_size") else len(getattr(pool, "_holders", []))
            free = pool.get_idle_size() if hasattr(pool, "get_idle_size") else None
            min_size = pool.get_min_size() if hasattr(pool, "get_min_size") else None
            max_size = pool.get_max_size() if hasattr(pool, "get_max_size") else None
            queue = getattr(pool, "_queue", None)
            # asyncpg's _queue is a LifoQueue pre-filled to max_size with
            # PoolConnectionHolder objects. qsize() therefore counts *available
            # holders*, not callers waiting on the pool — the previous "waiters"
            # label here was the opposite of what it suggested. The actual count
            # of awaiters is len(_queue._getters), nonzero only when qsize()==0.
            free_holders = queue.qsize() if queue is not None and hasattr(queue, "qsize") else None
            getters = getattr(queue, "_getters", None) if queue is not None else None
            pending_acquires = len(getters) if getters is not None else None

            parts = [f"size={size}"]
            if min_size is not None and max_size is not None:
                parts.append(f"limits={min_size}-{max_size}")
            if free is not None:
                parts.append(f"idle={free}")
                parts.append(f"in_use={size - free}")
            if free_holders is not None:
                parts.append(f"free_holders={free_holders}")
            if pending_acquires is not None:
                parts.append(f"pending_acquires={pending_acquires}")
            return " ".join(parts)
        except Exception as e:
            logger.debug(f"Pool stats unavailable: {e}")
            return "unavailable"

    def _log_pending_breakdown(self, breakdown: dict[str, dict[str, int]]) -> None:
        """Emit one [PENDING_BREAKDOWN] line bucketing pending rows by claimability.

        Each bucket mirrors a predicate in the claim query:
          * payload_null   - row has no task_payload (e.g. batch_retain parent
                             whose reconciliation never fired); claim query
                             skips it forever
          * retry_blocked  - next_retry_at is still in the future
          * assigned       - worker_id already set; another worker owns it

        ``claimable`` is the residual that *should* be picked up on the next
        poll. If ``claimable > 0`` while workers report free slots, the bug is
        somewhere else (lock contention, tenant discovery, etc.) - this line
        narrows the search.
        """
        if not breakdown:
            return

        parts = []
        for op_type in sorted(breakdown):
            b = breakdown[op_type]
            claimable = b["total"] - b["payload_null"] - b["retry_blocked"] - b["assigned"]
            parts.append(
                f"{op_type}: total={b['total']} claimable={claimable} "
                f"payload_null={b['payload_null']} retry_blocked={b['retry_blocked']} "
                f"assigned={b['assigned']}"
            )
        logger.info(f"[PENDING_BREAKDOWN] {' | '.join(parts)}")

    def _log_per_task_lines(self, active_tasks: dict[str, ActiveTaskInfo], now: float) -> None:
        """Emit one [WORKER_TASK] line per in-flight task and dump stuck stacks.

        Sorted by age desc so the oldest (most likely stuck) tasks appear first.
        """
        if not active_tasks:
            return

        # Sort by age descending; tie-break on op_id for determinism.
        ordered = sorted(
            active_tasks.items(),
            key=lambda kv: (now - kv[1].started_at, kv[0]),
            reverse=True,
        )

        for op_id, info in ordered:
            age_s = now - info.started_at
            holder = info.stage_holder
            stage = holder.stage if holder is not None else "unknown"
            stage_age_s = (now - holder.updated_at) if holder is not None else 0.0
            stuck_marker = "[STUCK?] " if age_s >= STUCK_STACK_INITIAL_THRESHOLD_S else ""
            schema_part = f" schema={info.schema}" if info.schema else ""
            logger.info(
                f"[WORKER_TASK] {stuck_marker}op={op_id} type={info.task_type} "
                f"op_type={info.op_type} bank={info.bank_id}{schema_part} "
                f"age={age_s:.0f}s stage={stage} stage_age={stage_age_s:.0f}s"
            )

            self._maybe_dump_stuck_stack(op_id, info, age_s)

    def _maybe_dump_stuck_stack(self, op_id: str, info: ActiveTaskInfo, age_s: float) -> None:
        """Dump a coroutine stack for tasks that crossed a stuck threshold.

        Each task gets one dump per threshold (5min, 10min, 20min, 40min...),
        gated by `info.last_stack_dump_threshold` so logs don't flood for tasks
        that legitimately take a long time (large LLM jobs, schema-retry loops).
        """
        if age_s < STUCK_STACK_INITIAL_THRESHOLD_S:
            return

        # Find the largest doubling-threshold that the task has crossed.
        threshold = STUCK_STACK_INITIAL_THRESHOLD_S
        crossed = STUCK_STACK_INITIAL_THRESHOLD_S
        while threshold <= age_s and threshold <= STUCK_STACK_MAX_THRESHOLD_S:
            crossed = threshold
            threshold *= 2

        if crossed <= info.last_stack_dump_threshold:
            return

        info.last_stack_dump_threshold = crossed

        try:
            buf = io.StringIO()
            info.bg_task.print_stack(file=buf, limit=15)
            stage = info.stage_holder.stage if info.stage_holder else "unknown"
            logger.warning(
                f"[STUCK_STACK] op={op_id} type={info.task_type} bank={info.bank_id} "
                f"age={age_s:.0f}s threshold={crossed}s stage={stage}\n{buf.getvalue()}"
            )
        except Exception as e:
            # Stack capture is best-effort - never crash the polling loop over it.
            logger.debug(f"Failed to capture stack for {op_id}: {e}")

    async def _log_db_waits(self) -> None:
        """Log any non-idle hindsight session that's waiting on a lock or other resource.

        Catches the case where a coroutine appears 'fine' from Python's perspective
        but is blocked on a Postgres row lock - which is exactly how the 3-phase
        retain pipeline deadlock would present.

        pg_stat_activity is PostgreSQL-specific; skip on other backends.
        """
        # pg_stat_activity is PG-specific — skip on non-PG backends.
        if self._backend.backend_type != "postgresql":
            return

        try:
            async with self._backend.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        pid,
                        application_name,
                        wait_event_type,
                        wait_event,
                        state,
                        EXTRACT(EPOCH FROM (now() - query_start))::int AS age_s,
                        LEFT(query, 200) AS query
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND state IS NOT NULL
                      AND state != 'idle'
                      AND wait_event IS NOT NULL
                      AND wait_event_type NOT IN ('Activity', 'Client')
                    ORDER BY age_s DESC NULLS LAST
                    LIMIT 20
                    """
                )
        except Exception as e:
            # pg_stat_activity may be restricted on managed Postgres - degrade silently.
            logger.debug(f"DB waits query failed: {e}")
            return

        if not rows:
            return

        for r in rows:
            logger.info(
                f"[DB_WAITS] pid={r['pid']} app={r['application_name']} "
                f"wait={r['wait_event_type']}.{r['wait_event']} state={r['state']} "
                f"age={r['age_s']}s query={r['query']!r}"
            )

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown.is_set()

    @property
    def seconds_since_last_poll(self) -> float | None:
        """Age of the last completed claim cycle, or None before the first one."""
        if self._last_poll_at is None:
            return None
        return round(time.monotonic() - self._last_poll_at, 1)
