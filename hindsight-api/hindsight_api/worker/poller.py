"""
Worker poller for distributed task execution.

Polls PostgreSQL for pending tasks and executes them using
FOR UPDATE SKIP LOCKED for safe concurrent claiming.
"""

import asyncio
import json
import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

    from hindsight_api.extensions.tenant import TenantExtension

logger = logging.getLogger(__name__)

# Progress logging interval in seconds
PROGRESS_LOG_INTERVAL = 30


def fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    if schema:
        return f'"{schema}".{table}'
    return table


@dataclass
class ClaimedTask:
    """A task claimed from the database with its schema context."""

    operation_id: str
    task_dict: dict[str, Any]
    schema: str | None


class WorkerPoller:
    """
    Polls PostgreSQL for pending tasks and executes them.

    Uses FOR UPDATE SKIP LOCKED for safe distributed claiming,
    allowing multiple workers to process tasks without conflicts.

    Supports dynamic multi-tenant discovery via tenant_extension.
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        worker_id: str,
        executor: Callable[[dict[str, Any]], Awaitable[None]],
        poll_interval_ms: int = 500,
        max_retries: int = 3,
        schema: str | None = None,
        tenant_extension: "TenantExtension | None" = None,
        max_slots: int = 10,
        consolidation_max_slots: int = 2,
    ):
        """
        Initialize the worker poller.

        Args:
            pool: asyncpg connection pool
            worker_id: Unique identifier for this worker
            executor: Async function to execute tasks (typically MemoryEngine.execute_task)
            poll_interval_ms: Interval between polls when no tasks found (milliseconds)
            max_retries: Maximum retry attempts before marking task as failed
            schema: Database schema for single-tenant support (deprecated, use tenant_extension)
            tenant_extension: Extension for dynamic multi-tenant discovery. If None, creates a
                            DefaultTenantExtension with the configured schema.
            max_slots: Maximum concurrent tasks per worker
            consolidation_max_slots: Maximum concurrent consolidation tasks per worker
        """
        self._pool = pool
        self._worker_id = worker_id
        self._executor = executor
        self._poll_interval_ms = poll_interval_ms
        self._max_retries = max_retries
        self._schema = schema
        # Always set tenant extension (use DefaultTenantExtension if none provided)
        if tenant_extension is None:
            from ..extensions.builtin.tenant import DefaultTenantExtension

            # Pass schema parameter to DefaultTenantExtension if explicitly provided
            config = {"schema": schema} if schema else {}
            tenant_extension = DefaultTenantExtension(config=config)
        self._tenant_extension = tenant_extension
        self._max_slots = max_slots
        self._consolidation_max_slots = consolidation_max_slots
        self._shutdown = asyncio.Event()
        self._current_tasks: set[asyncio.Task] = set()
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()
        self._last_progress_log = 0.0
        self._tasks_completed_since_log = 0
        # Track active tasks locally: operation_id -> (op_type, bank_id, schema, asyncio.Task)
        self._active_tasks: dict[str, tuple[str, str, str | None, asyncio.Task]] = {}
        # Track in-flight tasks by operation type
        self._in_flight_by_type: dict[str, int] = {}

    async def _get_schemas(self) -> list[str | None]:
        """Get list of schemas to poll. Returns [None] for default schema (no prefix)."""
        from ..config import DEFAULT_DATABASE_SCHEMA

        tenants = await self._tenant_extension.list_tenants()
        # Convert default schema to None for SQL compatibility (no prefix), keep others as-is
        return [t.schema if t.schema != DEFAULT_DATABASE_SCHEMA else None for t in tenants]

    async def _get_available_slots(self) -> tuple[int, int]:
        """
        Calculate available slots for claiming tasks.

        Returns:
            (total_available, consolidation_available) tuple
        """
        async with self._in_flight_lock:
            total_in_flight = self._in_flight_count
            consolidation_in_flight = self._in_flight_by_type.get("consolidation", 0)

        total_available = max(0, self._max_slots - total_in_flight)
        consolidation_available = max(0, self._consolidation_max_slots - consolidation_in_flight)

        return total_available, consolidation_available

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
        respecting slot limits (total and consolidation).

        Uses FOR UPDATE SKIP LOCKED to ensure no conflicts with other workers.

        Returns:
            List of ClaimedTask objects containing operation_id, task_dict, and schema
        """
        # Calculate available slots
        total_available, consolidation_available = await self._get_available_slots()

        if total_available <= 0:
            return []

        schemas = await self._get_schemas()
        all_tasks: list[ClaimedTask] = []
        remaining_total = total_available
        remaining_consolidation = consolidation_available

        for schema in schemas:
            if remaining_total <= 0:
                break

            tasks = await self._claim_batch_for_schema(schema, remaining_total, remaining_consolidation)

            # Update remaining slots based on what was claimed
            for task in tasks:
                op_type = task.task_dict.get("operation_type", "unknown")
                if op_type == "consolidation":
                    remaining_consolidation -= 1

            all_tasks.extend(tasks)
            remaining_total -= len(tasks)

        return all_tasks

    async def _claim_batch_for_schema(
        self, schema: str | None, limit: int, consolidation_limit: int
    ) -> list[ClaimedTask]:
        """Claim tasks from a specific schema respecting slot limits."""
        try:
            return await self._claim_batch_for_schema_inner(schema, limit, consolidation_limit)
        except Exception as e:
            # Format schema for logging: custom schemas in quotes, None as-is
            schema_display = f'"{schema}"' if schema else str(schema)
            logger.warning(f"Worker {self._worker_id} failed to claim tasks for schema {schema_display}: {e}")
            return []

    async def _claim_batch_for_schema_inner(
        self, schema: str | None, limit: int, consolidation_limit: int
    ) -> list[ClaimedTask]:
        """Inner implementation for claiming tasks from a specific schema with slot limits."""
        table = fq_table("async_operations", schema)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Strategy: Claim non-consolidation tasks first, then consolidation up to limit

                # 1. Claim non-consolidation tasks (up to limit)
                non_consolidation_rows = await conn.fetch(
                    f"""
                    SELECT operation_id, task_payload
                    FROM {table}
                    WHERE status = 'pending'
                      AND task_payload IS NOT NULL
                      AND operation_type != 'consolidation'
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    limit,
                )

                claimed_count = len(non_consolidation_rows)
                remaining_limit = limit - claimed_count

                # 2. Claim consolidation tasks (up to consolidation_limit and remaining_limit)
                consolidation_rows = []
                if consolidation_limit > 0 and remaining_limit > 0:
                    consolidation_rows = await conn.fetch(
                        f"""
                        SELECT operation_id, task_payload
                        FROM {table} AS pending
                        WHERE status = 'pending'
                          AND task_payload IS NOT NULL
                          AND operation_type = 'consolidation'
                          AND NOT EXISTS (
                              SELECT 1 FROM {table} AS processing
                              WHERE processing.bank_id = pending.bank_id
                                AND processing.operation_type = 'consolidation'
                                AND processing.status = 'processing'
                          )
                        ORDER BY created_at
                        LIMIT $1
                        FOR UPDATE SKIP LOCKED
                        """,
                        min(consolidation_limit, remaining_limit),
                    )

                all_rows = non_consolidation_rows + consolidation_rows

                if not all_rows:
                    return []

                # Claim the tasks by updating status and worker_id
                operation_ids = [row["operation_id"] for row in all_rows]
                await conn.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'processing', worker_id = $1, claimed_at = now(), updated_at = now()
                    WHERE operation_id = ANY($2)
                    """,
                    self._worker_id,
                    operation_ids,
                )

                # Parse and return task payloads with schema context
                return [
                    ClaimedTask(
                        operation_id=str(row["operation_id"]),
                        task_dict=json.loads(row["task_payload"]),
                        schema=schema,
                    )
                    for row in all_rows
                ]

    async def _mark_completed(self, operation_id: str, schema: str | None):
        """Mark a task as completed."""
        table = fq_table("async_operations", schema)
        await self._pool.execute(
            f"""
            UPDATE {table}
            SET status = 'completed', completed_at = now(), updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
        )

    async def _mark_failed(self, operation_id: str, error_message: str, schema: str | None):
        """Mark a task as failed with error message."""
        table = fq_table("async_operations", schema)
        # Truncate error message if too long (max 5000 chars in schema)
        error_message = error_message[:5000] if len(error_message) > 5000 else error_message
        await self._pool.execute(
            f"""
            UPDATE {table}
            SET status = 'failed', error_message = $2, completed_at = now(), updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
            error_message,
        )

    async def _retry_or_fail(self, operation_id: str, error_message: str, schema: str | None):
        """Increment retry count or mark as failed if max retries exceeded."""
        table = fq_table("async_operations", schema)

        # Get current retry count
        row = await self._pool.fetchrow(
            f"SELECT retry_count FROM {table} WHERE operation_id = $1",
            operation_id,
        )

        if row is None:
            logger.warning(f"Operation {operation_id} not found, cannot retry")
            return

        retry_count = row["retry_count"]

        if retry_count >= self._max_retries:
            # Max retries exceeded, mark as failed
            await self._mark_failed(
                operation_id, f"Max retries ({self._max_retries}) exceeded. Last error: {error_message}", schema
            )
            logger.error(f"Task {operation_id} failed after {retry_count} retries")
        else:
            # Increment retry and reset to pending
            await self._pool.execute(
                f"""
                UPDATE {table}
                SET status = 'pending', worker_id = NULL, claimed_at = NULL,
                    retry_count = retry_count + 1, updated_at = now()
                WHERE operation_id = $1
                """,
                operation_id,
            )
            logger.warning(f"Task {operation_id} failed, will retry (attempt {retry_count + 1}/{self._max_retries})")

    async def execute_task(self, task: ClaimedTask):
        """Execute a single task as a background job (fire-and-forget)."""
        task_type = task.task_dict.get("type", "unknown")
        operation_type = task.task_dict.get("operation_type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        # Create background task
        bg_task = asyncio.create_task(self._execute_task_inner(task))

        # Track this task as active
        async with self._in_flight_lock:
            self._active_tasks[task.operation_id] = (task_type, bank_id, task.schema, bg_task)
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

    async def _execute_task_inner(self, task: ClaimedTask):
        """Inner task execution with error handling."""
        task_type = task.task_dict.get("type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        try:
            schema_info = f", schema={task.schema}" if task.schema else ""
            logger.debug(f"Executing task {task.operation_id} (type={task_type}, bank={bank_id}{schema_info})")
            if task.schema:
                task.task_dict["_schema"] = task.schema
            await self._executor(task.task_dict)
            await self._mark_completed(task.operation_id, task.schema)
            logger.debug(f"Task {task.operation_id} completed successfully")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Task {task.operation_id} failed: {e}")
            await self._retry_or_fail(task.operation_id, error_msg, task.schema)

    async def recover_own_tasks(self) -> int:
        """
        Recover tasks that were assigned to this worker but not completed.

        This handles the case where a worker crashes while processing tasks.
        On startup, we reset any tasks stuck in 'processing' for this worker_id
        back to 'pending' so they can be picked up again.

        If tenant_extension is configured, recovers across all tenant schemas.

        Returns:
            Number of tasks recovered
        """
        schemas = await self._get_schemas()
        total_count = 0

        for schema in schemas:
            try:
                table = fq_table("async_operations", schema)

                result = await self._pool.execute(
                    f"""
                    UPDATE {table}
                    SET status = 'pending', worker_id = NULL, claimed_at = NULL, updated_at = now()
                    WHERE status = 'processing' AND worker_id = $1
                    """,
                    self._worker_id,
                )

                # Parse "UPDATE N" to get count
                count = int(result.split()[-1]) if result else 0
                total_count += count
            except Exception as e:
                # Format schema for logging: custom schemas in quotes, None as-is
                schema_display = f'"{schema}"' if schema else str(schema)
                logger.warning(f"Worker {self._worker_id} failed to recover tasks for schema {schema_display}: {e}")

        if total_count > 0:
            logger.info(f"Worker {self._worker_id} recovered {total_count} stale tasks from previous run")
        return total_count

    async def run(self):
        """
        Main polling loop with fire-and-forget task execution.

        Continuously polls for pending tasks, spawns them as background tasks,
        and immediately continues polling (up to slot limits).
        """
        await self.recover_own_tasks()

        logger.info(
            f"Worker {self._worker_id} starting polling loop "
            f"(max_slots={self._max_slots}, consolidation_max_slots={self._consolidation_max_slots})"
        )

        while not self._shutdown.is_set():
            try:
                # Claim a batch of tasks (respecting slot limits)
                tasks = await self.claim_batch()

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
                logger.error(f"Worker {self._worker_id} error in polling loop: {e}")
                traceback.print_exc()
                # Backoff on error
                await asyncio.sleep(1)

        logger.info(f"Worker {self._worker_id} polling loop stopped")

    async def shutdown_graceful(self, timeout: float = 30.0):
        """
        Signal shutdown and wait for current tasks to complete.

        Args:
            timeout: Maximum time to wait for in-flight tasks (seconds)
        """
        logger.info(f"Worker {self._worker_id} initiating graceful shutdown")
        self._shutdown.set()

        # Wait for in-flight tasks to complete
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                active_task_objects = [task_info[3] for task_info in self._active_tasks.values()]

            if in_flight == 0:
                logger.info(f"Worker {self._worker_id} graceful shutdown complete")
                return

            logger.info(f"Worker {self._worker_id} waiting for {in_flight} in-flight tasks")

            # Wait for at least one task to complete
            if active_task_objects:
                done, _ = await asyncio.wait(active_task_objects, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.5)

        logger.warning(f"Worker {self._worker_id} shutdown timeout after {timeout}s, cancelling remaining tasks")

        # Cancel remaining tasks
        async with self._in_flight_lock:
            for operation_id, (_, _, _, bg_task) in list(self._active_tasks.items()):
                if not bg_task.done():
                    bg_task.cancel()

    async def _log_progress_if_due(self):
        """Log progress stats every PROGRESS_LOG_INTERVAL seconds."""
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

            consolidation_count = in_flight_by_type.get("consolidation", 0)
            available_slots = self._max_slots - in_flight
            available_consolidation_slots = self._consolidation_max_slots - consolidation_count

            # Build local processing breakdown
            task_groups: dict[tuple[str, str], int] = {}
            for op_type, bank_id, _, _ in active_tasks.values():
                key = (op_type, bank_id)
                task_groups[key] = task_groups.get(key, 0) + 1

            processing_info = [f"{op}:{bank}({cnt})" for (op, bank), cnt in task_groups.items()]
            processing_str = ", ".join(processing_info[:10]) if processing_info else "none"
            if len(processing_info) > 10:
                processing_str += f" +{len(processing_info) - 10} more"

            # Get global stats from DB
            schemas = await self._get_schemas()
            global_pending = 0
            all_worker_counts: dict[str, int] = {}

            async with self._pool.acquire() as conn:
                for schema in schemas:
                    table = fq_table("async_operations", schema)

                    row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table} WHERE status = 'pending'")
                    global_pending += row["count"] if row else 0

                    worker_rows = await conn.fetch(
                        f"""
                        SELECT worker_id, COUNT(*) as count
                        FROM {table}
                        WHERE status = 'processing'
                        GROUP BY worker_id
                        """
                    )
                    for wr in worker_rows:
                        wid = wr["worker_id"] or "unknown"
                        all_worker_counts[wid] = all_worker_counts.get(wid, 0) + wr["count"]

            other_workers = []
            for wid, cnt in all_worker_counts.items():
                if wid != self._worker_id:
                    other_workers.append(f"{wid}:{cnt}")
            others_str = ", ".join(other_workers) if other_workers else "none"

            # Display None as "default" in logs
            schemas_str = ", ".join(s if s else "default" for s in schemas)
            logger.info(
                f"[WORKER_STATS] worker={self._worker_id} "
                f"slots={in_flight}/{self._max_slots} (consolidation={consolidation_count}/{self._consolidation_max_slots}) | "
                f"available={available_slots} (consolidation={available_consolidation_slots}) | "
                f"global: pending={global_pending} (schemas: {schemas_str}) | "
                f"others: {others_str} | "
                f"my_active: {processing_str}"
            )

        except Exception as e:
            logger.debug(f"Failed to log progress stats: {e}")

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown.is_set()
