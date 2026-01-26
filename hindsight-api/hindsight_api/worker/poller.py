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
        batch_size: int = 10,
        max_retries: int = 3,
        schema: str | None = None,
        tenant_extension: "TenantExtension | None" = None,
    ):
        """
        Initialize the worker poller.

        Args:
            pool: asyncpg connection pool
            worker_id: Unique identifier for this worker
            executor: Async function to execute tasks (typically MemoryEngine.execute_task)
            poll_interval_ms: Interval between polls when no tasks found (milliseconds)
            batch_size: Maximum number of tasks to claim per poll cycle
            max_retries: Maximum retry attempts before marking task as failed
            schema: Database schema for single-tenant support (ignored if tenant_extension is set)
            tenant_extension: Extension for dynamic multi-tenant discovery. If set, list_tenants()
                            is called on each poll cycle to discover schemas dynamically.
        """
        self._pool = pool
        self._worker_id = worker_id
        self._executor = executor
        self._poll_interval_ms = poll_interval_ms
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._schema = schema
        self._tenant_extension = tenant_extension
        self._shutdown = asyncio.Event()
        self._current_tasks: set[asyncio.Task] = set()
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()
        self._last_progress_log = 0.0
        self._tasks_completed_since_log = 0
        # Track active tasks locally: operation_id -> (op_type, bank_id, schema)
        self._active_tasks: dict[str, tuple[str, str, str | None]] = {}

    async def _get_schemas(self) -> list[str | None]:
        """Get list of schemas to poll. Returns [None] for public schema."""
        if self._tenant_extension is not None:
            tenants = await self._tenant_extension.list_tenants()
            # Convert "public" to None for SQL compatibility, keep others as-is
            return [t.schema if t.schema != "public" else None for t in tenants]
        # Single schema mode
        return [self._schema]

    async def claim_batch(self) -> list[ClaimedTask]:
        """
        Claim up to batch_size pending tasks atomically across all tenant schemas.

        Uses FOR UPDATE SKIP LOCKED to ensure no conflicts with other workers.

        For consolidation tasks specifically, skips pending tasks if there's already
        a processing consolidation for the same bank (to avoid duplicate work).

        If tenant_extension is configured, dynamically discovers schemas on each call.

        Returns:
            List of ClaimedTask objects containing operation_id, task_dict, and schema
        """
        schemas = await self._get_schemas()
        all_tasks: list[ClaimedTask] = []
        remaining_batch = self._batch_size

        for schema in schemas:
            if remaining_batch <= 0:
                break

            tasks = await self._claim_batch_for_schema(schema, remaining_batch)
            all_tasks.extend(tasks)
            remaining_batch -= len(tasks)

        return all_tasks

    async def _claim_batch_for_schema(self, schema: str | None, limit: int) -> list[ClaimedTask]:
        """Claim tasks from a specific schema."""
        table = fq_table("async_operations", schema)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Select and lock pending tasks
                # For consolidation: skip if same bank already has one processing
                rows = await conn.fetch(
                    f"""
                    SELECT operation_id, task_payload
                    FROM {table} AS pending
                    WHERE status = 'pending' AND task_payload IS NOT NULL
                    AND (
                        -- Non-consolidation tasks: always claimable
                        operation_type != 'consolidation'
                        OR
                        -- Consolidation: only if no other consolidation processing for same bank
                        NOT EXISTS (
                            SELECT 1 FROM {table} AS processing
                            WHERE processing.bank_id = pending.bank_id
                            AND processing.operation_type = 'consolidation'
                            AND processing.status = 'processing'
                        )
                    )
                    ORDER BY created_at
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    limit,
                )

                if not rows:
                    return []

                # Claim the tasks by updating status and worker_id
                operation_ids = [row["operation_id"] for row in rows]
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
                    for row in rows
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
        """Execute a single task and update its status."""
        task_type = task.task_dict.get("type", "unknown")
        bank_id = task.task_dict.get("bank_id", "unknown")

        # Track this task as active
        async with self._in_flight_lock:
            self._active_tasks[task.operation_id] = (task_type, bank_id, task.schema)

        try:
            schema_info = f", schema={task.schema}" if task.schema else ""
            logger.debug(f"Executing task {task.operation_id} (type={task_type}, bank={bank_id}{schema_info})")
            # Pass schema to executor so it can set the correct context
            if task.schema:
                task.task_dict["_schema"] = task.schema
            await self._executor(task.task_dict)
            await self._mark_completed(task.operation_id, task.schema)
            logger.debug(f"Task {task.operation_id} completed successfully")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Task {task.operation_id} failed: {e}")
            await self._retry_or_fail(task.operation_id, error_msg, task.schema)
        finally:
            # Remove from active tasks
            async with self._in_flight_lock:
                self._active_tasks.pop(task.operation_id, None)

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

        if total_count > 0:
            logger.info(f"Worker {self._worker_id} recovered {total_count} stale tasks from previous run")
        return total_count

    async def run(self):
        """
        Main polling loop.

        Continuously polls for pending tasks, claims them, and executes them
        until shutdown is signaled.

        If tenant_extension is configured, dynamically discovers schemas on each poll.
        """
        # Recover any tasks from a previous crash before starting
        await self.recover_own_tasks()

        logger.info(f"Worker {self._worker_id} starting polling loop")

        while not self._shutdown.is_set():
            try:
                # Claim a batch of tasks (across all tenant schemas if configured)
                tasks = await self.claim_batch()

                if tasks:
                    # Log batch info
                    task_types: dict[str, int] = {}
                    schemas_seen: set[str | None] = set()
                    for task in tasks:
                        t = task.task_dict.get("type", "unknown")
                        task_types[t] = task_types.get(t, 0) + 1
                        schemas_seen.add(task.schema)
                    types_str = ", ".join(f"{k}:{v}" for k, v in task_types.items())
                    schemas_str = ", ".join(s or "public" for s in schemas_seen)
                    logger.info(
                        f"Worker {self._worker_id} claimed {len(tasks)} tasks: {types_str} (schemas: {schemas_str})"
                    )

                    # Track in-flight tasks
                    async with self._in_flight_lock:
                        self._in_flight_count += len(tasks)

                    # Execute tasks concurrently
                    try:
                        await asyncio.gather(
                            *[self.execute_task(task) for task in tasks],
                            return_exceptions=True,
                        )
                    finally:
                        async with self._in_flight_lock:
                            self._in_flight_count -= len(tasks)
                else:
                    # No tasks found, wait before polling again
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

            if in_flight == 0:
                logger.info(f"Worker {self._worker_id} graceful shutdown complete")
                return

            logger.info(f"Worker {self._worker_id} waiting for {in_flight} in-flight tasks")
            await asyncio.sleep(0.5)

        logger.warning(f"Worker {self._worker_id} shutdown timeout after {timeout}s")

    async def _log_progress_if_due(self):
        """Log progress stats every PROGRESS_LOG_INTERVAL seconds."""
        now = time.time()
        if now - self._last_progress_log < PROGRESS_LOG_INTERVAL:
            return

        self._last_progress_log = now

        try:
            # Get local active tasks (this worker only)
            async with self._in_flight_lock:
                in_flight = self._in_flight_count
                active_tasks = dict(self._active_tasks)  # Copy to avoid holding lock

            # Build local processing breakdown grouped by (op_type, bank_id)
            task_groups: dict[tuple[str, str], int] = {}
            for op_type, bank_id, _ in active_tasks.values():
                key = (op_type, bank_id)
                task_groups[key] = task_groups.get(key, 0) + 1

            processing_info = [f"{op}:{bank}({cnt})" for (op, bank), cnt in task_groups.items()]
            processing_str = ", ".join(processing_info[:10]) if processing_info else "none"
            if len(processing_info) > 10:
                processing_str += f" +{len(processing_info) - 10} more"

            # Get global stats from DB across all schemas
            schemas = await self._get_schemas()
            global_pending = 0
            all_worker_counts: dict[str, int] = {}

            async with self._pool.acquire() as conn:
                for schema in schemas:
                    table = fq_table("async_operations", schema)

                    row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table} WHERE status = 'pending'")
                    global_pending += row["count"] if row else 0

                    # Get processing breakdown by worker
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

            # Format other workers' processing counts
            other_workers = []
            for wid, cnt in all_worker_counts.items():
                if wid != self._worker_id:
                    other_workers.append(f"{wid}:{cnt}")
            others_str = ", ".join(other_workers) if other_workers else "none"

            schemas_str = ", ".join(s or "public" for s in schemas)
            logger.info(
                f"[WORKER_STATS] worker={self._worker_id} in_flight={in_flight} | "
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
