"""
Worker poller for distributed task execution.

Polls PostgreSQL for pending tasks and executes them using
FOR UPDATE SKIP LOCKED for safe concurrent claiming.
"""

import asyncio
import json
import logging
import traceback
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


def fq_table(table: str, schema: str | None = None) -> str:
    """Get fully-qualified table name with optional schema prefix."""
    if schema:
        return f'"{schema}".{table}'
    return table


class WorkerPoller:
    """
    Polls PostgreSQL for pending tasks and executes them.

    Uses FOR UPDATE SKIP LOCKED for safe distributed claiming,
    allowing multiple workers to process tasks without conflicts.
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
            schema: Database schema for multi-tenant support (optional)
        """
        self._pool = pool
        self._worker_id = worker_id
        self._executor = executor
        self._poll_interval_ms = poll_interval_ms
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._schema = schema
        self._shutdown = asyncio.Event()
        self._current_tasks: set[asyncio.Task] = set()
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()

    async def claim_batch(self) -> list[tuple[str, dict[str, Any]]]:
        """
        Claim up to batch_size pending tasks atomically.

        Uses FOR UPDATE SKIP LOCKED to ensure no conflicts with other workers.

        For consolidation tasks specifically, skips pending tasks if there's already
        a processing consolidation for the same bank (to avoid duplicate work).

        Returns:
            List of tuples (operation_id, task_dict)
        """
        table = fq_table("async_operations", self._schema)

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
                    self._batch_size,
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

                # Parse and return task payloads
                return [(str(row["operation_id"]), json.loads(row["task_payload"])) for row in rows]

    async def _mark_completed(self, operation_id: str):
        """Mark a task as completed."""
        table = fq_table("async_operations", self._schema)
        await self._pool.execute(
            f"""
            UPDATE {table}
            SET status = 'completed', completed_at = now(), updated_at = now()
            WHERE operation_id = $1
            """,
            operation_id,
        )

    async def _mark_failed(self, operation_id: str, error_message: str):
        """Mark a task as failed with error message."""
        table = fq_table("async_operations", self._schema)
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

    async def _retry_or_fail(self, operation_id: str, error_message: str):
        """Increment retry count or mark as failed if max retries exceeded."""
        table = fq_table("async_operations", self._schema)

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
                operation_id, f"Max retries ({self._max_retries}) exceeded. Last error: {error_message}"
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

    async def execute_task(self, operation_id: str, task_dict: dict[str, Any]):
        """Execute a single task and update its status."""
        task_type = task_dict.get("type", "unknown")
        bank_id = task_dict.get("bank_id", "unknown")

        try:
            logger.debug(f"Executing task {operation_id} (type={task_type}, bank={bank_id})")
            await self._executor(task_dict)
            await self._mark_completed(operation_id)
            logger.debug(f"Task {operation_id} completed successfully")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error(f"Task {operation_id} failed: {e}")
            await self._retry_or_fail(operation_id, error_msg)

    async def recover_own_tasks(self) -> int:
        """
        Recover tasks that were assigned to this worker but not completed.

        This handles the case where a worker crashes while processing tasks.
        On startup, we reset any tasks stuck in 'processing' for this worker_id
        back to 'pending' so they can be picked up again.

        Returns:
            Number of tasks recovered
        """
        table = fq_table("async_operations", self._schema)

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
        if count > 0:
            logger.info(f"Worker {self._worker_id} recovered {count} stale tasks from previous run")
        return count

    async def run(self):
        """
        Main polling loop.

        Continuously polls for pending tasks, claims them, and executes them
        until shutdown is signaled.
        """
        # Recover any tasks from a previous crash before starting
        await self.recover_own_tasks()

        logger.info(f"Worker {self._worker_id} starting polling loop")

        while not self._shutdown.is_set():
            try:
                # Claim a batch of tasks
                tasks = await self.claim_batch()

                if tasks:
                    # Log batch info
                    task_types = {}
                    for _, task_dict in tasks:
                        t = task_dict.get("type", "unknown")
                        task_types[t] = task_types.get(t, 0) + 1
                    types_str = ", ".join(f"{k}:{v}" for k, v in task_types.items())
                    logger.info(f"Worker {self._worker_id} claimed {len(tasks)} tasks: {types_str}")

                    # Track in-flight tasks
                    async with self._in_flight_lock:
                        self._in_flight_count += len(tasks)

                    # Execute tasks concurrently
                    try:
                        await asyncio.gather(
                            *[self.execute_task(op_id, task_dict) for op_id, task_dict in tasks],
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

    @property
    def worker_id(self) -> str:
        """Get the worker ID."""
        return self._worker_id

    @property
    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown.is_set()
