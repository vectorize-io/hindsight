"""
Task backend for distributed task processing.

This provides an abstraction for task storage and execution:
- BrokerTaskBackend: Uses PostgreSQL as broker (production)
- SyncTaskBackend: Executes tasks immediately (testing/embedded)
"""

import json
import logging
from abc import ABC, abstractmethod
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


class TaskBackend(ABC):
    """
    Abstract base class for task execution backends.

    Implementations must:
    1. Store/publish task events (as serializable dicts)
    2. Execute tasks through a provided executor callback (optional)

    The backend treats tasks as pure dictionaries that can be serialized
    and stored in the database. The executor (typically MemoryEngine.execute_task)
    receives the dict and routes it to the appropriate handler.
    """

    def __init__(self):
        """Initialize the task backend."""
        self._executor: Callable[[dict[str, Any]], Awaitable[None]] | None = None
        self._initialized = False

    def set_executor(self, executor: Callable[[dict[str, Any]], Awaitable[None]]):
        """
        Set the executor callback for processing tasks.

        Args:
            executor: Async function that takes a task dict and executes it
        """
        self._executor = executor

    @abstractmethod
    async def initialize(self):
        """
        Initialize the backend (e.g., connect to database).
        """
        pass

    @abstractmethod
    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Submit a task for execution.

        Args:
            task_dict: Task as a dictionary (must be serializable)
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """
        Shutdown the backend gracefully.
        """
        pass

    async def _execute_task(self, task_dict: dict[str, Any]):
        """
        Execute a task through the registered executor.

        Args:
            task_dict: Task dictionary to execute
        """
        if self._executor is None:
            task_type = task_dict.get("type", "unknown")
            logger.warning(f"No executor registered, skipping task {task_type}")
            return

        try:
            await self._executor(task_dict)
        except Exception as e:
            task_type = task_dict.get("type", "unknown")
            logger.error(f"Error executing task {task_type}: {e}")
            import traceback

            traceback.print_exc()


class SyncTaskBackend(TaskBackend):
    """
    Synchronous task backend that executes tasks immediately.

    This is useful for tests and embedded/CLI usage where we don't want
    background workers. Tasks are executed inline rather than being queued.
    """

    async def initialize(self):
        """No-op for sync backend."""
        self._initialized = True
        logger.debug("SyncTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Execute the task immediately (synchronously).

        Args:
            task_dict: Task dictionary to execute
        """
        if not self._initialized:
            await self.initialize()

        await self._execute_task(task_dict)

    async def shutdown(self):
        """No-op for sync backend."""
        self._initialized = False
        logger.debug("SyncTaskBackend shutdown")


class BrokerTaskBackend(TaskBackend):
    """
    Task backend using PostgreSQL as broker.

    submit_task() stores task_payload in async_operations table.
    Actual polling and execution is handled separately by WorkerPoller.

    This backend is used by the API to store tasks. Workers poll
    the database separately to claim and execute tasks.
    """

    def __init__(
        self,
        pool_getter: Callable[[], "asyncpg.Pool"],
        schema: str | None = None,
        schema_getter: Callable[[], str | None] | None = None,
    ):
        """
        Initialize the broker task backend.

        Args:
            pool_getter: Callable that returns the asyncpg connection pool
            schema: Database schema for multi-tenant support (optional, static)
            schema_getter: Callable that returns current schema dynamically (optional).
                          If set, takes precedence over static schema for submit_task.
        """
        super().__init__()
        self._pool_getter = pool_getter
        self._schema = schema
        self._schema_getter = schema_getter

    async def initialize(self):
        """Initialize the backend."""
        self._initialized = True
        logger.info("BrokerTaskBackend initialized")

    async def submit_task(self, task_dict: dict[str, Any]):
        """
        Store task payload in async_operations table.

        The task_dict should contain an 'operation_id' if updating an existing
        operation record, otherwise a new operation will be created.

        Args:
            task_dict: Task dictionary to store (must be JSON serializable)
        """
        if not self._initialized:
            await self.initialize()

        pool = self._pool_getter()
        operation_id = task_dict.get("operation_id")
        task_type = task_dict.get("type", "unknown")
        bank_id = task_dict.get("bank_id")

        # Custom encoder to handle datetime objects
        from datetime import datetime

        def datetime_encoder(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        payload_json = json.dumps(task_dict, default=datetime_encoder)

        schema = self._schema_getter() if self._schema_getter else self._schema
        table = fq_table("async_operations", schema)

        if operation_id:
            # Update existing operation with task payload
            await pool.execute(
                f"""
                UPDATE {table}
                SET task_payload = $1::jsonb, updated_at = now()
                WHERE operation_id = $2
                """,
                payload_json,
                operation_id,
            )
            logger.debug(f"Updated task payload for operation {operation_id}")
        else:
            # Insert new operation (for tasks without pre-created records)
            # e.g., access_count_update tasks
            import uuid

            new_id = uuid.uuid4()
            await pool.execute(
                f"""
                INSERT INTO {table} (operation_id, bank_id, operation_type, status, task_payload)
                VALUES ($1, $2, $3, 'pending', $4::jsonb)
                """,
                new_id,
                bank_id,
                task_type,
                payload_json,
            )
            logger.debug(f"Created new operation {new_id} for task type {task_type}")

    async def shutdown(self):
        """Shutdown the backend."""
        self._initialized = False
        logger.info("BrokerTaskBackend shutdown")

    async def wait_for_pending_tasks(self, timeout: float = 120.0):
        """
        Wait for pending tasks to be processed.

        In the broker model, this polls the database to check if tasks
        for this process have been completed. This is useful in tests
        when worker_enabled=True (API processes its own tasks).

        Args:
            timeout: Maximum time to wait in seconds
        """
        import asyncio

        pool = self._pool_getter()
        schema = self._schema_getter() if self._schema_getter else self._schema
        table = fq_table("async_operations", schema)

        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if there are any pending tasks with payloads
            count = await pool.fetchval(
                f"""
                SELECT COUNT(*) FROM {table}
                WHERE status = 'pending' AND task_payload IS NOT NULL
                """
            )

            if count == 0:
                return

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for pending tasks after {timeout}s")
