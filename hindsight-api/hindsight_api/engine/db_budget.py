"""
Database connection budget management.

Limits concurrent database connections per operation to prevent
a single operation (e.g., recall with parallel queries) from
exhausting the connection pool.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class OperationBudget:
    """
    Tracks connection budget for a single operation.

    Each operation gets a semaphore limiting its concurrent connections.
    """

    operation_id: str
    max_connections: int
    semaphore: asyncio.Semaphore = field(init=False)
    active_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.semaphore = asyncio.Semaphore(self.max_connections)


class ConnectionBudgetManager:
    """
    Manages per-operation connection budgets.

    Usage:
        manager = ConnectionBudgetManager(default_budget=4)

        # Start an operation
        async with manager.operation(max_connections=2) as op:
            # Acquire connections within the budget
            async with op.acquire(pool) as conn:
                await conn.fetch(...)

            # Multiple connections respect the budget
            async with op.acquire(pool) as conn1, op.acquire(pool) as conn2:
                # At most 2 concurrent connections for this operation
                ...
    """

    def __init__(self, default_budget: int = 4):
        """
        Initialize the budget manager.

        Args:
            default_budget: Default max connections per operation
        """
        self.default_budget = default_budget
        self._operations: dict[str, OperationBudget] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def operation(
        self,
        max_connections: int | None = None,
        operation_id: str | None = None,
    ) -> AsyncIterator["BudgetedOperation"]:
        """
        Create a budgeted operation context.

        Args:
            max_connections: Max concurrent connections for this operation.
                           Defaults to manager's default_budget.
            operation_id: Optional custom operation ID. Auto-generated if not provided.

        Yields:
            BudgetedOperation context for acquiring connections
        """
        op_id = operation_id or f"op-{uuid.uuid4().hex[:12]}"
        budget = max_connections or self.default_budget

        async with self._lock:
            if op_id in self._operations:
                raise ValueError(f"Operation {op_id} already exists")
            self._operations[op_id] = OperationBudget(op_id, budget)

        try:
            yield BudgetedOperation(self, op_id)
        finally:
            async with self._lock:
                self._operations.pop(op_id, None)

    def _get_budget(self, operation_id: str) -> OperationBudget:
        """Get budget for an operation (internal use)."""
        budget = self._operations.get(operation_id)
        if not budget:
            raise ValueError(f"Operation {operation_id} not found")
        return budget


class BudgetedOperation:
    """
    A single operation with connection budget.

    Provides methods to acquire connections within the budget.
    """

    def __init__(self, manager: ConnectionBudgetManager, operation_id: str):
        self._manager = manager
        self.operation_id = operation_id

    @property
    def budget(self) -> OperationBudget:
        """Get the budget for this operation."""
        return self._manager._get_budget(self.operation_id)

    @asynccontextmanager
    async def acquire(self, pool: "asyncpg.Pool") -> AsyncIterator["asyncpg.Connection"]:
        """
        Acquire a connection within the operation's budget.

        Blocks if the operation has reached its connection limit.

        Args:
            pool: asyncpg connection pool

        Yields:
            Database connection
        """
        budget = self.budget
        async with budget.semaphore:
            budget.active_count += 1
            conn = await pool.acquire()
            try:
                yield conn
            finally:
                budget.active_count -= 1
                await pool.release(conn)

    def wrap_pool(self, pool: "asyncpg.Pool") -> "BudgetedPool":
        """
        Wrap a pool with this operation's budget.

        The returned BudgetedPool can be passed to functions expecting a pool,
        and all acquire() calls will be limited by this operation's budget.

        Args:
            pool: asyncpg connection pool to wrap

        Returns:
            BudgetedPool that limits connections to this operation's budget
        """
        return BudgetedPool(pool, self)

    async def acquire_many(
        self,
        pool: "asyncpg.Pool",
        count: int,
    ) -> AsyncIterator[list["asyncpg.Connection"]]:
        """
        Acquire multiple connections within the budget.

        Note: This acquires connections sequentially to respect the budget.
        For parallel acquisition, use multiple acquire() calls with asyncio.gather().

        Args:
            pool: asyncpg connection pool
            count: Number of connections to acquire

        Yields:
            List of database connections
        """
        connections = []
        try:
            for _ in range(count):
                conn = await pool.acquire()
                connections.append(conn)
            yield connections
        finally:
            for conn in connections:
                await pool.release(conn)


# Global default manager instance
_default_manager: ConnectionBudgetManager | None = None


def get_budget_manager(default_budget: int = 4) -> ConnectionBudgetManager:
    """
    Get or create the global budget manager.

    Args:
        default_budget: Default max connections per operation

    Returns:
        Global ConnectionBudgetManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ConnectionBudgetManager(default_budget=default_budget)
    return _default_manager


@asynccontextmanager
async def budgeted_operation(
    max_connections: int | None = None,
    operation_id: str | None = None,
    default_budget: int = 4,
) -> AsyncIterator[BudgetedOperation]:
    """
    Convenience function to create a budgeted operation.

    Args:
        max_connections: Max concurrent connections for this operation
        operation_id: Optional custom operation ID
        default_budget: Default budget if manager not yet created

    Yields:
        BudgetedOperation context

    Example:
        async with budgeted_operation(max_connections=2) as op:
            async with op.acquire(pool) as conn:
                await conn.fetch(...)
    """
    manager = get_budget_manager(default_budget)
    async with manager.operation(max_connections, operation_id) as op:
        yield op


class BudgetedPool:
    """
    A pool wrapper that limits concurrent connection acquisitions.

    This can be passed to functions expecting a pool, and acquire()
    calls will be limited by the budget semaphore.

    Usage:
        async with budgeted_operation(max_connections=4) as op:
            budgeted_pool = op.wrap_pool(pool)
            # Pass budgeted_pool to functions that expect a pool
            await some_function(budgeted_pool, ...)
    """

    def __init__(self, pool: "asyncpg.Pool", operation: BudgetedOperation):
        self._pool = pool
        self._operation = operation

    async def acquire(self) -> "asyncpg.Connection":
        """
        Acquire a connection within the budget.

        Note: Caller must release the connection when done.
        Prefer using as context manager via acquire_with_retry or op.acquire().
        """
        budget = self._operation.budget
        await budget.semaphore.acquire()
        budget.active_count += 1
        try:
            return await self._pool.acquire()
        except Exception:
            budget.active_count -= 1
            budget.semaphore.release()
            raise

    async def release(self, conn: "asyncpg.Connection") -> None:
        """Release a connection back to the pool."""
        budget = self._operation.budget
        try:
            await self._pool.release(conn)
        finally:
            budget.active_count -= 1
            budget.semaphore.release()

    def __getattr__(self, name):
        """Proxy other attributes to the underlying pool."""
        return getattr(self._pool, name)
