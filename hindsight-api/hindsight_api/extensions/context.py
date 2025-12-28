"""Extension context providing a controlled API for extensions to interact with the system."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_api.engine.interface import MemoryEngineInterface


class ExtensionContext(ABC):
    """
    Abstract context providing a controlled API for extensions.

    Extensions receive this context instead of direct access to internal
    components like MemoryEngine or database connections. This provides:
    - A stable API that won't break when internals change
    - Security by limiting what extensions can access
    - Clear documentation of what extensions can do

    Built-in implementation:
        hindsight_api.extensions.builtin.context.DefaultExtensionContext

    Example usage in an extension:
        class MyTenantExtension(TenantExtension):
            async def on_startup(self) -> None:
                # Run migrations for a new tenant schema
                await self.context.run_migration("tenant_acme")

        class MyHttpExtension(HttpExtension):
            def get_router(self, memory):
                # Use memory engine for custom endpoints
                engine = self.context.get_memory_engine()
                ...
    """

    @abstractmethod
    async def run_migration(self, schema: str) -> None:
        """
        Run database migrations for a specific schema.

        This creates the schema if it doesn't exist and runs all pending
        migrations. Uses advisory locks to coordinate between distributed workers.

        Args:
            schema: PostgreSQL schema name (e.g., "tenant_acme").
                    The schema will be created if it doesn't exist.

        Raises:
            RuntimeError: If migrations fail to complete.

        Example:
            # Provision a new tenant schema
            await context.run_migration("tenant_acme")
        """
        ...

    @abstractmethod
    def get_memory_engine(self) -> "MemoryEngineInterface":
        """
        Get the memory engine interface.

        Returns the MemoryEngineInterface for performing memory operations
        like retain, recall, reflect, and entity/document management.

        Returns:
            MemoryEngineInterface instance.

        Example:
            engine = context.get_memory_engine()
            result = await engine.recall_async(bank_id, query)
        """
        ...


class DefaultExtensionContext(ExtensionContext):
    """
    Default implementation of ExtensionContext.

    Uses the system's database URL and migration infrastructure.
    """

    def __init__(
        self,
        database_url: str,
        memory_engine: "MemoryEngineInterface | None" = None,
    ):
        """
        Initialize the context.

        Args:
            database_url: SQLAlchemy database URL for migrations.
            memory_engine: Optional MemoryEngine instance for memory operations.
        """
        self._database_url = database_url
        self._memory_engine = memory_engine

    async def run_migration(self, schema: str) -> None:
        """Run migrations for a specific schema."""
        from hindsight_api.migrations import run_migrations

        # Prefer getting URL from memory engine (handles pg0 case where URL is set after init)
        db_url = self._database_url
        if self._memory_engine is not None:
            engine_url = getattr(self._memory_engine, "db_url", None)
            if engine_url:
                db_url = engine_url

        run_migrations(db_url, schema=schema)

    def get_memory_engine(self) -> "MemoryEngineInterface":
        """Get the memory engine interface."""
        if self._memory_engine is None:
            raise RuntimeError(
                "Memory engine not configured in ExtensionContext. "
                "Ensure the context was created with a memory_engine parameter."
            )
        return self._memory_engine
