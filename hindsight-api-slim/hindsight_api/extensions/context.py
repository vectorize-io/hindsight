"""Extension context providing a controlled API for extensions to interact with the system."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hindsight_api.engine.interface import MemoryEngineInterface
    from hindsight_api.webhooks.manager import WebhookManager


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
        webhook_manager: "WebhookManager | None" = None,
        current_schema: str | None = None,
    ):
        """
        Initialize the context.

        Args:
            database_url: SQLAlchemy database URL for migrations.
            memory_engine: Optional MemoryEngine instance for memory operations.
            webhook_manager: Optional WebhookManager for firing webhooks.
            current_schema: Optional current schema name for tenant context.
        """
        self._database_url = database_url
        self._memory_engine = memory_engine
        self.webhook_manager = webhook_manager
        self.current_schema = current_schema

    async def run_migration(self, schema: str) -> None:
        """Run migrations for a specific schema."""
        import asyncio

        from hindsight_api.config import get_config
        from hindsight_api.migrations import (
            ensure_embedding_dimension,
            ensure_text_search_extension,
            ensure_vector_extension,
            run_migrations,
        )

        # Prefer getting URL from memory engine (handles pg0 case where URL is set after init)
        db_url = self._database_url
        if self._memory_engine is not None:
            engine_url = getattr(self._memory_engine, "db_url", None)
            if engine_url:
                db_url = engine_url

        # Run synchronous migration functions in a thread so the asyncio event loop
        # remains free. This is critical for single-machine deployments where the
        # worker runs in-process: if run_migrations() blocks the event loop, any
        # in-flight asyncpg transactions cannot flush their COMMIT, and
        # CREATE INDEX CONCURRENTLY inside the migration waits for those transactions
        # forever — a deadlock.
        config = get_config()
        await asyncio.to_thread(
            run_migrations, db_url, schema=schema, migration_database_url=config.migration_database_url
        )

        # Ensure embedding column dimension matches the model's dimension
        # This is needed because migrations create columns with default dimension
        if self._memory_engine is not None:
            embeddings = getattr(self._memory_engine, "embeddings", None)
            if embeddings is not None:
                dimension = getattr(embeddings, "dimension", None)
                if dimension is not None:
                    await asyncio.to_thread(
                        ensure_embedding_dimension,
                        db_url,
                        dimension,
                        schema=schema,
                        vector_extension=config.vector_extension,
                    )

        # Ensure vector indexes match the configured extension
        await asyncio.to_thread(
            ensure_vector_extension, db_url, vector_extension=config.vector_extension, schema=schema
        )

        # Ensure text search columns/indexes match the configured extension
        await asyncio.to_thread(
            ensure_text_search_extension,
            db_url,
            text_search_extension=config.text_search_extension,
            pg_search_tokenizer=config.text_search_extension_pg_search_tokenizer,
            schema=schema,
        )

    def get_memory_engine(self) -> "MemoryEngineInterface":
        """Get the memory engine interface."""
        if self._memory_engine is None:
            raise RuntimeError(
                "Memory engine not configured in ExtensionContext. "
                "Ensure the context was created with a memory_engine parameter."
            )
        return self._memory_engine
