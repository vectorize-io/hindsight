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

    @property
    def is_primary(self) -> bool:
        """Whether this context belongs to the loop that owns process-level work.

        A multi-loop server builds one application per loop and designates exactly one of
        them primary; that one runs what belongs to the PROCESS rather than to a loop --
        migrations, the background pollers, and any one-off startup provisioning an
        extension does. An extension that installs shared infrastructure on startup should
        guard it with this, or it does that work once per loop.

        Always True for a single-loop server, which is every deployment that does not opt
        into more, so an extension that ignores this keeps its current behaviour.
        """
        return True

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
        is_primary: bool = True,
    ):
        """
        Initialize the context.

        Args:
            database_url: SQLAlchemy database URL for migrations.
            memory_engine: Optional MemoryEngine instance for memory operations.
            webhook_manager: Optional WebhookManager for firing webhooks.
            current_schema: Optional current schema name for tenant context.
            is_primary: Whether this context's loop owns process-level work. Defaults to
                True so a single-loop server, and any caller that predates the flag, keeps
                doing that work.
        """
        self._database_url = database_url
        self._memory_engine = memory_engine
        self.webhook_manager = webhook_manager
        self.current_schema = current_schema
        self._is_primary = is_primary

    async def run_migration(self, schema: str) -> None:
        """Run migrations for a specific schema."""
        import asyncio

        from hindsight_api.config import get_config
        from hindsight_api.migrations import run_migrations_for_schemas

        # Prefer getting URL from memory engine (handles pg0 case where URL is set after init)
        db_url = self._database_url
        if self._memory_engine is not None:
            engine_url = getattr(self._memory_engine, "db_url", None)
            if engine_url:
                db_url = engine_url

        # Ensure the embedding column dimension matches the model's dimension: migrations
        # create the columns with a default dimension.
        embedding_dimension: int | None = None
        if self._memory_engine is not None:
            embeddings = getattr(self._memory_engine, "embeddings", None)
            if embeddings is not None:
                embedding_dimension = getattr(embeddings, "dimension", None)

        # One call, not four: run_migrations_for_schemas is the migration-isolation
        # boundary. Calling run_migrations() and then the ensure_* helpers separately
        # would run each of them here -- and every one opens SQLAlchemy's sync engine,
        # i.e. psycopg2, which has no free-threaded build. On a python3.14t interpreter
        # that import re-enables the GIL for the life of the server (and, with the
        # strict free-threading guard on, fails the request outright). Startup already
        # goes through this entrypoint; runtime tenant provisioning must too, or
        # provisioning a new bank on a free-threaded API 500s.
        config = get_config()
        await asyncio.to_thread(
            run_migrations_for_schemas,
            db_url,
            [schema],
            migration_database_url=config.migration_database_url,
            embedding_dimension=embedding_dimension,
            vector_extension=config.vector_extension,
            text_search_extension=config.text_search_extension,
            pg_search_tokenizer=config.text_search_extension_pg_search_tokenizer,
        )

        # Provision any extension-owned bank-scoped tables for this schema,
        # right after core migrations, so extension schema evolves on the same
        # lifecycle as core schema (instead of via a lazy per-request path).
        # No-op unless a tenant extension declares a provisioner; errors
        # propagate so a failed provision surfaces here, not at request time.
        engine = self._memory_engine
        get_pool = getattr(engine, "_get_pool", None)
        tenant_extension = getattr(engine, "tenant_extension", None)
        if get_pool is not None and tenant_extension is not None:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await tenant_extension.provision_bank_tables(conn, schema)

    @property
    def is_primary(self) -> bool:
        """Whether this context's loop owns process-level work."""
        return self._is_primary

    def get_memory_engine(self) -> "MemoryEngineInterface":
        """Get the memory engine interface."""
        if self._memory_engine is None:
            raise RuntimeError(
                "Memory engine not configured in ExtensionContext. "
                "Ensure the context was created with a memory_engine parameter."
            )
        return self._memory_engine
