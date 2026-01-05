"""
Memory Engine for Memory Banks.

This implements a sophisticated memory architecture that combines:
1. Temporal links: Memories connected by time proximity
2. Semantic links: Memories connected by meaning/similarity
3. Entity links: Memories connected by shared entities (PERSON, ORG, etc.)
4. Spreading activation: Search through the graph with activation decay
5. Dynamic weighting: Recency and frequency-based importance
"""

import asyncio
import contextvars
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ..config import get_config

# Context variable for current schema (async-safe, per-task isolation)
_current_schema: contextvars.ContextVar[str] = contextvars.ContextVar("current_schema", default="public")


def get_current_schema() -> str:
    """Get the current schema from context (default: 'public')."""
    return _current_schema.get()


def fq_table(table_name: str) -> str:
    """
    Get fully-qualified table name with current schema.

    Example:
        fq_table("memory_units") -> "public.memory_units"
        fq_table("memory_units") -> "tenant_xyz.memory_units" (if schema is set)
    """
    return f"{get_current_schema()}.{table_name}"


# Tables that must be schema-qualified (for runtime validation)
_PROTECTED_TABLES = frozenset(
    [
        "memory_units",
        "memory_links",
        "unit_entities",
        "entities",
        "entity_cooccurrences",
        "banks",
        "documents",
        "chunks",
        "async_operations",
        "fact_usefulness",
        "usefulness_signals",
        "query_pattern_stats",
    ]
)

# Enable runtime SQL validation (can be disabled in production for performance)
_VALIDATE_SQL_SCHEMAS = True


class UnqualifiedTableError(Exception):
    """Raised when SQL contains unqualified table references."""

    pass


def validate_sql_schema(sql: str) -> None:
    """
    Validate that SQL doesn't contain unqualified table references.

    This is a runtime safety check to prevent cross-tenant data access.
    Raises UnqualifiedTableError if any protected table is referenced
    without a schema prefix.

    Args:
        sql: The SQL query to validate

    Raises:
        UnqualifiedTableError: If unqualified table reference found
    """
    if not _VALIDATE_SQL_SCHEMAS:
        return

    import re

    sql_upper = sql.upper()

    for table in _PROTECTED_TABLES:
        table_upper = table.upper()

        # Pattern: SQL keyword followed by unqualified table name
        # Matches: FROM memory_units, JOIN memory_units, INTO memory_units, UPDATE memory_units
        patterns = [
            rf"FROM\s+{table_upper}(?:\s|$|,|\)|;)",
            rf"JOIN\s+{table_upper}(?:\s|$|,|\)|;)",
            rf"INTO\s+{table_upper}(?:\s|$|\()",
            rf"UPDATE\s+{table_upper}(?:\s|$)",
            rf"DELETE\s+FROM\s+{table_upper}(?:\s|$|;)",
        ]

        for pattern in patterns:
            match = re.search(pattern, sql_upper)
            if match:
                # Check if it's actually qualified (preceded by schema.)
                # Look backwards from match to see if there's a dot
                start = match.start()
                # Find the table name position in the match
                table_pos = sql_upper.find(table_upper, start)
                if table_pos > 0:
                    # Check character before table name (skip whitespace)
                    prefix = sql[:table_pos].rstrip()
                    if not prefix.endswith("."):
                        raise UnqualifiedTableError(
                            f"Unqualified table reference '{table}' in SQL. "
                            f"Use fq_table('{table}') for schema safety. "
                            f"SQL snippet: ...{sql[max(0, start - 10) : start + 50]}..."
                        )


import asyncpg
import numpy as np
from pydantic import BaseModel, Field

from .cross_encoder import CrossEncoderModel
from .embeddings import Embeddings, create_embeddings_from_env
from .interface import MemoryEngineInterface

if TYPE_CHECKING:
    from hindsight_api.extensions import OperationValidatorExtension, TenantExtension
    from hindsight_api.models import RequestContext


from enum import Enum

from ..pg0 import EmbeddedPostgres
from .entity_resolver import EntityResolver
from .llm_wrapper import LLMConfig
from .query_analyzer import QueryAnalyzer
from .response_models import VALID_RECALL_FACT_TYPES, EntityObservation, EntityState, MemoryFact, ReflectResult
from .response_models import RecallResult as RecallResultModel
from .retain import bank_utils, embedding_utils
from .retain.types import RetainContentDict
from .search import observation_utils, think_utils
from .search.reranking import CrossEncoderReranker
from .task_backend import AsyncIOQueueBackend, TaskBackend


class Budget(str, Enum):
    """Budget levels for recall/reflect operations."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(UTC)


# Logger for memory system
logger = logging.getLogger(__name__)

import tiktoken

from .db_utils import acquire_with_retry

# Cache tiktoken encoding for token budget filtering (module-level singleton)
_TIKTOKEN_ENCODING = None


def _get_tiktoken_encoding():
    """Get cached tiktoken encoding (cl100k_base for GPT-4/3.5)."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


class MemoryEngine(MemoryEngineInterface):
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.

    This class provides:
    - Embedding generation for semantic search
    - Entity, temporal, and semantic link creation
    - Think operations for formulating answers with opinions
    - bank profile and disposition management
    """

    def __init__(
        self,
        db_url: str | None = None,
        memory_llm_provider: str | None = None,
        memory_llm_api_key: str | None = None,
        memory_llm_model: str | None = None,
        memory_llm_base_url: str | None = None,
        embeddings: Embeddings | None = None,
        cross_encoder: CrossEncoderModel | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        pool_min_size: int = 5,
        pool_max_size: int = 100,
        task_backend: TaskBackend | None = None,
        run_migrations: bool = True,
        operation_validator: "OperationValidatorExtension | None" = None,
        tenant_extension: "TenantExtension | None" = None,
        skip_llm_verification: bool | None = None,
        lazy_reranker: bool | None = None,
    ):
        """
        Initialize the temporal + semantic memory system.

        All parameters are optional and will be read from environment variables if not provided.
        See hindsight_api.config for environment variable names and defaults.

        Args:
            db_url: PostgreSQL connection URL. Defaults to HINDSIGHT_API_DATABASE_URL env var or "pg0".
                    Also supports pg0 URLs: "pg0" or "pg0://instance-name" or "pg0://instance-name:port"
            memory_llm_provider: LLM provider. Defaults to HINDSIGHT_API_LLM_PROVIDER env var or "groq".
            memory_llm_api_key: API key for the LLM provider. Defaults to HINDSIGHT_API_LLM_API_KEY env var.
            memory_llm_model: Model name. Defaults to HINDSIGHT_API_LLM_MODEL env var.
            memory_llm_base_url: Base URL for the LLM API. Defaults based on provider.
            embeddings: Embeddings implementation. If not provided, created from env vars.
            cross_encoder: Cross-encoder model. If not provided, created from env vars.
            query_analyzer: Query analyzer implementation. If not provided, uses DateparserQueryAnalyzer.
            pool_min_size: Minimum number of connections in the pool (default: 5)
            pool_max_size: Maximum number of connections in the pool (default: 100)
            task_backend: Custom task backend. If not provided, uses AsyncIOQueueBackend.
            run_migrations: Whether to run database migrations during initialize(). Default: True
            operation_validator: Optional extension to validate operations before execution.
                                If provided, retain/recall/reflect operations will be validated.
            tenant_extension: Optional extension for multi-tenancy and API key authentication.
                             If provided, operations require a RequestContext for authentication.
            skip_llm_verification: Skip LLM connection verification during initialization.
                                  Defaults to HINDSIGHT_API_SKIP_LLM_VERIFICATION env var or False.
            lazy_reranker: Delay reranker initialization until first use. Useful for retain-only
                          operations that don't need the cross-encoder. Defaults to
                          HINDSIGHT_API_LAZY_RERANKER env var or False.
        """
        # Load config from environment for any missing parameters
        from ..config import get_config

        config = get_config()

        # Apply optimization flags from config if not explicitly provided
        self._skip_llm_verification = (
            skip_llm_verification if skip_llm_verification is not None else config.skip_llm_verification
        )
        self._lazy_reranker = lazy_reranker if lazy_reranker is not None else config.lazy_reranker

        # Apply defaults from config
        db_url = db_url or config.database_url
        memory_llm_provider = memory_llm_provider or config.llm_provider
        memory_llm_api_key = memory_llm_api_key or config.llm_api_key
        # Ollama doesn't require an API key
        if not memory_llm_api_key and memory_llm_provider != "ollama":
            raise ValueError("LLM API key is required. Set HINDSIGHT_API_LLM_API_KEY environment variable.")
        memory_llm_model = memory_llm_model or config.llm_model
        memory_llm_base_url = memory_llm_base_url or config.get_llm_base_url() or None
        # Track pg0 instance (if used)
        self._pg0: EmbeddedPostgres | None = None
        self._pg0_instance_name: str | None = None

        # Initialize PostgreSQL connection URL
        # The actual URL will be set during initialize() after starting the server
        # Supports: "pg0" (default instance), "pg0://instance-name" (named instance), or regular postgresql:// URL
        if db_url == "pg0":
            self._use_pg0 = True
            self._pg0_instance_name = "hindsight"
            self._pg0_port = None  # Use default port
            self.db_url = None
        elif db_url.startswith("pg0://"):
            self._use_pg0 = True
            # Parse instance name and optional port: pg0://instance-name or pg0://instance-name:port
            url_part = db_url[6:]  # Remove "pg0://"
            if ":" in url_part:
                self._pg0_instance_name, port_str = url_part.rsplit(":", 1)
                self._pg0_port = int(port_str)
            else:
                self._pg0_instance_name = url_part or "hindsight"
                self._pg0_port = None  # Use default port
            self.db_url = None
        else:
            self._use_pg0 = False
            self._pg0_instance_name = None
            self._pg0_port = None
            self.db_url = db_url

        # Set default base URL if not provided
        if memory_llm_base_url is None:
            if memory_llm_provider.lower() == "groq":
                memory_llm_base_url = "https://api.groq.com/openai/v1"
            elif memory_llm_provider.lower() == "ollama":
                memory_llm_base_url = "http://localhost:11434/v1"
            else:
                memory_llm_base_url = ""

        # Connection pool (will be created in initialize())
        self._pool = None
        self._initialized = False
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._run_migrations = run_migrations

        # Initialize entity resolver (will be created in initialize())
        self.entity_resolver = None

        # Initialize embeddings (from env vars if not provided)
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = create_embeddings_from_env()

        # Initialize query analyzer
        if query_analyzer is not None:
            self.query_analyzer = query_analyzer
        else:
            from .query_analyzer import DateparserQueryAnalyzer

            self.query_analyzer = DateparserQueryAnalyzer()

        # Initialize LLM configuration
        self._llm_config = LLMConfig(
            provider=memory_llm_provider,
            api_key=memory_llm_api_key,
            base_url=memory_llm_base_url,
            model=memory_llm_model,
        )

        # Store client and model for convenience (deprecated: use _llm_config.call() instead)
        self._llm_client = self._llm_config._client
        self._llm_model = self._llm_config.model

        # Initialize cross-encoder reranker (cached for performance)
        self._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=cross_encoder)

        # Initialize task backend
        self._task_backend = task_backend or AsyncIOQueueBackend(batch_size=100, batch_interval=1.0)

        # Backpressure mechanism: limit concurrent searches to prevent overwhelming the database
        # Limit concurrent searches to prevent connection pool exhaustion
        # Each search can use 2-4 connections, so with 10 concurrent searches
        # we use ~20-40 connections max, staying well within pool limits
        self._search_semaphore = asyncio.Semaphore(10)

        # Backpressure for put operations: limit concurrent puts to prevent database contention
        # Each put_batch holds a connection for the entire transaction, so we limit to 5
        # concurrent puts to avoid connection pool exhaustion and reduce write contention
        self._put_semaphore = asyncio.Semaphore(5)

        # initialize encoding eagerly to avoid delaying the first time
        _get_tiktoken_encoding()

        # Store operation validator extension (optional)
        self._operation_validator = operation_validator

        # Store tenant extension (optional)
        self._tenant_extension = tenant_extension

    async def _validate_operation(self, validation_coro) -> None:
        """
        Run validation if an operation validator is configured.

        Args:
            validation_coro: Coroutine that returns a ValidationResult

        Raises:
            OperationValidationError: If validation fails
        """
        if self._operation_validator is None:
            return

        from hindsight_api.extensions import OperationValidationError

        result = await validation_coro
        if not result.allowed:
            raise OperationValidationError(result.reason or "Operation not allowed", result.status_code)

    async def _authenticate_tenant(self, request_context: "RequestContext | None") -> str:
        """
        Authenticate tenant and set schema in context variable.

        The schema is stored in a contextvar for async-safe, per-task isolation.
        Use fq_table(table_name) to get fully-qualified table names.

        Args:
            request_context: The request context with API key. Required if tenant_extension is configured.

        Returns:
            Schema name that was set in the context.

        Raises:
            AuthenticationError: If authentication fails or request_context is missing when required.
        """
        if self._tenant_extension is None:
            _current_schema.set("public")
            return "public"

        from hindsight_api.extensions import AuthenticationError

        if request_context is None:
            raise AuthenticationError("RequestContext is required when tenant extension is configured")

        # Let AuthenticationError propagate - HTTP layer will convert to 401
        tenant_context = await self._tenant_extension.authenticate(request_context)

        _current_schema.set(tenant_context.schema_name)
        return tenant_context.schema_name

    async def _handle_access_count_update(self, task_dict: dict[str, Any]):
        """
        Handler for access count update tasks.

        Args:
            task_dict: Dict with 'node_ids' key containing list of node IDs to update

        Raises:
            Exception: Any exception from database operations (propagates to execute_task for retry)
        """
        node_ids = task_dict.get("node_ids", [])
        if not node_ids:
            return

        pool = await self._get_pool()
        # Convert string UUIDs to UUID type for faster matching
        uuid_list = [uuid.UUID(nid) for nid in node_ids]
        async with acquire_with_retry(pool) as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET access_count = access_count + 1 WHERE id = ANY($1::uuid[])",
                uuid_list,
            )

    async def _handle_batch_retain(self, task_dict: dict[str, Any]):
        """
        Handler for batch retain tasks.

        Args:
            task_dict: Dict with 'bank_id', 'contents'

        Raises:
            ValueError: If bank_id is missing
            Exception: Any exception from retain_batch_async (propagates to execute_task for retry)
        """
        bank_id = task_dict.get("bank_id")
        if not bank_id:
            raise ValueError("bank_id is required for batch retain task")
        contents = task_dict.get("contents", [])

        logger.info(
            f"[BATCH_RETAIN_TASK] Starting background batch retain for bank_id={bank_id}, {len(contents)} items"
        )

        # Use internal request context for background tasks
        from hindsight_api.models import RequestContext

        internal_context = RequestContext()
        await self.retain_batch_async(bank_id=bank_id, contents=contents, request_context=internal_context)

        logger.info(f"[BATCH_RETAIN_TASK] Completed background batch retain for bank_id={bank_id}")

    async def execute_task(self, task_dict: dict[str, Any]):
        """
        Execute a task by routing it to the appropriate handler.

        This method is called by the task backend to execute tasks.
        It receives a plain dict that can be serialized and sent over the network.

        Args:
            task_dict: Task dictionary with 'type' key and other payload data
                      Example: {'type': 'access_count_update', 'node_ids': [...]}
        """
        task_type = task_dict.get("type")
        operation_id = task_dict.get("operation_id")
        retry_count = task_dict.get("retry_count", 0)
        max_retries = 3

        # Check if operation was cancelled (only for tasks with operation_id)
        if operation_id:
            try:
                pool = await self._get_pool()
                async with acquire_with_retry(pool) as conn:
                    result = await conn.fetchrow(
                        f"SELECT operation_id FROM {fq_table('async_operations')} WHERE operation_id = $1",
                        uuid.UUID(operation_id),
                    )
                    if not result:
                        # Operation was cancelled, skip processing
                        logger.info(f"Skipping cancelled operation: {operation_id}")
                        return
            except Exception as e:
                logger.error(f"Failed to check operation status {operation_id}: {e}")
                # Continue with processing if we can't check status

        try:
            if task_type == "access_count_update":
                await self._handle_access_count_update(task_dict)
            elif task_type == "reinforce_opinion":
                await self._handle_reinforce_opinion(task_dict)
            elif task_type == "form_opinion":
                await self._handle_form_opinion(task_dict)
            elif task_type == "batch_retain":
                await self._handle_batch_retain(task_dict)
            elif task_type == "regenerate_observations":
                await self._handle_regenerate_observations(task_dict)
            else:
                logger.error(f"Unknown task type: {task_type}")
                # Don't retry unknown task types
                if operation_id:
                    await self._delete_operation_record(operation_id)
                return

            # Task succeeded - delete operation record
            if operation_id:
                await self._delete_operation_record(operation_id)

        except Exception as e:
            # Task failed - check if we should retry
            logger.error(
                f"Task execution failed (attempt {retry_count + 1}/{max_retries + 1}): {task_type}, error: {e}"
            )
            import traceback

            error_traceback = traceback.format_exc()
            traceback.print_exc()

            if retry_count < max_retries:
                # Reschedule with incremented retry count
                task_dict["retry_count"] = retry_count + 1
                logger.info(f"Rescheduling task {task_type} (retry {retry_count + 1}/{max_retries})")
                await self._task_backend.submit_task(task_dict)
            else:
                # Max retries exceeded - mark operation as failed
                logger.error(f"Max retries exceeded for task {task_type}, marking as failed")
                if operation_id:
                    await self._mark_operation_failed(operation_id, str(e), error_traceback)

    async def _delete_operation_record(self, operation_id: str):
        """Helper to delete an operation record from the database."""
        try:
            pool = await self._get_pool()
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"DELETE FROM {fq_table('async_operations')} WHERE operation_id = $1", uuid.UUID(operation_id)
                )
        except Exception as e:
            logger.error(f"Failed to delete async operation record {operation_id}: {e}")

    async def _mark_operation_failed(self, operation_id: str, error_message: str, error_traceback: str):
        """Helper to mark an operation as failed in the database."""
        try:
            pool = await self._get_pool()
            # Truncate error message to avoid extremely long strings
            full_error = f"{error_message}\n\nTraceback:\n{error_traceback}"
            truncated_error = full_error[:5000] if len(full_error) > 5000 else full_error

            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET status = 'failed', error_message = $2
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(operation_id),
                    truncated_error,
                )
            logger.info(f"Marked async operation as failed: {operation_id}")
        except Exception as e:
            logger.error(f"Failed to mark operation as failed {operation_id}: {e}")

    async def initialize(self):
        """Initialize the connection pool, models, and background workers.

        Loads models (embeddings, cross-encoder) in parallel with pg0 startup
        for faster overall initialization.
        """
        if self._initialized:
            return

        # Run model loading in thread pool (CPU-bound) in parallel with pg0 startup
        loop = asyncio.get_event_loop()

        async def start_pg0():
            """Start pg0 if configured."""
            if self._use_pg0:
                kwargs = {"name": self._pg0_instance_name}
                if self._pg0_port is not None:
                    kwargs["port"] = self._pg0_port
                pg0 = EmbeddedPostgres(**kwargs)  # type: ignore[invalid-argument-type] - dict kwargs
                # Check if pg0 is already running before we start it
                was_already_running = await pg0.is_running()
                self.db_url = await pg0.ensure_running()
                # Only track pg0 (to stop later) if WE started it
                if not was_already_running:
                    self._pg0 = pg0

        async def init_embeddings():
            """Initialize embedding model."""
            # For local providers, run in thread pool to avoid blocking event loop
            if self.embeddings.provider_name == "local":
                await loop.run_in_executor(None, lambda: asyncio.run(self.embeddings.initialize()))
            else:
                await self.embeddings.initialize()

        async def init_cross_encoder():
            """Initialize cross-encoder model."""
            cross_encoder = self._cross_encoder_reranker.cross_encoder
            # For local providers, run in thread pool to avoid blocking event loop
            if cross_encoder.provider_name == "local":
                await loop.run_in_executor(None, lambda: asyncio.run(cross_encoder.initialize()))
            else:
                await cross_encoder.initialize()
            # Mark reranker as initialized
            self._cross_encoder_reranker._initialized = True

        async def init_query_analyzer():
            """Initialize query analyzer model."""
            # Query analyzer load is sync and CPU-bound
            await loop.run_in_executor(None, self.query_analyzer.load)

        async def verify_llm():
            """Verify LLM connection is working."""
            if not self._skip_llm_verification:
                await self._llm_config.verify_connection()

        # Build list of initialization tasks
        init_tasks = [
            start_pg0(),
            init_embeddings(),
            init_query_analyzer(),
        ]

        # Only init cross-encoder eagerly if not using lazy initialization
        if not self._lazy_reranker:
            init_tasks.append(init_cross_encoder())

        # Only verify LLM if not skipping
        if not self._skip_llm_verification:
            init_tasks.append(verify_llm())

        # Run pg0 and selected model initializations in parallel
        await asyncio.gather(*init_tasks)

        # Run database migrations if enabled
        if self._run_migrations:
            from ..migrations import ensure_embedding_dimension, run_migrations

            if not self.db_url:
                raise ValueError("Database URL is required for migrations")
            logger.info("Running database migrations...")
            run_migrations(self.db_url)

            # Ensure embedding column dimension matches the model's dimension
            # This is done after migrations and after embeddings.initialize()
            ensure_embedding_dimension(self.db_url, self.embeddings.dimension)

        logger.info(f"Connecting to PostgreSQL at {self.db_url}")

        # Create connection pool
        # For read-heavy workloads with many parallel think/search operations,
        # we need a larger pool. Read operations don't need strong isolation.
        self._pool = await asyncpg.create_pool(
            self.db_url,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            command_timeout=60,
            statement_cache_size=0,  # Disable prepared statement cache
            timeout=30,  # Connection acquisition timeout (seconds)
        )

        # Initialize entity resolver with pool
        self.entity_resolver = EntityResolver(self._pool)

        # Set executor for task backend and initialize
        self._task_backend.set_executor(self.execute_task)
        await self._task_backend.initialize()

        self._initialized = True
        logger.info("Memory system initialized (pool and task backend started)")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool (must call initialize() first)."""
        if not self._initialized:
            await self.initialize()
        return self._pool

    async def _acquire_connection(self):
        """
        Acquire a connection from the pool with retry logic.

        Returns an async context manager that yields a connection.
        Retries on transient connection errors with exponential backoff.
        """
        pool = await self._get_pool()

        async def acquire():
            return await pool.acquire()

        return await _retry_with_backoff(acquire)

    async def health_check(self) -> dict:
        """
        Perform a health check by querying the database.

        Returns:
            dict with status and optional error message

        Note:
            Returns unhealthy until initialize() has completed successfully.
        """
        # Not healthy until fully initialized
        if not self._initialized:
            return {"status": "unhealthy", "reason": "not_initialized"}

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    return {"status": "healthy", "database": "connected"}
                else:
                    return {"status": "unhealthy", "database": "unexpected response"}
        except Exception as e:
            return {"status": "unhealthy", "database": "error", "error": str(e)}

    async def close(self):
        """Close the connection pool and shutdown background workers."""
        logger.info("close() started")

        # Shutdown task backend
        await self._task_backend.shutdown()

        # Close pool
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

        self._initialized = False

        # Stop pg0 if we started it
        if self._pg0 is not None:
            logger.info("Stopping pg0...")
            await self._pg0.stop()
            self._pg0 = None
            logger.info("pg0 stopped")

    async def wait_for_background_tasks(self):
        """
        Wait for all pending background tasks to complete.

        This is useful in tests to ensure background tasks (like opinion reinforcement)
        complete before making assertions.
        """
        if hasattr(self._task_backend, "wait_for_pending_tasks"):
            await self._task_backend.wait_for_pending_tasks()

    def _format_readable_date(self, dt: datetime) -> str:
        """
        Format a datetime into a readable string for temporal matching.

        Examples:
            - June 2024
            - January 15, 2024
            - December 2023

        This helps queries like "camping in June" match facts that happened in June.

        Args:
            dt: datetime object to format

        Returns:
            Readable date string
        """
        # Format as "Month Year" for most cases
        # Could be extended to include day for very specific dates if needed
        month_name = dt.strftime("%B")  # Full month name (e.g., "June")
        year = dt.strftime("%Y")  # Year (e.g., "2024")

        # For now, use "Month Year" format
        # Could check if day is significant (not 1st or 15th) and include it
        return f"{month_name} {year}"

    async def _find_duplicate_facts_batch(
        self,
        conn,
        bank_id: str,
        texts: list[str],
        embeddings: list[list[float]],
        event_date: datetime,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.95,
    ) -> list[bool]:
        """
        Check which facts are duplicates using semantic similarity + temporal window.

        For each new fact, checks if a semantically similar fact already exists
        within the time window. Uses pgvector cosine similarity for efficiency.

        Args:
            conn: Database connection
            bank_id: bank IDentifier
            texts: List of fact texts to check
            embeddings: Corresponding embeddings
            event_date: Event date for temporal filtering
            time_window_hours: Hours before/after event_date to search (default: 24)
            similarity_threshold: Minimum cosine similarity to consider duplicate (default: 0.95)

        Returns:
            List of booleans - True if fact is a duplicate (should skip), False if new
        """
        if not texts:
            return []

        # Handle edge cases where event_date is at datetime boundaries
        try:
            time_lower = event_date - timedelta(hours=time_window_hours)
        except OverflowError:
            time_lower = datetime.min
        try:
            time_upper = event_date + timedelta(hours=time_window_hours)
        except OverflowError:
            time_upper = datetime.max

        # Fetch ALL existing facts in time window ONCE (much faster than N queries)
        import time as time_mod

        fetch_start = time_mod.time()
        existing_facts = await conn.fetch(
            f"""
            SELECT id, text, embedding
            FROM {fq_table("memory_units")}
            WHERE bank_id = $1
              AND event_date BETWEEN $2 AND $3
            """,
            bank_id,
            time_lower,
            time_upper,
        )

        # If no existing facts, nothing is duplicate
        if not existing_facts:
            return [False] * len(texts)

        # Compute similarities in Python (vectorized with numpy)
        is_duplicate = []

        # Convert existing embeddings to numpy for faster computation
        embedding_arrays = []
        for row in existing_facts:
            raw_emb = row["embedding"]
            # Handle different pgvector formats
            if isinstance(raw_emb, str):
                # Parse string format: "[1.0, 2.0, ...]"
                import json

                emb = np.array(json.loads(raw_emb), dtype=np.float32)
            elif isinstance(raw_emb, (list, tuple)):
                emb = np.array(raw_emb, dtype=np.float32)
            else:
                # Try direct conversion
                emb = np.array(raw_emb, dtype=np.float32)
            embedding_arrays.append(emb)

        if not embedding_arrays:
            existing_embeddings = np.array([])
        elif len(embedding_arrays) == 1:
            # Single embedding: reshape to (1, dim)
            existing_embeddings = embedding_arrays[0].reshape(1, -1)
        else:
            # Multiple embeddings: vstack
            existing_embeddings = np.vstack(embedding_arrays)

        comp_start = time_mod.time()
        for embedding in embeddings:
            # Compute cosine similarity with all existing facts
            emb_array = np.array(embedding)
            # Cosine similarity = 1 - cosine distance
            # For normalized vectors: cosine_sim = dot product
            similarities = np.dot(existing_embeddings, emb_array)

            # Check if any existing fact is too similar
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            is_duplicate.append(max_similarity > similarity_threshold)

        return is_duplicate

    def retain(
        self,
        bank_id: str,
        content: str,
        context: str = "",
        event_date: datetime | None = None,
        request_context: "RequestContext | None" = None,
    ) -> list[str]:
        """
        Store content as memory units (synchronous wrapper).

        This is a synchronous wrapper around retain_async() for convenience.
        For best performance, use retain_async() directly.

        Args:
            bank_id: Unique identifier for the bank
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)
            request_context: Request context for authentication (optional, uses internal context if not provided)

        Returns:
            List of created unit IDs
        """
        # Run async version synchronously
        from hindsight_api.models import RequestContext as RC

        ctx = request_context if request_context is not None else RC()
        return asyncio.run(self.retain_async(bank_id, content, context, event_date, request_context=ctx))

    async def retain_async(
        self,
        bank_id: str,
        content: str,
        context: str = "",
        event_date: datetime | None = None,
        document_id: str | None = None,
        fact_type_override: str | None = None,
        confidence_score: float | None = None,
        *,
        request_context: "RequestContext",
    ) -> list[str]:
        """
        Store content as memory units with temporal and semantic links (ASYNC version).

        This is a convenience wrapper around retain_batch_async for a single content item.

        Args:
            bank_id: Unique identifier for the bank
            content: Text content to store
            context: Context about when/why this memory was formed
            event_date: When the event occurred (defaults to now)
            document_id: Optional document ID for tracking (always upserts if document already exists)
            fact_type_override: Override fact type ('world', 'experience', 'opinion')
            confidence_score: Confidence score for opinions (0.0 to 1.0)
            request_context: Request context for authentication.

        Returns:
            List of created unit IDs
        """
        # Build content dict
        content_dict: RetainContentDict = {"content": content, "context": context}  # type: ignore[typeddict-item] - building incrementally
        if event_date:
            content_dict["event_date"] = event_date
        if document_id:
            content_dict["document_id"] = document_id

        # Use retain_batch_async with a single item (avoids code duplication)
        result = await self.retain_batch_async(
            bank_id=bank_id,
            contents=[content_dict],
            request_context=request_context,
            fact_type_override=fact_type_override,
            confidence_score=confidence_score,
        )

        # Return the first (and only) list of unit IDs
        return result[0] if result else []

    async def retain_batch_async(
        self,
        bank_id: str,
        contents: list[RetainContentDict],
        *,
        request_context: "RequestContext",
        document_id: str | None = None,
        fact_type_override: str | None = None,
        confidence_score: float | None = None,
    ) -> list[list[str]]:
        """
        Store multiple content items as memory units in ONE batch operation.

        This is MUCH more efficient than calling retain_async multiple times:
        - Extracts facts from all contents in parallel
        - Generates ALL embeddings in ONE batch
        - Does ALL database operations in ONE transaction
        - Automatically chunks large batches to prevent timeouts

        Args:
            bank_id: Unique identifier for the bank
            contents: List of dicts with keys:
                - "content" (required): Text content to store
                - "context" (optional): Context about the memory
                - "event_date" (optional): When the event occurred
                - "document_id" (optional): Document ID for this specific content item
            document_id: **DEPRECATED** - Use "document_id" key in each content dict instead.
                        Applies the same document_id to ALL content items that don't specify their own.
            fact_type_override: Override fact type for all facts ('world', 'experience', 'opinion')
            confidence_score: Confidence score for opinions (0.0 to 1.0)

        Returns:
            List of lists of unit IDs (one list per content item)

        Example (new style - per-content document_id):
            unit_ids = await memory.retain_batch_async(
                bank_id="user123",
                contents=[
                    {"content": "Alice works at Google", "document_id": "doc1"},
                    {"content": "Bob loves Python", "document_id": "doc2"},
                    {"content": "More about Alice", "document_id": "doc1"},
                ]
            )
            # Returns: [["unit-id-1"], ["unit-id-2"], ["unit-id-3"]]

        Example (deprecated style - batch-level document_id):
            unit_ids = await memory.retain_batch_async(
                bank_id="user123",
                contents=[
                    {"content": "Alice works at Google"},
                    {"content": "Bob loves Python"},
                ],
                document_id="meeting-2024-01-15"
            )
            # Returns: [["unit-id-1"], ["unit-id-2"]]
        """
        start_time = time.time()

        if not contents:
            return []

        # Authenticate tenant and set schema in context (for fq_table())
        await self._authenticate_tenant(request_context)

        # Validate operation if validator is configured
        contents_copy = [dict(c) for c in contents]  # Convert TypedDict to regular dict for extension
        if self._operation_validator:
            from hindsight_api.extensions import RetainContext

            ctx = RetainContext(
                bank_id=bank_id,
                contents=contents_copy,
                request_context=request_context,
                document_id=document_id,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
            )
            await self._validate_operation(self._operation_validator.validate_retain(ctx))

        # Apply batch-level document_id to contents that don't have their own (backwards compatibility)
        if document_id:
            for item in contents:
                if "document_id" not in item:
                    item["document_id"] = document_id

        # Auto-chunk large batches by character count to avoid timeouts and memory issues
        # Calculate total character count
        total_chars = sum(len(item.get("content", "")) for item in contents)

        CHARS_PER_BATCH = 600_000

        if total_chars > CHARS_PER_BATCH:
            # Split into smaller batches based on character count
            logger.info(
                f"Large batch detected ({total_chars:,} chars from {len(contents)} items). Splitting into sub-batches of ~{CHARS_PER_BATCH:,} chars each..."
            )

            sub_batches = []
            current_batch = []
            current_batch_chars = 0

            for item in contents:
                item_chars = len(item.get("content", ""))

                # If adding this item would exceed the limit, start a new batch
                # (unless current batch is empty - then we must include it even if it's large)
                if current_batch and current_batch_chars + item_chars > CHARS_PER_BATCH:
                    sub_batches.append(current_batch)
                    current_batch = [item]
                    current_batch_chars = item_chars
                else:
                    current_batch.append(item)
                    current_batch_chars += item_chars

            # Add the last batch
            if current_batch:
                sub_batches.append(current_batch)

            logger.info(f"Split into {len(sub_batches)} sub-batches: {[len(b) for b in sub_batches]} items each")

            # Process each sub-batch using internal method (skip chunking check)
            all_results = []
            for i, sub_batch in enumerate(sub_batches, 1):
                sub_batch_chars = sum(len(item.get("content", "")) for item in sub_batch)
                logger.info(
                    f"Processing sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_chars:,} chars"
                )

                sub_results = await self._retain_batch_async_internal(
                    bank_id=bank_id,
                    contents=sub_batch,
                    document_id=document_id,
                    is_first_batch=i == 1,  # Only upsert on first batch
                    fact_type_override=fact_type_override,
                    confidence_score=confidence_score,
                )
                all_results.extend(sub_results)

            total_time = time.time() - start_time
            logger.info(
                f"RETAIN_BATCH_ASYNC (chunked) COMPLETE: {len(all_results)} results from {len(contents)} contents in {total_time:.3f}s"
            )
            result = all_results
        else:
            # Small batch - use internal method directly
            result = await self._retain_batch_async_internal(
                bank_id=bank_id,
                contents=contents,
                document_id=document_id,
                is_first_batch=True,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
            )

        # Call post-operation hook if validator is configured
        if self._operation_validator:
            from hindsight_api.extensions import RetainResult

            result_ctx = RetainResult(
                bank_id=bank_id,
                contents=contents_copy,
                request_context=request_context,
                document_id=document_id,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
                unit_ids=result,
                success=True,
                error=None,
            )
            try:
                await self._operation_validator.on_retain_complete(result_ctx)
            except Exception as e:
                logger.warning(f"Post-retain hook error (non-fatal): {e}")

        return result

    async def _retain_batch_async_internal(
        self,
        bank_id: str,
        contents: list[RetainContentDict],
        document_id: str | None = None,
        is_first_batch: bool = True,
        fact_type_override: str | None = None,
        confidence_score: float | None = None,
    ) -> list[list[str]]:
        """
        Internal method for batch processing without chunking logic.

        Assumes contents are already appropriately sized (< 50k chars).
        Called by retain_batch_async after chunking large batches.

        Uses semaphore for backpressure to limit concurrent retains.

        Args:
            bank_id: Unique identifier for the bank
            contents: List of dicts with content, context, event_date
            document_id: Optional document ID (always upserts if exists)
            is_first_batch: Whether this is the first batch (for chunked operations, only delete on first batch)
            fact_type_override: Override fact type for all facts
            confidence_score: Confidence score for opinions
        """
        # Backpressure: limit concurrent retains to prevent database contention
        async with self._put_semaphore:
            # Use the new modular orchestrator
            from .retain import orchestrator

            pool = await self._get_pool()
            return await orchestrator.retain_batch(
                pool=pool,
                embeddings_model=self.embeddings,
                llm_config=self._llm_config,
                entity_resolver=self.entity_resolver,
                task_backend=self._task_backend,
                format_date_fn=self._format_readable_date,
                duplicate_checker_fn=self._find_duplicate_facts_batch,
                bank_id=bank_id,
                contents_dicts=contents,
                document_id=document_id,
                is_first_batch=is_first_batch,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
            )

    def recall(
        self,
        bank_id: str,
        query: str,
        fact_type: str,
        budget: Budget = Budget.MID,
        max_tokens: int = 4096,
        enable_trace: bool = False,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        """
        Recall memories using 4-way parallel retrieval (synchronous wrapper).

        This is a synchronous wrapper around recall_async() for convenience.
        For best performance, use recall_async() directly.

        Args:
            bank_id: bank ID to recall for
            query: Recall query
            fact_type: Required filter for fact type ('world', 'experience', or 'opinion')
            budget: Budget level for graph traversal (low=100, mid=300, high=600 units)
            max_tokens: Maximum tokens to return (counts only 'text' field, default 4096)
            enable_trace: If True, returns detailed trace object

        Returns:
            Tuple of (results, trace)
        """
        # Run async version synchronously - deprecated sync method, passing None for request_context
        from hindsight_api.models import RequestContext

        return asyncio.run(
            self.recall_async(
                bank_id,
                query,
                budget=budget,
                max_tokens=max_tokens,
                enable_trace=enable_trace,
                fact_type=[fact_type],
                request_context=RequestContext(),
            )
        )

    async def recall_async(
        self,
        bank_id: str,
        query: str,
        *,
        budget: Budget | None = None,
        max_tokens: int = 4096,
        enable_trace: bool = False,
        fact_type: list[str] | None = None,
        question_date: datetime | None = None,
        include_entities: bool = False,
        max_entity_tokens: int = 500,
        include_chunks: bool = False,
        max_chunk_tokens: int = 8192,
        boost_by_usefulness: bool = False,
        usefulness_weight: float = 0.3,
        min_usefulness: float = 0.0,
        request_context: "RequestContext",
    ) -> RecallResultModel:
        """
        Recall memories using N*4-way parallel retrieval (N fact types × 4 retrieval methods).

        This implements the core RECALL operation:
        1. Retrieval: For each fact type, run 4 parallel retrievals (semantic vector, BM25 keyword, graph activation, temporal graph)
        2. Merge: Combine using Reciprocal Rank Fusion (RRF)
        3. Rerank: Score using selected reranker (heuristic or cross-encoder)
        4. Diversify: Apply MMR for diversity
        5. Token Filter: Return results up to max_tokens budget

        Args:
            bank_id: bank ID to recall for
            query: Recall query
            fact_type: List of fact types to recall (e.g., ['world', 'experience'])
            budget: Budget level for graph traversal (low=100, mid=300, high=600 units)
            max_tokens: Maximum tokens to return (counts only 'text' field, default 4096)
                       Results are returned until token budget is reached, stopping before
                       including a fact that would exceed the limit
            enable_trace: Whether to return trace for debugging (deprecated)
            question_date: Optional date when question was asked (for temporal filtering)
            include_entities: Whether to include entity observations in the response
            max_entity_tokens: Maximum tokens for entity observations (default 500)
            include_chunks: Whether to include raw chunks in the response
            max_chunk_tokens: Maximum tokens for chunks (default 8192)

        Returns:
            RecallResultModel containing:
            - results: List of MemoryFact objects
            - trace: Optional trace information for debugging
            - entities: Optional dict of entity states (if include_entities=True)
            - chunks: Optional dict of chunks (if include_chunks=True)
        """
        # Authenticate tenant and set schema in context (for fq_table())
        await self._authenticate_tenant(request_context)

        # Default to all fact types if not specified
        if fact_type is None:
            fact_type = list(VALID_RECALL_FACT_TYPES)

        # Validate fact types early
        invalid_types = set(fact_type) - VALID_RECALL_FACT_TYPES
        if invalid_types:
            raise ValueError(
                f"Invalid fact type(s): {', '.join(sorted(invalid_types))}. "
                f"Must be one of: {', '.join(sorted(VALID_RECALL_FACT_TYPES))}"
            )

        # Validate operation if validator is configured
        if self._operation_validator:
            from hindsight_api.extensions import RecallContext

            ctx = RecallContext(
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                budget=budget,
                max_tokens=max_tokens,
                enable_trace=enable_trace,
                fact_types=list(fact_type),
                question_date=question_date,
                include_entities=include_entities,
                max_entity_tokens=max_entity_tokens,
                include_chunks=include_chunks,
                max_chunk_tokens=max_chunk_tokens,
            )
            await self._validate_operation(self._operation_validator.validate_recall(ctx))

        # Map budget enum to thinking_budget number (default to MID if None)
        budget_mapping = {Budget.LOW: 100, Budget.MID: 300, Budget.HIGH: 1000}
        effective_budget = budget if budget is not None else Budget.MID
        thinking_budget = budget_mapping[effective_budget]

        # Backpressure: limit concurrent recalls to prevent overwhelming the database
        result = None
        error_msg = None
        async with self._search_semaphore:
            # Retry loop for connection errors
            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    result = await self._search_with_retries(
                        bank_id,
                        query,
                        fact_type,
                        thinking_budget,
                        max_tokens,
                        enable_trace,
                        question_date,
                        include_entities,
                        max_entity_tokens,
                        include_chunks,
                        max_chunk_tokens,
                        boost_by_usefulness,
                        usefulness_weight,
                        min_usefulness,
                        request_context,
                    )
                    break  # Success - exit retry loop
                except Exception as e:
                    # Check if it's a connection error
                    is_connection_error = (
                        isinstance(e, asyncpg.TooManyConnectionsError)
                        or isinstance(e, asyncpg.CannotConnectNowError)
                        or (isinstance(e, asyncpg.PostgresError) and "connection" in str(e).lower())
                    )

                    if is_connection_error and attempt < max_retries:
                        # Wait with exponential backoff before retry
                        wait_time = 0.5 * (2**attempt)  # 0.5s, 1s, 2s
                        logger.warning(
                            f"Connection error on search attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        # Not a connection error or out of retries - call post-hook and raise
                        error_msg = str(e)
                        if self._operation_validator:
                            from hindsight_api.extensions.operation_validator import RecallResult

                            result_ctx = RecallResult(
                                bank_id=bank_id,
                                query=query,
                                request_context=request_context,
                                budget=budget,
                                max_tokens=max_tokens,
                                enable_trace=enable_trace,
                                fact_types=list(fact_type),
                                question_date=question_date,
                                include_entities=include_entities,
                                max_entity_tokens=max_entity_tokens,
                                include_chunks=include_chunks,
                                max_chunk_tokens=max_chunk_tokens,
                                result=None,
                                success=False,
                                error=error_msg,
                            )
                            try:
                                await self._operation_validator.on_recall_complete(result_ctx)
                            except Exception as hook_err:
                                logger.warning(f"Post-recall hook error (non-fatal): {hook_err}")
                        raise
            else:
                # Exceeded max retries
                error_msg = "Exceeded maximum retries for search due to connection errors."
                if self._operation_validator:
                    from hindsight_api.extensions.operation_validator import RecallResult

                    result_ctx = RecallResult(
                        bank_id=bank_id,
                        query=query,
                        request_context=request_context,
                        budget=budget,
                        max_tokens=max_tokens,
                        enable_trace=enable_trace,
                        fact_types=list(fact_type),
                        question_date=question_date,
                        include_entities=include_entities,
                        max_entity_tokens=max_entity_tokens,
                        include_chunks=include_chunks,
                        max_chunk_tokens=max_chunk_tokens,
                        result=None,
                        success=False,
                        error=error_msg,
                    )
                    try:
                        await self._operation_validator.on_recall_complete(result_ctx)
                    except Exception as hook_err:
                        logger.warning(f"Post-recall hook error (non-fatal): {hook_err}")
                raise Exception(error_msg)

        # Call post-operation hook for success
        if self._operation_validator and result is not None:
            from hindsight_api.extensions.operation_validator import RecallResult

            result_ctx = RecallResult(
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                budget=budget,
                max_tokens=max_tokens,
                enable_trace=enable_trace,
                fact_types=list(fact_type),
                question_date=question_date,
                include_entities=include_entities,
                max_entity_tokens=max_entity_tokens,
                include_chunks=include_chunks,
                max_chunk_tokens=max_chunk_tokens,
                result=result,
                success=True,
                error=None,
            )
            try:
                await self._operation_validator.on_recall_complete(result_ctx)
            except Exception as e:
                logger.warning(f"Post-recall hook error (non-fatal): {e}")

        return result

    async def _search_with_retries(
        self,
        bank_id: str,
        query: str,
        fact_type: list[str],
        thinking_budget: int,
        max_tokens: int,
        enable_trace: bool,
        question_date: datetime | None = None,
        include_entities: bool = False,
        max_entity_tokens: int = 500,
        include_chunks: bool = False,
        max_chunk_tokens: int = 8192,
        boost_by_usefulness: bool = False,
        usefulness_weight: float = 0.3,
        min_usefulness: float = 0.0,
        request_context: "RequestContext" = None,
    ) -> RecallResultModel:
        """
        Search implementation with modular retrieval and reranking.

        Architecture:
        1. Retrieval: 4-way parallel (semantic, keyword, graph, temporal graph)
        2. Merge: RRF to combine ranked lists
        3. Reranking: Pluggable strategy (heuristic or cross-encoder)
        4. Diversity: MMR with λ=0.5
        5. Token Filter: Limit results to max_tokens budget

        Args:
            bank_id: bank IDentifier
            query: Search query
            fact_type: Type of facts to search
            thinking_budget: Nodes to explore in graph traversal
            max_tokens: Maximum tokens to return (counts only 'text' field)
            enable_trace: Whether to return search trace (deprecated)
            include_entities: Whether to include entity observations
            max_entity_tokens: Maximum tokens for entity observations
            include_chunks: Whether to include raw chunks
            max_chunk_tokens: Maximum tokens for chunks

        Returns:
            RecallResultModel with results, trace, optional entities, and optional chunks
        """
        # Initialize tracer if requested
        from .search.tracer import SearchTracer

        tracer = SearchTracer(query, thinking_budget, max_tokens) if enable_trace else None
        if tracer:
            tracer.start()

        pool = await self._get_pool()
        recall_start = time.time()

        # Buffer logs for clean output in concurrent scenarios
        recall_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
        log_buffer = []
        log_buffer.append(
            f"[RECALL {recall_id}] Query: '{query[:50]}...' (budget={thinking_budget}, max_tokens={max_tokens})"
        )

        try:
            # Step 1: Generate query embedding (for semantic search)
            step_start = time.time()
            query_embedding = embedding_utils.generate_embedding(self.embeddings, query)
            step_duration = time.time() - step_start
            log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")

            if tracer:
                tracer.record_query_embedding(query_embedding)
                tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Step 2: N*4-Way Parallel Retrieval (N fact types × 4 retrieval methods)
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            from .search.retrieval import retrieve_parallel

            # Track each retrieval start time
            retrieval_start = time.time()

            # Run retrieval for each fact type in parallel
            retrieval_tasks = [
                retrieve_parallel(
                    pool, query, query_embedding_str, bank_id, ft, thinking_budget, question_date, self.query_analyzer
                )
                for ft in fact_type
            ]
            all_retrievals = await asyncio.gather(*retrieval_tasks)

            # Combine all results from all fact types and aggregate timings
            semantic_results = []
            bm25_results = []
            graph_results = []
            temporal_results = []
            aggregated_timings = {"semantic": 0.0, "bm25": 0.0, "graph": 0.0, "temporal": 0.0}

            detected_temporal_constraint = None
            for idx, retrieval_result in enumerate(all_retrievals):
                # Log fact types in this retrieval batch
                ft_name = fact_type[idx] if idx < len(fact_type) else "unknown"
                logger.debug(
                    f"[RECALL {recall_id}] Fact type '{ft_name}': semantic={len(retrieval_result.semantic)}, bm25={len(retrieval_result.bm25)}, graph={len(retrieval_result.graph)}, temporal={len(retrieval_result.temporal) if retrieval_result.temporal else 0}"
                )

                semantic_results.extend(retrieval_result.semantic)
                bm25_results.extend(retrieval_result.bm25)
                graph_results.extend(retrieval_result.graph)
                if retrieval_result.temporal:
                    temporal_results.extend(retrieval_result.temporal)
                # Track max timing for each method (since they run in parallel across fact types)
                for method, duration in retrieval_result.timings.items():
                    aggregated_timings[method] = max(aggregated_timings.get(method, 0.0), duration)
                # Capture temporal constraint (same across all fact types)
                if retrieval_result.temporal_constraint:
                    detected_temporal_constraint = retrieval_result.temporal_constraint

            # If no temporal results from any fact type, set to None
            if not temporal_results:
                temporal_results = None

            # Sort combined results by score (descending) so higher-scored results
            # get better ranks in the trace, regardless of fact type
            semantic_results.sort(key=lambda r: r.similarity if hasattr(r, "similarity") else 0, reverse=True)
            bm25_results.sort(key=lambda r: r.bm25_score if hasattr(r, "bm25_score") else 0, reverse=True)
            graph_results.sort(key=lambda r: r.activation if hasattr(r, "activation") else 0, reverse=True)
            if temporal_results:
                temporal_results.sort(
                    key=lambda r: r.combined_score if hasattr(r, "combined_score") else 0, reverse=True
                )

            retrieval_duration = time.time() - retrieval_start

            step_duration = time.time() - step_start
            total_retrievals = len(fact_type) * (4 if temporal_results else 3)
            # Format per-method timings
            timing_parts = [
                f"semantic={len(semantic_results)}({aggregated_timings['semantic']:.3f}s)",
                f"bm25={len(bm25_results)}({aggregated_timings['bm25']:.3f}s)",
                f"graph={len(graph_results)}({aggregated_timings['graph']:.3f}s)",
            ]
            temporal_info = ""
            if detected_temporal_constraint:
                start_dt, end_dt = detected_temporal_constraint
                temporal_count = len(temporal_results) if temporal_results else 0
                timing_parts.append(f"temporal={temporal_count}({aggregated_timings['temporal']:.3f}s)")
                temporal_info = f" | temporal_range={start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
            log_buffer.append(
                f"  [2] {total_retrievals}-way retrieval ({len(fact_type)} fact_types): {', '.join(timing_parts)} in {step_duration:.3f}s{temporal_info}"
            )

            # Record retrieval results for tracer - per fact type
            if tracer:
                # Convert RetrievalResult to old tuple format for tracer
                def to_tuple_format(results):
                    return [(r.id, r.__dict__) for r in results]

                # Add retrieval results per fact type (to show parallel execution in UI)
                for idx, rr in enumerate(all_retrievals):
                    ft_name = fact_type[idx] if idx < len(fact_type) else "unknown"

                    # Add semantic retrieval results for this fact type
                    tracer.add_retrieval_results(
                        method_name="semantic",
                        results=to_tuple_format(rr.semantic),
                        duration_seconds=rr.timings.get("semantic", 0.0),
                        score_field="similarity",
                        metadata={"limit": thinking_budget},
                        fact_type=ft_name,
                    )

                    # Add BM25 retrieval results for this fact type
                    tracer.add_retrieval_results(
                        method_name="bm25",
                        results=to_tuple_format(rr.bm25),
                        duration_seconds=rr.timings.get("bm25", 0.0),
                        score_field="bm25_score",
                        metadata={"limit": thinking_budget},
                        fact_type=ft_name,
                    )

                    # Add graph retrieval results for this fact type
                    tracer.add_retrieval_results(
                        method_name="graph",
                        results=to_tuple_format(rr.graph),
                        duration_seconds=rr.timings.get("graph", 0.0),
                        score_field="activation",
                        metadata={"budget": thinking_budget},
                        fact_type=ft_name,
                    )

                    # Add temporal retrieval results for this fact type (even if empty, to show it ran)
                    if rr.temporal is not None:
                        tracer.add_retrieval_results(
                            method_name="temporal",
                            results=to_tuple_format(rr.temporal),
                            duration_seconds=rr.timings.get("temporal", 0.0),
                            score_field="temporal_score",
                            metadata={"budget": thinking_budget},
                            fact_type=ft_name,
                        )

                # Record entry points (from semantic results) for legacy graph view
                for rank, retrieval in enumerate(semantic_results[:10], start=1):  # Top 10 as entry points
                    tracer.add_entry_point(retrieval.id, retrieval.text, retrieval.similarity or 0.0, rank)

                tracer.add_phase_metric(
                    "parallel_retrieval",
                    step_duration,
                    {
                        "semantic_count": len(semantic_results),
                        "bm25_count": len(bm25_results),
                        "graph_count": len(graph_results),
                        "temporal_count": len(temporal_results) if temporal_results else 0,
                    },
                )

            # Step 3: Merge with RRF
            step_start = time.time()
            from .search.fusion import reciprocal_rank_fusion

            # Merge 3 or 4 result lists depending on temporal constraint
            if temporal_results:
                merged_candidates = reciprocal_rank_fusion(
                    [semantic_results, bm25_results, graph_results, temporal_results]
                )
            else:
                merged_candidates = reciprocal_rank_fusion([semantic_results, bm25_results, graph_results])

            step_duration = time.time() - step_start
            log_buffer.append(f"  [3] RRF merge: {len(merged_candidates)} unique candidates in {step_duration:.3f}s")

            if tracer:
                # Convert MergedCandidate to old tuple format for tracer
                tracer_merged = [
                    (mc.id, mc.retrieval.__dict__, {"rrf_score": mc.rrf_score, **mc.source_ranks})
                    for mc in merged_candidates
                ]
                tracer.add_rrf_merged(tracer_merged)
                tracer.add_phase_metric("rrf_merge", step_duration, {"candidates_merged": len(merged_candidates)})

            # Step 4: Rerank using cross-encoder (MergedCandidate -> ScoredResult)
            step_start = time.time()
            reranker_instance = self._cross_encoder_reranker

            # Ensure reranker is initialized (for lazy initialization mode)
            await reranker_instance.ensure_initialized()

            # Rerank using cross-encoder
            scored_results = reranker_instance.rerank(query, merged_candidates)

            step_duration = time.time() - step_start
            log_buffer.append(f"  [4] Reranking: {len(scored_results)} candidates scored in {step_duration:.3f}s")

            # Step 4.5: Combine cross-encoder score with retrieval signals
            # This preserves retrieval work (RRF, temporal, recency) instead of pure cross-encoder ranking
            if scored_results:
                # Normalize RRF scores to [0, 1] range using min-max normalization
                rrf_scores = [sr.candidate.rrf_score for sr in scored_results]
                max_rrf = max(rrf_scores) if rrf_scores else 0.0
                min_rrf = min(rrf_scores) if rrf_scores else 0.0
                rrf_range = max_rrf - min_rrf  # Don't force to 1.0, let fallback handle it

                # Calculate recency based on occurred_start (more recent = higher score)
                now = utcnow()
                for sr in scored_results:
                    # Normalize RRF score (0-1 range, 0.5 if all same)
                    if rrf_range > 0:
                        sr.rrf_normalized = (sr.candidate.rrf_score - min_rrf) / rrf_range
                    else:
                        # All RRF scores are the same, use neutral value
                        sr.rrf_normalized = 0.5

                    # Calculate recency (decay over 365 days, minimum 0.1)
                    sr.recency = 0.5  # default for missing dates
                    if sr.retrieval.occurred_start:
                        occurred = sr.retrieval.occurred_start
                        if hasattr(occurred, "tzinfo") and occurred.tzinfo is None:
                            occurred = occurred.replace(tzinfo=UTC)
                        days_ago = (now - occurred).total_seconds() / 86400
                        sr.recency = max(0.1, 1.0 - (days_ago / 365))  # Linear decay over 1 year

                    # Get temporal proximity if available (already 0-1)
                    sr.temporal = (
                        sr.retrieval.temporal_proximity if sr.retrieval.temporal_proximity is not None else 0.5
                    )

                    # Weighted combination
                    # Cross-encoder: 60% (semantic relevance)
                    # RRF: 20% (retrieval consensus)
                    # Temporal proximity: 10% (time relevance for temporal queries)
                    # Recency: 10% (prefer recent facts)
                    sr.combined_score = (
                        0.6 * sr.cross_encoder_score_normalized
                        + 0.2 * sr.rrf_normalized
                        + 0.1 * sr.temporal
                        + 0.1 * sr.recency
                    )
                    sr.weight = sr.combined_score  # Update weight for final ranking

                # Re-sort by combined score
                scored_results.sort(key=lambda x: x.weight, reverse=True)
                log_buffer.append(
                    "  [4.6] Combined scoring: cross_encoder(0.6) + rrf(0.2) + temporal(0.1) + recency(0.1)"
                )

            # Step 4.7: Apply usefulness boost if enabled
            if boost_by_usefulness and scored_results:
                # Get usefulness scores for all candidate facts
                # Pass query_embedding for query-context aware scoring
                fact_ids = [sr.id for sr in scored_results]
                async with acquire_with_retry(pool) as usefulness_conn:
                    usefulness_scores = await self._get_usefulness_scores_for_facts(
                        usefulness_conn, bank_id, fact_ids, query_embedding=query_embedding
                    )

                # Filter by min_usefulness
                if min_usefulness > 0.0:
                    original_count = len(scored_results)
                    scored_results = [sr for sr in scored_results if usefulness_scores.get(sr.id, 0.5) >= min_usefulness]
                    log_buffer.append(
                        f"  [4.7] Usefulness filter: {original_count} -> {len(scored_results)} results "
                        f"(min_usefulness={min_usefulness})"
                    )

                # Apply usefulness boost to weight
                # New weight = (1 - usefulness_weight) * old_weight + usefulness_weight * usefulness_score
                for sr in scored_results:
                    usefulness = usefulness_scores.get(sr.id, 0.5)
                    sr.weight = (1 - usefulness_weight) * sr.weight + usefulness_weight * usefulness

                # Re-sort by updated weight
                scored_results.sort(key=lambda x: x.weight, reverse=True)
                log_buffer.append(
                    f"  [4.7] Usefulness boost applied: weight={usefulness_weight}, "
                    f"boosted {len(scored_results)} results"
                )

            # Add reranked results to tracer AFTER combined scoring (so normalized values are included)
            if tracer:
                results_dict = [sr.to_dict() for sr in scored_results]
                tracer_merged = [
                    (mc.id, mc.retrieval.__dict__, {"rrf_score": mc.rrf_score, **mc.source_ranks})
                    for mc in merged_candidates
                ]
                tracer.add_reranked(results_dict, tracer_merged)
                tracer.add_phase_metric(
                    "reranking",
                    step_duration,
                    {"reranker_type": "cross-encoder", "candidates_reranked": len(scored_results)},
                )

            # Step 5: Truncate to thinking_budget * 2 for token filtering
            rerank_limit = thinking_budget * 2
            top_scored = scored_results[:rerank_limit]
            log_buffer.append(f"  [5] Truncated to top {len(top_scored)} results")

            # Step 6: Token budget filtering
            step_start = time.time()

            # Convert to dict for token filtering (backward compatibility)
            top_dicts = [sr.to_dict() for sr in top_scored]
            filtered_dicts, total_tokens = self._filter_by_token_budget(top_dicts, max_tokens)

            # Convert back to list of IDs and filter scored_results
            filtered_ids = {d["id"] for d in filtered_dicts}
            top_scored = [sr for sr in top_scored if sr.id in filtered_ids]

            step_duration = time.time() - step_start
            log_buffer.append(
                f"  [6] Token filtering: {len(top_scored)} results, {total_tokens}/{max_tokens} tokens in {step_duration:.3f}s"
            )

            if tracer:
                tracer.add_phase_metric(
                    "token_filtering",
                    step_duration,
                    {"results_selected": len(top_scored), "tokens_used": total_tokens, "max_tokens": max_tokens},
                )

            # Record visits for all retrieved nodes
            if tracer:
                for sr in scored_results:
                    tracer.visit_node(
                        node_id=sr.id,
                        text=sr.retrieval.text,
                        context=sr.retrieval.context or "",
                        event_date=sr.retrieval.occurred_start,
                        access_count=sr.retrieval.access_count,
                        is_entry_point=(sr.id in [ep.node_id for ep in tracer.entry_points]),
                        parent_node_id=None,  # In parallel retrieval, there's no clear parent
                        link_type=None,
                        link_weight=None,
                        activation=sr.candidate.rrf_score,  # Use RRF score as activation
                        semantic_similarity=sr.retrieval.similarity or 0.0,
                        recency=sr.recency,
                        frequency=0.0,
                        final_weight=sr.weight,
                    )

            # Step 8: Queue access count updates for visited nodes
            visited_ids = list(set([sr.id for sr in scored_results[:50]]))  # Top 50
            if visited_ids:
                await self._task_backend.submit_task({"type": "access_count_update", "node_ids": visited_ids})
                log_buffer.append(f"  [7] Queued access count updates for {len(visited_ids)} nodes")

            # Log fact_type distribution in results
            fact_type_counts = {}
            for sr in top_scored:
                ft = sr.retrieval.fact_type
                fact_type_counts[ft] = fact_type_counts.get(ft, 0) + 1

            fact_type_summary = ", ".join([f"{ft}={count}" for ft, count in sorted(fact_type_counts.items())])

            # Convert ScoredResult to dicts with ISO datetime strings
            top_results_dicts = []
            for sr in top_scored:
                result_dict = sr.to_dict()
                # Convert datetime objects to ISO strings for JSON serialization
                if result_dict.get("occurred_start"):
                    occurred_start = result_dict["occurred_start"]
                    result_dict["occurred_start"] = (
                        occurred_start.isoformat() if hasattr(occurred_start, "isoformat") else occurred_start
                    )
                if result_dict.get("occurred_end"):
                    occurred_end = result_dict["occurred_end"]
                    result_dict["occurred_end"] = (
                        occurred_end.isoformat() if hasattr(occurred_end, "isoformat") else occurred_end
                    )
                if result_dict.get("mentioned_at"):
                    mentioned_at = result_dict["mentioned_at"]
                    result_dict["mentioned_at"] = (
                        mentioned_at.isoformat() if hasattr(mentioned_at, "isoformat") else mentioned_at
                    )
                top_results_dicts.append(result_dict)

            # Get entities for each fact if include_entities is requested
            fact_entity_map = {}  # unit_id -> list of (entity_id, entity_name)
            if include_entities and top_scored:
                unit_ids = [uuid.UUID(sr.id) for sr in top_scored]
                if unit_ids:
                    async with acquire_with_retry(pool) as entity_conn:
                        entity_rows = await entity_conn.fetch(
                            f"""
                            SELECT ue.unit_id, e.id as entity_id, e.canonical_name
                            FROM {fq_table("unit_entities")} ue
                            JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                            WHERE ue.unit_id = ANY($1::uuid[])
                            """,
                            unit_ids,
                        )
                        for row in entity_rows:
                            unit_id = str(row["unit_id"])
                            if unit_id not in fact_entity_map:
                                fact_entity_map[unit_id] = []
                            fact_entity_map[unit_id].append(
                                {"entity_id": str(row["entity_id"]), "canonical_name": row["canonical_name"]}
                            )

            # Convert results to MemoryFact objects
            memory_facts = []
            for result_dict in top_results_dicts:
                result_id = str(result_dict.get("id"))
                # Get entity names for this fact
                entity_names = None
                if include_entities and result_id in fact_entity_map:
                    entity_names = [e["canonical_name"] for e in fact_entity_map[result_id]]

                memory_facts.append(
                    MemoryFact(
                        id=result_id,
                        text=result_dict.get("text"),
                        fact_type=result_dict.get("fact_type", "world"),
                        entities=entity_names,
                        context=result_dict.get("context"),
                        occurred_start=result_dict.get("occurred_start"),
                        occurred_end=result_dict.get("occurred_end"),
                        mentioned_at=result_dict.get("mentioned_at"),
                        document_id=result_dict.get("document_id"),
                        chunk_id=result_dict.get("chunk_id"),
                    )
                )

            # Fetch entity observations if requested
            entities_dict = None
            total_entity_tokens = 0
            total_chunk_tokens = 0
            if include_entities and fact_entity_map:
                # Collect unique entities in order of fact relevance (preserving order from top_scored)
                # Use a list to maintain order, but track seen entities to avoid duplicates
                entities_ordered = []  # list of (entity_id, entity_name) tuples
                seen_entity_ids = set()

                # Iterate through facts in relevance order
                for sr in top_scored:
                    unit_id = sr.id
                    if unit_id in fact_entity_map:
                        for entity in fact_entity_map[unit_id]:
                            entity_id = entity["entity_id"]
                            entity_name = entity["canonical_name"]
                            if entity_id not in seen_entity_ids:
                                entities_ordered.append((entity_id, entity_name))
                                seen_entity_ids.add(entity_id)

                # Fetch observations for each entity (respect token budget, in order)
                entities_dict = {}
                encoding = _get_tiktoken_encoding()

                for entity_id, entity_name in entities_ordered:
                    if total_entity_tokens >= max_entity_tokens:
                        break

                    observations = await self.get_entity_observations(
                        bank_id, entity_id, limit=5, request_context=request_context
                    )

                    # Calculate tokens for this entity's observations
                    entity_tokens = 0
                    included_observations = []
                    for obs in observations:
                        obs_tokens = len(encoding.encode(obs.text))
                        if total_entity_tokens + entity_tokens + obs_tokens <= max_entity_tokens:
                            included_observations.append(obs)
                            entity_tokens += obs_tokens
                        else:
                            break

                    if included_observations:
                        entities_dict[entity_name] = EntityState(
                            entity_id=entity_id, canonical_name=entity_name, observations=included_observations
                        )
                        total_entity_tokens += entity_tokens

            # Fetch chunks if requested
            chunks_dict = None
            if include_chunks and top_scored:
                from .response_models import ChunkInfo

                # Collect chunk_ids in order of fact relevance (preserving order from top_scored)
                # Use a list to maintain order, but track seen chunks to avoid duplicates
                chunk_ids_ordered = []
                seen_chunk_ids = set()
                for sr in top_scored:
                    chunk_id = sr.retrieval.chunk_id
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        chunk_ids_ordered.append(chunk_id)
                        seen_chunk_ids.add(chunk_id)

                if chunk_ids_ordered:
                    # Fetch chunk data from database using chunk_ids (no ORDER BY to preserve input order)
                    async with acquire_with_retry(pool) as conn:
                        chunks_rows = await conn.fetch(
                            f"""
                            SELECT chunk_id, chunk_text, chunk_index
                            FROM {fq_table("chunks")}
                            WHERE chunk_id = ANY($1::text[])
                            """,
                            chunk_ids_ordered,
                        )

                    # Create a lookup dict for fast access
                    chunks_lookup = {row["chunk_id"]: row for row in chunks_rows}

                    # Apply token limit and build chunks_dict in the order of chunk_ids_ordered
                    chunks_dict = {}
                    encoding = _get_tiktoken_encoding()

                    for chunk_id in chunk_ids_ordered:
                        if chunk_id not in chunks_lookup:
                            continue

                        row = chunks_lookup[chunk_id]
                        chunk_text = row["chunk_text"]
                        chunk_tokens = len(encoding.encode(chunk_text))

                        # Check if adding this chunk would exceed the limit
                        if total_chunk_tokens + chunk_tokens > max_chunk_tokens:
                            # Truncate the chunk to fit within the remaining budget
                            remaining_tokens = max_chunk_tokens - total_chunk_tokens
                            if remaining_tokens > 0:
                                # Truncate to remaining tokens
                                truncated_text = encoding.decode(encoding.encode(chunk_text)[:remaining_tokens])
                                chunks_dict[chunk_id] = ChunkInfo(
                                    chunk_text=truncated_text, chunk_index=row["chunk_index"], truncated=True
                                )
                                total_chunk_tokens = max_chunk_tokens
                            # Stop adding more chunks once we hit the limit
                            break
                        else:
                            chunks_dict[chunk_id] = ChunkInfo(
                                chunk_text=chunk_text, chunk_index=row["chunk_index"], truncated=False
                            )
                            total_chunk_tokens += chunk_tokens

            # Finalize trace if enabled
            trace_dict = None
            if tracer:
                trace = tracer.finalize(top_results_dicts)
                trace_dict = trace.to_dict() if trace else None

            # Log final recall stats
            total_time = time.time() - recall_start
            num_chunks = len(chunks_dict) if chunks_dict else 0
            num_entities = len(entities_dict) if entities_dict else 0
            log_buffer.append(
                f"[RECALL {recall_id}] Complete: {len(top_scored)} facts ({total_tokens} tok), {num_chunks} chunks ({total_chunk_tokens} tok), {num_entities} entities ({total_entity_tokens} tok) | {fact_type_summary} | {total_time:.3f}s"
            )
            logger.info("\n" + "\n".join(log_buffer))

            return RecallResultModel(results=memory_facts, trace=trace_dict, entities=entities_dict, chunks=chunks_dict)

        except Exception as e:
            log_buffer.append(f"[RECALL {recall_id}] ERROR after {time.time() - recall_start:.3f}s: {str(e)}")
            logger.error("\n" + "\n".join(log_buffer))
            raise Exception(f"Failed to search memories: {str(e)}")

    def _filter_by_token_budget(
        self, results: list[dict[str, Any]], max_tokens: int
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Filter results to fit within token budget.

        Counts tokens only for the 'text' field using tiktoken (cl100k_base encoding).
        Stops before including a fact that would exceed the budget.

        Args:
            results: List of search results
            max_tokens: Maximum tokens allowed

        Returns:
            Tuple of (filtered_results, total_tokens_used)
        """
        encoding = _get_tiktoken_encoding()

        filtered_results = []
        total_tokens = 0

        for result in results:
            text = result.get("text", "")
            text_tokens = len(encoding.encode(text))

            # Check if adding this result would exceed budget
            if total_tokens + text_tokens <= max_tokens:
                filtered_results.append(result)
                total_tokens += text_tokens
            else:
                # Stop before including a fact that would exceed limit
                break

        return filtered_results, total_tokens

    async def get_document(
        self,
        document_id: str,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Retrieve document metadata and statistics.

        Args:
            document_id: Document ID to retrieve
            bank_id: bank ID that owns the document
            request_context: Request context for authentication.

        Returns:
            Dictionary with document info or None if not found
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            doc = await conn.fetchrow(
                f"""
                SELECT d.id, d.bank_id, d.original_text, d.content_hash,
                       d.created_at, d.updated_at, COUNT(mu.id) as unit_count
                FROM {fq_table("documents")} d
                LEFT JOIN {fq_table("memory_units")} mu ON mu.document_id = d.id
                WHERE d.id = $1 AND d.bank_id = $2
                GROUP BY d.id, d.bank_id, d.original_text, d.content_hash, d.created_at, d.updated_at
                """,
                document_id,
                bank_id,
            )

            if not doc:
                return None

            return {
                "id": doc["id"],
                "bank_id": doc["bank_id"],
                "original_text": doc["original_text"],
                "content_hash": doc["content_hash"],
                "memory_unit_count": doc["unit_count"],
                "created_at": doc["created_at"].isoformat() if doc["created_at"] else None,
                "updated_at": doc["updated_at"].isoformat() if doc["updated_at"] else None,
            }

    async def delete_document(
        self,
        document_id: str,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Delete a document and all its associated memory units and links.

        Args:
            document_id: Document ID to delete
            bank_id: bank ID that owns the document
            request_context: Request context for authentication.

        Returns:
            Dictionary with counts of deleted items
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                # Count units before deletion
                units_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE document_id = $1", document_id
                )

                # Delete document (cascades to memory_units and all their links)
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id",
                    document_id,
                    bank_id,
                )

                return {"document_deleted": 1 if deleted else 0, "memory_units_deleted": units_count if deleted else 0}

    async def delete_memory_unit(
        self,
        unit_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Delete a single memory unit and all its associated links.

        Due to CASCADE DELETE constraints, this will automatically delete:
        - All links from this unit (memory_links where from_unit_id = unit_id)
        - All links to this unit (memory_links where to_unit_id = unit_id)
        - All entity associations (unit_entities where unit_id = unit_id)

        Args:
            unit_id: UUID of the memory unit to delete
            request_context: Request context for authentication.

        Returns:
            Dictionary with deletion result
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                # Delete the memory unit (cascades to links and associations)
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 RETURNING id", unit_id
                )

                return {
                    "success": deleted is not None,
                    "unit_id": str(deleted) if deleted else None,
                    "message": "Memory unit and all its links deleted successfully"
                    if deleted
                    else "Memory unit not found",
                }

    async def delete_bank(
        self,
        bank_id: str,
        fact_type: str | None = None,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Delete all data for a specific agent (multi-tenant cleanup).

        This is much more efficient than dropping all tables and allows
        multiple agents to coexist in the same database.

        Deletes (with CASCADE):
        - All memory units for this bank (optionally filtered by fact_type)
        - All entities for this bank (if deleting all memory units)
        - All associated links, unit-entity associations, and co-occurrences

        Args:
            bank_id: bank ID to delete
            fact_type: Optional fact type filter (world, experience, opinion). If provided, only deletes memories of that type.
            request_context: Request context for authentication.

        Returns:
            Dictionary with counts of deleted items
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Ensure connection is not in read-only mode (can happen with connection poolers)
            await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
            async with conn.transaction():
                try:
                    if fact_type:
                        # Delete only memories of a specific fact type
                        units_count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
                            bank_id,
                            fact_type,
                        )
                        await conn.execute(
                            f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
                            bank_id,
                            fact_type,
                        )

                        # Note: We don't delete entities when fact_type is specified,
                        # as they may be referenced by other memory units
                        return {"memory_units_deleted": units_count, "entities_deleted": 0}
                    else:
                        # Delete all data for the bank
                        units_count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id
                        )
                        entities_count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {fq_table('entities')} WHERE bank_id = $1", bank_id
                        )
                        documents_count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {fq_table('documents')} WHERE bank_id = $1", bank_id
                        )

                        # Delete documents (cascades to chunks)
                        await conn.execute(f"DELETE FROM {fq_table('documents')} WHERE bank_id = $1", bank_id)

                        # Delete memory units (cascades to unit_entities, memory_links)
                        await conn.execute(f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1", bank_id)

                        # Delete entities (cascades to unit_entities, entity_cooccurrences, memory_links with entity_id)
                        await conn.execute(f"DELETE FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)

                        # Delete the bank profile itself
                        await conn.execute(f"DELETE FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)

                        return {
                            "memory_units_deleted": units_count,
                            "entities_deleted": entities_count,
                            "documents_deleted": documents_count,
                            "bank_deleted": True,
                        }

                except Exception as e:
                    raise Exception(f"Failed to delete agent data: {str(e)}")

    async def get_graph_data(
        self,
        bank_id: str | None = None,
        fact_type: str | None = None,
        *,
        request_context: "RequestContext",
    ):
        """
        Get graph data for visualization.

        Args:
            bank_id: Filter by bank ID
            fact_type: Filter by fact type (world, experience, opinion)
            request_context: Request context for authentication.

        Returns:
            Dict with nodes, edges, and table_rows
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get memory units, optionally filtered by bank_id and fact_type
            query_conditions = []
            query_params = []
            param_count = 0

            if bank_id:
                param_count += 1
                query_conditions.append(f"bank_id = ${param_count}")
                query_params.append(bank_id)

            if fact_type:
                param_count += 1
                query_conditions.append(f"fact_type = ${param_count}")
                query_params.append(fact_type)

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            units = await conn.fetch(
                f"""
                SELECT id, text, event_date, context, occurred_start, occurred_end, mentioned_at, document_id, chunk_id, fact_type
                FROM {fq_table("memory_units")}
                {where_clause}
                ORDER BY mentioned_at DESC NULLS LAST, event_date DESC
                LIMIT 1000
            """,
                *query_params,
            )

            # Get links, filtering to only include links between units of the selected agent
            # Use DISTINCT ON with LEAST/GREATEST to deduplicate bidirectional links
            unit_ids = [row["id"] for row in units]
            if unit_ids:
                links = await conn.fetch(
                    f"""
                    SELECT DISTINCT ON (LEAST(ml.from_unit_id, ml.to_unit_id), GREATEST(ml.from_unit_id, ml.to_unit_id), ml.link_type, COALESCE(ml.entity_id, '00000000-0000-0000-0000-000000000000'::uuid))
                        ml.from_unit_id,
                        ml.to_unit_id,
                        ml.link_type,
                        ml.weight,
                        e.canonical_name as entity_name
                    FROM {fq_table("memory_links")} ml
                    LEFT JOIN {fq_table("entities")} e ON ml.entity_id = e.id
                    WHERE ml.from_unit_id = ANY($1::uuid[]) AND ml.to_unit_id = ANY($1::uuid[])
                    ORDER BY LEAST(ml.from_unit_id, ml.to_unit_id), GREATEST(ml.from_unit_id, ml.to_unit_id), ml.link_type, COALESCE(ml.entity_id, '00000000-0000-0000-0000-000000000000'::uuid), ml.weight DESC
                """,
                    unit_ids,
                )
            else:
                links = []

            # Get entity information
            unit_entities = await conn.fetch(f"""
                SELECT ue.unit_id, e.canonical_name
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                ORDER BY ue.unit_id
            """)

        # Build entity mapping
        entity_map = {}
        for row in unit_entities:
            unit_id = row["unit_id"]
            entity_name = row["canonical_name"]
            if unit_id not in entity_map:
                entity_map[unit_id] = []
            entity_map[unit_id].append(entity_name)

        # Build nodes
        nodes = []
        for row in units:
            unit_id = row["id"]
            text = row["text"]
            event_date = row["event_date"]
            context = row["context"]

            entities = entity_map.get(unit_id, [])
            entity_count = len(entities)

            # Color by entity count
            if entity_count == 0:
                color = "#e0e0e0"
            elif entity_count == 1:
                color = "#90caf9"
            else:
                color = "#42a5f5"

            nodes.append(
                {
                    "data": {
                        "id": str(unit_id),
                        "label": f"{text[:30]}..." if len(text) > 30 else text,
                        "text": text,
                        "date": event_date.isoformat() if event_date else "",
                        "context": context if context else "",
                        "entities": ", ".join(entities) if entities else "None",
                        "color": color,
                    }
                }
            )

        # Build edges
        edges = []
        for row in links:
            from_id = str(row["from_unit_id"])
            to_id = str(row["to_unit_id"])
            link_type = row["link_type"]
            weight = row["weight"]
            entity_name = row["entity_name"]

            # Color by link type
            if link_type == "temporal":
                color = "#00bcd4"
                line_style = "dashed"
            elif link_type == "semantic":
                color = "#ff69b4"
                line_style = "solid"
            elif link_type == "entity":
                color = "#ffd700"
                line_style = "solid"
            else:
                color = "#999999"
                line_style = "solid"

            edges.append(
                {
                    "data": {
                        "id": f"{from_id}-{to_id}-{link_type}",
                        "source": from_id,
                        "target": to_id,
                        "linkType": link_type,
                        "weight": weight,
                        "entityName": entity_name if entity_name else "",
                        "color": color,
                        "lineStyle": line_style,
                    }
                }
            )

        # Build table rows
        table_rows = []
        for row in units:
            unit_id = row["id"]
            entities = entity_map.get(unit_id, [])

            table_rows.append(
                {
                    "id": str(unit_id),
                    "text": row["text"],
                    "context": row["context"] if row["context"] else "N/A",
                    "occurred_start": row["occurred_start"].isoformat() if row["occurred_start"] else None,
                    "occurred_end": row["occurred_end"].isoformat() if row["occurred_end"] else None,
                    "mentioned_at": row["mentioned_at"].isoformat() if row["mentioned_at"] else None,
                    "date": row["event_date"].strftime("%Y-%m-%d %H:%M")
                    if row["event_date"]
                    else "N/A",  # Deprecated, kept for backwards compatibility
                    "entities": ", ".join(entities) if entities else "None",
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"] if row["chunk_id"] else None,
                    "fact_type": row["fact_type"],
                }
            )

        return {"nodes": nodes, "edges": edges, "table_rows": table_rows, "total_units": len(units)}

    async def list_memory_units(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ):
        """
        List memory units for table view with optional full-text search.

        Args:
            bank_id: Filter by bank ID
            fact_type: Filter by fact type (world, experience, opinion)
            search_query: Full-text search query (searches text and context fields)
            limit: Maximum number of results to return
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of memory units) and total count
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Build query conditions
            query_conditions = []
            query_params = []
            param_count = 0

            if bank_id:
                param_count += 1
                query_conditions.append(f"bank_id = ${param_count}")
                query_params.append(bank_id)

            if fact_type:
                param_count += 1
                query_conditions.append(f"fact_type = ${param_count}")
                query_params.append(fact_type)

            if search_query:
                # Full-text search on text and context fields using ILIKE
                param_count += 1
                query_conditions.append(f"(text ILIKE ${param_count} OR context ILIKE ${param_count})")
                query_params.append(f"%{search_query}%")

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM {fq_table("memory_units")}
                {where_clause}
            """
            count_result = await conn.fetchrow(count_query, *query_params)
            total = count_result["total"]

            # Get units with limit and offset
            param_count += 1
            limit_param = f"${param_count}"
            query_params.append(limit)

            param_count += 1
            offset_param = f"${param_count}"
            query_params.append(offset)

            units = await conn.fetch(
                f"""
                SELECT id, text, event_date, context, fact_type, mentioned_at, occurred_start, occurred_end, chunk_id
                FROM {fq_table("memory_units")}
                {where_clause}
                ORDER BY mentioned_at DESC NULLS LAST, created_at DESC
                LIMIT {limit_param} OFFSET {offset_param}
            """,
                *query_params,
            )

            # Get entity information for these units
            if units:
                unit_ids = [row["id"] for row in units]
                unit_entities = await conn.fetch(
                    f"""
                    SELECT ue.unit_id, e.canonical_name
                    FROM {fq_table("unit_entities")} ue
                    JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                    WHERE ue.unit_id = ANY($1::uuid[])
                    ORDER BY ue.unit_id
                """,
                    unit_ids,
                )
            else:
                unit_entities = []

            # Build entity mapping
            entity_map = {}
            for row in unit_entities:
                unit_id = row["unit_id"]
                entity_name = row["canonical_name"]
                if unit_id not in entity_map:
                    entity_map[unit_id] = []
                entity_map[unit_id].append(entity_name)

            # Build result items
            items = []
            for row in units:
                unit_id = row["id"]
                entities = entity_map.get(unit_id, [])

                items.append(
                    {
                        "id": str(unit_id),
                        "text": row["text"],
                        "context": row["context"] if row["context"] else "",
                        "date": row["event_date"].isoformat() if row["event_date"] else "",
                        "fact_type": row["fact_type"],
                        "mentioned_at": row["mentioned_at"].isoformat() if row["mentioned_at"] else None,
                        "occurred_start": row["occurred_start"].isoformat() if row["occurred_start"] else None,
                        "occurred_end": row["occurred_end"].isoformat() if row["occurred_end"] else None,
                        "entities": ", ".join(entities) if entities else "",
                        "chunk_id": row["chunk_id"] if row["chunk_id"] else None,
                    }
                )

            return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def list_documents(
        self,
        bank_id: str,
        *,
        search_query: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ):
        """
        List documents with optional search and pagination.

        Args:
            bank_id: bank ID (required)
            search_query: Search in document ID
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of documents without original_text) and total count
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Build query conditions
            query_conditions = []
            query_params = []
            param_count = 0

            param_count += 1
            query_conditions.append(f"bank_id = ${param_count}")
            query_params.append(bank_id)

            if search_query:
                # Search in document ID
                param_count += 1
                query_conditions.append(f"id ILIKE ${param_count}")
                query_params.append(f"%{search_query}%")

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""

            # Get total count
            count_query = f"""
                SELECT COUNT(*) as total
                FROM {fq_table("documents")}
                {where_clause}
            """
            count_result = await conn.fetchrow(count_query, *query_params)
            total = count_result["total"]

            # Get documents with limit and offset (without original_text for performance)
            param_count += 1
            limit_param = f"${param_count}"
            query_params.append(limit)

            param_count += 1
            offset_param = f"${param_count}"
            query_params.append(offset)

            documents = await conn.fetch(
                f"""
                SELECT
                    id,
                    bank_id,
                    content_hash,
                    created_at,
                    updated_at,
                    LENGTH(original_text) as text_length,
                    retain_params
                FROM {fq_table("documents")}
                {where_clause}
                ORDER BY created_at DESC
                LIMIT {limit_param} OFFSET {offset_param}
            """,
                *query_params,
            )

            # Get memory unit count for each document
            if documents:
                doc_ids = [(row["id"], row["bank_id"]) for row in documents]

                # Create placeholders for the query
                placeholders = []
                params_for_count = []
                for i, (doc_id, bank_id_val) in enumerate(doc_ids):
                    idx_doc = i * 2 + 1
                    idx_agent = i * 2 + 2
                    placeholders.append(f"(document_id = ${idx_doc} AND bank_id = ${idx_agent})")
                    params_for_count.extend([doc_id, bank_id_val])

                where_clause_count = " OR ".join(placeholders)

                unit_counts = await conn.fetch(
                    f"""
                    SELECT document_id, bank_id, COUNT(*) as unit_count
                    FROM {fq_table("memory_units")}
                    WHERE {where_clause_count}
                    GROUP BY document_id, bank_id
                """,
                    *params_for_count,
                )
            else:
                unit_counts = []

            # Build count mapping
            count_map = {(row["document_id"], row["bank_id"]): row["unit_count"] for row in unit_counts}

            # Build result items
            items = []
            for row in documents:
                doc_id = row["id"]
                bank_id_val = row["bank_id"]
                unit_count = count_map.get((doc_id, bank_id_val), 0)

                items.append(
                    {
                        "id": doc_id,
                        "bank_id": bank_id_val,
                        "content_hash": row["content_hash"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
                        "text_length": row["text_length"] or 0,
                        "memory_unit_count": unit_count,
                        "retain_params": row["retain_params"] if row["retain_params"] else None,
                    }
                )

            return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def get_chunk(
        self,
        chunk_id: str,
        *,
        request_context: "RequestContext",
    ):
        """
        Get a specific chunk by its ID.

        Args:
            chunk_id: Chunk ID (format: bank_id_document_id_chunk_index)
            request_context: Request context for authentication.

        Returns:
            Dict with chunk details including chunk_text, or None if not found
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            chunk = await conn.fetchrow(
                f"""
                SELECT
                    chunk_id,
                    document_id,
                    bank_id,
                    chunk_index,
                    chunk_text,
                    created_at
                FROM {fq_table("chunks")}
                WHERE chunk_id = $1
            """,
                chunk_id,
            )

            if not chunk:
                return None

            return {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "bank_id": chunk["bank_id"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "created_at": chunk["created_at"].isoformat() if chunk["created_at"] else "",
            }

    async def _evaluate_opinion_update_async(
        self,
        opinion_text: str,
        opinion_confidence: float,
        new_event_text: str,
        entity_name: str,
    ) -> dict[str, Any] | None:
        """
        Evaluate if an opinion should be updated based on a new event.

        Args:
            opinion_text: Current opinion text (includes reasons)
            opinion_confidence: Current confidence score (0.0-1.0)
            new_event_text: Text of the new event
            entity_name: Name of the entity this opinion is about

        Returns:
            Dict with 'action' ('keep'|'update'), 'new_confidence', 'new_text' (if action=='update')
            or None if no changes needed
        """

        class OpinionEvaluation(BaseModel):
            """Evaluation of whether an opinion should be updated."""

            action: str = Field(description="Action to take: 'keep' (no change) or 'update' (modify opinion)")
            reasoning: str = Field(description="Brief explanation of why this action was chosen")
            new_confidence: float = Field(
                description="New confidence score (0.0-1.0). Can be higher, lower, or same as before."
            )
            new_opinion_text: str | None = Field(
                default=None,
                description="If action is 'update', the revised opinion text that acknowledges the previous view. Otherwise None.",
            )

        evaluation_prompt = f"""You are evaluating whether an existing opinion should be updated based on new information.

ENTITY: {entity_name}

EXISTING OPINION:
{opinion_text}
Current confidence: {opinion_confidence:.2f}

NEW EVENT:
{new_event_text}

Evaluate whether this new event:
1. REINFORCES the opinion (increase confidence, keep text)
2. WEAKENS the opinion (decrease confidence, keep text)
3. CHANGES the opinion (update both text and confidence, noting "Previously I thought X, but now Y...")
4. IRRELEVANT (keep everything as is)

Guidelines:
- Only suggest 'update' action if the new event genuinely contradicts or significantly modifies the opinion
- If updating the text, acknowledge the previous opinion and explain the change
- Confidence should reflect accumulated evidence (0.0 = no confidence, 1.0 = very confident)
- Small changes in confidence are normal; large jumps should be rare"""

        try:
            result = await self._llm_config.call(
                messages=[
                    {"role": "system", "content": "You evaluate and update opinions based on new information."},
                    {"role": "user", "content": evaluation_prompt},
                ],
                response_format=OpinionEvaluation,
                scope="memory_evaluate_opinion",
                temperature=0.3,  # Lower temperature for more consistent evaluation
            )

            # Only return updates if something actually changed
            if result.action == "keep" and abs(result.new_confidence - opinion_confidence) < 0.01:
                return None

            return {
                "action": result.action,
                "reasoning": result.reasoning,
                "new_confidence": result.new_confidence,
                "new_text": result.new_opinion_text if result.action == "update" else None,
            }

        except Exception as e:
            logger.warning(f"Failed to evaluate opinion update: {str(e)}")
            return None

    async def _handle_form_opinion(self, task_dict: dict[str, Any]):
        """
        Handler for form opinion tasks.

        Args:
            task_dict: Dict with keys: 'bank_id', 'answer_text', 'query', 'tenant_id'
        """
        bank_id = task_dict["bank_id"]
        answer_text = task_dict["answer_text"]
        query = task_dict["query"]
        tenant_id = task_dict.get("tenant_id")

        await self._extract_and_store_opinions_async(
            bank_id=bank_id, answer_text=answer_text, query=query, tenant_id=tenant_id
        )

    async def _handle_reinforce_opinion(self, task_dict: dict[str, Any]):
        """
        Handler for reinforce opinion tasks.

        Args:
            task_dict: Dict with keys: 'bank_id', 'created_unit_ids', 'unit_texts', 'unit_entities'
        """
        bank_id = task_dict["bank_id"]
        created_unit_ids = task_dict["created_unit_ids"]
        unit_texts = task_dict["unit_texts"]
        unit_entities = task_dict["unit_entities"]

        await self._reinforce_opinions_async(
            bank_id=bank_id, created_unit_ids=created_unit_ids, unit_texts=unit_texts, unit_entities=unit_entities
        )

    async def _reinforce_opinions_async(
        self,
        bank_id: str,
        created_unit_ids: list[str],
        unit_texts: list[str],
        unit_entities: list[list[dict[str, str]]],
    ):
        """
        Background task to reinforce opinions based on newly ingested events.

        This runs asynchronously and does not block the put operation.

        Args:
            bank_id: bank ID
            created_unit_ids: List of newly created memory unit IDs
            unit_texts: Texts of the newly created units
            unit_entities: Entities extracted from each unit
        """
        try:
            # Extract all unique entity names from the new units
            entity_names = set()
            for entities_list in unit_entities:
                for entity in entities_list:
                    # Handle both Entity objects and dicts
                    if hasattr(entity, "text"):
                        entity_names.add(entity.text)
                    elif isinstance(entity, dict):
                        entity_names.add(entity["text"])

            if not entity_names:
                return

            pool = await self._get_pool()
            async with acquire_with_retry(pool) as conn:
                # Find all opinions related to these entities
                opinions = await conn.fetch(
                    f"""
                    SELECT DISTINCT mu.id, mu.text, mu.confidence_score, e.canonical_name
                    FROM {fq_table("memory_units")} mu
                    JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
                    JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                    WHERE mu.bank_id = $1
                      AND mu.fact_type = 'opinion'
                      AND e.canonical_name = ANY($2::text[])
                    """,
                    bank_id,
                    list(entity_names),
                )

                if not opinions:
                    return

                # Use cached LLM config
                if self._llm_config is None:
                    logger.error("[REINFORCE] LLM config not available, skipping opinion reinforcement")
                    return

                # Evaluate each opinion against the new events
                updates_to_apply = []
                for opinion in opinions:
                    opinion_id = str(opinion["id"])
                    opinion_text = opinion["text"]
                    opinion_confidence = opinion["confidence_score"]
                    entity_name = opinion["canonical_name"]

                    # Find all new events mentioning this entity
                    relevant_events = []
                    for unit_text, entities_list in zip(unit_texts, unit_entities):
                        if any(e["text"] == entity_name for e in entities_list):
                            relevant_events.append(unit_text)

                    if not relevant_events:
                        continue

                    # Combine all relevant events
                    combined_events = "\n".join(relevant_events)

                    # Evaluate if opinion should be updated
                    evaluation = await self._evaluate_opinion_update_async(
                        opinion_text, opinion_confidence, combined_events, entity_name
                    )

                    if evaluation:
                        updates_to_apply.append({"opinion_id": opinion_id, "evaluation": evaluation})

                # Apply all updates in a single transaction
                if updates_to_apply:
                    async with conn.transaction():
                        for update in updates_to_apply:
                            opinion_id = update["opinion_id"]
                            evaluation = update["evaluation"]

                            if evaluation["action"] == "update" and evaluation["new_text"]:
                                # Update both text and confidence
                                await conn.execute(
                                    f"""
                                    UPDATE {fq_table("memory_units")}
                                    SET text = $1, confidence_score = $2, updated_at = NOW()
                                    WHERE id = $3
                                    """,
                                    evaluation["new_text"],
                                    evaluation["new_confidence"],
                                    uuid.UUID(opinion_id),
                                )
                            else:
                                # Only update confidence
                                await conn.execute(
                                    f"""
                                    UPDATE {fq_table("memory_units")}
                                    SET confidence_score = $1, updated_at = NOW()
                                    WHERE id = $2
                                    """,
                                    evaluation["new_confidence"],
                                    uuid.UUID(opinion_id),
                                )

                else:
                    pass  # No opinions to update

        except Exception as e:
            logger.error(f"[REINFORCE] Error during opinion reinforcement: {str(e)}")
            import traceback

            traceback.print_exc()

    # ==================== bank profile Methods ====================

    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get bank profile (name, disposition + background).
        Auto-creates agent with default values if not exists.

        Args:
            bank_id: bank IDentifier
            request_context: Request context for authentication.

        Returns:
            Dict with name, disposition traits, and background
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        profile = await bank_utils.get_bank_profile(pool, bank_id)
        disposition = profile["disposition"]
        return {
            "bank_id": bank_id,
            "name": profile["name"],
            "disposition": disposition,
            "background": profile["background"],
        }

    async def update_bank_disposition(
        self,
        bank_id: str,
        disposition: dict[str, int],
        *,
        request_context: "RequestContext",
    ) -> None:
        """
        Update bank disposition traits.

        Args:
            bank_id: bank IDentifier
            disposition: Dict with skepticism, literalism, empathy (all 1-5)
            request_context: Request context for authentication.
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        await bank_utils.update_bank_disposition(pool, bank_id, disposition)

    async def merge_bank_background(
        self,
        bank_id: str,
        new_info: str,
        *,
        update_disposition: bool = True,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Merge new background information with existing background using LLM.
        Normalizes to first person ("I") and resolves conflicts.
        Optionally infers disposition traits from the merged background.

        Args:
            bank_id: bank IDentifier
            new_info: New background information to add/merge
            update_disposition: If True, infer Big Five traits from background (default: True)
            request_context: Request context for authentication.

        Returns:
            Dict with 'background' (str) and optionally 'disposition' (dict) keys
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        return await bank_utils.merge_bank_background(pool, self._llm_config, bank_id, new_info, update_disposition)

    async def list_banks(
        self,
        *,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """
        List all agents in the system.

        Args:
            request_context: Request context for authentication.

        Returns:
            List of dicts with bank_id, name, disposition, background, created_at, updated_at
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        return await bank_utils.list_banks(pool)

    # ==================== Reflect Methods ====================

    async def reflect_async(
        self,
        bank_id: str,
        query: str,
        *,
        budget: Budget | None = None,
        context: str | None = None,
        max_tokens: int = 4096,
        response_schema: dict | None = None,
        request_context: "RequestContext",
    ) -> ReflectResult:
        """
        Reflect and formulate an answer using bank identity, world facts, and opinions.

        This method:
        1. Retrieves experience (conversations and events)
        2. Retrieves world facts (general knowledge)
        3. Retrieves existing opinions (bank's formed perspectives)
        4. Uses LLM to formulate an answer
        5. Extracts and stores any new opinions formed during reflection
        6. Optionally generates structured output based on response_schema
        7. Returns plain text answer and the facts used

        Args:
            bank_id: bank identifier
            query: Question to answer
            budget: Budget level for memory exploration (low=100, mid=300, high=600 units)
            context: Additional context string to include in LLM prompt (not used in recall)
            response_schema: Optional JSON Schema for structured output

        Returns:
            ReflectResult containing:
                - text: Plain text answer (no markdown)
                - based_on: Dict with 'world', 'experience', and 'opinion' fact lists (MemoryFact objects)
                - new_opinions: List of newly formed opinions
                - structured_output: Optional dict if response_schema was provided
        """
        # Use cached LLM config
        if self._llm_config is None:
            raise ValueError("Memory LLM API key not set. Set HINDSIGHT_API_LLM_API_KEY environment variable.")

        # Authenticate tenant and set schema in context (for fq_table())
        await self._authenticate_tenant(request_context)

        # Validate operation if validator is configured
        if self._operation_validator:
            from hindsight_api.extensions import ReflectContext

            ctx = ReflectContext(
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                budget=budget,
                context=context,
            )
            await self._validate_operation(self._operation_validator.validate_reflect(ctx))

        reflect_start = time.time()
        reflect_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
        log_buffer = []
        log_buffer.append(f"[REFLECT {reflect_id}] Query: '{query[:50]}...'")

        # Steps 1-3: Run multi-fact-type search (12-way retrieval: 4 methods × 3 fact types)
        recall_start = time.time()
        search_result = await self.recall_async(
            bank_id=bank_id,
            query=query,
            budget=budget,
            max_tokens=4096,
            enable_trace=False,
            fact_type=["experience", "world", "opinion"],
            include_entities=True,
            request_context=request_context,
        )
        recall_time = time.time() - recall_start

        all_results = search_result.results

        # Split results by fact type for structured response
        agent_results = [r for r in all_results if r.fact_type == "experience"]
        world_results = [r for r in all_results if r.fact_type == "world"]
        opinion_results = [r for r in all_results if r.fact_type == "opinion"]

        log_buffer.append(
            f"[REFLECT {reflect_id}] Recall: {len(all_results)} facts (experience={len(agent_results)}, world={len(world_results)}, opinion={len(opinion_results)}) in {recall_time:.3f}s"
        )

        # Format facts for LLM
        agent_facts_text = think_utils.format_facts_for_prompt(agent_results)
        world_facts_text = think_utils.format_facts_for_prompt(world_results)
        opinion_facts_text = think_utils.format_facts_for_prompt(opinion_results)

        # Get bank profile (name, disposition + background)
        profile = await self.get_bank_profile(bank_id, request_context=request_context)
        name = profile["name"]
        disposition = profile["disposition"]  # Typed as DispositionTraits
        background = profile["background"]

        # Build the prompt
        prompt = think_utils.build_think_prompt(
            agent_facts_text=agent_facts_text,
            world_facts_text=world_facts_text,
            opinion_facts_text=opinion_facts_text,
            query=query,
            name=name,
            disposition=disposition,
            background=background,
            context=context,
        )

        log_buffer.append(f"[REFLECT {reflect_id}] Prompt: {len(prompt)} chars")

        system_message = think_utils.get_system_message(disposition)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]

        # Prepare response_format if schema provided
        response_format = None
        if response_schema is not None:
            # Wrapper class to provide Pydantic-like interface for raw JSON schemas
            class JsonSchemaWrapper:
                def __init__(self, schema: dict):
                    self._schema = schema

                def model_json_schema(self):
                    return self._schema

            response_format = JsonSchemaWrapper(response_schema)

        llm_start = time.time()
        result = await self._llm_config.call(
            messages=messages,
            scope="memory_reflect",
            max_completion_tokens=max_tokens,
            response_format=response_format,
            skip_validation=True if response_format else False,
            # Don't enforce strict_schema - not all providers support it and may retry forever
            # Soft enforcement (schema in prompt + json_object mode) is sufficient
            strict_schema=False,
        )
        llm_time = time.time() - llm_start

        # Handle response based on whether structured output was requested
        if response_schema is not None:
            structured_output = result
            answer_text = ""  # Empty for backward compatibility
            log_buffer.append(f"[REFLECT {reflect_id}] Structured output generated")
        else:
            structured_output = None
            answer_text = result.strip()

        # Submit form_opinion task for background processing
        # Pass tenant_id from request context for internal authentication in background task
        await self._task_backend.submit_task(
            {
                "type": "form_opinion",
                "bank_id": bank_id,
                "answer_text": answer_text,
                "query": query,
                "tenant_id": getattr(request_context, "tenant_id", None) if request_context else None,
            }
        )

        total_time = time.time() - reflect_start
        log_buffer.append(
            f"[REFLECT {reflect_id}] Complete: {len(answer_text)} chars response, LLM {llm_time:.3f}s, total {total_time:.3f}s"
        )
        logger.info("\n" + "\n".join(log_buffer))

        # Return response with facts split by type
        result = ReflectResult(
            text=answer_text,
            based_on={"world": world_results, "experience": agent_results, "opinion": opinion_results},
            new_opinions=[],  # Opinions are being extracted asynchronously
            structured_output=structured_output,
        )

        # Call post-operation hook if validator is configured
        if self._operation_validator:
            from hindsight_api.extensions.operation_validator import ReflectResultContext

            result_ctx = ReflectResultContext(
                bank_id=bank_id,
                query=query,
                request_context=request_context,
                budget=budget,
                context=context,
                result=result,
                success=True,
                error=None,
            )
            try:
                await self._operation_validator.on_reflect_complete(result_ctx)
            except Exception as e:
                logger.warning(f"Post-reflect hook error (non-fatal): {e}")

        return result

    async def _extract_and_store_opinions_async(
        self, bank_id: str, answer_text: str, query: str, tenant_id: str | None = None
    ):
        """
        Background task to extract and store opinions from think response.

        This runs asynchronously and does not block the think response.

        Args:
            bank_id: bank IDentifier
            answer_text: The generated answer text
            query: The original query
            tenant_id: Tenant identifier for internal authentication
        """
        try:
            # Extract opinions from the answer
            new_opinions = await think_utils.extract_opinions_from_text(self._llm_config, text=answer_text, query=query)

            # Store new opinions
            if new_opinions:
                from datetime import datetime

                current_time = datetime.now(UTC)
                # Use internal context with tenant_id for background authentication
                # Extension can check internal=True to bypass normal auth
                from hindsight_api.models import RequestContext

                internal_context = RequestContext(tenant_id=tenant_id, internal=True)
                for opinion in new_opinions:
                    await self.retain_async(
                        bank_id=bank_id,
                        content=opinion.opinion,
                        context=f"formed during thinking about: {query}",
                        event_date=current_time,
                        fact_type_override="opinion",
                        confidence_score=opinion.confidence,
                        request_context=internal_context,
                    )

        except Exception as e:
            logger.warning(f"[REFLECT] Failed to extract/store opinions: {str(e)}")

    async def get_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        *,
        limit: int = 10,
        request_context: "RequestContext",
    ) -> list[Any]:
        """
        Get observations linked to an entity.

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID to get observations for
            limit: Maximum number of observations to return
            request_context: Request context for authentication.

        Returns:
            List of EntityObservation objects
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(
                f"""
                SELECT mu.text, mu.mentioned_at
                FROM {fq_table("memory_units")} mu
                JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
                WHERE mu.bank_id = $1
                  AND mu.fact_type = 'observation'
                  AND ue.entity_id = $2
                ORDER BY mu.mentioned_at DESC
                LIMIT $3
                """,
                bank_id,
                uuid.UUID(entity_id),
                limit,
            )

            observations = []
            for row in rows:
                mentioned_at = row["mentioned_at"].isoformat() if row["mentioned_at"] else None
                observations.append(EntityObservation(text=row["text"], mentioned_at=mentioned_at))
            return observations

    async def list_entities(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """
        List all entities for a bank.

        Args:
            bank_id: bank IDentifier
            limit: Maximum number of entities to return
            request_context: Request context for authentication.

        Returns:
            List of entity dicts with id, canonical_name, mention_count, first_seen, last_seen
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, canonical_name, mention_count, first_seen, last_seen, metadata
                FROM {fq_table("entities")}
                WHERE bank_id = $1
                ORDER BY mention_count DESC, last_seen DESC
                LIMIT $2
                """,
                bank_id,
                limit,
            )

            entities = []
            for row in rows:
                # Handle metadata - may be dict, JSON string, or None
                metadata = row["metadata"]
                if metadata is None:
                    metadata = {}
                elif isinstance(metadata, str):
                    import json

                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                entities.append(
                    {
                        "id": str(row["id"]),
                        "canonical_name": row["canonical_name"],
                        "mention_count": row["mention_count"],
                        "first_seen": row["first_seen"].isoformat() if row["first_seen"] else None,
                        "last_seen": row["last_seen"].isoformat() if row["last_seen"] else None,
                        "metadata": metadata,
                    }
                )
            return entities

    async def get_entity_state(
        self,
        bank_id: str,
        entity_id: str,
        entity_name: str,
        *,
        limit: int = 10,
        request_context: "RequestContext",
    ) -> EntityState:
        """
        Get the current state (mental model) of an entity.

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID
            entity_name: Canonical name of the entity
            limit: Maximum number of observations to include
            request_context: Request context for authentication.

        Returns:
            EntityState with observations
        """
        observations = await self.get_entity_observations(
            bank_id, entity_id, limit=limit, request_context=request_context
        )
        return EntityState(entity_id=entity_id, canonical_name=entity_name, observations=observations)

    async def regenerate_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        entity_name: str,
        *,
        version: str | None = None,
        conn=None,
        request_context: "RequestContext",
    ) -> None:
        """
        Regenerate observations for an entity by:
        1. Checking version for deduplication (if provided)
        2. Searching all facts mentioning the entity
        3. Using LLM to synthesize observations (no personality)
        4. Deleting old observations for this entity
        5. Storing new observations linked to the entity

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID
            entity_name: Canonical name of the entity
            version: Entity's last_seen timestamp when task was created (for deduplication)
            conn: Optional database connection (for transactional atomicity with caller)
            request_context: Request context for authentication.
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        entity_uuid = uuid.UUID(entity_id)

        # Helper to run a query with provided conn or acquire one
        async def fetch_with_conn(query, *args):
            if conn is not None:
                return await conn.fetch(query, *args)
            else:
                async with acquire_with_retry(pool) as acquired_conn:
                    return await acquired_conn.fetch(query, *args)

        async def fetchval_with_conn(query, *args):
            if conn is not None:
                return await conn.fetchval(query, *args)
            else:
                async with acquire_with_retry(pool) as acquired_conn:
                    return await acquired_conn.fetchval(query, *args)

        # Step 1: Check version for deduplication
        if version:
            current_last_seen = await fetchval_with_conn(
                f"""
                SELECT last_seen
                FROM {fq_table("entities")}
                WHERE id = $1 AND bank_id = $2
                """,
                entity_uuid,
                bank_id,
            )

            if current_last_seen and current_last_seen.isoformat() != version:
                return []

        # Step 2: Get all facts mentioning this entity (exclude observations themselves)
        rows = await fetch_with_conn(
            f"""
            SELECT mu.id, mu.text, mu.context, mu.occurred_start, mu.fact_type
            FROM {fq_table("memory_units")} mu
            JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
            WHERE mu.bank_id = $1
              AND ue.entity_id = $2
              AND mu.fact_type IN ('world', 'experience')
            ORDER BY mu.occurred_start DESC
            LIMIT 50
            """,
            bank_id,
            entity_uuid,
        )

        if not rows:
            return []

        # Convert to MemoryFact objects for the observation extraction
        facts = []
        for row in rows:
            occurred_start = row["occurred_start"].isoformat() if row["occurred_start"] else None
            facts.append(
                MemoryFact(
                    id=str(row["id"]),
                    text=row["text"],
                    fact_type=row["fact_type"],
                    context=row["context"],
                    occurred_start=occurred_start,
                )
            )

        # Step 3: Extract observations using LLM (no personality)
        observations = await observation_utils.extract_observations_from_facts(self._llm_config, entity_name, facts)

        if not observations:
            return []

        # Step 4: Delete old observations and insert new ones
        # If conn provided, we're already in a transaction - don't start another
        # If conn is None, acquire one and start a transaction
        async def do_db_operations(db_conn):
            # Delete old observations for this entity
            await db_conn.execute(
                f"""
                DELETE FROM {fq_table("memory_units")}
                WHERE id IN (
                    SELECT mu.id
                    FROM {fq_table("memory_units")} mu
                    JOIN {fq_table("unit_entities")} ue ON mu.id = ue.unit_id
                    WHERE mu.bank_id = $1
                      AND mu.fact_type = 'observation'
                      AND ue.entity_id = $2
                )
                """,
                bank_id,
                entity_uuid,
            )

            # Generate embeddings for new observations
            embeddings = await embedding_utils.generate_embeddings_batch(self.embeddings, observations)

            # Insert new observations
            current_time = utcnow()
            created_ids = []

            for obs_text, embedding in zip(observations, embeddings):
                result = await db_conn.fetchrow(
                    f"""
                    INSERT INTO {fq_table("memory_units")} (
                        bank_id, text, embedding, context, event_date,
                        occurred_start, occurred_end, mentioned_at,
                        fact_type, access_count
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'observation', 0)
                    RETURNING id
                    """,
                    bank_id,
                    obs_text,
                    str(embedding),
                    f"observation about {entity_name}",
                    current_time,
                    current_time,
                    current_time,
                    current_time,
                )
                obs_id = str(result["id"])
                created_ids.append(obs_id)

                # Link observation to entity
                await db_conn.execute(
                    f"""
                    INSERT INTO {fq_table("unit_entities")} (unit_id, entity_id)
                    VALUES ($1, $2)
                    """,
                    uuid.UUID(obs_id),
                    entity_uuid,
                )

            return created_ids

        if conn is not None:
            # Use provided connection (already in a transaction)
            return await do_db_operations(conn)
        else:
            # Acquire connection and start our own transaction
            async with acquire_with_retry(pool) as acquired_conn:
                async with acquired_conn.transaction():
                    return await do_db_operations(acquired_conn)

    async def _regenerate_observations_sync(
        self,
        bank_id: str,
        entity_ids: list[str],
        min_facts: int | None = None,
        conn=None,
        request_context: "RequestContext | None" = None,
    ) -> None:
        """
        Regenerate observations for entities synchronously (called during retain).

        Processes entities in PARALLEL for faster execution.

        Args:
            bank_id: Bank identifier
            entity_ids: List of entity IDs to process
            min_facts: Minimum facts required to regenerate observations (uses config default if None)
            conn: Optional database connection (for transactional atomicity)
        """
        if not bank_id or not entity_ids:
            return

        # Use config default if min_facts not specified
        if min_facts is None:
            min_facts = get_config().observation_min_facts

        # Convert to UUIDs
        entity_uuids = [uuid.UUID(eid) if isinstance(eid, str) else eid for eid in entity_ids]

        # Use provided connection or acquire a new one
        if conn is not None:
            # Use the provided connection (transactional with caller)
            entity_rows = await conn.fetch(
                f"""
                SELECT id, canonical_name FROM {fq_table("entities")}
                WHERE id = ANY($1) AND bank_id = $2
                """,
                entity_uuids,
                bank_id,
            )
            entity_names = {row["id"]: row["canonical_name"] for row in entity_rows}

            fact_counts = await conn.fetch(
                f"""
                SELECT ue.entity_id, COUNT(*) as cnt
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("memory_units")} mu ON ue.unit_id = mu.id
                WHERE ue.entity_id = ANY($1) AND mu.bank_id = $2
                GROUP BY ue.entity_id
                """,
                entity_uuids,
                bank_id,
            )
            entity_fact_counts = {row["entity_id"]: row["cnt"] for row in fact_counts}
        else:
            # Acquire a new connection (standalone call)
            pool = await self._get_pool()
            async with pool.acquire() as acquired_conn:
                entity_rows = await acquired_conn.fetch(
                    f"""
                    SELECT id, canonical_name FROM {fq_table("entities")}
                    WHERE id = ANY($1) AND bank_id = $2
                    """,
                    entity_uuids,
                    bank_id,
                )
                entity_names = {row["id"]: row["canonical_name"] for row in entity_rows}

                fact_counts = await acquired_conn.fetch(
                    f"""
                    SELECT ue.entity_id, COUNT(*) as cnt
                    FROM {fq_table("unit_entities")} ue
                    JOIN {fq_table("memory_units")} mu ON ue.unit_id = mu.id
                    WHERE ue.entity_id = ANY($1) AND mu.bank_id = $2
                    GROUP BY ue.entity_id
                    """,
                    entity_uuids,
                    bank_id,
                )
                entity_fact_counts = {row["entity_id"]: row["cnt"] for row in fact_counts}

        # Filter entities that meet the threshold
        entities_to_process = []
        for entity_id in entity_ids:
            entity_uuid = uuid.UUID(entity_id) if isinstance(entity_id, str) else entity_id
            if entity_uuid not in entity_names:
                continue
            fact_count = entity_fact_counts.get(entity_uuid, 0)
            if fact_count >= min_facts:
                entities_to_process.append((entity_id, entity_names[entity_uuid]))

        if not entities_to_process:
            return

        # Use internal context if not provided (for internal/background calls)
        from hindsight_api.models import RequestContext as RC

        ctx = request_context if request_context is not None else RC()

        # Process all entities in PARALLEL (LLM calls are the bottleneck)
        async def process_entity(entity_id: str, entity_name: str):
            try:
                await self.regenerate_entity_observations(
                    bank_id, entity_id, entity_name, version=None, conn=conn, request_context=ctx
                )
            except Exception as e:
                logger.error(f"[OBSERVATIONS] Error processing entity {entity_id}: {e}")

        await asyncio.gather(*[process_entity(eid, name) for eid, name in entities_to_process])

    async def _handle_regenerate_observations(self, task_dict: dict[str, Any]):
        """
        Handler for regenerate_observations tasks.

        Args:
            task_dict: Dict with 'bank_id' and either:
                       - 'entity_ids' (list): Process multiple entities
                       - 'entity_id', 'entity_name': Process single entity (legacy)

        Raises:
            ValueError: If required fields are missing
            Exception: Any exception from regenerate_entity_observations (propagates to execute_task for retry)
        """
        bank_id = task_dict.get("bank_id")
        # Use internal request context for background tasks
        from hindsight_api.models import RequestContext

        internal_context = RequestContext()

        # New format: multiple entity_ids
        if "entity_ids" in task_dict:
            entity_ids = task_dict.get("entity_ids", [])
            min_facts = task_dict.get("min_facts", 5)

            if not bank_id or not entity_ids:
                raise ValueError(f"[OBSERVATIONS] Missing required fields in task: {task_dict}")

            # Process each entity
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                for entity_id in entity_ids:
                    try:
                        # Fetch entity name and check fact count
                        import uuid as uuid_module

                        entity_uuid = uuid_module.UUID(entity_id) if isinstance(entity_id, str) else entity_id

                        # First check if entity exists
                        entity_exists = await conn.fetchrow(
                            f"SELECT canonical_name FROM {fq_table('entities')} WHERE id = $1 AND bank_id = $2",
                            entity_uuid,
                            bank_id,
                        )

                        if not entity_exists:
                            logger.debug(f"[OBSERVATIONS] Entity {entity_id} not yet in bank {bank_id}, skipping")
                            continue

                        entity_name = entity_exists["canonical_name"]

                        # Count facts linked to this entity
                        fact_count = (
                            await conn.fetchval(
                                f"SELECT COUNT(*) FROM {fq_table('unit_entities')} WHERE entity_id = $1",
                                entity_uuid,
                            )
                            or 0
                        )

                        # Only regenerate if entity has enough facts
                        if fact_count >= min_facts:
                            await self.regenerate_entity_observations(
                                bank_id, entity_id, entity_name, version=None, request_context=internal_context
                            )
                        else:
                            logger.debug(
                                f"[OBSERVATIONS] Skipping {entity_name} ({fact_count} facts < {min_facts} threshold)"
                            )

                    except Exception as e:
                        # Log but continue processing other entities - individual entity failures
                        # shouldn't fail the whole batch
                        logger.error(f"[OBSERVATIONS] Error processing entity {entity_id}: {e}")
                        continue

        # Legacy format: single entity
        else:
            entity_id = task_dict.get("entity_id")
            entity_name = task_dict.get("entity_name")
            version = task_dict.get("version")

            if not all([bank_id, entity_id, entity_name]):
                raise ValueError(f"[OBSERVATIONS] Missing required fields in task: {task_dict}")

            # Type assertions after validation
            assert isinstance(bank_id, str) and isinstance(entity_id, str) and isinstance(entity_name, str)
            await self.regenerate_entity_observations(
                bank_id, entity_id, entity_name, version=version, request_context=internal_context
            )

    # =========================================================================
    # Statistics & Operations (for HTTP API layer)
    # =========================================================================

    async def get_bank_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Get statistics about memory nodes and links for a bank."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            # Get node counts by fact_type
            node_stats = await conn.fetch(
                f"""
                SELECT fact_type, COUNT(*) as count
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                GROUP BY fact_type
                """,
                bank_id,
            )

            # Get link counts by link_type
            link_stats = await conn.fetch(
                f"""
                SELECT ml.link_type, COUNT(*) as count
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
                WHERE mu.bank_id = $1
                GROUP BY ml.link_type
                """,
                bank_id,
            )

            # Get link counts by fact_type (from nodes)
            link_fact_type_stats = await conn.fetch(
                f"""
                SELECT mu.fact_type, COUNT(*) as count
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
                WHERE mu.bank_id = $1
                GROUP BY mu.fact_type
                """,
                bank_id,
            )

            # Get link counts by fact_type AND link_type
            link_breakdown_stats = await conn.fetch(
                f"""
                SELECT mu.fact_type, ml.link_type, COUNT(*) as count
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
                WHERE mu.bank_id = $1
                GROUP BY mu.fact_type, ml.link_type
                """,
                bank_id,
            )

            # Get pending and failed operations counts
            ops_stats = await conn.fetch(
                f"""
                SELECT status, COUNT(*) as count
                FROM {fq_table("async_operations")}
                WHERE bank_id = $1
                GROUP BY status
                """,
                bank_id,
            )

            return {
                "bank_id": bank_id,
                "node_counts": {row["fact_type"]: row["count"] for row in node_stats},
                "link_counts": {row["link_type"]: row["count"] for row in link_stats},
                "link_counts_by_fact_type": {row["fact_type"]: row["count"] for row in link_fact_type_stats},
                "link_breakdown": [
                    {"fact_type": row["fact_type"], "link_type": row["link_type"], "count": row["count"]}
                    for row in link_breakdown_stats
                ],
                "operations": {row["status"]: row["count"] for row in ops_stats},
            }

    async def get_entity(
        self,
        bank_id: str,
        entity_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Get entity details including metadata and observations."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            entity_row = await conn.fetchrow(
                f"""
                SELECT id, canonical_name, mention_count, first_seen, last_seen, metadata
                FROM {fq_table("entities")}
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                uuid.UUID(entity_id),
            )

        if not entity_row:
            return None

        # Get observations for the entity
        observations = await self.get_entity_observations(bank_id, entity_id, limit=20, request_context=request_context)

        return {
            "id": str(entity_row["id"]),
            "canonical_name": entity_row["canonical_name"],
            "mention_count": entity_row["mention_count"],
            "first_seen": entity_row["first_seen"].isoformat() if entity_row["first_seen"] else None,
            "last_seen": entity_row["last_seen"].isoformat() if entity_row["last_seen"] else None,
            "metadata": entity_row["metadata"] or {},
            "observations": observations,
        }

    async def list_operations(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List async operations for a bank."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            operations = await conn.fetch(
                f"""
                SELECT operation_id, bank_id, operation_type, created_at, status, error_message, result_metadata
                FROM {fq_table("async_operations")}
                WHERE bank_id = $1
                ORDER BY created_at DESC
                """,
                bank_id,
            )

            def parse_metadata(metadata):
                if metadata is None:
                    return {}
                if isinstance(metadata, str):
                    import json

                    return json.loads(metadata)
                return metadata

            return [
                {
                    "id": str(row["operation_id"]),
                    "task_type": row["operation_type"],
                    "items_count": parse_metadata(row["result_metadata"]).get("items_count", 0),
                    "document_id": parse_metadata(row["result_metadata"]).get("document_id"),
                    "created_at": row["created_at"].isoformat(),
                    "status": row["status"],
                    "error_message": row["error_message"],
                }
                for row in operations
            ]

    async def cancel_operation(
        self,
        bank_id: str,
        operation_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Cancel a pending async operation."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        op_uuid = uuid.UUID(operation_id)

        async with acquire_with_retry(pool) as conn:
            # Check if operation exists and belongs to this memory bank
            result = await conn.fetchrow(
                f"SELECT bank_id FROM {fq_table('async_operations')} WHERE operation_id = $1 AND bank_id = $2",
                op_uuid,
                bank_id,
            )

            if not result:
                raise ValueError(f"Operation {operation_id} not found for bank {bank_id}")

            # Delete the operation
            await conn.execute(f"DELETE FROM {fq_table('async_operations')} WHERE operation_id = $1", op_uuid)

            return {
                "success": True,
                "message": f"Operation {operation_id} cancelled",
                "operation_id": operation_id,
                "bank_id": bank_id,
            }

    async def update_bank(
        self,
        bank_id: str,
        *,
        name: str | None = None,
        background: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Update bank name and/or background."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            if name is not None:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("banks")}
                    SET name = $2, updated_at = NOW()
                    WHERE bank_id = $1
                    """,
                    bank_id,
                    name,
                )

            if background is not None:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("banks")}
                    SET background = $2, updated_at = NOW()
                    WHERE bank_id = $1
                    """,
                    bank_id,
                    background,
                )

        # Return updated profile
        return await self.get_bank_profile(bank_id, request_context=request_context)

    async def submit_async_retain(
        self,
        bank_id: str,
        contents: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Submit a batch retain operation to run asynchronously."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        import json

        operation_id = uuid.uuid4()

        # Insert operation record into database
        async with acquire_with_retry(pool) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("async_operations")} (operation_id, bank_id, operation_type, result_metadata)
                VALUES ($1, $2, $3, $4)
                """,
                operation_id,
                bank_id,
                "retain",
                json.dumps({"items_count": len(contents)}),
            )

        # Submit task to background queue
        await self._task_backend.submit_task(
            {
                "type": "batch_retain",
                "operation_id": str(operation_id),
                "bank_id": bank_id,
                "contents": contents,
            }
        )

        logger.info(f"Retain task queued for bank_id={bank_id}, {len(contents)} items, operation_id={operation_id}")

        return {
            "operation_id": str(operation_id),
            "items_count": len(contents),
        }

    # =========================================================================
    # Feedback Signal Methods
    # =========================================================================

    # Signal weights for usefulness score calculation
    SIGNAL_WEIGHTS = {
        "used": 1.0,
        "ignored": -0.5,
        "helpful": 1.5,
        "not_helpful": -1.0,
    }

    # Cosine similarity threshold for matching similar queries
    # Queries with similarity >= this threshold are considered "the same" query context
    QUERY_SIMILARITY_THRESHOLD = 0.85

    async def submit_signals(
        self,
        bank_id: str,
        signals: list[dict],
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Submit feedback signals for facts.

        This method updates both global usefulness scores (for backwards compatibility)
        and query-context aware scores (for accurate per-query feedback).

        Args:
            bank_id: The memory bank ID
            signals: List of signal dicts with fact_id, signal_type, confidence, query, context
            request_context: Request context for authentication

        Returns:
            Dict with success status and list of updated fact IDs
        """
        import hashlib

        await self._authenticate_tenant(request_context)

        pool = await self._get_pool()
        updated_facts = set()

        # Step 1: Extract unique queries and generate embeddings
        unique_queries = list({s["query"] for s in signals if s.get("query")})
        query_embeddings: dict[str, list[float]] = {}

        if unique_queries:
            embeddings_list = embedding_utils.generate_embedding(self.embeddings, unique_queries[0])
            # For batch, we need to generate all embeddings
            if len(unique_queries) == 1:
                query_embeddings[unique_queries[0]] = embeddings_list
            else:
                # Generate embeddings for all unique queries
                all_embeddings = await embedding_utils.generate_embeddings_batch(
                    self.embeddings, unique_queries
                )
                for query_text, emb in zip(unique_queries, all_embeddings):
                    query_embeddings[query_text] = emb

        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                for signal in signals:
                    fact_id = signal["fact_id"]
                    signal_type = signal["signal_type"]
                    confidence = signal.get("confidence", 1.0)
                    query = signal["query"]  # Now required
                    context = signal.get("context")

                    # Generate query hash for pattern tracking
                    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

                    # Get the query embedding
                    query_embedding = query_embeddings.get(query)
                    query_embedding_str = str(query_embedding) if query_embedding else None

                    # Insert signal record with query embedding
                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table("usefulness_signals")}
                        (fact_id, bank_id, signal_type, confidence, query_hash, context, query_embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        uuid.UUID(fact_id),
                        bank_id,
                        signal_type,
                        confidence,
                        query_hash,
                        context,
                        query_embedding_str,
                    )

                    # Calculate delta for usefulness score
                    weight = self.SIGNAL_WEIGHTS.get(signal_type, 0)
                    delta = weight * confidence * 0.1

                    # Update global fact_usefulness (for backwards compatibility)
                    # Decay formula: new_score = old_score * (0.95 ^ (seconds_since_last_decay / 604800))
                    # 604800 seconds = 1 week
                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table("fact_usefulness")}
                        (fact_id, bank_id, usefulness_score, signal_count, last_signal_at, last_decay_at)
                        VALUES ($1, $2, GREATEST(0.0, LEAST(1.0, 0.5 + $3)), 1, now(), now())
                        ON CONFLICT (fact_id) DO UPDATE SET
                            usefulness_score = GREATEST(0.0, LEAST(1.0,
                                -- Apply decay first
                                (fact_usefulness.usefulness_score *
                                 POWER(0.95, EXTRACT(EPOCH FROM (now() - fact_usefulness.last_decay_at)) / 604800.0))
                                -- Then apply new signal delta
                                + $3
                            )),
                            signal_count = fact_usefulness.signal_count + 1,
                            last_signal_at = now(),
                            last_decay_at = now(),
                            updated_at = now()
                        """,
                        uuid.UUID(fact_id),
                        bank_id,
                        delta,
                    )

                    # Update query-context aware usefulness (query_fact_usefulness)
                    if query_embedding:
                        # Find existing entry with similar query for this fact
                        # Uses cosine similarity: 1 - cosine_distance
                        existing_row = await conn.fetchrow(
                            f"""
                            SELECT id, usefulness_score, signal_count, last_decay_at,
                                   1 - (query_embedding <=> $1) as similarity
                            FROM {fq_table("query_fact_usefulness")}
                            WHERE bank_id = $2 AND fact_id = $3
                              AND 1 - (query_embedding <=> $1) >= $4
                            ORDER BY similarity DESC
                            LIMIT 1
                            """,
                            query_embedding_str,
                            bank_id,
                            uuid.UUID(fact_id),
                            self.QUERY_SIMILARITY_THRESHOLD,
                        )

                        if existing_row:
                            # Update existing entry (similar query context exists)
                            # Apply time decay then new signal
                            await conn.execute(
                                f"""
                                UPDATE {fq_table("query_fact_usefulness")}
                                SET usefulness_score = GREATEST(0.0, LEAST(1.0,
                                        (usefulness_score *
                                         POWER(0.95, EXTRACT(EPOCH FROM (now() - last_decay_at)) / 604800.0))
                                        + $1
                                    )),
                                    signal_count = signal_count + 1,
                                    last_signal_at = now(),
                                    last_decay_at = now(),
                                    updated_at = now()
                                WHERE id = $2
                                """,
                                delta,
                                existing_row["id"],
                            )
                        else:
                            # Insert new entry (new query context for this fact)
                            await conn.execute(
                                f"""
                                INSERT INTO {fq_table("query_fact_usefulness")}
                                (bank_id, fact_id, query_embedding, query_example, usefulness_score, signal_count, last_signal_at)
                                VALUES ($1, $2, $3, $4, GREATEST(0.0, LEAST(1.0, 0.5 + $5)), 1, now())
                                """,
                                bank_id,
                                uuid.UUID(fact_id),
                                query_embedding_str,
                                query[:200],  # Store truncated example
                                delta,
                            )

                    updated_facts.add(fact_id)

                    # Update query pattern stats
                    count_column = {
                        "used": "used_count",
                        "ignored": "ignored_count",
                        "helpful": "helpful_count",
                        "not_helpful": "not_helpful_count",
                    }.get(signal_type, "used_count")

                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table("query_pattern_stats")}
                        (bank_id, query_hash, query_example, total_signals, {count_column})
                        VALUES ($1, $2, $3, 1, 1)
                        ON CONFLICT (bank_id, query_hash) DO UPDATE SET
                            total_signals = query_pattern_stats.total_signals + 1,
                            {count_column} = query_pattern_stats.{count_column} + 1,
                            updated_at = now()
                        """,
                        bank_id,
                        query_hash,
                        query[:200],  # Store truncated example
                    )

        return {
            "success": True,
            "signals_processed": len(signals),
            "updated_facts": list(updated_facts),
        }

    async def get_fact_usefulness_stats(
        self,
        bank_id: str,
        fact_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """
        Get usefulness statistics for a specific fact.

        Args:
            bank_id: The memory bank ID
            fact_id: The fact UUID
            request_context: Request context for authentication

        Returns:
            Dict with usefulness stats or None if no signals exist
        """
        await self._authenticate_tenant(request_context)

        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get aggregate stats with live decay calculation
            row = await conn.fetchrow(
                f"""
                SELECT
                    fact_id,
                    usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - last_decay_at)) / 604800.0) as usefulness_score,
                    signal_count,
                    last_signal_at,
                    created_at
                FROM {fq_table("fact_usefulness")}
                WHERE fact_id = $1 AND bank_id = $2
                """,
                uuid.UUID(fact_id),
                bank_id,
            )

            if not row:
                return None

            # Get signal breakdown
            breakdown = await conn.fetch(
                f"""
                SELECT signal_type, COUNT(*) as count
                FROM {fq_table("usefulness_signals")}
                WHERE fact_id = $1 AND bank_id = $2
                GROUP BY signal_type
                """,
                uuid.UUID(fact_id),
                bank_id,
            )

            signal_breakdown = {r["signal_type"]: r["count"] for r in breakdown}

            return {
                "fact_id": str(row["fact_id"]),
                "usefulness_score": max(0.0, min(1.0, row["usefulness_score"])),
                "signal_count": row["signal_count"],
                "signal_breakdown": signal_breakdown,
                "last_signal_at": row["last_signal_at"].isoformat() if row["last_signal_at"] else None,
                "created_at": row["created_at"].isoformat(),
            }

    async def get_bank_usefulness_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get aggregate usefulness statistics for a bank.

        Args:
            bank_id: The memory bank ID
            request_context: Request context for authentication

        Returns:
            Dict with bank-level usefulness stats
        """
        await self._authenticate_tenant(request_context)

        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get aggregate stats with time decay applied
            agg_row = await conn.fetchrow(
                f"""
                SELECT
                    COUNT(*) as total_facts,
                    AVG(usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - last_decay_at)) / 604800.0)) as avg_score,
                    SUM(signal_count) as total_signals
                FROM {fq_table("fact_usefulness")}
                WHERE bank_id = $1
                """,
                bank_id,
            )

            # Get signal distribution
            dist = await conn.fetch(
                f"""
                SELECT signal_type, COUNT(*) as count
                FROM {fq_table("usefulness_signals")}
                WHERE bank_id = $1
                GROUP BY signal_type
                """,
                bank_id,
            )

            signal_distribution = {r["signal_type"]: r["count"] for r in dist}

            # Get top 10 most useful facts
            top_facts = await conn.fetch(
                f"""
                SELECT
                    fu.fact_id,
                    fu.usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - fu.last_decay_at)) / 604800.0) as score,
                    mu.text
                FROM {fq_table("fact_usefulness")} fu
                JOIN {fq_table("memory_units")} mu ON fu.fact_id = mu.id
                WHERE fu.bank_id = $1
                ORDER BY score DESC
                LIMIT 10
                """,
                bank_id,
            )

            # Get bottom 10 least useful facts
            bottom_facts = await conn.fetch(
                f"""
                SELECT
                    fu.fact_id,
                    fu.usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - fu.last_decay_at)) / 604800.0) as score,
                    mu.text
                FROM {fq_table("fact_usefulness")} fu
                JOIN {fq_table("memory_units")} mu ON fu.fact_id = mu.id
                WHERE fu.bank_id = $1
                ORDER BY score ASC
                LIMIT 10
                """,
                bank_id,
            )

            return {
                "bank_id": bank_id,
                "total_facts_with_signals": agg_row["total_facts"] or 0,
                "average_usefulness": float(agg_row["avg_score"]) if agg_row["avg_score"] else 0.5,
                "total_signals": agg_row["total_signals"] or 0,
                "signal_distribution": signal_distribution,
                "top_useful_facts": [
                    {"fact_id": str(r["fact_id"]), "score": float(r["score"]), "text": r["text"][:100] if r["text"] else ""}
                    for r in top_facts
                ],
                "least_useful_facts": [
                    {"fact_id": str(r["fact_id"]), "score": float(r["score"]), "text": r["text"][:100] if r["text"] else ""}
                    for r in bottom_facts
                ],
            }

    async def _get_usefulness_scores_for_facts(
        self,
        conn,
        bank_id: str,
        fact_ids: list[str],
        query_embedding: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Get usefulness scores for a list of facts (internal method for recall integration).

        Uses query-context aware scoring when query_embedding is provided:
        1. First tries to find scores for similar queries (cosine similarity >= 0.85)
        2. Falls back to global scores for facts without query-specific signals
        3. Falls back to 0.5 (neutral) for facts without any signals

        Args:
            conn: Database connection
            bank_id: The memory bank ID
            fact_ids: List of fact UUIDs
            query_embedding: Optional query embedding for context-aware scoring

        Returns:
            Dict mapping fact_id to usefulness_score (0.5 default for facts without signals)
        """
        if not fact_ids:
            return {}

        uuid_list = [uuid.UUID(fid) for fid in fact_ids]

        # Default to 0.5 (neutral) for all facts
        scores = {fid: 0.5 for fid in fact_ids}

        # Step 1: Try query-context aware scores if query_embedding provided
        if query_embedding:
            query_embedding_str = str(query_embedding)

            # Find the best matching query context for each fact
            # Uses cosine similarity to find semantically similar queries
            query_rows = await conn.fetch(
                f"""
                WITH ranked_queries AS (
                    SELECT
                        fact_id,
                        usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - last_decay_at)) / 604800.0) as score,
                        1 - (query_embedding <=> $1) as similarity,
                        ROW_NUMBER() OVER (PARTITION BY fact_id ORDER BY 1 - (query_embedding <=> $1) DESC) as rn
                    FROM {fq_table("query_fact_usefulness")}
                    WHERE fact_id = ANY($2::uuid[])
                      AND bank_id = $3
                      AND 1 - (query_embedding <=> $1) >= $4
                )
                SELECT fact_id, score, similarity
                FROM ranked_queries
                WHERE rn = 1
                """,
                query_embedding_str,
                uuid_list,
                bank_id,
                self.QUERY_SIMILARITY_THRESHOLD,
            )

            # Apply query-context scores
            for row in query_rows:
                scores[str(row["fact_id"])] = max(0.0, min(1.0, row["score"]))

        # Step 2: Fill in remaining facts with global scores
        # (facts that didn't have query-specific signals)
        facts_needing_global = [fid for fid in fact_ids if scores[fid] == 0.5]

        if facts_needing_global:
            global_uuid_list = [uuid.UUID(fid) for fid in facts_needing_global]

            global_rows = await conn.fetch(
                f"""
                SELECT
                    fact_id,
                    usefulness_score * POWER(0.95, EXTRACT(EPOCH FROM (now() - last_decay_at)) / 604800.0) as score
                FROM {fq_table("fact_usefulness")}
                WHERE fact_id = ANY($1::uuid[]) AND bank_id = $2
                """,
                global_uuid_list,
                bank_id,
            )

            for row in global_rows:
                fact_id_str = str(row["fact_id"])
                # Only update if still at default (0.5) - don't override query-specific scores
                if scores[fact_id_str] == 0.5:
                    scores[fact_id_str] = max(0.0, min(1.0, row["score"]))

        return scores
