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
from ..metrics import get_metrics_collector
from .db_budget import budgeted_operation

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

from ..metrics import get_metrics_collector
from ..pg0 import EmbeddedPostgres, parse_pg0_url
from .entity_resolver import EntityResolver
from .llm_wrapper import LLMConfig
from .query_analyzer import QueryAnalyzer
from .reflect import run_reflect_agent
from .reflect.models import MentalModelInput
from .reflect.tools import tool_expand, tool_learn, tool_lookup, tool_recall
from .response_models import (
    VALID_RECALL_FACT_TYPES,
    EntityObservation,
    EntityState,
    LLMCallTrace,
    MemoryFact,
    MentalModelRef,
    ReflectResult,
    TokenUsage,
    ToolCallTrace,
)
from .response_models import RecallResult as RecallResultModel
from .retain import bank_utils, embedding_utils
from .retain.types import RetainContentDict
from .search import think_utils
from .search.reranking import CrossEncoderReranker
from .search.tags import TagsMatch
from .task_backend import AsyncIOQueueBackend, NoopTaskBackend, TaskBackend


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
        # Per-operation LLM config (optional, falls back to memory_llm_* params)
        retain_llm_provider: str | None = None,
        retain_llm_api_key: str | None = None,
        retain_llm_model: str | None = None,
        retain_llm_base_url: str | None = None,
        reflect_llm_provider: str | None = None,
        reflect_llm_api_key: str | None = None,
        reflect_llm_model: str | None = None,
        reflect_llm_base_url: str | None = None,
        embeddings: Embeddings | None = None,
        cross_encoder: CrossEncoderModel | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        db_command_timeout: int | None = None,
        db_acquire_timeout: int | None = None,
        task_backend: TaskBackend | None = None,
        task_batch_size: int | None = None,
        task_batch_interval: float | None = None,
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
            retain_llm_provider: LLM provider for retain operations. Falls back to memory_llm_provider.
            retain_llm_api_key: API key for retain LLM. Falls back to memory_llm_api_key.
            retain_llm_model: Model for retain operations. Falls back to memory_llm_model.
            retain_llm_base_url: Base URL for retain LLM. Falls back to memory_llm_base_url.
            reflect_llm_provider: LLM provider for reflect operations. Falls back to memory_llm_provider.
            reflect_llm_api_key: API key for reflect LLM. Falls back to memory_llm_api_key.
            reflect_llm_model: Model for reflect operations. Falls back to memory_llm_model.
            reflect_llm_base_url: Base URL for reflect LLM. Falls back to memory_llm_base_url.
            embeddings: Embeddings implementation. If not provided, created from env vars.
            cross_encoder: Cross-encoder model. If not provided, created from env vars.
            query_analyzer: Query analyzer implementation. If not provided, uses DateparserQueryAnalyzer.
            pool_min_size: Minimum number of connections in the pool. Defaults to HINDSIGHT_API_DB_POOL_MIN_SIZE.
            pool_max_size: Maximum number of connections in the pool. Defaults to HINDSIGHT_API_DB_POOL_MAX_SIZE.
            db_command_timeout: PostgreSQL command timeout in seconds. Defaults to HINDSIGHT_API_DB_COMMAND_TIMEOUT.
            db_acquire_timeout: Connection acquisition timeout in seconds. Defaults to HINDSIGHT_API_DB_ACQUIRE_TIMEOUT.
            task_backend: Custom task backend. If not provided, uses AsyncIOQueueBackend.
            task_batch_size: Background task batch size. Defaults to HINDSIGHT_API_TASK_BATCH_SIZE.
            task_batch_interval: Background task batch interval in seconds. Defaults to HINDSIGHT_API_TASK_BATCH_INTERVAL.
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
        # Ollama and mock don't require an API key
        if not memory_llm_api_key and memory_llm_provider not in ("ollama", "mock"):
            raise ValueError("LLM API key is required. Set HINDSIGHT_API_LLM_API_KEY environment variable.")
        memory_llm_model = memory_llm_model or config.llm_model
        memory_llm_base_url = memory_llm_base_url or config.get_llm_base_url() or None
        # Track pg0 instance (if used)
        self._pg0: EmbeddedPostgres | None = None

        # Initialize PostgreSQL connection URL
        # The actual URL will be set during initialize() after starting the server
        # Supports: "pg0" (default instance), "pg0://instance-name" (named instance), or regular postgresql:// URL
        self._use_pg0, self._pg0_instance_name, self._pg0_port = parse_pg0_url(db_url)
        if self._use_pg0:
            self.db_url = None
        else:
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
        self._pool_min_size = pool_min_size if pool_min_size is not None else config.db_pool_min_size
        self._pool_max_size = pool_max_size if pool_max_size is not None else config.db_pool_max_size
        self._db_command_timeout = db_command_timeout if db_command_timeout is not None else config.db_command_timeout
        self._db_acquire_timeout = db_acquire_timeout if db_acquire_timeout is not None else config.db_acquire_timeout
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

        # Initialize LLM configuration (default, used as fallback)
        self._llm_config = LLMConfig(
            provider=memory_llm_provider,
            api_key=memory_llm_api_key,
            base_url=memory_llm_base_url,
            model=memory_llm_model,
        )

        # Store client and model for convenience (deprecated: use _llm_config.call() instead)
        self._llm_client = self._llm_config._client
        self._llm_model = self._llm_config.model

        # Initialize per-operation LLM configs (fall back to default if not specified)
        # Retain LLM config - for fact extraction (benefits from strong structured output)
        retain_provider = retain_llm_provider or config.retain_llm_provider or memory_llm_provider
        retain_api_key = retain_llm_api_key or config.retain_llm_api_key or memory_llm_api_key
        retain_model = retain_llm_model or config.retain_llm_model or memory_llm_model
        retain_base_url = retain_llm_base_url or config.retain_llm_base_url or memory_llm_base_url
        # Apply provider-specific base URL defaults for retain
        if retain_base_url is None:
            if retain_provider.lower() == "groq":
                retain_base_url = "https://api.groq.com/openai/v1"
            elif retain_provider.lower() == "ollama":
                retain_base_url = "http://localhost:11434/v1"
            else:
                retain_base_url = ""

        self._retain_llm_config = LLMConfig(
            provider=retain_provider,
            api_key=retain_api_key,
            base_url=retain_base_url,
            model=retain_model,
        )

        # Reflect LLM config - for think/observe operations (can use lighter models)
        reflect_provider = reflect_llm_provider or config.reflect_llm_provider or memory_llm_provider
        reflect_api_key = reflect_llm_api_key or config.reflect_llm_api_key or memory_llm_api_key
        reflect_model = reflect_llm_model or config.reflect_llm_model or memory_llm_model
        reflect_base_url = reflect_llm_base_url or config.reflect_llm_base_url or memory_llm_base_url
        # Apply provider-specific base URL defaults for reflect
        if reflect_base_url is None:
            if reflect_provider.lower() == "groq":
                reflect_base_url = "https://api.groq.com/openai/v1"
            elif reflect_provider.lower() == "ollama":
                reflect_base_url = "http://localhost:11434/v1"
            else:
                reflect_base_url = ""

        self._reflect_llm_config = LLMConfig(
            provider=reflect_provider,
            api_key=reflect_api_key,
            base_url=reflect_base_url,
            model=reflect_model,
        )

        # Initialize cross-encoder reranker (cached for performance)
        self._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=cross_encoder)

        # Initialize task backend
        _task_batch_size = task_batch_size if task_batch_size is not None else config.task_backend_memory_batch_size
        _task_batch_interval = (
            task_batch_interval if task_batch_interval is not None else config.task_backend_memory_batch_interval
        )
        self._task_backend = task_backend or AsyncIOQueueBackend(
            batch_size=_task_batch_size, batch_interval=_task_batch_interval
        )

        # Backpressure mechanism: limit concurrent searches to prevent overwhelming the database
        # Configurable via HINDSIGHT_API_RECALL_MAX_CONCURRENT (default: 50)
        self._search_semaphore = asyncio.Semaphore(get_config().recall_max_concurrent)

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

    async def _handle_refresh_mental_models(self, task_dict: dict[str, Any]):
        """
        Handler for refresh mental models tasks.

        This is the main background job that:
        1. Identifies mental models (structural from mission + emergent from entities)
        2. Generates summaries for each mental model

        Args:
            task_dict: Dict with 'bank_id', 'operation_id', optional 'tags', optional 'subtype'
        """
        import time

        bank_id = task_dict.get("bank_id")
        operation_id = task_dict.get("operation_id")
        tags = task_dict.get("tags")  # Tags to apply to created mental models
        subtype = task_dict.get("subtype")  # Optional filter: "structural", "emergent", "pinned", or "learned"
        if not bank_id:
            raise ValueError("bank_id is required for refresh mental models task")

        refresh_structural = subtype is None or subtype == "structural"
        refresh_emergent = subtype is None or subtype == "emergent"
        refresh_pinned = subtype is None or subtype == "pinned"
        refresh_learned = subtype is None or subtype == "learned"
        subtype_desc = f" (subtype={subtype})" if subtype else " (all)"

        from hindsight_api.models import RequestContext

        internal_context = RequestContext()
        pool = await self._get_pool()

        from .mental_models.emergent import (
            detect_entity_candidates,
            evaluate_emergent_models,
            filter_candidates_by_mission,
        )

        # ===== Phase 1: Identify mental models (with buffered logging) =====
        phase1_start = time.perf_counter()
        id_log: list[str] = []  # Log buffer for identification phase

        # Step 1: Get the bank's mission (required - should have been validated before scheduling)
        profile = await self.get_bank_profile(bank_id, request_context=internal_context)
        mission = profile.get("mission") or ""
        if not mission:
            raise ValueError(f"Cannot refresh mental models: no mission is set for bank '{bank_id}'")

        structural_removed: list[str] = []
        emergent_removed: list[str] = []
        emergent_promoted: list[str] = []

        # Step 2: Derive structural models (LLM sees existing ones and decides what to keep)
        if refresh_structural:
            existing_structural = await self.list_mental_models(
                bank_id, subtype="structural", request_context=internal_context
            )
            id_log.append(f"structural: {len(existing_structural) if existing_structural else 0} existing")
            models_to_remove = await self._derive_structural_models_internal(
                bank_id, mission, pool, existing_models=existing_structural, tags=tags
            )
            for model_id in models_to_remove:
                structural_removed.append(model_id)
                await self.delete_mental_model(bank_id, model_id, request_context=internal_context)
            if structural_removed:
                id_log.append(f"structural removed: {structural_removed}")
        else:
            id_log.append("structural: skipped (subtype filter)")

        # Step 3: Evaluate existing emergent models
        removed_entity_ids: set[str] = set()  # Track entity_ids we removed (to prevent re-promotion)
        if refresh_emergent:
            existing_emergent = await self.list_mental_models(
                bank_id, subtype="emergent", request_context=internal_context
            )
            if existing_emergent:
                id_log.append(f"emergent: {len(existing_emergent)} existing")
                # Build model_id -> entity_id mapping for tracking
                model_to_entity = {m["id"]: m.get("entity_id") for m in existing_emergent}
                models_to_remove = await evaluate_emergent_models(self._llm_config, existing_emergent)
                for model_id in models_to_remove:
                    emergent_removed.append(model_id)
                    # Track the entity_id so we don't re-promote it
                    entity_id = model_to_entity.get(model_id)
                    if entity_id:
                        removed_entity_ids.add(str(entity_id))
                    await self.delete_mental_model(bank_id, model_id, request_context=internal_context)
                if emergent_removed:
                    id_log.append(f"emergent removed: {emergent_removed}")
            else:
                id_log.append("emergent: 0 existing")

            # Step 4: Detect emergent candidates (entities worth promoting)
            candidates = await detect_entity_candidates(pool, bank_id)
            id_log.append(f"emergent candidates detected: {len(candidates)}")

            # Step 5: Filter candidates by mission relevance
            if candidates and mission:
                candidates = await filter_candidates_by_mission(self._llm_config, mission, candidates)
                id_log.append(f"emergent candidates after mission filter: {len(candidates)}")

            # Step 6: Filter out candidates whose entity was just removed (they failed evaluation)
            if removed_entity_ids:
                original_count = len(candidates)
                candidates = [c for c in candidates if c.entity_id not in removed_entity_ids]
                if len(candidates) < original_count:
                    id_log.append(f"emergent excluded (failed evaluation): {original_count - len(candidates)}")

            # Step 7: Promote filtered candidates to mental models (with tags if provided)
            for candidate in candidates:
                if candidate.entity_id:
                    emergent_promoted.append(candidate.name)
                    await self._promote_entity_internal(bank_id, candidate.entity_id, pool, tags=tags)
            if emergent_promoted:
                id_log.append(f"emergent promoted: {emergent_promoted}")
        else:
            id_log.append("emergent: skipped (subtype filter)")

        phase1_duration_ms = int((time.perf_counter() - phase1_start) * 1000)

        # Output single log for Phase 1
        logger.info(
            f"[MENTAL_MODELS] Identification complete for bank={bank_id}{subtype_desc} "
            f"in {phase1_duration_ms}ms: {', '.join(id_log)}"
        )

        # ===== Phase 2: Generate summaries in parallel =====
        models = await self.list_mental_models(bank_id, request_context=internal_context)

        # Filter models to only those being refreshed based on subtype
        models_to_refresh = []
        for m in models:
            model_subtype = m["subtype"]
            if model_subtype == "structural" and refresh_structural:
                models_to_refresh.append(m)
            elif model_subtype == "emergent" and refresh_emergent:
                models_to_refresh.append(m)
            elif model_subtype == "pinned" and refresh_pinned:
                models_to_refresh.append(m)
            elif model_subtype == "learned" and refresh_learned:
                models_to_refresh.append(m)

        # Get concurrency limit from config
        from ..config import get_config

        config = get_config()
        concurrency = config.mental_model_refresh_concurrency

        # Use semaphore to limit concurrent refreshes
        semaphore = asyncio.Semaphore(concurrency)
        # Track results with timing: model_id -> {status, duration_ms, iterations, tool_calls, observations}
        refresh_results: dict[str, dict[str, Any]] = {}

        async def refresh_with_semaphore(model: dict) -> None:
            """Refresh a single model with semaphore-controlled concurrency."""
            async with semaphore:
                model_id = model["id"]
                model_name = model["name"]
                start_time = time.perf_counter()
                try:
                    result = await self.refresh_mental_model(
                        bank_id=bank_id,
                        model_id=model_id,
                        request_context=internal_context,
                        _return_agent_result=True,  # Get agent stats for logging
                    )
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    if result and isinstance(result, tuple):
                        _, agent_result = result
                        refresh_results[model_id] = {
                            "status": "success",
                            "name": model_name,
                            "duration_ms": duration_ms,
                            "iterations": agent_result.iterations if agent_result else 0,
                            "tool_calls": agent_result.tools_called if agent_result else 0,
                            "observations": len(agent_result.observations) if agent_result else 0,
                        }
                    else:
                        refresh_results[model_id] = {
                            "status": "success",
                            "name": model_name,
                            "duration_ms": duration_ms,
                        }
                except Exception as e:
                    duration_ms = int((time.perf_counter() - start_time) * 1000)
                    refresh_results[model_id] = {
                        "status": "failed",
                        "name": model_name,
                        "duration_ms": duration_ms,
                        "error": str(e),
                    }

        # Run all refreshes in parallel (bounded by semaphore)
        phase2_start = time.perf_counter()
        await asyncio.gather(*[refresh_with_semaphore(m) for m in models_to_refresh])
        phase2_duration_ms = int((time.perf_counter() - phase2_start) * 1000)

        # Build summary for each model
        model_summaries: list[str] = []
        for model_id, info in refresh_results.items():
            if info["status"] == "success":
                parts = [f"{info['name']}"]
                if "iterations" in info:
                    parts.append(f"iter={info['iterations']}")
                if "tool_calls" in info:
                    parts.append(f"tools={info['tool_calls']}")
                if "observations" in info:
                    parts.append(f"obs={info['observations']}")
                parts.append(f"{info['duration_ms']}ms")
                model_summaries.append(f"[{' '.join(parts)}]")
            else:
                model_summaries.append(
                    f"[{info['name']} FAILED: {info.get('error', 'unknown')} {info['duration_ms']}ms]"
                )

        success_count = sum(1 for r in refresh_results.values() if r["status"] == "success")
        failed_count = len(refresh_results) - success_count

        # Output single log for Phase 2
        logger.info(
            f"[MENTAL_MODELS] Refresh complete for bank={bank_id}, operation={operation_id}: "
            f"{success_count}/{len(models_to_refresh)} succeeded in {phase2_duration_ms}ms (concurrency={concurrency}). "
            f"Models: {' '.join(model_summaries)}"
        )

    async def _handle_generate_mental_model(self, task_dict: dict[str, Any]):
        """
        Handler for single mental model generation tasks.

        Generates/refreshes content for a specific mental model.

        Args:
            task_dict: Dict with 'bank_id', 'model_id', 'operation_id'
        """
        bank_id = task_dict.get("bank_id")
        model_id = task_dict.get("model_id")
        operation_id = task_dict.get("operation_id")

        if not bank_id or not model_id:
            raise ValueError("bank_id and model_id are required for generate mental model task")

        logger.info(
            f"[MENTAL_MODEL_TASK] Starting generation for model_id={model_id}, bank_id={bank_id}, operation_id={operation_id}"
        )

        from hindsight_api.models import RequestContext

        internal_context = RequestContext()

        # Generate content for the model (reuses the same logic as refresh)
        result = await self.refresh_mental_model(
            bank_id=bank_id,
            model_id=model_id,
            request_context=internal_context,
        )

        if result:
            logger.info(f"[MENTAL_MODEL_TASK] Completed generation for model_id={model_id}, bank_id={bank_id}")
        else:
            logger.warning(f"[MENTAL_MODEL_TASK] Model not found: model_id={model_id}, bank_id={bank_id}")

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
            elif task_type == "batch_retain":
                await self._handle_batch_retain(task_dict)
            elif task_type == "refresh_mental_models":
                await self._handle_refresh_mental_models(task_dict)
            elif task_type == "generate_mental_model":
                await self._handle_generate_mental_model(task_dict)
            else:
                logger.error(f"Unknown task type: {task_type}")
                # Don't retry unknown task types
                if operation_id:
                    await self._delete_operation_record(operation_id)
                return

            # Task succeeded - mark operation as completed
            if operation_id:
                await self._mark_operation_completed(operation_id)

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
                    SET status = 'failed', error_message = $2, updated_at = NOW()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(operation_id),
                    truncated_error,
                )
            logger.info(f"Marked async operation as failed: {operation_id}")
        except Exception as e:
            logger.error(f"Failed to mark operation as failed {operation_id}: {e}")

    async def _mark_operation_completed(self, operation_id: str):
        """Helper to mark an operation as completed in the database."""
        try:
            pool = await self._get_pool()
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET status = 'completed', updated_at = NOW(), completed_at = NOW()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(operation_id),
                )
            logger.info(f"Marked async operation as completed: {operation_id}")
        except Exception as e:
            logger.error(f"Failed to mark operation as completed {operation_id}: {e}")

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
            """Verify LLM connections are working for all unique configs."""
            if not self._skip_llm_verification:
                # Verify default config
                await self._llm_config.verify_connection()
                # Verify retain config if different from default
                retain_is_different = (
                    self._retain_llm_config.provider != self._llm_config.provider
                    or self._retain_llm_config.model != self._llm_config.model
                )
                if retain_is_different:
                    await self._retain_llm_config.verify_connection()
                # Verify reflect config if different from default and retain
                reflect_is_different = (
                    self._reflect_llm_config.provider != self._llm_config.provider
                    or self._reflect_llm_config.model != self._llm_config.model
                ) and (
                    self._reflect_llm_config.provider != self._retain_llm_config.provider
                    or self._reflect_llm_config.model != self._retain_llm_config.model
                )
                if reflect_is_different:
                    await self._reflect_llm_config.verify_connection()

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
            command_timeout=self._db_command_timeout,
            statement_cache_size=0,  # Disable prepared statement cache
            timeout=self._db_acquire_timeout,  # Connection acquisition timeout (seconds)
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

        This is useful in tests to ensure background tasks complete before making assertions.
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
        document_tags: list[str] | None = None,
        return_usage: bool = False,
    ):
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
            return_usage: If True, returns tuple of (unit_ids, TokenUsage). Default False for backward compatibility.

        Returns:
            If return_usage=False: List of lists of unit IDs (one list per content item)
            If return_usage=True: Tuple of (unit_ids, TokenUsage)

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
            if return_usage:
                return [], TokenUsage()
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
        total_usage = TokenUsage()

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

            # Process each sub-batch
            all_results = []
            for i, sub_batch in enumerate(sub_batches, 1):
                sub_batch_chars = sum(len(item.get("content", "")) for item in sub_batch)
                logger.info(
                    f"Processing sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_chars:,} chars"
                )

                sub_results, sub_usage = await self._retain_batch_async_internal(
                    bank_id=bank_id,
                    contents=sub_batch,
                    document_id=document_id,
                    is_first_batch=i == 1,  # Only upsert on first batch
                    fact_type_override=fact_type_override,
                    confidence_score=confidence_score,
                    document_tags=document_tags,
                )
                all_results.extend(sub_results)
                total_usage = total_usage + sub_usage

            total_time = time.time() - start_time
            logger.info(
                f"RETAIN_BATCH_ASYNC (chunked) COMPLETE: {len(all_results)} results from {len(contents)} contents in {total_time:.3f}s"
            )
            result = all_results
        else:
            # Small batch - use internal method directly
            result, total_usage = await self._retain_batch_async_internal(
                bank_id=bank_id,
                contents=contents,
                document_id=document_id,
                is_first_batch=True,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
                document_tags=document_tags,
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

        if return_usage:
            return result, total_usage
        return result

    async def _retain_batch_async_internal(
        self,
        bank_id: str,
        contents: list[RetainContentDict],
        document_id: str | None = None,
        is_first_batch: bool = True,
        fact_type_override: str | None = None,
        confidence_score: float | None = None,
        document_tags: list[str] | None = None,
    ) -> tuple[list[list[str]], "TokenUsage"]:
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
            document_tags: Tags applied to all items in this batch

        Returns:
            Tuple of (unit ID lists, token usage for fact extraction)
        """
        # Backpressure: limit concurrent retains to prevent database contention
        async with self._put_semaphore:
            # Use the new modular orchestrator
            from .retain import orchestrator

            pool = await self._get_pool()
            return await orchestrator.retain_batch(
                pool=pool,
                embeddings_model=self.embeddings,
                llm_config=self._retain_llm_config,
                entity_resolver=self.entity_resolver,
                format_date_fn=self._format_readable_date,
                duplicate_checker_fn=self._find_duplicate_facts_batch,
                bank_id=bank_id,
                contents_dicts=contents,
                document_id=document_id,
                is_first_batch=is_first_batch,
                fact_type_override=fact_type_override,
                confidence_score=confidence_score,
                document_tags=document_tags,
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
        request_context: "RequestContext",
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        _connection_budget: int | None = None,
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
            tags: Optional list of tags for visibility filtering (OR matching - returns
                  memories that have at least one matching tag)

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

        # Filter out 'opinion' - opinions are no longer returned from recall
        # (learnings are now stored as mental models instead)
        fact_type = [ft for ft in fact_type if ft != "opinion"]
        if not fact_type:
            # All requested types were opinions - return empty result
            return RecallResult(results=[], entities={}, chunks={})

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

        # Log recall start with tags if present
        tags_info = f", tags={tags} ({tags_match})" if tags else ""
        logger.info(f"[RECALL {bank_id[:8]}] Starting recall for query: {query[:50]}...{tags_info}")

        # Backpressure: limit concurrent recalls to prevent overwhelming the database
        result = None
        error_msg = None
        semaphore_wait_start = time.time()
        async with self._search_semaphore:
            semaphore_wait = time.time() - semaphore_wait_start
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
                        request_context,
                        semaphore_wait=semaphore_wait,
                        tags=tags,
                        tags_match=tags_match,
                        connection_budget=_connection_budget,
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
        request_context: "RequestContext" = None,
        semaphore_wait: float = 0.0,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        connection_budget: int | None = None,
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

        tracer = (
            SearchTracer(query, thinking_budget, max_tokens, tags=tags, tags_match=tags_match) if enable_trace else None
        )
        if tracer:
            tracer.start()

        pool = await self._get_pool()
        recall_start = time.time()

        # Buffer logs for clean output in concurrent scenarios
        recall_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}"
        log_buffer = []
        tags_info = f", tags={tags}, tags_match={tags_match}" if tags else ""
        log_buffer.append(
            f"[RECALL {recall_id}] Query: '{query[:50]}...' (budget={thinking_budget}, max_tokens={max_tokens}{tags_info})"
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

            # Step 2: Optimized parallel retrieval using batched queries
            # - Semantic + BM25 combined in 1 CTE query for ALL fact types
            # - Graph runs per fact type (complex traversal)
            # - Temporal runs per fact type (if constraint detected)
            step_start = time.time()
            query_embedding_str = str(query_embedding)

            from .search.retrieval import (
                get_default_graph_retriever,
                retrieve_all_fact_types_parallel,
            )

            # Track each retrieval start time
            retrieval_start = time.time()

            # Run optimized retrieval with connection budget
            config = get_config()
            effective_connection_budget = (
                connection_budget if connection_budget is not None else config.recall_connection_budget
            )
            async with budgeted_operation(
                max_connections=effective_connection_budget,
                operation_id=f"recall-{recall_id}",
            ) as op:
                budgeted_pool = op.wrap_pool(pool)
                parallel_start = time.time()
                multi_result = await retrieve_all_fact_types_parallel(
                    budgeted_pool,
                    query,
                    query_embedding_str,
                    bank_id,
                    fact_type,  # Pass all fact types at once
                    thinking_budget,
                    question_date,
                    self.query_analyzer,
                    tags=tags,
                    tags_match=tags_match,
                )
                parallel_duration = time.time() - parallel_start

            # Combine all results from all fact types and aggregate timings
            semantic_results = []
            bm25_results = []
            graph_results = []
            temporal_results = []
            aggregated_timings = {
                "semantic": 0.0,
                "bm25": 0.0,
                "graph": 0.0,
                "temporal": 0.0,
                "temporal_extraction": 0.0,
            }
            all_mpfp_timings = []

            detected_temporal_constraint = None
            max_conn_wait = multi_result.max_conn_wait
            for ft in fact_type:
                retrieval_result = multi_result.results_by_fact_type.get(ft)
                if not retrieval_result:
                    continue

                # Log fact types in this retrieval batch
                logger.debug(
                    f"[RECALL {recall_id}] Fact type '{ft}': semantic={len(retrieval_result.semantic)}, bm25={len(retrieval_result.bm25)}, graph={len(retrieval_result.graph)}, temporal={len(retrieval_result.temporal) if retrieval_result.temporal else 0}"
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
                f"temporal_extraction={aggregated_timings['temporal_extraction']:.3f}s",
            ]
            temporal_info = ""
            if detected_temporal_constraint:
                start_dt, end_dt = detected_temporal_constraint
                temporal_count = len(temporal_results) if temporal_results else 0
                timing_parts.append(f"temporal={temporal_count}({aggregated_timings['temporal']:.3f}s)")
                temporal_info = f" | temporal_range={start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
            log_buffer.append(
                f"  [2] Parallel retrieval ({len(fact_type)} fact_types): {', '.join(timing_parts)} in {parallel_duration:.3f}s{temporal_info}"
            )

            # Log graph retriever timing breakdown if available
            if all_mpfp_timings:
                retriever_name = get_default_graph_retriever().name.upper()
                mpfp_total = all_mpfp_timings[0]  # Take first fact type's timing as representative
                mpfp_parts = [
                    f"db_queries={mpfp_total.db_queries}",
                    f"edge_load={mpfp_total.edge_load_time:.3f}s",
                    f"edges={mpfp_total.edge_count}",
                    f"patterns={mpfp_total.pattern_count}",
                ]
                if mpfp_total.seeds_time > 0.01:
                    mpfp_parts.append(f"seeds={mpfp_total.seeds_time:.3f}s")
                if mpfp_total.fusion > 0.001:
                    mpfp_parts.append(f"fusion={mpfp_total.fusion:.3f}s")
                if mpfp_total.fetch > 0.001:
                    mpfp_parts.append(f"fetch={mpfp_total.fetch:.3f}s")
                log_buffer.append(f"      [{retriever_name}] {', '.join(mpfp_parts)}")
                # Log detailed hop timing for debugging slow queries
                if mpfp_total.hop_details:
                    for hd in mpfp_total.hop_details:
                        log_buffer.append(
                            f"        hop{hd['hop']}: exec={hd.get('exec_time', 0) * 1000:.0f}ms, "
                            f"uncached={hd.get('uncached_after_filter', 0)}, "
                            f"load={hd.get('load_time', 0) * 1000:.0f}ms, "
                            f"edges={hd.get('edges_loaded', 0)}"
                        )

            # Record temporal constraint in tracer if detected
            if tracer and detected_temporal_constraint:
                start_dt, end_dt = detected_temporal_constraint
                tracer.record_temporal_constraint(start_dt, end_dt)

            # Record retrieval results for tracer - per fact type
            if tracer:
                # Convert RetrievalResult to old tuple format for tracer
                def to_tuple_format(results):
                    return [(r.id, r.__dict__) for r in results]

                # Add retrieval results per fact type (to show parallel execution in UI)
                for ft_name in fact_type:
                    rr = multi_result.results_by_fact_type.get(ft_name)
                    if not rr:
                        continue

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

                    # Add temporal retrieval results for this fact type
                    # Show temporal even with 0 results if constraint was detected
                    if rr.temporal is not None or rr.temporal_constraint is not None:
                        temporal_metadata = {"budget": thinking_budget}
                        if rr.temporal_constraint:
                            start_dt, end_dt = rr.temporal_constraint
                            temporal_metadata["constraint"] = {
                                "start": start_dt.isoformat() if start_dt else None,
                                "end": end_dt.isoformat() if end_dt else None,
                            }
                        tracer.add_retrieval_results(
                            method_name="temporal",
                            results=to_tuple_format(rr.temporal or []),
                            duration_seconds=rr.timings.get("temporal", 0.0),
                            score_field="temporal_score",
                            metadata=temporal_metadata,
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

            # Pre-filter candidates to reduce reranking cost (RRF already provides good ranking)
            # This is especially important for remote rerankers with network latency
            reranker_max_candidates = get_config().reranker_max_candidates
            pre_filtered_count = 0
            if len(merged_candidates) > reranker_max_candidates:
                # Sort by RRF score and take top candidates
                merged_candidates.sort(key=lambda mc: mc.rrf_score, reverse=True)
                pre_filtered_count = len(merged_candidates) - reranker_max_candidates
                merged_candidates = merged_candidates[:reranker_max_candidates]

            # Rerank using cross-encoder
            scored_results = await reranker_instance.rerank(query, merged_candidates)

            step_duration = time.time() - step_start
            pre_filter_note = f" (pre-filtered {pre_filtered_count})" if pre_filtered_count > 0 else ""
            log_buffer.append(
                f"  [4] Reranking: {len(scored_results)} candidates scored in {step_duration:.3f}s{pre_filter_note}"
            )

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
                await self._task_backend.submit_task(
                    {
                        "type": "access_count_update",
                        "bank_id": bank_id,
                        "node_ids": visited_ids,
                    }
                )
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
                        tags=result_dict.get("tags"),
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

                # Return entities with empty observations (summaries now live in mental models)
                entities_dict = {}
                for entity_id, entity_name in entities_ordered:
                    entities_dict[entity_name] = EntityState(
                        entity_id=entity_id,
                        canonical_name=entity_name,
                        observations=[],  # Mental models provide this now
                    )

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
            # Include wait times in log if significant
            wait_parts = []
            if semaphore_wait > 0.01:
                wait_parts.append(f"sem={semaphore_wait:.3f}s")
            if max_conn_wait > 0.01:
                wait_parts.append(f"conn={max_conn_wait:.3f}s")
            wait_info = f" | waits: {', '.join(wait_parts)}" if wait_parts else ""
            log_buffer.append(
                f"[RECALL {recall_id}] Complete: {len(top_scored)} facts ({total_tokens} tok), {num_chunks} chunks ({total_chunk_tokens} tok), {num_entities} entities ({total_entity_tokens} tok) | {fact_type_summary} | {total_time:.3f}s{wait_info}"
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
                       d.created_at, d.updated_at, d.tags, COUNT(mu.id) as unit_count
                FROM {fq_table("documents")} d
                LEFT JOIN {fq_table("memory_units")} mu ON mu.document_id = d.id
                WHERE d.id = $1 AND d.bank_id = $2
                GROUP BY d.id, d.bank_id, d.original_text, d.content_hash, d.created_at, d.updated_at, d.tags
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
                "tags": list(doc["tags"]) if doc["tags"] else [],
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
                # Get memory unit IDs before deletion (for mental model invalidation)
                unit_rows = await conn.fetch(
                    f"SELECT id FROM {fq_table('memory_units')} WHERE document_id = $1", document_id
                )
                unit_ids = [str(row["id"]) for row in unit_rows]
                units_count = len(unit_ids)

                # Delete document (cascades to memory_units and all their links)
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id",
                    document_id,
                    bank_id,
                )

                # Invalidate deleted fact IDs from mental models
                if deleted and unit_ids:
                    await self._invalidate_facts_from_mental_models(conn, bank_id, unit_ids)

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
                # Get bank_id before deletion (for mental model invalidation)
                bank_id = await conn.fetchval(f"SELECT bank_id FROM {fq_table('memory_units')} WHERE id = $1", unit_id)

                # Delete the memory unit (cascades to links and associations)
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 RETURNING id", unit_id
                )

                # Invalidate deleted fact ID from mental models
                if deleted and bank_id:
                    await self._invalidate_facts_from_mental_models(conn, bank_id, [str(deleted)])

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
        limit: int = 1000,
        request_context: "RequestContext",
    ):
        """
        Get graph data for visualization.

        Args:
            bank_id: Filter by bank ID
            fact_type: Filter by fact type (world, experience, opinion)
            limit: Maximum number of items to return (default: 1000)
            request_context: Request context for authentication.

        Returns:
            Dict with nodes, edges, table_rows, total_units, and limit
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

            # Get total count first
            total_count_result = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as total
                FROM {fq_table("memory_units")}
                {where_clause}
            """,
                *query_params,
            )
            total_count = total_count_result["total"] if total_count_result else 0

            # Get units with limit
            param_count += 1
            units = await conn.fetch(
                f"""
                SELECT id, text, event_date, context, occurred_start, occurred_end, mentioned_at, document_id, chunk_id, fact_type
                FROM {fq_table("memory_units")}
                {where_clause}
                ORDER BY mentioned_at DESC NULLS LAST, event_date DESC
                LIMIT ${param_count}
            """,
                *query_params,
                limit,
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

        return {"nodes": nodes, "edges": edges, "table_rows": table_rows, "total_units": total_count, "limit": limit}

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

    async def get_memory_unit(
        self,
        bank_id: str,
        memory_id: str,
        request_context: "RequestContext",
    ):
        """
        Get a single memory unit by ID.

        Args:
            bank_id: Bank ID
            memory_id: Memory unit ID
            request_context: Request context for authentication.

        Returns:
            Dict with memory unit data or None if not found
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get the memory unit
            row = await conn.fetchrow(
                f"""
                SELECT id, text, context, event_date, occurred_start, occurred_end,
                       mentioned_at, fact_type, document_id, chunk_id, tags
                FROM {fq_table("memory_units")}
                WHERE id = $1 AND bank_id = $2
                """,
                memory_id,
                bank_id,
            )

            if not row:
                return None

            # Get entity information
            entities_rows = await conn.fetch(
                f"""
                SELECT e.canonical_name
                FROM {fq_table("unit_entities")} ue
                JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                WHERE ue.unit_id = $1
                """,
                row["id"],
            )
            entities = [r["canonical_name"] for r in entities_rows]

            return {
                "id": str(row["id"]),
                "text": row["text"],
                "context": row["context"] if row["context"] else "",
                "date": row["event_date"].isoformat() if row["event_date"] else "",
                "type": row["fact_type"],
                "mentioned_at": row["mentioned_at"].isoformat() if row["mentioned_at"] else None,
                "occurred_start": row["occurred_start"].isoformat() if row["occurred_start"] else None,
                "occurred_end": row["occurred_end"].isoformat() if row["occurred_end"] else None,
                "entities": entities,
                "document_id": row["document_id"] if row["document_id"] else None,
                "chunk_id": str(row["chunk_id"]) if row["chunk_id"] else None,
                "tags": row["tags"] if row["tags"] else [],
            }

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

    # ==================== bank profile Methods ====================

    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get bank profile (name, disposition + mission).
        Auto-creates agent with default values if not exists.

        Args:
            bank_id: bank IDentifier
            request_context: Request context for authentication.

        Returns:
            Dict with name, disposition traits, and mission
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        profile = await bank_utils.get_bank_profile(pool, bank_id)
        disposition = profile["disposition"]
        return {
            "bank_id": bank_id,
            "name": profile["name"],
            "disposition": disposition,
            "mission": profile["mission"],
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

    async def set_bank_mission(
        self,
        bank_id: str,
        mission: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Set the mission for a bank.

        Args:
            bank_id: bank IDentifier
            mission: The mission text
            request_context: Request context for authentication.

        Returns:
            Dict with bank_id and mission.
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        await bank_utils.set_bank_mission(pool, bank_id, mission)
        return {"bank_id": bank_id, "mission": mission}

    async def merge_bank_mission(
        self,
        bank_id: str,
        new_info: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Merge new mission information with existing mission using LLM.
        Normalizes to first person ("I") and resolves conflicts.

        Args:
            bank_id: bank IDentifier
            new_info: New mission information to add/merge
            request_context: Request context for authentication.

        Returns:
            Dict with 'mission' (str) key
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        return await bank_utils.merge_bank_mission(pool, self._reflect_llm_config, bank_id, new_info)

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
            List of dicts with bank_id, name, disposition, mission, created_at, updated_at
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
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
    ) -> ReflectResult:
        """
        Reflect and formulate an answer using an agentic loop with tools.

        The reflect agent iteratively uses tools to:
        1. lookup: Get mental models (synthesized knowledge)
        2. recall: Search facts (semantic + temporal retrieval)
        3. learn: Create/update mental models with new insights
        4. expand: Get chunk/document context for memories

        The agent starts with empty context and must call tools to gather
        information. On the last iteration, tools are removed to force a
        final text response.

        Args:
            bank_id: bank identifier
            query: Question to answer
            budget: Budget level (currently unused, reserved for future)
            context: Additional context string to include in agent prompt
            max_tokens: Max tokens (currently unused, reserved for future)
            response_schema: Optional JSON Schema for structured output (not yet supported)

        Returns:
            ReflectResult containing:
                - text: Plain text answer
                - based_on: Empty dict (agent retrieves facts dynamically)
                - new_opinions: Empty list (learnings stored as mental models)
                - structured_output: None (not yet supported for agentic reflect)
        """
        # Use cached LLM config
        if self._reflect_llm_config is None:
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
        tags_info = f", tags={tags} ({tags_match})" if tags else ""
        logger.info(f"[REFLECT {reflect_id}] Starting agentic reflect for query: {query[:50]}...{tags_info}")

        # Get bank profile for agent identity
        profile = await self.get_bank_profile(bank_id, request_context=request_context)

        # NOTE: Mental models are NOT pre-loaded to keep the initial prompt small.
        # The agent can call lookup() to list available models if needed.
        # This is critical for banks with many mental models to avoid huge prompts.

        # Compute max iterations based on budget
        config = get_config()
        base_max_iterations = config.reflect_max_iterations
        # Budget multipliers: low=0.5x, mid=1x, high=2x
        budget_multipliers = {Budget.LOW: 0.5, Budget.MID: 1.0, Budget.HIGH: 2.0}
        effective_budget = budget or Budget.LOW
        max_iterations = max(1, int(base_max_iterations * budget_multipliers.get(effective_budget, 1.0)))

        # Run agentic loop - acquire connections only when needed for DB operations
        # (not held during LLM calls which can be slow)
        pool = await self._get_pool()

        # Create tool callbacks that acquire connections only when needed
        async def lookup_fn(model_id: str | None = None) -> dict[str, Any]:
            async with pool.acquire() as conn:
                return await tool_lookup(conn, bank_id, model_id, tags=tags, tags_match=tags_match)

        async def recall_fn(q: str, max_tokens: int = 4096) -> dict[str, Any]:
            return await tool_recall(
                self, bank_id, q, request_context, max_tokens=max_tokens, tags=tags, tags_match=tags_match
            )

        async def learn_fn(input: MentalModelInput) -> dict[str, Any]:
            async with pool.acquire() as conn:
                result = await tool_learn(conn, bank_id, input, tags=tags)
            # If a new model was created, trigger background generation
            if result.get("status") == "created" and result.get("model_id"):
                try:
                    await self.generate_mental_model_async(
                        bank_id=bank_id,
                        model_id=result["model_id"],
                        request_context=request_context,
                    )
                    logger.info(f"[REFLECT] Triggered background generation for learned model: {result['model_id']}")
                except Exception as e:
                    logger.warning(f"[REFLECT] Failed to trigger generation for {result['model_id']}: {e}")
            return result

        async def expand_fn(memory_ids: list[str], depth: str) -> dict[str, Any]:
            async with pool.acquire() as conn:
                return await tool_expand(conn, bank_id, memory_ids, depth)

        # Run the agent
        agent_result = await run_reflect_agent(
            llm_config=self._reflect_llm_config,
            bank_id=bank_id,
            query=query,
            bank_profile=profile,
            lookup_fn=lookup_fn,
            recall_fn=recall_fn,
            learn_fn=learn_fn,
            expand_fn=expand_fn,
            context=context,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

        total_time = time.time() - reflect_start
        logger.info(
            f"[REFLECT {reflect_id}] Complete: {len(agent_result.text)} chars, "
            f"{agent_result.iterations} iterations, {agent_result.tools_called} tool calls | {total_time:.3f}s"
        )

        # Convert agent tool trace to ToolCallTrace objects
        tool_trace_result = [
            ToolCallTrace(
                tool=tc.tool,
                input=tc.input,
                output=tc.output,
                duration_ms=tc.duration_ms,
                iteration=tc.iteration,
            )
            for tc in agent_result.tool_trace
        ]

        # Convert agent LLM trace to LLMCallTrace objects
        llm_trace_result = [LLMCallTrace(scope=lc.scope, duration_ms=lc.duration_ms) for lc in agent_result.llm_trace]

        # Extract memories from recall tool outputs - only include memories the agent actually used
        # agent_result.used_memory_ids contains validated IDs from the done action
        used_memory_ids_set = set(agent_result.used_memory_ids) if agent_result.used_memory_ids else set()
        based_on: dict[str, list[MemoryFact]] = {"world": [], "experience": [], "opinion": []}
        seen_memory_ids: set[str] = set()
        for tc in agent_result.tool_trace:
            if tc.tool == "recall" and "memories" in tc.output:
                for memory_data in tc.output["memories"]:
                    memory_id = memory_data.get("id")
                    # Only include memories that the agent declared as used (or all if none specified)
                    if memory_id and memory_id not in seen_memory_ids:
                        if used_memory_ids_set and memory_id not in used_memory_ids_set:
                            continue  # Skip memories not actually used by the agent
                        seen_memory_ids.add(memory_id)
                        fact_type = memory_data.get("type", "world")
                        if fact_type in based_on:
                            based_on[fact_type].append(
                                MemoryFact(
                                    id=memory_id,
                                    text=memory_data.get("text", ""),
                                    fact_type=fact_type,
                                    context=None,
                                    occurred_start=memory_data.get("occurred"),
                                    occurred_end=memory_data.get("occurred"),
                                )
                            )

        # Extract mental models from lookup tool outputs - only include models the agent actually used
        # agent_result.used_model_ids contains validated IDs from the done action
        used_model_ids_set = set(agent_result.used_model_ids) if agent_result.used_model_ids else set()
        based_on["mental_model"] = []
        mental_models_result: list[MentalModelRef] = []
        seen_model_ids: set[str] = set()
        for tc in agent_result.tool_trace:
            if tc.tool == "get_mental_model":
                # Single model lookup (with full details)
                if tc.output.get("found") and "model" in tc.output:
                    model = tc.output["model"]
                    model_id = model.get("id")
                    if model_id and model_id not in seen_model_ids:
                        # Only include models that the agent declared as used (or all if none specified)
                        if used_model_ids_set and model_id not in used_model_ids_set:
                            continue  # Skip models not actually used by the agent
                        seen_model_ids.add(model_id)
                        # Add to based_on as MemoryFact with type "mental_model"
                        model_name = model.get("name", "")
                        model_summary = model.get("summary") or model.get("description", "")
                        based_on["mental_model"].append(
                            MemoryFact(
                                id=model_id,
                                text=f"{model_name}: {model_summary}",
                                fact_type="mental_model",
                                context=f"{model.get('type', 'concept')} ({model.get('subtype', 'structural')})",
                                occurred_start=None,
                                occurred_end=None,
                            )
                        )
                        mental_models_result.append(
                            MentalModelRef(
                                id=model_id,
                                name=model_name,
                                type=model.get("type", "concept"),
                                subtype=model.get("subtype", "structural"),
                                description=model.get("description", ""),
                                summary=model.get("summary"),
                            )
                        )
                # List all models lookup - don't add to based_on (too verbose, just a listing)

        # Return response (compatible with existing API)
        result = ReflectResult(
            text=agent_result.text,
            based_on=based_on,
            new_opinions=[],  # Learnings stored as mental models
            structured_output=agent_result.structured_output,
            usage=None,  # Token tracking not yet implemented for agentic loop
            tool_trace=tool_trace_result,
            llm_trace=llm_trace_result,
            mental_models=mental_models_result,
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

    async def get_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        *,
        limit: int = 10,
        request_context: "RequestContext",
    ) -> list[Any]:
        """
        Get observations for an entity.

        NOTE: Entity observations/summaries have been moved to mental models.
        This method returns an empty list. Use mental models for entity summaries.

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID to get observations for
            limit: Ignored (kept for backwards compatibility)
            request_context: Request context for authentication.

        Returns:
            Empty list (observations now in mental models)
        """
        await self._authenticate_tenant(request_context)
        return []

    async def list_entities(
        self,
        bank_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List all entities for a bank with pagination.

        Args:
            bank_id: bank IDentifier
            limit: Maximum number of entities to return
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items, total, limit, offset
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Get total count
            total_row = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as total
                FROM {fq_table("entities")}
                WHERE bank_id = $1
                """,
                bank_id,
            )
            total = total_row["total"] if total_row else 0

            # Get paginated entities
            rows = await conn.fetch(
                f"""
                SELECT id, canonical_name, mention_count, first_seen, last_seen, metadata
                FROM {fq_table("entities")}
                WHERE bank_id = $1
                ORDER BY mention_count DESC, last_seen DESC, id ASC
                LIMIT $2 OFFSET $3
                """,
                bank_id,
                limit,
                offset,
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
            return {
                "items": entities,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    async def list_tags(
        self,
        bank_id: str,
        *,
        pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List all unique tags for a bank with usage counts.

        Use this to discover available tags or expand wildcard patterns.
        Supports '*' as wildcard for flexible matching (case-insensitive):
        - 'user:*' matches user:alice, user:bob
        - '*-admin' matches role-admin, super-admin
        - 'env*-prod' matches env-prod, environment-prod

        Args:
            bank_id: Bank identifier
            pattern: Wildcard pattern to filter tags (use '*' as wildcard, case-insensitive)
            limit: Maximum number of tags to return
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of {tag, count}), total, limit, offset
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()
        async with acquire_with_retry(pool) as conn:
            # Build pattern filter if provided (convert * to % for ILIKE)
            pattern_clause = ""
            params: list[Any] = [bank_id]
            if pattern:
                # Convert wildcard pattern: * -> % for SQL ILIKE
                sql_pattern = pattern.replace("*", "%")
                pattern_clause = "AND tag ILIKE $2"
                params.append(sql_pattern)

            # Get total count of distinct tags matching pattern
            total_row = await conn.fetchrow(
                f"""
                SELECT COUNT(DISTINCT tag) as total
                FROM {fq_table("memory_units")}, unnest(tags) AS tag
                WHERE bank_id = $1 AND tags IS NOT NULL AND tags != '{{}}'
                {pattern_clause}
                """,
                *params,
            )
            total = total_row["total"] if total_row else 0

            # Get paginated tags with counts, ordered by frequency
            limit_param = len(params) + 1
            offset_param = len(params) + 2
            params.extend([limit, offset])

            rows = await conn.fetch(
                f"""
                SELECT tag, COUNT(*) as count
                FROM {fq_table("memory_units")}, unnest(tags) AS tag
                WHERE bank_id = $1 AND tags IS NOT NULL AND tags != '{{}}'
                {pattern_clause}
                GROUP BY tag
                ORDER BY count DESC, tag ASC
                LIMIT ${limit_param} OFFSET ${offset_param}
                """,
                *params,
            )

            items = [{"tag": row["tag"], "count": row["count"]} for row in rows]

            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset,
            }

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
        Get the current state of an entity.

        NOTE: Entity observations/summaries have been moved to mental models.
        This method returns an entity with empty observations.

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID
            entity_name: Canonical name of the entity
            limit: Maximum number of observations to include (kept for backwards compat)
            request_context: Request context for authentication.

        Returns:
            EntityState with empty observations (summaries now in mental models)
        """
        await self._authenticate_tenant(request_context)
        return EntityState(entity_id=entity_id, canonical_name=entity_name, observations=[])

    async def regenerate_entity_observations(
        self,
        bank_id: str,
        entity_id: str,
        entity_name: str,
        *,
        version: str | None = None,
        conn=None,
        request_context: "RequestContext",
    ) -> list[str]:
        """
        Regenerate observations for an entity.

        NOTE: Entity observations/summaries have been moved to mental models.
        This method is now a no-op and returns an empty list.

        Args:
            bank_id: bank IDentifier
            entity_id: Entity UUID
            entity_name: Canonical name of the entity
            version: Entity's last_seen timestamp when task was created (for deduplication)
            conn: Optional database connection (ignored)
            request_context: Request context for authentication.

        Returns:
            Empty list (observations now in mental models)
        """
        await self._authenticate_tenant(request_context)
        return []

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

    # =========================================================================
    # Mental Models
    # =========================================================================

    async def list_mental_models(
        self,
        bank_id: str,
        *,
        subtype: str | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List mental models for a bank, optionally filtered by subtype or tags.

        Args:
            bank_id: Bank identifier
            subtype: Filter by subtype (structural, emergent, pinned)
            tags: Filter by tags - returns models that match according to tags_match
            tags_match: How to match tags - "any" (OR), "all" (AND), or "exact"
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        query = f"""
            SELECT id, bank_id, subtype, name, description, observations,
                   entity_id, links, tags, last_updated, created_at
            FROM {fq_table("mental_models")}
            WHERE bank_id = $1
        """
        params: list[Any] = [bank_id]

        if subtype:
            query += f" AND subtype = ${len(params) + 1}"
            params.append(subtype)

        # Tags filtering: include untagged models OR models with matching tags
        if tags:
            if tags_match == "any":
                # OR match: model has no tags OR model has at least one matching tag
                query += f" AND (tags = '{{}}' OR tags && ${len(params) + 1})"
            elif tags_match == "all":
                # AND match: model has no tags OR model has all specified tags
                query += f" AND (tags = '{{}}' OR tags @> ${len(params) + 1})"
            else:  # exact
                # Exact match: model has no tags OR model has exactly the specified tags
                query += f" AND (tags = '{{}}' OR tags = ${len(params) + 1})"
            params.append(tags)

        query += " ORDER BY created_at ASC"

        async with acquire_with_retry(pool) as conn:
            rows = await conn.fetch(query, *params)

        return [self._row_to_mental_model(row) for row in rows]

    async def get_mental_model(
        self,
        bank_id: str,
        model_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Get a mental model by ID."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, bank_id, subtype, name, description, observations,
                       entity_id, links, tags, last_updated, created_at
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                model_id,
            )

        return self._row_to_mental_model(row) if row else None

    async def refresh_mental_model(
        self,
        bank_id: str,
        model_id: str,
        *,
        request_context: "RequestContext",
        _return_agent_result: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any] | None, Any] | None:
        """Refresh the summary for a mental model using the reflect agent.

        Uses the model's stored tags to filter recall results.

        Args:
            bank_id: Bank identifier
            model_id: Mental model ID
            request_context: Request context for authentication
            _return_agent_result: Internal flag to return (model, agent_result) tuple for logging

        Returns:
            Updated mental model dict, or (model, agent_result) tuple if _return_agent_result=True
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        # Get the mental model
        model = await self.get_mental_model(bank_id, model_id, request_context=request_context)
        if not model:
            return None

        # Use the model's stored tags for filtering recall
        model_tags = model.get("tags") or None

        # Import reflect agent and tools
        from .reflect.agent import run_reflect_agent
        from .reflect.tools import tool_expand, tool_lookup, tool_recall

        # Get bank profile for agent context
        profile = await self.get_bank_profile(bank_id, request_context=request_context)
        bank_profile = {
            "name": profile.get("name", "Assistant"),
            "mission": profile.get("mission", ""),
        }

        # Build query for the agent - instruct to generate multiple structured observations
        query = f"""Generate comprehensive observations about '{model["name"]}': {model["description"]}

To create thorough observations:
1. Use multiple recall queries to explore different aspects and facets of this topic
2. Search for related events, relationships, preferences, patterns, and historical context
3. Use expand to get full context when a fact seems important but incomplete
4. Create multiple distinct observations, each covering a different dimension or aspect

Each observation should be self-contained and focus on a specific theme (e.g., preferences, history, relationships, patterns)."""

        # Run the reflect agent with tools (no learn tool for summary generation)
        # Use observations mode to get structured observations instead of a single text answer
        config = get_config()
        metrics = get_metrics_collector()
        # Use 2x iterations for observations mode - needs more iterations for:
        # multiple recall queries + expand verification + observations creation
        observations_max_iterations = config.reflect_max_iterations * 2
        with metrics.record_operation("mental_model_refresh", bank_id=bank_id, source="api"):
            async with acquire_with_retry(pool) as conn:
                result = await run_reflect_agent(
                    llm_config=self._reflect_llm_config,
                    bank_id=bank_id,
                    query=query,
                    bank_profile=bank_profile,
                    lookup_fn=lambda mid=None: tool_lookup(conn, bank_id, mid),
                    recall_fn=lambda q, mt=4096: tool_recall(
                        self,
                        bank_id,
                        q,
                        request_context,
                        max_tokens=mt,
                        tags=model_tags,
                        tags_match="any",
                    ),
                    expand_fn=lambda mem_ids, depth: tool_expand(conn, bank_id, mem_ids, depth),
                    learn_fn=None,  # Disable learn tool for summary generation
                    max_iterations=observations_max_iterations,
                    output_mode="observations",  # Get structured observations
                )

        # Update the model with the structured observations from the agent
        import json

        # Use observations from the agent result directly, or fall back to single observation from text
        if result.observations:
            observations_list = [
                {"title": obs.title, "text": obs.text, "memory_ids": obs.memory_ids} for obs in result.observations
            ]
        else:
            # Fallback if no structured observations returned
            observations_list = [{"title": "", "text": result.text, "memory_ids": result.used_memory_ids or []}]

        observations_json = {"observations": observations_list}
        async with acquire_with_retry(pool) as conn:
            updated_row = await conn.fetchrow(
                f"""
                UPDATE {fq_table("mental_models")}
                SET observations = $1::jsonb, last_updated = NOW()
                WHERE bank_id = $2 AND id = $3
                RETURNING id, bank_id, subtype, name, description, observations,
                          entity_id, links, tags, last_updated, created_at
                """,
                json.dumps(observations_json),
                bank_id,
                model_id,
            )

        model_result = self._row_to_mental_model(updated_row) if updated_row else None
        if _return_agent_result:
            return (model_result, result)
        return model_result

    async def generate_mental_model_async(
        self,
        bank_id: str,
        model_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Submit a background job to generate/refresh a specific mental model.

        This is useful for:
        - Generating content for newly created learned models
        - Re-generating content for pinned models after description changes
        - Manual refresh of a specific model without touching others

        Args:
            bank_id: Bank identifier
            model_id: Mental model ID to generate

        Returns:
            Dict with operation_id to track progress
        """
        await self._authenticate_tenant(request_context)

        # Verify the model exists
        model = await self.get_mental_model(bank_id, model_id, request_context=request_context)
        if not model:
            raise ValueError(f"Mental model '{model_id}' not found in bank '{bank_id}'")

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
                "generate_mental_model",
                json.dumps({"model_id": model_id}),
            )

        # Submit task to background queue
        task_payload = {
            "type": "generate_mental_model",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "model_id": model_id,
        }

        await self._task_backend.submit_task(task_payload)

        logger.info(
            f"[MENTAL_MODEL] Generation task queued for model_id={model_id}, bank_id={bank_id}, operation_id={operation_id}"
        )

        return {
            "operation_id": str(operation_id),
            "model_id": model_id,
            "status": "queued",
        }

    async def refresh_mental_models(
        self,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        subtype: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Submit a background job to refresh mental models for a bank.

        The background job will (depending on subtype filter):
        1. Derive structural models from the bank's mission (if subtype is None or "structural")
        2. Detect emergent candidates (entities worth promoting) (if subtype is None or "emergent")
        3. Filter candidates by mission relevance
        4. Create/update mental models with specified tags
        5. Generate summaries for refreshed mental models

        Args:
            bank_id: Bank identifier
            tags: Tags to apply to newly created mental models
            subtype: Only refresh models of this subtype ("structural" or "emergent").
                     If None, refreshes all subtypes.

        Raises:
            ValueError: If no mission is set for the bank

        Returns:
            Dict with operation_id to track progress
        """
        await self._authenticate_tenant(request_context)

        # Check that mission is set before scheduling the task
        profile = await self.get_bank_profile(bank_id, request_context=request_context)
        mission = profile.get("mission") or ""
        if not mission:
            raise ValueError(
                f"Cannot refresh mental models: no mission is set for bank '{bank_id}'. Set a mission first."
            )

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
                "refresh_mental_models",
                json.dumps({}),
            )

        # Submit task to background queue
        task_payload = {
            "type": "refresh_mental_models",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
        }
        if tags:
            task_payload["tags"] = tags
        if subtype:
            task_payload["subtype"] = subtype

        await self._task_backend.submit_task(task_payload)

        logger.info(f"[MENTAL_MODELS] Refresh task queued for bank_id={bank_id}, operation_id={operation_id}")

        return {
            "operation_id": str(operation_id),
            "status": "queued",
        }

    async def _derive_structural_models_internal(
        self,
        bank_id: str,
        mission: str,
        pool,
        existing_models: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        Internal method to derive structural models without auth check.

        Args:
            bank_id: Bank identifier
            mission: The bank's mission
            pool: Database connection pool
            existing_models: Optional list of existing structural models
            tags: Tags to apply to created mental models

        Returns:
            List of model IDs to remove (existing models not in LLM output)
        """
        from .mental_models.models import MentalModelSubtype
        from .mental_models.structural import derive_structural_models

        templates, models_to_remove = await derive_structural_models(
            self._llm_config, mission, existing_models=existing_models
        )

        model_tags = tags or []
        created_count = 0
        async with acquire_with_retry(pool) as conn:
            for template in templates:
                try:
                    await conn.fetchrow(
                        f"""
                        INSERT INTO {fq_table("mental_models")}
                        (id, bank_id, subtype, name, description, tags)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (id, bank_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            description = EXCLUDED.description,
                            tags = EXCLUDED.tags
                        RETURNING id
                        """,
                        template.id,
                        bank_id,
                        MentalModelSubtype.STRUCTURAL.value,
                        template.name,
                        template.description,
                        model_tags,
                    )
                    created_count += 1
                except Exception as e:
                    logger.warning(f"[MENTAL_MODELS] Failed to create structural model {template.id}: {e}")

        logger.info(f"[MENTAL_MODELS] Created/updated {created_count} structural models for bank {bank_id}")
        return models_to_remove

    async def _promote_entity_internal(
        self, bank_id: str, entity_id: str, pool, tags: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Internal method to promote entity to mental model without auth check.

        Args:
            bank_id: Bank identifier
            entity_id: Entity ID to promote
            pool: Database connection pool
            tags: Tags to apply to the created mental model
        """
        from .mental_models.models import MentalModelSubtype

        async with acquire_with_retry(pool) as conn:
            # Get entity info
            entity = await conn.fetchrow(
                f"SELECT id, canonical_name FROM {fq_table('entities')} WHERE id = $1 AND bank_id = $2",
                uuid.UUID(entity_id),
                bank_id,
            )

            if not entity:
                return None

            # Create mental model from entity
            model_id = f"entity-{entity['canonical_name'].lower().replace(' ', '-')}"
            row = await conn.fetchrow(
                f"""
                INSERT INTO {fq_table("mental_models")}
                (id, bank_id, subtype, name, description, entity_id, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id, bank_id) DO NOTHING
                RETURNING id, bank_id, subtype, name, description, observations,
                          entity_id, links, tags, last_updated, created_at
                """,
                model_id,
                bank_id,
                MentalModelSubtype.EMERGENT.value,
                entity["canonical_name"],
                f"Mental model for {entity['canonical_name']}",
                entity["id"],
                tags or [],  # Apply tags from refresh operation
            )

        return self._row_to_mental_model(row) if row else None

    async def create_mental_model(
        self,
        bank_id: str,
        name: str,
        description: str,
        *,
        tags: list[str] | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Create a pinned mental model.

        Pinned mental models are user-defined and persist across refreshes.
        They are not automatically removed when mental models are regenerated.

        Args:
            bank_id: Bank identifier
            name: Human-readable name for the mental model
            description: One-liner description for quick scanning
            tags: Tags for scoped visibility

        Returns:
            The created mental model
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        from .mental_models.models import MentalModelSubtype

        # Generate stable ID from name
        model_id = f"pinned-{name.lower().replace(' ', '-').replace('/', '-')}"

        async with acquire_with_retry(pool) as conn:
            # Check if model already exists
            existing = await conn.fetchrow(
                f"SELECT id FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                model_id,
            )
            if existing:
                raise ValueError(f"Mental model with name '{name}' already exists")

            row = await conn.fetchrow(
                f"""
                INSERT INTO {fq_table("mental_models")}
                (id, bank_id, subtype, name, description, tags)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, bank_id, subtype, name, description, observations,
                          entity_id, links, tags, last_updated, created_at
                """,
                model_id,
                bank_id,
                MentalModelSubtype.PINNED.value,
                name,
                description,
                tags or [],
            )

        logger.info(f"[MENTAL_MODELS] Created pinned mental model '{name}' (id={model_id}) for bank {bank_id}")
        return self._row_to_mental_model(row)

    async def delete_mental_model(
        self,
        bank_id: str,
        model_id: str,
        *,
        request_context: "RequestContext",
    ) -> bool:
        """Delete a mental model."""
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        async with acquire_with_retry(pool) as conn:
            result = await conn.execute(
                f"DELETE FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                model_id,
            )

        return result == "DELETE 1"

    def _row_to_mental_model(self, row) -> dict[str, Any]:
        """Convert a database row to a mental model dict."""
        import json

        # Parse observations JSON - can be a dict {"observations": [...]} or a list []
        observations_data = row.get("observations")
        if observations_data is None:
            observations_raw = []
        elif isinstance(observations_data, str):
            observations_data = json.loads(observations_data)
            observations_raw = (
                observations_data.get("observations", []) if isinstance(observations_data, dict) else observations_data
            )
        elif isinstance(observations_data, list):
            observations_raw = observations_data
        elif isinstance(observations_data, dict):
            observations_raw = observations_data.get("observations", [])
        else:
            observations_raw = []

        # Normalize observation format: map memory_ids/fact_ids to based_on
        observations = []
        for obs in observations_raw:
            if isinstance(obs, dict):
                # Get memory IDs from either memory_ids (new) or fact_ids (legacy)
                based_on = obs.get("memory_ids") or obs.get("fact_ids") or []
                observations.append(
                    {
                        "title": obs.get("title", ""),
                        "text": obs.get("text", ""),
                        "based_on": based_on,
                    }
                )

        return {
            "id": row["id"],
            "bank_id": row["bank_id"],
            "subtype": row["subtype"],
            "name": row["name"],
            "description": row["description"],
            "observations": observations,
            "entity_id": str(row["entity_id"]) if row["entity_id"] else None,
            "links": row["links"] or [],
            "tags": list(row["tags"]) if row.get("tags") else [],
            "last_updated": row["last_updated"].isoformat() if row["last_updated"] else None,
            "created_at": row["created_at"].isoformat(),
        }

    async def _invalidate_facts_from_mental_models(
        self,
        conn,
        bank_id: str,
        fact_ids: list[str],
    ) -> int:
        """
        Remove fact IDs from mental model observations when memories are deleted.

        Uses JSONB path operations to find and update mental models that reference
        the deleted fact IDs in their observations.

        Args:
            conn: Database connection
            bank_id: Bank identifier
            fact_ids: List of fact IDs to remove from mental models

        Returns:
            Number of mental models updated
        """
        if not fact_ids:
            return 0

        # Convert fact_ids to a jsonb array for efficient comparison
        import json

        fact_ids_json = json.dumps(fact_ids)

        # Update mental models by removing the deleted fact IDs from all observations
        # This uses jsonb_set to update each observation's fact_ids array
        result = await conn.execute(
            f"""
            UPDATE {fq_table("mental_models")}
            SET observations = jsonb_set(
                observations,
                '{{observations}}',
                (
                    SELECT COALESCE(jsonb_agg(
                        jsonb_set(
                            observation,
                            '{{fact_ids}}',
                            (
                                SELECT COALESCE(jsonb_agg(fid), '[]'::jsonb)
                                FROM jsonb_array_elements_text(observation->'fact_ids') AS fid
                                WHERE NOT (fid::text = ANY($2::text[]))
                            )
                        )
                    ), '[]'::jsonb)
                    FROM jsonb_array_elements(observations->'observations') AS observation
                )
            ),
            last_updated = NOW()
            WHERE bank_id = $1
            AND EXISTS (
                SELECT 1
                FROM jsonb_array_elements(observations->'observations') AS observation,
                     jsonb_array_elements_text(observation->'fact_ids') AS fid
                WHERE fid::text = ANY($2::text[])
            )
            """,
            bank_id,
            fact_ids,
        )

        # Parse the result to get number of updated rows
        updated_count = int(result.split()[-1]) if result and "UPDATE" in result else 0
        if updated_count > 0:
            logger.info(
                f"[MENTAL_MODELS] Invalidated {len(fact_ids)} fact IDs from {updated_count} mental models in bank {bank_id}"
            )
        return updated_count

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
            # Get total count
            total_row = await conn.fetchrow(
                f"SELECT COUNT(*) as total FROM {fq_table('async_operations')} WHERE bank_id = $1",
                bank_id,
            )
            total = total_row["total"] if total_row else 0

            # Get recent operations
            operations = await conn.fetch(
                f"""
                SELECT operation_id, operation_type, created_at, status, error_message
                FROM {fq_table("async_operations")}
                WHERE bank_id = $1
                ORDER BY created_at DESC
                LIMIT 50
                """,
                bank_id,
            )

            return {
                "total": total,
                "operations": [
                    {
                        "id": str(row["operation_id"]),
                        "task_type": row["operation_type"],
                        "items_count": 0,
                        "document_id": None,
                        "created_at": row["created_at"].isoformat(),
                        "status": row["status"],
                        "error_message": row["error_message"],
                    }
                    for row in operations
                ],
            }

    async def get_operation_status(
        self,
        bank_id: str,
        operation_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Get the status of a specific async operation.

        Returns:
            - status: "pending", "completed", or "failed"
            - updated_at: last update timestamp
            - completed_at: completion timestamp (if completed)
        """
        await self._authenticate_tenant(request_context)
        pool = await self._get_pool()

        op_uuid = uuid.UUID(operation_id)

        async with acquire_with_retry(pool) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT operation_id, operation_type, created_at, updated_at, completed_at, status, error_message
                FROM {fq_table("async_operations")}
                WHERE operation_id = $1 AND bank_id = $2
                """,
                op_uuid,
                bank_id,
            )

            if row:
                # Map DB status to API status (processing -> pending for simplicity)
                db_status = row["status"]
                api_status = "pending" if db_status in ("pending", "processing") else db_status
                return {
                    "operation_id": operation_id,
                    "status": api_status,
                    "operation_type": row["operation_type"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                    "error_message": row["error_message"],
                }
            else:
                # Operation not found
                return {
                    "operation_id": operation_id,
                    "status": "not_found",
                    "operation_type": None,
                    "created_at": None,
                    "updated_at": None,
                    "completed_at": None,
                    "error_message": None,
                }

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
        mission: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Update bank name and/or mission."""
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

            if mission is not None:
                await conn.execute(
                    f"""
                    UPDATE {fq_table("banks")}
                    SET mission = $2, updated_at = NOW()
                    WHERE bank_id = $1
                    """,
                    bank_id,
                    mission,
                )

        # Return updated profile
        return await self.get_bank_profile(bank_id, request_context=request_context)

    async def submit_async_retain(
        self,
        bank_id: str,
        contents: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
        document_tags: list[str] | None = None,
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
        task_payload = {
            "type": "batch_retain",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "contents": contents,
        }
        if document_tags:
            task_payload["document_tags"] = document_tags

        await self._task_backend.submit_task(task_payload)

        logger.info(f"Retain task queued for bank_id={bank_id}, {len(contents)} items, operation_id={operation_id}")

        return {
            "operation_id": str(operation_id),
            "items_count": len(contents),
        }
