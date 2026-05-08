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
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import asyncpg
import httpx
import tiktoken

from ..config import (
    DEFAULT_RECALL_CHUNKS_MAX_TOKENS,
    DEFAULT_RECALL_INCLUDE_CHUNKS,
    DEFAULT_RECALL_MAX_TOKENS,
    DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS,
    get_config,
)
from ..db_url import to_libpq_url
from ..metrics import get_metrics_collector
from ..tracing import create_operation_span
from ..utils import mask_network_location
from ..worker.exceptions import DeferOperation, RetryTaskAt
from ..worker.stage import set_stage
from .audit import AuditLogger, audit_context
from .db import DatabaseBackend, create_database_backend
from .db_budget import budgeted_operation
from .operation_metadata import (
    BatchRetainChildMetadata,
    BatchRetainParentMetadata,
    ConsolidationMetadata,
    RefreshMentalModelMetadata,
    RetainMetadata,
)
from .sql import SQLDialect, create_sql_dialect

# Context variable for current schema (async-safe, per-task isolation)
# Note: default is None, actual default comes from config via get_current_schema()
_current_schema: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_schema", default=None)


def get_current_schema() -> str:
    """Get the current schema from context (falls back to config default)."""
    schema = _current_schema.get()
    if schema is None:
        # Fall back to configured default schema
        return get_config().database_schema
    return schema


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (cl100k_base encoding for GPT-4/3.5)."""
    return len(_get_tiktoken_encoding().encode(text))


def fq_table(table_name: str) -> str:
    """Get fully-qualified table name with current schema.

    Delegates to :func:`engine.schema.fq_table` — kept here for backward
    compatibility (many modules import ``fq_table`` from ``memory_engine``).
    """
    from .schema import fq_table as _fq_table

    return _fq_table(table_name)


def _json_default(obj: Any) -> str:
    """JSON serializer for types commonly carried through async task payloads."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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
        "file_storage",
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
from .llm_wrapper import LLMConfig, requires_api_key, sanitize_llm_output
from .query_analyzer import QueryAnalyzer
from .reflect import run_reflect_agent
from .reflect.prompts import DELTA_SYSTEM_PROMPT, build_delta_prompt
from .reflect.tools import tool_expand, tool_recall, tool_search_mental_models, tool_search_observations
from .response_models import (
    VALID_RECALL_FACT_TYPES,
    EntityObservation,
    EntityState,
    LLMCallTrace,
    MemoryFact,
    ObservationRef,
    ReflectResult,
    TokenUsage,
    ToolCallTrace,
)
from .response_models import RecallResult as RecallResultModel
from .retain import bank_utils, embedding_utils
from .retain.types import RetainContentDict
from .search import think_utils
from .search.reranking import CrossEncoderReranker, apply_combined_scoring
from .search.tags import TagGroup, TagsMatch, build_tags_where_clause
from .task_backend import TaskBackend


def _is_oracledb_connection_error(e: Exception) -> bool:
    """Check if an exception is an Oracle connection/interface error."""
    try:
        import oracledb  # type: ignore[import-not-found]
    except ImportError:
        return False
    return isinstance(e, (oracledb.InterfaceError, oracledb.OperationalError))


def _is_oracledb_integrity_error(e: Exception) -> bool:
    """Check if an exception is an Oracle integrity constraint error."""
    try:
        import oracledb  # type: ignore[import-not-found]
    except ImportError:
        return False
    return isinstance(e, oracledb.IntegrityError)


class Budget(str, Enum):
    """Budget levels for recall/reflect operations."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


def _resolve_thinking_budget(config_dict: dict, budget: "Budget | None", max_tokens: int) -> int:
    """
    Map a Budget enum level to the integer thinking_budget passed to retrieval.

    Reads the bank-resolved config to decide between two functions:
    - "fixed": returns recall_budget_fixed_<level> directly (legacy default).
    - "adaptive": returns round(max_tokens * recall_budget_adaptive_<level>),
                  clamped to [recall_budget_min, recall_budget_max].

    A None budget falls back to MID (preserves legacy default).
    """
    effective_budget = budget if budget is not None else Budget.MID
    function = config_dict.get("recall_budget_function", "fixed")

    if function == "adaptive":
        ratios = {
            Budget.LOW: config_dict.get("recall_budget_adaptive_low", 0.025),
            Budget.MID: config_dict.get("recall_budget_adaptive_mid", 0.075),
            Budget.HIGH: config_dict.get("recall_budget_adaptive_high", 0.25),
        }
        raw = round(max_tokens * float(ratios[effective_budget]))
        floor = int(config_dict.get("recall_budget_min", 20))
        ceiling = int(config_dict.get("recall_budget_max", 2000))
        return max(floor, min(ceiling, raw))

    fixed = {
        Budget.LOW: config_dict.get("recall_budget_fixed_low", 100),
        Budget.MID: config_dict.get("recall_budget_fixed_mid", 300),
        Budget.HIGH: config_dict.get("recall_budget_fixed_high", 1000),
    }
    return int(fixed[effective_budget])


def utcnow():
    """Get current UTC time with timezone info."""
    return datetime.now(UTC)


# Logger for memory system
logger = logging.getLogger(__name__)

from .db_utils import acquire_with_retry

# Cache tiktoken encoding for token budget filtering (module-level singleton)
_TIKTOKEN_ENCODING = None


def _get_tiktoken_encoding():
    """Get cached tiktoken encoding (cl100k_base for GPT-4/3.5)."""
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


@dataclass(frozen=True)
class _TimeseriesPeriodConfig:
    """How one period slices the time axis for the memories-ingested chart."""

    interval: str  # postgres interval literal used in the `now() - interval '...'` filter
    trunc: str  # date_trunc unit (minute/hour/day)
    step: timedelta  # distance between adjacent buckets
    count: int  # total buckets rendered for the period


_MEMORIES_TIMESERIES_PERIODS: dict[str, _TimeseriesPeriodConfig] = {
    "1h": _TimeseriesPeriodConfig("1 hour", "minute", timedelta(minutes=1), 60),
    "12h": _TimeseriesPeriodConfig("12 hours", "hour", timedelta(hours=1), 12),
    "1d": _TimeseriesPeriodConfig("24 hours", "hour", timedelta(hours=1), 24),
    "7d": _TimeseriesPeriodConfig("7 days", "day", timedelta(days=1), 7),
    "30d": _TimeseriesPeriodConfig("30 days", "day", timedelta(days=1), 30),
    "90d": _TimeseriesPeriodConfig("90 days", "day", timedelta(days=1), 90),
}


@dataclass
class MemoryTimeseriesBucketData:
    """One bucket of the memories-ingested time series (engine-side)."""

    time: str
    world: int = 0
    experience: int = 0
    observation: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "world": self.world,
            "experience": self.experience,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class RefreshTagFiltering:
    """Resolved tag filtering parameters for mental model refresh."""

    tags: list[str] | None
    tags_match: TagsMatch
    tag_groups: list[TagGroup] | None


def _resolve_refresh_tag_filtering(
    model_tags: list[str] | None,
    trigger_data: dict[str, Any],
) -> RefreshTagFiltering:
    """Resolve tag filtering parameters for mental model refresh.

    Takes raw trigger dict from DB (JSONB with no fixed schema guarantee)
    and resolves the tag filtering to use during reflect.

    Priority:
    - If trigger has tag_groups, use those (overrides flat tags entirely)
    - If trigger has tags_match, use model's tags with that match mode
    - Otherwise default to all_strict when tags present (security isolation)
    """
    trigger_tag_groups = trigger_data.get("tag_groups")
    if trigger_tag_groups is not None:
        from pydantic import TypeAdapter

        adapter = TypeAdapter(TagGroup)
        parsed = [adapter.validate_python(tg) for tg in trigger_tag_groups]
        return RefreshTagFiltering(tags=None, tags_match="any", tag_groups=parsed)

    trigger_tags_match = trigger_data.get("tags_match")
    tags_match: TagsMatch = trigger_tags_match if trigger_tags_match else ("all_strict" if model_tags else "any")
    return RefreshTagFiltering(tags=model_tags, tags_match=tags_match, tag_groups=None)


class MemoryEngine(MemoryEngineInterface):
    """
    Advanced memory system using temporal and semantic linking with PostgreSQL.

    This class provides:
    - Embedding generation for semantic search
    - Entity, temporal, and semantic link creation
    - Think operations for formulating answers with observations
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
        consolidation_llm_provider: str | None = None,
        consolidation_llm_api_key: str | None = None,
        consolidation_llm_model: str | None = None,
        consolidation_llm_base_url: str | None = None,
        embeddings: Embeddings | None = None,
        cross_encoder: CrossEncoderModel | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        db_command_timeout: int | None = None,
        db_acquire_timeout: int | None = None,
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
            retain_llm_provider: LLM provider for retain operations. Falls back to memory_llm_provider.
            retain_llm_api_key: API key for retain LLM. Falls back to memory_llm_api_key.
            retain_llm_model: Model for retain operations. Falls back to memory_llm_model.
            retain_llm_base_url: Base URL for retain LLM. Falls back to memory_llm_base_url.
            reflect_llm_provider: LLM provider for reflect operations. Falls back to memory_llm_provider.
            reflect_llm_api_key: API key for reflect LLM. Falls back to memory_llm_api_key.
            reflect_llm_model: Model for reflect operations. Falls back to memory_llm_model.
            reflect_llm_base_url: Base URL for reflect LLM. Falls back to memory_llm_base_url.
            consolidation_llm_provider: LLM provider for consolidation operations. Falls back to memory_llm_provider.
            consolidation_llm_api_key: API key for consolidation LLM. Falls back to memory_llm_api_key.
            consolidation_llm_model: Model for consolidation operations. Falls back to memory_llm_model.
            consolidation_llm_base_url: Base URL for consolidation LLM. Falls back to memory_llm_base_url.
            embeddings: Embeddings implementation. If not provided, created from env vars.
            cross_encoder: Cross-encoder model. If not provided, created from env vars.
            query_analyzer: Query analyzer implementation. If not provided, uses DateparserQueryAnalyzer.
            pool_min_size: Minimum number of connections in the pool. Defaults to HINDSIGHT_API_DB_POOL_MIN_SIZE.
            pool_max_size: Maximum number of connections in the pool. Defaults to HINDSIGHT_API_DB_POOL_MAX_SIZE.
            db_command_timeout: PostgreSQL command timeout in seconds. Defaults to HINDSIGHT_API_DB_COMMAND_TIMEOUT.
            db_acquire_timeout: Connection acquisition timeout in seconds. Defaults to HINDSIGHT_API_DB_ACQUIRE_TIMEOUT.
            task_backend: Custom task backend. If not provided, uses BrokerTaskBackend for distributed processing.
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

        # Force skip LLM verification when provider is "none" (no LLM to verify)
        if memory_llm_provider == "none":
            self._skip_llm_verification = True
        memory_llm_api_key = memory_llm_api_key or config.llm_api_key
        if not memory_llm_api_key and requires_api_key(memory_llm_provider):
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

        # Database backend and SQL dialect (created during initialize())
        self._database_backend_type = config.database_backend
        self._backend: DatabaseBackend | None = None
        self._dialect: SQLDialect | None = None
        # Connection pool — set from backend.get_pool() for backward compatibility
        self._pool = None
        self._read_backend: DatabaseBackend | None = None
        self._read_database_url: str | None = (
            config.read_database_url if self._database_backend_type == "postgresql" else None
        )
        self._initialized = False
        self._pool_min_size = pool_min_size if pool_min_size is not None else config.db_pool_min_size
        self._pool_max_size = pool_max_size if pool_max_size is not None else config.db_pool_max_size
        self._read_pool_min_size = config.read_db_pool_min_size
        self._read_pool_max_size = config.read_db_pool_max_size
        self._db_command_timeout = db_command_timeout if db_command_timeout is not None else config.db_command_timeout
        self._db_acquire_timeout = db_acquire_timeout if db_acquire_timeout is not None else config.db_acquire_timeout
        self._db_statement_timeout = config.db_statement_timeout
        self._run_migrations = run_migrations
        self._retain_entity_lookup = config.retain_entity_lookup

        # Webhook manager (will be created in initialize() after pool is ready)
        self._webhook_manager = None
        self._http_client: httpx.AsyncClient | None = None

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
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.llm_litellmrouter_config,
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
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.retain_llm_litellmrouter_config or config.llm_litellmrouter_config,
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
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.reflect_llm_litellmrouter_config or config.llm_litellmrouter_config,
        )

        # Consolidation LLM config - for mental model consolidation (can use efficient models)
        consolidation_provider = consolidation_llm_provider or config.consolidation_llm_provider or memory_llm_provider
        consolidation_api_key = consolidation_llm_api_key or config.consolidation_llm_api_key or memory_llm_api_key
        consolidation_model = consolidation_llm_model or config.consolidation_llm_model or memory_llm_model
        consolidation_base_url = consolidation_llm_base_url or config.consolidation_llm_base_url or memory_llm_base_url
        # Apply provider-specific base URL defaults for consolidation
        if consolidation_base_url is None:
            if consolidation_provider.lower() == "groq":
                consolidation_base_url = "https://api.groq.com/openai/v1"
            elif consolidation_provider.lower() == "ollama":
                consolidation_base_url = "http://localhost:11434/v1"
            else:
                consolidation_base_url = ""

        self._consolidation_llm_config = LLMConfig(
            provider=consolidation_provider,
            api_key=consolidation_api_key,
            base_url=consolidation_base_url,
            model=consolidation_model,
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.consolidation_llm_litellmrouter_config or config.llm_litellmrouter_config,
        )

        # Initialize cross-encoder reranker (cached for performance)
        self._cross_encoder_reranker = CrossEncoderReranker(cross_encoder=cross_encoder)

        # Initialize task backend.
        # All backends use BrokerTaskBackend + WorkerPoller for async background execution.
        # Create the backend object early so we can query its capabilities.
        self._backend = create_database_backend(self._database_backend_type)
        if task_backend:
            self._task_backend = task_backend
        else:
            self._task_backend = self._backend.create_task_backend(
                pool_getter=lambda: self._backend,
                schema_getter=get_current_schema,
            )

        # Audit logger for feature usage tracking
        config = get_config()
        self._audit_logger = AuditLogger(
            pool_getter=lambda: self._backend,
            schema_getter=get_current_schema,
            enabled=config.audit_log_enabled,
            allowed_actions=config.audit_log_actions,
            retention_days=config.audit_log_retention_days,
        )

        # Backpressure mechanism: limit concurrent searches to prevent overwhelming the database
        # Configurable via HINDSIGHT_API_RECALL_MAX_CONCURRENT (default: 50)
        self._search_semaphore = asyncio.Semaphore(get_config().recall_max_concurrent)

        # Backpressure for retain DB writes: limit concurrent transactions to prevent contention
        # on entity/link tables. Acquired in the orchestrator *after* LLM extraction completes,
        # so LLM calls run in full parallelism while only the DB-heavy phase is throttled.
        # Configurable via HINDSIGHT_API_RETAIN_MAX_CONCURRENT (default: 4).
        self._put_semaphore = asyncio.Semaphore(get_config().retain_max_concurrent)

        # initialize encoding eagerly to avoid delaying the first time
        _get_tiktoken_encoding()

        # Store operation validator extension (optional)
        self._operation_validator = operation_validator

        # Store tenant extension (always set, use default if none provided)
        if tenant_extension is None:
            from ..extensions.builtin.tenant import DefaultTenantExtension

            tenant_extension = DefaultTenantExtension(config={})
        self._tenant_extension = tenant_extension

    @property
    def audit_logger(self) -> AuditLogger:
        """The audit logger for feature usage tracking."""
        return self._audit_logger

    @property
    def tenant_extension(self) -> "TenantExtension | None":
        """The configured tenant extension, if any."""
        return self._tenant_extension

    async def _validate_operation(self, validation_coro) -> "ValidationResult | None":
        """
        Run validation if an operation validator is configured.

        Args:
            validation_coro: Coroutine that returns a ValidationResult

        Returns:
            The ValidationResult (may contain enrichment fields), or None if no validator.

        Raises:
            OperationValidationError: If validation fails
        """
        if self._operation_validator is None:
            return None

        from hindsight_api.extensions import OperationValidationError, ValidationResult

        result = await validation_coro
        if not result.allowed:
            raise OperationValidationError(result.reason or "Operation not allowed", result.status_code)
        return result

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
        from hindsight_api.extensions import AuthenticationError

        if request_context is None:
            raise AuthenticationError("RequestContext is required")

        # For internal/background operations (e.g., worker tasks), skip extension authentication.
        # The task was already authenticated at submission time, and execute_task sets _current_schema
        # from the task's _schema field.
        if request_context.internal:
            return _current_schema.get()

        # For MCP requests already authenticated via MCP_AUTH_TOKEN, skip tenant re-validation.
        # The MCP transport layer already verified the token; re-validating against the tenant
        # extension would fail when MCP_AUTH_TOKEN and TENANT_API_KEY differ.
        if request_context.mcp_authenticated:
            return _current_schema.get()

        # Authenticate through tenant extension (always set, may be default no-auth extension)
        tenant_context = await self._tenant_extension.authenticate(request_context)

        _current_schema.set(tenant_context.schema_name)
        return tenant_context.schema_name

    async def _handle_batch_retain(self, task_dict: dict[str, Any]):
        """
        Handler for batch retain tasks.

        Args:
            task_dict: Dict with 'bank_id', 'contents', 'operation_id'

        Raises:
            ValueError: If bank_id is missing
            Exception: Any exception from retain_batch_async (propagates to execute_task for retry)
        """
        bank_id = task_dict.get("bank_id")
        if not bank_id:
            raise ValueError("bank_id is required for batch retain task")
        contents = task_dict.get("contents", [])
        document_tags = task_dict.get("document_tags")
        operation_id = task_dict.get("operation_id")  # For batch API crash recovery
        strategy = task_dict.get("strategy")

        logger.info(
            f"[BATCH_RETAIN_TASK] Starting background batch retain for bank_id={bank_id}, {len(contents)} items, operation_id={operation_id}"
        )

        # Restore tenant_id/api_key_id from task payload so extensions
        # (e.g., operation validators) can attribute the operation correctly.
        # internal=True to skip extension auth (worker has no API key),
        # user_initiated=True so extensions know this originated from a user request.
        from hindsight_api.models import RequestContext

        context = RequestContext(
            internal=True,
            user_initiated=True,
            tenant_id=task_dict.get("_tenant_id"),
            api_key_id=task_dict.get("_api_key_id"),
            retry_count=task_dict.get("_retry_count", 0),
        )
        await self.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            document_tags=document_tags,
            request_context=context,
            operation_id=operation_id,
            strategy=strategy,
            outbox_callback=self._build_retain_outbox_callback(
                bank_id=bank_id,
                contents=contents,
                operation_id=operation_id,
                schema=_current_schema.get(),
            ),
        )

        # If this retain was triggered by file conversion, update document with file metadata
        file_metadata = task_dict.get("_file_metadata")
        if file_metadata and len(contents) == 1:
            doc_id = contents[0].get("document_id")
            if doc_id:
                backend = await self._get_backend()
                async with acquire_with_retry(backend) as conn:
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("documents")}
                        SET file_storage_key = $3,
                            file_original_name = $4,
                            file_content_type = $5,
                            updated_at = NOW()
                        WHERE id = $1 AND bank_id = $2
                        """,
                        doc_id,
                        bank_id,
                        file_metadata["file_storage_key"],
                        file_metadata["file_original_name"],
                        file_metadata["file_content_type"],
                    )

        logger.info(f"[BATCH_RETAIN_TASK] Completed background batch retain for bank_id={bank_id}")

    async def _handle_file_convert_retain(self, task_dict: dict[str, Any]):
        """
        Handler for file conversion tasks.

        Converts a file to markdown, then submits a separate async retain operation
        and marks this conversion as completed — all in a single transaction.
        This avoids holding a worker slot during the expensive retain pipeline.

        Args:
            task_dict: Dict with 'bank_id', 'storage_key', 'parser', etc.

        Raises:
            ValueError: If required fields are missing
            Exception: Any exception from conversion (includes filename in error)
        """
        bank_id = task_dict.get("bank_id")
        storage_key = task_dict.get("storage_key")
        document_id = task_dict.get("document_id")
        operation_id = task_dict.get("operation_id")
        filename = task_dict.get("original_filename", "unknown")

        if not all([bank_id, storage_key, document_id]):
            raise ValueError("bank_id, storage_key, and document_id are required for file_convert_retain task")

        logger.info(f"[FILE_CONVERT_RETAIN] Starting for bank_id={bank_id}, document_id={document_id}, file={filename}")

        try:
            # Retrieve file from storage
            file_data = await self._file_storage.retrieve(storage_key)

            # Convert to markdown using the ordered fallback chain stored in the task payload.
            # task_dict["parser"] is always a list[str] set at submission time.
            parser_chain: list[str] = task_dict.get("parser") or []
            if not parser_chain:
                raise ValueError("No parser chain defined for file_convert_retain task")
            convert_result = await self._parser_registry.convert_with_fallback(
                parsers=parser_chain,
                file_data=file_data,
                filename=filename,
                content_type=task_dict.get("content_type"),
            )
            markdown_content = sanitize_llm_output(convert_result.content) or ""
            winning_parser = convert_result.parser_name
        except Exception as e:
            # Re-raise with filename context for better error reporting
            error_msg = f"Failed to parse file '{filename}': {str(e)}"
            logger.error(f"[FILE_CONVERT_RETAIN] {error_msg}")
            raise RuntimeError(error_msg) from e

        logger.info(
            f"[FILE_CONVERT_RETAIN] Converted file for bank_id={bank_id}, "
            f"document_id={document_id}, {len(markdown_content)} chars. Submitting retain task."
        )

        # Fire file conversion hook (e.g., for Iris billing)
        if self._operation_validator:
            try:
                from hindsight_api.extensions.operation_validator import FileConvertResult
                from hindsight_api.models import RequestContext

                convert_context = RequestContext(
                    internal=True,
                    user_initiated=True,
                    tenant_id=task_dict.get("_tenant_id"),
                    api_key_id=task_dict.get("_api_key_id"),
                    retry_count=task_dict.get("_retry_count", 0),
                )
                await self._operation_validator.on_file_convert_complete(
                    FileConvertResult(
                        bank_id=bank_id,
                        parser_name=winning_parser,
                        filename=filename,
                        output_chars=len(markdown_content),
                        output_text=markdown_content,
                        request_context=convert_context,
                    )
                )
            except Exception as e:
                logger.warning(f"[FILE_CONVERT_RETAIN] on_file_convert_complete hook failed: {e}")

        # Build retain task payload
        retain_content: dict[str, Any] = {
            "content": markdown_content,
            "document_id": document_id,
            "context": task_dict.get("context"),
            "metadata": task_dict.get("metadata", {}),
            "tags": task_dict.get("tags", []),
        }
        file_timestamp = task_dict.get("timestamp")
        if file_timestamp == "unset":
            retain_content["event_date"] = None
        elif file_timestamp:
            retain_content["event_date"] = file_timestamp
        retain_contents = [retain_content]
        document_tags = task_dict.get("document_tags")

        retain_task_payload: dict[str, Any] = {"contents": retain_contents}
        if document_tags:
            retain_task_payload["document_tags"] = document_tags
        if task_dict.get("strategy"):
            retain_task_payload["strategy"] = task_dict["strategy"]

        # Pass tenant/api_key context through to retain task
        if task_dict.get("_tenant_id"):
            retain_task_payload["_tenant_id"] = task_dict["_tenant_id"]
        if task_dict.get("_api_key_id"):
            retain_task_payload["_api_key_id"] = task_dict["_api_key_id"]

        # File metadata to attach after retain creates the document
        retain_task_payload["_file_metadata"] = {
            "file_storage_key": storage_key,
            "file_original_name": task_dict["original_filename"],
            "file_content_type": task_dict["content_type"],
        }

        # Include task_payload in the INSERT atomically. Previously this was a
        # two-step process (INSERT without payload, then UPDATE to set it) which
        # left null-payload rows when a crash or timeout occurred between the two
        # statements. The worker claim query filters on `task_payload IS NOT NULL`,
        # so those orphaned rows became permanently stuck as unclaimed pending tasks.
        retain_operation_id = uuid.uuid4()
        full_retain_payload = {
            "type": "batch_retain",
            "operation_id": str(retain_operation_id),
            "bank_id": bank_id,
            **retain_task_payload,
        }
        payload_json = json.dumps(full_retain_payload, default=_json_default)

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table("async_operations")}
                    (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                    """,
                    retain_operation_id,
                    bank_id,
                    "retain",
                    json.dumps({}),
                    "pending",
                    payload_json,
                )

                if operation_id:
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET status = 'completed', updated_at = NOW(), completed_at = NOW()
                        WHERE operation_id = $1
                        """,
                        uuid.UUID(operation_id),
                    )

        # For SyncTaskBackend: executes the retain task inline.
        # For BrokerTaskBackend: no-op (submit_task's UPDATE skips rows whose
        # task_payload is already set, which it is after the INSERT above).
        await self._task_backend.submit_task(full_retain_payload)

        logger.info(
            f"[FILE_CONVERT_RETAIN] Completed conversion for bank_id={bank_id}, "
            f"document_id={document_id}. Retain task submitted as operation {retain_operation_id}"
        )

        # Delete file bytes from storage if configured (saves storage costs)
        from ..config import get_config

        config = get_config()
        if config.file_delete_after_retain:
            try:
                await self._file_storage.delete(storage_key)
                logger.info(f"[FILE_CONVERT_RETAIN] Deleted file bytes for {storage_key} (conversion completed)")
            except Exception as e:
                # Non-fatal - log and continue
                logger.warning(f"[FILE_CONVERT_RETAIN] Failed to delete file {storage_key}: {e}")

    async def _handle_consolidation(self, task_dict: dict[str, Any]):
        """
        Handler for consolidation tasks.

        Consolidates new memories into mental models for a bank.

        Args:
            task_dict: Dict with 'bank_id'

        Raises:
            ValueError: If bank_id is missing
            Exception: Any exception from consolidation (propagates to execute_task for retry)
        """
        bank_id = task_dict.get("bank_id")
        if not bank_id:
            raise ValueError("bank_id is required for consolidation task")

        # Skip consolidation when LLM provider is "none"
        if self._llm_config.provider == "none":
            logger.info(f"[CONSOLIDATION] Skipping consolidation for bank {bank_id}: LLM provider is 'none'")
            return {"memories_processed": 0, "skipped": True}

        from hindsight_api.models import RequestContext

        from .consolidation import run_consolidation_job

        # Restore tenant_id/api_key_id from task payload so downstream operations
        # (e.g., mental model refreshes) can attribute usage to the correct org.
        internal_context = RequestContext(
            internal=True,
            tenant_id=task_dict.get("_tenant_id"),
            api_key_id=task_dict.get("_api_key_id"),
            retry_count=task_dict.get("_retry_count", 0),
        )
        result = await run_consolidation_job(
            memory_engine=self,
            bank_id=bank_id,
            request_context=internal_context,
            operation_id=task_dict.get("operation_id"),
        )

        logger.info(f"[CONSOLIDATION] bank={bank_id} completed: {result.get('memories_processed', 0)} processed")
        return result

    async def _handle_refresh_mental_model(self, task_dict: dict[str, Any]):
        """
        Handler for refresh_mental_model tasks.

        Delegates to ``refresh_mental_model`` so async (worker-driven) refreshes
        and synchronous refreshes share the same code path — including the
        structured-delta logic. Previously this handler had its own copy of the
        reflect+update pipeline, which silently bypassed structured delta when
        the UI/worker queued the task. The duplication caused the original
        "delta refresh produced full-document drift" bug to persist even after
        delta was implemented on the synchronous path.

        Args:
            task_dict: Dict with 'bank_id', 'mental_model_id', 'operation_id'

        Raises:
            ValueError: If required fields are missing
            Exception: Any exception from refresh_mental_model (propagates for retry)
        """
        bank_id = task_dict.get("bank_id")
        mental_model_id = task_dict.get("mental_model_id")

        if not bank_id or not mental_model_id:
            raise ValueError("bank_id and mental_model_id are required for refresh_mental_model task")

        logger.info(f"[REFRESH_MENTAL_MODEL_TASK] Starting for bank_id={bank_id}, mental_model_id={mental_model_id}")

        from hindsight_api.models import RequestContext

        # Restore tenant_id/api_key_id from task payload so extensions can
        # attribute the mental_model_refresh operation to the correct org.
        internal_context = RequestContext(
            internal=True,
            tenant_id=task_dict.get("_tenant_id"),
            api_key_id=task_dict.get("_api_key_id"),
            retry_count=task_dict.get("_retry_count", 0),
        )

        refreshed = await self.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=mental_model_id,
            request_context=internal_context,
        )
        if refreshed is None:
            raise ValueError(f"Mental model {mental_model_id} not found in bank {bank_id}")

        # Compute facts/mental_models counts for the post-op validator hook.
        # refresh_mental_model already persisted everything; the hook only needs
        # tallies that derive from the stored reflect_response payload.
        rr = refreshed.get("reflect_response") or {}
        based_on = rr.get("based_on") or {}
        facts_used = 0
        mental_models_used = 0
        for fact_type, facts in based_on.items():
            n = len(facts) if facts else 0
            if fact_type in ("mental_models", "mental-models"):
                mental_models_used += n
            else:
                facts_used += n
        source_query = refreshed.get("source_query") or ""
        generated_content = refreshed.get("content") or ""

        # Call post-operation hook if validator is configured
        if self._operation_validator:
            from hindsight_api.extensions.operation_validator import MentalModelRefreshResult

            # Estimate tokens
            query_tokens = len(source_query) // 4 if source_query else 0
            output_tokens = len(generated_content) // 4 if generated_content else 0
            context_tokens = 0  # refresh doesn't use additional context

            result_ctx = MentalModelRefreshResult(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=internal_context,
                query_tokens=query_tokens,
                output_tokens=output_tokens,
                context_tokens=context_tokens,
                facts_used=facts_used,
                mental_models_used=mental_models_used,
                success=True,
            )
            try:
                await self._operation_validator.on_mental_model_refresh_complete(result_ctx)
            except Exception as hook_err:
                logger.warning(f"Post-mental-model-refresh hook error (non-fatal): {hook_err}")

        logger.info(f"[REFRESH_MENTAL_MODEL_TASK] Completed for bank_id={bank_id}, mental_model_id={mental_model_id}")

    async def execute_task(self, task_dict: dict[str, Any]):
        """
        Execute a task by routing it to the appropriate handler.

        This method is called by the task backend to execute tasks.
        It receives a plain dict that can be serialized and sent over the network.

        Args:
            task_dict: Task dictionary with 'type' key and other payload data
                      Example: {'type': 'batch_retain', 'bank_id': '...', 'contents': [...]}
        """
        task_type = task_dict.get("type")
        operation_id = task_dict.get("operation_id")

        # Set schema context for multi-tenant task execution
        schema = task_dict.pop("_schema", None)
        if schema:
            _current_schema.set(schema)

        # Check if operation was cancelled (only for tasks with operation_id)
        if operation_id:
            try:
                backend = await self._get_backend()
                async with acquire_with_retry(backend) as conn:
                    result = await conn.fetchrow(
                        f"SELECT status FROM {fq_table('async_operations')} WHERE operation_id = $1",
                        uuid.UUID(operation_id),
                    )
                    if not result or result["status"] == "cancelled":
                        # Operation was cancelled, skip processing
                        logger.info(f"Skipping cancelled operation: {operation_id}")
                        return
            except Exception as e:
                logger.error(f"Failed to check operation status {operation_id}: {e}")
                # Continue with processing if we can't check status

        consolidation_result: dict | None = None
        bank_id = task_dict.get("bank_id")
        async with audit_context(
            self._audit_logger, task_type or "unknown", "system", bank_id, request=task_dict
        ) as audit_entry:
            try:
                # Stage breadcrumb for the worker poller's WORKER_TASK log line.
                # No-op outside a worker context.
                set_stage(f"task.{task_type}")
                if task_type == "batch_retain":
                    await self._handle_batch_retain(task_dict)
                elif task_type == "file_convert_retain":
                    await self._handle_file_convert_retain(task_dict)
                elif task_type == "consolidation":
                    consolidation_result = await self._handle_consolidation(task_dict)
                elif task_type == "refresh_mental_model":
                    await self._handle_refresh_mental_model(task_dict)
                elif task_type == "webhook_delivery":
                    await self._handle_webhook_delivery(task_dict)
                else:
                    logger.error(f"Unknown task type: {task_type}")
                    # Don't retry unknown task types
                    if operation_id:
                        await self._delete_operation_record(operation_id)
                    return

                # Task succeeded - mark operation as completed
                # file_convert_retain marks itself as completed in a transaction, skip double-marking
                if operation_id and task_type not in ("file_convert_retain",):
                    if task_type == "consolidation":
                        # Atomically mark completed AND queue webhook delivery in one transaction
                        await self._mark_operation_completed_and_fire_webhook(
                            operation_id=operation_id,
                            bank_id=task_dict.get("bank_id", ""),
                            status="completed",
                            result=consolidation_result,
                            schema=schema,
                        )
                    else:
                        await self._mark_operation_completed(operation_id)

                audit_entry.response = {"status": "completed", "operation_id": operation_id}

            except RetryTaskAt:
                # Task-owned retry: let the poller handle scheduling
                raise
            except DeferOperation:
                # Task-owned defer: let the poller handle re-scheduling without
                # bumping retry_count or writing error_message. Pairs with the
                # DeferOperation catch in poller._execute_task_inner (PR #1105);
                # without this passthrough, the generic-exception branch below
                # would convert a legitimate defer into a 60-second RetryTaskAt
                # and lose the "not a failure" semantics entirely.
                raise
            except Exception as e:
                logger.error(f"Task execution failed: {task_type}, error: {e}")
                import traceback

                error_traceback = traceback.format_exc()
                traceback.print_exc()

                if task_type == "file_convert_retain":
                    # Non-retryable: mark as failed immediately.
                    # Conversion failures won't improve on retry (missing OCR, corrupted file, etc.)
                    logger.error(f"Not retrying task {task_type} (non-retryable), marking as failed")
                    if operation_id:
                        await self._mark_operation_failed(operation_id, str(e), error_traceback)
                elif isinstance(e, asyncpg.exceptions.IntegrityConstraintViolationError) or (
                    _is_oracledb_integrity_error(e)
                ):
                    # Non-retryable: deterministic integrity violations (PG or Oracle)
                    # (UniqueViolationError, ForeignKeyViolationError, CheckViolationError,
                    # NotNullViolationError, ExclusionViolationError / ORA-00001, ORA-02291, etc.)
                    # will never succeed on retry — the offending row state is already committed.
                    # Retrying just burns worker capacity. See vectorize-io/hindsight#980.
                    logger.error(
                        f"Not retrying task {task_type} (integrity violation, deterministic): {type(e).__name__}"
                    )
                    if task_type == "consolidation" and operation_id:
                        await self._fire_consolidation_webhook(
                            bank_id=task_dict.get("bank_id", ""),
                            operation_id=operation_id,
                            status="failed",
                            result=None,
                            error_message=str(e),
                            schema=schema,
                        )
                    if operation_id:
                        await self._mark_operation_failed(operation_id, str(e), error_traceback)
                else:
                    if task_type == "consolidation" and operation_id:
                        # Fire failure webhook (non-transactional — operation not yet marked failed;
                        # poller will mark it failed after this raise)
                        await self._fire_consolidation_webhook(
                            bank_id=task_dict.get("bank_id", ""),
                            operation_id=operation_id,
                            status="failed",
                            result=None,
                            error_message=str(e),
                            schema=schema,
                        )
                    # Retryable: use RetryTaskAt if under the retry limit, else re-raise (poller marks failed)
                    retry_count = task_dict.get("_retry_count", 0)
                    if retry_count < 3:
                        raise RetryTaskAt(retry_at=datetime.now(UTC) + timedelta(seconds=60), message=str(e))
                    raise

    async def _fire_consolidation_webhook(
        self,
        bank_id: str,
        operation_id: str,
        status: str,
        result: dict | None,
        error_message: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Fire a consolidation webhook event. Non-fatal - logs errors but does not raise."""
        if not self._webhook_manager:
            return
        try:
            from datetime import datetime, timezone

            from ..webhooks.models import ConsolidationEventData, WebhookEvent, WebhookEventType

            data = ConsolidationEventData(
                observations_created=result.get("observations_created") if result else None,
                observations_updated=result.get("observations_updated") if result else None,
                observations_deleted=result.get("observations_deleted") if result else None,
                error_message=error_message,
            )
            event = WebhookEvent(
                event=WebhookEventType.CONSOLIDATION_COMPLETED,
                bank_id=bank_id,
                operation_id=operation_id,
                status=status,
                timestamp=datetime.now(timezone.utc),
                data=data,
            )
            await self._webhook_manager.fire_event(event, schema=schema)
        except Exception as e:
            logger.error(f"Failed to fire consolidation webhook for operation {operation_id}: {e}")

    def _build_retain_outbox_callback(
        self,
        bank_id: str,
        contents: list[dict],
        operation_id: str | None,
        schema: str | None = None,
    ) -> "Callable[[asyncpg.Connection], Awaitable[None]] | None":
        """Build a transactional outbox callback for retain.completed webhook events.

        Returns a coroutine function that queues one webhook delivery row per content
        item using the provided connection (inside the retain transaction). Returns None
        if no webhook manager is configured.
        """
        webhook_manager = getattr(self, "_webhook_manager", None)
        if not webhook_manager:
            return None

        from ..webhooks.models import RetainEventData, WebhookEvent, WebhookEventType

        now = datetime.now(UTC)
        op_id = operation_id or uuid.uuid4().hex
        events = []
        for content in contents:
            doc_id = content.get("document_id")
            tags = content.get("tags")
            data = RetainEventData(
                document_id=doc_id,
                tags=tags if isinstance(tags, list) else None,
            )
            events.append(
                WebhookEvent(
                    event=WebhookEventType.RETAIN_COMPLETED,
                    bank_id=bank_id,
                    operation_id=op_id,
                    status="completed",
                    timestamp=now,
                    data=data,
                )
            )

        async def _callback(conn: asyncpg.Connection) -> None:
            # Resolve schema at call time (not at callback creation time) because
            # _current_schema contextvar may not yet be set when the callback is built
            # from the HTTP path (http.py calls _build_retain_outbox_callback before
            # retain_batch_async which is where _authenticate_tenant sets the schema).
            resolved_schema = schema or _current_schema.get()
            for event in events:
                await webhook_manager.fire_event_with_conn(event, conn, schema=resolved_schema)

        return _callback

    async def _update_webhook_delivery_metadata(
        self, operation_id: str, status_code: int | None, response_body: str | None
    ) -> None:
        """Persist last HTTP attempt info into async_operations.result_metadata."""
        try:
            backend = await self._get_backend()
            meta = json.dumps(
                {
                    "last_status_code": status_code,
                    "last_response_body": (response_body or "")[:2048],
                    "last_attempt_at": datetime.now(UTC).isoformat(),
                }
            )
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"UPDATE {fq_table('async_operations')} SET result_metadata = $2::jsonb, updated_at = now() WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                    meta,
                )
        except Exception as meta_err:
            logger.debug(f"Failed to update webhook delivery metadata: {meta_err}")

    async def _handle_webhook_delivery(self, task_dict: dict[str, Any]) -> None:
        """Deliver a webhook event via HTTP.

        Raises RetryTaskAt to schedule a retry on failure (up to MAX_ATTEMPTS).
        Raises the original exception when retries are exhausted (poller marks failed).
        Response status code and body are stored in result_metadata for debugging.
        """
        from ..webhooks.manager import MAX_ATTEMPTS, RETRY_DELAYS
        from ..webhooks.models import WebhookHttpConfig

        url = task_dict["url"]
        secret = task_dict.get("secret")
        event_type = task_dict["event_type"]
        raw_payload = task_dict["payload"]
        retry_count = task_dict.get("_retry_count", 0)
        operation_id: str | None = task_dict.get("_operation_id")
        http_config = WebhookHttpConfig.model_validate(task_dict.get("http_config") or {})

        if isinstance(raw_payload, dict):
            payload_bytes = json.dumps(raw_payload).encode()
        else:
            payload_bytes = str(raw_payload).encode()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Hindsight-Event": event_type,
            **http_config.headers,
        }
        if secret and self._webhook_manager:
            headers["X-Hindsight-Signature"] = self._webhook_manager._sign_payload(secret, payload_bytes)

        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")

        response = None
        try:
            request_kwargs: dict[str, Any] = {
                "headers": headers,
                "params": http_config.params if http_config.params else None,
                "timeout": http_config.timeout_seconds,
            }
            if http_config.method.upper() == "GET":
                response = await self._http_client.get(url, **request_kwargs)
            else:
                response = await self._http_client.post(url, content=payload_bytes, **request_kwargs)
            response.raise_for_status()
            if operation_id:
                await self._update_webhook_delivery_metadata(operation_id, response.status_code, response.text)
        except Exception as e:
            status_code = response.status_code if response is not None else None
            response_body = response.text if response is not None else None
            if operation_id:
                await self._update_webhook_delivery_metadata(operation_id, status_code, response_body)
            if retry_count >= MAX_ATTEMPTS - 1:
                logger.error(
                    f"webhook_delivery permanently_failed url={url} attempts={retry_count + 1} "
                    f"status_code={status_code} error={e}"
                )
                raise
            delay = RETRY_DELAYS[retry_count] if retry_count < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
            retry_at = datetime.now(UTC) + timedelta(seconds=delay)
            logger.warning(
                f"webhook_delivery failed url={url} attempt={retry_count + 1}/{MAX_ATTEMPTS} "
                f"status_code={status_code} retry_in={delay}s error={e}"
            )
            raise RetryTaskAt(retry_at=retry_at, message=str(e))

    async def _delete_operation_record(self, operation_id: str):
        """Helper to delete an operation record from the database."""
        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"DELETE FROM {fq_table('async_operations')} WHERE operation_id = $1", uuid.UUID(operation_id)
                )
        except Exception as e:
            logger.error(f"Failed to delete async operation record {operation_id}: {e}")

    async def _check_op_alive(self, operation_id: str) -> bool:
        """Return False if the operation was cancelled or no longer exists (e.g. bank deleted via CASCADE).

        Long-running operations should call this at natural checkpoints (e.g. after each
        committed batch) to detect cancellation or bank deletion early and abort cleanly.
        """
        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                row = await conn.fetchrow(
                    f"SELECT status FROM {fq_table('async_operations')} WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                )
                return row is not None and row["status"] != "cancelled"
        except Exception as e:
            logger.error(f"Failed to check operation liveness {operation_id}: {e}")
            return True  # Assume alive on DB error to avoid false-positive aborts

    async def _mark_operation_failed(self, operation_id: str, error_message: str, error_traceback: str):
        """Helper to mark an operation as failed in the database.

        Also checks if this is a child operation and updates the parent if all siblings are done.
        Uses a single transaction to avoid race conditions when multiple children fail simultaneously.
        """
        try:
            backend = await self._get_backend()
            # Truncate error message to avoid extremely long strings
            full_error = f"{error_message}\n\nTraceback:\n{error_traceback}"
            truncated_error = full_error[:5000] if len(full_error) > 5000 else full_error

            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    # Mark this operation as failed
                    row = await conn.fetchrow(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET status = 'failed', error_message = $2, updated_at = NOW()
                        WHERE operation_id = $1
                        RETURNING operation_id
                        """,
                        uuid.UUID(operation_id),
                        truncated_error,
                    )
                    if row is None:
                        logger.info(f"Operation {operation_id} no longer exists (bank deleted), skipping mark-failed")
                        return
                    logger.info(f"Marked async operation as failed: {operation_id}")

                    # Check if this is a child operation and update parent if all siblings are done
                    # This happens in the same transaction after the child status is updated
                    await self._maybe_update_parent_operation(operation_id, conn)
        except Exception as e:
            logger.error(f"Failed to mark operation as failed {operation_id}: {e}")

    async def _mark_operation_completed(self, operation_id: str):
        """Helper to mark an operation as completed in the database.

        Also checks if this is a child operation and updates the parent if all siblings are done.
        Uses a single transaction to avoid race conditions when multiple children complete simultaneously.
        """
        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    # Mark this operation as completed
                    row = await conn.fetchrow(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET status = 'completed', updated_at = NOW(), completed_at = NOW()
                        WHERE operation_id = $1
                        RETURNING operation_id
                        """,
                        uuid.UUID(operation_id),
                    )
                    if row is None:
                        logger.info(
                            f"Operation {operation_id} no longer exists (bank deleted), skipping mark-completed"
                        )
                        return
                    logger.info(f"Marked async operation as completed: {operation_id}")

                    # Check if this is a child operation and update parent if all siblings are done
                    # This happens in the same transaction after the child status is updated
                    await self._maybe_update_parent_operation(operation_id, conn)
        except Exception as e:
            logger.error(f"Failed to mark operation as completed {operation_id}: {e}")

    async def _mark_operation_completed_and_fire_webhook(
        self,
        operation_id: str,
        bank_id: str,
        status: str,
        result: dict | None,
        schema: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark an operation as completed and queue webhook deliveries in a single transaction.

        Uses the transactional outbox pattern: the webhook delivery row is inserted in the
        same database transaction as the status update. This guarantees at-least-once delivery
        even if the process crashes immediately after committing.
        """
        from ..webhooks.models import ConsolidationEventData, WebhookEvent, WebhookEventType

        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        f"""
                        UPDATE {fq_table("async_operations")}
                        SET status = 'completed', updated_at = NOW(), completed_at = NOW()
                        WHERE operation_id = $1
                        RETURNING operation_id
                        """,
                        uuid.UUID(operation_id),
                    )
                    if row is None:
                        logger.info(
                            f"Operation {operation_id} no longer exists (bank deleted), skipping mark-completed"
                        )
                        return
                    logger.info(f"Marked async operation as completed: {operation_id}")
                    await self._maybe_update_parent_operation(operation_id, conn)

                    # Queue webhook deliveries inside the same transaction
                    if self._webhook_manager:
                        data = ConsolidationEventData(
                            observations_created=result.get("observations_created") if result else None,
                            observations_updated=result.get("observations_updated") if result else None,
                            observations_deleted=result.get("observations_deleted") if result else None,
                            error_message=error_message,
                        )
                        event = WebhookEvent(
                            event=WebhookEventType.CONSOLIDATION_COMPLETED,
                            bank_id=bank_id,
                            operation_id=operation_id,
                            status=status,
                            timestamp=datetime.now(UTC),
                            data=data,
                        )
                        await self._webhook_manager.fire_event_with_conn(event, conn, schema=schema)
        except Exception as e:
            logger.error(f"Failed to mark operation completed and fire webhook {operation_id}: {e}")

    async def _maybe_update_parent_operation(self, child_operation_id: str, conn):
        """Check if this is a child operation and update parent status if all siblings are done.

        Must be called within an active transaction that has already updated the child's status.
        Uses SELECT FOR UPDATE to lock the parent and prevent race conditions.

        Args:
            child_operation_id: The operation ID that just completed or failed
            conn: Database connection with an active transaction
        """
        try:
            # Get this operation's metadata to check if it has a parent
            row = await conn.fetchrow(
                f"""
                SELECT result_metadata, bank_id
                FROM {fq_table("async_operations")}
                WHERE operation_id = $1
                """,
                uuid.UUID(child_operation_id),
            )

            if not row:
                return

            raw_rm = row["result_metadata"]
            result_metadata = conn.parse_json(raw_rm) or {}
            parent_operation_id = result_metadata.get("parent_operation_id")

            if not parent_operation_id:
                # Not a child operation
                return

            bank_id = row["bank_id"]

            # Lock the parent operation to prevent concurrent updates from other children
            # Use FOR UPDATE to ensure only one child can update the parent at a time
            parent_row = await conn.fetchrow(
                f"""
                SELECT operation_id
                FROM {fq_table("async_operations")}
                WHERE operation_id = $1 AND bank_id = $2
                FOR UPDATE
                """,
                uuid.UUID(parent_operation_id),
                bank_id,
            )

            if not parent_row:
                # Parent doesn't exist (shouldn't happen)
                return

            # Get all sibling operations (including this one).
            # This query runs in the same transaction, so it sees the current
            # child's updated status. Pull error_message too so a parent that
            # fails can inherit a representative child reason -- otherwise
            # downstream consumers (dashboards, alert filters) lose the actual
            # cause once a batch has children. See the worker poller's
            # _summarise_child_error_messages for the propagation rationale.
            siblings = await conn.fetch(
                f"""
                SELECT status, error_message
                FROM {fq_table("async_operations")}
                WHERE bank_id = $1
                AND result_metadata::jsonb @> $2::jsonb
                """,
                bank_id,
                json.dumps({"parent_operation_id": parent_operation_id}),
            )

            if not siblings:
                return

            # Check if all siblings are done (completed or failed)
            all_completed = all(sib["status"] == "completed" for sib in siblings)
            any_failed = any(sib["status"] == "failed" for sib in siblings)
            all_done = all(sib["status"] in ("completed", "failed") for sib in siblings)

            if not all_done:
                # Some siblings still pending/processing
                return

            # All siblings are done - update parent status
            if any_failed:
                new_status = "failed"
                # Set parent error message to indicate child failure. Inherit
                # the most-common failed-child error_message rather than a
                # generic string so downstream filters can attribute the
                # cause correctly.
                from hindsight_api.worker.poller import _summarise_child_error_messages

                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET status = $2, error_message = $3, updated_at = NOW()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                    new_status,
                    _summarise_child_error_messages(siblings),
                )
            elif all_completed:
                new_status = "completed"
                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET status = $2, updated_at = NOW(), completed_at = NOW()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                    new_status,
                )

            logger.info(f"Updated parent operation {parent_operation_id} to status '{new_status}' (all children done)")

        except Exception as e:
            logger.error(f"Failed to update parent operation for child {child_operation_id}: {e}")
            # Re-raise to rollback the transaction
            raise

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
                pg0 = EmbeddedPostgres(**kwargs)
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
            """Verify LLM connections are working for all unique configs.

            Failures are logged as warnings instead of raising — the server will
            still start so queued operations can be processed once the LLM
            provider becomes available (e.g. after a quota reset).
            """
            if not self._skip_llm_verification:
                configs_to_verify: list[tuple[str, LLMConfig]] = [("default", self._llm_config)]

                # Verify retain config if different from default
                retain_is_different = (
                    self._retain_llm_config.provider != self._llm_config.provider
                    or self._retain_llm_config.model != self._llm_config.model
                )
                if retain_is_different:
                    configs_to_verify.append(("retain", self._retain_llm_config))

                # Verify reflect config if different from default and retain
                reflect_is_different = (
                    self._reflect_llm_config.provider != self._llm_config.provider
                    or self._reflect_llm_config.model != self._llm_config.model
                ) and (
                    self._reflect_llm_config.provider != self._retain_llm_config.provider
                    or self._reflect_llm_config.model != self._retain_llm_config.model
                )
                if reflect_is_different:
                    configs_to_verify.append(("reflect", self._reflect_llm_config))

                # Verify consolidation config if different from all others
                consolidation_is_different = (
                    (
                        self._consolidation_llm_config.provider != self._llm_config.provider
                        or self._consolidation_llm_config.model != self._llm_config.model
                    )
                    and (
                        self._consolidation_llm_config.provider != self._retain_llm_config.provider
                        or self._consolidation_llm_config.model != self._retain_llm_config.model
                    )
                    and (
                        self._consolidation_llm_config.provider != self._reflect_llm_config.provider
                        or self._consolidation_llm_config.model != self._reflect_llm_config.model
                    )
                )
                if consolidation_is_different:
                    configs_to_verify.append(("consolidation", self._consolidation_llm_config))

                for config_name, llm_config in configs_to_verify:
                    try:
                        await llm_config.verify_connection()
                    except Exception as e:
                        logger.warning(
                            "LLM connection verification failed for '%s' config: %s. "
                            "Server will start but LLM-dependent operations may fail "
                            "until the provider is available.",
                            config_name,
                            e,
                        )

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
            if not self.db_url:
                raise ValueError("Database URL is required for migrations")

            config = get_config()

            # Run schema migrations via the backend's migration runner.
            # Each backend handles its own migration strategy:
            # - PG: Alembic migrations with schema support
            # - Oracle: idempotent DDL runner (no Alembic)
            logger.info("Running database migrations...")
            tenants = await self._tenant_extension.list_tenants()
            if tenants:
                logger.info(f"Running migrations on {len(tenants)} schema(s)...")
                for tenant in tenants:
                    schema = tenant.schema
                    if schema:
                        schema = self._backend.normalize_schema(schema)
                        self._backend.run_migrations(self.db_url, schema=schema)
                logger.info("Schema migrations completed")

            # PG-specific post-migration steps: ensure vector/text search extensions
            # and embedding dimensions match configuration. These are no-ops for
            # non-PG backends since they use different indexing strategies.
            if self._backend.supports_bm25:
                from ..migrations import (
                    ensure_embedding_dimension,
                    ensure_text_search_extension,
                    ensure_vector_extension,
                )

                if tenants:
                    for tenant in tenants:
                        schema = tenant.schema
                        if schema:
                            ensure_embedding_dimension(
                                self.db_url,
                                self.embeddings.dimension,
                                schema=schema,
                                vector_extension=config.vector_extension,
                            )

                    for tenant in tenants:
                        schema = tenant.schema
                        if schema:
                            ensure_vector_extension(
                                self.db_url, vector_extension=config.vector_extension, schema=schema
                            )

                    for tenant in tenants:
                        schema = tenant.schema
                        if schema:
                            ensure_text_search_extension(
                                self.db_url, text_search_extension=config.text_search_extension, schema=schema
                            )

        logger.info(f"Connecting to database at {mask_network_location(self.db_url)}")

        # Create SQL dialect via abstraction layer
        # (backend was created in __init__ so we can use it for migrations and task backend)
        self._dialect = create_sql_dialect(self._database_backend_type)

        stmt_timeout_s = self._db_statement_timeout

        # Per-connection initialization callback (PostgreSQL-specific for now)
        async def _init_connection(conn: asyncpg.Connection) -> None:
            # SET (not SET LOCAL) so it persists for the connection lifetime.
            # ef_search=200 improves HNSW recall quality for the per-fact_type
            # semantic queries in retrieve_semantic_bm25_combined().
            try:
                await conn.execute("SET hnsw.ef_search = 200")
            except Exception:
                logger.debug("Could not set hnsw.ef_search — extension may not support it")

            # Server-side safety net for runaway queries. Migrations use a
            # separate SQLAlchemy/psycopg2 engine, so long-running DDL is
            # unaffected. 0 disables.
            if stmt_timeout_s > 0:
                await conn.execute(f"SET statement_timeout = '{stmt_timeout_s}s'")

        await self._backend.initialize(
            self.db_url,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
            command_timeout=self._db_command_timeout,
            acquire_timeout=self._db_acquire_timeout,
            statement_cache_size=0,
            init_callback=_init_connection,
        )

        # Expose raw pool for backward compatibility with consumers that
        # still use pool.acquire() / acquire_with_retry(pool) directly.
        # These will be migrated to use self._backend.acquire() over time.
        self._pool = self._backend.get_pool()

        if self._read_database_url:
            logger.info(
                f"Opening read backend against {mask_network_location(self._read_database_url)} for recall queries"
            )
            self._read_backend = create_database_backend(self._database_backend_type)
            await self._read_backend.initialize(
                self._read_database_url,
                min_size=self._read_pool_min_size,
                max_size=self._read_pool_max_size,
                command_timeout=self._db_command_timeout,
                acquire_timeout=self._db_acquire_timeout,
                statement_cache_size=0,
                init_callback=_init_connection,
            )
        else:
            self._read_backend = self._backend

        # Initialize entity resolver with pool and configured lookup strategy
        self.entity_resolver = EntityResolver(
            self._backend,
            entity_lookup=self._retain_entity_lookup,
        )

        # Initialize config resolver for hierarchical configuration
        from ..config_resolver import ConfigResolver

        self._config_resolver = ConfigResolver(backend=self._backend, tenant_extension=self._tenant_extension)
        logger.debug("Config resolver initialized for hierarchical configuration")

        # Initialize file storage
        from .storage import create_file_storage

        config = get_config()
        self._file_storage = create_file_storage(
            storage_type=config.file_storage_type,
            pool_getter=lambda: self._backend,
            schema_getter=get_current_schema,
        )
        logger.debug(f"File storage initialized ({config.file_storage_type})")

        # Initialize parser registry
        from .parsers import FileParserRegistry, IrisParser, LlamaParseParser, MarkitdownParser

        self._parser_registry = FileParserRegistry()
        try:
            self._parser_registry.register(MarkitdownParser())
            logger.debug("Registered markitdown parser")
        except ImportError:
            logger.warning("markitdown not available - file parsing disabled")
        iris_token = config.file_parser_iris_token
        iris_org_id = config.file_parser_iris_org_id
        if iris_token and iris_org_id:
            self._parser_registry.register(IrisParser(token=iris_token, org_id=iris_org_id))
            logger.debug("Registered iris parser")
        else:
            logger.debug("Iris parser not registered (VECTORIZE_TOKEN or VECTORIZE_ORG_ID not set)")
        llama_parse_key = config.file_parser_llama_parse_api_key
        if llama_parse_key:
            self._parser_registry.register(LlamaParseParser(api_key=llama_parse_key))
            logger.debug("Registered llama_parse parser")
        else:
            logger.debug("LlamaParse parser not registered (HINDSIGHT_API_FILE_PARSER_LLAMA_PARSE_API_KEY not set)")

        # Initialize webhook manager
        from ..webhooks import WebhookManager
        from ..webhooks.models import WebhookConfig

        webhook_global: list[WebhookConfig] = []
        if config.webhook_url:
            webhook_global = [
                WebhookConfig(
                    id="",  # No DB row for env-configured global webhook
                    bank_id=None,
                    url=config.webhook_url,
                    secret=config.webhook_secret,
                    event_types=config.webhook_event_types,
                    enabled=True,
                )
            ]
        self._webhook_manager = WebhookManager(
            backend=self._backend,
            global_webhooks=webhook_global,
            tenant_extension=self._tenant_extension,
        )
        logger.debug("Webhook manager initialized")

        # Long-lived HTTP client for webhook delivery tasks
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Set executor for task backend and initialize
        self._task_backend.set_executor(self.execute_task)
        await self._task_backend.initialize()

        # Start audit log retention sweep (if configured)
        self._audit_logger.start_retention_sweep()

        self._initialized = True
        logger.info("Memory system initialized (pool and task backend started)")

    async def _get_pool(self) -> asyncpg.Pool:
        """Get the connection pool (must call initialize() first)."""
        if not self._initialized:
            await self.initialize()
        return self._pool

    async def _get_read_backend(self) -> DatabaseBackend:
        """Get the read-only backend (replica when configured, otherwise primary).

        Writes MUST NOT be issued through this backend.
        """
        if not self._initialized:
            await self.initialize()
        return self._read_backend

    async def _get_backend(self) -> DatabaseBackend:
        """Get the database backend, auto-initializing if needed."""
        if not self._initialized:
            await self.initialize()
        return self._backend

    async def _acquire_connection(self):
        """
        Acquire a connection from the database backend.

        Yields a DatabaseConnection from the backend's connection pool.
        """
        backend = await self._get_backend()
        async with backend.acquire() as conn:
            yield conn

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
            backend = await self._get_backend()
            async with backend.acquire() as conn:
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

        # Stop audit log retention sweep
        await self._audit_logger.stop_retention_sweep()

        # Shutdown task backend
        await self._task_backend.shutdown()

        # Close HTTP client used for webhook delivery
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

        if self._read_backend is not None and self._read_backend is not self._backend:
            await self._read_backend.shutdown()
        self._read_backend = None

        # Close primary database backend (shuts down pool)
        if self._backend is not None:
            await self._backend.shutdown()
            self._backend = None
            self._pool = None

        self._initialized = False

        # Clean up LLM providers (e.g. stop llamacpp subprocess)
        for llm_config in (
            self._llm_config,
            self._retain_llm_config,
            self._reflect_llm_config,
            self._consolidation_llm_config,
        ):
            try:
                await llm_config.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up LLM provider: {e}")

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
            fact_type_override: Override fact type ('world', 'experience')
            request_context: Request context for authentication.

        Returns:
            List of created unit IDs
        """
        # Build content dict
        content_dict: RetainContentDict = {"content": content, "context": context}
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
        document_tags: list[str] | None = None,
        return_usage: bool = False,
        operation_id: str | None = None,
        outbox_callback: "Callable[[asyncpg.Connection], Awaitable[None]] | None" = None,
        strategy: str | None = None,
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
            fact_type_override: Override fact type for all facts ('world', 'experience')
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
            )
            result = await self._validate_operation(self._operation_validator.validate_retain(ctx))
            if result and result.contents is not None:
                contents = result.contents

        # Engine-owned copy: the orchestrator clears per-item "content" strings
        # after building the document's combined text (memory pressure
        # optimization, see retain/orchestrator.py). Without an internal copy
        # those mutations leak back to the caller's dicts.
        contents = cast(list[RetainContentDict], [dict(c) for c in contents])

        # Apply batch-level document_id to contents that don't have their own (backwards compatibility)
        if document_id:
            for item in contents:
                if "document_id" not in item:
                    item["document_id"] = document_id

        # Validate no duplicate document_ids in the batch
        # Having duplicate document_ids causes race conditions in document upserts during parallel processing
        doc_ids = [item.get("document_id") for item in contents if item.get("document_id")]
        if len(doc_ids) != len(set(doc_ids)):
            from collections import Counter

            duplicates = [doc_id for doc_id, count in Counter(doc_ids).items() if count > 1]
            raise ValueError(
                f"Batch contains duplicate document_ids: {duplicates}. "
                f"Each content item in a batch must have a unique document_id to avoid race conditions."
            )

        # Validate update_mode=append requires document_id
        for item in contents:
            if item.get("update_mode") == "append" and not item.get("document_id"):
                raise ValueError("update_mode='append' requires a document_id")

        # Auto-chunk large batches by token count to avoid timeouts and memory issues
        # Calculate total token count
        total_tokens = sum(count_tokens(item.get("content", "")) for item in contents)
        total_usage = TokenUsage()
        # Aggregate "content tokens that actually went through extraction after
        # chunk-level dedup" across sub-batches. ``None`` in any sub-batch
        # means that sub-batch bypassed dedup, so the aggregate is None
        # (see RetainResult.processed_content_tokens).
        total_processed_content_tokens: int | None = 0

        # Get batch size threshold from config
        config = get_config()
        tokens_per_batch = config.retain_batch_tokens

        if total_tokens > tokens_per_batch:
            # Split into smaller batches based on token count
            logger.info(
                f"Large batch detected ({total_tokens:,} tokens from {len(contents)} items). Splitting into sub-batches of ~{tokens_per_batch:,} tokens each..."
            )

            sub_batches = []
            current_batch = []
            current_batch_tokens = 0

            for item in contents:
                item_tokens = count_tokens(item.get("content", ""))

                # If adding this item would exceed the limit, start a new batch
                # (unless current batch is empty - then we must include it even if it's large)
                if current_batch and current_batch_tokens + item_tokens > tokens_per_batch:
                    sub_batches.append(current_batch)
                    current_batch = [item]
                    current_batch_tokens = item_tokens
                else:
                    current_batch.append(item)
                    current_batch_tokens += item_tokens

            # Add the last batch
            if current_batch:
                sub_batches.append(current_batch)

            logger.info(f"Split into {len(sub_batches)} sub-batches: {[len(b) for b in sub_batches]} items each")

            # Process each sub-batch
            all_results = []
            for i, sub_batch in enumerate(sub_batches, 1):
                # Checkpoint: abort if the operation was deleted (bank was deleted) between sub-batches.
                if operation_id and not await self._check_op_alive(operation_id):
                    logger.info(
                        f"[BATCH_RETAIN] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping after {i - 1}/{len(sub_batches)} sub-batches"
                    )
                    if return_usage:
                        return all_results, total_usage
                    return all_results

                sub_batch_tokens = sum(count_tokens(item.get("content", "")) for item in sub_batch)
                logger.info(
                    f"Processing sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_tokens:,} tokens"
                )

                sub_results, sub_usage, sub_processed = await self._retain_batch_async_internal(
                    bank_id=bank_id,
                    contents=sub_batch,
                    request_context=request_context,
                    document_id=document_id,
                    is_first_batch=i == 1,  # Only upsert on first batch
                    fact_type_override=fact_type_override,
                    document_tags=document_tags,
                    operation_id=operation_id,
                    strategy=strategy,
                    # Outbox callback runs inside the last sub-batch's transaction so the
                    # webhook delivery row is committed atomically with the final retain data.
                    outbox_callback=outbox_callback if i == len(sub_batches) else None,
                )
                all_results.extend(sub_results)
                total_usage = total_usage + sub_usage
                if total_processed_content_tokens is None or sub_processed is None:
                    total_processed_content_tokens = None
                else:
                    total_processed_content_tokens = total_processed_content_tokens + sub_processed

            total_time = time.time() - start_time
            logger.info(
                f"RETAIN_BATCH_ASYNC (chunked) COMPLETE: {len(all_results)} results from {len(contents)} contents in {total_time:.3f}s"
            )
            result = all_results
        else:
            # Small batch - use internal method directly
            result, total_usage, total_processed_content_tokens = await self._retain_batch_async_internal(
                bank_id=bank_id,
                contents=contents,
                request_context=request_context,
                document_id=document_id,
                is_first_batch=True,
                fact_type_override=fact_type_override,
                document_tags=document_tags,
                operation_id=operation_id,
                strategy=strategy,
                outbox_callback=outbox_callback,
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
                unit_ids=result,
                success=True,
                error=None,
                llm_input_tokens=total_usage.input_tokens,
                llm_output_tokens=total_usage.output_tokens,
                llm_total_tokens=total_usage.total_tokens,
                processed_content_tokens=total_processed_content_tokens,
            )
            try:
                await self._operation_validator.on_retain_complete(result_ctx)
            except Exception as e:
                logger.warning(f"Post-retain hook error (non-fatal): {e}")

        # Trigger consolidation as a tracked async operation if enabled
        # Resolve bank-specific config to check if observations are enabled for this bank
        config = await self._config_resolver.resolve_full_config(bank_id, request_context)
        if config.enable_observations:
            try:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                # Log but don't fail the retain - consolidation is non-critical
                logger.warning(f"Failed to submit consolidation task for bank {bank_id}: {e}")

        if return_usage:
            return result, total_usage
        return result

    async def _retain_batch_async_internal(
        self,
        bank_id: str,
        contents: list[RetainContentDict],
        request_context: "RequestContext",
        document_id: str | None = None,
        is_first_batch: bool = True,
        fact_type_override: str | None = None,
        document_tags: list[str] | None = None,
        operation_id: str | None = None,
        outbox_callback: "Callable[[asyncpg.Connection], Awaitable[None]] | None" = None,
        strategy: str | None = None,
    ) -> tuple[list[list[str]], "TokenUsage", int | None]:
        """
        Internal method for batch processing without chunking logic.

        Assumes contents are already appropriately sized (< 50k chars).
        Called by retain_batch_async after chunking large batches.

        Uses semaphore for backpressure to limit concurrent retains.

        Args:
            bank_id: Unique identifier for the bank
            contents: List of dicts with content, context, event_date
            request_context: Request context for config resolution
            document_id: Optional document ID (always upserts if exists)
            is_first_batch: Whether this is the first batch (for chunked operations, only delete on first batch)
            fact_type_override: Override fact type for all facts
            document_tags: Tags applied to all items in this batch

        Returns:
            Tuple of (unit ID lists, LLM token usage, processed_content_tokens).
            See ``RetainResult.processed_content_tokens`` for the semantics of
            the third element.
        """
        # Use the new modular orchestrator
        from .retain import orchestrator

        backend = await self._get_backend()

        # Resolve bank-specific config for this operation
        resolved_config = await self._config_resolver.resolve_full_config(bank_id, request_context)

        # Force chunks mode when LLM provider is "none" (no LLM available for fact extraction)
        if self._llm_config.provider == "none":
            resolved_config.retain_extraction_mode = "chunks"
            resolved_config.enable_observations = False

        # Apply strategy overrides: explicit strategy > bank default strategy
        from hindsight_api.config_resolver import apply_strategy

        effective_strategy = strategy or resolved_config.retain_default_strategy
        if effective_strategy:
            resolved_config = apply_strategy(resolved_config, effective_strategy)

        # Create parent span for retain operation
        with create_operation_span("retain", bank_id):
            return await orchestrator.retain_batch(
                pool=self._backend,
                embeddings_model=self.embeddings,
                llm_config=self._retain_llm_config.with_config(resolved_config),
                entity_resolver=self.entity_resolver,
                format_date_fn=self._format_readable_date,
                bank_id=bank_id,
                contents_dicts=contents,
                document_id=document_id,
                is_first_batch=is_first_batch,
                fact_type_override=fact_type_override,
                document_tags=document_tags,
                config=resolved_config,
                operation_id=operation_id,
                schema=_current_schema.get(),
                outbox_callback=outbox_callback,
                db_semaphore=self._put_semaphore,
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
            fact_type: Required filter for fact type ('world' or 'experience')
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
        include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096,
        max_source_facts_tokens_per_observation: int = -1,
        request_context: "RequestContext",
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        _connection_budget: int | None = None,
        _quiet: bool = False,
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
                             NOTE: Chunks are fetched independently of max_tokens filtering.
                             This means setting max_tokens=0 will return 0 facts but can still
                             return chunks from the top-scored (reranked) results.
                             Chunks are fetched in batches (estimated as (max_chunk_tokens // retain_chunk_size) * 2)
                             until the token budget is exhausted or all chunks are fetched.
                             This handles varying chunk sizes across documents.
            tags: Optional list of tags for visibility filtering (OR matching - returns
                  memories that have at least one matching tag)

        Returns:
            RecallResultModel containing:
            - results: List of MemoryFact objects (filtered by max_tokens)
            - trace: Optional trace information for debugging
            - entities: Optional dict of entity states (if include_entities=True)
            - chunks: Optional dict of chunks (if include_chunks=True, independent of max_tokens)
        """
        # Authenticate tenant and set schema in context (for fq_table())
        await self._authenticate_tenant(request_context)

        # Default to all fact types if not specified
        if fact_type is None:
            fact_type = list(VALID_RECALL_FACT_TYPES)

        # Filter out 'opinion' (removed fact type, silently ignore for backwards compat)
        fact_type = [ft for ft in fact_type if ft != "opinion"]
        if not fact_type:
            return RecallResultModel(results=[], entities={}, chunks={})

        # Validate fact types
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
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
            )
            result = await self._validate_operation(self._operation_validator.validate_recall(ctx))
            if result:
                if result.tags is not None:
                    tags = result.tags
                if result.tags_match is not None:
                    tags_match = result.tags_match
                if result.tag_groups is not None:
                    tag_groups = result.tag_groups

        # Map budget enum to thinking_budget number using bank-resolved config.
        # Function "fixed" preserves legacy {LOW: 100, MID: 300, HIGH: 1000}; function "adaptive"
        # derives from max_tokens and clamps to [recall_budget_min, recall_budget_max].
        budget_config_dict = await self._config_resolver.get_bank_config(bank_id, request_context)
        thinking_budget = _resolve_thinking_budget(budget_config_dict, budget, max_tokens)

        # Log recall start with tags if present (skip if quiet mode for internal operations)
        if not _quiet:
            tags_info = f", tags={tags} ({tags_match})" if tags else ""
            logger.info(f"[RECALL {bank_id[:8]}] Starting recall for query: {query[:50]}...{tags_info}")

        # Create parent span for recall operation
        from ..tracing import get_tracer

        tracer = get_tracer()
        # Use start_as_current_span to ensure child spans are linked properly
        recall_span_context = tracer.start_as_current_span("hindsight.recall")
        recall_span = recall_span_context.__enter__()
        recall_span.set_attribute("hindsight.bank_id", bank_id)
        recall_span.set_attribute("hindsight.query", query[:100])
        recall_span.set_attribute("hindsight.fact_types", ",".join(fact_type))
        recall_span.set_attribute("hindsight.thinking_budget", thinking_budget)
        recall_span.set_attribute("hindsight.max_tokens", max_tokens)

        try:
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
                            tag_groups=tag_groups,
                            created_after=created_after,
                            created_before=created_before,
                            connection_budget=_connection_budget,
                            quiet=_quiet,
                            include_source_facts=include_source_facts,
                            max_source_facts_tokens=max_source_facts_tokens,
                            max_source_facts_tokens_per_observation=max_source_facts_tokens_per_observation,
                        )
                        break  # Success - exit retry loop
                    except Exception as e:
                        # Check if it's a connection error (PG or Oracle)
                        is_connection_error = (
                            isinstance(e, asyncpg.TooManyConnectionsError)
                            or isinstance(e, asyncpg.CannotConnectNowError)
                            or (isinstance(e, asyncpg.PostgresError) and "connection" in str(e).lower())
                            or _is_oracledb_connection_error(e)
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
        finally:
            recall_span_context.__exit__(None, None, None)

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
        tag_groups: list[TagGroup] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        connection_budget: int | None = None,
        quiet: bool = False,
        include_source_facts: bool = False,
        max_source_facts_tokens: int = 4096,
        max_source_facts_tokens_per_observation: int = -1,
    ) -> RecallResultModel:
        """
        Search implementation with modular retrieval and reranking.

        Architecture:
        1. Retrieval: 4-way parallel (semantic, keyword, graph, temporal graph)
        2. Merge: RRF to combine ranked lists
        3. Reranking: Pluggable strategy (heuristic or cross-encoder)
        4. Diversity: MMR with λ=0.5
        5. Chunks: Fetch chunks from top-scored results (BEFORE token filtering)
        6. Token Filter: Limit facts to max_tokens budget

        Args:
            bank_id: bank IDentifier
            query: Search query
            fact_type: Type of facts to search
            thinking_budget: Nodes to explore in graph traversal
            max_tokens: Maximum tokens to return (counts only 'text' field)
            enable_trace: Whether to return search trace (deprecated)
            include_entities: Whether to include entity observations
            max_entity_tokens: Maximum tokens for entity observations
            include_chunks: Whether to include raw chunks (fetched before max_tokens filtering)
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

        backend = await self._get_read_backend()
        recall_start = time.time()

        # Buffer logs for clean output in concurrent scenarios.
        # Include a uuid suffix so two recalls on the same bank within the
        # same millisecond don't collide on the budgeted_operation key
        # (`recall-{recall_id}`), which would raise "Operation ... already exists".
        recall_id = f"{bank_id[:8]}-{int(time.time() * 1000) % 100000}-{uuid.uuid4().hex[:6]}"
        log_buffer = []
        tags_info = f", tags={tags}, tags_match={tags_match}" if tags else ""
        log_buffer.append(
            f"[RECALL {recall_id}] Query: '{query[:50]}...' (budget={thinking_budget}, max_tokens={max_tokens}{tags_info})"
        )

        # Import tracing utilities
        from ..tracing import get_tracer

        tracer_otel = get_tracer()

        try:
            # Step 1: Generate query embedding (for semantic search)
            step_start = time.time()

            embedding_span = tracer_otel.start_span("hindsight.recall_embedding")
            embedding_span.set_attribute("hindsight.bank_id", bank_id)
            embedding_span.set_attribute("hindsight.query", query[:100])

            try:
                query_embeddings = await embedding_utils.generate_embeddings_batch(self.embeddings, [query])
                query_embedding = query_embeddings[0]
                step_duration = time.time() - step_start
                log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")
            finally:
                embedding_span.end()

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

            retrieval_span = tracer_otel.start_span("hindsight.recall_retrieval")
            retrieval_span.set_attribute("hindsight.bank_id", bank_id)
            retrieval_span.set_attribute("hindsight.fact_types", ",".join(fact_type))
            retrieval_span.set_attribute("hindsight.thinking_budget", thinking_budget)

            try:
                # Run optimized retrieval with connection budget
                config = get_config()
                effective_connection_budget = (
                    connection_budget if connection_budget is not None else config.recall_connection_budget
                )
                async with budgeted_operation(
                    max_connections=effective_connection_budget,
                    operation_id=f"recall-{recall_id}",
                ) as op:
                    budgeted_pool = op.wrap_pool(backend)
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
                        tag_groups=tag_groups,
                        created_after=created_after,
                        created_before=created_before,
                    )
                    parallel_duration = time.time() - parallel_start
            finally:
                retrieval_span.end()

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
            all_graph_timings = []

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
            if all_graph_timings:
                retriever_name = get_default_graph_retriever().name.upper()
                graph_total = all_graph_timings[0]  # Take first fact type's timing as representative
                graph_parts = [
                    f"db_queries={graph_total.db_queries}",
                    f"edge_load={graph_total.edge_load_time:.3f}s",
                    f"edges={graph_total.edge_count}",
                    f"patterns={graph_total.pattern_count}",
                ]
                if graph_total.seeds_time > 0.01:
                    graph_parts.append(f"seeds={graph_total.seeds_time:.3f}s")
                if graph_total.fusion > 0.001:
                    graph_parts.append(f"fusion={graph_total.fusion:.3f}s")
                if graph_total.fetch > 0.001:
                    graph_parts.append(f"fetch={graph_total.fetch:.3f}s")
                log_buffer.append(f"      [{retriever_name}] {', '.join(graph_parts)}")
                # Log detailed hop timing for debugging slow queries
                if graph_total.hop_details:
                    for hd in graph_total.hop_details:
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
                # Also expose each retrieval method as its own phase so
                # benchmarks can pinpoint which sub-query drives latency.
                for _method, _dur in aggregated_timings.items():
                    if _dur > 0:
                        tracer.add_phase_metric(f"retrieval_{_method}", _dur)

            # Step 3: Merge with RRF
            step_start = time.time()
            from .search.fusion import reciprocal_rank_fusion

            fusion_span = tracer_otel.start_span("hindsight.recall_fusion")
            fusion_span.set_attribute("hindsight.bank_id", bank_id)
            fusion_span.set_attribute("hindsight.semantic_count", len(semantic_results))
            fusion_span.set_attribute("hindsight.bm25_count", len(bm25_results))
            fusion_span.set_attribute("hindsight.graph_count", len(graph_results))
            fusion_span.set_attribute("hindsight.temporal_count", len(temporal_results) if temporal_results else 0)

            try:
                # Merge 3 or 4 result lists depending on temporal constraint
                if temporal_results:
                    merged_candidates = reciprocal_rank_fusion(
                        [semantic_results, bm25_results, graph_results, temporal_results]
                    )
                else:
                    merged_candidates = reciprocal_rank_fusion([semantic_results, bm25_results, graph_results])

                step_duration = time.time() - step_start
                log_buffer.append(
                    f"  [3] RRF merge: {len(merged_candidates)} unique candidates in {step_duration:.3f}s"
                )
            finally:
                fusion_span.set_attribute("hindsight.merged_count", len(merged_candidates))
                fusion_span.end()

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

            rerank_span = tracer_otel.start_span("hindsight.recall_rerank")
            rerank_span.set_attribute("hindsight.bank_id", bank_id)
            rerank_span.set_attribute("hindsight.candidates_count", len(merged_candidates))

            scored_results: list = []
            pre_filtered_count = 0
            try:
                # Ensure reranker is initialized (for lazy initialization mode)
                await reranker_instance.ensure_initialized()

                # Pre-filter candidates to reduce reranking cost (RRF already provides good ranking)
                # This is especially important for remote rerankers with network latency
                reranker_max_candidates = get_config().reranker_max_candidates
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
            finally:
                rerank_span.set_attribute("hindsight.scored_count", len(scored_results))
                if pre_filtered_count > 0:
                    rerank_span.set_attribute("hindsight.pre_filtered_count", pre_filtered_count)
                rerank_span.end()

            # Step 4.5: Combine cross-encoder score with retrieval signals via multiplicative boosts.
            # See apply_combined_scoring for the full rationale and formula.
            # is_passthrough_reranker tells the scoring code to seed CE scores
            # from RRF rank — only meaningful when the configured reranker is
            # the slim/passthrough one that returns a constant score per pair.
            if scored_results:
                ce = reranker_instance.cross_encoder
                is_passthrough = ce is not None and ce.provider_name == "rrf"
                apply_combined_scoring(scored_results, now=utcnow(), is_passthrough_reranker=is_passthrough)
                scored_results.sort(key=lambda x: x.weight, reverse=True)
                log_buffer.append("  [4.6] Combined scoring: ce * recency_boost(0.2) * temporal_boost(0.2)")

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

            # Step 5.5: Fetch chunks from top-scored results (before token filtering)
            # Chunks are fetched independently of max_tokens filtering
            chunks_dict = None
            total_chunk_tokens = 0
            if include_chunks and top_scored:
                from .response_models import ChunkInfo

                # Collect chunk_ids in order of fact relevance (preserving order from top_scored).
                # Observations have no direct chunk_id — use a placeholder so their source
                # chunks end up at the observation's rank position, not appended at the end.
                # ordered_items: list of ('chunk', chunk_id) | ('obs', sr.id)
                ordered_items: list[tuple[str, str]] = []
                seen_chunk_ids: set[str] = set()
                observation_ids_ordered: list[uuid.UUID] = []
                for sr in top_scored:
                    chunk_id = sr.retrieval.chunk_id
                    if chunk_id and chunk_id not in seen_chunk_ids:
                        ordered_items.append(("chunk", chunk_id))
                        seen_chunk_ids.add(chunk_id)
                    elif not chunk_id and sr.retrieval.fact_type == "observation":
                        ordered_items.append(("obs", sr.id))
                        observation_ids_ordered.append(uuid.UUID(sr.id))

                # Resolve source chunk_ids for all observations in a single query,
                # ordered by observation rank so per-observation results stay grouped correctly.
                obs_chunk_ids: dict[str, list[str]] = {}
                if observation_ids_ordered:
                    async with acquire_with_retry(backend) as obs_conn:
                        if self._backend.ops.uses_observation_sources_table:
                            obs_source_rows = await obs_conn.fetch(
                                f"""
                                SELECT os.observation_id AS obs_id, mu.chunk_id
                                FROM {fq_table("observation_sources")} os
                                JOIN {fq_table("memory_units")} mu
                                  ON mu.id = os.source_id
                                WHERE os.observation_id = ANY($1::uuid[])
                                  AND mu.chunk_id IS NOT NULL
                                ORDER BY array_position($1::uuid[], os.observation_id)
                                """,
                                observation_ids_ordered,
                            )
                        else:
                            obs_source_rows = await obs_conn.fetch(
                                f"""
                                SELECT obs.id AS obs_id, mu.chunk_id
                                FROM {fq_table("memory_units")} obs
                                JOIN {fq_table("memory_units")} mu
                                  ON mu.id = ANY(obs.source_memory_ids)
                                WHERE obs.id = ANY($1::uuid[])
                                  AND mu.chunk_id IS NOT NULL
                                ORDER BY array_position($1::uuid[], obs.id)
                                """,
                                observation_ids_ordered,
                            )
                    for row in obs_source_rows:
                        obs_id = str(row["obs_id"])
                        cid = row["chunk_id"]
                        if cid not in seen_chunk_ids:
                            obs_chunk_ids.setdefault(obs_id, []).append(cid)
                            seen_chunk_ids.add(cid)

                # Flatten ordered_items into chunk_ids_ordered, expanding obs placeholders
                chunk_ids_ordered = []
                for item_type, item_id in ordered_items:
                    if item_type == "chunk":
                        chunk_ids_ordered.append(item_id)
                    else:
                        chunk_ids_ordered.extend(obs_chunk_ids.get(item_id, []))

                if chunk_ids_ordered:
                    chunks_dict = {}
                    encoding = _get_tiktoken_encoding()

                    # Fetch all candidate chunks in a single query. Token-budget accounting
                    # happens in Python after the fetch — one round-trip is always faster
                    # than multiple batched round-trips when the candidate set is large.
                    async with acquire_with_retry(backend) as conn:
                        chunks_rows = await conn.fetch(
                            f"""
                            SELECT chunk_id, chunk_text, chunk_index
                            FROM {fq_table("chunks")}
                            WHERE chunk_id = ANY($1::text[])
                            """,
                            chunk_ids_ordered,
                        )

                    chunks_lookup = {row["chunk_id"]: row for row in chunks_rows}

                    # Process chunks in relevance order, respecting token budget
                    for chunk_id in chunk_ids_ordered:
                        if chunk_id not in chunks_lookup:
                            continue

                        row = chunks_lookup[chunk_id]
                        chunk_text = row["chunk_text"]
                        chunk_tokens = len(encoding.encode(chunk_text))

                        if total_chunk_tokens + chunk_tokens > max_chunk_tokens:
                            remaining_tokens = max_chunk_tokens - total_chunk_tokens
                            if remaining_tokens > 0:
                                truncated_text = encoding.decode(encoding.encode(chunk_text)[:remaining_tokens])
                                chunks_dict[chunk_id] = ChunkInfo(
                                    chunk_text=truncated_text, chunk_index=row["chunk_index"], truncated=True
                                )
                                total_chunk_tokens = max_chunk_tokens
                            break
                        else:
                            chunks_dict[chunk_id] = ChunkInfo(
                                chunk_text=chunk_text, chunk_index=row["chunk_index"], truncated=False
                            )
                            total_chunk_tokens += chunk_tokens

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

            # Fetch source facts for observation-type results (mirrors chunks pattern)
            source_fact_ids_by_obs: dict[str, list[str]] = {}  # obs_id -> [source_id, ...]
            source_facts_dict: dict[str, MemoryFact] | None = None
            if include_source_facts:
                observation_ids = [uuid.UUID(sr.id) for sr in top_scored if sr.retrieval.fact_type == "observation"]
                if observation_ids:
                    async with acquire_with_retry(backend) as sf_conn:
                        # Fetch source_memory_ids for all observation results
                        obs_rows = await sf_conn.fetch(
                            f"""
                            SELECT id, source_memory_ids
                            FROM {fq_table("memory_units")}
                            WHERE id = ANY($1::uuid[]) AND fact_type = 'observation'
                            """,
                            observation_ids,
                        )

                        # Collect unique source IDs in order of first appearance
                        seen_source_ids: set[str] = set()
                        source_ids_ordered: list[str] = []
                        for obs_row in obs_rows:
                            obs_id = str(obs_row["id"])
                            sids = [str(s) for s in (obs_row["source_memory_ids"] or [])]
                            source_fact_ids_by_obs[obs_id] = sids
                            for sid in sids:
                                if sid not in seen_source_ids:
                                    source_ids_ordered.append(sid)
                                    seen_source_ids.add(sid)

                        # Fetch source fact content up to token budget
                        if source_ids_ordered:
                            import uuid as uuid_module

                            source_rows = await sf_conn.fetch(
                                f"""
                                SELECT id, text, fact_type, context, occurred_start, occurred_end,
                                       mentioned_at, document_id, chunk_id, tags, metadata
                                FROM {fq_table("memory_units")}
                                WHERE id = ANY($1::uuid[])
                                """,
                                [uuid_module.UUID(sid) for sid in source_ids_ordered],
                            )
                            source_row_by_id = {str(r["id"]): r for r in source_rows}

                            encoding = _get_tiktoken_encoding()
                            source_facts_dict = {}

                            def _make_source_fact(sid: str, r: Any) -> MemoryFact:
                                return MemoryFact(
                                    id=sid,
                                    text=r["text"],
                                    fact_type=r["fact_type"],
                                    context=r["context"],
                                    occurred_start=r["occurred_start"].isoformat() if r["occurred_start"] else None,
                                    occurred_end=r["occurred_end"].isoformat() if r["occurred_end"] else None,
                                    mentioned_at=r["mentioned_at"].isoformat() if r["mentioned_at"] else None,
                                    document_id=r["document_id"],
                                    metadata=r["metadata"],
                                    chunk_id=str(r["chunk_id"]) if r["chunk_id"] else None,
                                    tags=r["tags"] or None,
                                )

                            if max_source_facts_tokens_per_observation >= 0:
                                # Per-observation capping: each observation independently selects
                                # source facts up to its token budget.
                                for obs_id, sids in source_fact_ids_by_obs.items():
                                    obs_tokens = 0
                                    for sid in sids:
                                        if sid not in source_row_by_id:
                                            continue
                                        r = source_row_by_id[sid]
                                        fact_tokens = len(encoding.encode(r["text"]))
                                        if obs_tokens + fact_tokens > max_source_facts_tokens_per_observation:
                                            break
                                        obs_tokens += fact_tokens
                                        if sid not in source_facts_dict:
                                            source_facts_dict[sid] = _make_source_fact(sid, r)
                            else:
                                # Global budget: fill in order of first appearance until exhausted.
                                total_source_tokens = 0
                                for sid in source_ids_ordered:
                                    if sid not in source_row_by_id:
                                        continue
                                    r = source_row_by_id[sid]
                                    fact_tokens = len(encoding.encode(r["text"]))
                                    if (
                                        max_source_facts_tokens >= 0
                                        and total_source_tokens + fact_tokens > max_source_facts_tokens
                                    ):
                                        break
                                    source_facts_dict[sid] = _make_source_fact(sid, r)
                                    total_source_tokens += fact_tokens

            # Get entities for each fact if include_entities is requested.
            # _entity_rows_for_units_sql resolves both direct unit_entities rows
            # and observation-via-source-memory inheritance in a single query.
            fact_entity_map = {}  # unit_id -> list of {entity_id, canonical_name}
            if include_entities and top_scored:
                unit_ids = [uuid.UUID(sr.id) for sr in top_scored]
                if unit_ids:
                    async with acquire_with_retry(backend) as entity_conn:
                        entity_rows = await entity_conn.fetch(
                            self._entity_rows_for_units_sql(unit_ids_placeholder=1),
                            unit_ids,
                        )
                        for row in entity_rows:
                            unit_id = str(row["unit_id"])
                            fact_entity_map.setdefault(unit_id, []).append(
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
                        metadata=result_dict.get("metadata"),
                        chunk_id=result_dict.get("chunk_id"),
                        tags=result_dict.get("tags"),
                        source_fact_ids=source_fact_ids_by_obs.get(result_id) if include_source_facts else None,
                    )
                )

            # Fetch entity observations if requested
            entities_dict = None
            total_entity_tokens = 0
            if include_entities and fact_entity_map:
                # Collect unique entities in order of fact relevance (preserving order from top_scored)
                entities_ordered = []  # list of (entity_id, entity_name) tuples
                seen_entity_ids = set()

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
            if not quiet:
                logger.info("\n" + "\n".join(log_buffer))

            return RecallResultModel(
                results=memory_facts,
                trace=trace_dict,
                entities=entities_dict,
                chunks=chunks_dict,
                source_facts=source_facts_dict,
            )

        except Exception as e:
            # Use repr(e) so exceptions with empty __str__ (e.g. raise SomeError())
            # still emit a discriminating class+args string into operations.error_message.
            log_buffer.append(
                f"[RECALL {recall_id}] ERROR after {time.time() - recall_start:.3f}s: {type(e).__name__}: {e!r}"
            )
            if not quiet:
                logger.error("\n" + "\n".join(log_buffer), exc_info=True)
            raise RuntimeError(f"Failed to search memories ({type(e).__name__}): {e!r}") from e

    def _entity_rows_for_units_sql(self, unit_ids_placeholder: int) -> str:
        """SQL SELECT producing ``(unit_id, entity_id, canonical_name)`` rows for
        the given unit IDs.

        Direct rows come from ``unit_entities``. Observations rarely carry
        direct rows there; their entity association lives transitively through
        their source memories (``source_memory_ids`` on PG, the
        ``observation_sources`` junction on Oracle). When an observation has
        no direct entity rows the SELECT inherits its source memories'
        entities, so the result is the same set callers would get from
        ``get_memory_unit``.

        ``unit_ids_placeholder`` is the 1-based parameter index that holds the
        ``uuid[]`` of unit IDs. The placeholder is referenced twice — both
        sides of the UNION need it — so callers should not reuse the slot.
        """
        ue = fq_table("unit_entities")
        ents = fq_table("entities")
        mu = fq_table("memory_units")
        p = unit_ids_placeholder

        direct = (
            f"SELECT ue.unit_id, e.id AS entity_id, e.canonical_name "
            f"FROM {ue} ue "
            f"JOIN {ents} e ON e.id = ue.entity_id "
            f"WHERE ue.unit_id = ANY(${p}::uuid[])"
        )

        if self._backend.ops.uses_observation_sources_table:
            os_t = fq_table("observation_sources")
            inherited = (
                f"SELECT os.observation_id AS unit_id, e.id AS entity_id, e.canonical_name "
                f"FROM {os_t} os "
                f"JOIN {ue} src_ue ON src_ue.unit_id = os.source_id "
                f"JOIN {ents} e ON e.id = src_ue.entity_id "
                f"WHERE os.observation_id = ANY(${p}::uuid[]) "
                f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = os.observation_id)"
            )
        else:
            inherited = (
                f"SELECT obs.id AS unit_id, e.id AS entity_id, e.canonical_name "
                f"FROM {mu} obs "
                f"CROSS JOIN LATERAL unnest(obs.source_memory_ids) AS src_id "
                f"JOIN {ue} src_ue ON src_ue.unit_id = src_id "
                f"JOIN {ents} e ON e.id = src_ue.entity_id "
                f"WHERE obs.id = ANY(${p}::uuid[]) "
                f"AND obs.fact_type = 'observation' "
                f"AND obs.source_memory_ids IS NOT NULL "
                f"AND NOT EXISTS (SELECT 1 FROM {ue} d WHERE d.unit_id = obs.id)"
            )

        return f"({direct}) UNION ({inherited})"

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

    def _observations_via_source_match_sql(
        self,
        source_column: str,
        source_placeholder: int,
        bank_placeholder: int | None,
    ) -> str:
        """SQL predicate matching `memory_units` rows that are observations
        whose source memories satisfy ``<source_column> = $source_placeholder``.

        Observations have no `document_id` / `chunk_id` of their own; the link
        to a source row lives in `source_memory_ids` (PG) or the
        `observation_sources` junction (Oracle).
        """
        if source_column not in ("document_id", "chunk_id"):
            raise ValueError(f"Unsupported source_column: {source_column!r}")
        if self._backend.ops.uses_observation_sources_table:
            bank_clause = f" AND src.bank_id = ${bank_placeholder}" if bank_placeholder else ""
            return (
                f"id IN (SELECT os.observation_id "
                f"FROM {fq_table('observation_sources')} os "
                f"JOIN {fq_table('memory_units')} src ON src.id = os.source_id "
                f"WHERE src.{source_column} = ${source_placeholder}{bank_clause})"
            )
        bank_clause = f" AND bank_id = ${bank_placeholder}" if bank_placeholder else ""
        return (
            f"source_memory_ids && (SELECT array_agg(id) "
            f"FROM {fq_table('memory_units')} "
            f"WHERE {source_column} = ${source_placeholder}{bank_clause})"
        )

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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_document", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            obs_match = self._observations_via_source_match_sql("document_id", source_placeholder=1, bank_placeholder=2)
            observation_count_sql = (
                f"(SELECT COUNT(*) FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $2 AND fact_type = 'observation' AND {obs_match})"
            )

            # Use a subquery for counts to avoid GROUP BY on CLOB columns
            # (Oracle cannot use CLOB types as comparison keys in GROUP BY).
            doc = await conn.fetchrow(
                f"""
                SELECT d.id, d.bank_id, d.original_text, d.content_hash,
                       d.created_at, d.updated_at, d.tags, d.retain_params,
                       COALESCE(stats.unit_count, 0) as unit_count,
                       COALESCE(stats.world_count, 0) as world_count,
                       COALESCE(stats.experience_count, 0) as experience_count,
                       COALESCE({observation_count_sql}, 0) as observation_count
                FROM {fq_table("documents")} d
                LEFT JOIN (
                    SELECT mu.document_id, mu.bank_id,
                           COUNT(mu.id) as unit_count,
                           COUNT(CASE WHEN mu.fact_type = 'world' THEN 1 END) as world_count,
                           COUNT(CASE WHEN mu.fact_type = 'experience' THEN 1 END) as experience_count
                    FROM {fq_table("memory_units")} mu
                    WHERE mu.document_id = $1 AND mu.bank_id = $2
                    GROUP BY mu.document_id, mu.bank_id
                ) stats ON stats.document_id = d.id AND stats.bank_id = d.bank_id
                WHERE d.id = $1 AND d.bank_id = $2
                """,
                document_id,
                bank_id,
            )

            if not doc:
                return None

            retain_params_parsed = conn.parse_json(doc["retain_params"])

            # document_metadata is sourced from retain_params.metadata
            document_metadata = retain_params_parsed.get("metadata") if retain_params_parsed else None

            return {
                "id": doc["id"],
                "bank_id": doc["bank_id"],
                "original_text": doc["original_text"],
                "content_hash": doc["content_hash"],
                "memory_unit_count": doc["unit_count"],
                "nodes_by_fact_type": {
                    "world": doc["world_count"],
                    "experience": doc["experience_count"],
                    "observation": doc["observation_count"],
                },
                "created_at": doc["created_at"].isoformat() if doc["created_at"] else None,
                "updated_at": doc["updated_at"].isoformat() if doc["updated_at"] else None,
                "tags": list(doc["tags"]) if doc["tags"] else [],
                "document_metadata": document_metadata or None,
                "retain_params": retain_params_parsed or None,
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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="delete_document", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        invalidated_obs = 0
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                # Get memory unit IDs before deletion (for observation cleanup)
                unit_rows = await conn.fetch(
                    f"SELECT id FROM {fq_table('memory_units')} WHERE document_id = $1 AND fact_type IN ('experience', 'world')",
                    document_id,
                )
                unit_ids = [str(row["id"]) for row in unit_rows]
                units_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE document_id = $1", document_id
                )

                # Delete document first (cascades to memory_units and all their links).
                # Running the stale-observation sweep AFTER the delete ensures we also
                # catch observations inserted concurrently by consolidation — otherwise
                # an insert that commits between the sweep and the delete would leave an
                # orphan referencing the just-deleted source memory.
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING id",
                    document_id,
                    bank_id,
                )

                # Invalidate observations referencing these (now-deleted) memories
                if unit_ids:
                    invalidated_obs = await self._delete_stale_observations_for_memories(conn, bank_id, unit_ids)

                result = {
                    "document_deleted": 1 if deleted else 0,
                    "memory_units_deleted": units_count if deleted else 0,
                }

        if invalidated_obs > 0:
            try:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit consolidation after document deletion for bank {bank_id}: {e}")

        return result

    async def update_document(
        self,
        document_id: str,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        request_context: "RequestContext",
    ) -> bool:
        """
        Update mutable fields on a document without re-processing its content.

        Tag changes propagate to all associated memory units and trigger observation
        invalidation + re-consolidation (same semantics as delete_document):
        - Observations referencing the document's memory units are deleted.
        - The document's own units and any co-source memories from other documents
          have consolidated_at reset so they are re-consolidated under the new tags.

        Args:
            document_id: Document ID to update
            bank_id: Bank ID that owns the document
            tags: New tags to apply to the document and all its memory units (optional)
            request_context: Request context for authentication.

        Returns:
            True if the document was found and updated, False if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_document", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        invalidated_obs = 0
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                set_parts: list[str] = ["updated_at = now()"]
                params: list[Any] = []
                p = 1

                if tags is not None:
                    set_parts.append(f"tags = ${p}")
                    params.append(tags)
                    p += 1

                params.extend([document_id, bank_id])
                doc_id_found = await conn.fetchval(
                    f"""
                    UPDATE {fq_table("documents")}
                    SET {", ".join(set_parts)}
                    WHERE id = ${p} AND bank_id = ${p + 1}
                    RETURNING id
                    """,
                    *params,
                )
                if not doc_id_found:
                    return False

                if tags is not None:
                    unit_rows = await conn.fetch(
                        f"SELECT id FROM {fq_table('memory_units')} WHERE document_id = $1 AND fact_type IN ('experience', 'world')",
                        document_id,
                    )
                    unit_ids = [str(row["id"]) for row in unit_rows]

                    await conn.execute(
                        f"UPDATE {fq_table('memory_units')} SET tags = $1 WHERE document_id = $2",
                        tags,
                        document_id,
                    )

                    if unit_ids:
                        import uuid as uuid_module

                        unit_uuids = [uuid_module.UUID(uid) for uid in unit_ids]
                        unit_uuid_set = {str(u) for u in unit_uuids}
                        if self._backend.ops.uses_observation_sources_table:
                            affected_obs = await conn.fetch(
                                f"""
                                SELECT mu.id, mu.source_memory_ids
                                FROM {fq_table("memory_units")} mu
                                WHERE mu.bank_id = $1
                                  AND mu.fact_type = 'observation'
                                  AND EXISTS (
                                      SELECT 1 FROM {fq_table("observation_sources")} os
                                      WHERE os.observation_id = mu.id
                                        AND os.source_id = ANY($2::uuid[])
                                  )
                                """,
                                bank_id,
                                unit_uuids,
                            )
                        else:
                            affected_obs = await conn.fetch(
                                f"""
                                SELECT id, source_memory_ids
                                FROM {fq_table("memory_units")}
                                WHERE bank_id = $1
                                  AND fact_type = 'observation'
                                  AND source_memory_ids && $2::uuid[]
                                """,
                                bank_id,
                                unit_uuids,
                            )
                        if affected_obs:
                            obs_ids = [obs["id"] for obs in affected_obs]

                            seen: set[str] = set()
                            other_source_uuids: list[uuid_module.UUID] = []
                            for obs in affected_obs:
                                for src_id in obs["source_memory_ids"] or []:
                                    src_str = str(src_id)
                                    if src_str not in unit_uuid_set and src_str not in seen:
                                        other_source_uuids.append(src_id)
                                        seen.add(src_str)

                            await conn.execute(
                                f"DELETE FROM {fq_table('memory_units')} WHERE id = ANY($1::uuid[])",
                                obs_ids,
                            )
                            await conn.execute(
                                f"""
                                UPDATE {fq_table("memory_units")}
                                SET consolidated_at = NULL
                                WHERE id = ANY($1::uuid[])
                                  AND fact_type IN ('experience', 'world')
                                """,
                                unit_uuids,
                            )
                            if other_source_uuids:
                                await conn.execute(
                                    f"""
                                    UPDATE {fq_table("memory_units")}
                                    SET consolidated_at = NULL
                                    WHERE id = ANY($1::uuid[])
                                      AND fact_type IN ('experience', 'world')
                                    """,
                                    other_source_uuids,
                                )
                            invalidated_obs = len(obs_ids)
                            logger.info(
                                f"[OBSERVATIONS] Deleted {invalidated_obs} observations, reset "
                                f"{len(unit_ids)} document source memories and "
                                f"{len(other_source_uuids)} co-source memories for re-consolidation "
                                f"after document update on '{document_id}' in bank {bank_id}"
                            )

        if invalidated_obs > 0:
            try:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit consolidation after document update for bank {bank_id}: {e}")

        return True

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

        Observations referencing this memory are deleted and their other source
        memories are reset for re-consolidation.

        Args:
            unit_id: UUID of the memory unit to delete
            request_context: Request context for authentication.

        Returns:
            Dictionary with deletion result

        Raises:
            ValueError: If unit_id is not a valid UUID
        """
        try:
            unit_uuid = uuid.UUID(unit_id)
        except ValueError:
            raise ValueError(f"Invalid unit_id: '{unit_id}' is not a valid UUID")
        await self._authenticate_tenant(request_context)
        backend = await self._get_backend()
        invalidated_obs = 0
        bank_id_for_consolidation: str | None = None
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                # Get bank_id and fact_type before deletion
                row = await conn.fetchrow(
                    f"SELECT bank_id, fact_type FROM {fq_table('memory_units')} WHERE id = $1",
                    str(unit_uuid),
                )
                bank_id = row["bank_id"] if row else None
                fact_type = row["fact_type"] if row else None

                # Delete the memory unit first (cascades to links and associations).
                # The stale-observation sweep runs AFTER the delete so it also catches
                # observations inserted concurrently by consolidation (otherwise a
                # racing insert committed between the sweep and the delete would
                # leave an orphan referencing this just-deleted source memory).
                deleted = await conn.fetchval(
                    f"DELETE FROM {fq_table('memory_units')} WHERE id = $1 RETURNING id", unit_id
                )

                # Invalidate observations referencing this (now-deleted) source memory
                if bank_id and fact_type in ("experience", "world"):
                    invalidated_obs = await self._delete_stale_observations_for_memories(conn, bank_id, [unit_id])
                    if invalidated_obs > 0:
                        bank_id_for_consolidation = bank_id

                result = {
                    "success": deleted is not None,
                    "unit_id": str(deleted) if deleted else None,
                    "message": "Memory unit and all its links deleted successfully"
                    if deleted
                    else "Memory unit not found",
                }

        if bank_id_for_consolidation:
            try:
                await self.submit_async_consolidation(
                    bank_id=bank_id_for_consolidation, request_context=request_context
                )
            except Exception as e:
                logger.warning(
                    f"Failed to submit consolidation after memory deletion for bank {bank_id_for_consolidation}: {e}"
                )

        return result

    async def delete_bank(
        self,
        bank_id: str,
        fact_type: str | None = None,
        *,
        delete_bank_profile: bool = True,
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
            fact_type: Optional fact type filter (world, experience). If provided, only deletes memories of that type.
            request_context: Request context for authentication.

        Returns:
            Dictionary with counts of deleted items
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="delete_bank", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        invalidated_obs = 0
        result: dict[str, int] = {}
        bank_internal_id: str | None = None
        async with acquire_with_retry(backend) as conn:
            # Ensure connection is not in read-only mode (can happen with connection poolers)
            await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
            async with conn.transaction():
                try:
                    if fact_type:
                        # For source memory types, capture ids so we can invalidate
                        # dependent observations AFTER the delete below. Running the
                        # stale-observation sweep post-delete ensures we also catch
                        # observations inserted concurrently by consolidation.
                        unit_ids: list[str] = []
                        if fact_type in ("experience", "world"):
                            unit_id_rows = await conn.fetch(
                                f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = $2",
                                bank_id,
                                fact_type,
                            )
                            unit_ids = [str(row["id"]) for row in unit_id_rows]

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

                        if unit_ids:
                            invalidated_obs = await self._delete_stale_observations_for_memories(
                                conn, bank_id, unit_ids
                            )

                        # Note: We don't delete entities when fact_type is specified,
                        # as they may be referenced by other memory units
                        result = {"memory_units_deleted": units_count, "entities_deleted": 0}
                    else:
                        # Delete all data for the bank — observations are included, no invalidation needed
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

                        result = {
                            "memory_units_deleted": units_count,
                            "entities_deleted": entities_count,
                            "documents_deleted": documents_count,
                        }

                        if delete_bank_profile:
                            # Delete the bank profile and retrieve internal_id for HNSW index cleanup
                            internal_id = await conn.fetchval(
                                f"DELETE FROM {fq_table('banks')} WHERE bank_id = $1 RETURNING internal_id", bank_id
                            )
                            if internal_id:
                                bank_internal_id = str(internal_id)
                            result["bank_deleted"] = True

                except Exception as e:
                    raise Exception(f"Failed to delete agent data: {str(e)}")

            # Drop per-bank vector indexes AFTER the transaction commits to avoid
            # AccessExclusiveLock deadlocks with concurrent bank deletions.
            # (DROP INDEX on memory_units conflicts with RowExclusiveLock from DELETE inside tx)
            if bank_internal_id:
                await bank_utils.drop_bank_vector_indexes(conn, bank_internal_id, ops=self._backend.ops)

        if invalidated_obs > 0:
            try:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit consolidation after bank deletion for bank {bank_id}: {e}")

        return result

    async def clear_observations(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Clear all observations for a bank (consolidated knowledge).

        Args:
            bank_id: Bank ID to clear observations for
            request_context: Request context for authentication.

        Returns:
            Dictionary with count of deleted observations
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="clear_observations", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                # Count observations before deletion
                count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
                    bank_id,
                )

                # Delete all observations
                await conn.execute(
                    f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
                    bank_id,
                )

                # Reset consolidated_at on source memories so they get re-consolidated
                await conn.execute(
                    f"UPDATE {fq_table('memory_units')} SET consolidated_at = NULL WHERE bank_id = $1 AND fact_type IN ('experience', 'world')",
                    bank_id,
                )

                # Reset consolidation timestamp
                await conn.execute(
                    f"UPDATE {fq_table('banks')} SET last_consolidated_at = NULL WHERE bank_id = $1",
                    bank_id,
                )

                return {"deleted_count": count or 0}

    async def retry_failed_consolidation(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Reset memories that previously failed consolidation so they are retried on the next
        consolidation run.

        Clears consolidation_failed_at (and consolidated_at) for all memories in the bank
        that were marked as permanently failed after exhausting all LLM retries and adaptive
        batch splitting. Does not delete any observations.

        Args:
            bank_id: Bank ID
            request_context: Request context for authentication.

        Returns:
            Dictionary with count of memories queued for retry.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(
                bank_id=bank_id, operation="retry_failed_consolidation", request_context=request_context
            )
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            count = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND consolidation_failed_at IS NOT NULL
                  AND fact_type IN ('experience', 'world')
                """,
                bank_id,
            )
            await conn.execute(
                f"""
                UPDATE {fq_table("memory_units")}
                SET consolidation_failed_at = NULL, consolidated_at = NULL
                WHERE bank_id = $1
                  AND consolidation_failed_at IS NOT NULL
                  AND fact_type IN ('experience', 'world')
                """,
                bank_id,
            )
            return {"retried_count": count or 0}

    async def clear_observations_for_memory(
        self,
        bank_id: str,
        memory_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Clear all observations derived from a specific memory and mark source memories
        (including the given memory itself) for re-consolidation.

        Unlike deleting the memory, the memory itself is preserved. This is useful
        when you want to force re-consolidation of a specific memory's observations
        without losing the underlying fact.

        Args:
            bank_id: Bank ID
            memory_id: ID of the memory whose observations should be cleared
            request_context: Request context for authentication.

        Returns:
            Dictionary with count of deleted observations
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(
                bank_id=bank_id, operation="clear_observations_for_memory", request_context=request_context
            )
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        deleted_count = 0

        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                import uuid as uuid_module

                deleted_count = await self._delete_stale_observations_for_memories(conn, bank_id, [memory_id])

                # Also reset this memory's own consolidated_at so it gets re-consolidated
                # (the memory was a source for the deleted observations, so it needs new ones)
                if deleted_count > 0:
                    await conn.execute(
                        f"""
                        UPDATE {fq_table("memory_units")}
                        SET consolidated_at = NULL
                        WHERE id = $1
                          AND bank_id = $2
                          AND fact_type IN ('experience', 'world')
                        """,
                        uuid_module.UUID(memory_id),
                        bank_id,
                    )

        if deleted_count > 0:
            await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)

        return {"deleted_count": deleted_count}

    async def run_consolidation(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, int]:
        """
        Run memory consolidation to create/update mental models.

        Args:
            bank_id: Bank ID to run consolidation for
            request_context: Request context for authentication.

        Returns:
            Dictionary with consolidation stats
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="run_consolidation", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        from .consolidation import run_consolidation_job

        # Create parent span for consolidation operation
        with create_operation_span("consolidation", bank_id):
            result = await run_consolidation_job(
                memory_engine=self,
                bank_id=bank_id,
                request_context=request_context,
            )

            return {
                "processed": result.get("processed", 0),
                "created": result.get("created", 0),
                "updated": result.get("updated", 0),
                "skipped": result.get("skipped", 0),
            }

    async def get_graph_data(
        self,
        bank_id: str | None = None,
        fact_type: str | None = None,
        *,
        limit: int = 1000,
        q: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "all_strict",
        document_id: str | None = None,
        chunk_id: str | None = None,
        request_context: "RequestContext",
    ):
        """
        Get graph data for visualization.

        Args:
            bank_id: Filter by bank ID
            fact_type: Filter by fact type (world, experience)
            limit: Maximum number of items to return (default: 1000)
            q: Full-text search query (searches text and context fields)
            tags: Filter by tags
            tags_match: Tag matching mode (default: all_strict)
            document_id: Filter by document ID
            chunk_id: Filter by chunk ID
            request_context: Request context for authentication.

        Returns:
            Dict with nodes, edges, table_rows, total_units, and limit
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_graph_data", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            # Get memory units, optionally filtered by bank_id and fact_type
            query_conditions = []
            query_params = []
            param_count = 0

            bank_id_placeholder: int | None = None
            if bank_id:
                param_count += 1
                bank_id_placeholder = param_count
                query_conditions.append(f"bank_id = ${param_count}")
                query_params.append(bank_id)

            if fact_type:
                param_count += 1
                query_conditions.append(f"fact_type = ${param_count}")
                query_params.append(fact_type)

            if document_id:
                param_count += 1
                obs_match = self._observations_via_source_match_sql(
                    "document_id", source_placeholder=param_count, bank_placeholder=bank_id_placeholder
                )
                query_conditions.append(
                    f"(document_id = ${param_count} OR (fact_type = 'observation' AND {obs_match}))"
                )
                query_params.append(document_id)

            if chunk_id:
                param_count += 1
                obs_match = self._observations_via_source_match_sql(
                    "chunk_id", source_placeholder=param_count, bank_placeholder=bank_id_placeholder
                )
                query_conditions.append(f"(chunk_id = ${param_count} OR (fact_type = 'observation' AND {obs_match}))")
                query_params.append(chunk_id)

            if q:
                param_count += 1
                query_conditions.append(f"(text ILIKE ${param_count} OR context ILIKE ${param_count})")
                query_params.append(f"%{q}%")

            if tags:
                from .search.tags import build_tags_where_clause_simple

                tag_clause = build_tags_where_clause_simple(tags, param_count + 1, match=tags_match)
                if tag_clause:
                    query_conditions.append(tag_clause.removeprefix("AND "))
                    param_count += 1
                    query_params.append(tags)

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
                SELECT id, text, event_date, context, occurred_start, occurred_end, mentioned_at, document_id, chunk_id, fact_type, tags, created_at, proof_count, source_memory_ids
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
            unit_id_set = set(unit_ids)

            # Collect source memory IDs from observations
            source_memory_ids = []
            for unit in units:
                if unit["source_memory_ids"]:
                    source_memory_ids.extend(unit["source_memory_ids"])
            source_memory_ids = list(set(source_memory_ids))  # Deduplicate

            # Fetch links where BOTH endpoints are in the visible set (or source memories)
            # Cap at 10k edges — the UI can't usefully render more, and uncapped queries
            # on highly-connected graphs (e.g. 1000 nodes with 500k+ edges) are too slow.
            max_edges = 10000
            all_relevant_ids = unit_ids + source_memory_ids
            if all_relevant_ids:
                links = await conn.fetch(
                    f"""
                    SELECT ml.from_unit_id,
                           ml.to_unit_id,
                           ml.link_type,
                           ml.weight,
                           e.canonical_name as entity_name
                    FROM {fq_table("memory_links")} ml
                    LEFT JOIN {fq_table("entities")} e ON ml.entity_id = e.id
                    WHERE ml.from_unit_id = ANY($1::uuid[]) AND ml.to_unit_id = ANY($1::uuid[])
                    ORDER BY ml.weight DESC NULLS LAST
                    LIMIT $2
                """,
                    all_relevant_ids,
                    max_edges,
                )
            else:
                links = []

            # Copy links from source memories to observations
            # Observations inherit links from their source memories via source_memory_ids
            # Build a map from source_id to observation_ids
            source_to_observations = {}
            for unit in units:
                if unit["source_memory_ids"]:
                    for source_id in unit["source_memory_ids"]:
                        if source_id not in source_to_observations:
                            source_to_observations[source_id] = []
                        source_to_observations[source_id].append(unit["id"])

            copied_links = []
            for link in links:
                from_id = link["from_unit_id"]
                to_id = link["to_unit_id"]

                # Get observations that should inherit this link
                from_observations = source_to_observations.get(from_id, [])
                to_observations = source_to_observations.get(to_id, [])

                # If from_id is a source memory, copy links to its observations
                if from_observations:
                    for obs_id in from_observations:
                        # Only include if the target is visible
                        if to_id in unit_id_set or to_observations:
                            target = to_observations[0] if to_observations and to_id not in unit_id_set else to_id
                            if target in unit_id_set and obs_id != target:
                                copied_links.append(
                                    {
                                        "from_unit_id": obs_id,
                                        "to_unit_id": target,
                                        "link_type": link["link_type"],
                                        "weight": link["weight"],
                                        "entity_name": link["entity_name"],
                                    }
                                )

                # If to_id is a source memory, copy links to its observations
                if to_observations and from_id in unit_id_set:
                    for obs_id in to_observations:
                        if from_id != obs_id:
                            copied_links.append(
                                {
                                    "from_unit_id": from_id,
                                    "to_unit_id": obs_id,
                                    "link_type": link["link_type"],
                                    "weight": link["weight"],
                                    "entity_name": link["entity_name"],
                                }
                            )

            # Keep only direct links between visible nodes
            direct_links = [
                link for link in links if link["from_unit_id"] in unit_id_set and link["to_unit_id"] in unit_id_set
            ]

            # Get entity information — only for visible units
            # Fetch entities for visible units AND their source memories
            # (so observations can inherit entities from source memories)
            entity_lookup_ids = unit_ids + source_memory_ids
            if entity_lookup_ids:
                unit_entities = await conn.fetch(
                    f"""
                    SELECT ue.unit_id, e.canonical_name
                    FROM {fq_table("unit_entities")} ue
                    JOIN {fq_table("entities")} e ON ue.entity_id = e.id
                    WHERE ue.unit_id = ANY($1::uuid[])
                    ORDER BY ue.unit_id
                """,
                    entity_lookup_ids,
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

        # For observations, inherit entities from source memories
        for unit in units:
            if unit["source_memory_ids"] and unit["id"] not in entity_map:
                # Collect entities from all source memories
                source_entities = []
                for source_id in unit["source_memory_ids"]:
                    if source_id in entity_map:
                        source_entities.extend(entity_map[source_id])
                if source_entities:
                    # Deduplicate while preserving order
                    entity_map[unit["id"]] = list(dict.fromkeys(source_entities))

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

        # Build observation-inferred links from inherited entities and shared source memories.
        # Observations never have direct memory_links rows, so all their links must be derived.
        observation_units = [unit for unit in units if unit["fact_type"] == "observation"]
        observation_ids = {unit["id"] for unit in observation_units}

        # Entity links: pair observations that share at least one inherited entity
        entity_to_observations: dict[str, list] = {}
        for obs_id in observation_ids:
            for entity_name in entity_map.get(obs_id, []):
                entity_to_observations.setdefault(entity_name, []).append(obs_id)

        # Semantic links: pair observations that share at least one source memory
        source_to_obs_for_semantic: dict = {}
        for unit in observation_units:
            if unit["source_memory_ids"]:
                for src_id in unit["source_memory_ids"]:
                    source_to_obs_for_semantic.setdefault(src_id, []).append(unit["id"])

        observation_inferred_links = []
        seen_inferred: set[tuple] = set()

        for entity_name, obs_ids in entity_to_observations.items():
            for i, obs_a in enumerate(obs_ids):
                for obs_b in obs_ids[i + 1 :]:
                    pair = (min(str(obs_a), str(obs_b)), max(str(obs_a), str(obs_b)), "entity", entity_name)
                    if pair not in seen_inferred:
                        seen_inferred.add(pair)
                        observation_inferred_links.append(
                            {
                                "from_unit_id": obs_a,
                                "to_unit_id": obs_b,
                                "link_type": "entity",
                                "weight": 1.0,
                                "entity_name": entity_name,
                            }
                        )

        for src_id, obs_ids in source_to_obs_for_semantic.items():
            for i, obs_a in enumerate(obs_ids):
                for obs_b in obs_ids[i + 1 :]:
                    pair = (min(str(obs_a), str(obs_b)), max(str(obs_a), str(obs_b)), "semantic", "")
                    if pair not in seen_inferred:
                        seen_inferred.add(pair)
                        observation_inferred_links.append(
                            {
                                "from_unit_id": obs_a,
                                "to_unit_id": obs_b,
                                "link_type": "semantic",
                                "weight": 1.0,
                                "entity_name": None,
                            }
                        )

        # Build edges (combine direct links, copied links from sources, and observation-inferred links)
        edges = []
        seen_edges: set[tuple] = set()
        all_links = direct_links + copied_links + observation_inferred_links
        for row in all_links:
            from_id = str(row["from_unit_id"])
            to_id = str(row["to_unit_id"])
            link_type = row["link_type"]
            weight = row["weight"]
            entity_name = row.get("entity_name")

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

            edge_key = (from_id, to_id, link_type, entity_name or "")
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

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
                    "tags": list(row["tags"]) if row["tags"] else [],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "proof_count": row["proof_count"] if row["proof_count"] else None,
                }
            )

        return {"nodes": nodes, "edges": edges, "table_rows": table_rows, "total_units": total_count, "limit": limit}

    async def list_memory_units(
        self,
        bank_id: str,
        *,
        fact_type: str | None = None,
        search_query: str | None = None,
        consolidation_state: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ):
        """
        List memory units for table view with optional full-text search.

        Args:
            bank_id: Filter by bank ID
            fact_type: Filter by fact type (world, experience)
            search_query: Full-text search query (searches text and context fields)
            consolidation_state: Optional filter on consolidation state. One of
                'failed' (consolidation permanently failed and awaiting recovery),
                'pending' (not yet consolidated, no failure), or
                'done' (successfully consolidated). Only applies to source memory
                types (world/experience).
            limit: Maximum number of results to return
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of memory units) and total count
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_memory_units", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
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

            if consolidation_state:
                state = consolidation_state.lower()
                if state == "failed":
                    query_conditions.append(
                        "consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')"
                    )
                elif state == "pending":
                    query_conditions.append(
                        "consolidated_at IS NULL AND consolidation_failed_at IS NULL "
                        "AND fact_type IN ('experience', 'world')"
                    )
                elif state == "done":
                    query_conditions.append("consolidated_at IS NOT NULL AND fact_type IN ('experience', 'world')")
                else:
                    raise ValueError(
                        f"Invalid consolidation_state '{consolidation_state}': expected 'failed', 'pending', or 'done'."
                    )

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
                SELECT id, text, event_date, context, fact_type, mentioned_at, occurred_start, occurred_end, chunk_id, proof_count, tags, consolidated_at, consolidation_failed_at
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
                        "proof_count": row["proof_count"] if row["proof_count"] is not None else 1,
                        "tags": list(row["tags"]) if row["tags"] else [],
                        "consolidated_at": row["consolidated_at"].isoformat() if row["consolidated_at"] else None,
                        "consolidation_failed_at": (
                            row["consolidation_failed_at"].isoformat() if row["consolidation_failed_at"] else None
                        ),
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

        Raises:
            ValueError: If memory_id is not a valid UUID
        """
        try:
            memory_uuid = uuid.UUID(memory_id)
        except ValueError:
            raise ValueError(f"Invalid memory_id: '{memory_id}' is not a valid UUID")
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_memory_unit", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            # Get the memory unit (include source_memory_ids for mental models)
            row = await conn.fetchrow(
                f"""
                SELECT id, text, context, event_date, occurred_start, occurred_end,
                       mentioned_at, fact_type, document_id, chunk_id, tags, source_memory_ids,
                       observation_scopes
                FROM {fq_table("memory_units")}
                WHERE id = $1 AND bank_id = $2
                """,
                str(memory_uuid),
                bank_id,
            )

            if not row:
                return None

            # Get entity information. _entity_rows_for_units_sql handles the
            # observation→source_memory_ids inheritance fallback in SQL, so a
            # single query covers direct rows and inherited ones.
            entities_rows = await conn.fetch(
                self._entity_rows_for_units_sql(unit_ids_placeholder=1),
                [row["id"]],
            )
            entities = [r["canonical_name"] for r in entities_rows]

            result = {
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
                "observation_scopes": row["observation_scopes"] if row["observation_scopes"] else None,
            }

            # For observations, include source_memory_ids
            # history is deprecated here - use GET /memories/{id}/history instead
            if row["fact_type"] == "observation":
                result["history"] = []

            if row["fact_type"] == "observation" and row["source_memory_ids"]:
                source_ids = row["source_memory_ids"]
                result["source_memory_ids"] = [str(sid) for sid in source_ids]

                # Fetch source memories
                source_rows = await conn.fetch(
                    f"""
                    SELECT id, text, fact_type, context, occurred_start, mentioned_at
                    FROM {fq_table("memory_units")}
                    WHERE id = ANY($1::uuid[])
                    ORDER BY mentioned_at DESC NULLS LAST
                    """,
                    source_ids,
                )
                result["source_memories"] = [
                    {
                        "id": str(r["id"]),
                        "text": r["text"],
                        "type": r["fact_type"],
                        "context": r["context"],
                        "occurred_start": r["occurred_start"].isoformat() if r["occurred_start"] else None,
                        "mentioned_at": r["mentioned_at"].isoformat() if r["mentioned_at"] else None,
                    }
                    for r in source_rows
                ]

            return result

    async def get_observation_history(
        self,
        bank_id: str,
        memory_id: str,
        request_context: "RequestContext",
    ) -> list[dict] | None:
        """
        Get the history of an observation, with source facts resolved to their text.

        Returns None if the memory is not found or is not an observation.
        Returns a list of history entries (most recent first), each with source_facts resolved.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_observation_history", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT fact_type, history, source_memory_ids
                FROM {fq_table("memory_units")}
                WHERE id = $1 AND bank_id = $2
                """,
                uuid.UUID(memory_id),
                bank_id,
            )
            if not row:
                return None
            if row["fact_type"] != "observation":
                return []

            raw_history = row["history"]
            if isinstance(raw_history, str):
                raw_history = json.loads(raw_history)
            if not raw_history:
                return []

            # Collect all source memory IDs (current full set + all historical new ones)
            current_source_ids: list[str] = [str(sid) for sid in (row["source_memory_ids"] or [])]
            all_source_ids: set[uuid.UUID] = set(uuid.UUID(sid) for sid in current_source_ids)
            for entry in raw_history:
                for sid in entry.get("new_source_memory_ids", []):
                    try:
                        all_source_ids.add(uuid.UUID(sid))
                    except (ValueError, AttributeError):
                        pass

            # Resolve all source memories in one query
            source_map: dict[str, dict] = {}
            if all_source_ids:
                source_rows = await conn.fetch(
                    f"""
                    SELECT id, text, fact_type, context
                    FROM {fq_table("memory_units")}
                    WHERE id = ANY($1::uuid[])
                    """,
                    list(all_source_ids),
                )
                for r in source_rows:
                    source_map[str(r["id"])] = {
                        "id": str(r["id"]),
                        "text": r["text"],
                        "type": r["fact_type"],
                        "context": r["context"] or None,
                    }

            # Reconstruct cumulative source IDs per change by working backwards from current state.
            # Source IDs are only ever accumulated (never removed), so:
            #   after_change_N = before_change_N + new_source_memory_ids_N
            cumulative_ids: list[str] = list(current_source_ids)
            enriched: list[dict] = []
            for entry in reversed(raw_history):
                new_ids_in_entry: set[str] = set(entry.get("new_source_memory_ids", []))
                source_facts = []
                for sid in cumulative_ids:
                    fact = source_map.get(sid, {"id": sid, "text": None, "type": None, "context": None})
                    source_facts.append({**fact, "is_new": sid in new_ids_in_entry})
                enriched_entry = dict(entry)
                enriched_entry["source_facts"] = source_facts
                enriched.append(enriched_entry)
                # Step back: remove the new IDs added by this change to get the prior state
                cumulative_ids = [sid for sid in cumulative_ids if sid not in new_ids_in_entry]

            enriched.reverse()
            return enriched

    async def list_documents(
        self,
        bank_id: str,
        *,
        search_query: str | None = None,
        tags: list[str] | None = None,
        tags_match: "TagsMatch" = "any_strict",
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ):
        """
        List documents with optional search and pagination.

        Args:
            bank_id: bank ID (required)
            search_query: Search in document ID
            tags: Filter by tags
            tags_match: How to match tags (any, all, any_strict, all_strict)
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of documents without original_text) and total count
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_documents", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
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

            tags_clause, tags_params, next_param = build_tags_where_clause(
                tags, param_offset=param_count + 1, match=tags_match
            )
            query_params.extend(tags_params)
            param_count = next_param - 1  # next_param is next available; convert to last used

            where_clause = "WHERE " + " AND ".join(query_conditions) if query_conditions else ""
            if tags_clause:
                # tags_clause starts with "AND", append after WHERE conditions
                where_clause = where_clause + " " + tags_clause if where_clause else "WHERE " + tags_clause[4:].lstrip()

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
                    retain_params,
                    tags
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

                retain_params_val = conn.parse_json(row["retain_params"])

                # document_metadata is sourced from retain_params.metadata
                document_metadata = retain_params_val.get("metadata") if retain_params_val else None

                items.append(
                    {
                        "id": doc_id,
                        "bank_id": bank_id_val,
                        "content_hash": row["content_hash"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
                        "text_length": row["text_length"] or 0,
                        "memory_unit_count": unit_count,
                        "retain_params": retain_params_val or None,
                        "document_metadata": document_metadata or None,
                        "tags": row["tags"] if row["tags"] else [],
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
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
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

            if self._operation_validator:
                from hindsight_api.extensions import BankReadContext

                ctx = BankReadContext(bank_id=chunk["bank_id"], operation="get_chunk", request_context=request_context)
                await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

            return {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "bank_id": chunk["bank_id"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "created_at": chunk["created_at"].isoformat() if chunk["created_at"] else "",
            }

    async def list_document_chunks(
        self,
        bank_id: str,
        document_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List all chunks for a given document, ordered by chunk_index.

        Args:
            bank_id: Bank ID
            document_id: Document ID
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication.

        Returns:
            Dict with items (list of chunks) and total count
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_document_chunks", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            # Verify document exists
            doc = await conn.fetchrow(
                f"SELECT id FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
            if not doc:
                return None

            count_result = await conn.fetchrow(
                f"""
                SELECT COUNT(*) as total
                FROM {fq_table("chunks")}
                WHERE document_id = $1 AND bank_id = $2
                """,
                document_id,
                bank_id,
            )
            total = count_result["total"]

            chunks = await conn.fetch(
                f"""
                SELECT chunk_id, document_id, bank_id, chunk_index, chunk_text, created_at
                FROM {fq_table("chunks")}
                WHERE document_id = $1 AND bank_id = $2
                ORDER BY chunk_index ASC
                LIMIT $3 OFFSET $4
                """,
                document_id,
                bank_id,
                limit,
                offset,
            )

            items = [
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "bank_id": row["bank_id"],
                    "chunk_index": row["chunk_index"],
                    "chunk_text": row["chunk_text"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                }
                for row in chunks
            ]

            return {"items": items, "total": total, "limit": limit, "offset": offset}

    async def reprocess_document(
        self,
        bank_id: str,
        document_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Reprocess a document by re-running retain with its existing content and parameters.

        Args:
            bank_id: Bank ID
            document_id: Document ID to reprocess
            request_context: Request context for authentication.

        Returns:
            Dict with operation result or None if document not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="reprocess_document", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        # Fetch the document
        doc = await self.get_document(document_id, bank_id, request_context=request_context)
        if not doc:
            return None

        original_text = doc.get("original_text")
        if not original_text:
            return None

        # Rebuild the content dict from retain_params
        retain_params = doc.get("retain_params") or {}
        content_dict: dict[str, Any] = {
            "content": original_text,
            "document_id": document_id,
            "update_mode": "replace",
        }
        if retain_params.get("context"):
            content_dict["context"] = retain_params["context"]
        if retain_params.get("event_date"):
            content_dict["event_date"] = retain_params["event_date"]
        if retain_params.get("metadata"):
            content_dict["metadata"] = retain_params["metadata"]
        if retain_params.get("entities"):
            content_dict["entities"] = retain_params["entities"]

        tags = doc.get("tags") or []
        if tags:
            content_dict["tags"] = tags
        if retain_params.get("observation_scopes") is not None:
            content_dict["observation_scopes"] = retain_params["observation_scopes"]

        strategy = retain_params.get("strategy")

        result = await self.submit_async_retain(
            bank_id,
            [content_dict],
            strategy=strategy,
            request_context=request_context,
        )

        return result

    # ==================== bank profile Methods ====================

    # Type-checker overloads: when create_if_missing is True (the default),
    # this method always returns a profile dict — the type checker can rely
    # on non-None for every existing caller. Only when create_if_missing is
    # explicitly False does the return become Optional.
    @overload
    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        create_if_missing: Literal[True] = True,
    ) -> dict[str, Any]: ...

    @overload
    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        create_if_missing: Literal[False],
    ) -> dict[str, Any] | None: ...

    async def get_bank_profile(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        create_if_missing: bool = True,
    ) -> dict[str, Any] | None:
        """
        Get bank profile (name, disposition + mission).

        Args:
            bank_id: bank IDentifier
            request_context: Request context for authentication.
            create_if_missing: If True (default), the bank is auto-created
                with defaults when it does not exist. Pass False from read-
                only callers (HTTP GET handlers, polling, etc.) so a missing
                bank surfaces as None rather than being silently created.
                The caller is then responsible for translating None to a
                404 (or similar).

        Returns:
            Dict with name, disposition traits, and mission, or None when
            create_if_missing=False and the bank does not exist.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_bank_profile", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        if not create_if_missing:
            existing = await bank_utils.get_bank_profile_if_exists(backend, bank_id)
            if existing is None:
                return None
            profile, created = existing, False
        else:
            profile, created = await bank_utils.get_or_create_bank_profile(backend, bank_id)

        # Apply HINDSIGHT_API_DEFAULT_BANK_TEMPLATE to freshly-created banks. Done
        # before reading the resolved config below so the template's overrides
        # (e.g. reflect_mission, dispositions) are visible on this very call.
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

        # reflect_mission and disposition in config take precedence over the legacy DB columns
        config_dict = await self._config_resolver.get_bank_config(bank_id, request_context)
        mission = config_dict.get("reflect_mission") or profile["mission"]

        # Overlay disposition from config if explicitly set; fall back to DB values
        db_disp = profile["disposition"]
        db_disp_dict = db_disp.model_dump() if hasattr(db_disp, "model_dump") else dict(db_disp)
        cfg_skep = config_dict.get("disposition_skepticism")
        cfg_lit = config_dict.get("disposition_literalism")
        cfg_emp = config_dict.get("disposition_empathy")
        disposition = {
            "skepticism": cfg_skep if cfg_skep is not None else db_disp_dict["skepticism"],
            "literalism": cfg_lit if cfg_lit is not None else db_disp_dict["literalism"],
            "empathy": cfg_emp if cfg_emp is not None else db_disp_dict["empathy"],
        }

        return {
            "bank_id": bank_id,
            "name": profile["name"],
            "disposition": disposition,
            "mission": mission,
        }

    async def _apply_default_bank_template(
        self,
        bank_id: str,
        request_context: "RequestContext",
    ) -> None:
        """Apply HINDSIGHT_API_DEFAULT_BANK_TEMPLATE to a freshly-created bank.

        No-op if the env var is unset. A malformed default template is logged
        and swallowed here rather than raised, so a bad server-level setting
        cannot wedge bank creation across all callers. Misconfiguration is
        still surfaced loudly via `logger.error`.
        """
        from ..config import get_config

        template_dict = get_config().default_bank_template
        if not template_dict:
            return

        # Lazy import to avoid a cycle (http.py imports memory_engine).
        from pydantic import ValidationError

        from hindsight_api.api.http import (
            BankTemplateManifest,
            apply_bank_template_manifest,
            validate_bank_template,
        )

        try:
            manifest = BankTemplateManifest.model_validate(template_dict)
        except ValidationError as e:
            errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
            logger.error(
                "HINDSIGHT_API_DEFAULT_BANK_TEMPLATE failed schema validation "
                f"and will be ignored for bank '{bank_id}': {'; '.join(errors)}"
            )
            return

        semantic_errors = validate_bank_template(manifest)
        if semantic_errors:
            logger.error(
                "HINDSIGHT_API_DEFAULT_BANK_TEMPLATE failed semantic validation "
                f"and will be ignored for bank '{bank_id}': {'; '.join(semantic_errors)}"
            )
            return

        try:
            await apply_bank_template_manifest(
                memory=self,
                bank_id=bank_id,
                manifest=manifest,
                request_context=request_context,
            )
            logger.info(f"Applied HINDSIGHT_API_DEFAULT_BANK_TEMPLATE to newly-created bank '{bank_id}'")
        except Exception as e:
            logger.error(f"Failed to apply HINDSIGHT_API_DEFAULT_BANK_TEMPLATE to bank '{bank_id}': {e}")

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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(
                bank_id=bank_id, operation="update_bank_disposition", request_context=request_context
            )
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        await bank_utils.update_bank_disposition(self._backend, bank_id, disposition)

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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="set_bank_mission", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        await bank_utils.set_bank_mission(self._backend, bank_id, mission)
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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="merge_bank_mission", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()
        return await bank_utils.merge_bank_mission(self._backend, self._reflect_llm_config, bank_id, new_info)

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
        backend = await self._get_backend()
        banks = await bank_utils.list_banks(self._backend)
        if self._operation_validator:
            from hindsight_api.extensions import BankListContext

            result = await self._operation_validator.filter_bank_list(
                BankListContext(banks=banks, request_context=request_context)
            )
            banks = result.banks
        return banks

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
        tag_groups: list[TagGroup] | None = None,
        exclude_mental_model_ids: list[str] | None = None,
        fact_types: list[str] | None = None,
        exclude_mental_models: bool = False,
        recall_include_chunks: bool | None = None,
        recall_max_tokens_override: int | None = None,
        recall_chunks_max_tokens_override: int | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        _skip_span: bool = False,
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
            tags: Optional tags to filter memories
            tags_match: How to match tags - "any" (OR), "all" (AND)
            exclude_mental_model_ids: Optional list of mental model IDs to exclude from search
                (used when refreshing a mental model to avoid circular reference)

        Returns:
            ReflectResult containing:
                - text: Plain text answer
                - based_on: Empty dict (agent retrieves facts dynamically)
                - structured_output: None (not yet supported for agentic reflect)
        """
        # Use cached LLM config
        if self._reflect_llm_config is None:
            raise ValueError("Memory LLM API key not set. Set HINDSIGHT_API_LLM_API_KEY environment variable.")

        # Block reflect when the reflect LLM provider is "none"
        if self._reflect_llm_config.provider == "none":
            from .providers.none_llm import LLMNotAvailableError

            raise LLMNotAvailableError(
                "Reflect requires an LLM provider. Current provider is set to 'none'. "
                "Set HINDSIGHT_API_LLM_PROVIDER to a real provider (e.g., openai, anthropic, gemini)."
            )

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

        resolved_reflect_config = await self._config_resolver.resolve_full_config(bank_id, request_context)

        # Compute max iterations based on budget
        config = get_config()
        base_max_iterations = config.reflect_max_iterations
        # Budget multipliers: low=0.5x, mid=1x, high=2x
        budget_multipliers = {Budget.LOW: 0.5, Budget.MID: 1.0, Budget.HIGH: 2.0}
        effective_budget = budget or Budget.LOW
        max_iterations = max(1, int(base_max_iterations * budget_multipliers.get(effective_budget, 1.0)))
        max_context_tokens = config.reflect_max_context_tokens
        wall_timeout = config.reflect_wall_timeout

        # Run agentic loop - acquire connections only when needed for DB operations
        # (not held during LLM calls which can be slow)
        backend = await self._get_backend()

        # Get bank stats for freshness info
        bank_stats = await self.get_bank_stats(bank_id, request_context=request_context)
        last_consolidated_at = bank_stats.last_consolidated_at if hasattr(bank_stats, "last_consolidated_at") else None
        pending_consolidation = bank_stats.pending_consolidation if hasattr(bank_stats, "pending_consolidation") else 0

        # Create tool callbacks that acquire connections only when needed
        from .retain import embedding_utils

        async def search_mental_models_fn(q: str, max_results: int = 5) -> dict[str, Any]:
            # Generate embedding for the query
            embeddings = await embedding_utils.generate_embeddings_batch(self.embeddings, [q])
            query_embedding = embeddings[0]
            async with backend.acquire() as conn:
                return await tool_search_mental_models(
                    self,
                    conn,
                    bank_id,
                    q,
                    query_embedding,
                    max_results=max_results,
                    tags=tags,
                    tags_match=tags_match,
                    tag_groups=tag_groups,
                    exclude_ids=exclude_mental_model_ids,
                )

        # Get reflect source facts config (hierarchical: env → tenant → bank)
        config_dict = await self._config_resolver.get_bank_config(bank_id, request_context)
        reflect_source_facts_max_tokens = config_dict.get(
            "reflect_source_facts_max_tokens", DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS
        )

        # Resolve recall overrides: caller arg (e.g. mental model trigger) → bank config → env default
        effective_recall_include_chunks = (
            recall_include_chunks
            if recall_include_chunks is not None
            else config_dict.get("recall_include_chunks", DEFAULT_RECALL_INCLUDE_CHUNKS)
        )
        effective_recall_max_tokens = (
            recall_max_tokens_override
            if recall_max_tokens_override is not None
            else config_dict.get("recall_max_tokens", DEFAULT_RECALL_MAX_TOKENS)
        )
        effective_recall_chunks_max_tokens = (
            recall_chunks_max_tokens_override
            if recall_chunks_max_tokens_override is not None
            else config_dict.get("recall_chunks_max_tokens", DEFAULT_RECALL_CHUNKS_MAX_TOKENS)
        )

        async def search_observations_fn(q: str, max_tokens: int = 5000) -> dict[str, Any]:
            return await tool_search_observations(
                self,
                bank_id,
                q,
                request_context,
                max_tokens=max_tokens,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                last_consolidated_at=last_consolidated_at,
                pending_consolidation=pending_consolidation,
                source_facts_max_tokens=reflect_source_facts_max_tokens,
                created_after=created_after,
                created_before=created_before,
            )

        # Determine which tools to enable based on fact_types and exclude_mental_models
        include_observations = fact_types is None or "observation" in fact_types
        recall_fact_types = [ft for ft in (fact_types or ["world", "experience"]) if ft in ("world", "experience")]
        include_recall = bool(recall_fact_types)

        # Defaults are bound at closure-definition time (re-evaluated on each
        # reflect_async call), so per-bank/per-trigger overrides apply when the
        # agent invokes recall without explicit token args.
        async def recall_fn(
            q: str,
            max_tokens: int = effective_recall_max_tokens,
            max_chunk_tokens: int = effective_recall_chunks_max_tokens,
        ) -> dict[str, Any]:
            return await tool_recall(
                self,
                bank_id,
                q,
                request_context,
                max_tokens=max_tokens,
                tags=tags,
                tags_match=tags_match,
                tag_groups=tag_groups,
                max_chunk_tokens=max_chunk_tokens,
                fact_types=recall_fact_types if fact_types is not None else None,
                include_chunks=effective_recall_include_chunks,
                created_after=created_after,
                created_before=created_before,
            )

        async def expand_fn(memory_ids: list[str], depth: str) -> dict[str, Any]:
            async with backend.acquire() as conn:
                return await tool_expand(conn, bank_id, memory_ids, depth)

        # Load directives from the dedicated directives table
        # Directives are hard rules that must be followed in all responses
        # Use isolation_mode=True to prevent tag-scoped directives from leaking into untagged operations
        # Use the same tags_match as the reflect request so directives respect the same scoping rules
        directives_raw = await self.list_directives(
            bank_id=bank_id,
            tags=tags,
            tags_match=tags_match,
            active_only=True,
            request_context=request_context,
            isolation_mode=True,
        )
        directives = directives_raw
        if directives:
            logger.info(f"[REFLECT {reflect_id}] Loaded {len(directives)} directives")

        # Check if the bank has any mental models (skip check if all mental models are excluded)
        has_mental_models = False
        if not exclude_mental_models:
            async with backend.acquire() as conn:
                mental_model_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {fq_table('mental_models')} WHERE bank_id = $1",
                    bank_id,
                )
            has_mental_models = mental_model_count > 0
            if has_mental_models:
                logger.info(f"[REFLECT {reflect_id}] Bank has {mental_model_count} mental models")

        # Run the agent with parent span for reflect operation (skip if called from another operation)
        if not _skip_span:
            span_context = create_operation_span("reflect", bank_id)
            span_context.__enter__()
        else:
            span_context = None

        try:
            try:
                agent_result = await asyncio.wait_for(
                    run_reflect_agent(
                        llm_config=self._reflect_llm_config.with_config(resolved_reflect_config),
                        bank_id=bank_id,
                        query=query,
                        bank_profile=profile,
                        search_mental_models_fn=search_mental_models_fn,
                        search_observations_fn=search_observations_fn,
                        recall_fn=recall_fn,
                        expand_fn=expand_fn,
                        context=context,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        response_schema=response_schema,
                        directives=directives,
                        has_mental_models=has_mental_models,
                        include_observations=include_observations,
                        include_recall=include_recall,
                        budget=effective_budget,
                        max_context_tokens=max_context_tokens,
                    ),
                    timeout=wall_timeout,
                )
            except asyncio.TimeoutError:
                total_time = time.time() - reflect_start
                logger.error(
                    "[REFLECT %s] Wall-clock timeout after %.1fs (limit: %ss) for query: %.50s...",
                    reflect_id,
                    total_time,
                    wall_timeout,
                    query,
                )
                raise TimeoutError(
                    f"Reflect operation timed out after {wall_timeout} seconds. "
                    f"Consider reducing the budget or simplifying the query."
                )

            total_time = time.time() - reflect_start
            logger.info(
                "[REFLECT %s] Complete: %d chars, %d iterations, %d tool calls | %.3fs",
                reflect_id,
                len(agent_result.text),
                agent_result.iterations,
                agent_result.tools_called,
                total_time,
            )

            # Convert agent tool trace to ToolCallTrace objects
            tool_trace_result = [
                ToolCallTrace(
                    tool=tc.tool,
                    reason=tc.reason,
                    input=tc.input,
                    output=tc.output,
                    duration_ms=tc.duration_ms,
                    iteration=tc.iteration,
                )
                for tc in agent_result.tool_trace
            ]

            # Convert agent LLM trace to LLMCallTrace objects
            llm_trace_result = [
                LLMCallTrace(scope=lc.scope, duration_ms=lc.duration_ms) for lc in agent_result.llm_trace
            ]

            # Extract memories and observations from tool outputs - only include those the agent actually used
            # agent_result.used_memory_ids / used_observation_ids contain validated IDs from the done action
            used_memory_ids_set = set(agent_result.used_memory_ids) if agent_result.used_memory_ids else set()
            used_observation_ids_set = (
                set(agent_result.used_observation_ids) if agent_result.used_observation_ids else set()
            )
            # based_on stores facts, mental models, and directives
            # Note: directives list stores raw directive dicts (not MemoryFact), which will be converted to Directive objects
            based_on: dict[str, list[MemoryFact] | list[dict[str, Any]]] = {
                "world": [],
                "experience": [],
                "opinion": [],
                "observation": [],
                "mental-models": [],
                "directives": [],
            }
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
                            fact_type = memory_data.get("fact_type", "world")
                            if fact_type in based_on:
                                based_on[fact_type].append(
                                    MemoryFact(
                                        id=memory_id,
                                        text=memory_data.get("text", ""),
                                        fact_type=fact_type,
                                        context=memory_data.get("context"),
                                        occurred_start=memory_data.get("occurred_start"),
                                        occurred_end=memory_data.get("occurred_end"),
                                    )
                                )
                elif tc.tool == "search_observations" and "observations" in tc.output:
                    for obs_data in tc.output["observations"]:
                        obs_id = obs_data.get("id")
                        if obs_id and obs_id not in seen_memory_ids:
                            if used_observation_ids_set and obs_id not in used_observation_ids_set:
                                continue  # Skip observations not actually used by the agent
                            seen_memory_ids.add(obs_id)
                            based_on["observation"].append(MemoryFact(**obs_data))

            # Extract mental models from tool outputs - only include models the agent actually used
            # agent_result.used_mental_model_ids contains validated IDs from the done action
            used_model_ids_set = (
                set(agent_result.used_mental_model_ids) if agent_result.used_mental_model_ids else set()
            )
            based_on["mental-models"] = []
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
                            # Add to based_on as MemoryFact with type "mental-models"
                            model_name = model.get("name", "")
                            model_content = model.get("content", "")
                            based_on["mental-models"].append(
                                MemoryFact(
                                    id=model_id,
                                    text=f"{model_name}: {model_content}",
                                    fact_type="mental-models",
                                    context=f"{model.get('type', 'concept')} ({model.get('subtype', 'structural')})",
                                    occurred_start=None,
                                    occurred_end=None,
                                )
                            )
                elif tc.tool == "search_mental_models":
                    # Search mental models - include all returned models (filtered by used_model_ids_set if specified)
                    for model in tc.output.get("mental_models", []):
                        model_id = model.get("id")
                        if model_id and model_id not in seen_model_ids:
                            # Only include models that the agent declared as used (or all if none specified)
                            if used_model_ids_set and model_id not in used_model_ids_set:
                                continue  # Skip models not actually used by the agent
                            seen_model_ids.add(model_id)
                            # Add to based_on as MemoryFact with type "mental-models"
                            model_name = model.get("name", "")
                            model_content = model.get("content", "")
                            based_on["mental-models"].append(
                                MemoryFact(
                                    id=model_id,
                                    text=f"{model_name}: {model_content}",
                                    fact_type="mental-models",
                                    context=f"{model.get('type', 'concept')} ({model.get('subtype', 'structural')})",
                                    occurred_start=None,
                                    occurred_end=None,
                                )
                            )

            # Add directives to based_on["directives"]
            # Store raw directive dicts (with id, name, content) for http.py to convert to ReflectDirective
            for directive_raw in directives_raw:
                based_on["directives"].append(
                    {
                        "id": directive_raw["id"],
                        "name": directive_raw["name"],
                        "content": directive_raw["content"],
                    }
                )

            # Build directives_applied from agent result
            from hindsight_api.engine.response_models import DirectiveRef

            directives_applied_result = [
                DirectiveRef(id=d.id, name=d.name, content=d.content) for d in agent_result.directives_applied
            ]

            # Convert agent usage to TokenUsage format
            from hindsight_api.engine.response_models import TokenUsage

            usage = TokenUsage(
                input_tokens=agent_result.usage.input_tokens,
                output_tokens=agent_result.usage.output_tokens,
                total_tokens=agent_result.usage.total_tokens,
            )

            # Return response (compatible with existing API)
            result = ReflectResult(
                text=agent_result.text,
                based_on=based_on,
                structured_output=agent_result.structured_output,
                usage=usage,
                tool_trace=tool_trace_result,
                llm_trace=llm_trace_result,
                directives_applied=directives_applied_result,
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
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_entities", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
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

    async def get_entity_graph(
        self,
        bank_id: str,
        *,
        limit: int = 1000,
        min_count: int = 1,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Get entity co-occurrence graph for visualization.

        Returns nodes for entities and edges from the materialized
        entity_cooccurrences table. Edges are ordered by cooccurrence_count DESC
        and capped at `limit` to keep the payload renderable.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_entity_graph", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            edge_rows = await conn.fetch(
                f"""
                SELECT ec.entity_id_1,
                       ec.entity_id_2,
                       ec.cooccurrence_count,
                       ec.last_cooccurred,
                       e1.canonical_name AS name_1,
                       e1.mention_count  AS mention_count_1,
                       e2.canonical_name AS name_2,
                       e2.mention_count  AS mention_count_2
                FROM {fq_table("entity_cooccurrences")} ec
                JOIN {fq_table("entities")} e1 ON e1.id = ec.entity_id_1
                JOIN {fq_table("entities")} e2 ON e2.id = ec.entity_id_2
                WHERE e1.bank_id = $1
                  AND e2.bank_id = $1
                  AND ec.cooccurrence_count >= $2
                ORDER BY ec.cooccurrence_count DESC, ec.last_cooccurred DESC
                LIMIT $3
                """,
                bank_id,
                min_count,
                limit,
            )

        @dataclass
        class _EntityNode:
            id: str
            label: str
            mention_count: int

        nodes_by_id: dict[str, _EntityNode] = {}
        edges: list[dict[str, Any]] = []
        for row in edge_rows:
            for eid, name, mentions in (
                (row["entity_id_1"], row["name_1"], row["mention_count_1"]),
                (row["entity_id_2"], row["name_2"], row["mention_count_2"]),
            ):
                key = str(eid)
                if key not in nodes_by_id:
                    nodes_by_id[key] = _EntityNode(id=key, label=name, mention_count=mentions or 0)

            from_id = str(row["entity_id_1"])
            to_id = str(row["entity_id_2"])
            count = row["cooccurrence_count"]
            edges.append(
                {
                    "data": {
                        "id": f"{from_id}-{to_id}",
                        "source": from_id,
                        "target": to_id,
                        "linkType": "cooccurrence",
                        "weight": count,
                        "color": "#ffd700",
                        "lineStyle": "solid",
                        "lastCooccurred": row["last_cooccurred"].isoformat() if row["last_cooccurred"] else None,
                    }
                }
            )

        nodes = [
            {
                "data": {
                    "id": n.id,
                    "label": n.label,
                    "mentionCount": n.mention_count,
                    "color": "#42a5f5" if n.mention_count > 1 else "#90caf9",
                }
            }
            for n in nodes_by_id.values()
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "total_entities": len(nodes),
            "total_edges": len(edges),
            "limit": limit,
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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_tags", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        return await self._list_tags_from_table(
            table="memory_units",
            bank_id=bank_id,
            pattern=pattern,
            limit=limit,
            offset=offset,
        )

    async def list_mental_model_tags(
        self,
        bank_id: str,
        *,
        pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        List all unique tags used on mental models in a bank with usage counts.

        Same wildcard semantics as list_tags. Useful to populate tag autocompletion
        for UIs filtering mental models by tag.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(
                bank_id=bank_id,
                operation="list_mental_model_tags",
                request_context=request_context,
            )
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        return await self._list_tags_from_table(
            table="mental_models",
            bank_id=bank_id,
            pattern=pattern,
            limit=limit,
            offset=offset,
        )

    async def _list_tags_from_table(
        self,
        *,
        table: str,
        bank_id: str,
        pattern: str | None,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            # Build pattern filter if provided (convert * to % for ILIKE)
            pattern_clause = ""
            params: list[Any] = [bank_id]
            if pattern:
                sql_pattern = pattern.replace("*", "%")
                pattern_clause = "AND tag ILIKE $2"
                params.append(sql_pattern)

            # Get backend-specific SQL fragments for tag listing
            tag_parts = self._backend.ops.build_tag_listing_parts(fq_table(table))
            tag_source = tag_parts.tag_source
            non_empty_check = tag_parts.non_empty_check
            tag_col = tag_parts.tag_col
            bank_prefix = tag_parts.bank_prefix

            tag_pattern_clause = pattern_clause.replace("tag", tag_col) if tag_col != "tag" else pattern_clause

            # Get total count of distinct tags matching pattern
            total_row = await conn.fetchrow(
                f"""
                SELECT COUNT(DISTINCT {tag_col}) as total
                FROM {tag_source}
                WHERE {bank_prefix}bank_id = $1 {non_empty_check}
                {tag_pattern_clause}
                """,
                *params,
            )
            total = total_row["total"] if total_row else 0

            limit_param = len(params) + 1
            offset_param = len(params) + 2
            params.extend([limit, offset])

            rows = await conn.fetch(
                f"""
                SELECT {tag_col} as tag, COUNT(*) as count
                FROM {tag_source}
                WHERE {bank_prefix}bank_id = $1 {non_empty_check}
                {tag_pattern_clause}
                GROUP BY {tag_col}
                ORDER BY count DESC, {tag_col} ASC
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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_entity_state", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        return EntityState(entity_id=entity_id, canonical_name=entity_name, observations=[])

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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_bank_stats", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
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

            # Link stats — filter on ml.bank_id (indexed) instead of joining through mu.bank_id.
            # With the idx_memory_links_bank_link_type index this turns a full-table hash join
            # into an indexed scan + PK lookups.  link_counts and link_counts_by_fact_type are
            # derived in Python from the breakdown.
            link_breakdown_stats = await conn.fetch(
                f"""
                SELECT mu.fact_type, ml.link_type, COUNT(*) as count
                FROM {fq_table("memory_links")} ml
                JOIN {fq_table("memory_units")} mu ON ml.from_unit_id = mu.id
                WHERE ml.bank_id = $1
                GROUP BY mu.fact_type, ml.link_type
                """,
                bank_id,
            )

            link_counts: dict[str, int] = {}
            link_counts_by_fact_type: dict[str, int] = {}
            for row in link_breakdown_stats:
                link_counts[row["link_type"]] = link_counts.get(row["link_type"], 0) + row["count"]
                link_counts_by_fact_type[row["fact_type"]] = (
                    link_counts_by_fact_type.get(row["fact_type"], 0) + row["count"]
                )

            ops_stats = await conn.fetch(
                f"""
                SELECT status, COUNT(*) as count
                FROM {fq_table("async_operations")}
                WHERE bank_id = $1
                GROUP BY status
                """,
                bank_id,
            )
            doc_count_row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM {fq_table('documents')} WHERE bank_id = $1",
                bank_id,
            )
            consolidation_row = await conn.fetchrow(
                f"""
                SELECT
                    MAX(consolidated_at) as last_consolidated_at,
                    COUNT(*) FILTER (WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')) as pending,
                    COUNT(*) FILTER (WHERE consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')) as failed
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                """,
                bank_id,
            )

            node_counts = {row["fact_type"]: row["count"] for row in node_stats}
            ops_by_status = {row["status"]: row["count"] for row in ops_stats}
            last_consolidated_at = consolidation_row["last_consolidated_at"] if consolidation_row else None

            return {
                "bank_id": bank_id,
                "node_counts": node_counts,
                "link_counts": link_counts,
                "link_counts_by_fact_type": link_counts_by_fact_type,
                "link_breakdown": [
                    {"fact_type": row["fact_type"], "link_type": row["link_type"], "count": row["count"]}
                    for row in link_breakdown_stats
                ],
                "operations": ops_by_status,
                "total_documents": doc_count_row["count"] if doc_count_row else 0,
                "last_consolidated_at": last_consolidated_at.isoformat() if last_consolidated_at else None,
                "pending_consolidation": consolidation_row["pending"] if consolidation_row else 0,
                "failed_consolidation": consolidation_row["failed"] if consolidation_row else 0,
                "total_observations": node_counts.get("observation", 0),
            }

    async def get_memories_timeseries(
        self,
        bank_id: str,
        *,
        period: str,
        request_context: "RequestContext",
        time_field: str = "created_at",
    ) -> dict[str, Any]:
        """Memory ingestion bucketed by time, broken down by fact_type.

        Always returns the full expected bucket set for the period so the
        chart line is continuous (empty buckets show as zeros). Buckets are
        anchored on UTC boundaries — we do this (rather than the PG session
        timezone) so the API response is deterministic regardless of where
        the database is deployed, and so the control-plane chart can match
        buckets by ISO key on the client side.

        ``time_field`` selects which timestamp column drives the bucket
        assignment. ``created_at`` (default) shows when records were ingested;
        ``mentioned_at`` / ``occurred_start`` reflect the event time carried
        over from the source data, which is what you want for migrated or
        backfilled corpora. For the event-time columns we fall back to
        ``created_at`` per-row via ``COALESCE`` so records that lack an event
        timestamp still show up in the chart.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_memories_timeseries", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        cfg = _MEMORIES_TIMESERIES_PERIODS.get(period) or _MEMORIES_TIMESERIES_PERIODS["7d"]
        if period not in _MEMORIES_TIMESERIES_PERIODS:
            period = "7d"

        # Whitelist time_field — it is interpolated into SQL, must never come from untrusted input.
        _ALLOWED_TIME_FIELDS = ("created_at", "mentioned_at", "occurred_start")
        if time_field not in _ALLOWED_TIME_FIELDS:
            time_field = "created_at"
        # COALESCE onto created_at for event-time fields so null rows don't vanish.
        bucket_expr = time_field if time_field == "created_at" else f"COALESCE({time_field}, created_at)"

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"""
                SELECT date_trunc('{cfg.trunc}', {bucket_expr} AT TIME ZONE 'UTC') AS bucket,
                       fact_type, COUNT(*) AS count
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1
                  AND {bucket_expr} >= now() - interval '{cfg.interval}'
                GROUP BY bucket, fact_type
                ORDER BY bucket
                """,
                bank_id,
            )

        # Build the canonical bucket list anchored on the most recent UTC boundary.
        # Use tz-aware UTC throughout so serialized ISO strings include a `+00:00`
        # offset; a naive ISO (`2026-04-18T00:00:00`) would be parsed by browsers
        # as local time per ECMA-262, producing an off-by-timezone display.
        now_utc = datetime.now(timezone.utc)
        if cfg.trunc == "minute":
            end = now_utc.replace(second=0, microsecond=0)
        elif cfg.trunc == "hour":
            end = now_utc.replace(minute=0, second=0, microsecond=0)
        else:
            end = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

        buckets: list[MemoryTimeseriesBucketData] = []
        by_iso: dict[str, MemoryTimeseriesBucketData] = {}
        for i in range(cfg.count):
            t = end - cfg.step * (cfg.count - 1 - i)
            entry = MemoryTimeseriesBucketData(time=t.isoformat())
            buckets.append(entry)
            by_iso[entry.time] = entry

        for row in rows:
            # asyncpg hands us a tz-aware datetime when the column is timestamptz;
            # ensure UTC so the ISO key matches `by_iso` (also tz-aware UTC).
            bucket_dt = row["bucket"]
            if bucket_dt.tzinfo is None:
                bucket_dt = bucket_dt.replace(tzinfo=timezone.utc)
            else:
                bucket_dt = bucket_dt.astimezone(timezone.utc)
            entry = by_iso.get(bucket_dt.isoformat())
            if entry is None:
                # Row fell outside the requested window (clock skew / edge case).
                continue
            ft = row["fact_type"]
            if ft == "world":
                entry.world += row["count"]
            elif ft == "experience":
                entry.experience += row["count"]
            elif ft == "observation":
                entry.observation += row["count"]

        return {
            "bank_id": bank_id,
            "period": period,
            "trunc": cfg.trunc,
            "time_field": time_field,
            "buckets": [b.as_dict() for b in buckets],
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
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_entity", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
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

        return {
            "id": str(entity_row["id"]),
            "canonical_name": entity_row["canonical_name"],
            "mention_count": entity_row["mention_count"],
            "first_seen": entity_row["first_seen"].isoformat() if entity_row["first_seen"] else None,
            "last_seen": entity_row["last_seen"].isoformat() if entity_row["last_seen"] else None,
            "metadata": entity_row["metadata"] or {},
            "observations": [],
        }

    def _parse_observations(self, observations_raw: list):
        """Parse raw observation dicts into typed Observation models.

        Returns list of Observation models with computed trend/evidence_span/evidence_count.
        """
        from .reflect.observations import Observation, ObservationEvidence

        observations: list[Observation] = []
        for obs in observations_raw:
            if not isinstance(obs, dict):
                continue

            try:
                parsed = Observation(
                    title=obs.get("title", ""),
                    content=obs.get("content", ""),
                    evidence=[
                        ObservationEvidence(
                            memory_id=ev.get("memory_id", ""),
                            quote=ev.get("quote", ""),
                            relevance=ev.get("relevance", ""),
                            timestamp=ev.get("timestamp"),
                        )
                        for ev in obs.get("evidence", [])
                        if isinstance(ev, dict)
                    ],
                    created_at=obs.get("created_at"),
                )
                observations.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse observation: {e}")
                continue

        return observations

    async def _count_memories_since(
        self,
        bank_id: str,
        since_timestamp: str | None,
        backend=None,
    ) -> int:
        """
        Count memories created after a given timestamp.

        Args:
            bank_id: Bank identifier
            since_timestamp: ISO timestamp string. If None, returns total count.
            backend: Optional database backend (uses default if not provided)

        Returns:
            Number of memories created since the timestamp
        """
        if backend is None:
            backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            if since_timestamp:
                # Parse the timestamp
                from datetime import datetime

                try:
                    ts = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
                except ValueError:
                    # Invalid timestamp, return total count
                    ts = None

                if ts:
                    count = await conn.fetchval(
                        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND created_at > $2",
                        bank_id,
                        ts,
                    )
                    return count or 0

            # No timestamp or invalid, return total count
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1",
                bank_id,
            )
            return count or 0

    async def _delete_stale_observations_for_memories(
        self,
        conn,
        bank_id: str,
        fact_ids: list[str],
    ) -> int:
        """Thin wrapper that delegates to ``fact_storage.delete_stale_observations_for_memories``.

        Kept on the engine class so the existing call sites here and the
        retain pipeline both end up running the same SQL. See the free
        function for the full contract.
        """
        from .retain.fact_storage import delete_stale_observations_for_memories

        return await delete_stale_observations_for_memories(conn, bank_id, fact_ids, ops=self._backend.ops)

    # =========================================================================
    # MENTAL MODELS (CONSOLIDATED) - Read-only access to auto-consolidated mental models
    # =========================================================================

    async def list_mental_models_consolidated(
        self,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        tags_match: str = "any",
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List auto-consolidated observations for a bank.

        Observations are stored in memory_units with fact_type='observation'.
        They are automatically created and updated by the consolidation engine.

        Args:
            bank_id: Bank identifier
            tags: Optional tags to filter by
            tags_match: How to match tags - 'any', 'all', or 'exact'
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication

        Returns:
            List of observation dicts
        """
        await self._authenticate_tenant(request_context)
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            # Build tag filter
            tag_filter = ""
            params: list[Any] = [bank_id, limit, offset]
            if tags:
                if tags_match == "all":
                    tag_filter = " AND tags @> $4::varchar[]"
                elif tags_match == "exact":
                    tag_filter = " AND tags = $4::varchar[]"
                else:  # any
                    tag_filter = " AND tags && $4::varchar[]"
                params.append(tags)

            rows = await conn.fetch(
                f"""
                SELECT id, bank_id, text, proof_count, history, tags, source_memory_ids, created_at, updated_at
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND fact_type = 'observation' {tag_filter}
                ORDER BY updated_at DESC NULLS LAST
                LIMIT $2 OFFSET $3
                """,
                *params,
            )

            return [self._row_to_observation_consolidated(row) for row in rows]

    async def get_observation_consolidated(
        self,
        bank_id: str,
        observation_id: str,
        *,
        include_source_memories: bool = True,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Get a single observation by ID.

        Args:
            bank_id: Bank identifier
            observation_id: Observation ID
            include_source_memories: Whether to include full source memory details
            request_context: Request context for authentication

        Returns:
            Observation dict or None if not found
        """
        await self._authenticate_tenant(request_context)
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, bank_id, text, proof_count, history, tags, source_memory_ids, created_at, updated_at
                FROM {fq_table("memory_units")}
                WHERE bank_id = $1 AND id = $2 AND fact_type = 'observation'
                """,
                bank_id,
                observation_id,
            )

            if not row:
                return None

            result = self._row_to_observation_consolidated(row)

            # Fetch source memories if requested and source_memory_ids exist
            if include_source_memories and result.get("source_memory_ids"):
                source_ids = [uuid.UUID(sid) if isinstance(sid, str) else sid for sid in result["source_memory_ids"]]
                source_rows = await conn.fetch(
                    f"""
                    SELECT id, text, fact_type, context, occurred_start, mentioned_at
                    FROM {fq_table("memory_units")}
                    WHERE id = ANY($1::uuid[])
                    ORDER BY mentioned_at DESC NULLS LAST
                    """,
                    source_ids,
                )
                result["source_memories"] = [
                    {
                        "id": str(r["id"]),
                        "text": r["text"],
                        "type": r["fact_type"],
                        "context": r["context"],
                        "occurred_start": r["occurred_start"].isoformat() if r["occurred_start"] else None,
                        "mentioned_at": r["mentioned_at"].isoformat() if r["mentioned_at"] else None,
                    }
                    for r in source_rows
                ]

            return result

    def _row_to_observation_consolidated(self, row: Any) -> dict[str, Any]:
        """Convert a database row to an observation dict."""
        import json

        history = row["history"]
        if isinstance(history, str):
            history = json.loads(history)
        elif history is None:
            history = []

        # Convert source_memory_ids to strings
        source_memory_ids = row.get("source_memory_ids") or []
        source_memory_ids = [str(sid) for sid in source_memory_ids]

        return {
            "id": str(row["id"]),
            "bank_id": row["bank_id"],
            "text": row["text"],
            "proof_count": row["proof_count"] or 1,
            "history": history,
            "tags": row["tags"] or [],
            "source_memory_ids": source_memory_ids,
            "source_memories": [],  # Populated separately when fetching full details
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    # =========================================================================
    # MENTAL MODELS CRUD
    # =========================================================================

    async def list_mental_models(
        self,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        tags_match: str = "any",
        detail: str = "full",
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List pinned mental models for a bank.

        Args:
            bank_id: Bank identifier
            tags: Optional tags to filter by
            tags_match: How to match tags - 'any', 'all', or 'exact'
            detail: Detail level - 'metadata', 'content', or 'full'
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication

        Returns:
            List of pinned mental model dicts
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_mental_models", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            # Build tag filter
            tag_filter = ""
            params: list[Any] = [bank_id, limit, offset]
            if tags:
                if tags_match == "all":
                    tag_filter = " AND tags @> $4::varchar[]"
                elif tags_match == "exact":
                    tag_filter = " AND tags = $4::varchar[]"
                else:  # any
                    tag_filter = " AND tags && $4::varchar[]"
                params.append(tags)

            rows = await conn.fetch(
                f"""
                SELECT id, bank_id, name, source_query, content, tags,
                       last_refreshed_at, created_at, reflect_response,
                       max_tokens, trigger, structured_content
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1 {tag_filter}
                ORDER BY last_refreshed_at DESC
                LIMIT $2 OFFSET $3
                """,
                *params,
            )

            return [self._row_to_mental_model(row, detail=detail) for row in rows]

    async def get_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        detail: str = "full",
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Get a single pinned mental model by ID.

        Args:
            bank_id: Bank identifier
            mental_model_id: Pinned mental model UUID
            detail: Detail level - 'metadata', 'content', or 'full'
            request_context: Request context for authentication

        Returns:
            Pinned mental model dict or None if not found
        """
        await self._authenticate_tenant(request_context)

        # Pre-operation validation (credit check / usage metering)
        if self._operation_validator:
            from hindsight_api.extensions.operation_validator import MentalModelGetContext

            ctx = MentalModelGetContext(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            await self._validate_operation(self._operation_validator.validate_mental_model_get(ctx))

        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, bank_id, name, source_query, content, tags,
                       last_refreshed_at, created_at, reflect_response,
                       max_tokens, trigger, structured_content
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mental_model_id,
            )

            result = self._row_to_mental_model(row, detail=detail) if row else None
            if result is not None and detail == "full":
                result["is_stale"] = await self.compute_mental_model_is_stale(conn, bank_id, row)

        # Post-operation hook (usage recording)
        if result and self._operation_validator:
            from hindsight_api.extensions.operation_validator import MentalModelGetResult

            content = result.get("content", "")
            output_tokens = len(content) // 4 if content else 0

            result_ctx = MentalModelGetResult(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
                output_tokens=output_tokens,
                success=True,
            )
            try:
                await self._operation_validator.on_mental_model_get_complete(result_ctx)
            except Exception as hook_err:
                logger.warning(f"Post-mental-model-get hook error (non-fatal): {hook_err}")

        return result

    async def get_mental_model_history(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        request_context: "RequestContext",
    ) -> list[dict] | None:
        """Get the refresh history of a mental model.

        Returns None if the mental model is not found.
        Returns a list of history entries (most recent first), each with previous_content and changed_at.

        """
        await self._authenticate_tenant(request_context)
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT history
                FROM {fq_table("mental_models")}
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                mental_model_id,
            )
            if row is None:
                return None
            raw_history = row["history"]
            if isinstance(raw_history, str):
                raw_history = json.loads(raw_history)
            if not raw_history:
                return []
            return list(reversed(raw_history))

    async def create_mental_model(
        self,
        bank_id: str,
        name: str,
        source_query: str,
        content: str,
        *,
        mental_model_id: str | None = None,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
        trigger: dict[str, Any] | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Create a new pinned mental model.

        Args:
            bank_id: Bank identifier
            name: Human-readable name for the mental model
            source_query: The query that generated this mental model
            content: The synthesized content
            mental_model_id: Optional UUID for the mental model (auto-generated if not provided)
            tags: Optional tags for scoped visibility
            max_tokens: Token limit for content generation during refresh
            trigger: Trigger settings (e.g., refresh_after_consolidation)
            request_context: Request context for authentication

        Returns:
            The created pinned mental model dict
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="create_mental_model", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        # Generate embedding for the content
        embedding_text = f"{name} {content}"
        embedding = await embedding_utils.generate_embeddings_batch(self.embeddings, [embedding_text])
        # Convert embedding to string for asyncpg vector type
        embedding_str = str(embedding[0]) if embedding else None

        if not mental_model_id:
            mental_model_id = f"mm-{uuid.uuid4().hex}"

        async with acquire_with_retry(backend) as conn:
            if mental_model_id:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {fq_table("mental_models")}
                    (id, bank_id, subtype, name, description, source_query, content, embedding, tags, max_tokens, trigger)
                    VALUES ($1, $2, 'pinned', $3, ' ', $4, $5, $6, $7, COALESCE($8, 2048), COALESCE($9, '{{"refresh_after_consolidation": false}}'::jsonb))
                    RETURNING id, bank_id, name, source_query, content, tags,
                              last_refreshed_at, created_at, reflect_response,
                              max_tokens, trigger, structured_content
                    """,
                    mental_model_id,
                    bank_id,
                    name,
                    source_query,
                    content,
                    embedding_str,
                    tags or [],
                    max_tokens,
                    json.dumps(trigger) if trigger else None,
                )
            else:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {fq_table("mental_models")}
                    (bank_id, subtype, name, description, source_query, content, embedding, tags, max_tokens, trigger)
                    VALUES ($1, 'pinned', $2, ' ', $3, $4, $5, $6, COALESCE($7, 2048), COALESCE($8, '{{"refresh_after_consolidation": false}}'::jsonb))
                    RETURNING id, bank_id, name, source_query, content, tags,
                              last_refreshed_at, created_at, reflect_response,
                              max_tokens, trigger, structured_content
                    """,
                    bank_id,
                    name,
                    source_query,
                    content,
                    embedding_str,
                    tags or [],
                    max_tokens,
                    json.dumps(trigger) if trigger else None,
                )

        logger.info(f"[MENTAL_MODELS] Created pinned mental model '{name}' for bank {bank_id}")
        return self._row_to_mental_model(row)

    async def refresh_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Refresh a pinned mental model by re-running its source query.

        This method:
        1. Gets the pinned mental model
        2. Runs the source_query through reflect
        3. Updates the content with the new synthesis
        4. Updates last_refreshed_at

        Args:
            bank_id: Bank identifier
            mental_model_id: Pinned mental model UUID
            request_context: Request context for authentication

        Returns:
            Updated pinned mental model dict or None if not found
        """
        await self._authenticate_tenant(request_context)

        # Get the current mental model
        mental_model = await self.get_mental_model(bank_id, mental_model_id, request_context=request_context)
        if not mental_model:
            return None

        # Create parent span for mental model refresh operation
        with create_operation_span("mental_model_refresh", bank_id):
            # Read reflect options from trigger (if stored)
            trigger_data = mental_model.get("trigger") or {}
            fact_types = trigger_data.get("fact_types")
            exclude_mental_models = trigger_data.get("exclude_mental_models", False)
            stored_exclude_ids: list[str] = trigger_data.get("exclude_mental_model_ids") or []
            recall_include_chunks_override = trigger_data.get("include_chunks")
            recall_max_tokens_override = trigger_data.get("recall_max_tokens")
            recall_chunks_max_tokens_override = trigger_data.get("recall_chunks_max_tokens")
            refresh_mode = trigger_data.get("mode") or "full"

            current_content = (mental_model.get("content") or "").strip()
            current_source_query = mental_model["source_query"]

            # Delta mode requires both existing content and an unchanged source_query.
            # When either condition fails, we fall back to a full regeneration: a
            # surgical edit has nothing to edit, or the topic itself has shifted.
            # The tracking column is only read when delta is requested so full-mode
            # refreshes don't pay for an extra query (and mock-based unit tests that
            # stub out the DB don't hit an unexpected pool access).
            use_delta = False
            stored_structured_content: dict[str, Any] | None = None
            if refresh_mode == "delta" and current_content:
                backend = await self._get_backend()
                async with acquire_with_retry(backend) as conn:
                    tracking_row = await conn.fetchrow(
                        f"SELECT last_refreshed_source_query, structured_content "
                        f"FROM {fq_table('mental_models')} "
                        f"WHERE bank_id = $1 AND id = $2",
                        bank_id,
                        mental_model_id,
                    )
                last_refreshed_source_query: str | None = (
                    tracking_row["last_refreshed_source_query"] if tracking_row else None
                )
                # Use delta when the user has content to anchor on AND the topic
                # hasn't shifted. The first delta refresh (no tracking row yet)
                # still uses the existing markdown as the baseline — users who
                # write a doc and then enable delta mode expect their content to
                # be the starting point, not discarded by a one-time full rebuild.
                use_delta = last_refreshed_source_query is None or last_refreshed_source_query == current_source_query
                if tracking_row is not None:
                    raw_struct = tracking_row["structured_content"]
                    if isinstance(raw_struct, str):
                        try:
                            stored_structured_content = json.loads(raw_struct)
                        except json.JSONDecodeError:
                            stored_structured_content = None
                    else:
                        stored_structured_content = raw_struct

            tag_filtering = _resolve_refresh_tag_filtering(mental_model.get("tags"), trigger_data)

            # Run reflect with the source query, excluding the mental model being refreshed
            # Skip creating a nested "hindsight.reflect" span since we already have "hindsight.mental_model_refresh"
            # Build context to guide the reflect agent: tell it what this mental
            # model is about so it stays on-topic and produces high-quality content.
            mm_name = mental_model.get("name") or mental_model_id
            refresh_context = (
                f'You are writing a document called "{mm_name}". '
                f"ONLY include content that directly answers the topic query. "
                f"Discard observations that are tangential or off-topic — retrieval may return "
                f"loosely related content that does not belong in this document.\n\n"
                f"Quality guidelines:\n"
                f"- Preserve concrete examples, before/after pairs, and sample sentences "
                f"from the observations. These teach more than abstract rules.\n"
                f"- If observations contain illustrative examples (e.g. ✅/❌ pairs, "
                f"rewrites, sample phrases), include them in your answer.\n"
                f"- Structure the document around the topic, not around the sources."
            )

            reflect_kwargs: dict[str, Any] = dict(
                bank_id=bank_id,
                query=mental_model["source_query"],
                context=refresh_context,
                request_context=request_context,
                tags=tag_filtering.tags,
                tags_match=tag_filtering.tags_match,
                tag_groups=tag_filtering.tag_groups,
                fact_types=fact_types,
                exclude_mental_models=exclude_mental_models,
                exclude_mental_model_ids=list({*stored_exclude_ids, mental_model_id}),
                recall_include_chunks=recall_include_chunks_override,
                recall_max_tokens_override=recall_max_tokens_override,
                recall_chunks_max_tokens_override=recall_chunks_max_tokens_override,
                _skip_span=True,
            )
            # Forward the per-model max_tokens so the final synthesis is capped at the
            # user-configured limit rather than the reflect_async default.
            stored_max_tokens = mental_model.get("max_tokens")
            if stored_max_tokens is not None:
                reflect_kwargs["max_tokens"] = stored_max_tokens

            # Delta mode: scope recall to memories created since the last refresh
            # so the agentic loop only retrieves genuinely new information.
            if use_delta:
                last_refreshed_at_raw = mental_model.get("last_refreshed_at")
                if last_refreshed_at_raw is not None:
                    if isinstance(last_refreshed_at_raw, str):
                        reflect_kwargs["created_after"] = datetime.fromisoformat(last_refreshed_at_raw)
                    else:
                        reflect_kwargs["created_after"] = last_refreshed_at_raw

            reflect_result = await self.reflect_async(**reflect_kwargs)

            # Build reflect_response payload to store
            # based_on contains MemoryFact objects for most types, but plain dicts for directives
            based_on_serialized_payload: dict[str, list[dict[str, Any]]] = {}
            for fact_type, facts in reflect_result.based_on.items():
                serialized_facts = []
                for fact in facts:
                    if isinstance(fact, dict):
                        # Plain dict (e.g., directives with id, name, content)
                        serialized_facts.append(
                            {
                                "id": str(fact["id"]),
                                "text": fact.get("text", fact.get("content", fact.get("name", ""))),
                                "type": fact_type,
                                "context": fact.get("context", None),
                            }
                        )
                    else:
                        # MemoryFact object with .id, .text, .context attributes
                        serialized_facts.append(
                            {
                                "id": str(fact.id),
                                "text": fact.text,
                                "type": fact_type,
                                "context": fact.context,
                            }
                        )
                based_on_serialized_payload[fact_type] = serialized_facts

            # In delta mode, based_on must accumulate: the mental model is
            # grounded on ALL facts ever used, not just the latest delta's new
            # ones. Merge previous based_on with current, deduplicating by id.
            if use_delta:
                prev_rr = mental_model.get("reflect_response") or {}
                prev_based_on = prev_rr.get("based_on") or {}
                for ftype, prev_facts in prev_based_on.items():
                    if not isinstance(prev_facts, list):
                        continue
                    new_ids = {f["id"] for f in based_on_serialized_payload.get(ftype, [])}
                    carried = [f for f in prev_facts if isinstance(f, dict) and f.get("id") not in new_ids]
                    if carried:
                        based_on_serialized_payload.setdefault(ftype, []).extend(carried)

            reflect_response_payload = {
                "text": reflect_result.text,
                "based_on": based_on_serialized_payload,
                "mental_models": [],  # Mental models are included in based_on["mental-models"]
            }

            # Delta-mode path: emit structured operations against the existing
            # structured doc, apply them, then re-render to markdown. Sections
            # not mentioned by any operation are physically untouched, so prose
            # drift is structurally impossible. Falls back to the full candidate
            # markdown if either the structuring or the LLM op call fails.
            from .reflect.delta_ops import (
                DeltaOperationList,
                apply_operations,
            )
            from .reflect.prompts import (
                STRUCTURED_DELTA_SYSTEM_PROMPT,
                build_structured_delta_prompt,
            )
            from .reflect.structured_doc import (
                StructuredDocument,
                parse_markdown,
                render_document,
            )

            final_content = reflect_result.text
            final_structured: StructuredDocument | None = None
            delta_applied = False
            applied_ops_summary: list[dict[str, Any]] = []
            skipped_ops_summary: list[dict[str, Any]] = []

            if use_delta:
                # Use the previously stored structured doc when available; otherwise
                # parse the existing markdown so the very first delta refresh can
                # still operate without waiting for a full rebuild.
                try:
                    if stored_structured_content is not None:
                        current_doc = StructuredDocument.model_validate(stored_structured_content)
                    else:
                        current_doc = parse_markdown(current_content)
                except Exception as exc:
                    logger.warning(
                        f"[MENTAL_MODELS] Could not load structured doc for {mental_model_id} "
                        f"({exc}); falling back to full synthesis"
                    )
                    current_doc = None

                if current_doc is not None:
                    supporting_facts: list[dict[str, Any]] = []
                    for _ftype, facts in based_on_serialized_payload.items():
                        supporting_facts.extend(facts)

                    # No new facts since last refresh — skip the delta LLM call
                    # and preserve existing content unchanged.
                    if not supporting_facts:
                        logger.info(
                            f"[MENTAL_MODELS] Delta refresh for {mental_model_id}: "
                            "no new facts found, preserving content"
                        )
                        reflect_response_payload["delta_applied"] = False
                        reflect_response_payload["delta_skipped_reason"] = "no_new_facts"
                        return await self.update_mental_model(
                            bank_id,
                            mental_model_id,
                            reflect_response=reflect_response_payload,
                            last_refreshed_source_query=current_source_query,
                            request_context=request_context,
                        )

                    # Op JSON is denser than the rendered markdown — each op
                    # carries the section_id, op type, and a full block payload
                    # whose ``text`` may quote the original passage. Budget 1.5×
                    # the document cap so the model can express several edits
                    # without truncating mid-string. The cap is also surfaced in
                    # the prompt so the model can self-trim if needed.
                    doc_max_tokens = mental_model.get("max_tokens") or 2048
                    delta_max_tokens = max(2048, int(doc_max_tokens * 1.5))
                    user_prompt = build_structured_delta_prompt(
                        current_document_json=current_doc.model_dump_json(indent=2),
                        candidate_markdown=reflect_result.text,
                        supporting_facts=supporting_facts,
                        source_query=current_source_query,
                        max_output_tokens=delta_max_tokens,
                    )
                    try:
                        # Text-mode call (not structured-output) because Pydantic's
                        # discriminated-union JSON schema isn't accepted by every
                        # provider — Gemini in particular rejects ``oneOf`` /
                        # ``discriminator``. We parse + validate the JSON ourselves
                        # so the same prompt works against any LLM.
                        raw = await self._reflect_llm_config.call(
                            messages=[
                                {"role": "system", "content": STRUCTURED_DELTA_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            max_completion_tokens=delta_max_tokens,
                            temperature=0.0,
                            scope="mental_model_delta_ops",
                        )
                        op_list: DeltaOperationList
                        if isinstance(raw, DeltaOperationList):
                            op_list = raw
                        elif isinstance(raw, dict):
                            op_list = DeltaOperationList.model_validate(raw)
                        else:
                            text = (raw or "").strip()
                            # Strip optional fenced code block.
                            if text.startswith("```"):
                                text = text.split("\n", 1)[1] if "\n" in text else ""
                                if text.endswith("```"):
                                    text = text[:-3].rstrip()
                            op_list = DeltaOperationList.model_validate_json(text)
                        outcome = apply_operations(current_doc, op_list.operations)
                        final_structured = outcome.document
                        final_content = render_document(outcome.document)
                        applied_ops_summary = outcome.applied
                        skipped_ops_summary = outcome.skipped
                        delta_applied = True
                        logger.info(
                            f"[MENTAL_MODELS] Delta refresh for {mental_model_id}: "
                            f"applied {len(applied_ops_summary)} op(s), "
                            f"skipped {len(skipped_ops_summary)}"
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[MENTAL_MODELS] Structured delta failed for {mental_model_id} "
                            f"({exc}); falling back to full synthesis"
                        )

                reflect_response_payload["delta_applied"] = delta_applied
                if delta_applied:
                    reflect_response_payload["delta_operations_applied"] = applied_ops_summary
                    reflect_response_payload["delta_operations_skipped"] = skipped_ops_summary

            # Refuse to overwrite existing content with an empty render.
            # The reflect agent can return an empty answer (small models, all
            # tool-call retries failing, transient provider errors) and the
            # delta merge can also fall back to that empty candidate. Writing
            # "" to the DB would destroy the working document — and since the
            # next refresh sees current_content == "" it would also skip the
            # delta path, compounding the problem. Preserve what's there and
            # surface the failure via reflect_response.
            if not final_content.strip() and current_content:
                logger.warning(
                    f"[MENTAL_MODELS] Refresh for {mental_model_id} produced empty content; "
                    "preserving previous content (likely an upstream LLM failure)"
                )
                reflect_response_payload["refresh_skipped"] = "empty_candidate"
                # Persist the reflect_response (so the failure is auditable) and
                # the source-query tracking, but do NOT touch content/structured.
                return await self.update_mental_model(
                    bank_id,
                    mental_model_id,
                    reflect_response=reflect_response_payload,
                    last_refreshed_source_query=current_source_query,
                    request_context=request_context,
                )

            # When delta is not applied (full mode, or delta fallback), parse the
            # candidate markdown so the next refresh has a structured baseline to
            # operate against.
            if final_structured is None:
                try:
                    final_structured = parse_markdown(final_content)
                except Exception as exc:
                    logger.warning(
                        f"[MENTAL_MODELS] Could not parse final markdown into structured form "
                        f"for {mental_model_id} ({exc}); leaving structured_content unchanged"
                    )

            # Update the mental model with new content and reflect_response.
            # Passing last_refreshed_source_query records the query used for this
            # refresh so a future delta-mode run can detect a topic change.
            return await self.update_mental_model(
                bank_id,
                mental_model_id,
                content=final_content,
                reflect_response=reflect_response_payload,
                last_refreshed_source_query=current_source_query,
                structured_content=(final_structured.model_dump() if final_structured is not None else None),
                request_context=request_context,
            )

    async def update_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        name: str | None = None,
        content: str | None = None,
        source_query: str | None = None,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        trigger: dict[str, Any] | None = None,
        reflect_response: dict[str, Any] | None = None,
        last_refreshed_source_query: str | None = None,
        structured_content: dict[str, Any] | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Update a pinned mental model.

        Args:
            bank_id: Bank identifier
            mental_model_id: Pinned mental model UUID
            name: New name (if changing)
            content: New content (if changing)
            source_query: New source query (if changing)
            max_tokens: New max tokens (if changing)
            tags: New tags (if changing)
            trigger: New trigger settings (if changing)
            reflect_response: Full reflect API response payload (if changing)
            request_context: Request context for authentication

        Returns:
            Updated pinned mental model dict or None if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_mental_model", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            # If content is changing, fetch current content + reflect_response to record history
            previous_content: str | None = None
            previous_reflect_response: dict[str, Any] | None = None
            if content is not None:
                current_row = await conn.fetchrow(
                    f"SELECT content, reflect_response FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    mental_model_id,
                )
                if current_row:
                    previous_content = current_row["content"]
                    raw_rr = current_row["reflect_response"]
                    if isinstance(raw_rr, str):
                        previous_reflect_response = json.loads(raw_rr) if raw_rr else None
                    else:
                        previous_reflect_response = raw_rr

            # Build dynamic update
            updates = []
            params: list[Any] = [bank_id, mental_model_id]
            param_idx = 3

            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1

            if content is not None:
                updates.append(f"content = ${param_idx}")
                params.append(content)
                param_idx += 1
                updates.append("last_refreshed_at = NOW()")
                # Record history entry with the previous content
                if get_config().enable_mental_model_history:
                    history_entry = json.dumps(
                        [
                            {
                                "previous_content": previous_content,
                                "previous_reflect_response": previous_reflect_response,
                                "changed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        ]
                    )
                    updates.append(f"history = COALESCE(history, '[]'::jsonb) || ${param_idx}::jsonb")
                    params.append(history_entry)
                    param_idx += 1
                # Also update embedding (convert to string for asyncpg vector type)
                embedding_text = f"{name or ''} {content}"
                embedding = await embedding_utils.generate_embeddings_batch(self.embeddings, [embedding_text])
                if embedding:
                    updates.append(f"embedding = ${param_idx}")
                    params.append(str(embedding[0]))
                    param_idx += 1

            if reflect_response is not None:
                updates.append(f"reflect_response = ${param_idx}")
                params.append(json.dumps(reflect_response))
                param_idx += 1

            if source_query is not None:
                updates.append(f"source_query = ${param_idx}")
                params.append(source_query)
                param_idx += 1

            if max_tokens is not None:
                updates.append(f"max_tokens = ${param_idx}")
                params.append(max_tokens)
                param_idx += 1

            if tags is not None:
                updates.append(f"tags = ${param_idx}")
                params.append(tags)
                param_idx += 1

            if trigger is not None:
                updates.append(f"trigger = ${param_idx}")
                params.append(json.dumps(trigger))
                param_idx += 1

            if last_refreshed_source_query is not None:
                updates.append(f"last_refreshed_source_query = ${param_idx}")
                params.append(last_refreshed_source_query)
                param_idx += 1

            if structured_content is not None:
                updates.append(f"structured_content = ${param_idx}")
                params.append(json.dumps(structured_content))
                param_idx += 1

            if not updates:
                return None

            query = f"""
                UPDATE {fq_table("mental_models")}
                SET {", ".join(updates)}
                WHERE bank_id = $1 AND id = $2
                RETURNING id, bank_id, name, source_query, content, tags,
                          last_refreshed_at, created_at, reflect_response,
                          max_tokens, trigger, structured_content
            """

            row = await conn.fetchrow(query, *params)

            return self._row_to_mental_model(row) if row else None

    async def delete_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        request_context: "RequestContext",
    ) -> bool:
        """Delete a pinned mental model.

        Args:
            bank_id: Bank identifier
            mental_model_id: Pinned mental model UUID
            request_context: Request context for authentication

        Returns:
            True if deleted, False if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="delete_mental_model", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            result = await conn.execute(
                f"DELETE FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                mental_model_id,
            )

        return result == "DELETE 1"

    async def compute_mental_model_is_stale(
        self,
        conn,
        bank_id: str,
        mm_row: Any,
    ) -> bool:
        """Check whether a mental model is out of date.

        A mental model is stale when a memory in its **scope** has been ingested after
        ``last_refreshed_at``. The scope is defined by the model's ``tags`` +
        ``trigger.tags_match`` semantics (``any`` / ``all`` / ``any_strict`` /
        ``all_strict``, matching recall semantics) and its ``trigger.fact_types`` filter
        when set. Memories still pending consolidation are included because they are
        already rows in ``memory_units``; no separate ``pending_consolidation`` signal is
        needed — it would bypass the tag scope and falsely flag unrelated MMs.

        Untagged mental model defaults to ``tags_match="any"`` so it matches any memory
        ingested in the bank (what a user would expect for a "global" MM).
        """
        from hindsight_api.engine.search.tags import _parse_tags_match

        def _get(key: str) -> Any:
            if isinstance(mm_row, dict):
                return mm_row.get(key)
            try:
                return mm_row[key]
            except (KeyError, TypeError):
                return None

        last_refreshed_at = _get("last_refreshed_at")
        if not last_refreshed_at:
            return True

        raw_tags = _get("tags")
        mm_tags: list[str] = list(raw_tags) if raw_tags else []

        trigger = _get("trigger")
        if isinstance(trigger, str):
            try:
                trigger = json.loads(trigger)
            except json.JSONDecodeError:
                trigger = None
        trigger = trigger or {}
        fact_types: list[str] = list(trigger.get("fact_types") or [])
        tags_match = trigger.get("tags_match")
        if not tags_match:
            tags_match = "any"  # default: untagged MM is "global", tagged MM matches any overlap

        params: list[Any] = [bank_id, last_refreshed_at]
        where = ["bank_id = $1", "updated_at > $2"]

        if mm_tags:
            operator, include_untagged = _parse_tags_match(tags_match)
            params.append(mm_tags)
            tag_idx = len(params)
            if include_untagged:
                where.append(f"(tags IS NULL OR tags = '{{}}' OR tags {operator} ${tag_idx}::varchar[])")
            else:
                where.append(f"(tags IS NOT NULL AND tags != '{{}}' AND tags {operator} ${tag_idx}::varchar[])")
        # else: untagged MM → no tag constraint, matches any ingested memory in scope

        if fact_types:
            params.append(fact_types)
            where.append(f"fact_type = ANY(${len(params)}::text[])")

        row = await conn.fetchrow(
            f"SELECT 1 FROM {fq_table('memory_units')} WHERE {' AND '.join(where)} LIMIT 1",
            *params,
        )
        return row is not None

    def _row_to_mental_model(self, row, *, detail: str = "full") -> dict[str, Any]:
        """Convert a database row to a mental model dict.

        Args:
            row: Database row
            detail: Detail level - 'metadata', 'content', or 'full'
        """
        result: dict[str, Any] = {
            "id": str(row["id"]),
            "bank_id": row["bank_id"],
            "name": row["name"],
            "tags": row["tags"] or [],
            "last_refreshed_at": row["last_refreshed_at"].isoformat() if row["last_refreshed_at"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        if detail == "metadata":
            return result

        trigger = row.get("trigger")
        if isinstance(trigger, str):
            try:
                trigger = json.loads(trigger)
            except json.JSONDecodeError:
                trigger = None
        result["source_query"] = row["source_query"]
        result["content"] = row["content"]
        result["max_tokens"] = row.get("max_tokens")
        result["trigger"] = trigger

        if detail == "full":
            reflect_response = row.get("reflect_response")
            if isinstance(reflect_response, str):
                try:
                    reflect_response = json.loads(reflect_response)
                except json.JSONDecodeError:
                    reflect_response = None
            result["reflect_response"] = reflect_response

            structured_content = row.get("structured_content")
            if isinstance(structured_content, str):
                try:
                    structured_content = json.loads(structured_content)
                except json.JSONDecodeError:
                    structured_content = None
            result["structured_content"] = structured_content

        return result

    # =========================================================================
    # Directives - Hard rules injected into prompts
    # =========================================================================

    async def list_directives(
        self,
        bank_id: str,
        *,
        tags: list[str] | None = None,
        tags_match: str = "any",
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
        isolation_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """List directives for a bank.

        Args:
            bank_id: Bank identifier
            tags: Optional tags to filter by
            tags_match: How to match tags - 'any', 'all', or 'exact'
            active_only: Only return active directives (default True)
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication
            isolation_mode: When True and tags=None, only return directives with no tags.
                This prevents tag-scoped directives from leaking into untagged operations.
                Default False (normal API behavior - returns all directives when tags=None)

        Returns:
            List of directive dicts
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_directives", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            # Build filters
            filters = ["bank_id = $1"]
            params: list[Any] = [bank_id]
            param_idx = 2

            if active_only:
                filters.append("is_active = TRUE")

            # Apply tags filter for directives:
            # Directives have special scoping rules:
            #   - Untagged directives (tags=[] or null) always apply regardless of reflect tags
            #   - Tagged directives only apply when the reflect operation includes matching tags
            #   - If tags=None and isolation_mode=True: only untagged directives (no leakage)
            #   - If tags=None and isolation_mode=False: all directives (normal API behavior)
            if tags:
                tags_clause, tags_params, param_idx = build_tags_where_clause(
                    tags=tags, param_offset=param_idx, table_alias="", match=tags_match
                )
                if tags_clause:
                    # Always include untagged directives; tagged ones must match the reflect tags
                    scoped_clause = tags_clause.replace("AND ", "", 1)
                    filters.append(f"((tags IS NULL OR tags = '{{}}') OR ({scoped_clause}))")
                    params.extend(tags_params)
            elif isolation_mode:
                # Isolation mode: only include directives with empty/null tags
                # This ensures tag-scoped directives don't apply to untagged operations
                filters.append("(tags IS NULL OR tags = '{}')")

            params.extend([limit, offset])

            rows = await conn.fetch(
                f"""
                SELECT id, bank_id, name, content, priority, is_active, tags, created_at, updated_at
                FROM {fq_table("directives")}
                WHERE {" AND ".join(filters)}
                ORDER BY priority DESC, created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
            )

            return [self._row_to_directive(row) for row in rows]

    async def get_directive(
        self,
        bank_id: str,
        directive_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Get a single directive by ID.

        Args:
            bank_id: Bank identifier
            directive_id: Directive UUID
            request_context: Request context for authentication

        Returns:
            Directive dict or None if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_directive", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, bank_id, name, content, priority, is_active, tags, created_at, updated_at
                FROM {fq_table("directives")}
                WHERE bank_id = $1 AND id = $2
                """,
                bank_id,
                directive_id,
            )

            return self._row_to_directive(row) if row else None

    async def create_directive(
        self,
        bank_id: str,
        name: str,
        content: str,
        *,
        priority: int = 0,
        is_active: bool = True,
        tags: list[str] | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Create a new directive.

        Args:
            bank_id: Bank identifier
            name: Human-readable name for the directive
            content: The directive text to inject into prompts
            priority: Higher priority directives are injected first (default 0)
            is_active: Whether this directive is active (default True)
            tags: Optional tags for filtering
            request_context: Request context for authentication

        Returns:
            The created directive dict
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="create_directive", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO {fq_table("directives")}
                (bank_id, name, content, priority, is_active, tags)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, bank_id, name, content, priority, is_active, tags, created_at, updated_at
                """,
                bank_id,
                name,
                content,
                priority,
                is_active,
                tags or [],
            )

        logger.info(f"[DIRECTIVES] Created directive '{name}' for bank {bank_id}")
        return self._row_to_directive(row)

    async def update_directive(
        self,
        bank_id: str,
        directive_id: str,
        *,
        name: str | None = None,
        content: str | None = None,
        priority: int | None = None,
        is_active: bool | None = None,
        tags: list[str] | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Update a directive.

        Args:
            bank_id: Bank identifier
            directive_id: Directive UUID
            name: New name (optional)
            content: New content (optional)
            priority: New priority (optional)
            is_active: New active status (optional)
            tags: New tags (optional)
            request_context: Request context for authentication

        Returns:
            Updated directive dict or None if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_directive", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        # Build update query dynamically
        updates = ["updated_at = now()"]
        params: list[Any] = []
        param_idx = 1

        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1

        if content is not None:
            updates.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1

        if priority is not None:
            updates.append(f"priority = ${param_idx}")
            params.append(priority)
            param_idx += 1

        if is_active is not None:
            updates.append(f"is_active = ${param_idx}")
            params.append(is_active)
            param_idx += 1

        if tags is not None:
            updates.append(f"tags = ${param_idx}")
            params.append(tags)
            param_idx += 1

        params.extend([bank_id, directive_id])

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {fq_table("directives")}
                SET {", ".join(updates)}
                WHERE bank_id = ${param_idx} AND id = ${param_idx + 1}
                RETURNING id, bank_id, name, content, priority, is_active, tags, created_at, updated_at
                """,
                *params,
            )

            return self._row_to_directive(row) if row else None

    async def delete_directive(
        self,
        bank_id: str,
        directive_id: str,
        *,
        request_context: "RequestContext",
    ) -> bool:
        """Delete a directive.

        Args:
            bank_id: Bank identifier
            directive_id: Directive UUID
            request_context: Request context for authentication

        Returns:
            True if deleted, False if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="delete_directive", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            result = await conn.execute(
                f"DELETE FROM {fq_table('directives')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                directive_id,
            )

        return result == "DELETE 1"

    def _row_to_directive(self, row) -> dict[str, Any]:
        """Convert a database row to a directive dict."""
        return {
            "id": str(row["id"]),
            "bank_id": row["bank_id"],
            "name": row["name"],
            "content": row["content"],
            "priority": row["priority"],
            "is_active": row["is_active"],
            "tags": row["tags"] or [],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def list_operations(
        self,
        bank_id: str,
        *,
        status: str | None = None,
        task_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
        exclude_parents: bool = False,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """List async operations for a bank with optional filtering and pagination.

        Args:
            bank_id: Bank identifier
            status: Optional status filter (pending, processing, completed, failed, cancelled)
            task_type: Optional operation type filter (retain, consolidation, etc.)
            limit: Maximum number of operations to return (default 20)
            offset: Number of operations to skip (default 0)
            exclude_parents: If True, exclude parent batch operations (is_parent=True in result_metadata)
            request_context: Request context for authentication

        Returns:
            Dict with total count and list of operations, sorted by most recent first
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_operations", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            # Build WHERE clause
            where_conditions = ["bank_id = $1"]
            params: list[Any] = [bank_id]

            if status:
                where_conditions.append(f"status = ${len(params) + 1}")
                params.append(status)

            if task_type:
                where_conditions.append(f"operation_type = ${len(params) + 1}")
                params.append(task_type)

            if exclude_parents:
                where_conditions.append("NOT (result_metadata::jsonb @> '{\"is_parent\": true}'::jsonb)")

            where_clause = " AND ".join(where_conditions)

            # Get total count (with filter)
            total_row = await conn.fetchrow(
                f"SELECT COUNT(*) as total FROM {fq_table('async_operations')} WHERE {where_clause}",
                *params,
            )
            total = total_row["total"] if total_row else 0

            # Get operations with pagination (include result_metadata to check for parent operations)
            operations = await conn.fetch(
                f"""
                SELECT operation_id, operation_type, created_at, status, error_message,
                       result_metadata, retry_count, next_retry_at
                FROM {fq_table("async_operations")}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )

            # Build operation list using status from database
            # Parent operations have their status updated when all children complete/fail
            operation_list = []
            for row in operations:
                # Map DB status to API status (pending includes processing)
                db_status = row["status"]
                api_status = "pending" if db_status in ("pending", "processing") else db_status

                result_metadata = conn.parse_json(row["result_metadata"]) or {}

                next_retry_at = row["next_retry_at"]
                operation_list.append(
                    {
                        "id": str(row["operation_id"]),
                        "task_type": row["operation_type"],
                        "items_count": result_metadata.get("items_count", 0),
                        "document_id": None,
                        "created_at": row["created_at"].isoformat(),
                        "status": row["status"],
                        "error_message": row["error_message"],
                        "retry_count": row["retry_count"] or 0,
                        "next_retry_at": next_retry_at.isoformat() if next_retry_at else None,
                    }
                )

            return {
                "total": total,
                "operations": operation_list,
            }

    async def get_operation_status(
        self,
        bank_id: str,
        operation_id: str,
        *,
        request_context: "RequestContext",
        include_payload: bool = False,
    ) -> dict[str, Any]:
        """Get the status of a specific async operation.

        For parent operations, the status is automatically updated in the database when all children complete/fail.

        Returns:
            - status: "pending", "completed", or "failed" (from database)
            - updated_at: last update timestamp
            - completed_at: completion timestamp (if completed)
            - child_operations: (for parent operations) list of child operation statuses
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_operation_status", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()

        op_uuid = uuid.UUID(operation_id)

        async with acquire_with_retry(backend) as conn:
            payload_column = ", task_payload" if include_payload else ""
            row = await conn.fetchrow(
                f"""
                SELECT operation_id, operation_type, created_at, updated_at, completed_at, status, error_message, result_metadata, retry_count, next_retry_at{payload_column}
                FROM {fq_table("async_operations")}
                WHERE operation_id = $1 AND bank_id = $2
                """,
                op_uuid,
                bank_id,
            )

            if row:
                # Check if this is a parent operation
                raw_rm = row["result_metadata"]
                result_metadata = conn.parse_json(raw_rm) or {}
                is_parent = result_metadata.get("is_parent", False)
                raw_tp = row["task_payload"] if include_payload else None
                task_payload = conn.parse_json(raw_tp) if include_payload else None

                # Status may be corrected by self-healing logic below for parent operations
                api_status = row["status"]

                # For parent operations, include child operations list
                if is_parent:
                    # Query child operations
                    child_rows = await conn.fetch(
                        f"""
                        SELECT operation_id, status, error_message, result_metadata
                        FROM {fq_table("async_operations")}
                        WHERE bank_id = $1
                        AND result_metadata::jsonb @> $2::jsonb
                        ORDER BY (result_metadata->>'sub_batch_index')::int
                        """,
                        bank_id,
                        json.dumps({"parent_operation_id": operation_id}),
                    )

                    # Build child operations list and check if parent status needs updating
                    child_statuses = []
                    all_done = True
                    any_failed = False
                    all_completed = True

                    for child_row in child_rows:
                        raw_crm = child_row["result_metadata"]
                        child_metadata = conn.parse_json(raw_crm) or {}
                        child_status = child_row["status"]

                        child_statuses.append(
                            {
                                "operation_id": str(child_row["operation_id"]),
                                "status": child_status,
                                "sub_batch_index": child_metadata.get("sub_batch_index"),
                                "items_count": child_metadata.get("items_count"),
                                "error_message": child_row["error_message"],
                            }
                        )

                        if child_status not in ("completed", "failed"):
                            all_done = False
                        if child_status == "failed":
                            any_failed = True
                        if child_status != "completed":
                            all_completed = False

                    # Self-healing: if parent status is out of sync with children, update it
                    if all_done and api_status == "pending":
                        correct_status = "failed" if any_failed else "completed"
                        logger.warning(
                            f"Parent operation {operation_id} status out of sync (DB: pending, should be: {correct_status}). Fixing."
                        )
                        await conn.execute(
                            f"""
                            UPDATE {fq_table("async_operations")}
                            SET status = $2, updated_at = NOW(), completed_at = NOW()
                            WHERE operation_id = $1
                            """,
                            op_uuid,
                            correct_status,
                        )
                        api_status = correct_status

                    return {
                        "operation_id": operation_id,
                        "status": api_status,
                        "operation_type": row["operation_type"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                        "error_message": row["error_message"],
                        "retry_count": row["retry_count"] or 0,
                        "next_retry_at": row["next_retry_at"].isoformat() if row["next_retry_at"] else None,
                        "result_metadata": result_metadata,
                        "child_operations": child_statuses,
                        "task_payload": task_payload,
                    }
                else:
                    # Regular operation (not a parent)
                    return {
                        "operation_id": operation_id,
                        "status": api_status,
                        "operation_type": row["operation_type"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                        "error_message": row["error_message"],
                        "retry_count": row["retry_count"] or 0,
                        "next_retry_at": row["next_retry_at"].isoformat() if row["next_retry_at"] else None,
                        "result_metadata": result_metadata,
                        "task_payload": task_payload,
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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="cancel_operation", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        op_uuid = uuid.UUID(operation_id)

        async with acquire_with_retry(backend) as conn:
            # Check if operation exists, belongs to this bank, and is in a cancellable state
            result = await conn.fetchrow(
                f"SELECT bank_id, status FROM {fq_table('async_operations')} WHERE operation_id = $1 AND bank_id = $2",
                op_uuid,
                bank_id,
            )

            if not result:
                raise ValueError(f"Operation {operation_id} not found for bank {bank_id}")

            if result["status"] != "pending":
                from hindsight_api.extensions import OperationValidationError

                raise OperationValidationError(
                    f"Operation {operation_id} cannot be cancelled: status is '{result['status']}', only 'pending' operations can be cancelled",
                    409,
                )

            # Mark the operation as cancelled
            await conn.execute(
                f"UPDATE {fq_table('async_operations')} SET status = 'cancelled', updated_at = now() WHERE operation_id = $1",
                op_uuid,
            )

            return {
                "success": True,
                "message": f"Operation {operation_id} cancelled",
                "operation_id": operation_id,
                "bank_id": bank_id,
            }

    async def retry_operation(
        self,
        bank_id: str,
        operation_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Re-queue a failed async operation."""
        await self._authenticate_tenant(request_context)
        from hindsight_api.extensions import OperationValidationError

        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="retry_operation", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        op_uuid = uuid.UUID(operation_id)

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"SELECT bank_id, status FROM {fq_table('async_operations')} WHERE operation_id = $1 AND bank_id = $2",
                op_uuid,
                bank_id,
            )

            if not row:
                raise ValueError(f"Operation {operation_id} not found for bank {bank_id}")

            if row["status"] not in ("failed", "cancelled"):
                raise OperationValidationError(
                    f"Operation {operation_id} cannot be retried: status is '{row['status']}', expected 'failed' or 'cancelled'",
                    409,
                )

            await conn.execute(
                f"""
                UPDATE {fq_table("async_operations")}
                SET status = 'pending',
                    error_message = NULL,
                    completed_at = NULL,
                    next_retry_at = NULL,
                    worker_id = NULL,
                    claimed_at = NULL,
                    retry_count = 0,
                    updated_at = NOW()
                WHERE operation_id = $1
                """,
                op_uuid,
            )

            return {
                "success": True,
                "message": f"Operation {operation_id} queued for retry",
                "operation_id": operation_id,
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
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_bank", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
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

    # =========================================================================
    # Webhook configuration methods
    #
    # These wrap the raw backend.ops.* calls used to be invoked directly from
    # the HTTP layer with ``fq_table("webhooks")``. Computing the fully-qualified
    # table name from the HTTP layer evaluates ``fq_table`` before the schema
    # contextvar is set, which means under deployments that resolve a
    # per-request target schema (multi-target-schema routing) the webhook rows
    # would land in the default schema while the rest of the bank's data lives
    # in a per-target schema. The fire path uses the bank's resolved schema
    # and would silently never see those webhook rows.
    #
    # Routing through engine methods that call ``_authenticate_tenant`` first
    # ensures ``fq_table`` resolves to the same schema used by retain,
    # consolidate, and every other bank-scoped operation.
    # =========================================================================

    async def create_webhook(
        self,
        bank_id: str,
        *,
        webhook_id: uuid.UUID,
        url: str,
        secret: str | None,
        event_types: list[str],
        enabled: bool,
        http_config_json: str,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Insert a webhook row in the bank's resolved schema.

        Authenticates the tenant first so ``fq_table("webhooks")`` resolves to
        the same schema as the rest of the bank's data.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="create_webhook", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        backend = await self._get_backend()

        # Ensure the bank row exists before inserting into webhooks (FK constraint).
        _, created = await bank_utils.get_or_create_bank_profile(backend, bank_id)
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

        async with acquire_with_retry(backend) as conn:
            row = await backend.ops.create_webhook(
                conn,
                fq_table("webhooks"),
                webhook_id,
                bank_id,
                url,
                secret,
                event_types,
                enabled,
                http_config_json,
            )
        return dict(row) if row is not None else None

    async def list_webhooks(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List webhooks for a bank in the bank's resolved schema."""
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_webhooks", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await backend.ops.list_webhooks_for_bank(
                conn,
                fq_table("webhooks"),
                bank_id,
            )
        return [dict(row) for row in rows]

    async def update_webhook(
        self,
        bank_id: str,
        webhook_id: uuid.UUID,
        *,
        set_clauses: list[str],
        params: list[Any],
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Update a webhook row in the bank's resolved schema.

        ``set_clauses`` and ``params`` are pre-built by the caller using PATCH
        semantics (only sent fields are updated). The first two ``params`` are
        ``webhook_id`` and ``bank_id``; subsequent params correspond to the
        SET clauses.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_webhook", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            row = await backend.ops.update_webhook(
                conn,
                fq_table("webhooks"),
                webhook_id,
                bank_id,
                set_clauses,
                params,
            )
        return dict(row) if row is not None else None

    async def delete_webhook(
        self,
        bank_id: str,
        webhook_id: uuid.UUID,
        *,
        request_context: "RequestContext",
    ) -> bool:
        """Delete a webhook row from the bank's resolved schema.

        Returns True if a row was deleted, False if no matching row was found.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="delete_webhook", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            return await backend.ops.delete_webhook(
                conn,
                fq_table("webhooks"),
                webhook_id,
                bank_id,
            )

    async def list_webhook_deliveries(
        self,
        bank_id: str,
        webhook_id: uuid.UUID,
        *,
        limit: int,
        cursor: str | None,
        request_context: "RequestContext",
    ) -> list[dict[str, Any]]:
        """List webhook delivery rows from the bank's resolved schema.

        First verifies the webhook belongs to this bank (in the same schema),
        then reads the delivery rows from ``async_operations``. Returns up to
        ``limit + 1`` rows so callers can determine whether more pages exist.

        Raises:
            LookupError: When the webhook does not exist in this bank.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_webhook_deliveries", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            webhook_row = await conn.fetchrow(
                f"SELECT id FROM {fq_table('webhooks')} WHERE id = $1 AND bank_id = $2",
                webhook_id,
                bank_id,
            )
            if not webhook_row:
                raise LookupError("Webhook not found")

            rows = await backend.ops.list_webhook_deliveries(
                conn,
                fq_table("async_operations"),
                str(webhook_id),
                bank_id,
                limit,
                cursor,
            )
        return [dict(row) for row in rows]

    async def _submit_async_operation(
        self,
        bank_id: str,
        operation_type: str,
        task_type: str,
        task_payload: dict[str, Any],
        *,
        result_metadata: dict[str, Any] | None = None,
        dedupe_by_bank: bool = False,
    ) -> dict[str, Any]:
        """Generic helper to submit an async operation.

        Args:
            bank_id: Bank identifier
            operation_type: Operation type for the async_operations record (e.g., 'consolidation', 'retain')
            task_type: Task type for the task payload (e.g., 'consolidation', 'batch_retain')
            task_payload: Additional task payload fields (operation_id and bank_id are added automatically)
            result_metadata: Optional metadata to store with the operation record
            dedupe_by_bank: If True, skip creating a new task if one is already pending for this bank+operation_type

        Returns:
            Dict with operation_id and optionally deduplicated=True if an existing task was found
        """
        import json

        backend = await self._get_backend()

        # Check for existing pending task if deduplication is enabled
        # Note: We only check 'pending', not 'processing', because a processing task
        # uses a watermark from when it started - new memories added after that point
        # would need another consolidation run to be processed.
        if dedupe_by_bank:
            async with acquire_with_retry(backend) as conn:
                existing = await conn.fetchrow(
                    f"""
                    SELECT operation_id FROM {fq_table("async_operations")}
                    WHERE bank_id = $1 AND operation_type = $2 AND status = 'pending'
                    LIMIT 1
                    """,
                    bank_id,
                    operation_type,
                )
                if existing:
                    logger.debug(
                        f"{operation_type} task already pending for bank_id={bank_id}, "
                        f"skipping duplicate (existing operation_id={existing['operation_id']})"
                    )
                    return {
                        "operation_id": str(existing["operation_id"]),
                        "deduplicated": True,
                    }

        operation_id = uuid.uuid4()

        # Build full payload before INSERT so task_payload is included atomically.
        # Previously the INSERT omitted task_payload and a separate submit_task call
        # did an UPDATE — a crash between the two left a null-payload row that the
        # worker's claim query (task_payload IS NOT NULL) could never pick up.
        full_payload = {
            "type": task_type,
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            **task_payload,
        }

        # Insert operation record with task_payload in a single atomic statement
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("async_operations")} (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                operation_id,
                bank_id,
                operation_type,
                json.dumps(result_metadata or {}, default=_json_default),
                "pending",
                json.dumps(full_payload, default=_json_default),
            )

        # For SyncTaskBackend: executes the task immediately.
        # For BrokerTaskBackend: no-op (submit_task's UPDATE skips rows whose
        # task_payload is already set, which it is after the INSERT above). The call
        # is kept for symmetry and to support any future notification mechanisms.
        await self._task_backend.submit_task(full_payload)

        logger.info(f"{operation_type} task queued for bank_id={bank_id}, operation_id={operation_id}")

        return {
            "operation_id": str(operation_id),
        }

    async def submit_async_retain(
        self,
        bank_id: str,
        contents: list[dict[str, Any]],
        *,
        request_context: "RequestContext",
        document_tags: list[str] | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Submit a batch retain operation to run asynchronously.

        For large batches (exceeding retain_batch_chars threshold), automatically splits
        into smaller sub-batches and creates a parent operation that tracks all children.
        """
        await self._authenticate_tenant(request_context)

        # Run operation validator (bank access, credits, etc.) before queuing
        if self._operation_validator:
            from hindsight_api.extensions import RetainContext

            ctx = RetainContext(
                bank_id=bank_id,
                contents=[dict(c) for c in contents],
                request_context=request_context,
            )
            result = await self._validate_operation(self._operation_validator.validate_retain(ctx))
            if result and result.contents is not None:
                contents = result.contents

        # Validate no duplicate document_ids in the batch
        # Having duplicate document_ids causes race conditions in document upserts during parallel processing
        doc_ids = [item.get("document_id") for item in contents if item.get("document_id")]
        if len(doc_ids) != len(set(doc_ids)):
            from collections import Counter

            duplicates = [doc_id for doc_id, count in Counter(doc_ids).items() if count > 1]
            raise ValueError(
                f"Batch contains duplicate document_ids: {duplicates}. "
                f"Each content item in a batch must have a unique document_id to avoid race conditions."
            )

        # Calculate total token count and determine if we need to split
        total_tokens = sum(count_tokens(item.get("content", "")) for item in contents)
        config = get_config()
        tokens_per_batch = config.retain_batch_tokens

        # Split into sub-batches based on token count
        sub_batches = []
        current_batch = []
        current_batch_tokens = 0

        for item in contents:
            item_tokens = count_tokens(item.get("content", ""))

            # If adding this item would exceed the limit, start a new batch
            # (unless current batch is empty - then we must include it even if it's large)
            if current_batch and current_batch_tokens + item_tokens > tokens_per_batch:
                sub_batches.append(current_batch)
                current_batch = [item]
                current_batch_tokens = item_tokens
            else:
                current_batch.append(item)
                current_batch_tokens += item_tokens

        # Add the last batch
        if current_batch:
            sub_batches.append(current_batch)

        # Log splitting info if we actually split
        if len(sub_batches) > 1:
            logger.info(
                f"Large async retain batch ({total_tokens:,} tokens from {len(contents)} items). "
                f"Split into {len(sub_batches)} sub-batches: {[len(b) for b in sub_batches]} items each"
            )

        # Always create parent operation (even for single batch - simpler, more reliable code path)
        import uuid

        parent_operation_id = uuid.uuid4()
        backend = await self._get_backend()

        # Ensure the bank row exists before inserting async_operations (which now has a FK).
        # Banks are created lazily on first retain, but the FK requires the row to exist first.
        _, created = await bank_utils.get_or_create_bank_profile(self._backend, bank_id)
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

        # Create typed metadata for parent operation
        parent_metadata = BatchRetainParentMetadata(
            items_count=len(contents),
            total_tokens=total_tokens,
            num_sub_batches=len(sub_batches),
        )

        # Persist the parent row and all child rows in a single transaction.
        #
        # The parent row is a status aggregator with NO task_payload (workers
        # skip rows where task_payload IS NULL because they're not directly
        # executable). Its lifecycle is driven by child completions: when all
        # children reach a terminal state, the parent gets promoted by the
        # aggregator.
        #
        # If the parent INSERT and child INSERTs are not transactionally
        # coupled, any failure between them (connection drop, timeout, schema
        # cache invalidation under concurrent load) leaves a parent row with
        # zero children. Workers ignore it forever (no task_payload), the
        # aggregator never fires (no children to complete), and the row sits
        # pending indefinitely — visible in queue-depth metrics and growing
        # without bound. Wrapping parent + children in one transaction makes
        # the create-batch operation atomic: either all rows are visible to
        # workers, or none are.
        #
        # submit_task() must run AFTER the transaction commits. SyncTaskBackend
        # (used in tests) executes the task synchronously, which would not see
        # the still-uncommitted child row. BrokerTaskBackend / WorkerTaskBackend
        # are effectively no-ops for already-populated task_payload, but we
        # defer them all uniformly for clarity.
        deferred_child_payloads: list[dict[str, Any]] = []

        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table("async_operations")} (operation_id, bank_id, operation_type, result_metadata, status)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    parent_operation_id,
                    bank_id,
                    "batch_retain",
                    json.dumps(parent_metadata.to_dict()),
                    "pending",  # Will be updated by status aggregation
                )

                for i, sub_batch in enumerate(sub_batches, 1):
                    if len(sub_batches) > 1:
                        sub_batch_tokens = sum(count_tokens(item.get("content", "")) for item in sub_batch)
                        logger.info(
                            f"Submitting sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_tokens:,} tokens"
                        )

                    task_payload: dict[str, Any] = {"contents": sub_batch}
                    if document_tags:
                        task_payload["document_tags"] = document_tags
                    if strategy:
                        task_payload["strategy"] = strategy
                    # Pass tenant_id and api_key_id through task payload
                    if request_context.tenant_id:
                        task_payload["_tenant_id"] = request_context.tenant_id
                    if request_context.api_key_id:
                        task_payload["_api_key_id"] = request_context.api_key_id

                    child_metadata = BatchRetainChildMetadata(
                        items_count=len(sub_batch),
                        parent_operation_id=str(parent_operation_id),
                        sub_batch_index=i,
                        total_sub_batches=len(sub_batches),
                    )

                    child_operation_id = uuid.uuid4()
                    full_payload = {
                        "type": "batch_retain",
                        "operation_id": str(child_operation_id),
                        "bank_id": bank_id,
                        **task_payload,
                    }

                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table("async_operations")} (operation_id, bank_id, operation_type, result_metadata, status, task_payload)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        """,
                        child_operation_id,
                        bank_id,
                        "retain",
                        json.dumps(child_metadata.to_dict(), default=_json_default),
                        "pending",
                        json.dumps(full_payload, default=_json_default),
                    )
                    deferred_child_payloads.append(full_payload)

        logger.info(f"Created parent operation {parent_operation_id} with {len(sub_batches)} child sub-batch(es)")

        # Notify the task backend after commit. For BrokerTaskBackend /
        # WorkerTaskBackend in production this is a no-op because task_payload
        # is already populated; for SyncTaskBackend in tests this kicks off
        # synchronous execution against the now-committed rows.
        for full_payload in deferred_child_payloads:
            await self._task_backend.submit_task(full_payload)

        return {
            "operation_id": str(parent_operation_id),
            "items_count": len(contents),
        }

    async def submit_async_file_retain(
        self,
        bank_id: str,
        file_items: list[dict[str, Any]],
        document_tags: list[str] | None,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """
        Submit batch file conversion + retain operation.

        Each file is converted to markdown and then retained as a memory.
        Files are stored in object storage and conversion happens asynchronously.

        Args:
            bank_id: Bank ID
            file_items: List of file items, each containing:
                - file: UploadFile object (FastAPI)
                - document_id: Document ID
                - context: Optional context
                - metadata: Optional metadata dict
                - tags: Optional tags list
                - timestamp: Optional timestamp
                - parser: Ordered list of parser names to try (fallback chain)
            document_tags: Tags applied to all documents
            request_context: Request context for authentication

        Returns:
            dict with operation_id and files_count
        """
        await self._authenticate_tenant(request_context)

        config = get_config()

        # Validate file count
        if len(file_items) > config.file_conversion_max_batch_size:
            raise ValueError(f"Too many files. Maximum {config.file_conversion_max_batch_size} files per request.")

        # Read all files and validate total batch size
        files_data = []
        total_batch_size = 0

        for item in file_items:
            file = item["file"]
            file_data = await file.read()
            total_batch_size += len(file_data)
            files_data.append((item, file, file_data))

        # Validate total batch size
        if total_batch_size > config.file_conversion_max_batch_size_bytes:
            total_mb = total_batch_size / (1024 * 1024)
            raise ValueError(
                f"Total batch size ({total_mb:.1f}MB) exceeds maximum of {config.file_conversion_max_batch_size_mb}MB"
            )

        # Submit individual operation for each file
        operation_ids = []
        for item, file, file_data in files_data:
            # Generate storage key
            storage_key = f"banks/{bank_id}/files/{item['document_id']}/{file.filename}"

            # Store file in object storage
            await self._file_storage.store(
                file_data=file_data,
                key=storage_key,
                metadata={
                    "content_type": file.content_type or "application/octet-stream",
                    "original_filename": file.filename,
                    "bank_id": bank_id,
                    "document_id": item["document_id"],
                },
            )

            # Create individual operation and submit task
            task_payload: dict[str, Any] = {
                "document_id": item["document_id"],
                "storage_key": storage_key,
                "original_filename": file.filename,
                "content_type": file.content_type or "application/octet-stream",
                "parser": item["parser"],
                "context": item.get("context"),
                "metadata": item.get("metadata", {}),
                "tags": item.get("tags", []),
                "document_tags": document_tags or [],
                "timestamp": item.get("timestamp"),
            }
            if item.get("strategy"):
                task_payload["strategy"] = item["strategy"]

            # Pass tenant_id and api_key_id through task payload
            if request_context.tenant_id:
                task_payload["_tenant_id"] = request_context.tenant_id
            if request_context.api_key_id:
                task_payload["_api_key_id"] = request_context.api_key_id

            result = await self._submit_async_operation(
                bank_id=bank_id,
                operation_type="file_convert_retain",
                task_type="file_convert_retain",
                task_payload=task_payload,
                result_metadata={
                    "original_filename": file.filename,
                },
                dedupe_by_bank=False,
            )
            operation_ids.append(result["operation_id"])

        return {
            "operation_ids": operation_ids,
            "files_count": len(file_items),
        }

    async def submit_async_consolidation(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Submit a consolidation operation to run asynchronously.

        Deduplicates by bank_id - if there's already a pending consolidation for this bank,
        returns the existing operation_id instead of creating a new one.

        Args:
            bank_id: Bank identifier
            request_context: Request context for authentication

        Returns:
            Dict with operation_id
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(
                bank_id=bank_id, operation="submit_async_consolidation", request_context=request_context
            )
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        # Pass tenant_id and api_key_id through task payload so the worker
        # can provide request context to extension hooks (e.g., usage metering
        # for mental model refreshes triggered by consolidation).
        task_payload: dict[str, Any] = {}
        if request_context.tenant_id:
            task_payload["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            task_payload["_api_key_id"] = request_context.api_key_id

        return await self._submit_async_operation(
            bank_id=bank_id,
            operation_type="consolidation",
            task_type="consolidation",
            task_payload=task_payload,
            dedupe_by_bank=True,
        )

    async def submit_async_refresh_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Submit an async mental model refresh operation.

        This schedules a background task to re-run the source query and update the content.

        Args:
            bank_id: Bank identifier
            mental_model_id: Mental model UUID to refresh
            request_context: Request context for authentication

        Returns:
            Dict with operation_id
        """
        # Block mental model refresh when LLM provider is "none"
        if self._llm_config.provider == "none":
            from .providers.none_llm import LLMNotAvailableError

            raise LLMNotAvailableError(
                "Mental model refresh requires an LLM provider. Current provider is set to 'none'. "
                "Set HINDSIGHT_API_LLM_PROVIDER to a real provider (e.g., openai, anthropic, gemini)."
            )

        await self._authenticate_tenant(request_context)

        # Pre-operation validation (credit check)
        if self._operation_validator:
            from hindsight_api.extensions.operation_validator import MentalModelRefreshContext

            ctx = MentalModelRefreshContext(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            await self._validate_operation(self._operation_validator.validate_mental_model_refresh(ctx))

        # Verify mental model exists
        mental_model = await self.get_mental_model(bank_id, mental_model_id, request_context=request_context)
        if not mental_model:
            raise ValueError(f"Mental model {mental_model_id} not found in bank {bank_id}")

        # Pass tenant_id and api_key_id through task payload so the worker
        # can provide request context to extension hooks.
        task_payload: dict[str, Any] = {
            "mental_model_id": mental_model_id,
        }
        if request_context.tenant_id:
            task_payload["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            task_payload["_api_key_id"] = request_context.api_key_id

        return await self._submit_async_operation(
            bank_id=bank_id,
            operation_type="refresh_mental_model",
            task_type="refresh_mental_model",
            task_payload=task_payload,
            result_metadata={"mental_model_id": mental_model_id, "name": mental_model["name"]},
            dedupe_by_bank=False,
        )
