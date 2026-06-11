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
import functools
import inspect
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast, overload

import asyncpg
import httpx

from .._vector_index import ann_search_tuning_settings, configured_vector_extension
from ..cancellation import OperationCancelledError
from ..config import (
    DEFAULT_RECALL_CHUNKS_MAX_TOKENS,
    DEFAULT_RECALL_INCLUDE_CHUNKS,
    DEFAULT_RECALL_MAX_TOKENS,
    DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS,
    ENV_MODEL_INIT_TIMEOUT,
    HindsightConfig,
    get_config,
)
from ..tracing import create_operation_span
from ..utils import mask_network_location
from ..worker.exceptions import DeferOperation, RetryTaskAt
from ..worker.stage import set_stage
from .audit import AuditLogger, audit_context
from .bank_stats_cache import BankStatsCache
from .db import DatabaseBackend, create_database_backend
from .db_budget import budgeted_operation
from .llm_trace import (
    LLMRequestEntry,
    LLMRequestListResponse,
    LLMRequestStatsBucket,
    LLMRequestStatsResponse,
    LLMRequestTokenSums,
    LLMTraceRecorder,
    trace_context_of,
)
from .operation_metadata import (
    BatchRetainChildMetadata,
    BatchRetainParentMetadata,
    RetainExtractionErrors,
    RetainOutcomeAggregate,
    RetainOutcomeMetadata,
)
from .sql import SQLDialect, create_sql_dialect

# Context variable for current schema (async-safe, per-task isolation)
# Note: default is None, actual default comes from config via get_current_schema()
_current_schema: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_schema", default=None)

# Context variable for the bank an operation runs for (async-safe, per-task isolation).
# Set by the engine wherever it learns the bank (recall/retain/batch/task execution) so
# downstream provider calls can attribute spend per bank — e.g. tagging the OpenAI `user`
# field for cost gateways. None outside a bank-scoped operation.
_current_bank_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_bank_id", default=None)
MENTAL_MODEL_PENDING_CONTENT = "Generating content..."


def get_current_schema() -> str:
    """Get the current schema from context (falls back to config default)."""
    schema = _current_schema.get()
    if schema is None:
        # Fall back to configured default schema
        return get_config().database_schema
    return schema


def get_current_bank_id() -> str | None:
    """Get the bank id of the in-flight operation, or None outside a bank-scoped context."""
    return _current_bank_id.get()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _bind_bank_id(
    arg: str = "bank_id", key: str | None = None
) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Bind ``_current_bank_id`` to an argument of the wrapped coroutine for the call's duration.

    ``arg`` names the parameter carrying the bank id; ``key`` optionally pulls it out of a
    dict-valued argument (e.g. ``task_dict["bank_id"]``). Token-based set/reset (including on
    exception) keeps the binding scoped to the call.
    """

    def decorate(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            value = sig.bind(*args, **kwargs).arguments.get(arg)
            if key is not None and isinstance(value, dict):
                value = value.get(key)
            token = _current_bank_id.set(value if isinstance(value, str) else None)
            try:
                return await func(*args, **kwargs)
            finally:
                _current_bank_id.reset(token)

        return wrapper

    return decorate


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

# Consolidation retry: indefinite retry with capped exponential backoff.
# Transient upstream outages (LLM provider down, DB flapping, tenant-ext
# blip) must eventually recover; the worker should keep trying rather than
# silently dead-lettering a bank's consolidation backlog. Deterministic
# failures (integrity violations, embedding dimension mismatches) are
# filtered upstream by `_is_non_retryable_task_error` and never reach the
# retry path. The dedup-by-bank guard prevents per-op retries from
# multiplying when a peer consolidation is already pending for the bank.
#
# Base is intentionally short so a momentary 5xx clears in seconds, not
# minutes; the cap is preserved so a genuine multi-hour outage doesn't hammer
# the upstream. Issue #1842 observed banks sitting idle for whole minutes on
# transient LLM blips because the prior 60s base overshot recovery by 10x+.
_CONSOLIDATION_RETRY_BACKOFF_BASE_SECONDS = 5
_CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS = 1800  # 30 min cap

# Upper bound on the per-bank LLM connectivity probe so a hung provider can't wedge
# the request. The probe is a deliberate, non-polled action (POST .../health/llm).
_LLM_PROBE_TIMEOUT_SECONDS = 10.0

# Substrings that identify an authentication/authorization failure across providers.
# A wrong API key is the single most common probe failure, so it gets its own status.
_AUTH_ERROR_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "api key not valid",
    "api_key_invalid",
    "authentication",
    "permission denied",
    "permissiondenied",
)


def _is_auth_error(error: Exception) -> bool:
    """Whether a probe exception looks like an auth failure (typically a bad API key).

    Walks the exception chain for an HTTP 401/403 status code, then falls back to
    matching known auth markers in the (provider-wrapped) message. Used only to pick a
    status label — the raw error itself is never returned to the client.
    """
    seen: list[Exception] = []
    current: BaseException | None = error
    for _ in range(6):  # bounded walk to avoid pathological cycles
        if current is None or current in seen:
            break
        seen.append(current)  # type: ignore[arg-type]
        code = getattr(current, "status_code", None) or getattr(current, "code", None)
        if code in (401, 403, "401", "403"):
            return True
        current = current.__cause__ or current.__context__
    text = " ".join(str(item) for item in seen).lower()
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


@dataclass
class LlmOperationHealthInfo:
    """Connectivity status for one operation's LLM. Status only — deliberately carries
    no provider/model/endpoint/error so the probe never leaks the LLM configuration."""

    operation: str
    ok: bool
    status: str
    latency_ms: float | None


@dataclass
class BankLlmHealthInfo:
    """Per-bank LLM connectivity probe across retain/consolidation/reflect (see
    MemoryEngine.check_bank_llm)."""

    bank_id: str
    operations: list[LlmOperationHealthInfo]


@dataclass
class _LlmProbeOutcome:
    """Internal result of probing a single LLM client (before it's tagged per operation)."""

    ok: bool
    status: str
    latency_ms: float | None


def _consolidation_retry_backoff_seconds(retry_count: int) -> int:
    """Capped exponential backoff: 5, 10, 20, 40, 80, 160, 320, 640, 1280, 1800, 1800, …"""
    return min(
        _CONSOLIDATION_RETRY_BACKOFF_BASE_SECONDS * (2**retry_count),
        _CONSOLIDATION_RETRY_BACKOFF_MAX_SECONDS,
    )


class UnqualifiedTableError(Exception):
    """Raised when SQL contains unqualified table references."""

    pass


class MentalModelRefreshError(Exception):
    """Raised when refresh_mental_model cannot produce new content.

    The previous content (if any) is preserved in the DB and the reflect_response
    audit trail is persisted before this is raised, so the failure is recoverable
    and auditable. Callers (worker queue, integration tests) should treat this
    as a retryable condition.
    """

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


from .cross_encoder import CrossEncoderModel
from .embeddings import Embeddings, create_embeddings_from_env
from .interface import MemoryEngineInterface

if TYPE_CHECKING:
    from hindsight_api.extensions import OperationValidatorExtension, TenantExtension, ValidationResult
    from hindsight_api.models import RequestContext

    from .audit import AuditLogListResponse, AuditLogStatsResponse
    from .transfer import BankImportResult, ImportResult


from enum import Enum

from ..pg0 import EmbeddedPostgres, parse_pg0_url
from .entity_resolver import EntityResolver
from .llm_wrapper import LLMConfig, requires_api_key, sanitize_llm_output, sanitize_text
from .query_analyzer import QueryAnalyzer
from .reflect import run_reflect_agent
from .reflect.tools import tool_expand, tool_recall, tool_search_mental_models, tool_search_observations
from .response_models import (
    VALID_RECALL_FACT_TYPES,
    EntityState,
    LLMCallTrace,
    MemoryFact,
    ReflectResult,
    TokenUsage,
    ToolCallTrace,
)
from .response_models import RecallResult as RecallResultModel
from .retain import bank_utils, embedding_utils
from .retain.types import RetainContentDict
from .search.reranking import CrossEncoderReranker, apply_combined_scoring
from .search.tags import TagGroup, TagsMatch, build_tag_groups_where_clause, build_tags_where_clause
from .search.types import ScoredResult
from .task_backend import TaskBackend

# Recall ranking strategy: how the per-arm (semantic/bm25/graph/temporal) results are
# fused and reranked into the final order.
#   "cross_encoder" — RRF fusion + cross-encoder rerank (default, user-facing recall).
#   "rrf"           — RRF fusion, no cross-encoder (RRF score is the order).
#   "interleave"    — round-robin interleave fusion, no cross-encoder. Guarantees each
#                     arm's top hits a slot (used by consolidation dedup recall, where RRF
#                     buried the near-identical twin below budget). See interleave_fusion.
RecallReranking = Literal["cross_encoder", "rrf", "interleave"]
from .token_encoding import get_token_encoding

RetainOutboxCallback = Callable[[asyncpg.Connection], Awaitable[None]]
RetainOutboxCallbackFactory = Callable[[list[RetainContentDict]], RetainOutboxCallback | None]


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


@dataclass
class _SubBatchSplit:
    """Result of packing retain contents into sub-batches.

    ``sub_batches[i]`` is a list of RetainContentDict items that should
    be processed together. ``origin_indices[i]`` lists the indices into
    the original ``contents`` list that contributed items to
    ``sub_batches[i]``; callers that present per-input results to the
    user (such as ``retain_batch_async``) use this mapping to merge
    results belonging to the same original content back together when
    an oversized item was chunked across multiple sub-batches.

    ``document_body_overrides[i]`` is the full original body of the
    oversized item that produced ``sub_batches[i]``, or ``None`` when
    the sub-batch was not produced by chunking an oversized item. The
    orchestrator uses this as the ``documents.original_text`` payload
    so that slicing an item across sub-batches does not persist a
    partial body (see issue #1838).
    """

    sub_batches: list[list[RetainContentDict]]
    origin_indices: list[list[int]]
    document_body_overrides: list[str | None] = field(default_factory=list)


def _split_contents_into_sub_batches(
    contents: list[RetainContentDict],
    tokens_per_batch: int,
) -> _SubBatchSplit:
    """Pack retain contents into sub-batches whose combined token count
    stays at or below ``tokens_per_batch``.

    Any single item that already exceeds the budget is chunked via
    ``fact_extraction.chunk_text`` (paragraph/sentence aware, or
    conversation-turn aware for JSON arrays and JSONL) and each chunk becomes its
    own single-item sub-batch. Without this, an oversized single item
    would pass through as a ``1/1`` sub-batch holding the entire
    payload — which contradicts the splitter's log and lets the
    orchestrator OOM under realistic memory limits (see issue #1571).

    Used by the in-process ``retain_batch_async`` path, which processes
    the returned sub-batches SEQUENTIALLY with ``is_first_batch=(i==1)``.
    The async submission path uses ``_split_contents_into_async_children``
    instead, which never fragments a single item across children — see
    that helper for the reasoning.
    """
    from .retain import fact_extraction

    # chunk_text takes a char budget; cl100k_base averages ~3-4 chars
    # per token on natural-language input. Use 3x for headroom so a
    # chunk's token count is comfortably under tokens_per_batch even
    # when content tokenizes denser than average (code, JSON).
    char_budget = max(tokens_per_batch * 3, 1)

    sub_batches: list[list[RetainContentDict]] = []
    origin_indices: list[list[int]] = []
    document_body_overrides: list[str | None] = []
    current_batch: list[RetainContentDict] = []
    current_batch_origins: list[int] = []
    current_batch_tokens = 0

    def _flush() -> None:
        nonlocal current_batch, current_batch_origins, current_batch_tokens
        if current_batch:
            sub_batches.append(current_batch)
            origin_indices.append(current_batch_origins)
            document_body_overrides.append(None)
            current_batch = []
            current_batch_origins = []
            current_batch_tokens = 0

    for original_idx, item in enumerate(contents):
        content_str = item.get("content", "") or ""
        item_tokens = count_tokens(content_str)

        if item_tokens > tokens_per_batch:
            # Oversized single item: flush anything in flight, then
            # chunk the content and emit each chunk as its own
            # single-item sub-batch. The sub-batches share the
            # original item's document_id and metadata so the
            # orchestrator's first-batch document tracking still
            # cascade-deletes the prior document version on slice 1.
            # Each slice carries ``content_str`` as the document body
            # override so the orchestrator writes the full original
            # text to documents.original_text — not just its own slice
            # (otherwise the last slice would clobber the body with a
            # truncated payload; see issue #1838).
            _flush()
            chunks = fact_extraction.chunk_text(content_str, char_budget)
            for chunk in chunks:
                chunk_item = cast(RetainContentDict, {**item, "content": chunk})
                sub_batches.append([chunk_item])
                origin_indices.append([original_idx])
                document_body_overrides.append(content_str)
            continue

        if current_batch and current_batch_tokens + item_tokens > tokens_per_batch:
            _flush()
        current_batch.append(item)
        current_batch_origins.append(original_idx)
        current_batch_tokens += item_tokens

    _flush()
    return _SubBatchSplit(
        sub_batches=sub_batches,
        origin_indices=origin_indices,
        document_body_overrides=document_body_overrides,
    )


def _split_contents_into_async_children(
    contents: list[RetainContentDict],
    tokens_per_batch: int,
) -> list[list[RetainContentDict]]:
    """Pack retain contents into child operations for async submission.

    Unlike ``_split_contents_into_sub_batches`` (used by the in-process
    path), this NEVER fragments a single input item across multiple
    children. Items where ``count_tokens(content) > tokens_per_batch``
    are emitted as their own single-item child holding the FULL
    un-chunked content; the in-process ``retain_batch_async`` then
    re-chunks them SEQUENTIALLY inside one worker slot with correct
    ``is_first_batch=(i==1)`` semantics.

    The previous behavior — chunking oversized items into N independent
    child async-operations sharing one ``document_id`` — let workers
    claim siblings concurrently with no per-document gate (the busy-bank
    guard in ``claim_tasks`` only covers consolidation). Each concurrent
    child ran ``handle_document_tracking(is_first_batch=True)``, which
    cascade-deletes the prior winner's ``memory_units`` for that
    document. The loser's final ANN pass then attempted to insert
    ``memory_links`` referencing now-deleted units → FK violations on
    ``fk_memory_links_from_unit_id_memory_units``, partial document
    state, and worker thread exhaustion from sentence-transformer pools
    spun up per concurrent child. See issue #1795.

    Items smaller than the budget are still packed together so genuinely
    independent items keep cross-worker parallelism.
    """
    children: list[list[RetainContentDict]] = []
    current: list[RetainContentDict] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        if current:
            children.append(current)
            current = []
            current_tokens = 0

    for item in contents:
        item_tokens = count_tokens(item.get("content", "") or "")

        if item_tokens > tokens_per_batch:
            # Oversized: flush in-flight items into their own child,
            # then emit this item AS-IS (un-chunked) into its own child.
            # The worker will sequentially chunk it inside retain_batch_async.
            _flush()
            children.append([item])
            continue

        if current and current_tokens + item_tokens > tokens_per_batch:
            _flush()
        current.append(item)
        current_tokens += item_tokens

    _flush()
    return children


def _is_invalid_embedding_dimension_error(e: Exception) -> bool:
    """Return True for deterministic embedding-dimension failures.

    These errors come from either PR #1670's preflight validation
    ("embedding 0 has dimension 0; expected 384") or from pgvector itself
    ("different vector dimensions 384 and 0"). Retrying the same poisoned
    embedding response only burns worker slots; a fresh retain request or a
    fixed embedding backend is required.
    """
    message = str(e).lower()
    return "different vector dimensions" in message or (
        "embedding" in message and "dimension" in message and "expected" in message
    )


def _is_non_retryable_task_error(e: Exception) -> bool:
    """Classify deterministic task failures that should skip worker retry."""
    return (
        isinstance(e, asyncpg.exceptions.IntegrityConstraintViolationError)
        or _is_oracledb_integrity_error(e)
        or _is_invalid_embedding_dimension_error(e)
    )


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


def _recall_scoring_now(question_date: datetime | None) -> datetime:
    """Return the reference time for recall scoring boosts."""
    if question_date is None:
        return utcnow()
    if question_date.tzinfo is None or question_date.utcoffset() is None:
        return question_date.replace(tzinfo=UTC)
    return question_date.astimezone(UTC)


# Logger for memory system
logger = logging.getLogger(__name__)

from .db_utils import acquire_with_retry


def _get_tiktoken_encoding():
    """Get cached tiktoken encoding (cl100k_base for GPT-4/3.5).

    Returns a wrapper that tolerates special-token literals in user content
    (see hindsight_api.engine.token_encoding).
    """
    return get_token_encoding()


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
            elif memory_llm_provider.lower() == "ollama-cloud":
                memory_llm_base_url = "https://ollama.com/v1"
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
        self._retain_entity_resolution_batch_size = config.retain_entity_resolution_batch_size

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
            reasoning_effort=config.llm_reasoning_effort,
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.llm_litellmrouter_config,
            bedrock_service_tier=config.llm_bedrock_service_tier,
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
            elif retain_provider.lower() == "ollama-cloud":
                retain_base_url = "https://ollama.com/v1"
            else:
                retain_base_url = ""

        self._retain_llm_config = LLMConfig(
            provider=retain_provider,
            api_key=retain_api_key,
            base_url=retain_base_url,
            model=retain_model,
            reasoning_effort=config.llm_reasoning_effort,
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.retain_llm_litellmrouter_config or config.llm_litellmrouter_config,
            bedrock_service_tier=config.llm_bedrock_service_tier,
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
            elif reflect_provider.lower() == "ollama-cloud":
                reflect_base_url = "https://ollama.com/v1"
            else:
                reflect_base_url = ""

        self._reflect_llm_config = LLMConfig(
            provider=reflect_provider,
            api_key=reflect_api_key,
            base_url=reflect_base_url,
            model=reflect_model,
            reasoning_effort=config.llm_reasoning_effort,
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.reflect_llm_litellmrouter_config or config.llm_litellmrouter_config,
            bedrock_service_tier=config.llm_bedrock_service_tier,
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
            elif consolidation_provider.lower() == "ollama-cloud":
                consolidation_base_url = "https://ollama.com/v1"
            else:
                consolidation_base_url = ""

        self._consolidation_llm_config = LLMConfig(
            provider=consolidation_provider,
            api_key=consolidation_api_key,
            base_url=consolidation_base_url,
            model=consolidation_model,
            reasoning_effort=config.llm_reasoning_effort,
            extra_body=config.llm_extra_body,
            default_headers=config.llm_default_headers,
            litellmrouter_config=config.consolidation_llm_litellmrouter_config or config.llm_litellmrouter_config,
            bedrock_service_tier=config.llm_bedrock_service_tier,
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
        )

        # Per-bank LLM request tracer (disabled by default). Registered as a
        # GenAI span recorder so it captures the same record_llm_call(...) events
        # providers already emit, alongside the OpenTelemetry exporter.
        self._llm_recorder = LLMTraceRecorder(
            pool_getter=lambda: self._backend,
            schema_getter=get_current_schema,
            enabled=config.llm_trace_enabled,
            allowed_scopes=config.llm_trace_scopes,
            max_chars=config.llm_trace_max_chars,
        )
        from ..tracing import register_span_recorder

        register_span_recorder(self._llm_recorder)

        # Background maintenance loop (retention sweeps + consolidation reconcile),
        # created in initialize() once the pool/backend is ready.
        from .maintenance import MaintenanceLoop

        self._maintenance_loop: MaintenanceLoop | None = None

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

        # Load memory defense extension; default to the regex extension when the
        # env var is unset. Lazy imports avoid a circular dependency:
        # extensions/__init__ imports MCPExtension which imports MemoryEngine at
        # module level.
        from ..extensions.builtin.memory_defense_regex import (  # noqa: PLC0415
            MemoryDefenseRegexExtension,
        )
        from ..extensions.context import DefaultExtensionContext  # noqa: PLC0415
        from ..extensions.loader import load_extension  # noqa: PLC0415
        from ..extensions.memory_defense import MemoryDefenseExtension  # noqa: PLC0415

        # Build the extension context now; webhook_manager is populated later in
        # initialize() once the pool is ready.  current_schema is a per-request
        # value written by _authenticate() and execute_task().
        self._ext_ctx = DefaultExtensionContext(
            database_url=config.database_url or "",
            memory_engine=self,
            webhook_manager=None,
            current_schema=None,
        )

        loaded = load_extension("MEMORY_DEFENSE", MemoryDefenseExtension, context=self._ext_ctx)
        if loaded is not None:
            self._memory_defense: MemoryDefenseExtension = loaded
        else:
            regex_defense = MemoryDefenseRegexExtension({})
            regex_defense.set_context(self._ext_ctx)
            self._memory_defense = regex_defense

        # Cache for get_bank_stats — short TTL + concurrent-loader coalescing.
        # The query joins memory_links to memory_units and can be a multi-second
        # parallel scan on large banks; a single polling client used to be able
        # to pin the primary by issuing several concurrent calls.
        self._bank_stats_cache = BankStatsCache(
            ttl_seconds=config.bank_stats_cache_ttl_seconds,
            max_entries=config.bank_stats_cache_max_entries,
        )

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

        from hindsight_api.extensions import OperationValidationError

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
        self._ext_ctx.current_schema = tenant_context.schema_name
        return tenant_context.schema_name

    async def _handle_import_documents(self, task_dict: dict[str, Any]):
        """Handler for async document-import tasks.

        Retrieves the stashed archive, runs the deterministic import, records the
        imported/skipped counts in the operation's ``result_metadata``, and
        deletes the archive. ``execute_task`` marks the operation completed.
        """
        import json

        bank_id = task_dict.get("bank_id")
        storage_key = task_dict.get("storage_key")
        on_conflict = task_dict.get("on_conflict", "skip")
        operation_id = task_dict.get("operation_id")
        if not bank_id or not storage_key:
            raise ValueError("bank_id and storage_key are required for import_documents task")

        from hindsight_api.models import RequestContext

        context = RequestContext(
            internal=True,
            user_initiated=True,
            tenant_id=task_dict.get("_tenant_id"),
            api_key_id=task_dict.get("_api_key_id"),
            retry_count=task_dict.get("_retry_count", 0),
        )

        archive_bytes = await self._file_storage.retrieve(storage_key)
        result = await self._run_import_documents(bank_id, archive_bytes, on_conflict, context)

        if operation_id:
            counts = {
                "documents_imported": result.documents_imported,
                "documents_skipped": result.documents_skipped,
                "facts_imported": result.facts_imported,
                "observations_imported": result.observations_imported,
                "observations_skipped": result.observations_skipped,
                "skipped_document_ids": result.skipped_document_ids,
                "remapped_document_ids": result.remapped_document_ids,
            }
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"UPDATE {fq_table('async_operations')} "
                    f"SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $1::jsonb "
                    f"WHERE operation_id = $2",
                    json.dumps(counts, default=_json_default),
                    uuid.UUID(operation_id),
                )

        # Best-effort cleanup of the transient upload.
        try:
            await self._file_storage.delete(storage_key)
        except Exception:
            logger.warning("Failed to delete import archive %s", storage_key, exc_info=True)

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
            outbox_callback_factory=self._build_retain_outbox_callback_factory(
                bank_id=bank_id,
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
            observation_scopes=task_dict.get("observation_scopes"),
        )

        logger.info(f"[CONSOLIDATION] bank={bank_id} completed: {result.get('memories_processed', 0)} processed")
        return result

    async def _handle_graph_maintenance(self, task_dict: dict[str, Any]):
        """Handler for graph_maintenance tasks. Drains graph_maintenance_queue for the bank."""
        bank_id = task_dict.get("bank_id")
        if not bank_id:
            raise ValueError("bank_id is required for graph_maintenance task")

        from hindsight_api.models import RequestContext

        from .graph_maintenance import run_graph_maintenance_job

        internal_context = RequestContext(
            internal=True,
            tenant_id=task_dict.get("_tenant_id"),
            api_key_id=task_dict.get("_api_key_id"),
            retry_count=task_dict.get("_retry_count", 0),
        )
        return await run_graph_maintenance_job(
            memory_engine=self,
            bank_id=bank_id,
            request_context=internal_context,
            operation_id=task_dict.get("operation_id"),
        )

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

    @_bind_bank_id("task_dict", key="bank_id")
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
            self._ext_ctx.current_schema = schema

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
                elif task_type == "import_documents":
                    await self._handle_import_documents(task_dict)
                elif task_type == "consolidation":
                    consolidation_result = await self._handle_consolidation(task_dict)
                elif task_type == "graph_maintenance":
                    await self._handle_graph_maintenance(task_dict)
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
                elif _is_non_retryable_task_error(e):
                    # Non-retryable: deterministic task failures (integrity violations,
                    # invalid embedding dimensions, etc.) will not succeed by rerunning
                    # the same payload. Retrying just burns worker capacity.
                    logger.error(f"Not retrying task {task_type} (deterministic failure): {type(e).__name__}")
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

                        # When another consolidation is already pending for the same
                        # bank, skip the retry. The pending op will process the same
                        # unconsolidated rows when it runs, so retrying ours just
                        # multiplies retry budgets during a long transient outage
                        # (every retain enqueues a fresh op, each independently
                        # consuming `_retry_count` slots — a retry storm).
                        bank_id_for_dedup = task_dict.get("bank_id", "")
                        if bank_id_for_dedup and await self._has_other_pending_consolidation(
                            bank_id=bank_id_for_dedup,
                            operation_id=operation_id,
                        ):
                            logger.info(
                                f"Consolidation {operation_id} for bank {bank_id_for_dedup} hit "
                                f"transient error; another consolidation is already pending for "
                                f"this bank — skipping retry."
                            )
                            raise

                        # Indefinite retry with capped exponential backoff.
                        # Transient outages (LLM provider down, DB flapping) must
                        # eventually recover; the alternative (cap after 3 retries
                        # and mark failed) silently dead-letters the bank's backlog.
                        # The dedup-by-bank guard above prevents this from causing
                        # a retry storm when multiple ops exist for the same bank.
                        retry_count = task_dict.get("_retry_count", 0)
                        backoff = _consolidation_retry_backoff_seconds(retry_count)
                        raise RetryTaskAt(
                            retry_at=datetime.now(UTC) + timedelta(seconds=backoff),
                            message=str(e),
                        )

                    # Retryable: use RetryTaskAt if under the retry limit, else re-raise (poller marks failed).
                    # Retry count and backoff come from config (HINDSIGHT_API_WORKER_MAX_RETRIES and
                    # HINDSIGHT_API_WORKER_TASK_RETRY_BACKOFF_SECONDS). Defaults of 3 x 60s give a
                    # 4-minute total window; operators expecting a longer provider outage can raise them.
                    config = get_config()
                    retry_count = task_dict.get("_retry_count", 0)
                    if retry_count < config.worker_max_retries:
                        raise RetryTaskAt(
                            retry_at=datetime.now(UTC) + timedelta(seconds=config.worker_task_retry_backoff_seconds),
                            message=str(e),
                        )
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
        contents: list[RetainContentDict],
        operation_id: str | None,
        schema: str | None = None,
    ) -> RetainOutboxCallback | None:
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

    def _build_retain_outbox_callback_factory(
        self,
        bank_id: str,
        operation_id: str | None,
        schema: str | None = None,
    ) -> RetainOutboxCallbackFactory:
        """Build retain outbox callbacks for grouped retain batches.

        The factory captures one operation_id so per-document webhook events from
        the same logical retain operation keep a shared event operation_id.
        """
        op_id = operation_id or uuid.uuid4().hex

        def _factory(
            callback_contents: list[RetainContentDict],
        ) -> RetainOutboxCallback | None:
            return self._build_retain_outbox_callback(
                bank_id=bank_id,
                contents=callback_contents,
                operation_id=op_id,
                schema=schema,
            )

        return _factory

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

    async def _write_operation_progress(
        self,
        operation_id: str | None,
        *,
        stage: str,
        processed: int | None = None,
        total: int | None = None,
        detail: dict[str, int] | None = None,
    ) -> None:
        """Persist a last-known progress snapshot for a long-running async operation.

        Merges a ``progress`` object into ``async_operations.result_metadata`` (top-level
        ``||`` merge, so sibling keys such as ``is_parent`` survive) and bumps
        ``updated_at``. Callers invoke this at coarse phase/batch boundaries — not per
        row — so an operator polling the operation status API can see the current stage
        and counters and tell a healthy long-running job from a frozen one.

        Best-effort: a failed heartbeat must never fail the underlying job, so all errors
        are swallowed with a debug log. A ``None`` operation_id (synchronous / untracked
        call sites) is a no-op.
        """
        if not operation_id:
            return
        snapshot: dict[str, Any] = {"stage": stage, "at": datetime.now(UTC).isoformat()}
        if processed is not None:
            snapshot["processed"] = processed
        if total is not None:
            snapshot["total"] = total
        if detail:
            snapshot["detail"] = detail
        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                await conn.execute(
                    f"UPDATE {fq_table('async_operations')} "
                    f"SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $2::jsonb, "
                    f"updated_at = now() WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                    json.dumps({"progress": snapshot}),
                )
        except Exception as e:
            logger.debug(f"Failed to write operation progress for {operation_id}: {e}")

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

    async def _write_retain_outcome_metadata(self, operation_id: str | None, unit_ids: list[list[str]]) -> None:
        """Persist completed retain outcome fields before the operation is marked completed."""
        if not operation_id:
            return

        unit_ids_count = sum(len(group) for group in unit_ids)
        try:
            backend = await self._get_backend()
            async with acquire_with_retry(backend) as conn:
                row = await conn.fetchrow(
                    f"SELECT result_metadata FROM {fq_table('async_operations')} WHERE operation_id = $1",
                    uuid.UUID(operation_id),
                )
                if not row:
                    return

                metadata = conn.parse_json(row["result_metadata"]) or {}
                extraction_errors = RetainExtractionErrors()
                extraction_errors.merge_metadata(metadata)
                outcome = RetainOutcomeMetadata(
                    unit_ids_count=unit_ids_count,
                    extraction_errors_count=extraction_errors.count,
                    extraction_errors_sample=extraction_errors.sample,
                )

                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $2::jsonb,
                        updated_at = now()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(operation_id),
                    json.dumps(outcome.to_dict()),
                )
        except Exception as e:
            # Best-effort, but log loudly: the whole point of this metadata is to
            # give clients a reliable success/silent-failure signal, so a missing
            # write silently regresses them to the ambiguous pre-fix behaviour.
            logger.warning(f"Failed to write retain outcome metadata for {operation_id}: {e}")

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
            # child's updated status. Pull result_metadata for completed
            # children so the parent exposes the same outcome counters as the
            # individual retain operations. Pull error_message too so a parent that
            # fails can inherit a representative child reason -- otherwise
            # downstream consumers (dashboards, alert filters) lose the actual
            # cause once a batch has children. See the worker poller's
            # _summarise_child_error_messages for the propagation rationale.
            siblings = await conn.fetch(
                f"""
                SELECT status, error_message, result_metadata
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
                outcome_aggregate = RetainOutcomeAggregate()
                for sibling in siblings:
                    sibling_metadata = conn.parse_json(sibling["result_metadata"]) or {}
                    outcome_aggregate.add_metadata(sibling_metadata)
                await conn.execute(
                    f"""
                    UPDATE {fq_table("async_operations")}
                    SET status = $2,
                        result_metadata = COALESCE(result_metadata, '{{}}'::jsonb) || $3::jsonb,
                        updated_at = NOW(),
                        completed_at = NOW()
                    WHERE operation_id = $1
                    """,
                    uuid.UUID(parent_operation_id),
                    new_status,
                    json.dumps(outcome_aggregate.to_outcome_metadata().to_dict()),
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

                # Validate batch API compatibility: if retain_batch_enabled is set,
                # the retain LLM provider must actually support the batch API.
                # Otherwise the server would silently fall back to sync mode on
                # every retain, which is confusing and wastes a config knob.
                config = get_config()
                if config.retain_batch_enabled:
                    supports_batch = await self._retain_llm_config._provider_impl.supports_batch_api()
                    if not supports_batch:
                        raise RuntimeError(
                            f"Configuration error: HINDSIGHT_API_RETAIN_BATCH_ENABLED=true "
                            f"but the retain LLM provider '{self._retain_llm_config.provider}' "
                            f"does not support the batch API. Either switch to a provider "
                            f"that supports batch operations (e.g. 'openai', 'groq', 'gemini') or "
                            f"set HINDSIGHT_API_RETAIN_BATCH_ENABLED=false."
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

        # Run pg0 and selected model initializations in parallel.
        # Cap the whole thing with a wall-clock timeout so a hung init task
        # (e.g. an offline HuggingFace download or an unreachable provider)
        # fails the daemon fast instead of leaving it stuck forever — neither
        # started nor errored.
        init_timeout = get_config().model_init_timeout
        try:
            await asyncio.wait_for(asyncio.gather(*init_tasks), timeout=init_timeout)
        except TimeoutError as e:
            raise RuntimeError(
                f"Model/connection initialization did not complete within "
                f"{init_timeout:g}s. A model load (embeddings/cross-encoder) or LLM "
                f"verification is likely blocked — e.g. an offline model download or "
                f"an unreachable provider. Increase {ENV_MODEL_INIT_TIMEOUT} if the "
                f"first-time model download legitimately needs more time."
            ) from e

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
                                self.db_url,
                                text_search_extension=config.text_search_extension,
                                pg_search_tokenizer=config.text_search_extension_pg_search_tokenizer,
                                schema=schema,
                            )

        logger.info(f"Connecting to database at {mask_network_location(self.db_url)}")

        # Create SQL dialect via abstraction layer
        # (backend was created in __init__ so we can use it for migrations and task backend)
        self._dialect = create_sql_dialect(self._database_backend_type)

        stmt_timeout_s = self._db_statement_timeout
        text_search_extension = get_config().text_search_extension

        # Per-connection initialization callback (PostgreSQL-specific for now)
        async def _init_connection(conn: asyncpg.Connection) -> None:
            # VectorChord BM25 registers its objects in dedicated schemas
            # (vchord_bm25 -> bm25_catalog, pg_tokenizer -> tokenizer_catalog).
            # The BM25 distance operator `<&>` resolves its operand types via the
            # session search_path, so a connection that lacks these schemas fails
            # recall with `type "bm25vector" does not exist` (and retain with
            # `function tokenize(...) does not exist`). The official vchord-suite
            # image masks this by shipping them in search_path; an external
            # Postgres does not, so we add them ourselves. Tenant tables are always
            # accessed via fully-qualified names (fq_table), so this does not
            # affect schema isolation. Only needed for the vchord backend.
            if text_search_extension == "vchord":
                await conn.execute('SET search_path TO "$user", public, bm25_catalog, tokenizer_catalog')

            # SET (not SET LOCAL) so per-backend ANN tuning persists for the
            # connection lifetime. The dispatcher returns only safe, portable
            # knobs for the configured extension; VectorChord probe tuning is
            # index-shaped and should be stored on vchordrq indexes instead.
            for guc, value in ann_search_tuning_settings(configured_vector_extension(), kind="high_recall"):
                try:
                    await conn.execute(f"SET {guc} = {value}")
                except asyncpg.exceptions.PostgresError:
                    # Defensive net for env mis-config (e.g. extension configured
                    # for vchord but the cluster only has pgvector). Narrow to
                    # PostgresError so genuine bugs in the pool/conn layer surface
                    # instead of being silently logged at debug level.
                    logger.debug("Could not set %s — extension may not support it", guc)

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
            entity_resolution_batch_size=self._retain_entity_resolution_batch_size,
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
        # Propagate the now-ready webhook manager to the extension context so
        # that the Memory Defense extension can fire webhooks.
        self._ext_ctx.webhook_manager = self._webhook_manager
        logger.debug("Webhook manager initialized")

        # Long-lived HTTP client for webhook delivery tasks
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Set executor for task backend and initialize
        self._task_backend.set_executor(self.execute_task)
        await self._task_backend.initialize()

        # Start the background maintenance loop: cross-tenant retention sweeps
        # (audit_log, llm_requests) plus the consolidation reconcile that
        # re-schedules banks with eligible-but-unscheduled facts.
        from .maintenance import MaintenanceLoop

        self._maintenance_loop = MaintenanceLoop(self)
        self._maintenance_loop.start()

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

        # Stop the background maintenance loop (retention sweeps + reconcile)
        if self._maintenance_loop is not None:
            await self._maintenance_loop.stop()

        # Unregister the LLM trace recorder span hook
        from ..tracing import unregister_span_recorder

        unregister_span_recorder(self._llm_recorder)

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

    @_bind_bank_id()
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

    @_bind_bank_id()
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
        outbox_callback: RetainOutboxCallback | None = None,
        outbox_callback_factory: RetainOutboxCallbackFactory | None = None,
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
                contents = cast(list[RetainContentDict], result.contents)

        # Engine-owned copy: the orchestrator clears per-item "content" strings
        # after building the document's combined text (memory pressure
        # optimization, see retain/orchestrator.py). Without an internal copy
        # those mutations leak back to the caller's dicts.
        contents = cast(list[RetainContentDict], [dict(c) for c in contents])

        # Sanitize content/context at ingress so lone UTF-16 surrogates (e.g. a
        # half-emoji a client serialized as a `\udXXX` escape) cannot crash the
        # embedder, cross-encoder, or logging with an HTTP 500 (see issue #1875).
        for item in contents:
            if "content" in item:
                item["content"] = sanitize_text(item["content"]) or ""
            if item.get("context"):
                item["context"] = sanitize_text(item["context"]) or ""

        # Apply batch-level document_id to contents that don't have their own (backwards compatibility)
        if document_id:
            for item in contents:
                if "document_id" not in item:
                    item["document_id"] = document_id

        if outbox_callback is None and outbox_callback_factory is not None:
            outbox_callback = outbox_callback_factory(contents)

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

        # Append mode rebuilds the full document by reading back the previously
        # stored original_text and prepending it. With store_document_text
        # disabled there is no stored text to read, so the append would silently
        # drop all prior content — reject it explicitly instead.
        if not get_config().store_document_text:
            for item in contents:
                if item.get("update_mode") == "append":
                    raise ValueError(
                        "update_mode='append' is not supported when HINDSIGHT_API_STORE_DOCUMENT_TEXT "
                        "is disabled: the prior document text is not stored and cannot be appended to. "
                        "Use update_mode='replace' instead."
                    )

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

            split = _split_contents_into_sub_batches(contents, tokens_per_batch)
            sub_batches = split.sub_batches
            origin_indices = split.origin_indices
            document_body_overrides = split.document_body_overrides

            sub_batch_sizes = [len(b) for b in sub_batches]
            # Keep the per-sub-batch sizes log compact when an oversize
            # single item gets chunked into many [1]-sized sub-batches.
            if len(sub_batches) <= 20:
                logger.info(f"Split into {len(sub_batches)} sub-batches: {sub_batch_sizes} items each")
            else:
                logger.info(
                    f"Split into {len(sub_batches)} sub-batches "
                    f"(items per sub-batch: min={min(sub_batch_sizes)}, "
                    f"max={max(sub_batch_sizes)}, total={sum(sub_batch_sizes)})"
                )

            # Preserve the public contract: one result list per input
            # content. When an oversize single item is chunked across
            # multiple sub-batches, unit_ids from every chunk get
            # appended back into that input's result slot.
            per_input_results: list[list[str]] = [[] for _ in contents]

            # Per-document chunk_index offsets. When an oversized single item is
            # sliced into several sub-batches that all share one document_id and
            # run sequentially, each sub-batch must continue the document's
            # chunk_index sequence rather than restart at 0 — otherwise the
            # derived chunk_id ({bank}_{doc}_{index}) collides and later
            # sub-batches overwrite earlier chunks, leaving only one sub-batch's
            # worth of chunks/memories (issue #1888). Counting uses the same
            # bank-resolved, strategy-applied chunk size the orchestrator chunks
            # with, so the offsets match the chunk_index values it assigns.
            from .retain import fact_extraction, fact_storage

            sub_chunk_size = await self._resolve_retain_chunk_size(bank_id, request_context, strategy)
            chunk_offsets: dict[str, int] = {}

            # In update_mode="append", retain_batch prepends the existing document
            # body to the FIRST sub-batch as an extra content item before chunking
            # (see orchestrator.retain_batch), consuming chunks(existing_body)
            # additional chunk_index slots ahead of that sub-batch's own content.
            # Capture that chunk count per document up front — the first sub-batch
            # overwrites documents.original_text when it commits, so it can't be
            # read back afterwards — and fold it into the offset so later
            # sub-batches continue past the prepended chunks instead of colliding.
            append_prepend_chunks: dict[str, int] = {}
            backend = await self._get_backend()
            append_doc_ids: set[str] = set()
            for item in contents:
                item_doc_id = item.get("document_id")
                if item.get("update_mode") == "append" and item_doc_id:
                    append_doc_ids.add(item_doc_id)
            for append_doc_id in append_doc_ids:
                async with acquire_with_retry(backend) as conn:
                    existing_text = await fact_storage.get_document_content(conn, bank_id, append_doc_id)
                if existing_text:
                    append_prepend_chunks[append_doc_id] = len(
                        fact_extraction.chunk_text(existing_text, sub_chunk_size)
                    )

            for i, (sub_batch, sub_origins) in enumerate(zip(sub_batches, origin_indices), 1):
                # Checkpoint: abort if the operation was deleted (bank was deleted) between sub-batches.
                if operation_id and not await self._check_op_alive(operation_id):
                    logger.info(
                        f"[BATCH_RETAIN] bank={bank_id} operation {operation_id} cancelled (bank deleted), stopping after {i - 1}/{len(sub_batches)} sub-batches"
                    )
                    if return_usage:
                        return per_input_results, total_usage
                    return per_input_results

                sub_batch_tokens = sum(count_tokens(item.get("content", "")) for item in sub_batch)
                logger.info(
                    f"Processing sub-batch {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_tokens:,} tokens"
                )
                # Live worker stage for the in-flight sub-batch; the durable progress
                # snapshot is written *after* the sub-batch commits (below) so processed
                # reflects work actually done and reaches total on completion.
                set_stage(f"batch_retain.sub_batch.{i}")

                # Resolve the document this sub-batch writes to so we can offset
                # its chunk_index past chunks already stored by earlier
                # sub-batches of the same document. Only the oversized-single-item
                # split shares a document_id across sub-batches; packed multi-item
                # sub-batches carry distinct document_ids (offset stays 0).
                sub_doc_id = document_id or (sub_batch[0].get("document_id") if len(sub_batch) == 1 else None)
                sub_offset = chunk_offsets.get(sub_doc_id, 0) if sub_doc_id else 0

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
                    outbox_callback_factory=outbox_callback_factory if i == len(sub_batches) else None,
                    document_body_override=document_body_overrides[i - 1],
                    chunk_index_offset=sub_offset,
                )

                # Advance the document's chunk_index cursor by the number of
                # chunks this sub-batch produced (computed with the same chunk
                # size the orchestrator uses), so the next sub-batch sharing the
                # document continues the sequence.
                if sub_doc_id:
                    sub_chunk_count = sum(
                        len(fact_extraction.chunk_text(item.get("content", "") or "", sub_chunk_size))
                        for item in sub_batch
                    )
                    # retain_batch only prepends the existing body on the global
                    # first sub-batch (is_first_batch == i == 1), so fold its chunk
                    # count in only there.
                    if i == 1:
                        sub_chunk_count += append_prepend_chunks.get(sub_doc_id, 0)
                    chunk_offsets[sub_doc_id] = sub_offset + sub_chunk_count
                # sub_results aligns 1:1 with sub_batch items; map each
                # back to its source input via origin_indices so callers
                # iterating with ``zip(contents, results)`` still align.
                for sub_idx, origin_idx in enumerate(sub_origins):
                    if sub_idx < len(sub_results):
                        per_input_results[origin_idx].extend(sub_results[sub_idx])
                total_usage = total_usage + sub_usage
                if total_processed_content_tokens is None or sub_processed is None:
                    total_processed_content_tokens = None
                else:
                    total_processed_content_tokens = total_processed_content_tokens + sub_processed
                # Per-sub-batch progress is intentionally not written here: the streaming
                # retain pipeline emits finer-grained "storing N/total chunks" snapshots
                # via progress_callback as each sub-batch's chunks commit.

            total_time = time.time() - start_time
            logger.info(
                f"RETAIN_BATCH_ASYNC (chunked) COMPLETE: {len(per_input_results)} results from {len(contents)} contents in {total_time:.3f}s"
            )
            result = per_input_results
        else:
            # Small batch - use internal method directly (single sub-batch).
            set_stage("batch_retain.sub_batch.1")
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
                outbox_callback_factory=outbox_callback_factory,
            )
            # Progress for this path is emitted by the streaming pipeline as
            # "storing N/total chunks" via progress_callback (see _retain_batch_async_internal).

        await self._write_retain_outcome_metadata(operation_id, result)

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

        # Same async side effects every fact insert triggers (retain or import).
        await self._submit_post_insert_maintenance(bank_id, request_context)

        if return_usage:
            return result, total_usage
        return result

    async def _submit_post_insert_maintenance(
        self,
        bank_id: str,
        request_context: "RequestContext",
        config: HindsightConfig | None = None,
    ) -> None:
        """Submit the async side effects that follow any fact insert (retain or import).

        Shared by the retain pipeline and the document-import pipeline so imported
        documents aren't second-class citizens:
          * auto-consolidation (when observations + auto-consolidation are enabled
            for the bank) so freshly inserted facts get observations;
          * graph maintenance, which short-circuits when no cleanup work was
            enqueued, so a plain insert pays a single cheap indexed SELECT here.

        Both are non-critical: failures are logged, never raised, so they can't
        fail the operation that produced the facts. Pass ``config`` when the caller
        already resolved it to avoid a redundant lookup.
        """
        if config is None:
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
        if config.enable_observations and config.enable_auto_consolidation:
            try:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit consolidation task for bank {bank_id}: {e}")
        try:
            await self.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
        except Exception as e:
            logger.warning(f"Failed to submit graph maintenance task for bank {bank_id}: {e}")

    async def _resolve_retain_chunk_size(
        self,
        bank_id: str,
        request_context: "RequestContext",
        strategy: str | None,
    ) -> int:
        """Resolve the effective ``retain_chunk_size`` for a bank.

        Mirrors the bank-config + strategy resolution that
        ``_retain_batch_async_internal`` applies before handing config to the
        orchestrator, so chunk-count estimates used for per-document
        chunk_index offsets match the chunk_index values the orchestrator
        actually assigns.
        """
        from hindsight_api.config_resolver import apply_strategy

        resolved_config = await self._config_resolver.resolve_full_config(bank_id, request_context)
        effective_strategy = strategy or resolved_config.retain_default_strategy
        if effective_strategy:
            resolved_config = apply_strategy(resolved_config, effective_strategy)
        return getattr(resolved_config, "retain_chunk_size", 3000)

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
        outbox_callback: RetainOutboxCallback | None = None,
        outbox_callback_factory: RetainOutboxCallbackFactory | None = None,
        strategy: str | None = None,
        document_body_override: str | None = None,
        chunk_index_offset: int = 0,
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

        await self._get_backend()

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
            retain_llm = self._retain_llm_config.with_config(resolved_config, bank_id=bank_id, operation="retain")
            result = await orchestrator.retain_batch(
                pool=self._backend,
                embeddings_model=self.embeddings,
                llm_config=retain_llm,
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
                outbox_callback_factory=outbox_callback_factory,
                db_semaphore=self._put_semaphore,
                document_body_override=document_body_override,
                chunk_index_offset=chunk_index_offset,
                # Stream chunk-level "storing N/total" progress to the operation row as
                # the document's chunks commit (more useful than the coarse sub-batch tick).
                progress_callback=self._write_operation_progress,
                webhook_manager=self._webhook_manager,
                memory_defense_extension=self._memory_defense,
                audit_logger=self._audit_logger,
            )
            # Map the created facts onto this retain's trace so the trace view can
            # show which memories the ingestion produced. result[0] is the
            # per-content-item list of created unit ids (see retain_batch).
            created_ids = [uid for group in result[0] for uid in group]
            # Fire-and-forget: the mapping is patched on a background task so it
            # never adds latency to the retain response.
            self._llm_recorder.attach_memory_ids(trace_context_of(retain_llm), created=created_ids)
            return result

    async def export_documents_async(
        self,
        bank_id: str,
        request_context: "RequestContext",
        document_ids: list[str] | None = None,
        include_observations: bool = False,
    ) -> bytes:
        """Export documents from a bank into a transfer ZIP archive (no LLM, no embeddings).

        See :mod:`hindsight_api.engine.transfer`. Embeddings and database ids are
        not included; the archive carries extracted facts, entity canonical
        names, causal links, and chunks so it can be replayed into another bank.
        When ``include_observations`` is set, consolidated observations are also
        exported (and restored on import) instead of being regenerated.
        """
        from .transfer import export_documents

        await self._get_backend()
        return await export_documents(self._backend, bank_id, document_ids, include_observations=include_observations)

    async def import_bank_async(
        self,
        archive_bytes: bytes,
        request_context: "RequestContext",
        *,
        target_bank_id: str | None = None,
        include_history: bool = False,
    ) -> "BankImportResult":
        """Restore a whole bank from an :func:`transfer.export_bank` archive.

        Re-embeds facts with this instance's embedding model and rebuilds links and
        indexes; restores bank config, mental models, directives and webhooks as
        exported (no consolidation/webhooks — a migration restores exact state). The
        target bank must not already exist (import restores a whole bank, not a merge).
        """
        from .transfer import import_bank
        from .transfer.importer import parse_bank_archive

        await self._authenticate_tenant(request_context)
        backend = await self._get_backend()
        # Parse up front so a bad archive fails fast and we can resolve the
        # target bank's config before the restore.
        parsed = parse_bank_archive(archive_bytes)
        bank_id = target_bank_id or parsed.manifest.source_bank_id
        resolved_config = await self._config_resolver.resolve_full_config(bank_id, request_context)
        return await import_bank(
            backend=backend,
            embeddings_model=self.embeddings,
            entity_resolver=self.entity_resolver,
            config=resolved_config,
            format_date_fn=self._format_readable_date,
            archive_bytes=archive_bytes,
            target_bank_id=target_bank_id,
            include_history=include_history,
        )

    async def import_documents_async(
        self,
        bank_id: str,
        archive_bytes: bytes,
        request_context: "RequestContext",
        on_conflict: str = "skip",
    ) -> dict[str, Any]:
        """Submit an async document-import operation and return its ``operation_id``.

        The archive is validated up front (so a bad zip fails fast), stashed in
        file storage, and processed by a worker — or inline when the engine uses
        a ``SyncTaskBackend`` (e.g. in tests). Poll the operations endpoint for
        status; the imported/skipped counts land in ``result_metadata``.
        Re-embeds facts and re-resolves entities — no LLM extraction is run.
        """
        from .transfer.importer import parse_archive

        if on_conflict not in ("skip", "replace", "new-id"):
            raise ValueError(f"Invalid on_conflict '{on_conflict}'; expected skip|replace|new-id")
        # Validate synchronously so a malformed/unsupported archive surfaces as an
        # immediate error to the caller rather than a background task failure.
        parse_archive(archive_bytes)

        await self._authenticate_tenant(request_context)
        await self._get_backend()
        # Ensure the bank (and its per-bank vector indexes) exist before inserts.
        # Import has no single write transaction to join — the archive is written
        # by a worker later — so the bank is created on its own connection.
        await self._ensure_bank_exists(bank_id, request_context)

        # Stash the archive in file storage and reference it by key in the task
        # payload, rather than base64-ing megabytes into the operation JSON.
        storage_key = f"banks/{bank_id}/imports/{uuid.uuid4()}/transfer.zip"
        await self._file_storage.store(
            file_data=archive_bytes,
            key=storage_key,
            metadata={"content_type": "application/zip", "bank_id": bank_id},
        )

        task_payload: dict[str, Any] = {"storage_key": storage_key, "on_conflict": on_conflict}
        if request_context.tenant_id:
            task_payload["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            task_payload["_api_key_id"] = request_context.api_key_id

        return await self._submit_async_operation(
            bank_id,
            operation_type="import_documents",
            task_type="import_documents",
            task_payload=task_payload,
        )

    async def _run_import_documents(
        self,
        bank_id: str,
        archive_bytes: bytes,
        on_conflict: str,
        request_context: "RequestContext",
    ) -> "ImportResult":
        """Run the deterministic import inline (shared by the worker handler).

        After inserting, runs the same post-retain side effects as a normal
        retain so imported documents aren't second-class citizens:
          * retain.completed webhooks (one per imported document, fired
            transactionally inside each document's insert);
          * auto-consolidation (so imported facts get observations — when the
            archive already carried observations, their sources are marked
            consolidated, so consolidation safely skips them);
          * graph maintenance (replace/new-id imports cascade-delete old data and
            enqueue relink work).
        """
        from .transfer import import_documents

        backend = await self._get_backend()
        # Imports insert across many per-document transactions, so the bank is
        # created up front on its own connection rather than coupled to a write.
        await self._ensure_bank_exists(bank_id, request_context)
        resolved_config = await self._config_resolver.resolve_full_config(bank_id, request_context)
        outbox_factory = self._build_retain_outbox_callback_factory(
            bank_id=bank_id, operation_id=None, schema=_current_schema.get()
        )
        result = await import_documents(
            backend=backend,
            embeddings_model=self.embeddings,
            entity_resolver=self.entity_resolver,
            config=resolved_config,
            format_date_fn=self._format_readable_date,
            bank_id=bank_id,
            archive_bytes=archive_bytes,
            on_conflict=on_conflict,
            outbox_callback_factory=outbox_factory,
        )

        # Fire the post-retain extension hook (usage tracking / metrics /
        # notifications) once per imported document, mirroring retain. Import runs
        # no LLM extraction, so token counts are zero and processed_content_tokens
        # is 0 ("nothing went through the extraction pipeline") — extensions that
        # meter LLM/extraction cost therefore correctly bill an import as free.
        if self._operation_validator:
            from hindsight_api.extensions import RetainResult

            for doc in result.imported_documents:
                try:
                    await self._operation_validator.on_retain_complete(
                        RetainResult(
                            bank_id=bank_id,
                            contents=[{"content": doc.content}],
                            request_context=request_context,
                            document_id=doc.document_id,
                            fact_type_override=None,
                            unit_ids=[doc.unit_ids],
                            success=True,
                            error=None,
                            llm_input_tokens=0,
                            llm_output_tokens=0,
                            llm_total_tokens=0,
                            processed_content_tokens=0,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Post-import hook error (non-fatal): {e}")

        # Same async side effects every fact insert triggers (retain or import).
        await self._submit_post_insert_maintenance(bank_id, request_context, config=resolved_config)

        return result

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

    @_bind_bank_id()
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
        reranking: RecallReranking = "cross_encoder",
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
            question_date: Optional date when question was asked (for temporal filtering and recency scoring)
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

        # Cooperative cancellation checkpoint: if the client already disconnected
        # while this request waited to be scheduled, abort before doing any work
        # (issue #2122). Further checkpoints sit at each pipeline stage boundary
        # inside _search_with_retries.
        request_context.raise_if_cancelled()

        # Sanitize the query at ingress: a client may serialize a half-emoji as a
        # lone UTF-16 surrogate, which crashes downstream logging, the embedder, and
        # the cross-encoder tokenizer with an HTTP 500 (see issue #1875). Cleaning it
        # here protects every sink that the query flows into.
        query = sanitize_text(query) or ""

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
                            reranking=reranking,
                        )
                        break  # Success - exit retry loop
                    except OperationCancelledError:
                        # Client disconnected — propagate to the HTTP layer (499);
                        # not a failure to retry or report via the post-op hook.
                        raise
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
        reranking: RecallReranking = "cross_encoder",
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
                query_embeddings = await embedding_utils.generate_embeddings_batch(
                    self.embeddings,
                    [query],
                    input_type="query",
                )
                query_embedding = query_embeddings[0]
                step_duration = time.time() - step_start
                log_buffer.append(f"  [1] Generate query embedding: {step_duration:.3f}s")
            finally:
                embedding_span.end()

            if tracer:
                tracer.record_query_embedding(query_embedding)
                tracer.add_phase_metric("generate_query_embedding", step_duration)

            # Cancellation checkpoint: bail before the DB-heavy retrieval stage
            # if the client has gone away (issue #2122).
            if request_context is not None:
                request_context.raise_if_cancelled()

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

            # Cap each source independently before fusion so a single
            # over-expanding backend (e.g. VectorChord returning hundreds of
            # weak candidates) cannot fill the reranker's global budget on its
            # own and crowd the other arms out of the final candidate pool.
            per_source_cap = get_config().recall_max_candidates_per_source
            if per_source_cap > 0:
                from .search.fusion import cap_per_source

                pre_cap_counts = (len(semantic_results), len(bm25_results), len(graph_results))
                semantic_results = cap_per_source(semantic_results, per_source_cap)
                bm25_results = cap_per_source(bm25_results, per_source_cap)
                graph_results = cap_per_source(graph_results, per_source_cap)
                if temporal_results:
                    temporal_results = cap_per_source(temporal_results, per_source_cap)
                if pre_cap_counts != (len(semantic_results), len(bm25_results), len(graph_results)):
                    logger.debug(
                        f"[RECALL {recall_id}] Per-source cap ({per_source_cap}) applied: "
                        f"semantic {pre_cap_counts[0]}->{len(semantic_results)}, "
                        f"bm25 {pre_cap_counts[1]}->{len(bm25_results)}, "
                        f"graph {pre_cap_counts[2]}->{len(graph_results)}"
                    )

            step_duration = time.time() - step_start
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

            # Step 3: Merge ranked lists. RRF by default; interleave (round-robin) when
            # requested by consolidation dedup recall — RRF averages a strong-in-one-arm
            # result down and buried the near-identical "twin" observation below budget
            # (semantic #1 -> outside the shown set), whereas interleave guarantees each
            # arm's top hits a slot. See interleave_fusion docstring.
            step_start = time.time()
            from .search.fusion import interleave_fusion, reciprocal_rank_fusion

            fusion_span = tracer_otel.start_span("hindsight.recall_fusion")
            fusion_span.set_attribute("hindsight.bank_id", bank_id)
            fusion_span.set_attribute("hindsight.semantic_count", len(semantic_results))
            fusion_span.set_attribute("hindsight.bm25_count", len(bm25_results))
            fusion_span.set_attribute("hindsight.graph_count", len(graph_results))
            fusion_span.set_attribute("hindsight.temporal_count", len(temporal_results) if temporal_results else 0)

            try:
                # Merge 3 or 4 result lists depending on temporal constraint
                result_lists = [semantic_results, bm25_results, graph_results]
                if temporal_results:
                    result_lists.append(temporal_results)
                fuse = interleave_fusion if reranking == "interleave" else reciprocal_rank_fusion
                merged_candidates = fuse(result_lists)

                step_duration = time.time() - step_start
                log_buffer.append(
                    f"  [3] {'interleave' if reranking == 'interleave' else 'RRF'} merge: "
                    f"{len(merged_candidates)} unique candidates in {step_duration:.3f}s"
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
            rerank_kind = "cross-encoder"
            try:
                # Pre-filter candidates by RRF before the (optional) cross-encoder.
                # RRF already provides good ranking; this caps cross-encoder cost.
                reranker_max_candidates = get_config().reranker_max_candidates
                if len(merged_candidates) > reranker_max_candidates:
                    # Sort by RRF score (boosted per-strategy if configured) and take top
                    # candidates. The weighted-RRF boost keeps boosted-arm candidates from
                    # being trimmed out of the reranker's global budget.
                    from .search.recall_boost import boosted_rrf_score

                    strategy_boosts = get_config().recall_strategy_boosts
                    merged_candidates.sort(key=lambda mc: boosted_rrf_score(mc, strategy_boosts), reverse=True)
                    pre_filtered_count = len(merged_candidates) - reranker_max_candidates
                    merged_candidates = merged_candidates[:reranker_max_candidates]

                if reranking == "cross_encoder":
                    # Cancellation checkpoint: the cross-encoder rerank is the
                    # single most CPU-expensive stage and runs in a worker thread
                    # that cannot be interrupted once dispatched (issue #2122).
                    # Skip it entirely if the client already disconnected during
                    # retrieval, rather than burning ~2 CPUs producing a result
                    # nobody will read.
                    if request_context is not None:
                        request_context.raise_if_cancelled()

                    # Ensure reranker is initialized (for lazy initialization mode)
                    await reranker_instance.ensure_initialized()
                    scored_results = await reranker_instance.rerank(query, merged_candidates)
                else:
                    # "rrf" / "interleave": skip the cross-encoder and keep the fusion order
                    # (rrf_score is descending by fusion position for both). The cross-encoder
                    # was observed to demote a near-identical existing observation (the dedup
                    # "twin") far below the budget cutoff (semantic rank #1 -> reranked #37),
                    # causing the LLM to never see it and create a duplicate.
                    rerank_kind = f"{reranking}-passthrough"
                    scored_results = [
                        ScoredResult(
                            candidate=mc,
                            cross_encoder_score=0.0,
                            cross_encoder_score_normalized=0.0,
                            weight=0.0,
                        )
                        for mc in sorted(merged_candidates, key=lambda mc: mc.rrf_score, reverse=True)
                    ]

                step_duration = time.time() - step_start
                pre_filter_note = f" (pre-filtered {pre_filtered_count})" if pre_filtered_count > 0 else ""
                log_buffer.append(
                    f"  [4] Reranking [{rerank_kind}]: {len(scored_results)} candidates "
                    f"scored in {step_duration:.3f}s{pre_filter_note}"
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
            if scored_results and reranking == "interleave":
                # Interleave order is authoritative for dedup recall: do NOT re-sort by the
                # recency/temporal boosts — that re-sort is precisely what buried the twin
                # under RRF. Seed weight from the interleave-position rrf_score so the order
                # survives Step 5 truncation and the Step 6 token-budget cut.
                for sr in scored_results:
                    sr.weight = sr.candidate.rrf_score
                log_buffer.append("  [4.6] Interleave order preserved (combined scoring skipped)")
            elif scored_results:
                ce = reranker_instance.cross_encoder
                # "rrf" mode is passthrough by construction; so is a configured "rrf" CE.
                is_passthrough = (reranking == "rrf") or (ce is not None and ce.provider_name == "rrf")
                apply_combined_scoring(
                    scored_results,
                    now=_recall_scoring_now(question_date),
                    is_passthrough_reranker=is_passthrough,
                )
                # Per-strategy additive boost: nudge candidates surfaced by a
                # prioritised retrieval arm up the final ordering.
                strategy_boosts = get_config().recall_strategy_boosts
                if strategy_boosts:
                    from .search.recall_boost import additive_strategy_boost

                    for sr in scored_results:
                        sr.weight += additive_strategy_boost(sr.candidate.source_ranks, strategy_boosts)
                scored_results.sort(key=lambda x: x.weight, reverse=True)
                log_buffer.append("  [4.6] Combined scoring: ce * recency_boost(0.2) * temporal_boost(0.2)")
                if strategy_boosts:
                    log_buffer.append(f"  [4.7] Strategy boosts applied: {strategy_boosts}")

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
                    {"reranker_type": rerank_kind, "candidates_reranked": len(scored_results)},
                )

            # Cancellation checkpoint: reranking is done; skip the remaining
            # enrichment (chunk/entity/source-fact fetches, each its own DB work)
            # if the client disconnected while we were reranking (issue #2122).
            if request_context is not None:
                request_context.raise_if_cancelled()

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

        except OperationCancelledError:
            # Client disconnected mid-recall — propagate the cancellation so the
            # HTTP layer can return 499. Must precede the broad handler below,
            # which would otherwise bury it inside a RuntimeError (issue #2122).
            raise
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

            # observation_scopes is captured into retain_params at retain time
            # (see _build_retain_params); surface it as a top-level field so the
            # UI can show which scoping was requested. Only present for documents
            # retained after this was added.
            observation_scopes = retain_params_parsed.get("observation_scopes") if retain_params_parsed else None

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
                "observation_scopes": observation_scopes or None,
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

                # Capture relink victims BEFORE the cascade — once the source
                # rows are gone, the join finding them returns nothing.
                if unit_ids:
                    from .graph_maintenance import enqueue_relink_victims

                    await enqueue_relink_victims(conn, bank_id, unit_ids, ops=backend.ops)

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
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            if config.enable_auto_consolidation:
                try:
                    await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
                except Exception as e:
                    logger.warning(f"Failed to submit consolidation after document deletion for bank {bank_id}: {e}")

        # Run graph_maintenance whenever any unit was removed — even if no
        # relink victims were enqueued, the deleted unit's entities may now
        # be orphans that the bank-wide sweep should clean up.
        if unit_ids:
            try:
                await self.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit graph maintenance after document deletion for bank {bank_id}: {e}")

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
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            if config.enable_auto_consolidation:
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
        bank_id_for_graph_maintenance: str | None = None
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                # Get bank_id and fact_type before deletion
                row = await conn.fetchrow(
                    f"SELECT bank_id, fact_type FROM {fq_table('memory_units')} WHERE id = $1",
                    str(unit_uuid),
                )
                bank_id = row["bank_id"] if row else None
                fact_type = row["fact_type"] if row else None

                # Capture relink victims BEFORE the cascade — once the row is
                # gone, the join finding them returns nothing.
                if bank_id and fact_type in ("experience", "world"):
                    from .graph_maintenance import enqueue_relink_victims

                    await enqueue_relink_victims(conn, bank_id, [unit_id], ops=backend.ops)

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

                # Run graph_maintenance whenever a source-memory unit was
                # removed — even if no relink victims were enqueued, the
                # deleted unit's entities may now be orphans.
                if deleted and bank_id and fact_type in ("experience", "world"):
                    bank_id_for_graph_maintenance = bank_id

                result = {
                    "success": deleted is not None,
                    "unit_id": str(deleted) if deleted else None,
                    "message": "Memory unit and all its links deleted successfully"
                    if deleted
                    else "Memory unit not found",
                }

        if bank_id_for_consolidation:
            config = await self._config_resolver.resolve_full_config(bank_id_for_consolidation, request_context)
            if config.enable_auto_consolidation:
                try:
                    await self.submit_async_consolidation(
                        bank_id=bank_id_for_consolidation, request_context=request_context
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to submit consolidation after memory deletion"
                        f" for bank {bank_id_for_consolidation}: {e}"
                    )

        if bank_id_for_graph_maintenance:
            try:
                await self.submit_async_graph_maintenance(
                    bank_id=bank_id_for_graph_maintenance, request_context=request_context
                )
            except Exception as e:
                logger.warning(
                    f"Failed to submit graph maintenance after memory deletion "
                    f"for bank {bank_id_for_graph_maintenance}: {e}"
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
                        # Curation archive holds invalidated facts of the same types.
                        await conn.execute(
                            f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1 AND fact_type = $2",
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

                        # Curation archive (rows with NULL document_id aren't covered by
                        # the documents cascade, so clear by bank explicitly).
                        await conn.execute(
                            f"DELETE FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1", bank_id
                        )

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

        # Drop any cached stats for this bank — counts have changed and the
        # TTL would otherwise serve pre-delete values for up to a minute.
        await self._bank_stats_cache.invalidate(get_current_schema(), bank_id)

        if invalidated_obs > 0:
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            if config.enable_auto_consolidation:
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

    async def list_observation_scopes(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """List the distinct scopes across a bank's observations.

        Every consolidated observation lives under a "scope": the exact set of
        tags it was consolidated with. This enumerates each distinct scope (tag
        order normalized so ``[a, b]`` and ``[b, a]`` collapse) together with the
        number of observations in it. The empty list ``[]`` is the "global" scope
        of untagged observations. Results are ordered most-populous first.

        Returns:
            Dict with ``scopes``: list of ``{"tags": list[str], "count": int}``.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="list_observation_scopes", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"""
                SELECT scope, COUNT(*) AS count
                FROM (
                    SELECT COALESCE(ARRAY(SELECT unnest(tags) ORDER BY 1), '{{}}'::text[]) AS scope
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1 AND fact_type = 'observation'
                ) s
                GROUP BY scope
                ORDER BY count DESC, scope
                """,
                bank_id,
            )
            return {"scopes": [{"tags": list(r["scope"]), "count": r["count"]} for r in rows]}

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
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            if config.enable_auto_consolidation:
                await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)

        return {"deleted_count": deleted_count}

    async def _reembed_memory_text(
        self,
        *,
        text: str,
        occurred_start: datetime | None,
        occurred_end: datetime | None,
        mentioned_at: datetime | None,
        entities: list[str],
    ) -> str | None:
        """Recompute a memory unit's embedding string the same way retain does.

        Mirrors the retain pipeline's date+entity augmentation so an edited or
        reverted memory embeds identically to a freshly-retained one. Returns the
        pgvector string form (or None if the embedder produced nothing).
        """
        from .retain import embedding_processing
        from .retain.types import ExtractedFact

        shim = ExtractedFact(
            fact_text=text,
            fact_type="world",
            entities=list(entities or []),
            occurred_start=occurred_start,
            occurred_end=occurred_end,
            mentioned_at=mentioned_at,
        )
        augmented = embedding_processing.augment_texts_with_dates([shim], self._format_readable_date)
        embeddings = await embedding_processing.generate_embeddings_batch(self.embeddings, augmented)
        return str(embeddings[0]) if embeddings else None

    async def _memory_unit_columns(self, conn) -> str:
        """Comma-joined, quoted ordinal column list of ``memory_units``.

        Used to move a row verbatim between ``memory_units`` and the curation
        archive (``invalidated_memory_units``) without hardcoding the
        migration-evolving column set — the archive is created via
        ``LIKE memory_units`` so the lists line up.
        """
        rows = await conn.fetch(
            f"SELECT a.attname FROM pg_attribute a "
            f"WHERE a.attrelid = '{fq_table('memory_units')}'::regclass "
            f"AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum"
        )
        return ", ".join(f'"{r["attname"]}"' for r in rows)

    async def update_memory_unit(
        self,
        bank_id: str,
        memory_id: str,
        *,
        text: str | None = None,
        context: str | None = None,
        occurred_start: str | None = None,
        occurred_end: str | None = None,
        new_fact_type: str | None = None,
        entities: list[str] | None = None,
        state: str | None = None,
        reason: str | None = None,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Curate a single raw memory unit: edit its fields and/or change its state.

        Invalidation keeps the recall hot-path clean by *moving* the row between
        tables rather than flagging it: live facts live in ``memory_units``,
        invalidated ones in ``invalidated_memory_units``. Recall/consolidation/
        graph queries therefore need no state predicate.

        - **Edit** (``text``/``context``/``occurred_start``/``occurred_end``/
          ``new_fact_type``/``entities``): correct what the LLM extracted.
          Re-embeds (text + dates + entities feed the embedding), drops derived
          observations + links, and re-consolidates. For date/context fields,
          ``""`` clears to NULL and ``None`` leaves unchanged; ``new_fact_type``
          must be world/experience. ``entities`` (when not None) replaces the
          unit's entity set: names are resolved/find-or-created via the same
          resolver retain uses, ``unit_entities`` + cooccurrence are rebuilt, and
          ``[]`` detaches all entities. Entities orphaned by the swap, and any
          now-stale cooccurrence rows, are reclaimed by the graph-maintenance
          sweep that this edit submits (entity edges live in ``unit_entities``,
          not ``memory_links``, so there is nothing to relink directly).
        - **Invalidate** (``state='invalidated'``): move the row to the archive
          (cascade-pruning its links/entity associations and re-deriving dependent
          observations). The embedding + an entity-id snapshot travel with it.
        - **Revert** (``state='valid'``): move the row back, restore its entity
          associations, and re-consolidate.

        Only ``world``/``experience`` facts can be curated — observations are
        derived and regenerate from their sources. Returns the updated memory
        (same shape as :meth:`get_memory_unit`) or None if not found.
        """
        try:
            memory_uuid = uuid.UUID(memory_id)
        except ValueError:
            raise ValueError(f"Invalid memory_id: '{memory_id}' is not a valid UUID")
        if state is not None and state not in ("valid", "invalidated"):
            raise ValueError(f"Invalid state '{state}': expected 'valid' or 'invalidated'.")
        if text is not None and not text.strip():
            raise ValueError("text must not be empty.")
        if new_fact_type is not None and new_fact_type not in ("world", "experience"):
            raise ValueError(f"Invalid fact_type '{new_fact_type}': expected 'world' or 'experience'.")
        # Normalize the entity list up front: drop blanks/whitespace and de-dup
        # case-insensitively (the resolver would coalesce these anyway). A
        # provided-but-empty list means "detach all entities"; None means leave
        # the unit's entities untouched.
        new_entities: list[str] | None = None
        if entities is not None:
            seen_names: set[str] = set()
            new_entities = []
            for name in entities:
                cleaned = name.strip()
                if cleaned and cleaned.lower() not in seen_names:
                    seen_names.add(cleaned.lower())
                    new_entities.append(cleaned)

        def _parse_edit_date(value: str | None) -> datetime | None:
            # "" clears to NULL; an ISO date/datetime parses (UTC if naive).
            if not value:
                return None
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="update_memory_unit", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        backend = await self._get_backend()
        from .graph_maintenance import enqueue_relink_victims
        from .retain.link_utils import resolve_entities_only

        # Resolve the bank's entity-label taxonomy once when re-resolving entities,
        # so corrected entities are matched with the same rules retain uses.
        entity_labels = None
        if new_entities is not None:
            edit_config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            entity_labels = getattr(edit_config, "entity_labels", None)

        mu = fq_table("memory_units")
        arch = fq_table("invalidated_memory_units")
        ue = fq_table("unit_entities")
        ml = fq_table("memory_links")
        ent = fq_table("entities")

        need_consolidation = False
        need_graph = False
        found = False

        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                live = await conn.fetchrow(
                    f"SELECT text, context, fact_type, event_date, occurred_start, occurred_end, mentioned_at "
                    f"FROM {mu} WHERE id = $1 AND bank_id = $2",
                    str(memory_uuid),
                    bank_id,
                )
                archived = None
                if not live:
                    archived = await conn.fetchrow(
                        f"SELECT fact_type FROM {arch} WHERE id = $1 AND bank_id = $2",
                        str(memory_uuid),
                        bank_id,
                    )
                record = live or archived
                if record is None:
                    return None
                found = True
                current_fact_type = record["fact_type"]
                if current_fact_type not in ("experience", "world"):
                    raise ValueError(
                        f"Memory '{memory_id}' is a {current_fact_type}; only world/experience facts can be "
                        "curated. Observations are derived and regenerate from their sources."
                    )

                collist = await self._memory_unit_columns(conn)

                # --- Edit fields (live rows only): text / context / dates / fact_type / entities ---
                doing_edit = any(
                    v is not None for v in (text, context, occurred_start, occurred_end, new_fact_type)
                ) or (new_entities is not None)
                if doing_edit:
                    if not live:
                        raise ValueError("Cannot edit an invalidated memory; revert it to 'valid' first.")
                    new_text = text if text is not None else live["text"]
                    new_context = (context or None) if context is not None else live["context"]
                    new_fact = new_fact_type if new_fact_type is not None else live["fact_type"]
                    new_occ_start = (
                        _parse_edit_date(occurred_start) if occurred_start is not None else live["occurred_start"]
                    )
                    new_occ_end = _parse_edit_date(occurred_end) if occurred_end is not None else live["occurred_end"]
                    # event_date (NOT NULL, legacy single date + used by temporal links)
                    # tracks the occurred start when it's set.
                    new_event_date = new_occ_start or live["event_date"]

                    # Rebuild the unit's entity set FIRST, so the re-embed below picks
                    # up the corrected canonical names. Reuses retain's resolver
                    # (find-or-create + cooccurrence) rather than touching entities
                    # directly. Orphaned entities + stale cooccurrence are swept by
                    # the graph-maintenance run this edit submits.
                    if new_entities is not None:
                        entity_date = new_occ_start or live["mentioned_at"]
                        _resolved_ids, _e2u, unit_to_entity_ids = await resolve_entities_only(
                            self.entity_resolver,
                            conn,
                            bank_id,
                            [str(memory_uuid)],
                            [new_text],
                            new_context or "",
                            [entity_date],
                            [[{"text": name, "type": "CONCEPT"} for name in new_entities]],
                            entity_labels=entity_labels,
                        )
                        await conn.execute(f"DELETE FROM {ue} WHERE unit_id = $1", str(memory_uuid))
                        resolved_for_unit = unit_to_entity_ids.get(str(memory_uuid), [])
                        if resolved_for_unit:
                            await self.entity_resolver.link_units_to_entities_batch(
                                [(str(memory_uuid), eid, entity_date) for eid in resolved_for_unit],
                                conn=conn,
                            )

                    ent_rows = await conn.fetch(
                        f"SELECT e.canonical_name FROM {ue} ue JOIN {ent} e ON ue.entity_id = e.id "
                        f"WHERE ue.unit_id = $1",
                        str(memory_uuid),
                    )
                    new_emb = await self._reembed_memory_text(
                        text=new_text,
                        occurred_start=new_occ_start,
                        occurred_end=new_occ_end,
                        mentioned_at=live["mentioned_at"],
                        entities=[r["canonical_name"] for r in ent_rows],
                    )
                    await enqueue_relink_victims(conn, bank_id, [memory_id], ops=backend.ops)
                    await conn.execute(
                        f"""
                        UPDATE {mu}
                        SET text = $3, context = $4, fact_type = $5, occurred_start = $6,
                            occurred_end = $7, event_date = $8, embedding = $9::vector,
                            consolidated_at = NULL, consolidation_failed_at = NULL,
                            edited_at = now(), updated_at = now()
                        WHERE id = $1 AND bank_id = $2
                        """,
                        str(memory_uuid),
                        bank_id,
                        new_text,
                        new_context,
                        new_fact,
                        new_occ_start,
                        new_occ_end,
                        new_event_date,
                        new_emb,
                    )
                    await conn.execute(f"DELETE FROM {ml} WHERE from_unit_id = $1 OR to_unit_id = $1", str(memory_uuid))
                    await self._delete_stale_observations_for_memories(conn, bank_id, [memory_id])
                    need_consolidation = True
                    need_graph = True

                # --- Invalidate: move live → archive ---
                if state == "invalidated" and live:
                    entity_ids = [
                        r["entity_id"]
                        for r in await conn.fetch(f"SELECT entity_id FROM {ue} WHERE unit_id = $1", str(memory_uuid))
                    ]
                    # Capture relink victims BEFORE the row (and its links) disappear.
                    await enqueue_relink_victims(conn, bank_id, [memory_id], ops=backend.ops)
                    await conn.execute(
                        f"INSERT INTO {arch} ({collist}, invalidation_reason, invalidated_at, entity_ids) "
                        f"SELECT {collist}, $2, now(), $3::uuid[] FROM {mu} WHERE id = $1 AND bank_id = $4",
                        str(memory_uuid),
                        reason,
                        entity_ids,
                        bank_id,
                    )
                    # Cascade prunes unit_entities + memory_links; sweep runs after
                    # the delete so it also catches a racing observation insert.
                    await conn.execute(f"DELETE FROM {mu} WHERE id = $1 AND bank_id = $2", str(memory_uuid), bank_id)
                    await self._delete_stale_observations_for_memories(conn, bank_id, [memory_id])
                    need_consolidation = True
                    need_graph = True
                elif state == "invalidated" and archived and reason is not None:
                    # Already archived — just update the recorded reason.
                    await conn.execute(
                        f"UPDATE {arch} SET invalidation_reason = $3 WHERE id = $1 AND bank_id = $2",
                        str(memory_uuid),
                        bank_id,
                        reason,
                    )

                # --- Revert: move archive → live ---
                elif state == "valid" and archived:
                    arch_row = await conn.fetchrow(
                        f"SELECT entity_ids FROM {arch} WHERE id = $1 AND bank_id = $2", str(memory_uuid), bank_id
                    )
                    await conn.execute(
                        f"INSERT INTO {mu} ({collist}) SELECT {collist} FROM {arch} WHERE id = $1 AND bank_id = $2",
                        str(memory_uuid),
                        bank_id,
                    )
                    # Re-consolidate from scratch; links are rebuilt by graph maintenance.
                    await conn.execute(
                        f"UPDATE {mu} SET consolidated_at = NULL, consolidation_failed_at = NULL, updated_at = now() "
                        f"WHERE id = $1 AND bank_id = $2",
                        str(memory_uuid),
                        bank_id,
                    )
                    # Restore entity associations for entities that still exist (some may
                    # have been pruned as orphans after the original move).
                    if arch_row and arch_row["entity_ids"]:
                        await conn.execute(
                            f"INSERT INTO {ue} (unit_id, entity_id) "
                            f"SELECT $1, eid FROM unnest($2::uuid[]) AS eid "
                            f"WHERE EXISTS (SELECT 1 FROM {ent} e WHERE e.id = eid AND e.bank_id = $3) "
                            f"ON CONFLICT DO NOTHING",
                            str(memory_uuid),
                            arch_row["entity_ids"],
                            bank_id,
                        )
                    await conn.execute(f"DELETE FROM {arch} WHERE id = $1 AND bank_id = $2", str(memory_uuid), bank_id)
                    need_consolidation = True
                    need_graph = True

        if not found:
            return None

        if need_consolidation:
            config = await self._config_resolver.resolve_full_config(bank_id, request_context)
            if config.enable_auto_consolidation:
                try:
                    await self.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
                except Exception as e:
                    logger.warning(f"Failed to submit consolidation after curating memory in bank {bank_id}: {e}")
        if need_graph:
            try:
                await self.submit_async_graph_maintenance(bank_id=bank_id, request_context=request_context)
            except Exception as e:
                logger.warning(f"Failed to submit graph maintenance after curating memory in bank {bank_id}: {e}")

        return await self.get_memory_unit(bank_id=bank_id, memory_id=memory_id, request_context=request_context)

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
            elif tags_match == "exact":
                # Exact match with no tags is the "global" scope: rows that carry no
                # tags at all. (Other match modes treat empty tags as "no filter".)
                query_conditions.append("(tags IS NULL OR tags = '{}')")

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

            # Fetch non-entity links where BOTH endpoints are in the visible set (or
            # source memories). Entity edges are derived below from unit_entities so
            # we don't materialize them in memory_links anymore.
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
                           NULL::text AS entity_name
                    FROM {fq_table("memory_links")} ml
                    WHERE ml.link_type <> 'entity'
                      AND ml.from_unit_id = ANY($1::uuid[])
                      AND ml.to_unit_id = ANY($1::uuid[])
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

        # Build derived links: entity edges for all visible units (from unit_entities)
        # and observation semantic edges via shared source memories.
        # Observations never have direct memory_links rows, so all their links are derived.
        observation_units = [unit for unit in units if unit["fact_type"] == "observation"]

        # Entity links: pair any visible units that share at least one entity.
        # Each unit links to up to max_neighbors_per_unit subsequent units in the
        # per-entity list, so every unit that shares an entity with another visible
        # unit gets edges (matches the historical writer cap, which was per-unit).
        # Bounds total edges to ~N * cap per entity instead of N² for hot entities.
        max_neighbors_per_unit = 10
        entity_to_units_visible: dict[str, list] = {}
        for unit_id in unit_ids:
            for entity_name in entity_map.get(unit_id, []):
                entity_to_units_visible.setdefault(entity_name, []).append(unit_id)

        # Semantic links: pair observations that share at least one source memory
        source_to_obs_for_semantic: dict = {}
        for unit in observation_units:
            if unit["source_memory_ids"]:
                for src_id in unit["source_memory_ids"]:
                    source_to_obs_for_semantic.setdefault(src_id, []).append(unit["id"])

        observation_inferred_links = []
        seen_inferred: set[tuple] = set()

        for entity_name, ent_unit_ids in entity_to_units_visible.items():
            n = len(ent_unit_ids)
            for i, unit_a in enumerate(ent_unit_ids):
                # Sliding window: link unit_a to its next max_neighbors_per_unit
                # in the list. Each pair is also "incoming" for the later unit,
                # so every unit ends up with up to ~2*max_neighbors_per_unit edges
                # for this entity (its successors + its predecessors via their pairs).
                for j in range(i + 1, min(i + 1 + max_neighbors_per_unit, n)):
                    unit_b = ent_unit_ids[j]
                    pair = (min(str(unit_a), str(unit_b)), max(str(unit_a), str(unit_b)), "entity", entity_name)
                    if pair not in seen_inferred:
                        seen_inferred.add(pair)
                        observation_inferred_links.append(
                            {
                                "from_unit_id": unit_a,
                                "to_unit_id": unit_b,
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
        state: str | None = None,
        document_id: str | None = None,
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
            document_id: Optional filter to a single source document.
            state: Optional curation-state filter ('valid' or 'invalidated').
                Invalidated facts live in a separate archive table; 'invalidated'
                reads that archive. Omitted/('valid') lists live facts.
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
        if state is not None and state not in ("valid", "invalidated"):
            raise ValueError(f"Invalid state '{state}': expected 'valid' or 'invalidated'.")
        # Invalidated facts live in a separate archive table; pick the source
        # accordingly. Default (state is None) lists live facts.
        is_archived = state == "invalidated"
        source_table = fq_table("invalidated_memory_units") if is_archived else fq_table("memory_units")
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

            if document_id:
                param_count += 1
                query_conditions.append(f"document_id = ${param_count}")
                query_params.append(document_id)

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
                FROM {source_table}
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

            # The archive carries invalidation bookkeeping; the live table doesn't.
            curation_cols = (
                "invalidation_reason, invalidated_at"
                if is_archived
                else "NULL::text AS invalidation_reason, NULL::timestamptz AS invalidated_at"
            )
            units = await conn.fetch(
                f"""
                SELECT id, text, event_date, context, fact_type, document_id,
                       mentioned_at, occurred_start, occurred_end, chunk_id, proof_count,
                       tags, consolidated_at, consolidation_failed_at, edited_at, {curation_cols}
                FROM {source_table}
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
                        "document_id": row["document_id"],
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
                        "state": "invalidated" if is_archived else "valid",
                        "invalidation_reason": row["invalidation_reason"],
                        "invalidated_at": row["invalidated_at"].isoformat() if row["invalidated_at"] else None,
                        "edited_at": row["edited_at"].isoformat() if row["edited_at"] else None,
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
            # Get the memory unit (include source_memory_ids for mental models).
            # Curation moves invalidated facts to invalidated_memory_units, so fall
            # back to the archive (with its invalidation bookkeeping) on a miss.
            select_cols = (
                "id, text, context, event_date, occurred_start, occurred_end, "
                "mentioned_at, fact_type, document_id, chunk_id, tags, source_memory_ids, "
                "observation_scopes, edited_at"
            )
            row = await conn.fetchrow(
                f"SELECT {select_cols}, NULL::text AS invalidation_reason, NULL::timestamptz AS invalidated_at "
                f"FROM {fq_table('memory_units')} WHERE id = $1 AND bank_id = $2",
                str(memory_uuid),
                bank_id,
            )
            unit_state = "valid"
            if not row:
                row = await conn.fetchrow(
                    f"SELECT {select_cols}, invalidation_reason, invalidated_at "
                    f"FROM {fq_table('invalidated_memory_units')} WHERE id = $1 AND bank_id = $2",
                    str(memory_uuid),
                    bank_id,
                )
                unit_state = "invalidated"

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
                "state": unit_state,
                "invalidation_reason": row["invalidation_reason"],
                "invalidated_at": row["invalidated_at"].isoformat() if row["invalidated_at"] else None,
                "edited_at": row["edited_at"].isoformat() if row["edited_at"] else None,
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
                SELECT fact_type, source_memory_ids
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

            # History now lives in the dedicated observation_history table
            # (one row per change), ordered oldest-first to match the prior
            # append-order semantics the reconstruction below relies on.
            history_rows = await conn.fetch(
                f"""
                SELECT content, changed_at
                FROM {fq_table("observation_history")}
                WHERE observation_id = $1
                ORDER BY changed_at ASC, id ASC
                """,
                uuid.UUID(memory_id),
            )
            if not history_rows:
                return []

            def _iso(v: Any) -> Any:
                return v.isoformat() if hasattr(v, "isoformat") else v

            def _as_list(v: Any) -> list:
                return list(v) if v else []

            raw_history = []
            for hr in history_rows:
                # The snapshot fields live in the JSONB ``content`` blob (str on
                # Oracle CLOB / when no jsonb codec is registered, dict otherwise).
                content = hr["content"]
                if isinstance(content, str):
                    content = json.loads(content) if content else {}
                content = content or {}
                raw_history.append(
                    {
                        "previous_text": content.get("previous_text"),
                        "previous_tags": _as_list(content.get("previous_tags")),
                        "previous_occurred_start": content.get("previous_occurred_start"),
                        "previous_occurred_end": content.get("previous_occurred_end"),
                        "previous_mentioned_at": content.get("previous_mentioned_at"),
                        "changed_at": _iso(hr["changed_at"]),
                        "new_source_memory_ids": [str(s) for s in _as_list(content.get("new_source_memory_ids"))],
                    }
                )

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

    # ==================== LLM request tracing read methods ====================

    # Column list shared by the flat and grouped llm_requests queries.
    _LLM_REQUEST_COLUMNS = (
        "id, bank_id, operation, scope, trace_id, span_id, parent_span_id, "
        "provider, model, status, started_at, ended_at, duration_ms, "
        "input_tokens, output_tokens, cached_tokens, total_tokens, "
        "input, output, error, llm_info, metadata"
    )

    @staticmethod
    def _llm_request_entry(conn: Any, row: Any) -> LLMRequestEntry:
        """Map a llm_requests row (selected via _LLM_REQUEST_COLUMNS) to the model."""
        return LLMRequestEntry(
            id=str(row["id"]),
            bank_id=row["bank_id"],
            operation=row["operation"],
            scope=row["scope"],
            trace_id=row["trace_id"],
            span_id=row["span_id"],
            parent_span_id=row["parent_span_id"],
            provider=row["provider"],
            model=row["model"],
            status=row["status"],
            started_at=row["started_at"].isoformat() if row["started_at"] else None,
            ended_at=row["ended_at"].isoformat() if row["ended_at"] else None,
            duration_ms=row["duration_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_tokens=row["cached_tokens"],
            total_tokens=row["total_tokens"],
            input=conn.parse_json(row["input"]) if row["input"] is not None else None,
            output=conn.parse_json(row["output"]) if row["output"] is not None else None,
            error=row["error"],
            llm_info=conn.parse_json(row["llm_info"]) if row["llm_info"] is not None else {},
            metadata=conn.parse_json(row["metadata"]) if row["metadata"] is not None else {},
        )

    async def list_llm_requests(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        status: str | None = None,
        operation: str | None = None,
        scope: str | None = None,
        provider: str | None = None,
        trace_id: str | None = None,
        document_id: str | None = None,
        memory_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        group: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> LLMRequestListResponse | None:
        """List traced LLM requests for a bank, newest first.

        When ``group`` is True, pagination is by operation run (all LLM calls
        sharing a ``trace_id``) rather than by individual call: ``limit``/
        ``offset`` and ``total`` count runs, and every returned run is complete
        (never split across pages), so the UI can render parent → child without
        gaps. When False, results are flat, paginated per call.

        Returns None when the bank does not exist (the HTTP layer maps this to a
        404). Authentication and tenant-schema resolution happen inside
        ``get_bank_profile`` before any query runs, so the queries below are
        scoped to the authenticated tenant's schema.
        """
        if await self.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False) is None:
            return None

        where_clauses = ["bank_id = $1"]
        params: list[Any] = [bank_id]
        idx = 2
        for column, value in (
            ("status", status),
            ("operation", operation),
            ("scope", scope),
            ("provider", provider),
            ("trace_id", trace_id),
        ):
            if value:
                where_clauses.append(f"{column} = ${idx}")
                params.append(value)
                idx += 1
        if document_id is not None:
            # document_id is carried in per-call metadata (set by the retain
            # extraction path); a document accrues one trace per retain run.
            where_clauses.append(f"metadata->>'document_id' = ${idx}")
            params.append(document_id)
            idx += 1
        if memory_id is not None:
            # Match the run(s) that produced this memory (metadata.memory_ids) or
            # consumed it as a consolidation source (metadata.source_memory_ids),
            # so a memory resolves both the trace that created it and the traces
            # that used it. The `?` operator tests array membership on the jsonb;
            # both clauses reference the same bind param.
            where_clauses.append(f"(metadata->'memory_ids' ? ${idx} OR metadata->'source_memory_ids' ? ${idx})")
            params.append(memory_id)
            idx += 1
        if start_date is not None:
            where_clauses.append(f"started_at >= ${idx}")
            params.append(start_date)
            idx += 1
        if end_date is not None:
            where_clauses.append(f"started_at < ${idx}")
            params.append(end_date)
            idx += 1

        where_sql = " AND ".join(where_clauses)
        table = fq_table("llm_requests")
        cols = self._LLM_REQUEST_COLUMNS

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            if group:
                # A "run" = one trace_id; untraced rows are each their own run.
                run_key = "COALESCE(trace_id, 'single:' || id::text)"
                count_row = await conn.fetchrow(
                    f"SELECT COUNT(*) AS total FROM (SELECT 1 FROM {table} WHERE {where_sql} GROUP BY {run_key}) q",
                    *params,
                )
                total = count_row["total"] if count_row else 0
                # Page of runs, most-recently-active first.
                key_rows = await conn.fetch(
                    f"""
                    SELECT {run_key} AS run_key, MAX(started_at) AS run_end
                    FROM {table} WHERE {where_sql}
                    GROUP BY {run_key}
                    ORDER BY run_end DESC
                    LIMIT ${idx} OFFSET ${idx + 1}
                    """,
                    *params,
                    limit,
                    offset,
                )
                run_keys = [r["run_key"] for r in key_rows]
                rows = []
                if run_keys:
                    rows = await conn.fetch(
                        f"""
                        SELECT {cols} FROM {table}
                        WHERE {where_sql} AND {run_key} = ANY(${idx}::text[])
                        ORDER BY started_at DESC
                        """,
                        *params,
                        run_keys,
                    )
            else:
                count_row = await conn.fetchrow(f"SELECT COUNT(*) AS total FROM {table} WHERE {where_sql}", *params)
                total = count_row["total"] if count_row else 0
                rows = await conn.fetch(
                    f"""
                    SELECT {cols} FROM {table}
                    WHERE {where_sql}
                    ORDER BY started_at DESC
                    LIMIT ${idx} OFFSET ${idx + 1}
                    """,
                    *params,
                    limit,
                    offset,
                )

            items = [self._llm_request_entry(conn, row) for row in rows]

        return LLMRequestListResponse(bank_id=bank_id, total=total, limit=limit, offset=offset, items=items)

    async def llm_request_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        operation: str | None = None,
        period: str = "7d",
    ) -> LLMRequestStatsResponse | None:
        """LLM request counts and token sums grouped by day, for charting.

        Returns None when the bank does not exist (mapped to 404 by the HTTP
        layer). Auth/tenant resolution happen in ``get_bank_profile``.
        """
        if await self.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False) is None:
            return None

        now = datetime.now(timezone.utc)
        trunc = "day"
        if period == "1d":
            start = now - timedelta(days=1)
        elif period == "30d":
            start = now - timedelta(days=30)
        else:  # 7d default
            start = now - timedelta(days=7)

        where_clauses = ["bank_id = $1", "started_at >= $2"]
        params: list[Any] = [bank_id, start]
        idx = 3
        if operation:
            where_clauses.append(f"operation = ${idx}")
            params.append(operation)
            idx += 1
        where_sql = " AND ".join(where_clauses)
        table = fq_table("llm_requests")

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"""
                SELECT date_trunc('{trunc}', started_at) AS bucket,
                       status,
                       COUNT(*) AS count,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                       COALESCE(SUM(total_tokens), 0) AS total_tokens
                FROM {table}
                WHERE {where_sql}
                GROUP BY bucket, status
                ORDER BY bucket ASC
                """,
                *params,
            )

        # Aggregate per bucket: call counts by status + summed token usage. Plain
        # dicts here (status names / bucket keys are dynamic); materialized into
        # typed models below.
        statuses_by_bucket: dict[str, dict[str, int]] = {}
        tokens_by_bucket: dict[str, dict[str, int]] = {}
        order: list[str] = []
        for row in rows:
            key = row["bucket"].isoformat()
            if key not in statuses_by_bucket:
                statuses_by_bucket[key] = {}
                tokens_by_bucket[key] = {"input": 0, "output": 0, "cached": 0, "total": 0}
                order.append(key)
            statuses_by_bucket[key][row["status"]] = row["count"]
            tok = tokens_by_bucket[key]
            tok["input"] += row["input_tokens"]
            tok["output"] += row["output_tokens"]
            tok["cached"] += row["cached_tokens"]
            tok["total"] += row["total_tokens"]

        return LLMRequestStatsResponse(
            bank_id=bank_id,
            period=period,
            trunc=trunc,
            start=start.isoformat(),
            buckets=[
                LLMRequestStatsBucket(
                    time=k,
                    statuses=statuses_by_bucket[k],
                    total=sum(statuses_by_bucket[k].values()),
                    tokens=LLMRequestTokenSums(**tokens_by_bucket[k]),
                )
                for k in order
            ],
        )

    # ==================== Audit log read methods ====================

    async def list_audit_logs(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        action: str | None = None,
        transport: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> "AuditLogListResponse | None":
        """List audit log entries for a bank, newest first.

        Returns None when the bank does not exist (the HTTP layer maps this to a
        404). Authentication and tenant-schema resolution happen inside
        ``get_bank_profile`` before any query runs, so the SELECT below is scoped
        to the authenticated tenant's schema.
        """
        from .audit import AuditLogEntry, AuditLogListResponse

        if await self.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False) is None:
            return None

        where_clauses = ["bank_id = $1"]
        params: list[Any] = [bank_id]
        idx = 2
        for column, value in (("action", action), ("transport", transport)):
            if value:
                where_clauses.append(f"{column} = ${idx}")
                params.append(value)
                idx += 1
        if start_date is not None:
            where_clauses.append(f"started_at >= ${idx}")
            params.append(start_date)
            idx += 1
        if end_date is not None:
            where_clauses.append(f"started_at < ${idx}")
            params.append(end_date)
            idx += 1

        where_sql = " AND ".join(where_clauses)
        table = fq_table("audit_log")

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            count_row = await conn.fetchrow(f"SELECT COUNT(*) AS total FROM {table} WHERE {where_sql}", *params)
            total = count_row["total"] if count_row else 0

            params.append(limit)
            params.append(offset)
            rows = await conn.fetch(
                f"""
                SELECT id, action, transport, bank_id, started_at, ended_at,
                       request, response, metadata
                FROM {table}
                WHERE {where_sql}
                ORDER BY started_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )

            items = []
            for row in rows:
                started = row["started_at"]
                ended = row["ended_at"]
                duration_ms = int((ended - started).total_seconds() * 1000) if started and ended else None
                items.append(
                    AuditLogEntry(
                        id=str(row["id"]),
                        action=row["action"],
                        transport=row["transport"],
                        bank_id=row["bank_id"],
                        started_at=started.isoformat() if started else None,
                        ended_at=ended.isoformat() if ended else None,
                        duration_ms=duration_ms,
                        request=conn.parse_json(row["request"]) if row["request"] is not None else None,
                        response=conn.parse_json(row["response"]) if row["response"] is not None else None,
                        metadata=conn.parse_json(row["metadata"]) if row["metadata"] is not None else {},
                    )
                )

        return AuditLogListResponse(bank_id=bank_id, total=total, limit=limit, offset=offset, items=items)

    async def audit_log_stats(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
        action: str | None = None,
        period: str = "7d",
    ) -> "AuditLogStatsResponse | None":
        """Audit log counts grouped by day and action, for charting.

        Returns None when the bank does not exist (mapped to 404 by the HTTP
        layer). Auth/tenant resolution happen in ``get_bank_profile``.
        """
        from .audit import AuditLogStatsBucket, AuditLogStatsResponse

        if await self.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False) is None:
            return None

        now = datetime.now(timezone.utc)
        trunc = "day"
        if period == "1d":
            start = now - timedelta(days=1)
        elif period == "30d":
            start = now - timedelta(days=30)
        else:  # 7d default
            start = now - timedelta(days=7)

        where_clauses = ["bank_id = $1", "started_at >= $2"]
        params: list[Any] = [bank_id, start]
        idx = 3
        if action:
            where_clauses.append(f"action = ${idx}")
            params.append(action)
            idx += 1
        where_sql = " AND ".join(where_clauses)
        table = fq_table("audit_log")

        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"""
                SELECT date_trunc('{trunc}', started_at) AS bucket,
                       action,
                       COUNT(*) AS count
                FROM {table}
                WHERE {where_sql}
                GROUP BY bucket, action
                ORDER BY bucket ASC
                """,
                *params,
            )

        # Aggregate per bucket: counts by action name (dynamic keys, so a plain
        # dict here; materialized into typed models below).
        actions_by_bucket: dict[str, dict[str, int]] = {}
        order: list[str] = []
        for row in rows:
            key = row["bucket"].isoformat()
            if key not in actions_by_bucket:
                actions_by_bucket[key] = {}
                order.append(key)
            actions_by_bucket[key][row["action"]] = row["count"]

        return AuditLogStatsResponse(
            bank_id=bank_id,
            period=period,
            trunc=trunc,
            start=start.isoformat(),
            buckets=[
                AuditLogStatsBucket(time=k, actions=actions_by_bucket[k], total=sum(actions_by_bucket[k].values()))
                for k in order
            ],
        )

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
            result = await bank_utils.get_or_create_bank_profile(backend, bank_id)
            profile, created = result.profile, result.created

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

    async def _ensure_bank_exists(
        self,
        bank_id: str,
        request_context: "RequestContext",
        *,
        conn=None,
    ) -> bool:
        """Lazily create the bank row (the FK target for bank-scoped writes).

        This is the single entry point every write path uses to mirror retain's
        lazy bank auto-create, so a first write to a new bank (pinned mental
        model, webhook, async operation, ...) behaves consistently instead of
        surfacing a raw FK violation.

        Transactionality:
          * Pass ``conn`` (a connection with an open transaction) to run the
            bank ``INSERT`` and its per-bank vector index creation on the
            caller's connection. The bank row then commits — or rolls back —
            atomically with the caller's write on that same transaction.
          * Omit ``conn`` to ensure the bank on a dedicated connection (used by
            paths that have no single write transaction to join, e.g. retain and
            import, whose data is written later across many per-document
            transactions).

        The ``HINDSIGHT_API_DEFAULT_BANK_TEMPLATE`` hook is best-effort, opens
        its own connections and can itself create pinned mental models, so it is
        never run inside the caller's transaction. When ``conn`` is omitted it is
        applied inline here. When ``conn`` is supplied the caller MUST apply it
        after committing, gated on the returned flag::

            async with acquire_with_retry(backend) as conn:
                async with conn.transaction():
                    created = await self._ensure_bank_exists(bank_id, rc, conn=conn)
                    ...  # bank-scoped write on the same conn
            if created:
                await self._apply_default_bank_template(bank_id, rc)

        Returns:
            True if the bank was freshly created on this call.
        """
        backend = await self._get_backend()
        if conn is not None:
            result = await bank_utils.get_or_create_bank_profile_on_conn(conn, bank_id, ops=backend.ops)
            return result.created

        result = await bank_utils.get_or_create_bank_profile(backend, bank_id)
        if result.created:
            await self._apply_default_bank_template(bank_id, request_context)
        return result.created

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
        await self._get_backend()
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
        await self._get_backend()
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
        await self._get_backend()
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
        await self._get_backend()
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
        _operation_label: str = "reflect",
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
        # Sanitize at ingress so lone UTF-16 surrogates in the question/context cannot
        # crash logging, recall's embedder, or the reflect LLM call (see issue #1875).
        query = sanitize_text(query) or ""
        context = sanitize_text(context)

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

        # Cooperative cancellation checkpoint: if the client already disconnected
        # while this request waited to be scheduled, abort before doing any work
        # (issue #2122). The agentic loop re-checks between iterations, and the
        # nested recall tool checks at its own stage boundaries.
        request_context.raise_if_cancelled()

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

        # Pull only the consolidation freshness — get_bank_stats also computes
        # link aggregations that reflect() does not use and which can take many
        # seconds on large banks.
        freshness = await self.get_bank_freshness(bank_id, request_context=request_context)
        last_consolidated_at = freshness.get("last_consolidated_at")
        pending_consolidation = freshness.get("pending_consolidation", 0)

        # Create tool callbacks that acquire connections only when needed
        from .retain import embedding_utils

        async def search_mental_models_fn(q: str, max_results: int = 5) -> dict[str, Any]:
            # Generate embedding for the query
            embeddings = await embedding_utils.generate_embeddings_batch(
                self.embeddings,
                [q],
                input_type="query",
            )
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
        # With document text storage disabled there is no raw chunk text, so
        # fetching chunks would only attach empty strings to every recall
        # result. Force it off (pairs with excluding the expand tool below).
        if not get_config().store_document_text:
            effective_recall_include_chunks = False
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
            tag_groups=tag_groups,
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
                        llm_config=self._reflect_llm_config.with_config(
                            resolved_reflect_config, bank_id=bank_id, operation=_operation_label
                        ),
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
                        llm_output_language=getattr(resolved_reflect_config, "llm_output_language", None),
                        cancel_check=request_context.raise_if_cancelled,
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
        """Get statistics about memory nodes and links for a bank.

        Results are served from a short-TTL per-process cache so a polling
        client cannot drive the link/unit aggregations multiple times per
        second; concurrent misses on the same bank are coalesced onto a
        single in-flight loader.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_bank_stats", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        schema = get_current_schema()
        return await self._bank_stats_cache.get_or_load(
            schema,
            bank_id,
            lambda: self._compute_bank_stats(bank_id),
        )

    async def _compute_bank_stats(self, bank_id: str) -> dict[str, Any]:
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

            # Non-entity link counts — no join, group by link_type only. With a
            # (bank_id, link_type) index this is an index-only scan; without one
            # it is at worst a single seq scan over memory_links rather than the
            # multi-second hash join through memory_units that this used to be.
            # The previous shape produced a (fact_type, link_type) matrix; only
            # the hindsight-cli `bank stats` renderer still consumes the
            # per-fact-type slice, and it tolerates empty maps (the section
            # prints with no rows). Response keys are kept populated below for
            # schema stability so existing SDK deserializers don't break.
            non_entity_link_rows = await conn.fetch(
                f"""
                SELECT link_type, COUNT(*) as count
                FROM {fq_table("memory_links")}
                WHERE bank_id = $1 AND link_type <> 'entity'
                GROUP BY link_type
                """,
                bank_id,
            )

            # Entity links are derived from unit_entities (no longer stored in
            # memory_links). Replicate the historical writer cap: each unit
            # linked bidirectionally to up to MAX_LINKS_PER_ENTITY others sharing
            # each entity. Aggregated to a single scalar — the per-fact-type
            # slice doubled the join cost and only fed link_counts_by_fact_type
            # / link_breakdown, which the UI ignores and the CLI renders into
            # sections that degrade gracefully when empty.
            max_links_per_entity = 10
            entity_total_row = await conn.fetchrow(
                f"""
                WITH per_entity AS (
                    SELECT ue.entity_id, COUNT(*) AS n
                    FROM {fq_table("unit_entities")} ue
                    JOIN {fq_table("memory_units")} mu ON mu.id = ue.unit_id
                    WHERE mu.bank_id = $1
                    GROUP BY ue.entity_id
                )
                SELECT COALESCE(SUM(LEAST(n - 1, $2)), 0)::bigint AS count
                FROM per_entity
                """,
                bank_id,
                max_links_per_entity,
            )
            entity_link_total = int(entity_total_row["count"] or 0) if entity_total_row else 0

            link_counts: dict[str, int] = {row["link_type"]: row["count"] for row in non_entity_link_rows}
            if entity_link_total > 0:
                link_counts["entity"] = entity_link_total

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

            # link_counts_by_fact_type and link_breakdown are retained in the
            # response shape but no longer populated — producing them required
            # the expensive memory_links⇒memory_units join we just deleted. The
            # UI does not read them; hindsight-cli `bank stats` does, and after
            # this change its "Links by Fact Type" section prints empty and the
            # "Detailed Link Breakdown" section is skipped (`is_empty()` guard).
            # Drop these keys, and the matching CLI rendering, once downstream
            # SDKs are regenerated.
            return {
                "bank_id": bank_id,
                "node_counts": node_counts,
                "link_counts": link_counts,
                "link_counts_by_fact_type": {},
                "link_breakdown": [],
                "operations": ops_by_status,
                "total_documents": doc_count_row["count"] if doc_count_row else 0,
                "last_consolidated_at": last_consolidated_at.isoformat() if last_consolidated_at else None,
                "pending_consolidation": consolidation_row["pending"] if consolidation_row else 0,
                "failed_consolidation": consolidation_row["failed"] if consolidation_row else 0,
                "total_observations": node_counts.get("observation", 0),
            }

    async def get_bank_freshness(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Cheap subset of bank stats consumed by reflect().

        Returns only the consolidation freshness fields: when the bank was last
        consolidated and how many units are pending or failed. reflect() calls
        this on every invocation, so it must not pay for any cross-table joins
        or link aggregations.
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_bank_stats", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        backend = await self._get_backend()
        # The current reflect() caller reads only last_consolidated_at and
        # pending_consolidation, but `failed` is part of this method's published
        # contract (see interface.get_bank_freshness) so the returned shape stays
        # a strict subset of get_bank_stats. All three come from one scan, so
        # keeping `failed` costs nothing extra.
        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
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

        if row is None:
            return {"last_consolidated_at": None, "pending_consolidation": 0, "failed_consolidation": 0}
        last = row["last_consolidated_at"]
        return {
            "last_consolidated_at": last.isoformat() if last else None,
            "pending_consolidation": row["pending"] or 0,
            "failed_consolidation": row["failed"] or 0,
        }

    async def _probe_llm(self, llm: Any) -> _LlmProbeOutcome:
        """Probe one LLM client (status only). The detailed provider error is logged
        server-side, never returned, so the probe leaks nothing about the LLM config."""
        # NoneLLM.verify_connection() is a no-op that succeeds, so detect "no LLM" by
        # the provider name rather than the probe result.
        if llm.provider == "none":
            return _LlmProbeOutcome(ok=False, status="not_configured", latency_ms=None)
        start = time.monotonic()
        try:
            await asyncio.wait_for(llm.verify_connection(), timeout=_LLM_PROBE_TIMEOUT_SECONDS)
            return _LlmProbeOutcome(ok=True, status="connected", latency_ms=(time.monotonic() - start) * 1000)
        except (TimeoutError, asyncio.TimeoutError):
            return _LlmProbeOutcome(ok=False, status="timeout", latency_ms=(time.monotonic() - start) * 1000)
        except Exception as e:
            logger.warning("LLM connectivity probe failed (provider=%s): %s", llm.provider, e)
            # A bad API key is the most common cause, so surface it distinctly (the
            # category leaks nothing — the raw error is only logged above).
            status = "auth_failed" if _is_auth_error(e) else "unreachable"
            return _LlmProbeOutcome(ok=False, status=status, latency_ms=(time.monotonic() - start) * 1000)

    async def check_bank_llm(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> BankLlmHealthInfo:
        """Probe the LLMs this bank would use for retain / consolidation / reflect (#2034).

        Deliberate, non-polled connectivity test so callers discover "not configured /
        unreachable" instead of a silent stall. Each operation can resolve to a different
        LLM, but they often share one; identical configs are probed **once** (keyed on
        provider/model/base_url/api_key) and the result fanned out. Returns status only —
        never the provider/model/endpoint or the raw error (those are logged server-side).
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankReadContext

            ctx = BankReadContext(bank_id=bank_id, operation="get_bank_stats", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_read(ctx))

        per_operation_llm = [
            ("retain", self._retain_llm_config),
            ("consolidation", self._consolidation_llm_config),
            ("reflect", self._reflect_llm_config),
        ]
        # Dedup key includes api_key so two ops with the same provider/model/url but
        # different keys are still probed separately. Keys never leave this method.
        probed: dict[tuple, _LlmProbeOutcome] = {}
        operations: list[LlmOperationHealthInfo] = []
        for operation, llm in per_operation_llm:
            key = (llm.provider, llm.model, llm.base_url, llm.api_key)
            if key not in probed:
                probed[key] = await self._probe_llm(llm)
            outcome = probed[key]
            operations.append(
                LlmOperationHealthInfo(
                    operation=operation, ok=outcome.ok, status=outcome.status, latency_ms=outcome.latency_ms
                )
            )
        return BankLlmHealthInfo(bank_id=bank_id, operations=operations)

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
            exists = await conn.fetchrow(
                f"SELECT id FROM {fq_table('mental_models')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                mental_model_id,
            )
            if exists is None:
                return None
            # History now lives in the dedicated mental_model_history table (one
            # row per refresh), returned most-recent-first. The snapshot fields
            # live in the JSONB ``content`` blob.
            rows = await conn.fetch(
                f"""
                SELECT content, changed_at
                FROM {fq_table("mental_model_history")}
                WHERE mental_model_id = $1 AND bank_id = $2
                ORDER BY changed_at DESC, id DESC
                """,
                mental_model_id,
                bank_id,
            )
            result: list[dict] = []
            for r in rows:
                content = r["content"]
                if isinstance(content, str):
                    content = json.loads(content) if content else {}
                content = content or {}
                changed_at = r["changed_at"]
                result.append(
                    {
                        "previous_content": content.get("previous_content"),
                        "previous_reflect_response": content.get("previous_reflect_response"),
                        "changed_at": changed_at.isoformat() if hasattr(changed_at, "isoformat") else changed_at,
                    }
                )
            return result

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

        # mental_models.bank_id has a FK to banks. Retain creates banks lazily;
        # pinned model creation must do the same. The lazy bank-create runs inside
        # the same transaction as the INSERT below, so a freshly-created bank never
        # outlives a mental-model insert that ultimately fails.
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                created = await self._ensure_bank_exists(bank_id, request_context, conn=conn)
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

        # Best-effort default-template hook runs after the bank-create commits
        # (it opens its own connections and can create pinned models).
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

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
            has_delta_baseline = bool(current_content) and current_content != MENTAL_MODEL_PENDING_CONTENT
            if refresh_mode == "delta" and has_delta_baseline:
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
                # Attribute these LLM calls to the mental-model refresh, not a
                # plain reflect, so traces group under the right operation.
                _operation_label="refresh_mental_model",
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
            # tool-call retries failing, transient provider errors, the cleaner
            # regex eating a JSON-dump that the LLM put in the answer field).
            # Writing "" to the DB would destroy the working document; on the
            # other hand silently returning the previous content masks upstream
            # failures from callers (workers, tests). So: preserve existing
            # content in the DB (and audit the failure via reflect_response),
            # then RAISE so the caller knows the refresh didn't happen.
            if not final_content.strip():
                logger.warning(
                    f"[MENTAL_MODELS] Refresh for {mental_model_id} produced empty content; "
                    "preserving previous content and raising MentalModelRefreshError."
                )
                reflect_response_payload["refresh_skipped"] = "empty_candidate"
                # Persist the reflect_response (so the failure is auditable) and
                # the source-query tracking, but do NOT touch content/structured.
                await self.update_mental_model(
                    bank_id,
                    mental_model_id,
                    reflect_response=reflect_response_payload,
                    last_refreshed_source_query=current_source_query,
                    request_context=request_context,
                )
                raise MentalModelRefreshError(
                    f"Refresh produced empty content for mental_model_id={mental_model_id} "
                    "(likely an upstream LLM failure). Previous content preserved in DB; "
                    "reflect_response.refresh_skipped == 'empty_candidate' for audit."
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

            # History snapshot is written to mental_model_history after the UPDATE.
            record_mm_history = False
            slim_reflect_response: dict[str, Any] | None = None

            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1

            if content is not None:
                updates.append(f"content = ${param_idx}")
                params.append(content)
                param_idx += 1
                updates.append("last_refreshed_at = NOW()")
                # Snapshot the previous version for history. The actual write goes
                # into the dedicated mental_model_history table after the UPDATE
                # (see _append_mental_model_history); we only store the slim slice
                # of previous_reflect_response that consumers read — `based_on`,
                # the fact references that backed that version. The full
                # reflect_response can be hundreds of KB (text, fact bodies,
                # scoring, embeddings), so persisting it per entry is wasteful.
                if get_config().enable_mental_model_history:
                    if previous_reflect_response is not None:
                        based_on = previous_reflect_response.get("based_on")
                        if based_on is not None:
                            slim_reflect_response = {"based_on": based_on}
                    record_mm_history = True
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

            # Persist the previous-version snapshot in the dedicated history table
            # (one row per refresh), then trim to the configured cap. Replaces the
            # old single-JSONB-column append, which rewrote the whole array (plus
            # TOAST) on every refresh and was capped by entry count, not size.
            if row is not None and record_mm_history:
                await self._append_mental_model_history(
                    conn,
                    bank_id,
                    mental_model_id,
                    previous_content,
                    slim_reflect_response,
                    get_config().mental_model_history_max_entries,
                )

            return self._row_to_mental_model(row) if row else None

    async def _append_mental_model_history(
        self,
        conn: Any,
        bank_id: str,
        mental_model_id: str,
        previous_content: str | None,
        previous_reflect_response: dict[str, Any] | None,
        max_entries: int,
    ) -> None:
        """Insert one refresh snapshot into mental_model_history, then delete the
        oldest rows beyond ``max_entries`` for this model. The snapshot is stored
        as a single JSONB ``content`` blob (per-row, so it stays small); bounding
        by row count keeps per-model history from growing without bound."""
        content = json.dumps(
            {"previous_content": previous_content, "previous_reflect_response": previous_reflect_response}
        )
        await conn.execute(
            f"""
            INSERT INTO {fq_table("mental_model_history")} (mental_model_id, bank_id, content, changed_at)
            VALUES ($1, $2, $3::jsonb, now())
            """,
            mental_model_id,
            bank_id,
            content,
        )
        if max_entries and max_entries > 0:
            await conn.execute(
                f"""
                DELETE FROM {fq_table("mental_model_history")}
                WHERE mental_model_id = $1 AND bank_id = $2
                  AND id NOT IN (
                      SELECT id FROM {fq_table("mental_model_history")}
                      WHERE mental_model_id = $1 AND bank_id = $2
                      ORDER BY changed_at DESC, id DESC
                      LIMIT $3
                  )
                """,
                mental_model_id,
                bank_id,
                max_entries,
            )

    async def clear_mental_model(
        self,
        bank_id: str,
        mental_model_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any] | None:
        """Clear a mental model's content so the next refresh performs a full re-synthesis.

        Resets content to an empty string and clears structured_content and
        last_refreshed_source_query.  This is useful for delta-mode models that
        have accumulated drift — after clearing, a normal /refresh will fall
        back to full mode because there is no delta baseline.

        Args:
            bank_id: Bank identifier
            mental_model_id: Mental model UUID
            request_context: Request context for authentication

        Returns:
            Updated mental model dict or None if not found
        """
        await self._authenticate_tenant(request_context)
        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(bank_id=bank_id, operation="clear_mental_model", request_context=request_context)
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))
        backend = await self._get_backend()

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE {fq_table("mental_models")}
                SET content = '',
                    structured_content = NULL,
                    last_refreshed_source_query = NULL
                WHERE bank_id = $1 AND id = $2
                RETURNING id, bank_id, name, source_query, content, tags,
                          last_refreshed_at, created_at, reflect_response,
                          max_tokens, trigger, structured_content
                """,
                bank_id,
                mental_model_id,
            )

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
        tags_match: TagsMatch = "any",
        tag_groups: list[TagGroup] | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
        request_context: "RequestContext",
        isolation_mode: bool = False,
    ) -> list[dict[str, Any]]:
        """List directives for a bank.

        Args:
            bank_id: Bank identifier
            tags: Optional flat tags to filter by
            tags_match: How to match tags - 'any', 'all', 'any_strict', or 'all_strict'
            tag_groups: Optional compound tag filter (mutually independent of tags;
                if both are provided each applies its own OR-with-untagged wrapping
                and the two are AND-ed together)
            active_only: Only return active directives (default True)
            limit: Maximum number of results
            offset: Offset for pagination
            request_context: Request context for authentication
            isolation_mode: When True and both tags and tag_groups are None, only
                return directives with no tags. This prevents tag-scoped directives
                from leaking into untagged operations. Default False (normal API
                behavior - returns all directives when no tag filter is supplied).

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

            # Apply tag filters for directives:
            # Directives have special scoping rules:
            #   - Untagged directives (tags=[] or null) always apply regardless of reflect tags
            #   - Tagged directives only apply when the reflect operation includes matching tags
            #   - If no tag filter is supplied and isolation_mode=True: only untagged directives
            #   - If no tag filter is supplied and isolation_mode=False: all directives (normal API behavior)
            #
            # When `tags` and `tag_groups` are both supplied (engine-level callers only;
            # the public reflect/recall API rejects the combo at the request validator),
            # both filters apply independently — each wrapped in the untagged-OR rule —
            # so the directive set is the intersection of what either filter would admit.
            if tags:
                tags_clause, tags_params, param_idx = build_tags_where_clause(
                    tags=tags, param_offset=param_idx, table_alias="", match=tags_match
                )
                if tags_clause:
                    # Always include untagged directives; tagged ones must match the reflect tags
                    scoped_clause = tags_clause.replace("AND ", "", 1)
                    filters.append(f"((tags IS NULL OR tags = '{{}}') OR ({scoped_clause}))")
                    params.extend(tags_params)
            if tag_groups:
                groups_clause, groups_params, param_idx = build_tag_groups_where_clause(
                    tag_groups, param_offset=param_idx
                )
                if groups_clause:
                    # Same untagged-OR rule as the flat-tags branch above.
                    scoped_clause = groups_clause.replace("AND ", "", 1)
                    filters.append(f"((tags IS NULL OR tags = '{{}}') OR ({scoped_clause}))")
                    params.extend(groups_params)
            if not tags and not tag_groups and isolation_mode:
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
                SELECT operation_id, operation_type, created_at, updated_at, status, error_message,
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
                result_metadata = conn.parse_json(row["result_metadata"]) or {}

                next_retry_at = row["next_retry_at"]
                operation_list.append(
                    {
                        "id": str(row["operation_id"]),
                        "task_type": row["operation_type"],
                        "items_count": result_metadata.get("items_count", 0),
                        "document_id": None,
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "status": row["status"],
                        "error_message": row["error_message"],
                        "retry_count": row["retry_count"] or 0,
                        "next_retry_at": next_retry_at.isoformat() if next_retry_at else None,
                        "progress": result_metadata.get("progress"),
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
                        "progress": result_metadata.get("progress"),
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
                        "progress": result_metadata.get("progress"),
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
        # The lazy bank-create shares the webhook insert's transaction so the two
        # commit (or roll back) atomically.
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                created = await self._ensure_bank_exists(bank_id, request_context, conn=conn)
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

        # Best-effort default-template hook runs after the bank-create commits.
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

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

    async def _has_other_pending_consolidation(
        self,
        *,
        bank_id: str,
        operation_id: str,
    ) -> bool:
        """Return True if any consolidation op other than ``operation_id`` is
        ``pending`` for ``bank_id``.

        Used by the task-retry path to skip retrying a transient consolidation
        failure when another pending op already covers the same bank — the other
        op will process the same unconsolidated rows when it runs.

        A check failure (DB hiccup) returns ``False`` so the caller proceeds
        with the normal retry path rather than swallowing a real failure.
        """
        backend = await self._get_backend()
        try:
            async with acquire_with_retry(backend) as conn:
                existing = await conn.fetchval(
                    f"""
                    SELECT 1 FROM {fq_table("async_operations")}
                    WHERE bank_id = $1
                      AND operation_type = 'consolidation'
                      AND status = 'pending'
                      AND operation_id != $2
                    LIMIT 1
                    """,
                    bank_id,
                    uuid.UUID(operation_id),
                )
            return existing is not None
        except Exception as e:
            logger.warning(f"Failed to check for other pending consolidation ops for bank {bank_id}: {e}")
            return False

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

        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                if dedupe_by_bank:
                    # Serialize concurrent submits for this bank so the dedup
                    # check-and-insert is atomic. A bare check-then-INSERT races
                    # under READ COMMITTED: two /consolidate calls (or a manual
                    # trigger racing a retain-driven submit / round-limit re-queue)
                    # both see no pending row and both insert, leaking duplicate
                    # pending ops that then pile up as retry_blocked and starve the
                    # bank (issue #1842). Locking the bank row serializes submits for
                    # this bank; it releases on commit below.
                    #
                    # FOR NO KEY UPDATE, not FOR UPDATE: async_operations has an FK to
                    # banks, so every async-op insert for this bank (a scoped
                    # consolidation, a batch-retain op, a webhook delivery, ...) takes a
                    # FOR KEY SHARE lock on the bank row. FOR UPDATE conflicts with
                    # FOR KEY SHARE and would block all of those during the submit;
                    # FOR NO KEY UPDATE still conflicts with itself (so two submits
                    # serialize) but not with FOR KEY SHARE (so those inserts proceed).
                    # On Oracle this rewrites to FOR UPDATE, which there does not block
                    # indexed-FK child inserts.
                    await conn.execute(
                        f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1 FOR NO KEY UPDATE",
                        bank_id,
                    )
                    # Only check 'pending', not 'processing': a processing task uses a
                    # watermark from when it started, so memories added after that need
                    # a fresh run regardless.
                    pending = await conn.fetch(
                        f"""
                        SELECT operation_id, task_payload FROM {fq_table("async_operations")}
                        WHERE bank_id = $1 AND operation_type = $2 AND status = 'pending'
                        """,
                        bank_id,
                        operation_type,
                    )
                    # Dedup only against an existing *unscoped* (full-bank) pending op.
                    # A pending scoped consolidation covers only its tag subset, so it
                    # must not swallow a full-bank sweep (#1842). The scope check is in
                    # Python because the JSON predicate isn't portable — Oracle's
                    # JSON_VALUE returns NULL for the array-valued observation_scopes.
                    # (Scoped submits never reach here: they pass dedupe_by_bank=False.)
                    for row in pending:
                        row_payload = row["task_payload"]
                        row_dict = json.loads(row_payload) if isinstance(row_payload, str) else (row_payload or {})
                        if row_dict.get("observation_scopes") is None:
                            logger.debug(
                                f"{operation_type} task already pending for bank_id={bank_id}, "
                                f"skipping duplicate (existing operation_id={row['operation_id']})"
                            )
                            return {
                                "operation_id": str(row["operation_id"]),
                                "deduplicated": True,
                            }

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

        # Pack items into child operations by token budget. An oversized
        # single item is emitted as its own un-chunked child rather than
        # being fragmented across siblings — workers have no
        # per-document serialization, so concurrent siblings would race
        # on the same document_id and trigger FK violations in the final
        # ANN pass (issue #1795). The worker's in-process splitter
        # handles intra-document chunking sequentially.
        sub_batches = _split_contents_into_async_children(cast(list[RetainContentDict], contents), tokens_per_batch)

        # Log splitting info if we actually split
        if len(sub_batches) > 1:
            sub_batch_sizes = [len(b) for b in sub_batches]
            if len(sub_batches) <= 20:
                logger.info(
                    f"Large async retain batch ({total_tokens:,} tokens from {len(contents)} items). "
                    f"Split into {len(sub_batches)} child operations: {sub_batch_sizes} items each"
                )
            else:
                logger.info(
                    f"Large async retain batch ({total_tokens:,} tokens from {len(contents)} items). "
                    f"Split into {len(sub_batches)} child operations "
                    f"(items per child: min={min(sub_batch_sizes)}, "
                    f"max={max(sub_batch_sizes)}, total={sum(sub_batch_sizes)})"
                )

        # Always create parent operation (even for single batch - simpler, more reliable code path)
        import uuid

        parent_operation_id = uuid.uuid4()
        backend = await self._get_backend()

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
                # async_operations.bank_id has a FK to banks. Create the bank
                # lazily inside this same transaction so it is atomic with the
                # parent + child operation rows.
                created = await self._ensure_bank_exists(bank_id, request_context, conn=conn)
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
                            f"Submitting child {i}/{len(sub_batches)}: {len(sub_batch)} items, {sub_batch_tokens:,} tokens"
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

        # Best-effort default-template hook runs after the bank-create commits.
        if created:
            await self._apply_default_bank_template(bank_id, request_context)

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
        observation_scopes: list[list[str]] | None = None,
    ) -> dict[str, Any]:
        """Submit a consolidation operation to run asynchronously.

        Deduplicates by bank_id - if there's already a pending consolidation for this bank,
        returns the existing operation_id instead of creating a new one.

        Args:
            bank_id: Bank identifier
            request_context: Request context for authentication
            observation_scopes: Optional list of tag scopes to consolidate. When provided,
                only unconsolidated memories matching at least one scope are processed.

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
        if observation_scopes is not None:
            task_payload["observation_scopes"] = observation_scopes

        # Skip bank-level deduplication when scoped — the caller wants a
        # targeted run that should not be merged into a pending full-bank sweep.
        dedupe = observation_scopes is None

        return await self._submit_async_operation(
            bank_id=bank_id,
            operation_type="consolidation",
            task_type="consolidation",
            task_payload=task_payload,
            dedupe_by_bank=dedupe,
        )

    async def submit_async_graph_maintenance(
        self,
        bank_id: str,
        *,
        request_context: "RequestContext",
    ) -> dict[str, Any]:
        """Submit a graph-maintenance job to drain ``graph_maintenance_queue`` for a bank.

        Idempotent: short-circuits with ``no_work=True`` when the queue is empty
        for this bank, so unconditional callers (e.g. every retain that may or
        may not have triggered a document upsert) don't generate empty worker
        tasks. Deduplicates by bank when an existing pending job is already
        scheduled.

        Returns:
            Dict with ``operation_id``. May contain ``no_work=True`` (and a
            null operation_id) when the queue was already empty.
        """
        await self._authenticate_tenant(request_context)

        # Cheap pre-check on the (bank_id, enqueued_at) index. Lets every
        # retain call this unconditionally without paying for an async_operations
        # row when there's nothing to do.
        backend = await self._get_backend()
        async with acquire_with_retry(backend) as conn:
            has_work = await conn.fetchval(
                f"SELECT 1 FROM {fq_table('graph_maintenance_queue')} WHERE bank_id = $1 LIMIT 1",
                bank_id,
            )
        if not has_work:
            return {"operation_id": None, "no_work": True}

        if self._operation_validator:
            from hindsight_api.extensions import BankWriteContext

            ctx = BankWriteContext(
                bank_id=bank_id, operation="submit_async_graph_maintenance", request_context=request_context
            )
            await self._validate_operation(self._operation_validator.validate_bank_write(ctx))

        task_payload: dict[str, Any] = {}
        if request_context.tenant_id:
            task_payload["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            task_payload["_api_key_id"] = request_context.api_key_id

        return await self._submit_async_operation(
            bank_id=bank_id,
            operation_type="graph_maintenance",
            task_type="graph_maintenance",
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
