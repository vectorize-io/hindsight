"""Operation Validator Extension for validating retain/recall/reflect/consolidate operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.response_models import RecallResult as RecallResultModel
    from hindsight_api.engine.response_models import ReflectResult
    from hindsight_api.engine.search.tags import TagGroup, TagsMatch
    from hindsight_api.models import RequestContext


class OperationValidationError(Exception):
    """Raised when an operation fails validation."""

    def __init__(self, reason: str, status_code: int = 403):
        self.reason = reason
        self.status_code = status_code
        super().__init__(f"Operation validation failed: {reason}")


@dataclass
class ValidationResult:
    """Result of an operation validation.

    Validators return this to accept or reject an operation. When accepting,
    validators can optionally return modified data that the engine will use
    instead of the original request parameters. This enables context enrichment
    (e.g., injecting tags or tag_groups).
    """

    allowed: bool
    reason: str | None = None
    status_code: int = 403  # Default to Forbidden
    # Optional enrichment fields — returned by validator, used by engine if present.
    # None means "no modification" (engine uses original values).
    contents: list[dict] | None = None  # Enriched retain contents (e.g., injected tags/strategy)
    tags: list[str] | None = None  # Enriched recall tags
    tags_match: "TagsMatch | None" = None  # Enriched recall tags match mode
    tag_groups: "list[TagGroup] | None" = None  # Enriched recall tag_groups

    @classmethod
    def accept(cls) -> "ValidationResult":
        """Create an accepted validation result (no enrichment)."""
        return cls(allowed=True)

    @classmethod
    def accept_with(
        cls,
        *,
        contents: list[dict] | None = None,
        tags: list[str] | None = None,
        tags_match: "TagsMatch | None" = None,
        tag_groups: "list[TagGroup] | None" = None,
    ) -> "ValidationResult":
        """Create an accepted validation result with enriched data.

        The engine will use the returned values instead of the original request
        parameters. Only non-None fields are applied; None means "keep original".
        """
        return cls(
            allowed=True,
            contents=contents,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    @classmethod
    def reject(cls, reason: str, status_code: int = 403) -> "ValidationResult":
        """Create a rejected validation result with a reason and HTTP status code."""
        return cls(allowed=False, reason=reason, status_code=status_code)


# =============================================================================
# Pre-operation Contexts (all user-provided parameters)
# =============================================================================


@dataclass
class RetainContext:
    """Context for a retain operation validation (pre-operation).

    Contains ALL user-provided parameters for the retain operation.
    To enrich contents (e.g., inject tags or strategy), return them
    via ValidationResult.accept_with(contents=...).
    """

    bank_id: str
    contents: list[dict]  # List of {content, context, event_date, document_id, tags, strategy}
    request_context: "RequestContext"
    document_id: str | None = None
    fact_type_override: str | None = None


@dataclass
class RecallContext:
    """Context for a recall operation validation (pre-operation).

    Contains ALL user-provided parameters for the recall operation.
    To enrich tag filters (e.g., inject tag_groups), return them
    via ValidationResult.accept_with(tag_groups=...).
    """

    bank_id: str
    query: str
    request_context: "RequestContext"
    budget: "Budget | None" = None
    max_tokens: int = 4096
    enable_trace: bool = False
    fact_types: list[str] = field(default_factory=list)
    question_date: datetime | None = None
    include_entities: bool = False
    max_entity_tokens: int = 500
    include_chunks: bool = False
    max_chunk_tokens: int = 8192
    tags: list[str] | None = None
    tags_match: "TagsMatch" = "any"
    tag_groups: "list[TagGroup] | None" = None


@dataclass
class ReflectContext:
    """Context for a reflect operation validation (pre-operation).

    Contains ALL user-provided parameters for the reflect operation.
    """

    bank_id: str
    query: str
    request_context: "RequestContext"
    budget: "Budget | None" = None
    context: str | None = None


# =============================================================================
# Consolidation Pre-operation Context
# =============================================================================


@dataclass
class ConsolidateContext:
    """Context for a consolidation operation validation (pre-operation)."""

    bank_id: str
    request_context: "RequestContext"


# =============================================================================
# Post-operation Contexts (includes results)
# =============================================================================


@dataclass
class RetainResult:
    """Result context for post-retain hook.

    Contains the operation parameters and the result.
    """

    bank_id: str
    contents: list[dict]
    request_context: "RequestContext"
    document_id: str | None
    fact_type_override: str | None
    # Result
    unit_ids: list[list[str]]  # List of unit IDs per content item
    success: bool = True
    error: str | None = None
    # Actual LLM token usage (populated by engine when available)
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_total_tokens: int | None = None


@dataclass
class RecallResult:
    """Result context for post-recall hook.

    Contains the operation parameters and the result.
    """

    bank_id: str
    query: str
    request_context: "RequestContext"
    budget: "Budget | None"
    max_tokens: int
    enable_trace: bool
    fact_types: list[str]
    question_date: datetime | None
    include_entities: bool
    max_entity_tokens: int
    include_chunks: bool
    max_chunk_tokens: int
    # Result
    result: "RecallResultModel | None" = None
    success: bool = True
    error: str | None = None


@dataclass
class ReflectResultContext:
    """Result context for post-reflect hook.

    Contains the operation parameters and the result.
    """

    bank_id: str
    query: str
    request_context: "RequestContext"
    budget: "Budget | None"
    context: str | None
    # Result
    result: "ReflectResult | None" = None
    success: bool = True
    error: str | None = None


# =============================================================================
# Consolidation Post-operation Context
# =============================================================================


@dataclass
class ConsolidateResult:
    """Result context for post-consolidation hook."""

    bank_id: str
    request_context: "RequestContext"
    # Result
    processed: int = 0
    created: int = 0
    updated: int = 0
    success: bool = True
    error: str | None = None


# =============================================================================
# Bank Management Contexts
# =============================================================================


@dataclass
class BankReadContext:
    """Context for a bank read operation validation (pre-operation)."""

    bank_id: str
    operation: str  # "get_bank_profile", "get_bank_stats"
    request_context: "RequestContext"


@dataclass
class BankWriteContext:
    """Context for a bank write operation validation (pre-operation)."""

    bank_id: str
    operation: str  # "delete_bank", "update_bank", "update_bank_disposition", "set_bank_mission", "merge_bank_mission", "clear_observations", "clear_observations_for_memory"
    request_context: "RequestContext"


@dataclass
class BankListContext:
    """Context for filtering the bank list (post-query)."""

    banks: list[dict]
    request_context: "RequestContext"


@dataclass
class BankListResult:
    """Result of filtering the bank list."""

    banks: list[dict]


# =============================================================================
# Mental Model Contexts
# =============================================================================


@dataclass
class MentalModelGetContext:
    """Context for a mental model GET operation validation (pre-operation)."""

    bank_id: str
    mental_model_id: str
    request_context: "RequestContext"


@dataclass
class MentalModelRefreshContext:
    """Context for a mental model refresh/create operation validation (pre-operation)."""

    bank_id: str
    mental_model_id: str | None  # None for create (not yet assigned)
    request_context: "RequestContext"


@dataclass
class MentalModelGetResult:
    """Result context for post-mental-model-GET hook."""

    bank_id: str
    mental_model_id: str
    request_context: "RequestContext"
    output_tokens: int  # tokens in the returned content
    success: bool = True
    error: str | None = None


@dataclass
class MentalModelRefreshResult:
    """Result context for post-mental-model-refresh hook."""

    bank_id: str
    mental_model_id: str
    request_context: "RequestContext"
    query_tokens: int  # tokens in source_query
    output_tokens: int  # tokens in generated content
    context_tokens: int  # tokens in context (if any)
    facts_used: int  # facts referenced in based_on
    mental_models_used: int  # mental models referenced in based_on
    success: bool = True
    error: str | None = None


# =============================================================================
# File Conversion Post-operation Context
# =============================================================================


@dataclass
class FileConvertResult:
    """Result context for post-file-conversion hook.

    Fired after a file is converted to markdown, before the retain step.
    """

    bank_id: str
    parser_name: str
    filename: str
    output_chars: int
    output_text: str
    request_context: "RequestContext"
    success: bool = True
    error: str | None = None


class OperationValidatorExtension(Extension, ABC):
    """
    Validates and hooks into retain/recall/reflect/consolidate operations.

    This extension allows implementing custom logic such as:
    - Rate limiting (pre-operation)
    - Quota enforcement (pre-operation)
    - Permission checks (pre-operation)
    - Content filtering (pre-operation)
    - Usage tracking (post-operation)
    - Audit logging (post-operation)
    - Metrics collection (post-operation)

    Enable via environment variable:
        HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator

    Configuration is passed from prefixed environment variables:
        HINDSIGHT_API_OPERATION_VALIDATOR_MAX_REQUESTS=100
        -> config = {"max_requests": "100"}

    Hook execution order:
        1. validate_* (pre-operation)
        2. [operation executes]
        3. on_*_complete (post-operation)

    Outcomes for `validate_*` hooks:
        - accept: return `ValidationResult.accept()` (or `accept_with(...)`)
        - reject: return `ValidationResult.reject(reason, status_code)`
          (raises `OperationValidationError` upstream)
        - defer: raise `DeferOperation(exec_date, reason)` from
          `hindsight_api.worker.exceptions` to requeue the task for a
          future time without bumping `retry_count`. Worker-only — do
          not raise from `validate_recall` / `validate_reflect` in
          synchronous HTTP request paths, where it surfaces as a 500.

    Supported operations:
        - retain, recall, reflect (core memory operations)
        - consolidate (mental models consolidation)
    """

    # =========================================================================
    # Pre-operation validation hooks (abstract - must be implemented)
    # =========================================================================

    @abstractmethod
    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        """
        Validate a retain operation before execution.

        Called before the retain operation is processed. Return ValidationResult.reject()
        to prevent the operation from executing.

        Args:
            ctx: Context containing all user-provided parameters:
                - bank_id: Bank identifier
                - contents: List of content dicts
                - request_context: Request context with auth info
                - document_id: Optional document ID
                - fact_type_override: Optional fact type override

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        ...

    @abstractmethod
    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        """
        Validate a recall operation before execution.

        Called before the recall operation is processed. Return ValidationResult.reject()
        to prevent the operation from executing.

        Args:
            ctx: Context containing all user-provided parameters:
                - bank_id: Bank identifier
                - query: Search query
                - request_context: Request context with auth info
                - budget: Budget level
                - max_tokens: Maximum tokens to return
                - enable_trace: Whether to include trace info
                - fact_types: List of fact types to search
                - question_date: Optional date context for query
                - include_entities: Whether to include entity data
                - max_entity_tokens: Max tokens for entities
                - include_chunks: Whether to include chunks
                - max_chunk_tokens: Max tokens for chunks

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        ...

    @abstractmethod
    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        """
        Validate a reflect operation before execution.

        Called before the reflect operation is processed. Return ValidationResult.reject()
        to prevent the operation from executing.

        Args:
            ctx: Context containing all user-provided parameters:
                - bank_id: Bank identifier
                - query: Question to answer
                - request_context: Request context with auth info
                - budget: Budget level
                - context: Optional additional context

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        ...

    # =========================================================================
    # Post-operation hooks (optional - override to implement)
    # =========================================================================

    async def on_retain_complete(self, result: RetainResult) -> None:
        """
        Called after a retain operation completes (success or failure).

        Override this method to implement post-operation logic such as:
        - Usage tracking
        - Audit logging
        - Metrics collection
        - Notifications

        Args:
            result: Result context containing:
                - All original operation parameters
                - unit_ids: List of created unit IDs (if success)
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    async def on_recall_complete(self, result: RecallResult) -> None:
        """
        Called after a recall operation completes (success or failure).

        Override this method to implement post-operation logic such as:
        - Usage tracking
        - Audit logging
        - Metrics collection
        - Query analytics

        Args:
            result: Result context containing:
                - All original operation parameters
                - result: RecallResultModel (if success)
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    async def on_reflect_complete(self, result: ReflectResultContext) -> None:
        """
        Called after a reflect operation completes (success or failure).

        Override this method to implement post-operation logic such as:
        - Usage tracking
        - Audit logging
        - Metrics collection
        - Response analytics

        Args:
            result: Result context containing:
                - All original operation parameters
                - result: ReflectResult (if success)
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    # =========================================================================
    # Consolidation - Pre-operation validation hook (optional - override to implement)
    # =========================================================================

    async def validate_consolidate(self, ctx: ConsolidateContext) -> ValidationResult:
        """
        Validate a consolidation operation before execution.

        Override to implement custom validation logic for consolidation.

        Args:
            ctx: Context containing:
                - bank_id: Bank identifier
                - request_context: Request context with auth info

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        return ValidationResult.accept()

    # =========================================================================
    # Consolidation - Post-operation hook (optional - override to implement)
    # =========================================================================

    async def on_consolidate_complete(self, result: ConsolidateResult) -> None:
        """
        Called after a consolidation operation completes (success or failure).

        Override to implement post-operation logic such as usage tracking or audit logging.

        Args:
            result: Result context containing:
                - bank_id: Bank identifier
                - processed: Number of memories processed
                - created: Number of mental models created
                - updated: Number of mental models updated
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    # =========================================================================
    # File Conversion - Post-operation hook (optional - override to implement)
    # =========================================================================

    async def on_file_convert_complete(self, result: FileConvertResult) -> None:
        """
        Called after a file is converted to markdown (before the retain step).

        Override to implement post-conversion logic such as:
        - Billing for premium parsers (e.g., Iris)
        - Usage tracking
        - Audit logging

        Args:
            result: Result context containing:
                - bank_id: Bank identifier
                - parser_name: Name of the parser used (e.g., 'markitdown', 'iris')
                - filename: Original filename
                - output_chars: Character count of the converted markdown
                - request_context: Request context with auth info
                - success: Whether the conversion succeeded
                - error: Error message (if failed)
        """
        pass

    # =========================================================================
    # Mental Model - Pre-operation validation hook (optional - override to implement)
    # =========================================================================

    async def validate_mental_model_get(self, ctx: MentalModelGetContext) -> ValidationResult:
        """
        Validate a mental model GET operation before execution.

        Override to implement custom validation logic for mental model retrieval.

        Args:
            ctx: Context containing:
                - bank_id: Bank identifier
                - mental_model_id: Mental model identifier
                - request_context: Request context with auth info

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        return ValidationResult.accept()

    async def validate_mental_model_refresh(self, ctx: MentalModelRefreshContext) -> ValidationResult:
        """
        Validate a mental model refresh/create operation before execution.

        Override to implement custom validation logic for mental model refresh.

        Args:
            ctx: Context containing:
                - bank_id: Bank identifier
                - mental_model_id: Mental model identifier (None for create)
                - request_context: Request context with auth info

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        return ValidationResult.accept()

    # =========================================================================
    # Mental Model - Post-operation hooks (optional - override to implement)
    # =========================================================================

    async def on_mental_model_get_complete(self, result: MentalModelGetResult) -> None:
        """
        Called after a mental model GET operation completes (success or failure).

        Override to implement post-operation logic such as tracking or audit logging.

        Args:
            result: Result context containing:
                - bank_id: Bank identifier
                - mental_model_id: Mental model identifier
                - output_tokens: Token count of the returned content
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    async def on_mental_model_refresh_complete(self, result: MentalModelRefreshResult) -> None:
        """
        Called after a mental model refresh operation completes (success or failure).

        Override to implement post-operation logic such as tracking or audit logging.

        Args:
            result: Result context containing:
                - bank_id: Bank identifier
                - mental_model_id: Mental model identifier
                - query_tokens: Tokens in source_query
                - output_tokens: Tokens in generated content
                - context_tokens: Tokens in context
                - facts_used: Number of facts referenced
                - mental_models_used: Number of mental models referenced
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass

    # =========================================================================
    # Bank Management - Validation hooks (optional - override to implement)
    # =========================================================================

    async def validate_bank_read(self, ctx: BankReadContext) -> ValidationResult:
        """
        Validate a bank read operation before execution.

        Override to implement custom validation logic for bank reads
        (get_bank_profile, get_bank_stats).

        Args:
            ctx: Context containing:
                - bank_id: Bank identifier
                - operation: Operation name
                - request_context: Request context with auth info

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        return ValidationResult.accept()

    async def validate_bank_write(self, ctx: BankWriteContext) -> ValidationResult:
        """
        Validate a bank write operation before execution.

        Override to implement custom validation logic for bank writes
        (delete_bank, update_bank, update_bank_disposition, set_bank_mission,
        merge_bank_mission, clear_observations, clear_observations_for_memory).

        Args:
            ctx: Context containing:
                - bank_id: Bank identifier
                - operation: Operation name
                - request_context: Request context with auth info

        Returns:
            ValidationResult indicating whether the operation is allowed.
        """
        return ValidationResult.accept()

    async def filter_bank_list(self, ctx: BankListContext) -> BankListResult:
        """
        Filter the bank list after querying.

        Unlike validate_* methods, this is a post-query filter that narrows results
        rather than a gate that blocks the operation.

        Override to implement custom filtering (e.g., restrict to allowed banks).

        Args:
            ctx: Context containing:
                - banks: List of bank dicts from the database
                - request_context: Request context with auth info

        Returns:
            BankListResult with the filtered list of banks.
        """
        return BankListResult(banks=ctx.banks)

    async def filter_mcp_tools(
        self,
        bank_id: str,
        request_context: "RequestContext",
        tools: frozenset[str],
    ) -> frozenset[str]:
        """
        Filter MCP tools visible to this user on this bank.

        Called during tools/list after bank-level mcp_enabled_tools filtering.
        The input set is already narrowed by bank config — this method can only
        remove tools, never add ones the bank config excluded.

        Default: return all tools unchanged (no per-user filtering).

        Args:
            bank_id: Target bank ID (from URL path or header).
            request_context: Authenticated context with tenant_id set.
            tools: Tools remaining after bank config filtering.

        Returns:
            Subset of tools this user should see.
        """
        return tools
