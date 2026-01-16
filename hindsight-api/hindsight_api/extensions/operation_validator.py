"""Operation Validator Extension for validating retain/recall/reflect operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from hindsight_api.extensions.base import Extension

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import Budget
    from hindsight_api.engine.response_models import RecallResult as RecallResultModel
    from hindsight_api.engine.response_models import ReflectResult
    from hindsight_api.models import RequestContext


class OperationValidationError(Exception):
    """Raised when an operation fails validation."""

    def __init__(self, reason: str, status_code: int = 403):
        self.reason = reason
        self.status_code = status_code
        super().__init__(f"Operation validation failed: {reason}")


@dataclass
class ValidationResult:
    """Result of an operation validation."""

    allowed: bool
    reason: str | None = None
    status_code: int = 403  # Default to Forbidden

    @classmethod
    def accept(cls) -> "ValidationResult":
        """Create an accepted validation result."""
        return cls(allowed=True)

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
    """

    bank_id: str
    contents: list[dict]  # List of {content, context, event_date, document_id}
    request_context: "RequestContext"
    document_id: str | None = None
    fact_type_override: str | None = None
    confidence_score: float | None = None


@dataclass
class RecallContext:
    """Context for a recall operation validation (pre-operation).

    Contains ALL user-provided parameters for the recall operation.
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


@dataclass
class RefreshMentalModelContext:
    """Context for a refresh mental model operation validation (pre-operation).

    Contains ALL user-provided parameters for the refresh mental model operation.
    """

    bank_id: str
    model_id: str
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
    confidence_score: float | None
    # Result
    unit_ids: list[list[str]]  # List of unit IDs per content item
    success: bool = True
    error: str | None = None


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


@dataclass
class RefreshMentalModelResult:
    """Result context for post-refresh-mental-model hook.

    Contains the operation parameters and the result including token usage.
    """

    bank_id: str
    model_id: str
    request_context: "RequestContext"
    # Result
    model_name: str | None = None
    observations_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    success: bool = True
    error: str | None = None


class OperationValidatorExtension(Extension, ABC):
    """
    Validates and hooks into retain/recall/reflect operations.

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
        1. validate_retain/validate_recall/validate_reflect (pre-operation)
        2. [operation executes]
        3. on_retain_complete/on_recall_complete/on_reflect_complete (post-operation)
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
                - confidence_score: Optional confidence score

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

    @abstractmethod
    async def validate_refresh_mental_model(self, ctx: RefreshMentalModelContext) -> ValidationResult:
        """
        Validate a refresh mental model operation before execution.

        Called before the refresh mental model operation is processed.
        Return ValidationResult.reject() to prevent the operation from executing.

        Args:
            ctx: Context containing all user-provided parameters:
                - bank_id: Bank identifier
                - model_id: Mental model ID to refresh
                - request_context: Request context with auth info

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

    async def on_refresh_mental_model_complete(self, result: RefreshMentalModelResult) -> None:
        """
        Called after a refresh mental model operation completes (success or failure).

        Override this method to implement post-operation logic such as:
        - Token usage tracking and billing
        - Audit logging
        - Metrics collection

        Args:
            result: Result context containing:
                - bank_id: Bank identifier
                - model_id: Mental model ID
                - request_context: Request context with auth info
                - model_name: Name of the mental model (if success)
                - observations_count: Number of observations generated
                - input_tokens: Number of input tokens used
                - output_tokens: Number of output tokens used
                - total_tokens: Total tokens used (input + output)
                - duration_ms: Total operation duration in milliseconds
                - success: Whether the operation succeeded
                - error: Error message (if failed)
        """
        pass
