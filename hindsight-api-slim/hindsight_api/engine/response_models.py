"""
Core response models for Hindsight memory system.

These models define the structure of data returned by the core MemoryEngine class.
API response models should be kept separate and convert from these core models to maintain
API stability even if internal models change.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .metadata_utils import as_string_metadata
from .search.tag_resolution import needs_resolution
from .search.tags import TagGroup, TagsMatch
from .structured_output import validate_response_schema

VALID_RECALL_FACT_TYPES = frozenset(["world", "experience", "observation"])


class Budget(str, Enum):
    """Budget levels for recall/reflect operations."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


def FieldWithDefault(default_factory: Callable, **kwargs) -> Any:
    """
    Field wrapper that ensures default_factory values appear in OpenAPI schema.

    Pydantic doesn't include default_factory in OpenAPI schemas, causing OpenAPI
    Generator to make fields Optional with default=None instead of non-optional
    with the correct default value.

    This wrapper adds json_schema_extra to include the default in the schema.
    """
    # Determine the default value for the schema based on the factory
    if default_factory is list:
        schema_default = []
    elif default_factory is dict:
        schema_default = {}
    else:
        # For custom factories (like IncludeOptions), use empty dict as placeholder
        schema_default = {}

    # Add or merge json_schema_extra
    json_extra = kwargs.pop("json_schema_extra", {})
    if isinstance(json_extra, dict):
        json_extra["default"] = schema_default
    else:
        # If json_schema_extra was a function, we can't merge easily
        # Fall back to just setting default
        json_extra = {"default": schema_default}

    return Field(default_factory=default_factory, json_schema_extra=json_extra, **kwargs)


class LLMToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(description="Arguments to pass to the tool")
    thought_signature: str | None = Field(
        default=None,
        description="Opaque token required by Gemini 3.1+ thinking models to preserve thought context across turns",
    )


class LLMToolCallResult(BaseModel):
    """Result from an LLM call that may include tool calls."""

    content: str | None = Field(default=None, description="Text content if any")
    tool_calls: list[LLMToolCall] = Field(default_factory=list, description="Tool calls requested by the LLM")
    finish_reason: str | None = Field(default=None, description="Reason the LLM stopped: 'stop', 'tool_calls', etc.")
    input_tokens: int = Field(
        default=0,
        description="Input tokens used in this call (includes any cached prefix tokens reported by the provider)",
    )
    output_tokens: int = Field(
        default=0, description="Visible output tokens used in this call (excludes reasoning/thoughts)"
    )
    cached_tokens: int = Field(
        default=0, description="Cached prefix tokens, when reported by the provider. Subset of input_tokens."
    )
    thoughts_tokens: int = Field(
        default=0,
        description="Reasoning/thinking tokens. Billed at the output rate by some providers but not part of visible output.",
    )


class ToolCallTrace(BaseModel):
    """A single tool call made during reflect."""

    tool: str = Field(description="Tool name: lookup, recall, learn, expand")
    reason: str | None = Field(default=None, description="Agent's reasoning for making this tool call")
    input: dict = Field(description="Tool input parameters")
    output: dict = Field(description="Tool output/result")
    duration_ms: int = Field(description="Execution time in milliseconds")
    iteration: int = Field(default=0, description="Iteration number (1-based) when this tool was called")


class LLMCallTrace(BaseModel):
    """A single LLM call made during reflect."""

    scope: str = Field(description="Call scope: agent_1, agent_2, final, etc.")
    duration_ms: int = Field(description="Execution time in milliseconds")


class ObservationRef(BaseModel):
    """Reference to an observation accessed during reflect."""

    id: str = Field(description="Observation ID")
    name: str = Field(description="Observation name")
    type: str = Field(description="Observation type: entity, concept, event")
    subtype: str = Field(description="Observation subtype: structural, emergent, learned")
    description: str = Field(description="Brief description")
    summary: str | None = Field(default=None, description="Full summary (when looked up in detail)")


class DirectiveRef(BaseModel):
    """Reference to a directive that was applied during reflect."""

    id: str = Field(description="Directive mental model ID")
    name: str = Field(description="Directive name")
    content: str = Field(description="Directive content")


class TokenUsage(BaseModel):
    """
    Token usage metrics for LLM calls.

    Tracks input/output tokens for a single request to enable
    per-request cost tracking and monitoring.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_tokens": 1500,
                "output_tokens": 500,
                "total_tokens": 2000,
            }
        }
    )

    input_tokens: int = Field(default=0, description="Number of input/prompt tokens consumed")
    output_tokens: int = Field(
        default=0, description="Number of visible output/completion tokens generated (excludes reasoning/thoughts)"
    )
    total_tokens: int = Field(default=0, description="Total tokens (input + output, excludes thoughts)")
    cached_tokens: int = Field(default=0, description="Cached/cache-read prompt tokens, when reported by the provider")
    thoughts_tokens: int = Field(
        default=0,
        description=(
            "Reasoning/thinking tokens generated by the model. Billed at the output rate by some providers "
            "(e.g. Gemini 2.5+ family) but not surfaced in the visible response."
        ),
    )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Allow aggregating token usage from multiple calls."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            thoughts_tokens=self.thoughts_tokens + other.thoughts_tokens,
        )


class LLMCallResult(BaseModel):
    """What one ``LLMInterface.call`` produced: the response, and what it cost.

    ``call`` used to return either the bare response or a ``(response, usage)``
    tuple depending on a ``return_usage: bool`` argument — a boolean that changed
    the return type, which is why all fourteen implementations were annotated
    ``-> Any`` and neither shape was ever checked. Callers that wanted the cost
    unpacked two names; callers that did not got the response directly, and
    nothing could tell you at the call site which you were looking at.

    Returning this unconditionally mirrors ``call_with_tools``, which has always
    returned a named ``LLMToolCallResult``. Usage is always available, so a caller
    that wants to record cost no longer has to change the call's return type to
    get it.
    """

    content: Any = Field(
        default=None,
        description=(
            "The parsed response when a response_format was given, otherwise the text content. "
            "Typed Any because the parsed shape is the caller's own response_format model."
        ),
    )
    usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Tokens this call consumed. Zero-valued when the provider reports none.",
    )


class ExtractedFact(BaseModel):
    """A single candidate fact produced by dry-run extraction (no resolution/links/persistence).

    A deliberate subset of the persisted memory-unit shape — only the fields a fresh extraction
    yields. Storage/consolidation/curation fields (id, document_id, chunk_id, proof_count, state, …)
    are omitted because nothing is stored. Entities are raw, unresolved names.
    """

    text: str = Field(description="The extracted fact text.")
    fact_type: str = Field(description="Perspective classification: 'world' or 'experience'.")
    occurred_start: str | None = Field(default=None, description="ISO timestamp the fact's event started, if dated.")
    occurred_end: str | None = Field(default=None, description="ISO timestamp the fact's event ended, if dated.")
    entities: list[str] = Field(
        default_factory=list, description="Raw (unresolved) entity names mentioned in the fact."
    )
    chunk_index: int | None = Field(
        default=None,
        description="Index into `chunks` of the chunk this fact came from; null if it could not be attributed.",
    )
    attachments: list["ExtractedFactAttachment"] = Field(
        default_factory=list,
        description="Attachments from user input that this fact is attributed to / associated with.",
    )


class ExtractedFactAttachment(BaseModel):
    """An attachment from multimodal input associated with an extracted fact."""

    block_index: int = Field(description="Index of the content block in user's input (0-based)")
    type: Literal["image", "file"] = Field(
        description="Content block type ('image' or 'file')",
        title="AttachmentType",
    )
    media_type: str = Field(description="MIME media type of the attachment, e.g. 'image/png'")


class ExtractionChunk(BaseModel):
    """One chunk the extractor was handed, and how much it yielded."""

    text: str = Field(description="The chunk as the extractor saw it.")
    fact_count: int = Field(description="How many facts came out of this chunk.")


class DryRunExtractionResult(BaseModel):
    """Result of dry-run fact extraction: candidate facts, the chunks they came from,
    and aggregated LLM token usage."""

    facts: list[ExtractedFact] = Field(
        default_factory=list, description="Candidate facts the retain step would extract."
    )
    chunks: list[ExtractionChunk] = Field(
        default_factory=list,
        description=(
            "The chunks the input was cut into before extraction. Already computed on every "
            "path; returned because `retain_chunk_size` is otherwise a number with no visible effect."
        ),
    )
    usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Aggregated token usage across the extraction LLM calls."
    )


class DispositionTraits(BaseModel):
    """
    Disposition traits for a memory bank.

    All traits are scored 1-5 where:
    - skepticism: 1=trusting, 5=skeptical (how much to doubt or question information)
    - literalism: 1=flexible interpretation, 5=literal interpretation (how strictly to interpret information)
    - empathy: 1=detached, 5=empathetic (how much to consider emotional context)
    """

    skepticism: int = Field(ge=1, le=5, description="How skeptical vs trusting (1=trusting, 5=skeptical)")
    literalism: int = Field(ge=1, le=5, description="How literally to interpret information (1=flexible, 5=literal)")
    empathy: int = Field(ge=1, le=5, description="How much to consider emotional context (1=detached, 5=empathetic)")

    model_config = ConfigDict(json_schema_extra={"example": {"skepticism": 3, "literalism": 3, "empathy": 3}})


class RecallScores(BaseModel):
    """Per-result recall scores from different stages of the pipeline.

    ``final`` is the value results are ranked by. The others are diagnostic and
    can be filtered on via the recall ``min_scores`` request parameter. ``semantic``
    and ``keyword`` are the raw per-strategy retrieval scores (``None`` when that
    strategy did not surface this result); ``reranker`` is the cross-encoder's
    normalized relevance.
    """

    final: float = Field(description="Final ranking score (combined reranker + recency/temporal/proof boosts)")
    reranker: float | None = Field(
        default=None,
        description="Cross-encoder relevance, normalized 0-1. None when the reranker is a passthrough (rrf/interleave modes).",
    )
    semantic: float | None = Field(
        default=None, description="Vector cosine similarity (0-1). None if this result was not surfaced semantically."
    )
    keyword: float | None = Field(
        default=None,
        description="Keyword/full-text (BM25) score (>= 0, unbounded). None if this result was not surfaced by keyword search.",
    )


class MinScores(BaseModel):
    """Optional per-stage score floors for recall. Every floor is inclusive (``>=``).

    The four floors act at two different levels, and the distinction decides what a
    returned result is guaranteed to satisfy.

    ``semantic`` and ``keyword`` are **retrieval-level** cutoffs pushed into their own
    SQL arm (overriding the global ``semantic_min_similarity`` / ``bm25_min_score``
    config for this request), so they prune weak matches before fusion. Each one
    constrains **only the arm it names**. Recall fuses four arms — semantic, keyword,
    graph and temporal — and a result reaches the response if *any* arm surfaced it,
    so a returned result may legitimately carry ``null`` for a stage it was not
    surfaced by, and results reached through the graph or temporal arm carry neither
    ``semantic`` nor ``keyword``. A *non-null* score always clears its floor — the
    gap is only ever a ``null``. Setting both does **not** restrict the response to
    results that clear both: they are not a predicate over each fused result. This is
    deliberate — an intersection would discard the strong single-arm matches that
    hybrid retrieval exists to find (a paraphrase with no lexical overlap, an exact
    identifier the embedding scores poorly).

    ``reranker`` and ``final`` are **post-query** filters applied to every scored
    result after fusion and reranking, so these *are* per-result predicates: a
    returned result always clears them. Use them, not the retrieval floors, to make
    recall abstain on low-confidence queries.

    Any field left None imposes no floor; all-None (the default) means no score
    filtering.
    """

    semantic: float | None = Field(
        default=None,
        description="Retrieval-level, semantic arm only: minimum vector similarity (0-1). A result the semantic "
        "arm did not surface reports `semantic: null` and is unaffected by this floor.",
    )
    keyword: float | None = Field(
        default=None,
        description="Retrieval-level, keyword arm only: minimum keyword/full-text (BM25) score. A result the "
        "keyword arm did not surface reports `keyword: null` and is unaffected by this floor.",
    )
    reranker: float | None = Field(
        default=None,
        description="Post-query: minimum normalized reranker score (0-1). Applied to every returned result. "
        "Rejected with HTTP 400 when the reranker scores by rank position (TypeSafe), since a floor would only "
        "keep a fixed share of the results.",
    )
    final: float | None = Field(
        default=None,
        description="Post-query: minimum final ranking score. Applied to every returned result.",
    )


class TemporalWindow(BaseModel):
    """A caller-supplied window for recall's temporal arm.

    Supplying this **replaces the date extraction** recall would otherwise run
    over the query text, so a caller that already knows the range it means does
    not have to phrase it in English and hope the parser agrees.

    It is **not a filter**. The temporal arm is one of four retrieval arms: it
    surfaces memories whose own dates (``mentioned_at`` / ``occurred_start`` /
    ``occurred_end``) fall inside the window so fusion can rank them higher.
    The semantic, keyword and graph arms are unaffected, so results dated
    outside the window are still returned. Narrowing results to a date range is
    a different operation this does not provide.

    Has no effect when the bank's ``enable_temporal_retrieval`` config is off:
    that flag gates the arm itself, and stays the single switch for it.
    """

    start: datetime = Field(description="Start of the window (inclusive).")
    end: datetime = Field(description="End of the window (inclusive).")

    @field_validator("start", "end")
    @classmethod
    def ensure_tz_aware(cls, v: datetime) -> datetime:
        # The arm coerces naive datetimes to UTC before querying; do it here
        # instead so both bounds are unambiguous the moment the request parses.
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @model_validator(mode="after")
    def validate_order(self) -> "TemporalWindow":
        if self.end < self.start:
            raise ValueError("temporal_window.end must not be earlier than temporal_window.start")
        return self


class MemoryFact(BaseModel):
    """
    A single memory fact returned by search or think operations.

    This represents a unit of information stored in the memory system,
    including both the content and metadata.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "Alice works at Google on the AI team",
                "fact_type": "world",
                "entities": ["Alice", "Google"],
                "context": "work info",
                "occurred_start": "2024-01-15T10:30:00Z",
                "occurred_end": "2024-01-15T10:30:00Z",
                "mentioned_at": "2024-01-15T10:30:00Z",
                "document_id": "session_abc123",
                "metadata": {"source": "slack"},
                "chunk_id": "bank123_session_abc123_0",
                "activation": 0.95,
                "tags": ["user_a", "session_123"],
            }
        }
    )

    id: str = Field(description="Unique identifier for the memory fact")
    text: str = Field(description="The actual text content of the memory")
    fact_type: str = Field(description="Type of fact: 'world', 'experience', or 'observation'")
    entities: list[str] | None = Field(None, description="Entity names mentioned in this fact")
    context: str | None = Field(None, description="Additional context for the memory")
    occurred_start: str | None = Field(None, description="ISO format date when the event started occurring")
    occurred_end: str | None = Field(None, description="ISO format date when the event ended occurring")
    mentioned_at: str | None = Field(None, description="ISO format date when the fact was mentioned/learned")
    document_id: str | None = Field(None, description="ID of the document this memory belongs to")
    metadata: dict[str, str] | None = Field(None, description="User-defined metadata")

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> dict[str, str] | None:
        """Parse metadata from JSON string if needed (asyncpg may return JSONB as str).

        Also coerces non-string dict values (e.g., integer IDs stored in JSONB)
        to strings, preventing ValidationError when consolidation encounters
        metadata like {"original_id": 348} instead of {"original_id": "348"}.
        Null-valued keys are dropped rather than stringified to "None" (issue
        #3209), so rows written before retain normalized its input stay
        readable without a data migration.
        """
        if v is None:
            return None
        if isinstance(v, str):
            import json

            v = json.loads(v)
        if isinstance(v, dict):
            return as_string_metadata(v)
        return v

    chunk_id: str | None = Field(
        None, description="ID of the chunk this fact was extracted from (format: bank_id_document_id_chunk_index)"
    )
    tags: list[str] | None = Field(None, description="Visibility scope tags associated with this fact")
    source_fact_ids: list[str] | None = Field(
        None,
        description="IDs of source facts this observation was derived from (observation type only, when source_facts is enabled)",
    )
    scores: RecallScores | None = Field(
        None,
        description="Recall scores from each pipeline stage (final/reranker/semantic/keyword). Not returned for source facts.",
    )
    # Internal, never serialised: the short ids of the attachments this fact was drawn
    # from, when the memories store returned them on the row. ``None`` means "not
    # carried", and the HTTP layer then reads them from `memory_units` instead; a list
    # (possibly empty) is resolved as-is, so a store that owns its rows is never asked twice.
    attachment_ids: list[str] | None = Field(None, exclude=True)


class ChunkInfo(BaseModel):
    """Information about a chunk."""

    chunk_text: str = Field(description="The raw chunk text")
    chunk_index: int = Field(description="Index of the chunk within the document")
    truncated: bool = Field(default=False, description="Whether the chunk was truncated due to token limits")


class ObservationResult(BaseModel):
    """An observation result from recall (consolidated knowledge synthesized from facts)."""

    id: str = Field(description="Unique observation ID")
    text: str = Field(description="The observation text")
    proof_count: int = Field(description="Number of facts supporting this observation")
    relevance: float = Field(default=0.0, description="Relevance score to the query")
    tags: list[str] | None = Field(default=None, description="Tags for visibility scoping")
    source_memory_ids: list[str] = Field(
        default_factory=list, description="IDs of facts that contribute to this observation"
    )


class MentalModelResult(BaseModel):
    """A mental model result from recall (stored reflect response)."""

    id: str = Field(description="Unique mental model ID")
    name: str = Field(description="Human-readable name")
    content: str = Field(description="The synthesized content")
    relevance: float = Field(default=0.0, description="Relevance score to the query")


class RecallResult(BaseModel):
    """
    Result from a recall operation.

    Contains a list of matching memory facts and optional trace information
    for debugging and transparency.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "text": "Alice works at Google on the AI team",
                        "fact_type": "world",
                        "context": "work info",
                        "occurred_start": "2024-01-15T10:30:00Z",
                        "occurred_end": "2024-01-15T10:30:00Z",
                        "activation": 0.95,
                    }
                ],
                "trace": {"query": "What did Alice say about machine learning?", "num_results": 1},
            }
        }
    )

    results: list[MemoryFact] = Field(description="List of memory facts matching the query")
    trace: dict[str, Any] | None = Field(None, description="Trace information for debugging")
    #: Per-stage timings, in microseconds, measured INSIDE a store that answered the whole recall
    #: (``MemoriesExtension.full_recall``). ``None`` when the engine ran its own pipeline, where the
    #: stages are already in the trace. Excluded from the API response — it exists so the recall
    #: trace can still attribute time once the stages no longer run here.
    store_stages: dict[str, int] | None = Field(default=None, exclude=True)
    entities: dict[str, "EntityState"] | None = Field(
        None, description="Entity states for entities mentioned in results (keyed by canonical name)"
    )
    chunks: dict[str, ChunkInfo] | None = Field(
        None, description="Chunks for facts, keyed by '{document_id}_{chunk_index}'"
    )
    source_facts: dict[str, MemoryFact] | None = Field(
        None, description="Source facts for observation-type results, keyed by fact ID"
    )
    source_facts_truncated: bool | None = Field(
        None,
        description=(
            "Whether the source_facts map was cut short by the token budget. When true, some IDs in "
            "results[].source_fact_ids have no entry in source_facts — the budget ran out, the "
            "references are not dangling. Only set when source facts were requested."
        ),
    )


class ReflectResult(BaseModel):
    """
    Result from a reflect operation.

    Contains the formulated answer, the facts it was based on (organized by type),
    any new opinions that were formed during the reflection process, and optionally
    structured output if a response schema was provided.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Based on my knowledge, machine learning is being actively used in healthcare...",
                "based_on": {
                    "world": [
                        {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "text": "Machine learning is used in medical diagnosis",
                            "fact_type": "world",
                            "context": "healthcare",
                            "occurred_start": "2024-01-15T10:30:00Z",
                            "occurred_end": "2024-01-15T10:30:00Z",
                        }
                    ],
                    "experience": [],
                    "opinion": [],
                    "observation": [],
                    "mental-models": [],
                    "directives": [
                        {
                            "id": "directive-123",
                            "name": "Response Style",
                            "rules": ["Always be concise"],
                        }
                    ],
                },
                "structured_output": {"summary": "ML in healthcare", "confidence": 0.9},
                "usage": {"input_tokens": 1500, "output_tokens": 500, "total_tokens": 2000},
            }
        }
    )

    text: str = Field(description="The formulated answer text")
    document: Any | None = Field(
        default=None,
        exclude=True,
        description=(
            "Internal: the structured document the agent emitted when the caller asked for one. "
            "``text`` is its deterministic render. Excluded from the API response — callers read "
            "the rendered text; the structure is persistence-side state."
        ),
    )
    based_on: dict[str, Any] = Field(
        description="Facts used to formulate the answer, organized by type (world, experience, observation, mental-models, directives)"
    )
    structured_output: dict[str, Any] | None = Field(
        default=None,
        description="Structured output parsed according to the provided response schema. Only present when response_schema was provided.",
    )
    structured_output_error: str | None = Field(
        default=None,
        description=(
            "Why structured output could not be produced, when a response_schema was provided and the "
            "extraction call failed. Absent when extraction succeeded, so a null structured_output "
            "without this field means the answer held nothing matching the schema."
        ),
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage metrics for the LLM calls made during this reflect operation.",
    )
    tool_trace: list[ToolCallTrace] = Field(
        default_factory=list,
        description="Trace of tool calls made during reflection. Only present when include.tool_calls is enabled.",
    )
    llm_trace: list[LLMCallTrace] = Field(
        default_factory=list,
        description="Trace of LLM calls made during reflection. Only present when include.tool_calls is enabled.",
    )
    directives_applied: list[DirectiveRef] = Field(
        default_factory=list,
        description="Directive mental models that were applied during this reflection.",
    )


class EntityObservation(BaseModel):
    """
    An observation about an entity.

    Observations are objective facts synthesized from multiple memory facts
    about an entity, without personality influence.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "John is detail-oriented and works at Google", "mentioned_at": "2024-01-15T10:30:00Z"}
        }
    )

    text: str = Field(description="The observation text")
    mentioned_at: str | None = Field(None, description="ISO format date when this observation was created")


class EntityState(BaseModel):
    """
    Current mental model of an entity.

    Contains observations synthesized from facts about the entity.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entity_id": "123e4567-e89b-12d3-a456-426614174000",
                "canonical_name": "John",
                "observations": [
                    {"text": "John is detail-oriented", "mentioned_at": "2024-01-15T10:30:00Z"},
                    {"text": "John works at Google on the AI team", "mentioned_at": "2024-01-14T09:00:00Z"},
                ],
            }
        }
    )

    entity_id: str = Field(description="Unique identifier for the entity")
    canonical_name: str = Field(description="Canonical name of the entity")
    observations: list[EntityObservation] = Field(
        default_factory=list, description="List of observations about this entity"
    )


class MentalModel(BaseModel):
    """
    A manually configured mental model for tracking specific topics/areas.

    Mental models are user-defined focus areas that the agent should track
    and maintain summaries for, unlike auto-extracted entities.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "team-dynamics",
                "name": "Team Dynamics",
                "description": "Track how the team collaborates, communication patterns, conflicts, and resolutions",
                "summary": "The team has strong collaboration...",
                "summary_updated_at": "2024-01-15T10:30:00Z",
                "created_at": "2024-01-10T08:00:00Z",
            }
        }
    )

    id: str = Field(description="Unique identifier (alphanumeric lowercase)")
    name: str = Field(description="Display name for the mental model")
    description: str = Field(description="Prompt/directions for what to track and summarize")
    summary: str | None = Field(None, description="Generated summary based on relevant facts")
    summary_updated_at: str | None = Field(None, description="ISO format date when summary was last updated")
    created_at: str = Field(description="ISO format date when the mental model was created")


STRATEGY_TAGS_MATCH_VALUES = ("all", "exact")


class ConsolidationScopePattern(BaseModel):
    """One rule of a consolidation strategy: tags, and how they must match.

    ``tags`` may be empty — that is a rule still being filled in, which the editor
    saves as typed and consolidation ignores. The type pins the *shape*, not
    completeness: a string where the tag list belongs is rejected at the door
    instead of being stored and silently ignored for the life of the bank.

    Unknown keys are rejected too, but by :class:`StrictConsolidationStrategySpec`
    on the write path rather than by ``extra="forbid"`` here: that would put
    ``additionalProperties: false`` in the schema, which openapi-generator cannot
    process ("Codegen Property not yet supported in getPydanticType").
    """

    tags: list[str] = Field(default_factory=list, description="fnmatch tag patterns, e.g. company:*")
    # A plain nullable string, checked by the validator below, rather than an
    # enum: both client generators choke on a nullable enum here — progenitor
    # (Rust) fails on an inline one with TypeError(InvalidValue), and
    # openapi-generator (Python) fails on `anyOf[$ref, null]` with "Codegen
    # Property not yet supported in getPydanticType". A nullable string is the
    # shape the rest of this spec uses, and both handle it.
    #
    # Optional rather than defaulting to "all" so a value written by the control
    # plane survives export and import unchanged: a non-optional default would be
    # filled in on the way out, and the stored config would grow a
    # `"tags_match": "all"` the editor never wrote.
    tags_match: str | None = Field(
        default=None,
        description='"all" (the default when omitted): the scope has every tag in the rule, other tags '
        'allowed. "exact": exactly these tags and no others.',
    )

    @field_validator("tags_match")
    @classmethod
    def _known_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in STRATEGY_TAGS_MATCH_VALUES:
            raise ValueError(f"must be one of {', '.join(STRATEGY_TAGS_MATCH_VALUES)}")
        return value


class ConsolidationStrategySpec(BaseModel):
    """One `consolidation_strategies` entry: the rules it claims scopes with, and
    the observation settings those scopes use. Every setting is optional; unset
    ones come from the bank-wide values."""

    scopes: list[ConsolidationScopePattern] = Field(
        default_factory=list, description="Alternatives: the strategy claims a scope when any rule matches it"
    )
    observations_mission: str | None = None
    max_observations_per_scope: int | None = None
    consolidation_source_facts_max_tokens: int | None = None
    consolidation_source_facts_max_tokens_per_observation: int | None = None


class StrictConsolidationScopePattern(ConsolidationScopePattern):
    """Validation-only: rejects unknown keys. Not used by any route, so it never
    reaches the OpenAPI schema (see :class:`ConsolidationScopePattern`)."""

    model_config = ConfigDict(extra="forbid")


class StrictConsolidationStrategySpec(ConsolidationStrategySpec):
    """Validation-only twin of :class:`ConsolidationStrategySpec` that rejects
    unknown keys — the "scope" for "scopes" typo that would otherwise be stored
    and silently ignored for the life of the bank."""

    model_config = ConfigDict(extra="forbid")

    scopes: list[StrictConsolidationScopePattern] = Field(default_factory=list)


class StrategyScopePreview(BaseModel):
    """One existing observation scope in a consolidation-strategy preview."""

    tags: list[str] = Field(description="The scope's tags (sorted)")
    count: int = Field(description="Observations in this scope")
    handled_by: int | None = Field(
        description="Index of the strategy that actually applies to this scope (the first that claims it), "
        "or null when no strategy does and Default applies"
    )


class StrategyRulePreview(BaseModel):
    """What one rule (one entry of a strategy's `scopes`) matches among existing scopes."""

    match_count: int = Field(description="Existing scopes this rule matches")
    taken_count: int = Field(description="Of those, how many an earlier strategy wins, so this one has no effect")
    observation_count: int = Field(description="Observations across the matching scopes")
    samples: list[StrategyScopePreview] = Field(description="The most populous matching scopes, up to sample_limit")


class StrategyPreview(BaseModel):
    """Preview of one strategy, aligned by position with the request."""

    active: bool = Field(description="False when the server would ignore this strategy (no usable rule, or no setting)")
    claimed_count: int = Field(description="Existing scopes this strategy actually applies to")
    rules: list[StrategyRulePreview] = Field(description="One entry per rule, aligned with the request")


class DefaultScopesPreview(BaseModel):
    """The scopes no strategy claims — they consolidate under the bank-wide settings."""

    match_count: int
    observation_count: int
    samples: list[StrategyScopePreview]


class ConsolidationStrategiesPreview(BaseModel):
    """Which existing observation scopes each consolidation strategy would apply to."""

    strategies: list[StrategyPreview]
    default: DefaultScopesPreview
    scopes_scanned: int = Field(description="Distinct scopes the preview was computed over")
    complete: bool = Field(
        description="False when the bank has more distinct scopes than the preview scans; counts are then lower bounds"
    )


class ReflectDefaultOptions(BaseModel):
    """Reflect options an operator can default per bank.

    Every field is ``None`` = "not set", so the same model is both the shape of
    the ``reflect_default_options`` bank config key and the set of fields a
    reflect request inherits from it. The resolution chain is: explicit request
    value -> bank ``reflect_default_options`` -> the shipped default.

    A mental-model refresh does NOT read this. It is its own operation with its
    own per-bank default (``knowledge_page_default_trigger``), so the same-named
    fields are declared on ``MentalModelTrigger`` instead — see there.
    """

    reflect_search_observations_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token budget for reflect's search_observations tool when the model names none. "
            "Observation evidence is often the largest contributor to the reflect context; "
            "lowering it trades the lowest-ranked observations for a smaller LLM context. "
            "None means use the shipped default (5000)."
        ),
    )
    reflect_search_observations_include_entities: bool | None = Field(
        default=None,
        description=(
            "Whether search_observations attaches resolved entity names to each observation. "
            "Entities can be more than half the serialized tool payload; turning them off keeps "
            "the same observations and ranking with a much smaller context. None means enabled."
        ),
    )
    reflect_search_mental_models_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token budget for the mental models search_mental_models returns in full, packed in rank order. "
            "The best-ranked one is always in full; the others that do not fit come back as a snippet the "
            "model can read with read_mental_models. Raise it for banks with long pages or answers spread "
            "across pages; lower it to cut the reflect context. None means use the shipped default (16000)."
        ),
    )


class MentalModelTrigger(BaseModel):
    """Trigger settings for a mental model.

    A refresh is not an ad-hoc reflect with different arguments: it synthesizes a
    whole document, so it wants its own retrieval and iteration settings. This
    trigger is therefore the only source for them — a bank's
    ``reflect_default_options`` deliberately does not reach a refresh. The
    per-bank default for these fields is ``knowledge_page_default_trigger``,
    which is merged over this same shape when a page is created.
    """

    budget: Budget | None = Field(
        default=None,
        description=(
            "How many agent iterations a refresh may spend, as a multiple of "
            "reflect_max_iterations: 'low' halves it, 'mid' keeps it, 'high' doubles it. "
            "A refresh is the heaviest reflect there is — it writes a whole document, and with "
            "exclude_mental_models it must read raw facts first — so null means 'mid', not the "
            "'low' an ad-hoc reflect defaults to."
        ),
    )

    mode: Literal["full", "delta"] = Field(
        default="full",
        description=(
            "Refresh mode. 'full' (default) regenerates the mental model content from scratch on each refresh. "
            "'delta' performs surgical edits against the existing content: unchanged sections are preserved "
            "byte-for-byte, stale content is removed, new content is added. If the mental model has no existing "
            "content, or if the source_query has changed since the last refresh, delta mode falls back to a full "
            "regeneration automatically."
        ),
    )
    refresh_after_consolidation: bool = Field(
        default=False,
        description="If true, refresh this mental model after observations consolidation (real-time mode)",
    )
    refresh_cron: str | None = Field(
        default=None,
        description=(
            "Cron expression (UTC, standard 5-field syntax, e.g. '0 3 * * *' for daily at 03:00 UTC) "
            "for refreshing this mental model on a fixed schedule. Mutually exclusive with "
            "refresh_after_consolidation — a model refreshes either after consolidation or on a cron "
            "schedule, not both. A scheduled refresh only runs when the model is stale (new memories in "
            "its scope since the last refresh); if nothing changed, the tick is skipped to avoid a "
            "wasted LLM call. null = no schedule."
        ),
    )
    full_refresh_cron: str | None = Field(
        default=None,
        description=(
            "Cron expression (UTC, standard 5-field syntax, e.g. '0 3 * * 0' for Sundays at 03:00 UTC) "
            "for a scheduled full refresh: on this schedule the model is refreshed in 'full' mode "
            "whatever its `mode` says, regenerating the content from scratch to clear the drift many "
            "delta edits accumulate. Independent of `refresh_cron` and `refresh_after_consolidation` — "
            "a delta model can refresh after consolidation and still get a weekly full refresh — and it "
            "does not change `mode`. A tick is skipped when no refresh has run since the last full one "
            "(there is nothing to correct). Like `refresh_cron`, it honours "
            "`min_refresh_interval_seconds` and pauses while the last refresh failed. null = no schedule."
        ),
    )
    min_refresh_interval_seconds: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum seconds between two automatic refreshes of this mental model. A triggered "
            "refresh that arrives sooner is not dropped: it is queued and parked until the window "
            "expires, and every further trigger in the meantime folds into that one queued refresh, "
            "so a burst of retains costs one refresh instead of one per retain. Applies to every "
            "automatic trigger: refresh_after_consolidation, refresh_cron and full_refresh_cron. Explicit refreshes (API, MCP, control "
            "plane) ignore it and run immediately. 0 disables the floor for this model; null falls "
            "back to the bank/global mental_model_min_refresh_interval_seconds setting (itself 0 by "
            "default, i.e. no floor)."
        ),
    )
    fact_types: list[Literal["world", "experience", "observation"]] | None = Field(
        default=None,
        description="Filter which fact types are retrieved during reflect. None means all types (world, experience, observation).",
    )
    exclude_mental_models: bool = Field(
        default=False,
        description="If true, exclude all mental models from the reflect loop (skip search_mental_models tool).",
    )
    exclude_mental_model_ids: list[str] | None = Field(
        default=None,
        description="Exclude specific mental models by ID from the reflect loop.",
    )
    tags_match: TagsMatch | None = Field(
        default=None,
        description=(
            "Override how the model's tags filter memories during refresh. "
            "If not set, defaults to 'all_strict' when the model has tags (security isolation) "
            "or 'any' when the model has no tags. Under 'all_strict' a memory must carry EVERY "
            "one of the model's tags and untagged memories are excluded, which is why a model "
            "tagged with labels its memories do not carry refreshes to empty content. "
            "Set to 'all' to keep requiring the tags while including untagged memories, or to "
            "'any' to include untagged memories alongside any single tag match. "
            "Staleness ignores that widening: an untagged write never marks a tagged model stale, "
            "in any mode — only a write that matches the model's tags does."
        ),
    )
    tag_groups: list[TagGroup] | None = Field(
        default=None,
        description=(
            "Compound boolean tag expressions to use during refresh instead of the model's own tags. "
            "When set, these tag groups are passed to reflect and the model's flat tags are NOT used for filtering. "
            "Supports nested and/or/not expressions for complex tag-based scoping."
        ),
    )
    include_chunks: bool | None = Field(
        default=None,
        description=(
            "Override whether the internal recall used during refresh returns raw chunk text. "
            "None means use the bank/global config default (recall_include_chunks)."
        ),
    )
    recall_max_tokens: int | None = Field(
        default=None,
        description=(
            "Override the token budget for facts returned by the internal recall during refresh. "
            "None means use the bank/global config default (recall_max_tokens)."
        ),
    )
    recall_chunks_max_tokens: int | None = Field(
        default=None,
        description=(
            "Override the token budget for raw chunks returned by the internal recall during refresh. "
            "None means use the bank/global config default (recall_chunks_max_tokens)."
        ),
    )
    reflect_search_observations_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Override the token budget for the refresh's search_observations calls. Observation "
            "evidence is often the largest contributor to the reflect context; lowering it trades "
            "the lowest-ranked observations for a smaller LLM context. null means the shipped 5000."
        ),
    )
    reflect_search_observations_include_entities: bool | None = Field(
        default=None,
        description=(
            "Override whether search_observations attaches resolved entity names to each "
            "observation. Entities can be more than half the serialized tool payload; turning them "
            "off keeps the same observations and ranking with a much smaller context. null means enabled."
        ),
    )
    response_schema: dict | None = Field(
        default=None,
        description=(
            "Optional JSON Schema for structured output. When set, each refresh runs the same "
            "structured-output extraction as reflect's response_schema and stores the parsed result "
            "under reflect_response.structured_output alongside the markdown content."
        ),
    )
    keep_trace: bool = Field(
        default=False,
        description=(
            "If true, every refresh of this mental model records how it reached its result under "
            "reflect_response.trace: the mode it ran in and why, the resolved scope and time window, "
            "how many facts retrieval returned versus how many the agent used, the tool and LLM calls, "
            "and any delta operations. Only the latest refresh's trace is kept. This is the only way to "
            "diagnose a cron- or consolidation-driven refresh after the fact, since no human sees those "
            "run. Tool outputs are reduced to result counts to keep the stored trace bounded; use "
            "LLM request tracing for raw prompts and responses."
        ),
    )

    @field_validator("fact_types")
    @classmethod
    def validate_fact_types(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("fact_types must not be empty. Use null to include all fact types.")
        return v

    @field_validator("tag_groups")
    @classmethod
    def validate_tag_groups_are_exact(cls, v: "list[TagGroup] | None") -> "list[TagGroup] | None":
        # A trigger's scope is read by two paths that resolve differently: the refresh runs
        # through reflect, which resolves fuzzy leaves, while the staleness check and the
        # scope watermark build SQL straight from the stored groups and do not. A stored
        # fuzzy leaf would therefore build content from the resolved tags while never being
        # marked stale by them, and would drift as the bank's tag vocabulary changes.
        # Resolution is a request-time step; stored scopes stay exact.
        if v is not None and needs_resolution(v):
            raise ValueError(
                "resolve='fuzzy' is not supported in a trigger's tag_groups. A stored scope must "
                "be exact; use it on a recall or reflect request instead."
            )
        return v

    @field_validator("response_schema")
    @classmethod
    def validate_trigger_response_schema(cls, v: dict | None) -> dict | None:
        if v is not None:
            validate_response_schema(v)
        return v

    @field_validator("refresh_cron", "full_refresh_cron")
    @classmethod
    def validate_refresh_cron(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        from croniter import croniter

        if not croniter.is_valid(v):
            raise ValueError(f"{info.field_name} is not a valid cron expression: {v!r}")
        return v

    @model_validator(mode="after")
    def validate_refresh_exclusivity(self) -> "MentalModelTrigger":
        # A mental model refreshes either after consolidation (real-time) or on a
        # cron schedule, never both — the two triggers would race and double-refresh.
        if self.refresh_after_consolidation and self.refresh_cron:
            raise ValueError(
                "refresh_after_consolidation and refresh_cron are mutually exclusive: "
                "a mental model refreshes either after consolidation or on a cron schedule, not both."
            )
        return self
