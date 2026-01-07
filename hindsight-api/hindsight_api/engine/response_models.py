"""
Core response models for Hindsight memory system.

These models define the structure of data returned by the core MemoryEngine class.
API response models should be kept separate and convert from these core models to maintain
API stability even if internal models change.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Valid fact types for recall operations (excludes 'observation' which is internal)
VALID_RECALL_FACT_TYPES = frozenset(["world", "experience", "opinion"])


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
    output_tokens: int = Field(default=0, description="Number of output/completion tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Allow aggregating token usage from multiple calls."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
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
            }
        }
    )

    id: str = Field(description="Unique identifier for the memory fact")
    text: str = Field(description="The actual text content of the memory")
    fact_type: str = Field(description="Type of fact: 'world', 'experience', 'opinion', or 'observation'")
    entities: list[str] | None = Field(None, description="Entity names mentioned in this fact")
    context: str | None = Field(None, description="Additional context for the memory")
    occurred_start: str | None = Field(None, description="ISO format date when the event started occurring")
    occurred_end: str | None = Field(None, description="ISO format date when the event ended occurring")
    mentioned_at: str | None = Field(None, description="ISO format date when the fact was mentioned/learned")
    document_id: str | None = Field(None, description="ID of the document this memory belongs to")
    metadata: dict[str, str] | None = Field(None, description="User-defined metadata")
    chunk_id: str | None = Field(
        None, description="ID of the chunk this fact was extracted from (format: bank_id_document_id_chunk_index)"
    )


class ChunkInfo(BaseModel):
    """Information about a chunk."""

    chunk_text: str = Field(description="The raw chunk text")
    chunk_index: int = Field(description="Index of the chunk within the document")
    truncated: bool = Field(default=False, description="Whether the chunk was truncated due to token limits")


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
    entities: dict[str, "EntityState"] | None = Field(
        None, description="Entity states for entities mentioned in results (keyed by canonical name)"
    )
    chunks: dict[str, ChunkInfo] | None = Field(
        None, description="Chunks for facts, keyed by '{document_id}_{chunk_index}'"
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
                },
                "new_opinions": ["Machine learning has great potential in healthcare"],
                "structured_output": {"summary": "ML in healthcare", "confidence": 0.9},
                "usage": {"input_tokens": 1500, "output_tokens": 500, "total_tokens": 2000},
            }
        }
    )

    text: str = Field(description="The formulated answer text")
    based_on: dict[str, list[MemoryFact]] = Field(
        description="Facts used to formulate the answer, organized by type (world, experience, opinion)"
    )
    new_opinions: list[str] = Field(default_factory=list, description="List of newly formed opinions during reflection")
    structured_output: dict[str, Any] | None = Field(
        default=None,
        description="Structured output parsed according to the provided response schema. Only present when response_schema was provided.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage metrics for the LLM calls made during this reflect operation.",
    )


class Opinion(BaseModel):
    """
    An opinion with confidence score.

    Opinions represent the bank's formed perspectives on topics,
    with a confidence level indicating strength of belief.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"text": "Machine learning has great potential in healthcare", "confidence": 0.85}
        }
    )

    text: str = Field(description="The opinion text")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


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
