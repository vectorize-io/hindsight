"""
Type definitions for the retain pipeline.

These dataclasses provide type safety throughout the retain operation,
from content input to fact storage.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypedDict
from uuid import UUID


class RetainContentDict(TypedDict, total=False):
    """Type definition for content items in retain_batch_async.

    Fields:
        content: Text content to store (required)
        context: Context about the content (optional)
        event_date: When the content occurred (optional, defaults to now)
        metadata: Custom key-value metadata (optional)
        document_id: Document ID for this content item (optional)
        entities: User-provided entities to merge with extracted entities (optional)
    """

    content: str  # Required
    context: str
    event_date: datetime
    metadata: dict[str, str]
    document_id: str
    entities: list[dict[str, str]]  # [{"text": "...", "type": "..."}]


def _now_utc() -> datetime:
    """Factory function for default event_date."""
    return datetime.now(UTC)


@dataclass
class RetainContent:
    """
    Input content item to be retained as memories.

    Represents a single piece of content to extract facts from.
    """

    content: str
    context: str = ""
    event_date: datetime = field(default_factory=_now_utc)
    metadata: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, str]] = field(default_factory=list)  # User-provided entities


@dataclass
class ChunkMetadata:
    """
    Metadata about a text chunk.

    Used to track which facts were extracted from which chunks.
    """

    chunk_text: str
    fact_count: int
    content_index: int  # Index of the source content
    chunk_index: int  # Global chunk index across all contents


@dataclass
class EntityRef:
    """
    Reference to an entity mentioned in a fact.

    Entities are extracted by the LLM during fact extraction.
    """

    name: str
    canonical_name: str | None = None  # Resolved canonical name
    entity_id: UUID | None = None  # Resolved entity ID


@dataclass
class CausalRelation:
    """
    Causal relationship between facts.

    Represents how one fact causes, enables, or prevents another.
    """

    relation_type: str  # "causes", "enables", "prevents", "caused_by"
    target_fact_index: int  # Index of the target fact in the batch
    strength: float = 1.0  # Strength of the causal relationship


@dataclass
class ExtractedFact:
    """
    Fact extracted from content by the LLM.

    This is the raw output from fact extraction before processing.
    """

    fact_text: str
    fact_type: str  # "world", "experience", "opinion", "observation"
    entities: list[str] = field(default_factory=list)
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    where: str | None = None  # WHERE the fact occurred or is about
    causal_relations: list[CausalRelation] = field(default_factory=list)

    # Context from the content item
    content_index: int = 0  # Which content this fact came from
    chunk_index: int = 0  # Which chunk this fact came from
    context: str = ""
    mentioned_at: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessedFact:
    """
    Fact after processing and ready for storage.

    Includes resolved entities, embeddings, and all necessary fields.
    """

    # Core fact data
    fact_text: str
    fact_type: str
    embedding: list[float]

    # Temporal data
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime

    # Context and metadata
    context: str
    metadata: dict[str, str]

    # Location data
    where: str | None = None

    # Entities
    entities: list[EntityRef] = field(default_factory=list)

    # Causal relations
    causal_relations: list[CausalRelation] = field(default_factory=list)

    # Chunk reference
    chunk_id: str | None = None

    # Document reference (denormalized for query performance)
    document_id: str | None = None

    # DB fields (set after insertion)
    unit_id: UUID | None = None

    # Track which content this fact came from (for user entity merging)
    content_index: int = 0

    @property
    def is_duplicate(self) -> bool:
        """Check if this fact was marked as a duplicate."""
        return self.unit_id is None

    @staticmethod
    def from_extracted_fact(
        extracted_fact: "ExtractedFact", embedding: list[float], chunk_id: str | None = None
    ) -> "ProcessedFact":
        """
        Create ProcessedFact from ExtractedFact.

        Args:
            extracted_fact: Source ExtractedFact
            embedding: Generated embedding vector
            chunk_id: Optional chunk ID

        Returns:
            ProcessedFact ready for storage
        """
        from datetime import datetime

        # Use occurred dates only if explicitly provided by LLM
        occurred_start = extracted_fact.occurred_start
        occurred_end = extracted_fact.occurred_end
        mentioned_at = extracted_fact.mentioned_at or datetime.now(UTC)

        # Convert entity strings to EntityRef objects
        entities = [EntityRef(name=name) for name in extracted_fact.entities]

        return ProcessedFact(
            fact_text=extracted_fact.fact_text,
            fact_type=extracted_fact.fact_type,
            embedding=embedding,
            occurred_start=occurred_start,
            occurred_end=occurred_end,
            mentioned_at=mentioned_at,
            context=extracted_fact.context,
            metadata=extracted_fact.metadata,
            entities=entities,
            causal_relations=extracted_fact.causal_relations,
            chunk_id=chunk_id,
            content_index=extracted_fact.content_index,
        )


@dataclass
class EntityLink:
    """
    Link between two memory units through a shared entity.

    Used for entity-based graph connections in the memory graph.
    """

    from_unit_id: UUID
    to_unit_id: UUID
    entity_id: UUID
    link_type: str = "entity"
    weight: float = 1.0


@dataclass
class RetainBatch:
    """
    A batch of content to retain.

    Tracks all facts, chunks, and metadata for a batch operation.
    """

    bank_id: str
    contents: list[RetainContent]
    document_id: str | None = None
    fact_type_override: str | None = None
    confidence_score: float | None = None

    # Extracted data (populated during processing)
    extracted_facts: list[ExtractedFact] = field(default_factory=list)
    processed_facts: list[ProcessedFact] = field(default_factory=list)
    chunks: list[ChunkMetadata] = field(default_factory=list)

    # Results (populated after storage)
    unit_ids_by_content: list[list[str]] = field(default_factory=list)

    def get_facts_for_content(self, content_index: int) -> list[ExtractedFact]:
        """Get all extracted facts for a specific content item."""
        return [f for f in self.extracted_facts if f.content_index == content_index]

    def get_chunks_for_content(self, content_index: int) -> list[ChunkMetadata]:
        """Get all chunks for a specific content item."""
        return [c for c in self.chunks if c.content_index == content_index]
