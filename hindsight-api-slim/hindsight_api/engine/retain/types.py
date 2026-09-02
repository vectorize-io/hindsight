"""
Type definitions for the retain pipeline.

These dataclasses provide type safety throughout the retain operation,
from content input to fact storage.
"""

import logging
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypedDict
from uuid import UUID

import numpy as np
import orjson

from ..metadata_utils import drop_null_values

logger = logging.getLogger(__name__)

# An embedding, packed. ``array("f")`` stores the vector as a contiguous block of C floats
# instead of one boxed ``PyFloat`` per dimension: a 384-dim vector measures 1,616 bytes
# packed against 12,344 bytes as ``list[float]``, a 7.6x difference that scales with the
# number of facts a retain batch holds at once (#3756).
#
# "f" (float32) is the width pgvector's ``vector`` column stores anyway, so nothing is lost
# on the way to the database — the rounding that used to happen at the INSERT now happens
# one step earlier, and the stored bytes are the same.
# Unparameterised on purpose: `array.array` is only generic in the type stubs, so
# `array[float]` raises TypeError at import time on CPython.
PackedEmbedding = array


def pack_embedding(values: Sequence[float]) -> PackedEmbedding:
    """Pack an embedding vector for carrying through the retain pipeline."""
    return array("f", values)


# What a function that merely carries or renders an embedding should accept. Retain holds the
# packed form; imports and re-embeds hold a plain float list; a few paths hold a rendered
# pgvector literal already. All three reach the same link/ANN helpers.
EmbeddingLike = PackedEmbedding | Sequence[float] | str


def _repr_literal(values: Iterable[object]) -> str:
    """The pre-orjson formatting, kept as the fallback for non-finite and non-float inputs."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"  # type: ignore[arg-type]


def _dumps_or_repr_fallback(payload: np.ndarray | list | tuple) -> str:
    """Serialize with orjson; fall back to repr() when the vector is non-finite.

    Finite-float JSON output consists only of [0-9.eE+-,[]], so the substring
    "null" can only mean orjson encoded a NaN/Inf element as JSON null.
    """
    try:
        rendered = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY).decode("ascii")
    except (orjson.JSONEncodeError, TypeError):
        return _repr_literal(payload)
    return _repr_literal(payload) if "null" in rendered else rendered


def embedding_to_pgvector(embedding: EmbeddingLike) -> str:
    """Render an embedding as the ``'[0.1,0.2,...]'`` literal asyncpg binds to ``vector``.

    Handles every form a caller may hold: the packed array retain carries, a plain float
    list (imports, re-embeds), or a literal that was already rendered. Uses orjson's SIMD
    float serializer for fast formatting without per-element Python allocations.

    Note: The formatted string uses float32 shortest representation when serialized via
    the PackedEmbedding/NumPy fast-path (e.g. ``0.1`` vs legacy ``0.10000000149011612``).
    When parsed back by PostgreSQL as a 32-bit float vector, the stored bytes are bit-identical.
    """
    if isinstance(embedding, str):
        return embedding
    if isinstance(embedding, array) and embedding.typecode == "f":
        return _dumps_or_repr_fallback(np.frombuffer(embedding, dtype=np.float32))
    if isinstance(embedding, (np.ndarray, list, tuple)):
        return _dumps_or_repr_fallback(embedding)
    return _repr_literal(embedding)


class RetainContentDict(TypedDict, total=False):
    """Type definition for content items in retain_batch_async.

    Fields:
        content: Text content to store (required)
        context: Context about the content (optional)
        event_date: When the content occurred (optional, defaults to now)
        metadata: Custom key-value metadata (optional)
        document_id: Document ID for this content item (optional)
        entities: User-provided entities to merge with extracted entities (optional)
        resolve_entities: Whether the supplied `entities` are resolved against the bank's
            existing entities (optional, default True). False takes them literally.
        tags: Visibility scope tags for this content item (optional)
        observation_scopes: How to scope observations for consolidation (optional).
            "per_tag" runs one pass per individual tag; "combined" (default) runs a
            single pass with all tags; "shared" runs a single pass over one global,
            untagged scope so memories consolidate together regardless of tags;
            a list[list[str]] specifies exact passes.
        update_mode: How to handle existing documents with the same document_id (optional).
            "replace" (default) deletes old data and reprocesses. "append" concatenates
            new content to the existing document and reprocesses.
        force_reextract: Re-run extraction even when the content is byte-identical to what
            is already stored (optional, default False). Internal — set by
            ``reprocess_document``, not accepted on the public retain API. See
            ``retain_batch`` for the two skips it suppresses.
    """

    content: str  # Required
    context: str
    event_date: datetime | None
    metadata: dict[str, str]
    document_id: str
    entities: list[dict[str, str]]  # [{"text": "...", "type": "..."}]
    resolve_entities: bool
    tags: list[str]  # Visibility scope tags
    observation_scopes: (
        Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]]
    )  # Observation scopes for consolidation
    update_mode: Literal["replace", "append"]
    force_reextract: bool


@dataclass
class UserEntities:
    """The entities a caller supplied for one retain content item, and how to match them.

    Kept together so the resolution choice travels with the names it applies to: retain merges
    these with the extractor's own entities into one batch, and only these are authoritative.
    """

    entities: list[dict[str, str]]
    resolve: bool = True


@dataclass
class RetainContent:
    """
    Input content item to be retained as memories.

    Represents a single piece of content to extract facts from.
    """

    content: str
    context: str = ""
    event_date: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, str]] = field(default_factory=list)  # User-provided entities
    # Whether the supplied `entities` are matched against the bank's existing entities. False
    # takes them literally; extracted entities are always resolved either way (#3479).
    resolve_entities: bool = True
    tags: list[str] = field(default_factory=list)  # Visibility scope tags
    observation_scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None = (
        None  # Observation scopes
    )

    def __post_init__(self) -> None:
        # Drop null-valued metadata keys (issue #3209): the retain API accepts
        # arbitrary JSON metadata, and a null value stored verbatim poisons the
        # read path, which validates MemoryFact.metadata as dict[str, str].
        # Non-string values are preserved; the read path coerces them. An
        # explicit ``"metadata": null`` in the request normalizes to {} so the
        # field always matches its declared type.
        self.metadata = drop_null_values(self.metadata)


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

    Retain emits only the backward-looking ``caused_by`` form. Transfer import
    reuses this structure to restore historical causal types without allowing
    normal retain writes to create them.
    """

    relation_type: str  # ``caused_by`` for retain; legacy types for transfer restore
    target_fact_index: int  # Index of the target fact in the batch


@dataclass
class ExtractedFact:
    """
    Fact extracted from content by the LLM.

    This is the raw output from fact extraction before processing.
    """

    fact_text: str
    fact_type: str  # "world", "experience", "observation"
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
    tags: list[str] = field(default_factory=list)  # Visibility scope tags
    observation_scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None = (
        None  # Observation scopes
    )


@dataclass
class ProcessedFact:
    """
    Fact after processing and ready for storage.

    Includes resolved entities, embeddings, and all necessary fields.
    """

    # Core fact data
    fact_text: str
    fact_type: str
    # Packed, not ``list[float]`` — see ``PackedEmbedding``. Render it for SQL with
    # ``embedding_to_pgvector``; ``list(...)`` recovers the plain float list.
    embedding: PackedEmbedding

    # Temporal data
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime | None

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

    # Visibility scope tags
    tags: list[str] = field(default_factory=list)

    # Observation scopes for consolidation
    observation_scopes: Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None = None

    @staticmethod
    def _is_degenerate_text(text: str) -> bool:
        """Check if fact text has zero information content.

        Rejects empty strings, whitespace-only, single punctuation marks,
        and common LLM hallucination patterns that carry no semantic meaning.
        """
        stripped = (text or "").strip()
        if not stripped:
            return True
        # Single or repeated punctuation patterns with no semantic content
        degenerate_patterns = {
            "...",
            "…",
            "-",
            "--",
            "---",
            ".",
            "..",
            "•",
            "·",
            "*",
            "**",
            "***",
            "_,_",
            "_, _, _",
        }
        if stripped in degenerate_patterns:
            return True
        # Strings composed entirely of punctuation and whitespace
        if all(c in ".,;:!?-–—…\"'`´ \t\n\r" for c in stripped):
            return True
        # Very short text (<= 2 chars) that is only punctuation
        if len(stripped) <= 2 and all(not c.isalnum() for c in stripped):
            return True
        return False

    @staticmethod
    def from_extracted_fact(
        extracted_fact: "ExtractedFact", embedding: Sequence[float], chunk_id: str | None = None
    ) -> "ProcessedFact | None":
        """
        Create ProcessedFact from ExtractedFact.

        Args:
            extracted_fact: Source ExtractedFact
            embedding: Generated embedding vector
            chunk_id: Optional chunk ID

        Returns:
            ProcessedFact ready for storage, or None if the fact text is degenerate
            (zero information content — punctuation-only, empty, etc.)
        """
        fact_text = extracted_fact.fact_text or ""
        if ProcessedFact._is_degenerate_text(fact_text):
            logger.warning(
                f"Rejected degenerate fact text: type={extracted_fact.fact_type}, "
                f"text={fact_text[:80]!r}, entities={extracted_fact.entities}"
            )
            return None

        # Use occurred dates only if explicitly provided by LLM
        occurred_start = extracted_fact.occurred_start
        occurred_end = extracted_fact.occurred_end
        mentioned_at = extracted_fact.mentioned_at  # May be None when caller opted into no timestamp

        # Convert entity strings to EntityRef objects
        entities = [EntityRef(name=name) for name in extracted_fact.entities]

        return ProcessedFact(
            fact_text=fact_text,
            fact_type=extracted_fact.fact_type,
            embedding=pack_embedding(embedding),
            occurred_start=occurred_start,
            occurred_end=occurred_end,
            mentioned_at=mentioned_at,
            context=extracted_fact.context,
            metadata=extracted_fact.metadata,
            entities=entities,
            causal_relations=extracted_fact.causal_relations,
            chunk_id=chunk_id,
            content_index=extracted_fact.content_index,
            tags=extracted_fact.tags,
            observation_scopes=extracted_fact.observation_scopes,
        )


@dataclass
class ResolvedEntity:
    """Identity of a resolved entity carried across the retain phase boundary.

    ``canonical_name`` is the value stored on the entity row (NOT the raw input
    mention), captured during Phase-1 resolution. It is threaded to Phase 2 so a
    parent pruned between phases can be re-created with its real name — the row
    is gone by then, so the name is otherwise unrecoverable (#2662).

    ``entity_kind`` mirrors the ``entities.entity_kind`` column ("regular" or
    "label"). It is carried for the same reason as ``canonical_name``: a pruned
    label parent re-created by the Phase-2 reassert must keep its kind, or it
    would re-enter the partial trigram index that label rows are excluded
    from (#3208).
    """

    entity_id: str
    canonical_name: str
    entity_kind: str = "regular"

    def __post_init__(self) -> None:
        # Callers pass UUID objects or strings; normalize once so downstream
        # comparisons, set membership, and SQL binds all see a plain str.
        self.entity_id = str(self.entity_id)


@dataclass
class EntityResolutionResult:
    """
    Result of Phase 1 entity resolution.

    Contains resolved entity identities and the mapping data needed to remap
    placeholder unit IDs to real IDs after fact insertion in Phase 2.
    """

    resolved_entities: list[ResolvedEntity]
    entity_to_unit: list[tuple]
    unit_to_entity_ids: dict[str, list[str]]

    @property
    def resolved_entity_ids(self) -> list[str]:
        """Entity IDs in flattened resolution order (used by link remapping)."""
        return [entity.entity_id for entity in self.resolved_entities]


@dataclass
class Phase1Result:
    """
    Full result of Phase 1 (entity resolution + optional semantic ANN).
    """

    entities: EntityResolutionResult
    semantic_ann_links: list[tuple]


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
    document_tags: list[str] = field(default_factory=list)  # Tags applied to all items

    # Extracted data (populated during processing)
    extracted_facts: list[ExtractedFact] = field(default_factory=list)
    processed_facts: list[ProcessedFact] = field(default_factory=list)
    chunks: list[ChunkMetadata] = field(default_factory=list)

    # Results (populated after storage)
    unit_ids_by_content: list[list[str]] = field(default_factory=list)


class ConcurrentAppendConflict(Exception):
    """An append-mode retain lost its read-modify-write race for a document.

    ``update_mode="append"`` reads ``documents.original_text``, concatenates the
    new content onto it, and reprocesses the result. That read and the write
    that follows it are separated by LLM extraction, so a second append
    committing in between would make this request overwrite a turn it never
    saw. Every write path that can observe the document having moved raises
    this instead of dropping the submission, so a lost race costs a retry
    rather than the caller's content.

    Retryable by construction: the retry re-reads the (now newer) stored text
    and re-appends the same submission on top of it.
    """
