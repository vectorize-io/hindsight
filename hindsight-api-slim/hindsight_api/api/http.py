"""
FastAPI application factory and API routes for memory system.

This module provides the create_app function to create and configure
the FastAPI application with all API endpoints.
"""

import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile

from hindsight_api.engine.audit import AuditEntry, AuditLogger
from hindsight_api.extensions import AuthenticationError


def _parse_metadata(metadata: Any) -> dict[str, Any]:
    """Parse metadata that may be a dict, JSON string, or None."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    return {}


from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hindsight_api import MemoryEngine


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


from hindsight_api.config import get_config
from hindsight_api.engine.memory_engine import Budget, _current_schema, _get_tiktoken_encoding, fq_table
from hindsight_api.engine.providers.none_llm import LLMNotAvailableError
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES, MemoryFact, TokenUsage
from hindsight_api.engine.search.tags import TagGroup, TagsMatch
from hindsight_api.extensions import HttpExtension, OperationValidationError, load_extension
from hindsight_api.metrics import create_metrics_collector, get_metrics_collector, initialize_metrics
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)


class EntityIncludeOptions(BaseModel):
    """Options for including entity observations in recall results."""

    max_tokens: int = Field(default=500, description="Maximum tokens for entity observations")


class ChunkIncludeOptions(BaseModel):
    """Options for including chunks in recall results."""

    max_tokens: int = Field(default=8192, description="Maximum tokens for chunks (chunks may be truncated)")


class SourceFactsIncludeOptions(BaseModel):
    """Options for including source facts for observation-type results."""

    max_tokens: int = Field(
        default=4096, description="Maximum total tokens for source facts across all observations (-1 = unlimited)"
    )
    max_tokens_per_observation: int = Field(
        default=-1, description="Maximum tokens of source facts per observation (-1 = unlimited)"
    )


class IncludeOptions(BaseModel):
    """Options for including additional data in recall results."""

    entities: EntityIncludeOptions | None = Field(
        default=EntityIncludeOptions(),
        description="Include entity observations. Set to null to disable entity inclusion.",
    )
    chunks: ChunkIncludeOptions | None = Field(
        default=None, description="Include raw chunks. Set to {} to enable, null to disable (default: disabled)."
    )
    source_facts: SourceFactsIncludeOptions | None = Field(
        default=None,
        description="Include source facts for observation-type results. Set to {} to enable, null to disable (default: disabled).",
    )


class RecallRequest(BaseModel):
    """Request model for recall endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What did Alice say about machine learning?",
                "types": ["world", "experience"],
                "budget": "mid",
                "max_tokens": 4096,
                "trace": True,
                "query_timestamp": "2023-05-30T23:40:00",
                "include": {"entities": {"max_tokens": 500}},
                "tags": ["user_a"],
                "tags_match": "any",
            }
        }
    )

    query: str
    types: list[str] | None = Field(
        default=None,
        description="List of fact types to recall: 'world', 'experience', 'observation'. Defaults to world and experience if not specified.",
    )
    budget: Budget = Budget.MID
    max_tokens: int = 4096
    trace: bool = False
    query_timestamp: str | None = Field(
        default=None, description="ISO format date string (e.g., '2023-05-30T23:40:00')"
    )
    include: IncludeOptions = FieldWithDefault(
        IncludeOptions,
        description="Options for including additional data (entities are included by default)",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Filter memories by tags. If not specified, all memories are returned.",
    )
    tags_match: TagsMatch = Field(
        default="any",
        description="How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), "
        "'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged).",
    )
    tag_groups: list[TagGroup] | None = Field(
        default=None,
        description="Compound tag filter using boolean groups. Groups in the list are AND-ed. "
        "Each group is a leaf {tags, match} or compound {and: [...]}, {or: [...]}, {not: ...}.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        from ..engine.search.retrieval import tokenize_query

        if not tokenize_query(v):
            raise ValueError("query must contain at least one word character after normalization")
        return v

    @model_validator(mode="after")
    def validate_tags_exclusive(self) -> "RecallRequest":
        if self.tags is not None and self.tag_groups is not None:
            raise ValueError("'tags' and 'tag_groups' are mutually exclusive. Use 'tag_groups' for compound filtering.")
        return self


class RecallResult(BaseModel):
    """Single recall result item."""

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "Alice works at Google on the AI team",
                "type": "world",
                "entities": ["Alice", "Google"],
                "context": "work info",
                "occurred_start": "2024-01-15T10:30:00Z",
                "occurred_end": "2024-01-15T10:30:00Z",
                "mentioned_at": "2024-01-15T10:30:00Z",
                "document_id": "session_abc123",
                "metadata": {"source": "slack"},
                "chunk_id": "456e7890-e12b-34d5-a678-901234567890",
                "tags": ["user_a", "user_b"],
            }
        },
    }

    id: str
    text: str
    type: str | None = None  # fact type: world, experience, opinion, observation
    entities: list[str] | None = None  # Entity names mentioned in this fact
    context: str | None = None
    occurred_start: str | None = None  # ISO format date when the event started
    occurred_end: str | None = None  # ISO format date when the event ended
    mentioned_at: str | None = None  # ISO format date when the fact was mentioned
    document_id: str | None = None  # Document this memory belongs to
    metadata: dict[str, str] | None = None  # User-defined metadata
    chunk_id: str | None = None  # Chunk this fact was extracted from
    tags: list[str] | None = None  # Visibility scope tags
    source_fact_ids: list[str] | None = (
        None  # IDs of source facts (observation type only, when source_facts is enabled)
    )


class EntityObservationResponse(BaseModel):
    """An observation about an entity."""

    text: str
    mentioned_at: str | None = None


class EntityStateResponse(BaseModel):
    """Current mental model of an entity."""

    entity_id: str
    canonical_name: str
    observations: list[EntityObservationResponse]


class EntityListItem(BaseModel):
    """Entity list item with summary."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "canonical_name": "John",
                "mention_count": 15,
                "first_seen": "2024-01-15T10:30:00Z",
                "last_seen": "2024-02-01T14:00:00Z",
            }
        }
    )

    id: str
    canonical_name: str
    mention_count: int
    first_seen: str | None = None
    last_seen: str | None = None
    metadata: dict[str, Any] | None = None


class EntityListResponse(BaseModel):
    """Response model for entity list endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "canonical_name": "John",
                        "mention_count": 15,
                        "first_seen": "2024-01-15T10:30:00Z",
                        "last_seen": "2024-02-01T14:00:00Z",
                    }
                ],
                "total": 150,
                "limit": 100,
                "offset": 0,
            }
        }
    )

    items: list[EntityListItem]
    total: int
    limit: int
    offset: int


class EntityGraphResponse(BaseModel):
    """Response model for entity co-occurrence graph endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes": [
                    {"data": {"id": "uuid-1", "label": "Alice", "mentionCount": 12, "color": "#42a5f5"}},
                    {"data": {"id": "uuid-2", "label": "Google", "mentionCount": 8, "color": "#42a5f5"}},
                ],
                "edges": [
                    {
                        "data": {
                            "id": "uuid-1-uuid-2",
                            "source": "uuid-1",
                            "target": "uuid-2",
                            "linkType": "cooccurrence",
                            "weight": 5,
                            "color": "#ffd700",
                            "lineStyle": "solid",
                            "lastCooccurred": "2024-02-01T14:00:00Z",
                        }
                    }
                ],
                "total_entities": 2,
                "total_edges": 1,
                "limit": 1000,
            }
        }
    )

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_entities: int
    total_edges: int
    limit: int


class EntityDetailResponse(BaseModel):
    """Response model for entity detail endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "canonical_name": "John",
                "mention_count": 15,
                "first_seen": "2024-01-15T10:30:00Z",
                "last_seen": "2024-02-01T14:00:00Z",
                "observations": [{"text": "John works at Google", "mentioned_at": "2024-01-15T10:30:00Z"}],
            }
        }
    )

    id: str
    canonical_name: str
    mention_count: int
    first_seen: str | None = None
    last_seen: str | None = None
    metadata: dict[str, Any] | None = None
    observations: list[EntityObservationResponse]


class ChunkData(BaseModel):
    """Chunk data for a single chunk."""

    id: str
    text: str
    chunk_index: int
    truncated: bool = Field(default=False, description="Whether the chunk text was truncated due to token limits")


class RecallResponse(BaseModel):
    """Response model for recall endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "text": "Alice works at Google on the AI team",
                        "type": "world",
                        "entities": ["Alice", "Google"],
                        "context": "work info",
                        "occurred_start": "2024-01-15T10:30:00Z",
                        "occurred_end": "2024-01-15T10:30:00Z",
                        "chunk_id": "456e7890-e12b-34d5-a678-901234567890",
                    }
                ],
                "trace": {
                    "query": "What did Alice say about machine learning?",
                    "num_results": 1,
                    "time_seconds": 0.123,
                },
                "entities": {
                    "Alice": {
                        "entity_id": "123e4567-e89b-12d3-a456-426614174001",
                        "canonical_name": "Alice",
                        "observations": [
                            {"text": "Alice works at Google on the AI team", "mentioned_at": "2024-01-15T10:30:00Z"}
                        ],
                    }
                },
                "chunks": {
                    "456e7890-e12b-34d5-a678-901234567890": {
                        "id": "456e7890-e12b-34d5-a678-901234567890",
                        "text": "Alice works at Google on the AI team. She's been there for 3 years...",
                        "chunk_index": 0,
                    }
                },
            }
        }
    )

    results: list[RecallResult]
    trace: dict[str, Any] | None = None
    entities: dict[str, EntityStateResponse] | None = Field(
        default=None, description="Entity states for entities mentioned in results"
    )
    chunks: dict[str, ChunkData] | None = Field(default=None, description="Chunks for facts, keyed by chunk_id")
    source_facts: dict[str, RecallResult] | None = Field(
        default=None, description="Source facts for observation-type results, keyed by fact ID"
    )


class EntityInput(BaseModel):
    """Entity to associate with retained content."""

    text: str = Field(description="The entity name/text")
    type: str | None = Field(default=None, description="Optional entity type (e.g., 'PERSON', 'ORG', 'CONCEPT')")


class MemoryItem(BaseModel):
    """Single memory item for retain."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Alice mentioned she's working on a new ML model",
                "timestamp": "2024-01-15T10:30:00Z",
                "context": "team meeting",
                "metadata": {"source": "slack", "channel": "engineering"},
                "document_id": "meeting_notes_2024_01_15",
                "entities": [{"text": "Alice"}, {"text": "ML model", "type": "CONCEPT"}],
                "tags": ["user_a", "user_b"],
            }
        },
    )

    content: str
    timestamp: datetime | str | None = Field(
        default=None,
        description=(
            "When the content occurred. "
            "Accepts an ISO 8601 datetime string (e.g. '2024-01-15T10:30:00Z'), null/omitted (defaults to now), "
            "or the special string 'unset' to explicitly store without any timestamp "
            "(use this for timeless content such as fictional documents or static reference material)."
        ),
    )
    context: str | None = None
    metadata: dict[str, str] | None = None
    document_id: str | None = Field(default=None, description="Optional document ID for this memory item.")
    entities: list[EntityInput] | None = Field(
        default=None,
        description="Optional entities to combine with auto-extracted entities.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Optional tags for visibility scoping. Memories with tags can be filtered during recall.",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def coerce_tags(cls, v):
        """Coerce JSON-string tags to list.

        MCP tool bridges sometimes serialize JSON arrays as strings during
        transport, e.g. '["a", "b"]' instead of ["a", "b"]. This validator
        parses such strings back into lists so the retain call succeeds.
        A plain non-JSON string is wrapped in a single-element list.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return [v]
        return v

    observation_scopes: Literal["per_tag", "combined", "all_combinations"] | list[list[str]] | None = Field(
        default=None,
        title="ObservationScopes",
        description=(
            "How to scope observations during consolidation. "
            "'per_tag' runs one consolidation pass per individual tag, creating separate observations for each tag. "
            "'combined' (default) runs a single pass with all tags together. "
            "A list of tag lists runs one pass per inner list, giving full control over which combinations to use."
        ),
    )
    strategy: str | None = Field(
        default=None,
        description="Named retain strategy for this item. Overrides the bank's default strategy for this item only. "
        "Strategies are defined in the bank config under 'retain_strategies'.",
    )
    update_mode: Literal["replace", "append"] | None = Field(
        default=None,
        description="How to handle an existing document with the same document_id. "
        "'replace' (default) deletes old data and reprocesses from scratch. "
        "'append' concatenates new content to the existing document text and reprocesses.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            if v.lower() == "unset":
                return "unset"
            try:
                # Try parsing as ISO format
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    f"Invalid timestamp/event_date format: '{v}'. Expected ISO format like '2024-01-15T10:30:00' or '2024-01-15T10:30:00Z', or the special value 'unset' to store without a timestamp."
                ) from e
        raise ValueError(f"timestamp must be a string or datetime, got {type(v).__name__}")


class RetainRequest(BaseModel):
    """Request model for retain endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"content": "Alice works at Google", "context": "work", "document_id": "conversation_123"},
                    {
                        "content": "Bob went hiking yesterday",
                        "timestamp": "2024-01-15T10:00:00Z",
                        "document_id": "conversation_123",
                    },
                ],
                "async": False,
            }
        }
    )

    items: list[MemoryItem]
    async_: bool = Field(
        default=False,
        alias="async",
        description="If true, process asynchronously in background. If false, wait for completion (default: false)",
    )
    document_tags: list[str] | None = Field(
        default=None,
        description="Deprecated. Use item-level tags instead.",
        deprecated=True,
    )


class FileRetainMetadata(BaseModel):
    """Metadata for a single file in file retain request."""

    document_id: str | None = Field(default=None, description="Document ID (auto-generated if not provided)")
    context: str | None = Field(default=None, description="Context for the file")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata")
    tags: list[str] | None = Field(default=None, description="Tags for this file")
    timestamp: str | None = Field(default=None, description="ISO timestamp")
    parser: str | list[str] | None = Field(
        default=None,
        description="Parser or ordered fallback chain for this file (overrides request-level parser). "
        "E.g. 'iris' or ['iris', 'markitdown'].",
    )
    strategy: str | None = Field(
        default=None,
        description="Named retain strategy for this file. Overrides the bank's default strategy. "
        "Strategies are defined in the bank config under 'retain_strategies'.",
    )


class FileRetainRequest(BaseModel):
    """Request model for file retain endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "parser": "iris",
                "files_metadata": [
                    {"document_id": "report_2024", "tags": ["quarterly"]},
                    {"context": "meeting notes", "parser": ["iris", "markitdown"]},
                ],
            }
        }
    )

    parser: str | list[str] | None = Field(
        default=None,
        description="Default parser or ordered fallback chain for all files in this request. "
        "E.g. 'markitdown' or ['iris', 'markitdown']. Falls back to server default if not set. "
        "Per-file 'parser' in files_metadata takes precedence over this value.",
    )
    files_metadata: list[FileRetainMetadata] | None = Field(
        default=None,
        description="Metadata for each file (optional, must match number of files if provided)",
    )


class RetainResponse(BaseModel):
    """Response model for retain endpoint."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "success": True,
                "bank_id": "user123",
                "items_count": 2,
                "async": False,
                "usage": {"input_tokens": 500, "output_tokens": 100, "total_tokens": 600},
            }
        },
    )

    success: bool
    bank_id: str
    items_count: int
    is_async: bool = Field(
        alias="async", serialization_alias="async", description="Whether the operation was processed asynchronously"
    )
    operation_id: str | None = Field(
        default=None,
        description="Operation ID for tracking async operations. Use GET /v1/default/banks/{bank_id}/operations to list operations. Only present when async=true. When items use different per-item strategies, use operation_ids instead.",
    )
    operation_ids: list[str] | None = Field(
        default=None,
        description="Operation IDs when items were submitted as multiple strategy groups (async=true with mixed per-item strategies). operation_id is set to the first entry for backward compatibility.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage metrics for LLM calls during fact extraction (only present for synchronous operations)",
    )


class FileRetainResponse(BaseModel):
    """Response model for file upload endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "operation_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "550e8400-e29b-41d4-a716-446655440001",
                    "550e8400-e29b-41d4-a716-446655440002",
                ],
            }
        },
    )

    operation_ids: list[str] = Field(
        description="Operation IDs for tracking file conversion operations. Use GET /v1/default/banks/{bank_id}/operations to list operations."
    )


class FactsIncludeOptions(BaseModel):
    """Options for including facts (based_on) in reflect results."""

    pass  # No additional options needed, just enable/disable


class ToolCallsIncludeOptions(BaseModel):
    """Options for including tool calls in reflect results."""

    output: bool = Field(
        default=True,
        description="Include tool outputs in the trace. Set to false to only include inputs (smaller payload).",
    )


class ReflectIncludeOptions(BaseModel):
    """Options for including additional data in reflect results."""

    facts: FactsIncludeOptions | None = Field(
        default=None,
        description="Include facts that the answer is based on. Set to {} to enable, null to disable (default: disabled).",
    )
    tool_calls: ToolCallsIncludeOptions | None = Field(
        default=None,
        description="Include tool calls trace. Set to {} for full trace (input+output), {output: false} for inputs only.",
    )


class ReflectRequest(BaseModel):
    """Request model for reflect endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What do you think about artificial intelligence?",
                "budget": "low",
                "max_tokens": 4096,
                "include": {"facts": {}},
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary", "key_points"],
                },
                "tags": ["user_a"],
                "tags_match": "any",
            }
        }
    )

    query: str
    budget: Budget = Budget.LOW
    context: str | None = Field(
        default=None,
        description="DEPRECATED: Additional context is now concatenated with the query. "
        "Pass context directly in the query field instead. "
        "If provided, it will be appended to the query for backward compatibility.",
        deprecated=True,
    )
    max_tokens: int = Field(default=4096, description="Maximum tokens for the response")
    include: ReflectIncludeOptions = Field(
        default_factory=ReflectIncludeOptions, description="Options for including additional data (disabled by default)"
    )
    response_schema: dict | None = Field(
        default=None,
        description="Optional JSON Schema for structured output. When provided, the response will include a 'structured_output' field with the LLM response parsed according to this schema.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Filter memories by tags during reflection. If not specified, all memories are considered.",
    )
    tags_match: TagsMatch = Field(
        default="any",
        description="How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), "
        "'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged).",
    )
    tag_groups: list[TagGroup] | None = Field(
        default=None,
        description="Compound tag filter using boolean groups. Groups in the list are AND-ed. "
        "Each group is a leaf {tags, match} or compound {and: [...]}, {or: [...]}, {not: ...}.",
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

    @field_validator("fact_types")
    @classmethod
    def validate_reflect_fact_types(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("fact_types must not be empty. Use null to include all fact types.")
        return v

    @model_validator(mode="after")
    def validate_tags_exclusive(self) -> "ReflectRequest":
        if self.tags is not None and self.tag_groups is not None:
            raise ValueError("'tags' and 'tag_groups' are mutually exclusive. Use 'tag_groups' for compound filtering.")
        return self


class ReflectFact(BaseModel):
    """A fact used in think response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "AI is used in healthcare",
                "type": "world",
                "context": "healthcare discussion",
                "occurred_start": "2024-01-15T10:30:00Z",
                "occurred_end": "2024-01-15T10:30:00Z",
            }
        }
    )

    id: str | None = None
    text: str = Field(
        description="Fact text. When type='observation', this contains markdown-formatted consolidated knowledge"
    )
    type: str | None = None  # fact type: world, experience, opinion, observation
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None


class ReflectDirective(BaseModel):
    """A directive applied during reflect."""

    id: str = Field(description="Directive ID")
    name: str = Field(description="Directive name")
    content: str = Field(description="Directive content")


class ReflectMentalModel(BaseModel):
    """A mental model used during reflect."""

    id: str = Field(description="Mental model ID")
    text: str = Field(description="Mental model content")
    context: str | None = Field(default=None, description="Additional context")


class ReflectToolCall(BaseModel):
    """A tool call made during reflect agent execution."""

    tool: str = Field(description="Tool name: lookup, recall, learn, expand")
    input: dict = Field(description="Tool input parameters")
    output: dict | None = Field(
        default=None, description="Tool output (only included when include.tool_calls.output is true)"
    )
    duration_ms: int = Field(description="Execution time in milliseconds")
    iteration: int = Field(default=0, description="Iteration number (1-based) when this tool was called")


class ReflectLLMCall(BaseModel):
    """An LLM call made during reflect agent execution."""

    scope: str = Field(description="Call scope: agent_1, agent_2, final, etc.")
    duration_ms: int = Field(description="Execution time in milliseconds")


class ReflectBasedOn(BaseModel):
    """Evidence the response is based on: memories, mental models, and directives."""

    memories: list[ReflectFact] = FieldWithDefault(list, description="Memory facts used to generate the response")
    mental_models: list[ReflectMentalModel] = FieldWithDefault(list, description="Mental models used during reflection")
    directives: list[ReflectDirective] = FieldWithDefault(list, description="Directives applied during reflection")


class ReflectTrace(BaseModel):
    """Execution trace of LLM and tool calls during reflection."""

    tool_calls: list[ReflectToolCall] = FieldWithDefault(list, description="Tool calls made during reflection")
    llm_calls: list[ReflectLLMCall] = FieldWithDefault(list, description="LLM calls made during reflection")


class ReflectResponse(BaseModel):
    """Response model for think endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "## AI Overview\n\nBased on my understanding, AI is a **transformative technology**:\n\n- Used extensively in healthcare\n- Discussed in recent conversations\n- Continues to evolve rapidly",
                "based_on": {
                    "memories": [
                        {"id": "123", "text": "AI is used in healthcare", "type": "world"},
                        {"id": "456", "text": "I discussed AI applications last week", "type": "experience"},
                    ],
                },
                "structured_output": {
                    "summary": "AI is transformative",
                    "key_points": ["Used in healthcare", "Discussed recently"],
                },
                "usage": {"input_tokens": 1500, "output_tokens": 500, "total_tokens": 2000},
                "trace": {
                    "tool_calls": [{"tool": "recall", "input": {"query": "AI"}, "duration_ms": 150}],
                    "llm_calls": [{"scope": "agent_1", "duration_ms": 1200}],
                    "observations": [
                        {
                            "id": "obs-1",
                            "name": "AI Technology",
                            "type": "concept",
                            "subtype": "structural",
                        }
                    ],
                },
            }
        }
    )

    text: str = Field(
        description="The reflect response as well-formatted markdown (headers, lists, bold/italic, code blocks, etc.)"
    )
    based_on: ReflectBasedOn | None = Field(
        default=None,
        description="Evidence used to generate the response. Only present when include.facts is set.",
    )
    structured_output: dict | None = Field(
        default=None,
        description="Structured output parsed according to the request's response_schema. Only present when response_schema was provided in the request.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage metrics for LLM calls during reflection.",
    )
    trace: ReflectTrace | None = Field(
        default=None,
        description="Execution trace of tool and LLM calls. Only present when include.tool_calls is set.",
    )


class DispositionTraits(BaseModel):
    """Disposition traits that influence how memories are formed and interpreted."""

    model_config = ConfigDict(json_schema_extra={"example": {"skepticism": 3, "literalism": 3, "empathy": 3}})

    skepticism: int = Field(ge=1, le=5, description="How skeptical vs trusting (1=trusting, 5=skeptical)")
    literalism: int = Field(ge=1, le=5, description="How literally to interpret information (1=flexible, 5=literal)")
    empathy: int = Field(ge=1, le=5, description="How much to consider emotional context (1=detached, 5=empathetic)")


class BankProfileResponse(BaseModel):
    """Response model for bank profile."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "user123",
                "name": "Alice",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "I am a software engineer helping my team stay organized and ship quality code",
            }
        }
    )

    bank_id: str
    name: str
    disposition: DispositionTraits
    mission: str = Field(description="The agent's mission - who they are and what they're trying to accomplish")
    # Deprecated: use mission instead. Kept for backwards compatibility.
    background: str | None = Field(default=None, description="Deprecated: use mission instead")


class UpdateDispositionRequest(BaseModel):
    """Request model for updating disposition traits."""

    disposition: DispositionTraits


class SetMissionRequest(BaseModel):
    """Request model for setting/updating the agent's mission."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"content": "I am a PM helping my engineering team stay organized"}}
    )

    content: str = Field(description="The mission content - who you are and what you're trying to accomplish")


class MissionResponse(BaseModel):
    """Response model for mission update."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mission": "I am a PM helping my engineering team stay organized and ship quality code.",
            }
        }
    )

    mission: str


class AddBackgroundRequest(BaseModel):
    """Request model for adding/merging background information. Deprecated: use SetMissionRequest instead."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"content": "I was born in Texas", "update_disposition": True}}
    )

    content: str = Field(description="New background information to add or merge")
    update_disposition: bool = Field(
        default=True, description="Deprecated - disposition is no longer auto-inferred from mission"
    )


class BackgroundResponse(BaseModel):
    """Response model for background update. Deprecated: use MissionResponse instead."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "mission": "I was born in Texas. I am a software engineer with 10 years of experience.",
            }
        }
    )

    mission: str
    # Deprecated fields kept for backwards compatibility
    background: str | None = Field(default=None, description="Deprecated: same as mission")
    disposition: DispositionTraits | None = None


class BankListItem(BaseModel):
    """Bank list item with profile summary."""

    bank_id: str
    name: str | None = None
    disposition: DispositionTraits
    mission: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class BankListResponse(BaseModel):
    """Response model for listing all banks."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "banks": [
                    {
                        "bank_id": "user123",
                        "name": "Alice",
                        "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                        "mission": "I am a software engineer helping my team ship quality code",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-16T14:20:00Z",
                    }
                ]
            }
        }
    )

    banks: list[BankListItem]


class CreateBankRequest(BaseModel):
    """Request model for creating/updating a bank."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "retain_mission": "Always include technical decisions and architectural trade-offs. Ignore meeting logistics.",
                "observations_mission": "Observations are stable facts about people and projects. Always include preferences and skills.",
            }
        }
    )

    # Deprecated fields — kept for backwards compatibility only
    name: str | None = Field(default=None, description="Deprecated: display label only, not advertised")
    disposition: DispositionTraits | None = Field(
        default=None, description="Deprecated: use update_bank_config instead"
    )
    disposition_skepticism: int | None = Field(
        default=None, ge=1, le=5, description="Deprecated: use update_bank_config instead"
    )
    disposition_literalism: int | None = Field(
        default=None, ge=1, le=5, description="Deprecated: use update_bank_config instead"
    )
    disposition_empathy: int | None = Field(
        default=None, ge=1, le=5, description="Deprecated: use update_bank_config instead"
    )
    # Deprecated: use update_bank_config with reflect_mission instead
    mission: str | None = Field(
        default=None, description="Deprecated: use update_bank_config with reflect_mission instead"
    )
    # Deprecated alias for mission
    background: str | None = Field(
        default=None, description="Deprecated: use update_bank_config with reflect_mission instead"
    )

    # Reflect configuration
    reflect_mission: str | None = Field(
        default=None,
        description="Mission/context for Reflect operations. Guides how Reflect interprets and uses memories.",
    )

    # Operational configuration (applied via config resolver)
    retain_mission: str | None = Field(
        default=None,
        description="Steers what gets extracted during retain(). Injected alongside built-in extraction rules.",
    )
    retain_extraction_mode: str | None = Field(
        default=None,
        description="Fact extraction mode: 'concise' (default), 'verbose', or 'custom'.",
    )
    retain_custom_instructions: str | None = Field(
        default=None,
        description="Custom extraction prompt. Only active when retain_extraction_mode is 'custom'.",
    )
    retain_chunk_size: int | None = Field(
        default=None,
        description="Maximum token size for each content chunk during retain.",
    )
    enable_observations: bool | None = Field(
        default=None,
        description="Toggle automatic observation consolidation after retain().",
    )
    observations_mission: str | None = Field(
        default=None,
        description="Controls what gets synthesised into observations. Replaces built-in consolidation rules entirely.",
    )

    def get_config_updates(self) -> dict[str, Any]:
        """Return only the config fields that were explicitly set.

        reflect_mission takes precedence over deprecated mission/background aliases.
        Individual disposition_* fields take priority over the deprecated disposition dict.
        """
        updates: dict[str, Any] = {}
        # Resolve reflect mission: reflect_mission (new) > mission (deprecated) > background (deprecated)
        resolved_reflect_mission = self.reflect_mission or self.mission or self.background
        if resolved_reflect_mission is not None:
            updates["reflect_mission"] = resolved_reflect_mission
        # Disposition: individual fields take priority over legacy disposition dict
        if self.disposition_skepticism is not None:
            updates["disposition_skepticism"] = self.disposition_skepticism
        elif self.disposition is not None:
            updates["disposition_skepticism"] = self.disposition.skepticism
        if self.disposition_literalism is not None:
            updates["disposition_literalism"] = self.disposition_literalism
        elif self.disposition is not None:
            updates["disposition_literalism"] = self.disposition.literalism
        if self.disposition_empathy is not None:
            updates["disposition_empathy"] = self.disposition_empathy
        elif self.disposition is not None:
            updates["disposition_empathy"] = self.disposition.empathy
        for field_name in (
            "retain_mission",
            "retain_extraction_mode",
            "retain_custom_instructions",
            "retain_chunk_size",
            "enable_observations",
            "observations_mission",
        ):
            value = getattr(self, field_name)
            if value is not None:
                updates[field_name] = value
        return updates


class BankConfigUpdate(BaseModel):
    """Request model for updating bank configuration."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "updates": {
                    "llm_model": "claude-sonnet-4-5",
                    "retain_extraction_mode": "verbose",
                    "retain_custom_instructions": "Extract technical details carefully",
                }
            }
        }
    )

    updates: dict[str, Any] = Field(
        description="Configuration overrides. Keys can be in Python field format (llm_provider) "
        "or environment variable format (HINDSIGHT_API_LLM_PROVIDER). "
        "Only hierarchical fields can be overridden per-bank."
    )


class BankConfigResponse(BaseModel):
    """Response model for bank configuration."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "my-bank",
                "config": {
                    "llm_provider": "openai",
                    "llm_model": "gpt-4",
                    "retain_extraction_mode": "verbose",
                },
                "overrides": {
                    "llm_model": "gpt-4",
                    "retain_extraction_mode": "verbose",
                },
            }
        }
    )

    bank_id: str = Field(description="Bank identifier")
    config: dict[str, Any] = Field(
        description="Fully resolved configuration with all hierarchical overrides applied (Python field names)"
    )
    overrides: dict[str, Any] = Field(description="Bank-specific configuration overrides only (Python field names)")


class GraphDataResponse(BaseModel):
    """Response model for graph data endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "nodes": [
                    {"id": "1", "label": "Alice works at Google", "type": "world"},
                    {"id": "2", "label": "Bob went hiking", "type": "world"},
                ],
                "edges": [{"from": "1", "to": "2", "type": "semantic", "weight": 0.8}],
                "table_rows": [
                    {
                        "id": "abc12345...",
                        "text": "Alice works at Google",
                        "context": "Work info",
                        "date": "2024-01-15 10:30",
                        "entities": "Alice (PERSON), Google (ORGANIZATION)",
                    }
                ],
                "total_units": 2,
                "limit": 1000,
            }
        }
    )

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    table_rows: list[dict[str, Any]]
    total_units: int
    limit: int


class ListMemoryUnitsResponse(BaseModel):
    """Response model for list memory units endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "text": "Alice works at Google on the AI team",
                        "context": "Work conversation",
                        "date": "2024-01-15T10:30:00Z",
                        "type": "world",
                        "entities": "Alice (PERSON), Google (ORGANIZATION)",
                    }
                ],
                "total": 150,
                "limit": 100,
                "offset": 0,
            }
        }
    )

    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class ListDocumentsResponse(BaseModel):
    """Response model for list documents endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": "session_1",
                        "bank_id": "user123",
                        "content_hash": "abc123",
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:30:00Z",
                        "text_length": 5420,
                        "memory_unit_count": 15,
                        "tags": ["user_a", "session_123"],
                    }
                ],
                "total": 50,
                "limit": 100,
                "offset": 0,
            }
        }
    )

    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class TagItem(BaseModel):
    """Single tag with usage count."""

    tag: str = Field(description="The tag value")
    count: int = Field(description="Number of memories with this tag")


class ListTagsResponse(BaseModel):
    """Response model for list tags endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"tag": "user:alice", "count": 42},
                    {"tag": "user:bob", "count": 15},
                    {"tag": "session:abc123", "count": 8},
                ],
                "total": 25,
                "limit": 100,
                "offset": 0,
            }
        }
    )

    items: list[TagItem]
    total: int
    limit: int
    offset: int


class DocumentResponse(BaseModel):
    """Response model for get document endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "session_1",
                "bank_id": "user123",
                "original_text": "Full document text here...",
                "content_hash": "abc123",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "memory_unit_count": 15,
                "tags": ["user_a", "session_123"],
                "document_metadata": {"source": "slack", "channel": "#general"},
                "retain_params": {"context": "Team meeting notes", "event_date": "2024-01-15"},
            }
        }
    )

    id: str
    bank_id: str
    original_text: str
    content_hash: str | None
    created_at: str
    updated_at: str
    memory_unit_count: int
    tags: list[str] = FieldWithDefault(list, description="Tags associated with this document")
    document_metadata: dict[str, Any] | None = Field(default=None, description="Document metadata")
    retain_params: dict[str, Any] | None = Field(default=None, description="Parameters used during retain")


class UpdateDocumentRequest(BaseModel):
    """Request model for updating a document's mutable fields."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tags": ["team-a", "team-b"],
            }
        }
    )

    tags: list[str] | None = Field(
        default=None,
        description="New tags for the document and its memory units. "
        "Triggers observation invalidation and re-consolidation.",
    )


class UpdateDocumentResponse(BaseModel):
    """Response model for update document endpoint."""

    success: bool = True


class DeleteDocumentResponse(BaseModel):
    """Response model for delete document endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Document 'session_1' and 5 associated memory units deleted successfully",
                "document_id": "session_1",
                "memory_units_deleted": 5,
            }
        }
    )

    success: bool
    message: str
    document_id: str
    memory_units_deleted: int


class ChunkResponse(BaseModel):
    """Response model for get chunk endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": "user123_session_1_0",
                "document_id": "session_1",
                "bank_id": "user123",
                "chunk_index": 0,
                "chunk_text": "This is the first chunk of the document...",
                "created_at": "2024-01-15T10:30:00Z",
            }
        }
    )

    chunk_id: str
    document_id: str
    bank_id: str
    chunk_index: int
    chunk_text: str
    created_at: str


class DeleteResponse(BaseModel):
    """Response model for delete operations."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"success": True, "message": "Deleted successfully", "deleted_count": 10}}
    )

    success: bool
    message: str | None = None
    deleted_count: int | None = None


class ClearMemoryObservationsResponse(BaseModel):
    """Response model for clearing observations for a specific memory."""

    model_config = ConfigDict(json_schema_extra={"example": {"deleted_count": 3}})

    deleted_count: int


class RecoverConsolidationResponse(BaseModel):
    """Response model for recovering failed consolidation."""

    model_config = ConfigDict(json_schema_extra={"example": {"retried_count": 42}})

    retried_count: int


class BankStatsResponse(BaseModel):
    """Response model for bank statistics endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "user123",
                "total_nodes": 150,
                "total_links": 300,
                "total_documents": 10,
                "nodes_by_fact_type": {"fact": 100, "preference": 30, "observation": 20},
                "links_by_link_type": {"temporal": 150, "semantic": 100, "entity": 50},
                "links_by_fact_type": {"fact": 200, "preference": 60, "observation": 40},
                "links_breakdown": {"fact": {"temporal": 100, "semantic": 60, "entity": 40}},
                "pending_operations": 2,
                "failed_operations": 0,
                "last_consolidated_at": "2024-01-15T10:30:00Z",
                "pending_consolidation": 0,
                "failed_consolidation": 0,
                "total_observations": 45,
            }
        }
    )

    bank_id: str
    total_nodes: int
    total_links: int
    total_documents: int
    nodes_by_fact_type: dict[str, int]
    links_by_link_type: dict[str, int]
    links_by_fact_type: dict[str, int]
    links_breakdown: dict[str, dict[str, int]]
    pending_operations: int
    failed_operations: int
    operations_by_status: dict[str, int] = Field(
        default_factory=dict,
        description="Async operations grouped by status (pending, in_progress, completed, failed, cancelled).",
    )
    # Consolidation stats
    last_consolidated_at: str | None = Field(default=None, description="When consolidation last ran (ISO format)")
    pending_consolidation: int = Field(default=0, description="Number of memories not yet processed into observations")
    failed_consolidation: int = Field(
        default=0,
        description="Number of source memories (world/experience) whose consolidation permanently failed and can be retried via the consolidation recovery endpoint.",
    )
    total_observations: int = Field(default=0, description="Total number of observations")


class MemoryTimeseriesBucket(BaseModel):
    """One bucket in the memory ingestion time-series."""

    time: str = Field(description="Bucket start timestamp in ISO-8601 (UTC).")
    world: int = Field(default=0, description="World-fact memories ingested in this bucket.")
    experience: int = Field(default=0, description="Experience memories ingested in this bucket.")
    observation: int = Field(default=0, description="Observations recorded in this bucket.")


class MemoriesTimeseriesResponse(BaseModel):
    """Time-series of memory ingestion bucketed by time and fact type."""

    bank_id: str
    period: str = Field(description="One of: 1h, 12h, 1d, 7d, 30d, 90d.")
    trunc: str = Field(description="Bucket granularity: minute, hour, day.")
    buckets: list[MemoryTimeseriesBucket] = Field(
        default_factory=list,
        description="Per-bucket counts, always returned fully padded for the requested period.",
    )


# Mental Model models


# =========================================================================
# Directive Models
# =========================================================================


class DirectiveResponse(BaseModel):
    """Response model for a directive."""

    id: str
    bank_id: str
    name: str
    content: str
    priority: int = 0
    is_active: bool = True
    tags: list[str] = FieldWithDefault(list)
    created_at: str | None = None
    updated_at: str | None = None


class DirectiveListResponse(BaseModel):
    """Response model for listing directives."""

    items: list[DirectiveResponse]


class CreateDirectiveRequest(BaseModel):
    """Request model for creating a directive."""

    name: str = Field(description="Human-readable name for the directive")
    content: str = Field(description="The directive text to inject into prompts")
    priority: int = Field(default=0, description="Higher priority directives are injected first")
    is_active: bool = Field(default=True, description="Whether this directive is active")
    tags: list[str] = FieldWithDefault(list, description="Tags for filtering")


class UpdateDirectiveRequest(BaseModel):
    """Request model for updating a directive."""

    name: str | None = Field(default=None, description="New name")
    content: str | None = Field(default=None, description="New content")
    priority: int | None = Field(default=None, description="New priority")
    is_active: bool | None = Field(default=None, description="New active status")
    tags: list[str] | None = Field(default=None, description="New tags")


# =========================================================================
# Mental Models (stored reflect responses)
# =========================================================================


class MentalModelTrigger(BaseModel):
    """Trigger settings for a mental model."""

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
            "or 'any' when the model has no tags. "
            "Set to 'any' to include untagged memories alongside tagged ones during refresh."
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

    @field_validator("fact_types")
    @classmethod
    def validate_fact_types(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("fact_types must not be empty. Use null to include all fact types.")
        return v


class MentalModelResponse(BaseModel):
    """Response model for a mental model (stored reflect response)."""

    id: str
    bank_id: str
    name: str
    source_query: str | None = None
    content: str | None = Field(
        default=None,
        description="The mental model content as well-formatted markdown (auto-generated from reflect endpoint)",
    )
    tags: list[str] = FieldWithDefault(list)
    max_tokens: int | None = Field(default=None)
    trigger: MentalModelTrigger | None = Field(default=None)
    last_refreshed_at: str | None = None
    created_at: str | None = None
    reflect_response: dict | None = Field(
        default=None,
        description="Full reflect API response payload including based_on facts and observations",
    )
    is_stale: bool | None = Field(
        default=None,
        description=(
            "True when new memories matching this mental model's tag/fact_type scope have been "
            "ingested since last_refreshed_at, or consolidation has pending items. Only populated "
            "when detail=full."
        ),
    )


class MentalModelListResponse(BaseModel):
    """Response model for listing mental models."""

    items: list[MentalModelResponse]


class CreateMentalModelRequest(BaseModel):
    """Request model for creating a mental model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "team-communication",
                "name": "Team Communication Preferences",
                "source_query": "How does the team prefer to communicate?",
                "tags": ["team"],
                "max_tokens": 2048,
                "trigger": {"refresh_after_consolidation": False},
            }
        }
    )

    id: str | None = Field(
        None, description="Optional custom ID for the mental model (alphanumeric lowercase with hyphens)"
    )
    name: str = Field(description="Human-readable name for the mental model")
    source_query: str = Field(description="The query to run to generate content")
    tags: list[str] = FieldWithDefault(list, description="Tags for scoped visibility")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Maximum tokens for generated content")
    trigger: MentalModelTrigger = FieldWithDefault(MentalModelTrigger, description="Trigger settings")


class CreateMentalModelResponse(BaseModel):
    """Response model for mental model creation."""

    mental_model_id: str | None = Field(None, description="ID of the created mental model")
    operation_id: str = Field(description="Operation ID to track refresh progress")


class UpdateMentalModelRequest(BaseModel):
    """Request model for updating a mental model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Team Communication Preferences",
                "source_query": "How does the team prefer to communicate?",
                "max_tokens": 4096,
                "tags": ["team", "communication"],
                "trigger": {"refresh_after_consolidation": True},
            }
        }
    )

    name: str | None = Field(default=None, description="New name for the mental model")
    source_query: str | None = Field(default=None, description="New source query for the mental model")
    max_tokens: int | None = Field(default=None, ge=256, le=8192, description="Maximum tokens for generated content")
    tags: list[str] | None = Field(default=None, description="Tags for scoped visibility")
    trigger: MentalModelTrigger | None = Field(default=None, description="Trigger settings")


# =========================================================================
# Bank Templates (import/export)
# =========================================================================

# Current manifest schema version. Bump when making breaking changes.
BANK_TEMPLATE_CURRENT_VERSION = "1"


class BankTemplateMentalModel(BaseModel):
    """A mental model definition within a bank template manifest."""

    id: str = Field(description="Unique ID for the mental model (alphanumeric lowercase with hyphens)")
    name: str = Field(description="Human-readable name for the mental model")
    source_query: str = Field(description="The query to run to generate content")
    tags: list[str] = FieldWithDefault(list, description="Tags for scoped visibility")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Maximum tokens for generated content")
    trigger: MentalModelTrigger = FieldWithDefault(MentalModelTrigger, description="Trigger settings")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", v):
            raise ValueError(
                f"Mental model id '{v}' must be alphanumeric lowercase with hyphens, starting with a letter or digit."
            )
        return v


class BankTemplateConfig(BaseModel):
    """Bank configuration fields within a template manifest.

    Only includes configurable (per-bank) fields. Credential fields
    (API keys, base URLs) are intentionally excluded for security.
    """

    reflect_mission: str | None = Field(default=None, description="Mission/context for Reflect operations")
    retain_mission: str | None = Field(default=None, description="Steers what gets extracted during retain")
    retain_extraction_mode: str | None = Field(
        default=None, description="Fact extraction mode: 'concise' (default), 'verbose', or 'custom'"
    )
    retain_custom_instructions: str | None = Field(
        default=None, description="Custom extraction prompt (when mode='custom')"
    )
    retain_chunk_size: int | None = Field(default=None, description="Max token size for each content chunk")
    enable_observations: bool | None = Field(default=None, description="Toggle observation consolidation")
    observations_mission: str | None = Field(default=None, description="Controls what gets synthesised")
    disposition_skepticism: int | None = Field(default=None, ge=1, le=5, description="Skepticism trait (1-5)")
    disposition_literalism: int | None = Field(default=None, ge=1, le=5, description="Literalism trait (1-5)")
    disposition_empathy: int | None = Field(default=None, ge=1, le=5, description="Empathy trait (1-5)")
    entity_labels: list[dict[str, Any]] | None = Field(
        default=None, description="Controlled vocabulary for entity labels"
    )
    entities_allow_free_form: bool | None = Field(
        default=None, description="Allow entities outside the label vocabulary"
    )
    retain_default_strategy: str | None = Field(
        default=None, description="Name of the default retain strategy (key into retain_strategies map)"
    )
    retain_strategies: dict | None = Field(
        default=None, description="Map of retain strategy name to per-strategy config dict"
    )
    retain_chunk_batch_size: int | None = Field(
        default=None, description="Max chunks per streaming batch (0 disables batching)"
    )
    mcp_enabled_tools: list[str] | None = Field(
        default=None, description="MCP tool allowlist for this bank (None = all tools)"
    )
    consolidation_llm_batch_size: int | None = Field(
        default=None, description="LLM batch size for observation consolidation"
    )
    consolidation_source_facts_max_tokens: int | None = Field(
        default=None, description="Max tokens of source facts per consolidation batch"
    )
    consolidation_source_facts_max_tokens_per_observation: int | None = Field(
        default=None, description="Max tokens of source facts per observation"
    )
    max_observations_per_scope: int | None = Field(
        default=None, description="Max observations to retain per consolidation scope"
    )
    reflect_source_facts_max_tokens: int | None = Field(
        default=None, description="Max tokens of source facts per reflect call"
    )
    llm_gemini_safety_settings: list | None = Field(
        default=None, description="Per-bank Gemini/VertexAI safety filter settings"
    )
    recall_budget_function: str | None = Field(
        default=None, description="Recall budget mapping function: 'fixed' or 'adaptive'"
    )
    recall_budget_fixed_low: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=low (function='fixed')"
    )
    recall_budget_fixed_mid: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=mid (function='fixed')"
    )
    recall_budget_fixed_high: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=high (function='fixed')"
    )
    recall_budget_adaptive_low: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=low (function='adaptive')"
    )
    recall_budget_adaptive_mid: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=mid (function='adaptive')"
    )
    recall_budget_adaptive_high: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=high (function='adaptive')"
    )
    recall_budget_min: int | None = Field(default=None, description="Floor for the adaptive function (after clamping)")
    recall_budget_max: int | None = Field(
        default=None, description="Ceiling for the adaptive function (after clamping)"
    )

    def get_config_updates(self) -> dict[str, Any]:
        """Return only the fields that were explicitly set (non-None)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BankTemplateDirective(BaseModel):
    """A directive definition within a bank template manifest.

    Directives are matched by name on re-import: existing directives
    with the same name are updated, new ones are created.
    """

    name: str = Field(description="Human-readable name for the directive (used as match key on re-import)")
    content: str = Field(description="The directive text to inject into prompts")
    priority: int = Field(default=0, description="Higher priority directives are injected first")
    is_active: bool = Field(default=True, description="Whether this directive is active")
    tags: list[str] = FieldWithDefault(list, description="Tags for filtering")


class BankTemplateManifest(BaseModel):
    """A bank template manifest for import/export.

    Version field enables forward-compatible schema evolution: the API
    auto-upgrades older manifest versions to the current schema on import.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": "1",
                "bank": {
                    "reflect_mission": "You are helping a support agent remember customer interactions.",
                    "retain_mission": "Extract customer issues, resolutions, and sentiment.",
                    "disposition_empathy": 5,
                    "enable_observations": True,
                },
                "mental_models": [
                    {
                        "id": "sentiment-overview",
                        "name": "Customer Sentiment Overview",
                        "source_query": "What is the overall sentiment trend?",
                        "trigger": {"refresh_after_consolidation": True},
                    }
                ],
                "directives": [
                    {
                        "name": "Always be empathetic",
                        "content": "Always respond with empathy and understanding.",
                        "priority": 10,
                    }
                ],
            }
        }
    )

    version: str = Field(description="Manifest schema version (currently '1')")
    bank: BankTemplateConfig | None = Field(
        default=None, description="Bank configuration to apply. Omit to leave config unchanged."
    )
    mental_models: list[BankTemplateMentalModel] | None = Field(
        default=None, description="Mental models to create or update (matched by id). Omit to leave unchanged."
    )
    directives: list[BankTemplateDirective] | None = Field(
        default=None, description="Directives to create or update (matched by name). Omit to leave unchanged."
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        try:
            ver = int(v)
        except ValueError:
            raise ValueError(f"version must be a numeric string, got '{v}'")
        if ver < 1:
            raise ValueError("version must be >= 1")
        if ver > int(BANK_TEMPLATE_CURRENT_VERSION):
            raise ValueError(
                f"version '{v}' is not supported by this server "
                f"(max supported: {BANK_TEMPLATE_CURRENT_VERSION}). Please upgrade Hindsight."
            )
        return v

    @field_validator("mental_models")
    @classmethod
    def validate_unique_mental_model_ids(
        cls,
        v: list[BankTemplateMentalModel] | None,
    ) -> list[BankTemplateMentalModel] | None:
        if v is None:
            return v
        ids = [m.id for m in v]
        duplicates = [mid for mid in ids if ids.count(mid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate mental model ids: {sorted(set(duplicates))}")
        return v

    @field_validator("directives")
    @classmethod
    def validate_unique_directive_names(
        cls,
        v: list[BankTemplateDirective] | None,
    ) -> list[BankTemplateDirective] | None:
        if v is None:
            return v
        names = [d.name for d in v]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"Duplicate directive names: {sorted(set(duplicates))}")
        return v


class BankTemplateImportResponse(BaseModel):
    """Response model for the bank template import endpoint."""

    bank_id: str = Field(description="Bank that was imported into")
    config_applied: bool = Field(description="Whether bank config was updated")
    mental_models_created: list[str] = FieldWithDefault(list, description="IDs of newly created mental models")
    mental_models_updated: list[str] = FieldWithDefault(list, description="IDs of updated mental models")
    directives_created: list[str] = FieldWithDefault(list, description="Names of newly created directives")
    directives_updated: list[str] = FieldWithDefault(list, description="Names of updated directives")
    operation_ids: list[str] = FieldWithDefault(
        list, description="Operation IDs for mental model content generation (async)"
    )
    dry_run: bool = Field(default=False, description="True if this was a validation-only run")


def validate_bank_template(manifest: "BankTemplateManifest") -> list[str]:
    """Validate a parsed manifest beyond Pydantic's structural checks.

    Returns a list of human-readable error strings (e.g. invalid
    extraction mode values, conflicting settings).
    """
    errors: list[str] = []
    if manifest.bank:
        bank = manifest.bank
        if bank.retain_extraction_mode is not None:
            valid_modes = ("concise", "verbose", "custom", "chunks")
            if bank.retain_extraction_mode not in valid_modes:
                errors.append(
                    f"bank.retain_extraction_mode: must be one of {valid_modes}, got '{bank.retain_extraction_mode}'"
                )
        if bank.retain_custom_instructions and bank.retain_extraction_mode != "custom":
            errors.append("bank.retain_custom_instructions: requires retain_extraction_mode='custom'")
    if manifest.mental_models:
        for i, mm in enumerate(manifest.mental_models):
            if not mm.name.strip():
                errors.append(f"mental_models[{i}].name: must not be empty")
            if not mm.source_query.strip():
                errors.append(f"mental_models[{i}].source_query: must not be empty")
    if manifest.directives:
        for i, d in enumerate(manifest.directives):
            if not d.name.strip():
                errors.append(f"directives[{i}].name: must not be empty")
            if not d.content.strip():
                errors.append(f"directives[{i}].content: must not be empty")
    return errors


async def apply_bank_template_manifest(
    memory,
    bank_id: str,
    manifest: "BankTemplateManifest",
    request_context: "RequestContext",
) -> "BankTemplateImportResponse":
    """Apply a validated BankTemplateManifest to an existing bank.

    Shared by the /import endpoint and the default-template-on-create hook
    driven by HINDSIGHT_API_DEFAULT_BANK_TEMPLATE. The bank MUST already
    exist; caller is responsible for validation (Pydantic + validate_bank_template).
    """
    config_applied = False
    if manifest.bank:
        config_updates = manifest.bank.get_config_updates()
        if config_updates:
            await memory._config_resolver.update_bank_config(bank_id, config_updates, request_context)
            config_applied = True

    created_ids: list[str] = []
    updated_ids: list[str] = []
    operation_ids: list[str] = []

    if manifest.mental_models:
        # Fetch existing mental models to decide create vs update
        existing = await memory.list_mental_models(bank_id=bank_id, request_context=request_context)
        existing_by_id = {m["id"]: m for m in existing}

        for mm in manifest.mental_models:
            if mm.id in existing_by_id:
                await memory.update_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm.id,
                    name=mm.name,
                    source_query=mm.source_query,
                    max_tokens=mm.max_tokens,
                    tags=mm.tags if mm.tags else None,
                    trigger=mm.trigger.model_dump() if mm.trigger else None,
                    request_context=request_context,
                )
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm.id,
                    request_context=request_context,
                )
                operation_ids.append(result["operation_id"])
                updated_ids.append(mm.id)
            else:
                mental_model = await memory.create_mental_model(
                    bank_id=bank_id,
                    name=mm.name,
                    source_query=mm.source_query,
                    content="Generating content...",
                    mental_model_id=mm.id,
                    tags=mm.tags if mm.tags else None,
                    max_tokens=mm.max_tokens,
                    trigger=mm.trigger.model_dump() if mm.trigger else None,
                    request_context=request_context,
                )
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mental_model["id"],
                    request_context=request_context,
                )
                operation_ids.append(result["operation_id"])
                created_ids.append(mm.id)

    directives_created: list[str] = []
    directives_updated: list[str] = []

    if manifest.directives:
        existing_directives = await memory.list_directives(
            bank_id=bank_id, active_only=False, request_context=request_context
        )
        existing_by_name = {d["name"]: d for d in existing_directives}

        for directive in manifest.directives:
            if directive.name in existing_by_name:
                await memory.update_directive(
                    bank_id=bank_id,
                    directive_id=existing_by_name[directive.name]["id"],
                    content=directive.content,
                    priority=directive.priority,
                    is_active=directive.is_active,
                    tags=directive.tags if directive.tags else None,
                    request_context=request_context,
                )
                directives_updated.append(directive.name)
            else:
                await memory.create_directive(
                    bank_id=bank_id,
                    name=directive.name,
                    content=directive.content,
                    priority=directive.priority,
                    is_active=directive.is_active,
                    tags=directive.tags if directive.tags else None,
                    request_context=request_context,
                )
                directives_created.append(directive.name)

    return BankTemplateImportResponse(
        bank_id=bank_id,
        config_applied=config_applied,
        mental_models_created=created_ids,
        mental_models_updated=updated_ids,
        directives_created=directives_created,
        directives_updated=directives_updated,
        operation_ids=operation_ids,
        dry_run=False,
    )


class OperationResponse(BaseModel):
    """Response model for a single async operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "task_type": "retain",
                "items_count": 5,
                "document_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "status": "pending",
                "error_message": None,
            }
        }
    )

    id: str
    task_type: str
    items_count: int
    document_id: str | None = None
    created_at: str
    status: str
    error_message: str | None


class ConsolidationResponse(BaseModel):
    """Response model for consolidation trigger endpoint."""

    operation_id: str = Field(description="ID of the async consolidation operation")
    deduplicated: bool = Field(default=False, description="True if an existing pending task was reused")


class OperationsListResponse(BaseModel):
    """Response model for list operations endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "user123",
                "total": 150,
                "limit": 20,
                "offset": 0,
                "operations": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "task_type": "retain",
                        "created_at": "2024-01-15T10:30:00Z",
                        "status": "pending",
                        "error_message": None,
                    }
                ],
            }
        }
    )

    bank_id: str
    total: int
    limit: int
    offset: int
    operations: list[OperationResponse]


class CancelOperationResponse(BaseModel):
    """Response model for cancel operation endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation 550e8400-e29b-41d4-a716-446655440000 cancelled",
                "operation_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    )

    success: bool
    message: str
    operation_id: str


class RetryOperationResponse(BaseModel):
    """Response model for retry operation endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Operation 550e8400-e29b-41d4-a716-446655440000 queued for retry",
                "operation_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    )

    success: bool
    message: str
    operation_id: str


class ChildOperationStatus(BaseModel):
    """Status of a child operation (for batch operations)."""

    operation_id: str
    status: str
    sub_batch_index: int | None = None
    items_count: int | None = None
    error_message: str | None = None


class OperationStatusResponse(BaseModel):
    """Response model for getting a single operation status."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "operation_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "operation_type": "refresh_mental_models",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:31:30Z",
                "completed_at": "2024-01-15T10:31:30Z",
                "error_message": None,
            }
        }
    )

    operation_id: str
    status: Literal["pending", "completed", "failed", "not_found"]
    operation_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    result_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Internal metadata for debugging. Structure may change without notice. Not for production use.",
    )
    child_operations: list[ChildOperationStatus] | None = Field(
        default=None, description="Child operations for batch operations (if applicable)"
    )
    task_payload: dict[str, Any] | None = Field(
        default=None,
        description="Raw task payload (params the operation was submitted with). Only populated when include_payload=true.",
    )


class AsyncOperationSubmitResponse(BaseModel):
    """Response model for submitting an async operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "operation_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
            }
        }
    )

    operation_id: str
    status: str


class FeaturesInfo(BaseModel):
    """Feature flags indicating which capabilities are enabled."""

    observations: bool = Field(description="Whether observations (auto-consolidation) are enabled")
    mcp: bool = Field(description="Whether MCP (Model Context Protocol) server is enabled")
    worker: bool = Field(description="Whether the background worker is enabled")
    bank_config_api: bool = Field(description="Whether per-bank configuration API is enabled")
    file_upload_api: bool = Field(description="Whether file upload/conversion API is enabled")


class VersionResponse(BaseModel):
    """Response model for the version/info endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_version": "0.4.0",
                "features": {
                    "observations": False,
                    "mcp": True,
                    "worker": True,
                    "bank_config_api": False,
                    "file_upload_api": True,
                },
            }
        }
    )

    api_version: str = Field(description="API version string")
    features: FeaturesInfo = Field(description="Enabled feature flags")


# =========================================================================
# Webhook Models
# =========================================================================


from hindsight_api.webhooks.models import WebhookHttpConfig


class CreateWebhookRequest(BaseModel):
    """Request model for registering a webhook."""

    url: str = Field(description="HTTP(S) endpoint URL to deliver events to")
    secret: str | None = Field(default=None, description="HMAC-SHA256 signing secret (optional)")
    event_types: list[str] = Field(
        default=["consolidation.completed"],
        description="List of event types to deliver. Currently supported: 'consolidation.completed'",
    )
    enabled: bool = Field(default=True, description="Whether this webhook is active")
    http_config: WebhookHttpConfig = Field(
        default_factory=WebhookHttpConfig,
        description="HTTP delivery configuration (method, timeout, headers, params)",
    )


class WebhookResponse(BaseModel):
    """Response model for a webhook."""

    id: str
    bank_id: str | None
    url: str
    secret: str | None = Field(default=None, description="Signing secret (redacted in responses)")
    event_types: list[str]
    enabled: bool
    http_config: WebhookHttpConfig = Field(default_factory=WebhookHttpConfig)
    created_at: str | None = None
    updated_at: str | None = None


class UpdateWebhookRequest(BaseModel):
    """Request model for updating a webhook. Only provided fields are updated."""

    url: str | None = Field(default=None, description="HTTP(S) endpoint URL")
    secret: str | None = Field(
        default=None, description="HMAC-SHA256 signing secret. Omit to keep existing; send null to clear."
    )
    event_types: list[str] | None = Field(default=None, description="List of event types")
    enabled: bool | None = Field(default=None, description="Whether this webhook is active")
    http_config: WebhookHttpConfig | None = Field(default=None, description="HTTP delivery configuration")


class WebhookListResponse(BaseModel):
    """Response model for listing webhooks."""

    items: list[WebhookResponse]


class WebhookDeliveryResponse(BaseModel):
    """Response model for a webhook delivery record."""

    id: str
    webhook_id: str | None
    url: str
    event_type: str
    status: str
    attempts: int
    next_retry_at: str | None = None
    last_error: str | None = None
    last_response_status: int | None = None
    last_response_body: str | None = None
    last_attempt_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_async_operation_row(cls, row: dict) -> "WebhookDeliveryResponse":
        import json as _json

        raw = row["task_payload"]
        if isinstance(raw, str):
            task_payload = _json.loads(raw)
        elif isinstance(raw, dict):
            task_payload = raw
        else:
            task_payload = {}

        raw_meta = row.get("result_metadata")
        if isinstance(raw_meta, str):
            result_metadata = _json.loads(raw_meta) if raw_meta else {}
        elif isinstance(raw_meta, dict):
            result_metadata = raw_meta
        else:
            result_metadata = {}

        return cls(
            id=str(row["operation_id"]),
            webhook_id=task_payload.get("webhook_id"),
            url=task_payload.get("url", ""),
            event_type=task_payload.get("event_type", ""),
            status=row["status"],
            attempts=row["retry_count"] + 1,
            next_retry_at=row["next_retry_at"],
            last_error=row["error_message"],
            last_response_status=result_metadata.get("last_status_code"),
            last_response_body=result_metadata.get("last_response_body"),
            last_attempt_at=result_metadata.get("last_attempt_at"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class WebhookDeliveryListResponse(BaseModel):
    """Response model for listing webhook deliveries."""

    items: list[WebhookDeliveryResponse]
    next_cursor: str | None = None


def _make_audited_http(audit_logger_getter: Callable[[], AuditLogger | None]):
    """Create an audit decorator bound to an audit logger getter.

    Returns a decorator factory that can be used as @audited("action_name").
    """
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    from functools import wraps
    from typing import Callable as _Callable

    def audited(action: str, *, request_param: str | None = "request"):
        """Decorator that wraps an HTTP handler with audit logging.

        Args:
            action: The audit action name (e.g. "retain", "recall").
            request_param: Name of the kwarg holding the Pydantic request model
                           (None if handler has no request body). Also supports "body".
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                al = audit_logger_getter()
                if al is None or not al.is_enabled(action):
                    return await func(*args, **kwargs)

                bank_id = kwargs.get("bank_id")
                started_at = _dt.now(_tz.utc)

                req_data = None
                if request_param:
                    req_obj = kwargs.get(request_param)
                    if req_obj is not None and hasattr(req_obj, "model_dump"):
                        req_data = req_obj.model_dump(mode="json")
                    elif req_obj is not None and isinstance(req_obj, dict):
                        req_data = req_obj

                entry = AuditEntry(
                    action=action,
                    transport="http",
                    bank_id=bank_id,
                    started_at=started_at,
                    request=req_data,
                )

                try:
                    result = await func(*args, **kwargs)
                    if hasattr(result, "model_dump"):
                        entry.response = result.model_dump(mode="json")
                    elif isinstance(result, dict):
                        entry.response = result
                    return result
                finally:
                    entry.ended_at = _dt.now(_tz.utc)
                    al.log_fire_and_forget(entry)

            # Preserve FastAPI's dependency injection signature
            import inspect

            wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
            return wrapper

        return decorator

    return audited


def create_app(
    memory: MemoryEngine,
    initialize_memory: bool = True,
    http_extension: HttpExtension | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        memory: MemoryEngine instance (already initialized with required parameters).
                Migrations are controlled by the MemoryEngine's run_migrations parameter.
        initialize_memory: Whether to initialize memory system on startup (default: True)
        http_extension: Optional HTTP extension to mount custom endpoints under /extension/.
                       If None, attempts to load from HINDSIGHT_API_HTTP_EXTENSION env var.

    Returns:
        Configured FastAPI application

    Note:
        When mounting this app as a sub-application, the lifespan events may not fire.
        In that case, you should call memory.initialize() manually before starting the server
        and memory.close() when shutting down.
    """
    # Load HTTP extension from environment if not provided
    if http_extension is None:
        http_extension = load_extension("HTTP", HttpExtension)
        if http_extension:
            logging.info(f"Loaded HTTP extension: {http_extension.__class__.__name__}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Lifespan context manager for startup and shutdown events.
        Note: This only fires when running the app standalone, not when mounted.
        """
        import asyncio
        import socket

        from hindsight_api.config import get_config
        from hindsight_api.worker import WorkerPoller

        config = get_config()
        poller = None
        poller_task = None

        # Initialize OpenTelemetry metrics
        try:
            prometheus_reader = initialize_metrics(service_name="hindsight-api", service_version="1.0.0")
            create_metrics_collector()
            app.state.prometheus_reader = prometheus_reader
            logging.info("Metrics initialized - available at /metrics endpoint")
        except Exception as e:
            logging.warning(f"Failed to initialize metrics: {e}. Metrics will be disabled (using no-op collector).")
            app.state.prometheus_reader = None
            # Metrics collector is already initialized as no-op by default

        # Initialize OpenTelemetry tracing if enabled
        if config.otel_traces_enabled:
            if not config.otel_exporter_otlp_endpoint:
                logging.warning("OTEL tracing enabled but no endpoint configured. Tracing disabled.")
            else:
                from hindsight_api.tracing import create_span_recorder, initialize_tracing

                try:
                    initialize_tracing(
                        service_name=config.otel_service_name,
                        endpoint=config.otel_exporter_otlp_endpoint,
                        headers=config.otel_exporter_otlp_headers,
                        deployment_environment=config.otel_deployment_environment,
                    )
                    create_span_recorder()
                    logging.info("OpenTelemetry tracing enabled and configured")
                except Exception as e:
                    logging.error(f"Failed to initialize tracing: {e}")
                    logging.warning("Continuing without tracing")

        # Startup: Initialize database and memory system (migrations run inside initialize if enabled)
        if initialize_memory:
            await memory.initialize()
            logging.info("Memory system initialized")

            # Set up DB pool metrics after memory initialization
            metrics_collector = get_metrics_collector()
            if memory._pool is not None and hasattr(metrics_collector, "set_db_pool"):
                metrics_collector.set_db_pool(memory._pool)
                logging.info("DB pool metrics configured")

        # Start worker poller if enabled (standalone mode)
        if config.worker_enabled and memory._pool is not None:
            from ..config import DEFAULT_DATABASE_SCHEMA

            worker_id = config.worker_id or socket.gethostname()
            # Convert default schema to None for SQL compatibility (no schema prefix)
            schema = None if config.database_schema == DEFAULT_DATABASE_SCHEMA else config.database_schema
            poller = WorkerPoller(
                pool=memory._pool,
                worker_id=worker_id,
                executor=memory.execute_task,
                poll_interval_ms=config.worker_poll_interval_ms,
                schema=schema,
                tenant_extension=memory._tenant_extension,
                max_slots=config.worker_max_slots,
                consolidation_max_slots=config.worker_consolidation_max_slots,
            )
            poller_task = asyncio.create_task(poller.run())
            logging.info(f"Worker poller started (worker_id={worker_id})")

        # Call tenant extension startup hook (e.g. JWKS fetch for Supabase)
        tenant_extension = memory.tenant_extension
        if tenant_extension:
            await tenant_extension.on_startup()
            logging.info("Tenant extension started")

        # Call HTTP extension startup hook
        if http_extension:
            await http_extension.on_startup()
            logging.info("HTTP extension started")

        yield

        # Shutdown worker poller if running
        if poller is not None:
            await poller.shutdown_graceful(timeout=30.0)
            if poller_task is not None:
                poller_task.cancel()
                try:
                    await poller_task
                except asyncio.CancelledError:
                    pass
            logging.info("Worker poller stopped")

        # Call tenant extension shutdown hook
        if tenant_extension:
            await tenant_extension.on_shutdown()
            logging.info("Tenant extension stopped")

        # Call HTTP extension shutdown hook
        if http_extension:
            await http_extension.on_shutdown()
            logging.info("HTTP extension stopped")

        # Shutdown: Cleanup memory system
        await memory.close()
        logging.info("Memory system closed")

    from hindsight_api import __version__
    from hindsight_api.config import get_config

    config = get_config()

    app = FastAPI(
        title="Hindsight HTTP API",
        version=__version__,
        description="HTTP API for Hindsight",
        contact={
            "name": "Memory System",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        lifespan=lifespan,
        root_path=config.base_path,
    )

    # IMPORTANT: Set memory on app.state immediately, don't wait for lifespan
    # This is required for mounted sub-applications where lifespan may not fire
    app.state.memory = memory
    app.state.audit_logger = memory.audit_logger

    # ---------------------------------------------------------------------------
    # Patch OpenAPI schema: align ValidationError with Pydantic v2 error format
    # ---------------------------------------------------------------------------
    # FastAPI auto-generates ValidationError with only loc/msg/type, but Pydantic
    # v2 actually returns additional fields: input (the rejected value), ctx (extra
    # context dict), and url (link to error docs). Without these in the spec,
    # generated clients using strict JSON decoding break on real 422 responses.
    _original_openapi = app.openapi

    def _patched_openapi() -> dict[str, Any]:
        schema = _original_openapi()
        ve = schema.get("components", {}).get("schemas", {}).get("ValidationError")
        if ve and "input" not in ve.get("properties", {}):
            ve["properties"]["input"] = {"title": "Input"}
            ve["properties"]["ctx"] = {"title": "Context", "type": "object"}
            ve["properties"]["url"] = {"title": "URL", "type": "string"}
        return schema

    app.openapi = _patched_openapi  # type: ignore[assignment]

    # Add unknown parameters detection middleware
    @app.middleware("http")
    async def unknown_params_middleware(request, call_next):
        """Detect unknown query params and body fields, log warning and set response header."""
        import inspect

        from starlette.routing import Match

        ignored_params: list[str] = []

        # --- Query parameters ---
        if request.query_params:
            for route in app.routes:
                match, _ = route.matches(request.scope)
                if match == Match.FULL:
                    endpoint = getattr(route, "endpoint", None)
                    if endpoint:
                        sig = inspect.signature(endpoint)
                        declared = set(sig.parameters.keys())
                        path_params = set(getattr(route, "param_convertors", {}).keys()) | set(
                            request.path_params.keys()
                        )
                        known_query = declared - path_params
                        for name in request.query_params:
                            if name not in known_query and name not in path_params:
                                ignored_params.append(name)
                    break

        # --- Body fields ---
        body_ignored: list[str] = []
        content_type = request.headers.get("content-type", "")
        if request.method in ("POST", "PUT", "PATCH") and "application/json" in content_type:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    body_json = json.loads(body_bytes)
                    if isinstance(body_json, dict):
                        for route in app.routes:
                            match, _ = route.matches(request.scope)
                            if match == Match.FULL:
                                endpoint = getattr(route, "endpoint", None)
                                if endpoint:
                                    sig = inspect.signature(endpoint)
                                    for param in sig.parameters.values():
                                        ann = param.annotation
                                        if isinstance(ann, type) and issubclass(ann, BaseModel):
                                            known_fields = set(ann.model_fields.keys())
                                            for key in body_json:
                                                if key not in known_fields:
                                                    body_ignored.append(key)
                                            break
                                break
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        all_ignored = ignored_params + body_ignored

        response = await call_next(request)

        if all_ignored:
            ignored_str = ", ".join(all_ignored)
            logger.warning(
                "Unknown parameters ignored: [%s] for %s %s",
                ignored_str,
                request.method,
                request.url.path,
            )
            response.headers["X-Ignored-Params"] = ignored_str

        return response

    # Add HTTP metrics middleware
    @app.middleware("http")
    async def http_metrics_middleware(request, call_next):
        """Record HTTP request metrics."""
        # Normalize endpoint path to reduce cardinality
        # Replace UUIDs and numeric IDs with placeholders
        import re

        from starlette.requests import Request

        path = request.url.path
        # Replace UUIDs
        path = re.sub(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{id}", path)
        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)

        status_code = [500]  # Default to 500, will be updated
        metrics_collector = get_metrics_collector()

        with metrics_collector.record_http_request(request.method, path, lambda: status_code[0]):
            response = await call_next(request)
            status_code[0] = response.status_code
            return response

    # Register all routes
    _register_routes(app)

    # Mount HTTP extension router if available
    if http_extension:
        extension_router = http_extension.get_router(memory)
        app.include_router(extension_router, prefix="/ext", tags=["Extension"])
        logging.info("HTTP extension router mounted at /ext/")

        # Mount root router if provided (for well-known endpoints, etc.)
        root_router = http_extension.get_root_router(memory)
        if root_router:
            app.include_router(root_router)
            logging.info("HTTP extension root router mounted")

    return app


def _register_routes(app: FastAPI):
    """Register all API routes on the given app instance."""

    # Create audit decorator bound to this app's audit logger
    audited = _make_audited_http(lambda: getattr(app.state, "audit_logger", None))

    def get_request_context(authorization: str | None = Header(default=None)) -> RequestContext:
        """
        Extract request context from Authorization header.

        Supports:
        - Bearer token: "Bearer <api_key>"
        - Direct API key: "<api_key>"

        Returns RequestContext with extracted API key (may be None if no auth header).
        """
        api_key = None
        if authorization:
            if authorization.lower().startswith("bearer "):
                api_key = authorization[7:].strip()
            else:
                api_key = authorization.strip()
        return RequestContext(api_key=api_key)

    # Global exception handler for authentication errors
    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(request, exc: AuthenticationError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=401,
            content={"detail": str(exc)},
        )

    @app.get(
        "/health",
        summary="Health check endpoint",
        description="Checks the health of the API and database connection",
        tags=["Monitoring"],
    )
    async def health_endpoint():
        """
        Health check endpoint that verifies database connectivity.

        Returns 200 if healthy, 503 if unhealthy.
        """
        from fastapi.responses import JSONResponse

        health = await app.state.memory.health_check()
        status_code = 200 if health.get("status") == "healthy" else 503
        return JSONResponse(content=health, status_code=status_code)

    @app.get(
        "/version",
        response_model=VersionResponse,
        summary="Get API version and feature flags",
        description="Returns API version information and enabled feature flags. "
        "Use this to check which capabilities are available in this deployment.",
        tags=["Monitoring"],
        operation_id="get_version",
    )
    async def version_endpoint() -> VersionResponse:
        """
        Get API version and enabled features.

        Returns version info and feature flags that can be used by clients
        to determine which capabilities are available.

        Note: observations flag shows the global default. Individual banks
        may override this setting via bank-specific configuration.
        """
        from hindsight_api import __version__
        from hindsight_api.config import _get_raw_config

        config = _get_raw_config()
        return VersionResponse(
            api_version=__version__,
            features=FeaturesInfo(
                observations=config.enable_observations,
                mcp=config.mcp_enabled,
                worker=config.worker_enabled,
                bank_config_api=config.enable_bank_config_api,
                file_upload_api=config.enable_file_upload_api,
            ),
        )

    @app.get(
        "/metrics",
        summary="Prometheus metrics endpoint",
        description="Exports metrics in Prometheus format for scraping",
        tags=["Monitoring"],
    )
    async def metrics_endpoint():
        """Return Prometheus metrics."""
        from fastapi.responses import Response
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)

    @app.get(
        "/v1/default/banks/{bank_id}/graph",
        response_model=GraphDataResponse,
        summary="Get memory graph data",
        description="Retrieve graph data for visualization, optionally filtered by type (world/experience/opinion).",
        operation_id="get_graph",
        tags=["Memory"],
    )
    async def api_graph(
        bank_id: str,
        type: str | None = None,
        limit: int = 1000,
        q: str | None = None,
        tags: list[str] | None = Query(None),
        tags_match: str = "all_strict",
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get graph data from database, filtered by bank_id and optionally by type."""
        try:
            data = await app.state.memory.get_graph_data(
                bank_id, type, limit=limit, q=q, tags=tags, tags_match=tags_match, request_context=request_context
            )
            return data
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/graph: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/memories/list",
        response_model=ListMemoryUnitsResponse,
        summary="List memory units",
        description="List memory units with pagination and optional full-text search. Supports filtering by type. Results are sorted by most recent first (mentioned_at DESC, then created_at DESC).",
        operation_id="list_memories",
        tags=["Memory"],
    )
    async def api_list(
        bank_id: str,
        type: str | None = None,
        q: str | None = None,
        consolidation_state: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """
        List memory units for table view with optional full-text search.

        Results are ordered by most recent first, using mentioned_at timestamp
        (when the memory was mentioned/learned), falling back to created_at.

        Args:
            bank_id: Memory Bank ID (from path)
            type: Filter by fact type (world, experience, opinion)
            q: Search query for full-text search (searches text and context)
            consolidation_state: Filter by consolidation state for source memories
                (world/experience). One of 'failed', 'pending', or 'done'.
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_memory_units(
                bank_id=bank_id,
                fact_type=type,
                search_query=q,
                consolidation_state=consolidation_state,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return data
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories/list: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/memories/{memory_id}",
        summary="Get memory unit",
        description="Get a single memory unit by ID with all its metadata including entities and tags. Note: the 'history' field is deprecated and always returns an empty list - use GET /memories/{memory_id}/history instead.",
        operation_id="get_memory",
        tags=["Memory"],
    )
    async def api_get_memory(
        bank_id: str,
        memory_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get a single memory unit by ID."""
        try:
            data = await app.state.memory.get_memory_unit(
                bank_id=bank_id,
                memory_id=memory_id,
                request_context=request_context,
            )
            if data is None:
                raise HTTPException(status_code=404, detail=f"Memory unit '{memory_id}' not found")
            return data
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories/{memory_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/memories/{memory_id}/history",
        summary="Get observation history",
        description="Get the full history of an observation, with each change's source facts resolved to their text.",
        operation_id="get_observation_history",
        tags=["Memory"],
    )
    async def api_get_observation_history(
        bank_id: str,
        memory_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get the history of a single observation by ID."""
        try:
            data = await app.state.memory.get_observation_history(
                bank_id=bank_id,
                memory_id=memory_id,
                request_context=request_context,
            )
            if data is None:
                raise HTTPException(status_code=404, detail=f"Memory unit '{memory_id}' not found")
            return data
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories/{memory_id}/history: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/memories/recall",
        response_model=RecallResponse,
        summary="Recall memory",
        description="Recall memory using semantic similarity and spreading activation.\n\n"
        "The type parameter is optional and must be one of:\n"
        "- `world`: General knowledge about people, places, events, and things that happen\n"
        "- `experience`: Memories about experience, conversations, actions taken, and tasks performed",
        operation_id="recall_memories",
        tags=["Memory"],
    )
    @audited("recall")
    async def api_recall(
        bank_id: str, request: RecallRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Run a recall and return results with trace."""
        import time

        handler_start = time.time()
        metrics = get_metrics_collector()

        # Validate query length to prevent expensive operations on oversized queries
        max_query_tokens = get_config().recall_max_query_tokens
        encoding = _get_tiktoken_encoding()
        query_tokens = len(encoding.encode(request.query))
        if query_tokens > max_query_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"Query too long: {query_tokens} tokens exceeds maximum of {max_query_tokens}. Please shorten your query.",
            )

        try:
            # Default to world and experience if not specified (exclude observation)
            fact_types = request.types if request.types else list(VALID_RECALL_FACT_TYPES)

            # Parse query_timestamp if provided
            question_date = None
            if request.query_timestamp:
                try:
                    question_date = datetime.fromisoformat(request.query_timestamp.replace("Z", "+00:00"))
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid query_timestamp format. Expected ISO format (e.g., '2023-05-30T23:40:00'): {str(e)}",
                    )

            # Determine entity inclusion settings
            include_entities = request.include.entities is not None
            max_entity_tokens = request.include.entities.max_tokens if include_entities else 500

            # Determine chunk inclusion settings
            include_chunks = request.include.chunks is not None
            max_chunk_tokens = request.include.chunks.max_tokens if include_chunks else 8192

            # Determine source facts inclusion settings
            include_source_facts = request.include.source_facts is not None
            max_source_facts_tokens = request.include.source_facts.max_tokens if include_source_facts else 4096
            max_source_facts_tokens_per_observation = (
                request.include.source_facts.max_tokens_per_observation if include_source_facts else -1
            )

            pre_recall = time.time() - handler_start
            # Run recall with tracing (record metrics)
            with metrics.record_operation(
                "recall", bank_id=bank_id, source="api", budget=request.budget.value, max_tokens=request.max_tokens
            ):
                recall_start = time.time()
                core_result = await app.state.memory.recall_async(
                    bank_id=bank_id,
                    query=request.query,
                    budget=request.budget,
                    max_tokens=request.max_tokens,
                    enable_trace=request.trace,
                    fact_type=fact_types,
                    question_date=question_date,
                    include_entities=include_entities,
                    max_entity_tokens=max_entity_tokens,
                    include_chunks=include_chunks,
                    max_chunk_tokens=max_chunk_tokens,
                    include_source_facts=include_source_facts,
                    max_source_facts_tokens=max_source_facts_tokens,
                    max_source_facts_tokens_per_observation=max_source_facts_tokens_per_observation,
                    request_context=request_context,
                    tags=request.tags,
                    tags_match=request.tags_match,
                    tag_groups=request.tag_groups,
                )

            # Convert core MemoryFact objects to API RecallResult objects (excluding internal metrics)
            def _fact_to_result(fact: "MemoryFact") -> RecallResult:
                return RecallResult(
                    id=fact.id,
                    text=fact.text,
                    type=fact.fact_type,
                    entities=fact.entities,
                    context=fact.context,
                    occurred_start=fact.occurred_start,
                    occurred_end=fact.occurred_end,
                    mentioned_at=fact.mentioned_at,
                    document_id=fact.document_id,
                    metadata=fact.metadata,
                    chunk_id=fact.chunk_id,
                    tags=fact.tags,
                    source_fact_ids=fact.source_fact_ids,
                )

            recall_results = [_fact_to_result(fact) for fact in core_result.results]

            # Convert chunks from engine to HTTP API format
            chunks_response = None
            if core_result.chunks:
                chunks_response = {}
                for chunk_id, chunk_info in core_result.chunks.items():
                    chunks_response[chunk_id] = ChunkData(
                        id=chunk_id,
                        text=chunk_info.chunk_text,
                        chunk_index=chunk_info.chunk_index,
                        truncated=chunk_info.truncated,
                    )

            # Convert core EntityState objects to API EntityStateResponse objects
            entities_response = None
            if core_result.entities:
                entities_response = {}
                for name, state in core_result.entities.items():
                    entities_response[name] = EntityStateResponse(
                        entity_id=state.entity_id,
                        canonical_name=state.canonical_name,
                        observations=[
                            EntityObservationResponse(text=obs.text, mentioned_at=obs.mentioned_at)
                            for obs in state.observations
                        ],
                    )

            # Convert source facts dict to API format
            source_facts_response = None
            if core_result.source_facts:
                source_facts_response = {
                    fact_id: _fact_to_result(fact) for fact_id, fact in core_result.source_facts.items()
                }

            response = RecallResponse(
                results=recall_results,
                trace=core_result.trace,
                entities=entities_response,
                chunks=chunks_response,
                source_facts=source_facts_response,
            )

            handler_duration = time.time() - handler_start
            recall_duration = time.time() - recall_start
            post_recall = handler_duration - pre_recall - recall_duration
            if handler_duration > 1.0:
                logging.info(
                    f"[RECALL HTTP] bank={bank_id} handler_total={handler_duration:.3f}s "
                    f"pre={pre_recall:.3f}s recall={recall_duration:.3f}s post={post_recall:.3f}s "
                    f"results={len(recall_results)} entities={len(entities_response) if entities_response else 0}"
                )

            return response
        except HTTPException:
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except (asyncio.TimeoutError, TimeoutError):
            handler_duration = time.time() - handler_start
            logger.error(
                f"[RECALL TIMEOUT] bank={bank_id} handler_duration={handler_duration:.3f}s - database query timed out"
            )
            raise HTTPException(
                status_code=504,
                detail="Request timed out while searching memories. Try a shorter or more specific query.",
            )
        except Exception as e:
            import traceback

            handler_duration = time.time() - handler_start
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(
                f"[RECALL ERROR] bank={bank_id} handler_duration={handler_duration:.3f}s error={str(e)}\n{error_detail}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/reflect",
        response_model=ReflectResponse,
        summary="Reflect and generate answer",
        description="Reflect and formulate an answer using bank identity, world facts, and opinions.\n\n"
        "This endpoint:\n"
        "1. Retrieves experience (conversations and events)\n"
        "2. Retrieves world facts relevant to the query\n"
        "3. Retrieves existing opinions (bank's perspectives)\n"
        "4. Uses LLM to formulate a contextual answer\n"
        "5. Returns plain text answer and the facts used",
        operation_id="reflect",
        tags=["Memory"],
    )
    @audited("reflect")
    async def api_reflect(
        bank_id: str, request: ReflectRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        metrics = get_metrics_collector()

        try:
            # Handle deprecated context field by concatenating with query
            query = request.query
            if request.context:
                query = f"{request.query}\n\nAdditional context: {request.context}"

            # Use the memory system's reflect_async method (record metrics)
            with metrics.record_operation("reflect", bank_id=bank_id, source="api", budget=request.budget.value):
                core_result = await app.state.memory.reflect_async(
                    bank_id=bank_id,
                    query=query,
                    budget=request.budget,
                    context=None,  # Deprecated, now concatenated with query
                    max_tokens=request.max_tokens,
                    response_schema=request.response_schema,
                    request_context=request_context,
                    tags=request.tags,
                    tags_match=request.tags_match,
                    tag_groups=request.tag_groups,
                    fact_types=request.fact_types,
                    exclude_mental_models=request.exclude_mental_models,
                    exclude_mental_model_ids=request.exclude_mental_model_ids,
                )

            # Build based_on (memories + mental_models + directives) if facts are requested
            based_on_result: ReflectBasedOn | None = None
            if request.include.facts is not None:
                memories = []
                mental_models = []
                directives = []
                for fact_type, facts in core_result.based_on.items():
                    if fact_type == "directives":
                        # Directives are dicts with id, name, content (not MemoryFact objects)
                        for directive in facts:
                            directives.append(
                                ReflectDirective(
                                    id=directive["id"],
                                    name=directive["name"],
                                    content=directive["content"],
                                )
                            )
                    elif fact_type == "mental-models":
                        # Mental models are MemoryFact with type "mental-models" (note: hyphen, not underscore)
                        for fact in facts:
                            mental_models.append(
                                ReflectMentalModel(
                                    id=fact.id,
                                    text=fact.text,
                                    context=fact.context,
                                )
                            )
                    else:
                        for fact in facts:
                            memories.append(
                                ReflectFact(
                                    id=fact.id,
                                    text=fact.text,
                                    type=fact.fact_type,
                                    context=fact.context,
                                    occurred_start=fact.occurred_start,
                                    occurred_end=fact.occurred_end,
                                )
                            )
                based_on_result = ReflectBasedOn(memories=memories, mental_models=mental_models, directives=directives)

            # Build trace (tool_calls + llm_calls + observations) if tool_calls is requested
            trace_result: ReflectTrace | None = None
            if request.include.tool_calls is not None:
                include_output = request.include.tool_calls.output
                tool_calls = [
                    ReflectToolCall(
                        tool=tc.tool,
                        input=tc.input,
                        output=tc.output if include_output else None,
                        duration_ms=tc.duration_ms,
                        iteration=tc.iteration,
                    )
                    for tc in core_result.tool_trace
                ]
                llm_calls = [ReflectLLMCall(scope=lc.scope, duration_ms=lc.duration_ms) for lc in core_result.llm_trace]
                trace_result = ReflectTrace(
                    tool_calls=tool_calls,
                    llm_calls=llm_calls,
                )

            return ReflectResponse(
                text=core_result.text,
                based_on=based_on_result,
                structured_output=core_result.structured_output,
                usage=core_result.usage,
                trace=trace_result,
            )

        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except LLMNotAvailableError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except TimeoutError as e:
            logger.error("Timeout in /v1/default/banks/%s/reflect: %s", bank_id, e)
            raise HTTPException(
                status_code=504,
                detail=str(e) or "Reflect operation timed out. Consider reducing the budget or simplifying the query.",
            )
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/reflect: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks",
        response_model=BankListResponse,
        summary="List all memory banks",
        description="Get a list of all agents with their profiles",
        operation_id="list_banks",
        tags=["Banks"],
    )
    async def api_list_banks(request_context: RequestContext = Depends(get_request_context)):
        """Get list of all banks with their profiles."""
        try:
            banks = await app.state.memory.list_banks(request_context=request_context)
            return BankListResponse(banks=banks)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/stats",
        response_model=BankStatsResponse,
        summary="Get statistics for memory bank",
        description="Get statistics about nodes and links for a specific agent",
        operation_id="get_agent_stats",
        tags=["Banks"],
    )
    async def api_stats(
        bank_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get statistics about memory nodes and links for a memory bank."""
        try:
            stats = await app.state.memory.get_bank_stats(bank_id, request_context=request_context)
            nodes_by_type = stats["node_counts"]
            links_by_type = stats["link_counts"]
            links_by_fact_type = stats["link_counts_by_fact_type"]
            links_breakdown: dict[str, dict[str, int]] = {}
            for row in stats["link_breakdown"]:
                ft = row["fact_type"]
                if ft not in links_breakdown:
                    links_breakdown[ft] = {}
                links_breakdown[ft][row["link_type"]] = row["count"]
            ops = stats["operations"]
            return BankStatsResponse(
                bank_id=bank_id,
                total_nodes=sum(nodes_by_type.values()),
                total_links=sum(links_by_type.values()),
                total_documents=stats["total_documents"],
                nodes_by_fact_type=nodes_by_type,
                links_by_link_type=links_by_type,
                links_by_fact_type=links_by_fact_type,
                links_breakdown=links_breakdown,
                pending_operations=ops.get("pending", 0),
                failed_operations=ops.get("failed", 0),
                operations_by_status=ops,
                last_consolidated_at=stats["last_consolidated_at"],
                pending_consolidation=stats["pending_consolidation"],
                failed_consolidation=stats.get("failed_consolidation", 0),
                total_observations=stats["total_observations"],
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/stats: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/stats/memories-timeseries",
        response_model=MemoriesTimeseriesResponse,
        summary="Memory ingestion time-series",
        description="Memories ingested over a period, bucketed by time and broken down by fact type.",
        operation_id="get_memories_timeseries",
        tags=["Banks"],
    )
    async def api_memories_timeseries(
        bank_id: str,
        period: str = "7d",
        request_context: RequestContext = Depends(get_request_context),
    ):
        try:
            data = await app.state.memory.get_memories_timeseries(
                bank_id, period=period, request_context=request_context
            )
            return MemoriesTimeseriesResponse(**data)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/stats/memories-timeseries: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/entities",
        response_model=EntityListResponse,
        summary="List entities",
        description="List all entities (people, organizations, etc.) known by the bank, ordered by mention count. Supports pagination.",
        operation_id="list_entities",
        tags=["Entities"],
    )
    async def api_list_entities(
        bank_id: str,
        limit: int = Query(default=100, description="Maximum number of entities to return"),
        offset: int = Query(default=0, description="Offset for pagination"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List entities for a memory bank with pagination."""
        try:
            data = await app.state.memory.list_entities(
                bank_id, limit=limit, offset=offset, request_context=request_context
            )
            return EntityListResponse(
                items=[EntityListItem(**e) for e in data["items"]],
                total=data["total"],
                limit=data["limit"],
                offset=data["offset"],
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/entities: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/entities/graph",
        response_model=EntityGraphResponse,
        summary="Get entity co-occurrence graph",
        description="Return a graph of entities (nodes) and their co-occurrences (edges) for visualization.",
        operation_id="get_entity_graph",
        tags=["Entities"],
    )
    async def api_entity_graph(
        bank_id: str,
        limit: int = Query(default=1000, description="Maximum number of co-occurrence edges to return"),
        min_count: int = Query(default=1, description="Minimum cooccurrence_count to include an edge"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Return entity co-occurrence graph for a bank."""
        try:
            return await app.state.memory.get_entity_graph(
                bank_id, limit=limit, min_count=min_count, request_context=request_context
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/entities/graph: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/entities/{entity_id}",
        response_model=EntityDetailResponse,
        summary="Get entity details",
        description="Get detailed information about an entity including observations (mental model).",
        operation_id="get_entity",
        tags=["Entities"],
    )
    async def api_get_entity(
        bank_id: str, entity_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """Get entity details with observations."""
        try:
            entity = await app.state.memory.get_entity(bank_id, entity_id, request_context=request_context)

            if entity is None:
                raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

            return EntityDetailResponse(
                id=entity["id"],
                canonical_name=entity["canonical_name"],
                mention_count=entity["mention_count"],
                first_seen=entity["first_seen"],
                last_seen=entity["last_seen"],
                metadata=_parse_metadata(entity["metadata"]),
                observations=[
                    EntityObservationResponse(text=obs.text, mentioned_at=obs.mentioned_at)
                    for obs in entity["observations"]
                ],
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/entities/{entity_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/entities/{entity_id}/regenerate",
        response_model=EntityDetailResponse,
        summary="Regenerate entity observations (deprecated)",
        description="This endpoint is deprecated. Entity observations have been replaced by mental models.",
        operation_id="regenerate_entity_observations",
        tags=["Entities"],
        deprecated=True,
    )
    async def api_regenerate_entity_observations(
        bank_id: str,
        entity_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Regenerate observations for an entity. DEPRECATED."""
        raise HTTPException(
            status_code=410,
            detail="This endpoint is deprecated. Entity observations are no longer supported.",
        )

    # =========================================================================
    # =========================================================================
    # MENTAL MODELS ENDPOINTS (stored reflect responses)
    # =========================================================================

    @app.get(
        "/v1/default/banks/{bank_id}/mental-models",
        response_model=MentalModelListResponse,
        summary="List mental models",
        description="List user-curated living documents that stay current.",
        operation_id="list_mental_models",
        tags=["Mental Models"],
    )
    async def api_list_mental_models(
        bank_id: str,
        tags_filter: list[str] | None = Query(None, alias="tags", description="Filter by tags"),
        tags_match: Literal["any", "all", "exact"] = Query("any", description="How to match tags"),
        detail: Literal["metadata", "content", "full"] = Query(
            "full",
            description="Detail level: 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response)",
        ),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List mental models for a bank."""
        try:
            mental_models = await app.state.memory.list_mental_models(
                bank_id=bank_id,
                tags=tags_filter,
                tags_match=tags_match,
                detail=detail,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return MentalModelListResponse(items=[MentalModelResponse(**m) for m in mental_models])
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/mental-models: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/mental-models/{mental_model_id}",
        response_model=MentalModelResponse,
        summary="Get mental model",
        description="Get a specific mental model by ID.",
        operation_id="get_mental_model",
        tags=["Mental Models"],
    )
    async def api_get_mental_model(
        bank_id: str,
        mental_model_id: str,
        detail: Literal["metadata", "content", "full"] = Query(
            "full",
            description="Detail level: 'metadata' (names/tags only), 'content' (adds content/config), 'full' (includes reflect_response)",
        ),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get a mental model by ID."""
        try:
            mental_model = await app.state.memory.get_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                detail=detail,
                request_context=request_context,
            )
            if mental_model is None:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")

            return MentalModelResponse(**mental_model)
        except (AuthenticationError, HTTPException):
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/mental-models/{mental_model_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/mental-models/{mental_model_id}/history",
        summary="Get mental model history",
        description="Get the refresh history of a mental model, showing content changes over time.",
        operation_id="get_mental_model_history",
        tags=["Mental Models"],
    )
    async def api_get_mental_model_history(
        bank_id: str,
        mental_model_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get the refresh history of a mental model."""
        try:
            data = await app.state.memory.get_mental_model_history(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            if data is None:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")
            return data
        except (AuthenticationError, HTTPException):
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(
                f"Error in GET /v1/default/banks/{bank_id}/mental-models/{mental_model_id}/history: {error_detail}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/mental-models",
        response_model=CreateMentalModelResponse,
        summary="Create mental model",
        description="Create a mental model by running reflect with the source query in the background. "
        "Returns an operation ID to track progress. The content is auto-generated by the reflect endpoint. "
        "Use the operations endpoint to check completion status.",
        operation_id="create_mental_model",
        tags=["Mental Models"],
    )
    @audited("create_mental_model", request_param="body")
    async def api_create_mental_model(
        bank_id: str,
        body: CreateMentalModelRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Create a mental model (async - returns operation_id)."""
        try:
            # 1. Create the mental model with placeholder content
            mental_model = await app.state.memory.create_mental_model(
                bank_id=bank_id,
                name=body.name,
                source_query=body.source_query,
                content="Generating content...",
                mental_model_id=body.id if body.id else None,
                tags=body.tags if body.tags else None,
                max_tokens=body.max_tokens,
                trigger=body.trigger.model_dump() if body.trigger else None,
                request_context=request_context,
            )
            # 2. Schedule a refresh to generate the actual content
            result = await app.state.memory.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model["id"],
                request_context=request_context,
            )
            return CreateMentalModelResponse(mental_model_id=mental_model["id"], operation_id=result["operation_id"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/mental-models: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/mental-models/{mental_model_id}/refresh",
        response_model=AsyncOperationSubmitResponse,
        summary="Refresh mental model",
        description="Submit an async task to re-run the source query through reflect and update the content.",
        operation_id="refresh_mental_model",
        tags=["Mental Models"],
    )
    @audited("refresh_mental_model", request_param=None)
    async def api_refresh_mental_model(
        bank_id: str,
        mental_model_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Refresh a mental model by re-running its source query (async)."""
        try:
            result = await app.state.memory.submit_async_refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            return AsyncOperationSubmitResponse(operation_id=result["operation_id"], status="queued")
        except LLMNotAvailableError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(
                f"Error in POST /v1/default/banks/{bank_id}/mental-models/{mental_model_id}/refresh: {error_detail}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}/mental-models/{mental_model_id}",
        response_model=MentalModelResponse,
        summary="Update mental model",
        description="Update a mental model's name and/or source query.",
        operation_id="update_mental_model",
        tags=["Mental Models"],
    )
    @audited("update_mental_model", request_param="body")
    async def api_update_mental_model(
        bank_id: str,
        mental_model_id: str,
        body: UpdateMentalModelRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Update a mental model."""
        try:
            mental_model = await app.state.memory.update_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                name=body.name,
                source_query=body.source_query,
                max_tokens=body.max_tokens,
                tags=body.tags,
                trigger=body.trigger.model_dump() if body.trigger else None,
                request_context=request_context,
            )
            if mental_model is None:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")
            return MentalModelResponse(**mental_model)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}/mental-models/{mental_model_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/mental-models/{mental_model_id}",
        summary="Delete mental model",
        description="Delete a mental model.",
        operation_id="delete_mental_model",
        tags=["Mental Models"],
    )
    @audited("delete_mental_model", request_param=None)
    async def api_delete_mental_model(
        bank_id: str,
        mental_model_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Delete a mental model."""
        try:
            deleted = await app.state.memory.delete_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")
            return {"status": "deleted"}
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/mental-models/{mental_model_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # DIRECTIVES ENDPOINTS
    # =========================================================================

    @app.get(
        "/v1/default/banks/{bank_id}/directives",
        response_model=DirectiveListResponse,
        summary="List directives",
        description="List hard rules that are injected into prompts.",
        operation_id="list_directives",
        tags=["Directives"],
    )
    async def api_list_directives(
        bank_id: str,
        tags_filter: list[str] | None = Query(None, alias="tags", description="Filter by tags"),
        tags_match: Literal["any", "all", "exact"] = Query("any", description="How to match tags"),
        active_only: bool = Query(True, description="Only return active directives"),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List directives for a bank."""
        try:
            directives = await app.state.memory.list_directives(
                bank_id=bank_id,
                tags=tags_filter,
                tags_match=tags_match,
                active_only=active_only,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return DirectiveListResponse(items=[DirectiveResponse(**d) for d in directives])
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/directives: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/directives/{directive_id}",
        response_model=DirectiveResponse,
        summary="Get directive",
        description="Get a specific directive by ID.",
        operation_id="get_directive",
        tags=["Directives"],
    )
    async def api_get_directive(
        bank_id: str,
        directive_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get a directive by ID."""
        try:
            directive = await app.state.memory.get_directive(
                bank_id=bank_id,
                directive_id=directive_id,
                request_context=request_context,
            )
            if directive is None:
                raise HTTPException(status_code=404, detail=f"Directive '{directive_id}' not found")
            return DirectiveResponse(**directive)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/directives/{directive_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/directives",
        response_model=DirectiveResponse,
        summary="Create directive",
        description="Create a hard rule that will be injected into prompts.",
        operation_id="create_directive",
        tags=["Directives"],
    )
    @audited("create_directive", request_param="body")
    async def api_create_directive(
        bank_id: str,
        body: CreateDirectiveRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Create a directive."""
        try:
            directive = await app.state.memory.create_directive(
                bank_id=bank_id,
                name=body.name,
                content=body.content,
                priority=body.priority,
                is_active=body.is_active,
                tags=body.tags,
                request_context=request_context,
            )
            return DirectiveResponse(**directive)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/directives: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}/directives/{directive_id}",
        response_model=DirectiveResponse,
        summary="Update directive",
        description="Update a directive's properties.",
        operation_id="update_directive",
        tags=["Directives"],
    )
    @audited("update_directive", request_param="body")
    async def api_update_directive(
        bank_id: str,
        directive_id: str,
        body: UpdateDirectiveRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Update a directive."""
        try:
            directive = await app.state.memory.update_directive(
                bank_id=bank_id,
                directive_id=directive_id,
                name=body.name,
                content=body.content,
                priority=body.priority,
                is_active=body.is_active,
                tags=body.tags,
                request_context=request_context,
            )
            if directive is None:
                raise HTTPException(status_code=404, detail=f"Directive '{directive_id}' not found")
            return DirectiveResponse(**directive)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}/directives/{directive_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/directives/{directive_id}",
        summary="Delete directive",
        description="Delete a directive.",
        operation_id="delete_directive",
        tags=["Directives"],
    )
    @audited("delete_directive", request_param=None)
    async def api_delete_directive(
        bank_id: str,
        directive_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Delete a directive."""
        try:
            deleted = await app.state.memory.delete_directive(
                bank_id=bank_id,
                directive_id=directive_id,
                request_context=request_context,
            )
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Directive '{directive_id}' not found")
            return {"status": "deleted"}
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/directives/{directive_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/documents",
        response_model=ListDocumentsResponse,
        summary="List documents",
        description="List documents with pagination and optional search. Documents are the source content from which memory units are extracted.",
        operation_id="list_documents",
        tags=["Documents"],
    )
    async def api_list_documents(
        bank_id: str,
        q: str | None = Query(
            None, description="Case-insensitive substring filter on document ID (e.g. 'report' matches 'report-2024')"
        ),
        tags: list[str] | None = Query(None, description="Filter documents by tags"),
        tags_match: str = Query(
            "any_strict", description="How to match tags: 'any', 'all', 'any_strict', 'all_strict'"
        ),
        limit: int = 100,
        offset: int = 0,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """
        List documents for a memory bank with optional search.

        Args:
            bank_id: Memory Bank ID (from path)
            q: Case-insensitive substring filter on document ID
            tags: Filter documents by tags
            tags_match: How to match tags (any, all, any_strict, all_strict)
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_documents(
                bank_id=bank_id,
                search_query=q,
                tags=tags,
                tags_match=tags_match,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return data
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/documents: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/documents/{document_id:path}",
        response_model=DocumentResponse,
        summary="Get document details",
        description="Get a specific document including its original text",
        operation_id="get_document",
        tags=["Documents"],
    )
    async def api_get_document(
        bank_id: str, document_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """
        Get a specific document with its original text.

        Args:
            bank_id: Memory Bank ID (from path)
            document_id: Document ID (from path)
        """
        try:
            document = await app.state.memory.get_document(document_id, bank_id, request_context=request_context)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            return document
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/tags",
        response_model=ListTagsResponse,
        summary="List tags",
        description="List all unique tags in a memory bank with usage counts. "
        "Supports wildcard search using '*' (e.g., 'user:*', '*-fred', 'tag*-2'). Case-insensitive.",
        operation_id="list_tags",
        tags=["Memory"],
    )
    async def api_list_tags(
        bank_id: str,
        q: str | None = Query(
            default=None,
            description="Wildcard pattern to filter tags (e.g., 'user:*' for user:alice, '*-admin' for role-admin). "
            "Use '*' as wildcard. Case-insensitive.",
        ),
        limit: int = Query(default=100, description="Maximum number of tags to return"),
        offset: int = Query(default=0, description="Offset for pagination"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """
        List all unique tags in a memory bank.

        Use this endpoint to discover available tags or expand wildcard patterns.
        Supports '*' wildcards for flexible matching (case-insensitive):
        - 'user:*' matches user:alice, user:bob
        - '*-admin' matches role-admin, super-admin
        - 'env*-prod' matches env-prod, environment-prod

        Args:
            bank_id: Memory Bank ID (from path)
            q: Wildcard pattern to filter tags (use '*' as wildcard)
            limit: Maximum number of tags to return (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_tags(
                bank_id=bank_id,
                pattern=q,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return data
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/tags: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/chunks/{chunk_id:path}",
        response_model=ChunkResponse,
        summary="Get chunk details",
        description="Get a specific chunk by its ID",
        operation_id="get_chunk",
        tags=["Documents"],
    )
    async def api_get_chunk(chunk_id: str, request_context: RequestContext = Depends(get_request_context)):
        """
        Get a specific chunk with its text.

        Args:
            chunk_id: Chunk ID (from path, format: bank_id_document_id_chunk_index)
        """
        try:
            chunk = await app.state.memory.get_chunk(chunk_id, request_context=request_context)
            if not chunk:
                raise HTTPException(status_code=404, detail="Chunk not found")
            return chunk
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/chunks/{chunk_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}/documents/{document_id:path}",
        response_model=UpdateDocumentResponse,
        summary="Update document",
        description="Update mutable fields on a document without re-processing its content.\n\n"
        "**Tags** (`tags`): Propagated to all associated memory units. Observations derived from "
        "those units are invalidated and queued for re-consolidation under the new tags. "
        "Co-source memories from other documents that shared those observations are also reset.\n\n"
        "At least one field must be provided.",
        operation_id="update_document",
        tags=["Documents"],
    )
    @audited("update_document", request_param="body")
    async def api_update_document(
        bank_id: str,
        document_id: str,
        body: UpdateDocumentRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """
        Update mutable fields on a document without re-processing its content.

        Args:
            bank_id: Memory Bank ID (from path)
            document_id: Document ID (from path)
            body: Fields to update (tags, metadata, context)
        """
        if body.tags is None:
            raise HTTPException(status_code=422, detail="At least one field (tags) must be provided")
        try:
            result = await app.state.memory.update_document(
                document_id,
                bank_id,
                tags=body.tags,
                request_context=request_context,
            )
            if not result:
                raise HTTPException(status_code=404, detail="Document not found")
            return UpdateDocumentResponse(success=True)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/documents/{document_id:path}",
        response_model=DeleteDocumentResponse,
        summary="Delete a document",
        description="Delete a document and all its associated memory units and links.\n\n"
        "This will cascade delete:\n"
        "- The document itself\n"
        "- All memory units extracted from this document\n"
        "- All links (temporal, semantic, entity) associated with those memory units\n\n"
        "This operation cannot be undone.",
        operation_id="delete_document",
        tags=["Documents"],
    )
    @audited("delete_document", request_param=None)
    async def api_delete_document(
        bank_id: str, document_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """
        Delete a document and all its associated memory units and links.

        Args:
            bank_id: Memory Bank ID (from path)
            document_id: Document ID to delete (from path)
        """
        try:
            result = await app.state.memory.delete_document(document_id, bank_id, request_context=request_context)

            if result["document_deleted"] == 0:
                raise HTTPException(status_code=404, detail="Document not found")

            return DeleteDocumentResponse(
                success=True,
                message=f"Document '{document_id}' and {result['memory_units_deleted']} associated memory units deleted successfully",
                document_id=document_id,
                memory_units_deleted=result["memory_units_deleted"],
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/operations",
        response_model=OperationsListResponse,
        summary="List async operations",
        description="Get a list of async operations for a specific agent, with optional filtering by status and operation type. Results are sorted by most recent first.",
        operation_id="list_operations",
        tags=["Operations"],
    )
    async def api_list_operations(
        bank_id: str,
        status: str | None = Query(default=None, description="Filter by status: pending, completed, or failed"),
        type: str | None = Query(
            default=None,
            description="Filter by operation type: retain, consolidation, refresh_mental_model, file_convert_retain, webhook_delivery",
        ),
        limit: int = Query(default=20, ge=1, le=100, description="Maximum number of operations to return"),
        offset: int = Query(default=0, ge=0, description="Number of operations to skip"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List async operations for a memory bank with optional filtering and pagination."""
        try:
            result = await app.state.memory.list_operations(
                bank_id, status=status, task_type=type, limit=limit, offset=offset, request_context=request_context
            )
            return OperationsListResponse(
                bank_id=bank_id,
                total=result["total"],
                limit=limit,
                offset=offset,
                operations=[OperationResponse(**op) for op in result["operations"]],
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/operations: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/operations/{operation_id}",
        response_model=OperationStatusResponse,
        summary="Get operation status",
        description="Get the status of a specific async operation. Returns 'pending', 'completed', or 'failed'. "
        "Completed operations are removed from storage, so 'completed' means the operation finished successfully.",
        operation_id="get_operation_status",
        tags=["Operations"],
    )
    async def api_get_operation_status(
        bank_id: str,
        operation_id: str,
        include_payload: bool = Query(
            default=False,
            description="Include the raw task payload (submission params) in the response. May be large.",
        ),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get the status of an async operation."""
        try:
            # Validate UUID format
            try:
                uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            result = await app.state.memory.get_operation_status(
                bank_id, operation_id, request_context=request_context, include_payload=include_payload
            )
            return OperationStatusResponse(**result)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/operations/{operation_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/operations/{operation_id}",
        response_model=CancelOperationResponse,
        summary="Cancel a pending async operation",
        description="Cancel a pending async operation by removing it from the queue",
        operation_id="cancel_operation",
        tags=["Operations"],
    )
    @audited("cancel_operation", request_param=None)
    async def api_cancel_operation(
        bank_id: str, operation_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """Cancel a pending async operation."""
        try:
            # Validate UUID format
            try:
                uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            result = await app.state.memory.cancel_operation(bank_id, operation_id, request_context=request_context)
            return CancelOperationResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/operations/{operation_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/operations/{operation_id}/retry",
        response_model=RetryOperationResponse,
        summary="Retry a failed async operation",
        description="Re-queue a failed async operation so the worker picks it up again",
        operation_id="retry_operation",
        tags=["Operations"],
    )
    @audited("retry_operation", request_param=None)
    async def api_retry_operation(
        bank_id: str, operation_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """Retry a failed async operation."""
        try:
            try:
                uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            result = await app.state.memory.retry_operation(bank_id, operation_id, request_context=request_context)
            return RetryOperationResponse(**result)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/operations/{operation_id}/retry: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/profile",
        response_model=BankProfileResponse,
        summary="Get memory bank profile",
        description="Get disposition traits and mission for a memory bank. Auto-creates agent with defaults if not exists.",
        operation_id="get_bank_profile",
        tags=["Banks"],
        deprecated=True,
    )
    async def api_get_bank_profile(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Get memory bank profile (disposition + mission)."""
        try:
            profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            # Convert DispositionTraits object to dict for Pydantic
            disposition_dict = (
                profile["disposition"].model_dump()
                if hasattr(profile["disposition"], "model_dump")
                else dict(profile["disposition"])
            )
            mission = profile.get("mission") or ""
            return BankProfileResponse(
                bank_id=bank_id,
                name=profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                mission=mission,
                background=mission,  # Backwards compat
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/profile: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put(
        "/v1/default/banks/{bank_id}/profile",
        response_model=BankProfileResponse,
        summary="Update memory bank disposition",
        description="Update bank's disposition traits (skepticism, literalism, empathy)",
        operation_id="update_bank_disposition",
        tags=["Banks"],
        deprecated=True,
    )
    async def api_update_bank_disposition(
        bank_id: str, request: UpdateDispositionRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Update bank disposition traits."""
        try:
            # Update disposition
            await app.state.memory.update_bank_disposition(
                bank_id, request.disposition.model_dump(), request_context=request_context
            )

            # Get updated profile
            profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            disposition_dict = (
                profile["disposition"].model_dump()
                if hasattr(profile["disposition"], "model_dump")
                else dict(profile["disposition"])
            )
            mission = profile.get("mission") or ""
            return BankProfileResponse(
                bank_id=bank_id,
                name=profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                mission=mission,
                background=mission,  # Backwards compat
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/profile: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/background",
        response_model=BackgroundResponse,
        summary="Add/merge memory bank background (deprecated)",
        description="Deprecated: Use PUT /mission instead. This endpoint now updates the mission field.",
        operation_id="add_bank_background",
        tags=["Banks"],
        deprecated=True,
    )
    async def api_add_bank_background(
        bank_id: str, request: AddBackgroundRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Deprecated: Add or merge bank background. Now updates mission field."""
        try:
            result = await app.state.memory.merge_bank_mission(
                bank_id, request.content, request_context=request_context
            )
            mission = result.get("mission") or ""
            return BackgroundResponse(mission=mission, background=mission)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/background: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put(
        "/v1/default/banks/{bank_id}",
        response_model=BankProfileResponse,
        summary="Create or update memory bank",
        description="Create a new agent or update existing agent with disposition and mission. Auto-fills missing fields with defaults.",
        operation_id="create_or_update_bank",
        tags=["Banks"],
    )
    @audited("create_bank")
    async def api_create_or_update_bank(
        bank_id: str, request: CreateBankRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Create or update an agent with disposition and mission."""
        try:
            # Ensure bank exists by getting profile (auto-creates with defaults)
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name if provided (stored in DB for display only, deprecated)
            if request.name is not None:
                await app.state.memory.update_bank(
                    bank_id,
                    name=request.name,
                    request_context=request_context,
                )

            # Apply all config overrides (includes reflect_mission, disposition, retain settings)
            config_updates = request.get_config_updates()
            if config_updates:
                await app.state.memory._config_resolver.update_bank_config(bank_id, config_updates, request_context)

            # Get final profile
            final_profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            disposition_dict = (
                final_profile["disposition"].model_dump()
                if hasattr(final_profile["disposition"], "model_dump")
                else dict(final_profile["disposition"])
            )
            mission = final_profile.get("mission") or ""
            return BankProfileResponse(
                bank_id=bank_id,
                name=final_profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                mission=mission,
                background=mission,  # Backwards compat
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}",
        response_model=BankProfileResponse,
        summary="Partial update memory bank",
        description="Partially update an agent's profile. Only provided fields will be updated.",
        operation_id="update_bank",
        tags=["Banks"],
    )
    @audited("update_bank")
    async def api_update_bank(
        bank_id: str, request: CreateBankRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Partially update an agent's profile (name, mission, disposition)."""
        try:
            # Ensure bank exists
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name if provided (stored in DB for display only, deprecated)
            if request.name is not None:
                await app.state.memory.update_bank(
                    bank_id,
                    name=request.name,
                    request_context=request_context,
                )

            # Apply all config overrides (includes reflect_mission, disposition, retain settings)
            config_updates = request.get_config_updates()
            if config_updates:
                await app.state.memory._config_resolver.update_bank_config(bank_id, config_updates, request_context)

            # Get final profile
            final_profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            disposition_dict = (
                final_profile["disposition"].model_dump()
                if hasattr(final_profile["disposition"], "model_dump")
                else dict(final_profile["disposition"])
            )
            mission = final_profile.get("mission") or ""
            return BankProfileResponse(
                bank_id=bank_id,
                name=final_profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                mission=mission,
                background=mission,  # Backwards compat
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}",
        response_model=DeleteResponse,
        summary="Delete memory bank",
        description="Delete an entire memory bank including all memories, entities, documents, and the bank profile itself. "
        "This is a destructive operation that cannot be undone.",
        operation_id="delete_bank",
        tags=["Banks"],
    )
    @audited("delete_bank", request_param=None)
    async def api_delete_bank(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Delete an entire memory bank and all its data."""
        try:
            result = await app.state.memory.delete_bank(bank_id, request_context=request_context)
            return DeleteResponse(
                success=True,
                message=f"Bank '{bank_id}' and all associated data deleted successfully",
                deleted_count=result.get("memory_units_deleted", 0)
                + result.get("entities_deleted", 0)
                + result.get("documents_deleted", 0),
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    # =====================================================================
    # Bank Template Import / Export
    # =====================================================================

    @app.post(
        "/v1/default/banks/{bank_id}/import",
        response_model=BankTemplateImportResponse,
        summary="Import bank template",
        description="Import a bank template manifest to create or update a bank's configuration, mental models, "
        "and directives. If the bank does not exist it is created. Config fields are applied as per-bank overrides. "
        "Mental models are matched by id, directives by name — existing ones are updated, new ones are created. "
        "Use dry_run=true to validate the manifest without applying changes.",
        operation_id="import_bank_template",
        tags=["Bank Templates"],
    )
    @audited("import_bank_template", request_param=None)
    async def api_import_bank_template(
        bank_id: str,
        request: Request,
        dry_run: bool = Query(default=False, description="Validate only, do not apply changes"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Import a bank template manifest."""
        try:
            # Parse raw JSON and validate against the Pydantic model manually
            # so we can return clean error messages instead of raw 422s.
            raw_body = await request.json()
            from pydantic import ValidationError

            try:
                body = BankTemplateManifest.model_validate(raw_body)
            except ValidationError as e:
                errors = [f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
                raise HTTPException(
                    status_code=400,
                    detail=f"Template schema validation failed: {'; '.join(errors)}",
                )

            # Semantic validation beyond Pydantic structural checks
            validation_errors = validate_bank_template(body)
            if validation_errors:
                raise HTTPException(
                    status_code=400,
                    detail=f"Template validation failed: {'; '.join(validation_errors)}",
                )
            if dry_run:
                return BankTemplateImportResponse(
                    bank_id=bank_id,
                    config_applied=body.bank is not None,
                    mental_models_created=[m.id for m in (body.mental_models or [])],
                    directives_created=[d.name for d in (body.directives or [])],
                    dry_run=True,
                )

            # Ensure bank exists (auto-creates with defaults if needed)
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            return await apply_bank_template_manifest(
                memory=app.state.memory,
                bank_id=bank_id,
                manifest=body,
                request_context=request_context,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/import: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/export",
        response_model=BankTemplateManifest,
        summary="Export bank template",
        description="Export a bank's current configuration, mental models, and directives as a template manifest. "
        "The exported manifest can be imported into another bank to replicate the setup.",
        operation_id="export_bank_template",
        tags=["Bank Templates"],
    )
    async def api_export_bank_template(
        bank_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Export a bank's config and mental models as a template manifest."""
        try:
            # Authenticate and ensure bank exists
            profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            if profile is None:
                raise HTTPException(status_code=404, detail=f"Bank '{bank_id}' not found")

            # Get bank-specific config overrides (not the fully resolved config,
            # so the template only contains what was explicitly set on this bank)
            await app.state.memory._authenticate_tenant(request_context)
            bank_overrides = await app.state.memory._config_resolver._load_bank_config(bank_id)

            # Filter to only BankTemplateConfig fields (exclude credentials, static fields)
            template_config_fields = set(BankTemplateConfig.model_fields.keys())
            filtered_overrides = {k: v for k, v in bank_overrides.items() if k in template_config_fields}
            bank_config = BankTemplateConfig(**filtered_overrides) if filtered_overrides else None

            # Get mental models
            mental_models_raw = await app.state.memory.list_mental_models(
                bank_id=bank_id, request_context=request_context
            )
            template_mental_models: list[BankTemplateMentalModel] = []
            for mm in mental_models_raw:
                trigger_data = mm.get("trigger", {})
                trigger = MentalModelTrigger(**trigger_data) if trigger_data else MentalModelTrigger()
                template_mental_models.append(
                    BankTemplateMentalModel(
                        id=mm["id"],
                        name=mm["name"],
                        source_query=mm["source_query"],
                        tags=mm.get("tags", []),
                        max_tokens=mm.get("max_tokens", 2048),
                        trigger=trigger,
                    )
                )

            # Get directives
            directives_raw = await app.state.memory.list_directives(
                bank_id=bank_id, active_only=False, request_context=request_context
            )
            template_directives: list[BankTemplateDirective] = []
            for d in directives_raw:
                template_directives.append(
                    BankTemplateDirective(
                        name=d["name"],
                        content=d["content"],
                        priority=d.get("priority", 0),
                        is_active=d.get("is_active", True),
                        tags=d.get("tags", []),
                    )
                )

            return BankTemplateManifest(
                version=BANK_TEMPLATE_CURRENT_VERSION,
                bank=bank_config,
                mental_models=template_mental_models if template_mental_models else None,
                directives=template_directives if template_directives else None,
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/export: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/bank-template-schema",
        summary="Get bank template JSON Schema",
        description="Returns the JSON Schema for the bank template manifest format. "
        "Use this to validate template manifests before importing.",
        operation_id="get_bank_template_schema",
        tags=["Bank Templates"],
    )
    async def api_get_bank_template_schema():
        """Return the JSON Schema for the bank template manifest."""
        return BankTemplateManifest.model_json_schema()

    @app.delete(
        "/v1/default/banks/{bank_id}/observations",
        response_model=DeleteResponse,
        summary="Clear all observations",
        description="Delete all observations for a memory bank. This is useful for resetting the consolidated knowledge.",
        operation_id="clear_observations",
        tags=["Banks"],
    )
    @audited("clear_observations", request_param=None)
    async def api_clear_observations(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Clear all observations for a bank."""
        try:
            result = await app.state.memory.clear_observations(bank_id, request_context=request_context)
            return DeleteResponse(
                success=True,
                message=f"Cleared {result.get('deleted_count', 0)} observations",
                deleted_count=result.get("deleted_count", 0),
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/observations: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/consolidation/recover",
        response_model=RecoverConsolidationResponse,
        summary="Recover failed consolidation",
        description=(
            "Reset all memories that were permanently marked as failed during consolidation "
            "(after exhausting all LLM retries and adaptive batch splitting) so they are "
            "picked up again on the next consolidation run. Does not delete any observations."
        ),
        operation_id="recover_consolidation",
        tags=["Banks"],
    )
    @audited("recover_consolidation", request_param=None)
    async def api_recover_consolidation(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Reset consolidation-failed memories for recovery."""
        try:
            result = await app.state.memory.retry_failed_consolidation(bank_id, request_context=request_context)
            return RecoverConsolidationResponse(retried_count=result["retried_count"])
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/consolidation/recover: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/memories/{memory_id}/observations",
        response_model=ClearMemoryObservationsResponse,
        summary="Clear observations for a memory",
        description="Delete all observations derived from a specific memory and reset it for re-consolidation. "
        "The memory itself is not deleted. A consolidation job is triggered automatically so the memory "
        "will produce fresh observations on the next consolidation run.",
        operation_id="clear_memory_observations",
        tags=["Memory"],
    )
    @audited("clear_memory_observations", request_param=None)
    async def api_clear_memory_observations(
        bank_id: str,
        memory_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Clear all observations derived from a specific memory."""
        try:
            result = await app.state.memory.clear_observations_for_memory(
                bank_id=bank_id,
                memory_id=memory_id,
                request_context=request_context,
            )
            return ClearMemoryObservationsResponse(deleted_count=result["deleted_count"])
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(
                f"Error in DELETE /v1/default/banks/{bank_id}/memories/{memory_id}/observations: {error_detail}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/config",
        response_model=BankConfigResponse,
        summary="Get bank configuration",
        description="Get fully resolved configuration for a bank including all hierarchical overrides (global → tenant → bank). "
        "The 'config' field contains all resolved config values. The 'overrides' field shows only bank-specific overrides.",
        operation_id="get_bank_config",
        tags=["Banks"],
    )
    async def api_get_bank_config(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Get configuration for a bank with all hierarchical overrides applied."""
        if not get_config().enable_bank_config_api:
            raise HTTPException(
                status_code=404,
                detail="Bank configuration API is disabled. Set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true to re-enable.",
            )
        try:
            # Authenticate and set schema context for multi-tenant DB queries
            await app.state.memory._authenticate_tenant(request_context)
            if app.state.memory._operation_validator:
                from hindsight_api.extensions import BankReadContext

                ctx = BankReadContext(bank_id=bank_id, operation="get_bank_config", request_context=request_context)
                await app.state.memory._validate_operation(
                    app.state.memory._operation_validator.validate_bank_read(ctx)
                )

            # Get resolved config from config resolver
            config_dict = await app.state.memory._config_resolver.get_bank_config(bank_id, request_context)

            # Get bank-specific overrides only
            bank_overrides = await app.state.memory._config_resolver._load_bank_config(bank_id)

            return BankConfigResponse(bank_id=bank_id, config=config_dict, overrides=bank_overrides)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/config: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}/config",
        response_model=BankConfigResponse,
        summary="Update bank configuration",
        description="Update configuration overrides for a bank. Only hierarchical fields can be overridden (LLM settings, retention parameters, etc.). "
        "Keys can be provided in Python field format (llm_provider) or environment variable format (HINDSIGHT_API_LLM_PROVIDER).",
        operation_id="update_bank_config",
        tags=["Banks"],
    )
    @audited("update_bank_config")
    async def api_update_bank_config(
        bank_id: str, request: BankConfigUpdate, request_context: RequestContext = Depends(get_request_context)
    ):
        """Update configuration overrides for a bank."""
        if not get_config().enable_bank_config_api:
            raise HTTPException(
                status_code=404,
                detail="Bank configuration API is disabled. Set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true to re-enable.",
            )
        try:
            # Authenticate and set schema context for multi-tenant DB queries
            await app.state.memory._authenticate_tenant(request_context)
            if app.state.memory._operation_validator:
                from hindsight_api.extensions import BankWriteContext

                ctx = BankWriteContext(bank_id=bank_id, operation="update_bank_config", request_context=request_context)
                await app.state.memory._validate_operation(
                    app.state.memory._operation_validator.validate_bank_write(ctx)
                )

            # Update config via config resolver (validates configurable fields and permissions)
            await app.state.memory._config_resolver.update_bank_config(bank_id, request.updates, request_context)

            # Return updated config
            config_dict = await app.state.memory._config_resolver.get_bank_config(bank_id, request_context)
            bank_overrides = await app.state.memory._config_resolver._load_bank_config(bank_id)

            return BankConfigResponse(bank_id=bank_id, config=config_dict, overrides=bank_overrides)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except ValueError as e:
            # Validation error (e.g., trying to override static field)
            raise HTTPException(status_code=400, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}/config: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/config",
        response_model=BankConfigResponse,
        summary="Reset bank configuration",
        description="Reset bank configuration to defaults by removing all bank-specific overrides. "
        "The bank will then use global and tenant-level configuration only.",
        operation_id="reset_bank_config",
        tags=["Banks"],
    )
    @audited("reset_bank_config", request_param=None)
    async def api_reset_bank_config(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Reset bank configuration to defaults (remove all overrides)."""
        if not get_config().enable_bank_config_api:
            raise HTTPException(
                status_code=404,
                detail="Bank configuration API is disabled. Set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true to re-enable.",
            )
        try:
            # Authenticate and set schema context for multi-tenant DB queries
            await app.state.memory._authenticate_tenant(request_context)
            if app.state.memory._operation_validator:
                from hindsight_api.extensions import BankWriteContext

                ctx = BankWriteContext(bank_id=bank_id, operation="reset_bank_config", request_context=request_context)
                await app.state.memory._validate_operation(
                    app.state.memory._operation_validator.validate_bank_write(ctx)
                )

            # Reset config via config resolver
            await app.state.memory._config_resolver.reset_bank_config(bank_id)

            # Return updated config (should match defaults now)
            config_dict = await app.state.memory._config_resolver.get_bank_config(bank_id, request_context)
            bank_overrides = await app.state.memory._config_resolver._load_bank_config(bank_id)

            return BankConfigResponse(bank_id=bank_id, config=config_dict, overrides=bank_overrides)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/config: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/consolidate",
        response_model=ConsolidationResponse,
        summary="Trigger consolidation",
        description="Run memory consolidation to create/update observations from recent memories.",
        operation_id="trigger_consolidation",
        tags=["Banks"],
    )
    @audited("consolidation", request_param=None)
    async def api_trigger_consolidation(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Trigger consolidation for a bank (async)."""
        try:
            result = await app.state.memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            return ConsolidationResponse(
                operation_id=result["operation_id"],
                deduplicated=result.get("deduplicated", False),
            )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/consolidate: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # Webhook Endpoints
    # =========================================================================

    @app.post(
        "/v1/default/banks/{bank_id}/webhooks",
        response_model=WebhookResponse,
        summary="Register webhook",
        description="Register a webhook endpoint to receive event notifications for this bank.",
        operation_id="create_webhook",
        tags=["Webhooks"],
        status_code=201,
    )
    @audited("create_webhook")
    async def api_create_webhook(
        bank_id: str,
        request: CreateWebhookRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Register a webhook for a bank."""
        try:
            pool = await app.state.memory._get_pool()
            from hindsight_api.engine.memory_engine import fq_table
            from hindsight_api.engine.retain import bank_utils

            # Ensure the bank row exists before inserting into webhooks (FK constraint).
            _, created = await bank_utils.get_or_create_bank_profile(pool, bank_id)
            if created:
                await app.state.memory._apply_default_bank_template(bank_id, request_context)

            webhook_id = uuid.uuid4()
            now = datetime.now(timezone.utc).isoformat()
            row = await pool.fetchrow(
                f"""
                INSERT INTO {fq_table("webhooks")}
                (id, bank_id, url, secret, event_types, enabled, http_config, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, NOW(), NOW())
                RETURNING id, bank_id, url, secret, event_types, enabled,
                          http_config::text, created_at::text, updated_at::text
                """,
                webhook_id,
                bank_id,
                request.url,
                request.secret,
                request.event_types,
                request.enabled,
                request.http_config.model_dump_json(),
            )
            return WebhookResponse(
                id=str(row["id"]),
                bank_id=row["bank_id"],
                url=row["url"],
                secret=None,  # Never return secret in responses
                event_types=list(row["event_types"]) if row["event_types"] else [],
                enabled=row["enabled"],
                http_config=WebhookHttpConfig.model_validate_json(row["http_config"])
                if row["http_config"]
                else WebhookHttpConfig(),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/webhooks: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/webhooks",
        response_model=WebhookListResponse,
        summary="List webhooks",
        description="List all webhooks registered for a bank.",
        operation_id="list_webhooks",
        tags=["Webhooks"],
    )
    async def api_list_webhooks(
        bank_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List webhooks for a bank."""
        try:
            pool = await app.state.memory._get_pool()
            from hindsight_api.engine.memory_engine import fq_table

            rows = await pool.fetch(
                f"""
                SELECT id, bank_id, url, secret, event_types, enabled,
                       http_config::text, created_at::text, updated_at::text
                FROM {fq_table("webhooks")}
                WHERE bank_id = $1
                ORDER BY created_at
                """,
                bank_id,
            )
            return WebhookListResponse(
                items=[
                    WebhookResponse(
                        id=str(row["id"]),
                        bank_id=row["bank_id"],
                        url=row["url"],
                        secret=None,  # Never return secret in responses
                        event_types=list(row["event_types"]) if row["event_types"] else [],
                        enabled=row["enabled"],
                        http_config=WebhookHttpConfig.model_validate_json(row["http_config"])
                        if row["http_config"]
                        else WebhookHttpConfig(),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/webhooks: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
        response_model=DeleteResponse,
        summary="Delete webhook",
        description="Remove a registered webhook.",
        operation_id="delete_webhook",
        tags=["Webhooks"],
    )
    @audited("delete_webhook", request_param=None)
    async def api_delete_webhook(
        bank_id: str,
        webhook_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Delete a webhook."""
        try:
            pool = await app.state.memory._get_pool()
            from hindsight_api.engine.memory_engine import fq_table

            result = await pool.execute(
                f"DELETE FROM {fq_table('webhooks')} WHERE id = $1 AND bank_id = $2",
                uuid.UUID(webhook_id),
                bank_id,
            )
            deleted = int(result.split()[-1]) if result else 0
            if deleted == 0:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return DeleteResponse(success=True)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/webhooks/{webhook_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.patch(
        "/v1/default/banks/{bank_id}/webhooks/{webhook_id}",
        response_model=WebhookResponse,
        summary="Update webhook",
        description="Update one or more fields of a registered webhook. Only provided fields are changed.",
        operation_id="update_webhook",
        tags=["Webhooks"],
    )
    @audited("update_webhook")
    async def api_update_webhook(
        bank_id: str,
        webhook_id: str,
        request: UpdateWebhookRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Update a webhook's fields (PATCH semantics — only sent fields are updated)."""
        try:
            pool = await app.state.memory._get_pool()
            from hindsight_api.engine.memory_engine import fq_table

            set_clauses: list[str] = []
            params: list = [uuid.UUID(webhook_id), bank_id]

            fields = request.model_fields_set
            if "url" in fields:
                params.append(request.url)
                set_clauses.append(f"url = ${len(params)}")
            if "secret" in fields:
                params.append(request.secret)
                set_clauses.append(f"secret = ${len(params)}")
            if "event_types" in fields:
                params.append(request.event_types)
                set_clauses.append(f"event_types = ${len(params)}")
            if "enabled" in fields:
                params.append(request.enabled)
                set_clauses.append(f"enabled = ${len(params)}")
            if "http_config" in fields:
                params.append(request.http_config.model_dump_json())
                set_clauses.append(f"http_config = ${len(params)}::jsonb")

            if not set_clauses:
                raise HTTPException(status_code=422, detail="No fields provided to update")

            set_clauses.append("updated_at = NOW()")
            row = await pool.fetchrow(
                f"""
                UPDATE {fq_table("webhooks")}
                SET {", ".join(set_clauses)}
                WHERE id = $1 AND bank_id = $2
                RETURNING id, bank_id, url, secret, event_types, enabled,
                          http_config::text, created_at::text, updated_at::text
                """,
                *params,
            )
            if not row:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return WebhookResponse(
                id=str(row["id"]),
                bank_id=row["bank_id"],
                url=row["url"],
                secret=None,
                event_types=list(row["event_types"]) if row["event_types"] else [],
                enabled=row["enabled"],
                http_config=WebhookHttpConfig.model_validate_json(row["http_config"])
                if row["http_config"]
                else WebhookHttpConfig(),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in PATCH /v1/default/banks/{bank_id}/webhooks/{webhook_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/webhooks/{webhook_id}/deliveries",
        response_model=WebhookDeliveryListResponse,
        summary="List webhook deliveries",
        description="Inspect delivery history for a webhook (useful for debugging).",
        operation_id="list_webhook_deliveries",
        tags=["Webhooks"],
    )
    async def api_list_webhook_deliveries(
        bank_id: str,
        webhook_id: str,
        limit: int = Query(default=50, le=200, description="Maximum number of deliveries to return"),
        cursor: str | None = Query(default=None, description="Pagination cursor (created_at of last item)"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List deliveries for a specific webhook, newest first. Use next_cursor for pagination."""
        try:
            pool = await app.state.memory._get_pool()
            from hindsight_api.engine.memory_engine import fq_table

            # Verify webhook belongs to this bank
            webhook_row = await pool.fetchrow(
                f"SELECT id FROM {fq_table('webhooks')} WHERE id = $1 AND bank_id = $2",
                uuid.UUID(webhook_id),
                bank_id,
            )
            if not webhook_row:
                raise HTTPException(status_code=404, detail="Webhook not found")

            # Fetch limit+1 to detect if there's a next page
            fetch_limit = limit + 1
            if cursor:
                rows = await pool.fetch(
                    f"""
                    SELECT operation_id, status, retry_count, next_retry_at::text,
                           error_message, task_payload, result_metadata::text, created_at::text, updated_at::text
                    FROM {fq_table("async_operations")}
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'webhook_id' = $2
                      AND created_at < $3::timestamptz
                    ORDER BY created_at DESC
                    LIMIT $4
                    """,
                    bank_id,
                    webhook_id,
                    cursor,
                    fetch_limit,
                )
            else:
                rows = await pool.fetch(
                    f"""
                    SELECT operation_id, status, retry_count, next_retry_at::text,
                           error_message, task_payload, result_metadata::text, created_at::text, updated_at::text
                    FROM {fq_table("async_operations")}
                    WHERE operation_type = 'webhook_delivery'
                      AND bank_id = $1
                      AND task_payload->>'webhook_id' = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    bank_id,
                    webhook_id,
                    fetch_limit,
                )

            has_more = len(rows) > limit
            page = rows[:limit]
            next_cursor = page[-1]["created_at"] if has_more and page else None
            return WebhookDeliveryListResponse(
                items=[WebhookDeliveryResponse.from_async_operation_row(dict(row)) for row in page],
                next_cursor=next_cursor,
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/webhooks/{webhook_id}/deliveries: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/memories",
        response_model=RetainResponse,
        summary="Retain memories",
        description="Retain memory items with automatic fact extraction.\n\n"
        "This is the main endpoint for storing memories. It supports both synchronous and asynchronous processing via the `async` parameter.\n\n"
        "**Features:**\n"
        "- Efficient batch processing\n"
        "- Automatic fact extraction from natural language\n"
        "- Entity recognition and linking\n"
        "- Document tracking with automatic upsert (when document_id is provided)\n"
        "- Temporal and semantic linking\n"
        "- Optional asynchronous processing\n\n"
        "**The system automatically:**\n"
        "1. Extracts semantic facts from the content\n"
        "2. Generates embeddings\n"
        "3. Deduplicates similar facts\n"
        "4. Creates temporal, semantic, and entity links\n"
        "5. Tracks document metadata\n\n"
        "**When `async=true`:** Returns immediately after queuing. Use the operations endpoint to monitor progress.\n\n"
        "**When `async=false` (default):** Waits for processing to complete.\n\n"
        "**Note:** If a memory item has a `document_id` that already exists, the old document and its memory units will be deleted before creating new ones (upsert behavior).",
        operation_id="retain_memories",
        tags=["Memory"],
    )
    @audited("retain")
    async def api_retain(
        bank_id: str, request: RetainRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Retain memories with optional async processing."""
        metrics = get_metrics_collector()

        try:
            # Group items by strategy
            strategy_groups: dict[str | None, list[dict]] = {}
            for item in request.items:
                effective = item.strategy
                if effective not in strategy_groups:
                    strategy_groups[effective] = []
                content_dict: dict = {"content": item.content}
                if item.timestamp == "unset":
                    content_dict["event_date"] = None
                elif item.timestamp:
                    content_dict["event_date"] = item.timestamp
                if item.context:
                    content_dict["context"] = item.context
                if item.metadata:
                    content_dict["metadata"] = item.metadata
                if item.document_id:
                    content_dict["document_id"] = item.document_id
                if item.entities:
                    content_dict["entities"] = [{"text": e.text, "type": e.type or "CONCEPT"} for e in item.entities]
                if item.tags:
                    content_dict["tags"] = item.tags
                if item.observation_scopes is not None:
                    content_dict["observation_scopes"] = item.observation_scopes
                if item.update_mode is not None:
                    content_dict["update_mode"] = item.update_mode
                strategy_groups[effective].append(content_dict)

            if request.async_:
                # Async processing: one submit per strategy group
                all_operation_ids = []
                total_items_count = 0
                for group_strategy, contents in strategy_groups.items():
                    result = await app.state.memory.submit_async_retain(
                        bank_id,
                        contents,
                        document_tags=request.document_tags,
                        strategy=group_strategy,
                        request_context=request_context,
                    )
                    all_operation_ids.append(result["operation_id"])
                    total_items_count += result["items_count"]
                return RetainResponse.model_validate(
                    {
                        "success": True,
                        "bank_id": bank_id,
                        "items_count": total_items_count,
                        "async": True,
                        "operation_id": all_operation_ids[0] if all_operation_ids else None,
                        "operation_ids": all_operation_ids if len(all_operation_ids) > 1 else None,
                    }
                )
            else:
                # Check if batch API is enabled - if so, require async mode
                from hindsight_api.config import get_config

                config = get_config()
                if config.retain_batch_enabled:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Batch API is enabled (HINDSIGHT_API_RETAIN_BATCH_ENABLED=true) but async=false. "
                            "Batch operations can take several minutes to hours and will timeout in synchronous mode. "
                            "Please set async=true in your request to use background processing, or disable batch API "
                            "by setting HINDSIGHT_API_RETAIN_BATCH_ENABLED=false in your environment."
                        ),
                    )

                # Synchronous processing: one batch per strategy group, aggregate results
                total_items_count = 0
                total_usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
                with metrics.record_operation("retain", bank_id=bank_id, source="api"):
                    for group_strategy, contents in strategy_groups.items():
                        result, usage = await app.state.memory.retain_batch_async(
                            bank_id=bank_id,
                            contents=contents,
                            document_tags=request.document_tags,
                            strategy=group_strategy,
                            request_context=request_context,
                            return_usage=True,
                            outbox_callback=app.state.memory._build_retain_outbox_callback(
                                bank_id=bank_id,
                                contents=contents,
                                operation_id=None,
                                schema=_current_schema.get(),
                            ),
                        )
                        total_items_count += len(contents)
                        if usage:
                            total_usage = TokenUsage(
                                input_tokens=total_usage.input_tokens + usage.input_tokens,
                                output_tokens=total_usage.output_tokens + usage.output_tokens,
                                total_tokens=total_usage.total_tokens + usage.total_tokens,
                            )

                return RetainResponse.model_validate(
                    {
                        "success": True,
                        "bank_id": bank_id,
                        "items_count": total_items_count,
                        "async": False,
                        "usage": total_usage,
                    }
                )
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            # Create a summary of the input for debugging
            input_summary = []
            for i, item in enumerate(request.items):
                content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                input_summary.append(
                    f"  [{i}] content={content_preview!r}, context={item.context}, timestamp={item.timestamp}"
                )
            input_debug = "\n".join(input_summary)

            error_detail = (
                f"{str(e)}\n\n"
                f"Input ({len(request.items)} items):\n{input_debug}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories (retain): {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/files/retain",
        response_model=FileRetainResponse,
        summary="Convert files to memories",
        description="Upload files (PDF, DOCX, etc.), convert them to markdown, and retain as memories.\n\n"
        "This endpoint handles file upload, conversion, and memory creation in a single operation.\n\n"
        "**Features:**\n"
        "- Supports PDF, DOCX, PPTX, XLSX, images (with OCR), audio (with transcription)\n"
        "- Automatic file-to-markdown conversion using pluggable parsers\n"
        "- Files stored in object storage (PostgreSQL by default, S3 for production)\n"
        "- Each file becomes a separate document with optional metadata/tags\n"
        "- Always processes asynchronously — returns operation IDs immediately\n\n"
        "**The system automatically:**\n"
        "1. Stores uploaded files in object storage\n"
        "2. Converts files to markdown\n"
        "3. Creates document records with file metadata\n"
        "4. Extracts facts and creates memory units (same as regular retain)\n\n"
        "Use the operations endpoint to monitor progress.\n\n"
        "**Request format:** multipart/form-data with:\n"
        "- `files`: One or more files to upload\n"
        "- `request`: JSON string with FileRetainRequest model\n\n"
        "**Parser selection:**\n"
        "- Set `parser` in the request body to override the server default for all files.\n"
        "- Set `parser` inside a `files_metadata` entry for per-file control.\n"
        "- Pass a list (e.g. `['iris', 'markitdown']`) to define an ordered fallback chain — "
        "each parser is tried in sequence until one succeeds.\n"
        "- Falls back to the server default (`HINDSIGHT_API_FILE_PARSER`) if not specified.\n"
        "- Only parsers enabled on the server may be requested; others return HTTP 400.",
        operation_id="file_retain",
        tags=["Files"],
    )
    @audited("file_retain", request_param=None)
    async def api_file_retain(
        bank_id: str,
        files: list[UploadFile] = File(..., description="Files to upload and convert"),
        request: str = Form(..., description="JSON string with FileRetainRequest model"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Upload and convert files to memories."""
        from hindsight_api.config import get_config

        config = get_config()

        # Check if file upload API is enabled
        if not config.enable_file_upload_api:
            raise HTTPException(
                status_code=404,
                detail="File upload API is disabled. Set HINDSIGHT_API_ENABLE_FILE_UPLOAD_API=true to enable.",
            )

        try:
            # Parse request JSON
            try:
                request_data = FileRetainRequest.model_validate_json(request)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request JSON: {str(e)}",
                )

            # Validate file count
            if len(files) > config.file_conversion_max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many files. Maximum {config.file_conversion_max_batch_size} files per request.",
                )

            # Validate files_metadata count matches files count if provided
            if request_data.files_metadata and len(request_data.files_metadata) != len(files):
                raise HTTPException(
                    status_code=400,
                    detail=f"files_metadata count ({len(request_data.files_metadata)}) must match files count ({len(files)})",
                )

            # Resolve the registered parser names for allowlist validation
            registered_parsers = app.state.memory._parser_registry.list_parsers()
            allowlist = config.file_parser_allowlist if config.file_parser_allowlist is not None else registered_parsers

            def _resolve_parser(raw: str | list[str] | None) -> list[str]:
                """Normalize parser value to a non-empty list of names."""
                if raw is None:
                    return config.file_parser
                return [raw] if isinstance(raw, str) else list(raw)

            def _validate_parsers(parsers: list[str], context: str) -> None:
                """Raise HTTP 400 if any parser name is not in the allowlist."""
                disallowed = [p for p in parsers if p not in allowlist]
                if disallowed:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Parser(s) not available ({context}): {disallowed}. Available: {allowlist}",
                    )

            # Validate request-level parser early (before reading files)
            if request_data.parser is not None:
                _validate_parsers(_resolve_parser(request_data.parser), "request-level parser")

            # Prepare file items and calculate total batch size
            import io

            file_items = []
            total_batch_size = 0

            for i, file in enumerate(files):
                # Read file content to check size
                file_content = await file.read()
                total_batch_size += len(file_content)

                # Create a mock UploadFile with the necessary attributes
                class FileWrapper:
                    def __init__(self, content, filename, content_type):
                        self._content = content
                        self.filename = filename
                        self.content_type = content_type

                    async def read(self):
                        return self._content

                wrapped_file = FileWrapper(file_content, file.filename, file.content_type)

                # Get per-file metadata
                file_meta = request_data.files_metadata[i] if request_data.files_metadata else FileRetainMetadata()
                doc_id = file_meta.document_id or f"file_{uuid.uuid4()}"

                # Resolve and validate per-file parser chain
                # Priority: per-file > request-level > server default
                raw_parser = file_meta.parser if file_meta.parser is not None else request_data.parser
                parser_chain = _resolve_parser(raw_parser)
                _validate_parsers(parser_chain, f"file '{file.filename}'")

                item = {
                    "file": wrapped_file,
                    "document_id": doc_id,
                    "context": file_meta.context,
                    "metadata": file_meta.metadata or {},
                    "tags": file_meta.tags or [],
                    "timestamp": file_meta.timestamp,
                    "parser": parser_chain,
                    "strategy": file_meta.strategy,
                }
                file_items.append(item)

            # Check total batch size after processing all files
            if total_batch_size > config.file_conversion_max_batch_size_bytes:
                total_mb = total_batch_size / (1024 * 1024)
                raise HTTPException(
                    status_code=400,
                    detail=f"Total batch size ({total_mb:.1f}MB) exceeds maximum of {config.file_conversion_max_batch_size_mb}MB",
                )

            result = await app.state.memory.submit_async_file_retain(
                bank_id=bank_id,
                file_items=file_items,
                document_tags=None,
                request_context=request_context,
            )
            return FileRetainResponse.model_validate(
                {
                    "operation_ids": result["operation_ids"],
                }
            )

        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/files/retain: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/memories",
        response_model=DeleteResponse,
        summary="Clear memory bank memories",
        description="Delete memory units for a memory bank. Optionally filter by type (world, experience, opinion) to delete only specific types. This is a destructive operation that cannot be undone. The bank profile (disposition and background) will be preserved.",
        operation_id="clear_bank_memories",
        tags=["Memory"],
    )
    @audited("clear_memories", request_param=None)
    async def api_clear_bank_memories(
        bank_id: str,
        type: str | None = Query(None, description="Optional fact type filter (world, experience, opinion)"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Clear memories for a memory bank, optionally filtered by type."""
        try:
            await app.state.memory.delete_bank(
                bank_id, fact_type=type, delete_bank_profile=False, request_context=request_context
            )

            return DeleteResponse(success=True)
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    # ---- Audit Logs ----

    class AuditLogEntry(BaseModel):
        """A single audit log entry."""

        id: str
        action: str
        transport: str
        bank_id: str | None
        started_at: str | None
        ended_at: str | None
        duration_ms: int | None = Field(
            default=None,
            description="Server-computed duration in milliseconds (started_at → ended_at). Null if not yet completed.",
        )
        request: dict[str, Any] | None
        response: dict[str, Any] | None
        metadata: dict[str, Any]

    class AuditLogListResponse(BaseModel):
        """Response model for list audit logs endpoint."""

        bank_id: str
        total: int
        limit: int
        offset: int
        items: list[AuditLogEntry]

    class AuditLogStatsBucket(BaseModel):
        """A single time bucket in audit log stats."""

        time: str
        actions: dict[str, int]
        total: int

    class AuditLogStatsResponse(BaseModel):
        """Response model for audit log stats endpoint."""

        bank_id: str
        period: str
        trunc: str
        start: str
        buckets: list[AuditLogStatsBucket]

    @app.get(
        "/v1/default/banks/{bank_id}/audit-logs",
        summary="List audit logs",
        description="List audit log entries for a bank, ordered by most recent first.",
        operation_id="list_audit_logs",
        tags=["Audit"],
        response_model=AuditLogListResponse,
    )
    async def api_list_audit_logs(
        bank_id: str,
        action: str | None = Query(None, description="Filter by action type"),
        transport: str | None = Query(None, description="Filter by transport (http, mcp, system)"),
        start_date: str | None = Query(None, description="Filter from this ISO datetime (inclusive)"),
        end_date: str | None = Query(None, description="Filter until this ISO datetime (exclusive)"),
        limit: int = Query(50, ge=1, le=500, description="Max items to return"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List audit log entries for a bank."""
        try:
            from hindsight_api.engine.memory_engine import fq_table

            pool = await app.state.memory._get_pool()

            # Ensure bank exists
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            from hindsight_api.engine.db_utils import acquire_with_retry

            async with acquire_with_retry(pool) as conn:
                where_clauses = ["bank_id = $1"]
                params: list[Any] = [bank_id]
                idx = 2

                if action:
                    where_clauses.append(f"action = ${idx}")
                    params.append(action)
                    idx += 1

                if transport:
                    where_clauses.append(f"transport = ${idx}")
                    params.append(transport)
                    idx += 1

                if start_date:
                    parsed_start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    where_clauses.append(f"started_at >= ${idx}")
                    params.append(parsed_start)
                    idx += 1

                if end_date:
                    parsed_end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    where_clauses.append(f"started_at < ${idx}")
                    params.append(parsed_end)
                    idx += 1

                where_sql = " AND ".join(where_clauses)
                table = fq_table("audit_log")

                # Get total count
                count_row = await conn.fetchrow(
                    f"SELECT COUNT(*) as total FROM {table} WHERE {where_sql}",
                    *params,
                )
                total = count_row["total"] if count_row else 0

                # Get paginated results
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
                    duration_ms = None
                    if row["started_at"] and row["ended_at"]:
                        duration_ms = int((row["ended_at"] - row["started_at"]).total_seconds() * 1000)

                    items.append(
                        {
                            "id": str(row["id"]),
                            "action": row["action"],
                            "transport": row["transport"],
                            "bank_id": row["bank_id"],
                            "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                            "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
                            "duration_ms": duration_ms,
                            "request": json.loads(row["request"]) if row["request"] else None,
                            "response": json.loads(row["response"]) if row["response"] else None,
                            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        }
                    )

                return {
                    "bank_id": bank_id,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "items": items,
                }
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            logger.error(f"Error listing audit logs: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/audit-logs/stats",
        summary="Audit log statistics",
        description="Get audit log counts grouped by time bucket for charting.",
        operation_id="audit_log_stats",
        tags=["Audit"],
        response_model=AuditLogStatsResponse,
    )
    async def api_audit_log_stats(
        bank_id: str,
        action: str | None = Query(None, description="Filter by action type"),
        period: str = Query("7d", description="Time period: 1d, 7d, or 30d"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get audit log counts grouped by time bucket."""
        try:
            from hindsight_api.engine.db_utils import acquire_with_retry
            from hindsight_api.engine.memory_engine import fq_table

            pool = await app.state.memory._get_pool()
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Determine time range (always per-day buckets)
            from datetime import timedelta as _td

            now = datetime.now(timezone.utc)
            trunc = "day"
            if period == "1d":
                start = now - _td(days=1)
            elif period == "30d":
                start = now - _td(days=30)
            else:  # 7d default
                start = now - _td(days=7)

            table = fq_table("audit_log")

            async with acquire_with_retry(pool) as conn:
                where_clauses = ["bank_id = $1", "started_at >= $2"]
                params: list[Any] = [bank_id, start]
                idx = 3

                if action:
                    where_clauses.append(f"action = ${idx}")
                    params.append(action)
                    idx += 1

                where_sql = " AND ".join(where_clauses)

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

                buckets: dict[str, dict[str, int]] = {}
                for row in rows:
                    bucket_key = row["bucket"].isoformat()
                    if bucket_key not in buckets:
                        buckets[bucket_key] = {}
                    buckets[bucket_key][row["action"]] = row["count"]

                return {
                    "bank_id": bank_id,
                    "period": period,
                    "trunc": trunc,
                    "start": start.isoformat(),
                    "buckets": [{"time": k, "actions": v, "total": sum(v.values())} for k, v in buckets.items()],
                }
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            logger.error(f"Error getting audit log stats: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
