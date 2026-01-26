"""
FastAPI application factory and API routes for memory system.

This module provides the create_app function to create and configure
the FastAPI application with all API endpoints.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query

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


from pydantic import BaseModel, ConfigDict, Field, field_validator

from hindsight_api import MemoryEngine
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import Budget, fq_table
from hindsight_api.engine.reflect.observations import Observation
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES, TokenUsage
from hindsight_api.engine.search.tags import TagsMatch
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


class IncludeOptions(BaseModel):
    """Options for including additional data in recall results."""

    entities: EntityIncludeOptions | None = Field(
        default=EntityIncludeOptions(),
        description="Include entity observations. Set to null to disable entity inclusion.",
    )
    chunks: ChunkIncludeOptions | None = Field(
        default=None, description="Include raw chunks. Set to {} to enable, null to disable (default: disabled)."
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
        description="List of fact types to recall: 'world', 'experience', 'observation'. Defaults to world and experience if not specified. "
        "Note: 'opinion' is accepted but ignored (opinions are excluded from recall).",
    )
    budget: Budget = Budget.MID
    max_tokens: int = 4096
    trace: bool = False
    query_timestamp: str | None = Field(
        default=None, description="ISO format date string (e.g., '2023-05-30T23:40:00')"
    )
    include: IncludeOptions = Field(
        default_factory=IncludeOptions,
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
    timestamp: datetime | None = None
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

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                # Try parsing as ISO format
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValueError(
                    f"Invalid timestamp/event_date format: '{v}'. Expected ISO format like '2024-01-15T10:30:00' or '2024-01-15T10:30:00Z'"
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
                "document_tags": ["user_a", "user_b"],
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
        description="Tags applied to all items in this request. These are merged with any item-level tags.",
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
        description="Operation ID for tracking async operations. Use GET /v1/default/banks/{bank_id}/operations to list operations and find this ID. Only present when async=true.",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="Token usage metrics for LLM calls during fact extraction (only present for synchronous operations)",
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


class OpinionItem(BaseModel):
    """Model for an opinion with confidence score."""

    text: str
    confidence: float


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
    text: str
    type: str | None = None  # fact type: world, experience, opinion
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None


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
    """Evidence the response is based on: memories and mental models."""

    memories: list[ReflectFact] = Field(default_factory=list, description="Memory facts used to generate the response")


class ReflectTrace(BaseModel):
    """Execution trace of LLM and tool calls during reflection."""

    tool_calls: list[ReflectToolCall] = Field(default_factory=list, description="Tool calls made during reflection")
    llm_calls: list[ReflectLLMCall] = Field(default_factory=list, description="LLM calls made during reflection")


class ReflectResponse(BaseModel):
    """Response model for think endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Based on my understanding, AI is a transformative technology...",
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

    text: str
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


class BanksResponse(BaseModel):
    """Response model for banks list endpoint."""

    model_config = ConfigDict(json_schema_extra={"example": {"banks": ["user123", "bank_alice", "bank_bob"]}})

    banks: list[str]


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
                "name": "Alice",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                "mission": "I am a PM helping my engineering team stay organized",
            }
        }
    )

    name: str | None = None
    disposition: DispositionTraits | None = None
    mission: str | None = Field(default=None, description="The agent's mission")
    # Deprecated: use mission instead
    background: str | None = Field(default=None, description="Deprecated: use mission instead")


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
    tags: list[str] = Field(default_factory=list, description="Tags associated with this document")


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
    # Consolidation stats
    last_consolidated_at: str | None = Field(default=None, description="When consolidation last ran (ISO format)")
    pending_consolidation: int = Field(default=0, description="Number of memories not yet processed into observations")
    total_observations: int = Field(default=0, description="Total number of observations")


# Mental Model models


class ObservationEvidenceResponse(BaseModel):
    """A single piece of evidence supporting an observation."""

    memory_id: str = Field(description="ID of the memory unit this evidence comes from")
    quote: str = Field(description="Exact quote from the memory supporting the observation")
    relevance: str = Field(description="Brief explanation of how this quote supports the observation")
    timestamp: str = Field(description="When the source memory was created (ISO format)")


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
    tags: list[str] = Field(default_factory=list)
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
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")


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


class MentalModelResponse(BaseModel):
    """Response model for a mental model (stored reflect response)."""

    id: str
    bank_id: str
    name: str
    source_query: str
    content: str
    tags: list[str] = Field(default_factory=list)
    last_refreshed_at: str | None = None
    created_at: str | None = None
    reflect_response: dict | None = Field(
        default=None,
        description="Full reflect API response payload including based_on facts and observations",
    )


class MentalModelListResponse(BaseModel):
    """Response model for listing mental models."""

    items: list[MentalModelResponse]


class CreateMentalModelRequest(BaseModel):
    """Request model for creating a mental model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Team Communication Preferences",
                "source_query": "How does the team prefer to communicate?",
                "tags": ["team"],
                "max_tokens": 2048,
            }
        }
    )

    name: str = Field(description="Human-readable name for the mental model")
    source_query: str = Field(description="The query to run to generate content")
    tags: list[str] = Field(default_factory=list, description="Tags for scoped visibility")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Maximum tokens for generated content")


class CreateMentalModelResponse(BaseModel):
    """Response model for mental model creation."""

    operation_id: str = Field(description="Operation ID to track progress")


class UpdateMentalModelRequest(BaseModel):
    """Request model for updating a mental model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Updated Team Communication Preferences",
            }
        }
    )

    name: str | None = Field(default=None, description="New name for the mental model")


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


class VersionResponse(BaseModel):
    """Response model for the version/info endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_version": "1.0.0",
                "features": {
                    "observations": False,
                    "mcp": True,
                    "worker": True,
                },
            }
        }
    )

    api_version: str = Field(description="API version string")
    features: FeaturesInfo = Field(description="Enabled feature flags")


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
            worker_id = config.worker_id or socket.gethostname()
            poller = WorkerPoller(
                pool=memory._pool,
                worker_id=worker_id,
                executor=memory.execute_task,
                poll_interval_ms=config.worker_poll_interval_ms,
                batch_size=config.worker_batch_size,
                max_retries=config.worker_max_retries,
            )
            poller_task = asyncio.create_task(poller.run())
            logging.info(f"Worker poller started (worker_id={worker_id})")

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

        # Call HTTP extension shutdown hook
        if http_extension:
            await http_extension.on_shutdown()
            logging.info("HTTP extension stopped")

        # Shutdown: Cleanup memory system
        await memory.close()
        logging.info("Memory system closed")

    from hindsight_api import __version__

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
    )

    # IMPORTANT: Set memory on app.state immediately, don't wait for lifespan
    # This is required for mounted sub-applications where lifespan may not fire
    app.state.memory = memory

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

    return app


def _register_routes(app: FastAPI):
    """Register all API routes on the given app instance."""

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
        """
        from hindsight_api.config import get_config

        config = get_config()
        return VersionResponse(
            api_version="1.0.0",
            features=FeaturesInfo(
                observations=config.enable_observations,
                mcp=config.mcp_enabled,
                worker=config.worker_enabled,
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
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get graph data from database, filtered by bank_id and optionally by type."""
        try:
            data = await app.state.memory.get_graph_data(bank_id, type, limit=limit, request_context=request_context)
            return data
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
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_memory_units(
                bank_id=bank_id,
                fact_type=type,
                search_query=q,
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return data
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
        description="Get a single memory unit by ID with all its metadata including entities and tags.",
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
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories/{memory_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/memories/recall",
        response_model=RecallResponse,
        summary="Recall memory",
        description="Recall memory using semantic similarity and spreading activation.\n\n"
        "The type parameter is optional and must be one of:\n"
        "- `world`: General knowledge about people, places, events, and things that happen\n"
        "- `experience`: Memories about experience, conversations, actions taken, and tasks performed\n"
        "- `opinion`: The bank's formed beliefs, perspectives, and viewpoints\n\n"
        "Set `include_entities=true` to get entity observations alongside recall results.",
        operation_id="recall_memories",
        tags=["Memory"],
    )
    async def api_recall(
        bank_id: str, request: RecallRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Run a recall and return results with trace."""
        import time

        handler_start = time.time()
        metrics = get_metrics_collector()

        try:
            # Default to world and experience if not specified (exclude observation and opinion)
            # Filter out 'opinion' even if requested - opinions are excluded from recall
            fact_types = request.types if request.types else list(VALID_RECALL_FACT_TYPES)
            fact_types = [ft for ft in fact_types if ft != "opinion"]

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
                    request_context=request_context,
                    tags=request.tags,
                    tags_match=request.tags_match,
                )

            # Convert core MemoryFact objects to API RecallResult objects (excluding internal metrics)
            recall_results = [
                RecallResult(
                    id=fact.id,
                    text=fact.text,
                    type=fact.fact_type,
                    entities=fact.entities,
                    context=fact.context,
                    occurred_start=fact.occurred_start,
                    occurred_end=fact.occurred_end,
                    mentioned_at=fact.mentioned_at,
                    document_id=fact.document_id,
                    chunk_id=fact.chunk_id,
                    tags=fact.tags,
                )
                for fact in core_result.results
            ]

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

            response = RecallResponse(
                results=recall_results,
                trace=core_result.trace,
                entities=entities_response,
                chunks=chunks_response,
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
        "5. Extracts and stores any new opinions formed\n"
        "6. Returns plain text answer, the facts used, and new opinions",
        operation_id="reflect",
        tags=["Memory"],
    )
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
                )

            # Build based_on (memories + observations) if facts are requested
            based_on_result: ReflectBasedOn | None = None
            if request.include.facts is not None:
                memories = []
                for fact_type, facts in core_result.based_on.items():
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
                based_on_result = ReflectBasedOn(memories=memories)

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
            # Authenticate and set tenant schema
            await app.state.memory._authenticate_tenant(request_context)
            pool = await app.state.memory._get_pool()
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
                ops_by_status = {row["status"]: row["count"] for row in ops_stats}
                pending_operations = ops_by_status.get("pending", 0)
                failed_operations = ops_by_status.get("failed", 0)

                # Get document count
                doc_count_result = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) as count
                    FROM {fq_table("documents")}
                    WHERE bank_id = $1
                    """,
                    bank_id,
                )
                total_documents = doc_count_result["count"] if doc_count_result else 0

                # Get consolidation stats from memory-level tracking
                consolidation_stats = await conn.fetchrow(
                    f"""
                    SELECT
                        MAX(consolidated_at) as last_consolidated_at,
                        COUNT(*) FILTER (WHERE consolidated_at IS NULL AND fact_type IN ('experience', 'world')) as pending
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1
                    """,
                    bank_id,
                )
                last_consolidated_at = consolidation_stats["last_consolidated_at"] if consolidation_stats else None
                pending_consolidation = consolidation_stats["pending"] if consolidation_stats else 0

                # Count total observations (consolidated knowledge)
                observation_count_result = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) as count
                    FROM {fq_table("memory_units")}
                    WHERE bank_id = $1 AND fact_type = 'observation'
                    """,
                    bank_id,
                )
                total_observations = observation_count_result["count"] if observation_count_result else 0

                # Format results
                nodes_by_type = {row["fact_type"]: row["count"] for row in node_stats}
                links_by_type = {row["link_type"]: row["count"] for row in link_stats}
                links_by_fact_type = {row["fact_type"]: row["count"] for row in link_fact_type_stats}

                # Build detailed breakdown: {fact_type: {link_type: count}}
                links_breakdown = {}
                for row in link_breakdown_stats:
                    fact_type = row["fact_type"]
                    link_type = row["link_type"]
                    count = row["count"]
                    if fact_type not in links_breakdown:
                        links_breakdown[fact_type] = {}
                    links_breakdown[fact_type][link_type] = count

                total_nodes = sum(nodes_by_type.values())
                total_links = sum(links_by_type.values())

                return BankStatsResponse(
                    bank_id=bank_id,
                    total_nodes=total_nodes,
                    total_links=total_links,
                    total_documents=total_documents,
                    nodes_by_fact_type=nodes_by_type,
                    links_by_link_type=links_by_type,
                    links_by_fact_type=links_by_fact_type,
                    links_breakdown=links_breakdown,
                    pending_operations=pending_operations,
                    failed_operations=failed_operations,
                    last_consolidated_at=(last_consolidated_at.isoformat() if last_consolidated_at else None),
                    pending_consolidation=pending_consolidation,
                    total_observations=total_observations,
                )

        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/stats: {error_detail}")
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
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/entities: {error_detail}")
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
                limit=limit,
                offset=offset,
                request_context=request_context,
            )
            return MentalModelListResponse(items=[MentalModelResponse(**m) for m in mental_models])
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
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Get a mental model by ID."""
        try:
            mental_model = await app.state.memory.get_mental_model(
                bank_id=bank_id,
                mental_model_id=mental_model_id,
                request_context=request_context,
            )
            if mental_model is None:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")
            return MentalModelResponse(**mental_model)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in GET /v1/default/banks/{bank_id}/mental-models/{mental_model_id}: {error_detail}")
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
    async def api_create_mental_model(
        bank_id: str,
        body: CreateMentalModelRequest,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Create a mental model (async - returns operation_id)."""
        try:
            result = await app.state.memory.submit_async_create_mental_model(
                bank_id=bank_id,
                name=body.name,
                source_query=body.source_query,
                tags=body.tags if body.tags else None,
                max_tokens=body.max_tokens,
                request_context=request_context,
            )
            return CreateMentalModelResponse(operation_id=result["operation_id"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
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
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except (AuthenticationError, HTTPException):
            raise
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
        description="Update a mental model's name.",
        operation_id="update_mental_model",
        tags=["Mental Models"],
    )
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
                request_context=request_context,
            )
            if mental_model is None:
                raise HTTPException(status_code=404, detail=f"Mental model '{mental_model_id}' not found")
            return MentalModelResponse(**mental_model)
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
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """
        List documents for a memory bank with optional search.

        Args:
            bank_id: Memory Bank ID (from path)
            q: Search query (searches document ID and metadata)
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_documents(
                bank_id=bank_id, search_query=q, limit=limit, offset=offset, request_context=request_context
            )
            return data
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
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/chunks/{chunk_id}: {error_detail}")
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
        description="Get a list of async operations for a specific agent, with optional filtering by status. Results are sorted by most recent first.",
        operation_id="list_operations",
        tags=["Operations"],
    )
    async def api_list_operations(
        bank_id: str,
        status: str | None = Query(default=None, description="Filter by status: pending, completed, or failed"),
        limit: int = Query(default=20, ge=1, le=100, description="Maximum number of operations to return"),
        offset: int = Query(default=0, ge=0, description="Number of operations to skip"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List async operations for a memory bank with optional filtering and pagination."""
        try:
            result = await app.state.memory.list_operations(
                bank_id, status=status, limit=limit, offset=offset, request_context=request_context
            )
            return OperationsListResponse(
                bank_id=bank_id,
                total=result["total"],
                limit=limit,
                offset=offset,
                operations=[OperationResponse(**op) for op in result["operations"]],
            )
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
        bank_id: str, operation_id: str, request_context: RequestContext = Depends(get_request_context)
    ):
        """Get the status of an async operation."""
        try:
            # Validate UUID format
            try:
                uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            result = await app.state.memory.get_operation_status(bank_id, operation_id, request_context=request_context)
            return OperationStatusResponse(**result)
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
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/operations/{operation_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/v1/default/banks/{bank_id}/profile",
        response_model=BankProfileResponse,
        summary="Get memory bank profile",
        description="Get disposition traits and mission for a memory bank. Auto-creates agent with defaults if not exists.",
        operation_id="get_bank_profile",
        tags=["Banks"],
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
    async def api_create_or_update_bank(
        bank_id: str, request: CreateBankRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Create or update an agent with disposition and mission."""
        try:
            # Ensure bank exists by getting profile (auto-creates with defaults)
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name and/or mission if provided (support both mission and deprecated background)
            mission_value = request.mission or request.background
            if request.name is not None or mission_value is not None:
                await app.state.memory.update_bank(
                    bank_id,
                    name=request.name,
                    mission=mission_value,
                    request_context=request_context,
                )

            # Update disposition if provided
            if request.disposition is not None:
                await app.state.memory.update_bank_disposition(
                    bank_id, request.disposition.model_dump(), request_context=request_context
                )

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
    async def api_update_bank(
        bank_id: str, request: CreateBankRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Partially update an agent's profile (name, mission, disposition)."""
        try:
            # Ensure bank exists
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name and/or mission if provided
            mission_value = request.mission or request.background
            if request.name is not None or mission_value is not None:
                await app.state.memory.update_bank(
                    bank_id,
                    name=request.name,
                    mission=mission_value,
                    request_context=request_context,
                )

            # Update disposition if provided
            if request.disposition is not None:
                await app.state.memory.update_bank_disposition(
                    bank_id, request.disposition.model_dump(), request_context=request_context
                )

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
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/v1/default/banks/{bank_id}/observations",
        response_model=DeleteResponse,
        summary="Clear all observations",
        description="Delete all observations for a memory bank. This is useful for resetting the consolidated knowledge.",
        operation_id="clear_observations",
        tags=["Banks"],
    )
    async def api_clear_observations(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Clear all observations for a bank."""
        try:
            result = await app.state.memory.clear_observations(bank_id, request_context=request_context)
            return DeleteResponse(
                success=True,
                message=f"Cleared {result.get('deleted_count', 0)} observations",
                deleted_count=result.get("deleted_count", 0),
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in DELETE /v1/default/banks/{bank_id}/observations: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/v1/default/banks/{bank_id}/consolidate",
        response_model=ConsolidationResponse,
        summary="Trigger consolidation",
        description="Run memory consolidation to create/update observations from recent memories.",
        operation_id="trigger_consolidation",
        tags=["Banks"],
    )
    async def api_trigger_consolidation(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Trigger consolidation for a bank (async)."""
        try:
            result = await app.state.memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
            return ConsolidationResponse(
                operation_id=result["operation_id"],
                deduplicated=result.get("deduplicated", False),
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in POST /v1/default/banks/{bank_id}/consolidate: {error_detail}")
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
    async def api_retain(
        bank_id: str, request: RetainRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Retain memories with optional async processing."""
        metrics = get_metrics_collector()

        try:
            # Prepare contents for processing
            contents = []
            for item in request.items:
                content_dict = {"content": item.content}
                if item.timestamp:
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
                contents.append(content_dict)

            if request.async_:
                # Async processing: queue task and return immediately
                result = await app.state.memory.submit_async_retain(
                    bank_id, contents, document_tags=request.document_tags, request_context=request_context
                )
                return RetainResponse.model_validate(
                    {
                        "success": True,
                        "bank_id": bank_id,
                        "items_count": result["items_count"],
                        "async": True,
                        "operation_id": result["operation_id"],
                    }
                )
            else:
                # Synchronous processing: wait for completion (record metrics)
                with metrics.record_operation("retain", bank_id=bank_id, source="api"):
                    result, usage = await app.state.memory.retain_batch_async(
                        bank_id=bank_id,
                        contents=contents,
                        document_tags=request.document_tags,
                        request_context=request_context,
                        return_usage=True,
                    )

                return RetainResponse.model_validate(
                    {"success": True, "bank_id": bank_id, "items_count": len(contents), "async": False, "usage": usage}
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

    @app.delete(
        "/v1/default/banks/{bank_id}/memories",
        response_model=DeleteResponse,
        summary="Clear memory bank memories",
        description="Delete memory units for a memory bank. Optionally filter by type (world, experience, opinion) to delete only specific types. This is a destructive operation that cannot be undone. The bank profile (disposition and background) will be preserved.",
        operation_id="clear_bank_memories",
        tags=["Memory"],
    )
    async def api_clear_bank_memories(
        bank_id: str,
        type: str | None = Query(None, description="Optional fact type filter (world, experience, opinion)"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Clear memories for a memory bank, optionally filtered by type."""
        try:
            await app.state.memory.delete_bank(bank_id, fact_type=type, request_context=request_context)

            return DeleteResponse(success=True)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))
