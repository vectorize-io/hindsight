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
from typing import Any

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
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES
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
            }
        }
    )

    query: str
    types: list[str] | None = Field(
        default=None, description="List of fact types to recall (defaults to all if not specified)"
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
                ]
            }
        }
    )

    items: list[EntityListItem]


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
            }
        },
    )

    content: str
    timestamp: datetime | None = None
    context: str | None = None
    metadata: dict[str, str] | None = None
    document_id: str | None = Field(default=None, description="Optional document ID for this memory item.")

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
            }
        }
    )

    items: list[MemoryItem]
    async_: bool = Field(
        default=False,
        alias="async",
        description="If true, process asynchronously in background. If false, wait for completion (default: false)",
    )


class RetainResponse(BaseModel):
    """Response model for retain endpoint."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {"success": True, "bank_id": "user123", "items_count": 2, "async": False}},
    )

    success: bool
    bank_id: str
    items_count: int
    is_async: bool = Field(
        alias="async", serialization_alias="async", description="Whether the operation was processed asynchronously"
    )


class FactsIncludeOptions(BaseModel):
    """Options for including facts (based_on) in reflect results."""

    pass  # No additional options needed, just enable/disable


class ReflectIncludeOptions(BaseModel):
    """Options for including additional data in reflect results."""

    facts: FactsIncludeOptions | None = Field(
        default=None,
        description="Include facts that the answer is based on. Set to {} to enable, null to disable (default: disabled).",
    )


class ReflectRequest(BaseModel):
    """Request model for reflect endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What do you think about artificial intelligence?",
                "budget": "low",
                "context": "This is for a research paper on AI ethics",
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
            }
        }
    )

    query: str
    budget: Budget = Budget.LOW
    context: str | None = None
    max_tokens: int = Field(default=4096, description="Maximum tokens for the response")
    include: ReflectIncludeOptions = Field(
        default_factory=ReflectIncludeOptions, description="Options for including additional data (disabled by default)"
    )
    response_schema: dict | None = Field(
        default=None,
        description="Optional JSON Schema for structured output. When provided, the response will include a 'structured_output' field with the LLM response parsed according to this schema.",
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


class ReflectResponse(BaseModel):
    """Response model for think endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Based on my understanding, AI is a transformative technology...",
                "based_on": [
                    {"id": "123", "text": "AI is used in healthcare", "type": "world"},
                    {"id": "456", "text": "I discussed AI applications last week", "type": "experience"},
                ],
                "structured_output": {
                    "summary": "AI is transformative",
                    "key_points": ["Used in healthcare", "Discussed recently"],
                },
            }
        }
    )

    text: str
    based_on: list[ReflectFact] = []  # Facts used to generate the response
    structured_output: dict | None = Field(
        default=None,
        description="Structured output parsed according to the request's response_schema. Only present when response_schema was provided in the request.",
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
                "background": "I am a software engineer with 10 years of experience in startups",
            }
        }
    )

    bank_id: str
    name: str
    disposition: DispositionTraits
    background: str


class UpdateDispositionRequest(BaseModel):
    """Request model for updating disposition traits."""

    disposition: DispositionTraits


class AddBackgroundRequest(BaseModel):
    """Request model for adding/merging background information."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"content": "I was born in Texas", "update_disposition": True}}
    )

    content: str = Field(description="New background information to add or merge")
    update_disposition: bool = Field(
        default=True, description="If true, infer disposition traits from the merged background (default: true)"
    )


class BackgroundResponse(BaseModel):
    """Response model for background update."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "background": "I was born in Texas. I am a software engineer with 10 years of experience.",
                "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
            }
        }
    )

    background: str
    disposition: DispositionTraits | None = None


class BankListItem(BaseModel):
    """Bank list item with profile summary."""

    bank_id: str
    name: str | None = None
    disposition: DispositionTraits
    background: str | None = None
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
                        "background": "I am a software engineer",
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
                "background": "I am a creative software engineer with 10 years of experience",
            }
        }
    )

    name: str | None = None
    disposition: DispositionTraits | None = None
    background: str | None = None


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
            }
        }
    )

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    table_rows: list[dict[str, Any]]
    total_units: int


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


class OperationResponse(BaseModel):
    """Response model for a single async operation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "task_type": "retain",
                "items_count": 5,
                "document_id": "meeting-notes-2024",
                "created_at": "2024-01-15T10:30:00Z",
                "status": "pending",
                "error_message": None,
            }
        }
    )

    id: str
    task_type: str
    items_count: int
    document_id: str | None
    created_at: str
    status: str
    error_message: str | None


class OperationsListResponse(BaseModel):
    """Response model for list operations endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bank_id": "user123",
                "operations": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "task_type": "retain",
                        "items_count": 5,
                        "document_id": None,
                        "created_at": "2024-01-15T10:30:00Z",
                        "status": "pending",
                        "error_message": None,
                    }
                ],
            }
        }
    )

    bank_id: str
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

        # Call HTTP extension startup hook
        if http_extension:
            await http_extension.on_startup()
            logging.info("HTTP extension started")

        yield

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
        description="Retrieve graph data for visualization, optionally filtered by type (world/experience/opinion). Limited to 1000 most recent items.",
        operation_id="get_graph",
        tags=["Memory"],
    )
    async def api_graph(
        bank_id: str, type: str | None = None, request_context: RequestContext = Depends(get_request_context)
    ):
        """Get graph data from database, filtered by bank_id and optionally by type."""
        try:
            data = await app.state.memory.get_graph_data(bank_id, type, request_context=request_context)
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
        metrics = get_metrics_collector()

        try:
            # Default to world, experience, opinion if not specified (exclude observation by default)
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

            # Run recall with tracing (record metrics)
            with metrics.record_operation(
                "recall", bank_id=bank_id, budget=request.budget.value, max_tokens=request.max_tokens
            ):
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

            return RecallResponse(
                results=recall_results, trace=core_result.trace, entities=entities_response, chunks=chunks_response
            )
        except HTTPException:
            raise
        except OperationValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.reason)
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/memories/recall: {error_detail}")
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
            # Use the memory system's reflect_async method (record metrics)
            with metrics.record_operation("reflect", bank_id=bank_id, budget=request.budget.value):
                core_result = await app.state.memory.reflect_async(
                    bank_id=bank_id,
                    query=request.query,
                    budget=request.budget,
                    context=request.context,
                    max_tokens=request.max_tokens,
                    response_schema=request.response_schema,
                    request_context=request_context,
                )

            # Convert core MemoryFact objects to API ReflectFact objects if facts are requested
            based_on_facts = []
            if request.include.facts is not None:
                for fact_type, facts in core_result.based_on.items():
                    for fact in facts:
                        based_on_facts.append(
                            ReflectFact(
                                id=fact.id,
                                text=fact.text,
                                type=fact.fact_type,
                                context=fact.context,
                                occurred_start=fact.occurred_start,
                                occurred_end=fact.occurred_end,
                            )
                        )

            return ReflectResponse(
                text=core_result.text,
                based_on=based_on_facts,
                structured_output=core_result.structured_output,
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
    async def api_stats(bank_id: str):
        """Get statistics about memory nodes and links for a memory bank."""
        try:
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
        description="List all entities (people, organizations, etc.) known by the bank, ordered by mention count.",
        operation_id="list_entities",
        tags=["Entities"],
    )
    async def api_list_entities(
        bank_id: str,
        limit: int = Query(default=100, description="Maximum number of entities to return"),
        request_context: RequestContext = Depends(get_request_context),
    ):
        """List entities for a memory bank."""
        try:
            entities = await app.state.memory.list_entities(bank_id, limit=limit, request_context=request_context)
            return EntityListResponse(items=[EntityListItem(**e) for e in entities])
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
        summary="Regenerate entity observations",
        description="Regenerate observations for an entity based on all facts mentioning it.",
        operation_id="regenerate_entity_observations",
        tags=["Entities"],
    )
    async def api_regenerate_entity_observations(
        bank_id: str,
        entity_id: str,
        request_context: RequestContext = Depends(get_request_context),
    ):
        """Regenerate observations for an entity."""
        try:
            # Get the entity to verify it exists and get canonical_name
            entity = await app.state.memory.get_entity(bank_id, entity_id, request_context=request_context)

            if entity is None:
                raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

            # Regenerate observations
            await app.state.memory.regenerate_entity_observations(
                bank_id=bank_id,
                entity_id=entity_id,
                entity_name=entity["canonical_name"],
                request_context=request_context,
            )

            # Get updated entity with new observations
            entity = await app.state.memory.get_entity(bank_id, entity_id, request_context=request_context)

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
            logger.error(f"Error in /v1/default/banks/{bank_id}/entities/{entity_id}/regenerate: {error_detail}")
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
        "/v1/default/banks/{bank_id}/documents/{document_id}",
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
        "/v1/default/chunks/{chunk_id}",
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
        "/v1/default/banks/{bank_id}/documents/{document_id}",
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
        description="Get a list of all async operations (pending and failed) for a specific agent, including error messages for failed operations",
        operation_id="list_operations",
        tags=["Operations"],
    )
    async def api_list_operations(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """List all async operations (pending and failed) for a memory bank."""
        try:
            operations = await app.state.memory.list_operations(bank_id, request_context=request_context)
            return OperationsListResponse(
                bank_id=bank_id,
                operations=[OperationResponse(**op) for op in operations],
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}/operations: {error_detail}")
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
        description="Get disposition traits and background for a memory bank. Auto-creates agent with defaults if not exists.",
        operation_id="get_bank_profile",
        tags=["Banks"],
    )
    async def api_get_bank_profile(bank_id: str, request_context: RequestContext = Depends(get_request_context)):
        """Get memory bank profile (disposition + background)."""
        try:
            profile = await app.state.memory.get_bank_profile(bank_id, request_context=request_context)
            # Convert DispositionTraits object to dict for Pydantic
            disposition_dict = (
                profile["disposition"].model_dump()
                if hasattr(profile["disposition"], "model_dump")
                else dict(profile["disposition"])
            )
            return BankProfileResponse(
                bank_id=bank_id,
                name=profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                background=profile["background"],
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
            return BankProfileResponse(
                bank_id=bank_id,
                name=profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                background=profile["background"],
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
        summary="Add/merge memory bank background",
        description="Add new background information or merge with existing. LLM intelligently resolves conflicts, normalizes to first person, and optionally infers disposition traits.",
        operation_id="add_bank_background",
        tags=["Banks"],
    )
    async def api_add_bank_background(
        bank_id: str, request: AddBackgroundRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Add or merge bank background information. Optionally infer disposition traits."""
        try:
            result = await app.state.memory.merge_bank_background(
                bank_id, request.content, update_disposition=request.update_disposition, request_context=request_context
            )

            response = BackgroundResponse(background=result["background"])
            if "disposition" in result:
                response.disposition = DispositionTraits(**result["disposition"])

            return response
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
        description="Create a new agent or update existing agent with disposition and background. Auto-fills missing fields with defaults.",
        operation_id="create_or_update_bank",
        tags=["Banks"],
    )
    async def api_create_or_update_bank(
        bank_id: str, request: CreateBankRequest, request_context: RequestContext = Depends(get_request_context)
    ):
        """Create or update an agent with disposition and background."""
        try:
            # Ensure bank exists by getting profile (auto-creates with defaults)
            await app.state.memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name and/or background if provided
            if request.name is not None or request.background is not None:
                await app.state.memory.update_bank(
                    bank_id,
                    name=request.name,
                    background=request.background,
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
            return BankProfileResponse(
                bank_id=bank_id,
                name=final_profile["name"],
                disposition=DispositionTraits(**disposition_dict),
                background=final_profile["background"],
            )
        except (AuthenticationError, HTTPException):
            raise
        except Exception as e:
            import traceback

            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(f"Error in /v1/default/banks/{bank_id}: {error_detail}")
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
                contents.append(content_dict)

            if request.async_:
                # Async processing: queue task and return immediately
                result = await app.state.memory.submit_async_retain(bank_id, contents, request_context=request_context)
                return RetainResponse.model_validate(
                    {
                        "success": True,
                        "bank_id": bank_id,
                        "items_count": result["items_count"],
                        "async": True,
                    }
                )
            else:
                # Synchronous processing: wait for completion (record metrics)
                with metrics.record_operation("retain", bank_id=bank_id):
                    result = await app.state.memory.retain_batch_async(
                        bank_id=bank_id, contents=contents, request_context=request_context
                    )

                return RetainResponse.model_validate(
                    {"success": True, "bank_id": bank_id, "items_count": len(contents), "async": False}
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
