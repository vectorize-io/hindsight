"""
FastAPI application factory and API routes for memory system.

This module provides the create_app function to create and configure
the FastAPI application with all API endpoints.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict

from hindsight_api import MemoryEngine
from hindsight_api.engine.db_utils import acquire_with_retry


class MetadataFilter(BaseModel):
    """Filter for metadata fields. Matches records where (key=value) OR (key not set) when match_unset=True."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "key": "source",
            "value": "slack",
            "match_unset": True
        }
    })

    key: str = Field(description="Metadata key to filter on")
    value: Optional[str] = Field(default=None, description="Value to match. If None with match_unset=True, matches any record where key is not set.")
    match_unset: bool = Field(default=True, description="If True, also match records where this metadata key is not set")


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What did Alice say about machine learning?",
            "fact_type": ["world", "agent"],
            "thinking_budget": 100,
            "max_tokens": 4096,
            "trace": True,
            "question_date": "2023-05-30T23:40:00",
            "metadata_filter": [{"key": "source", "value": "slack", "match_unset": True}]
        }
    })

    query: str
    fact_type: Optional[List[str]] = None  # List of fact types to search (defaults to all if not specified)
    thinking_budget: int = 100
    max_tokens: int = 4096
    trace: bool = False
    question_date: Optional[str] = None  # ISO format date string (e.g., "2023-05-30T23:40:00")
    metadata_filter: Optional[List[MetadataFilter]] = Field(default=None, description="Filter by metadata. Multiple filters are ANDed together.")


class SearchResult(BaseModel):
    """Single search result item."""
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "Alice works at Google on the AI team",
                "type": "world",
                "context": "work info",
                "event_date": "2024-01-15T10:30:00Z",
                "document_id": "session_abc123",
                "metadata": {"source": "slack"}
            }
        }
    }

    id: str
    text: str
    type: Optional[str] = None  # fact type: world, agent, opinion
    context: Optional[str] = None
    event_date: Optional[str] = None  # ISO format date string
    document_id: Optional[str] = None  # Document this memory belongs to
    metadata: Optional[Dict[str, str]] = None  # User-defined metadata


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "results": [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "text": "Alice works at Google on the AI team",
                    "type": "world",
                    "context": "work info",
                    "event_date": "2024-01-15T10:30:00Z"
                }
            ],
            "trace": {
                "query": "What did Alice say about machine learning?",
                "num_results": 1,
                "time_seconds": 0.123
            }
        }
    })

    results: List[SearchResult]
    trace: Optional[Dict[str, Any]] = None


class MemoryItem(BaseModel):
    """Single memory item for batch put."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "content": "Alice mentioned she's working on a new ML model",
            "event_date": "2024-01-15T10:30:00Z",
            "context": "team meeting",
            "metadata": {"source": "slack", "channel": "engineering"}
        }
    })

    content: str
    event_date: Optional[datetime] = None
    context: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class BatchPutRequest(BaseModel):
    """Request model for batch put endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [
                {
                    "content": "Alice works at Google",
                    "context": "work"
                },
                {
                    "content": "Bob went hiking yesterday",
                    "event_date": "2024-01-15T10:00:00Z"
                }
            ],
            "document_id": "conversation_123"
        }
    })

    items: List[MemoryItem]
    document_id: Optional[str] = None


class BatchPutResponse(BaseModel):
    """Response model for batch put endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "message": "Successfully stored 2 memory items",
            "agent_id": "user123",
            "document_id": "conversation_123",
            "items_count": 2
        }
    })

    success: bool
    message: str
    agent_id: str
    document_id: Optional[str] = None
    items_count: int


class BatchPutAsyncResponse(BaseModel):
    """Response model for async batch put endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "message": "Batch put task queued for background processing",
            "agent_id": "user123",
            "document_id": "conversation_123",
            "items_count": 2,
            "queued": True
        }
    })

    success: bool
    message: str
    agent_id: str
    document_id: Optional[str] = None
    items_count: int
    queued: bool


class ThinkRequest(BaseModel):
    """Request model for think endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What do you think about artificial intelligence?",
            "thinking_budget": 50,
            "context": "This is for a research paper on AI ethics",
            "metadata_filter": [{"key": "source", "value": "slack", "match_unset": True}]
        }
    })

    query: str
    thinking_budget: int = 50
    context: Optional[str] = None
    metadata_filter: Optional[List[MetadataFilter]] = Field(default=None, description="Filter by metadata. Multiple filters are ANDed together.")


class OpinionItem(BaseModel):
    """Model for an opinion with confidence score."""
    text: str
    confidence: float


class ThinkFact(BaseModel):
    """A fact used in think response."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "text": "AI is used in healthcare",
            "type": "world",
            "context": "healthcare discussion",
            "event_date": "2024-01-15T10:30:00Z"
        }
    })

    id: Optional[str] = None
    text: str
    type: Optional[str] = None  # fact type: world, agent, opinion
    context: Optional[str] = None
    event_date: Optional[str] = None


class ThinkResponse(BaseModel):
    """Response model for think endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "Based on my understanding, AI is a transformative technology...",
            "based_on": [
                {
                    "id": "123",
                    "text": "AI is used in healthcare",
                    "type": "world"
                },
                {
                    "id": "456",
                    "text": "I discussed AI applications last week",
                    "type": "agent"
                }
            ],
            "new_opinions": [
                "AI has great potential when used responsibly"
            ]
        }
    })

    text: str
    based_on: List[ThinkFact] = []  # Facts used to generate the response
    new_opinions: List[str] = []  # Simplified to list of opinion strings


class AgentsResponse(BaseModel):
    """Response model for agents list endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agents": ["user123", "agent_alice", "agent_bob"]
        }
    })

    agents: List[str]


class PersonalityTraits(BaseModel):
    """Personality traits based on Big Five model."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.5,
            "agreeableness": 0.7,
            "neuroticism": 0.3,
            "bias_strength": 0.7
        }
    })

    openness: float = Field(ge=0.0, le=1.0, description="Openness to experience (0-1)")
    conscientiousness: float = Field(ge=0.0, le=1.0, description="Conscientiousness (0-1)")
    extraversion: float = Field(ge=0.0, le=1.0, description="Extraversion (0-1)")
    agreeableness: float = Field(ge=0.0, le=1.0, description="Agreeableness (0-1)")
    neuroticism: float = Field(ge=0.0, le=1.0, description="Neuroticism (0-1)")
    bias_strength: float = Field(ge=0.0, le=1.0, description="How strongly personality influences opinions (0-1)")


class AgentProfileResponse(BaseModel):
    """Response model for agent profile."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agent_id": "user123",
            "name": "Alice",
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.5,
                "agreeableness": 0.7,
                "neuroticism": 0.3,
                "bias_strength": 0.7
            },
            "background": "I am a software engineer with 10 years of experience in startups"
        }
    })

    agent_id: str
    name: str
    personality: PersonalityTraits
    background: str


class UpdatePersonalityRequest(BaseModel):
    """Request model for updating personality traits."""
    personality: PersonalityTraits


class AddBackgroundRequest(BaseModel):
    """Request model for adding/merging background information."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "content": "I was born in Texas",
            "update_personality": True
        }
    })

    content: str = Field(description="New background information to add or merge")
    update_personality: bool = Field(
        default=True,
        description="If true, infer Big Five personality traits from the merged background (default: true)"
    )


class BackgroundResponse(BaseModel):
    """Response model for background update."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "background": "I was born in Texas. I am a software engineer with 10 years of experience.",
            "personality": {
                "openness": 0.7,
                "conscientiousness": 0.6,
                "extraversion": 0.5,
                "agreeableness": 0.8,
                "neuroticism": 0.4,
                "bias_strength": 0.6
            }
        }
    })

    background: str
    personality: Optional[PersonalityTraits] = None


class AgentListItem(BaseModel):
    """Agent list item with profile summary."""
    agent_id: str
    name: str
    personality: PersonalityTraits
    background: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AgentListResponse(BaseModel):
    """Response model for listing all agents."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "agents": [
                {
                    "agent_id": "user123",
                    "name": "Alice",
                    "personality": {
                        "openness": 0.5,
                        "conscientiousness": 0.5,
                        "extraversion": 0.5,
                        "agreeableness": 0.5,
                        "neuroticism": 0.5,
                        "bias_strength": 0.5
                    },
                    "background": "I am a software engineer",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-16T14:20:00Z"
                }
            ]
        }
    })

    agents: List[AgentListItem]


class CreateAgentRequest(BaseModel):
    """Request model for creating/updating an agent."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Alice",
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.5,
                "agreeableness": 0.7,
                "neuroticism": 0.3,
                "bias_strength": 0.7
            },
            "background": "I am a creative software engineer with 10 years of experience"
        }
    })

    name: Optional[str] = None
    personality: Optional[PersonalityTraits] = None
    background: Optional[str] = None


class GraphDataResponse(BaseModel):
    """Response model for graph data endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "nodes": [
                {"id": "1", "label": "Alice works at Google", "type": "world"},
                {"id": "2", "label": "Bob went hiking", "type": "world"}
            ],
            "edges": [
                {"from": "1", "to": "2", "type": "semantic", "weight": 0.8}
            ],
            "table_rows": [
                {"id": "abc12345...", "text": "Alice works at Google", "context": "Work info", "date": "2024-01-15 10:30", "entities": "Alice (PERSON), Google (ORGANIZATION)"}
            ],
            "total_units": 2
        }
    })

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    table_rows: List[Dict[str, Any]]
    total_units: int


class ListMemoryUnitsResponse(BaseModel):
    """Response model for list memory units endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "text": "Alice works at Google on the AI team",
                    "context": "Work conversation",
                    "date": "2024-01-15T10:30:00Z",
                    "fact_type": "world",
                    "entities": "Alice (PERSON), Google (ORGANIZATION)"
                }
            ],
            "total": 150,
            "limit": 100,
            "offset": 0
        }
    })

    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class ListDocumentsResponse(BaseModel):
    """Response model for list documents endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [
                {
                    "id": "session_1",
                    "agent_id": "user123",
                    "content_hash": "abc123",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:30:00Z",
                    "text_length": 5420,
                    "memory_unit_count": 15
                }
            ],
            "total": 50,
            "limit": 100,
            "offset": 0
        }
    })

    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


class DocumentResponse(BaseModel):
    """Response model for get document endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "session_1",
            "agent_id": "user123",
            "original_text": "Full document text here...",
            "content_hash": "abc123",
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
            "memory_unit_count": 15
        }
    })

    id: str
    agent_id: str
    original_text: str
    content_hash: Optional[str]
    created_at: str
    updated_at: str
    memory_unit_count: int


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "message": "Resource deleted successfully"
        }
    })

    success: bool
    message: str


def create_app(memory: MemoryEngine, run_migrations: bool = True, initialize_memory: bool = True) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        memory: MemoryEngine instance (already initialized with required parameters)
        run_migrations: Whether to run database migrations on startup (default: True)
        initialize_memory: Whether to initialize memory system on startup (default: True)

    Returns:
        Configured FastAPI application

    Note:
        When mounting this app as a sub-application, the lifespan events may not fire.
        In that case, you should call memory.initialize() manually before starting the server
        and memory.close() when shutting down.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        Lifespan context manager for startup and shutdown events.
        Note: This only fires when running the app standalone, not when mounted.
        """
        # Startup: Initialize database and memory system
        if initialize_memory:
            await memory.initialize()
            logging.info("Memory system initialized")

        if run_migrations:
            from hindsight_api.migrations import run_migrations as do_migrations
            do_migrations(memory.db_url)
            logging.info("Database migrations applied")



        yield

        # Shutdown: Cleanup memory system
        await memory.close()
        logging.info("Memory system closed")

    app = FastAPI(
        title="Agent Memory API",
        version="1.0.0",
        description="""
A temporal-semantic memory system for AI agents that stores, retrieves, and reasons over memories.

## Features

* **Batch Memory Storage**: Store multiple memories efficiently with automatic fact extraction
* **Semantic Search**: Find relevant memories using natural language queries
* **Fact Type Filtering**: Search across world facts, agent actions, and opinions separately
* **Think Endpoint**: Generate contextual answers based on agent identity and memories
* **Graph Visualization**: Interactive memory graph visualization
* **Document Tracking**: Track and manage memory documents with upsert support

## Architecture

The system uses:
- **Temporal Links**: Connect memories that are close in time
- **Semantic Links**: Connect semantically similar memories
- **Entity Links**: Connect memories that mention the same entities
- **Spreading Activation**: Intelligent traversal for memory retrieval
        """,
        contact={
            "name": "Memory System",
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        lifespan=lifespan
    )

    # IMPORTANT: Set memory on app.state immediately, don't wait for lifespan
    # This is required for mounted sub-applications where lifespan may not fire
    app.state.memory = memory

    # Register all routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI):
    """Register all API routes on the given app instance."""


    @app.get(
        "/api/v1/agents/{agent_id}/graph",
        response_model=GraphDataResponse,
        tags=["Visualization"],
        summary="Get memory graph data",
        description="Retrieve graph data for visualization, optionally filtered by fact_type (world/agent/opinion). Limited to 1000 most recent items.",
        operation_id="get_graph"
    )
    async def api_graph(
        agent_id: str,
        fact_type: Optional[str] = None
    ):
        """Get graph data from database, filtered by agent_id and optionally by fact_type."""
        try:
            data = await app.state.memory.get_graph_data(agent_id, fact_type)
            return data
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/graph: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/v1/agents/{agent_id}/memories/list",
        response_model=ListMemoryUnitsResponse,
        tags=["Memory Operations"],
        summary="List memory units",
        description="List memory units with pagination and optional full-text search. Supports filtering by fact_type.",
        operation_id="list_memories"
    )
    async def api_list(
        agent_id: str,
        fact_type: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List memory units for table view with optional full-text search.

        Args:
            agent_id: Agent ID (from path)
            fact_type: Filter by fact type (world, agent, opinion)
            q: Search query for full-text search (searches text and context)
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_memory_units(
                agent_id=agent_id,
                fact_type=fact_type,
                search_query=q,
                limit=limit,
                offset=offset
            )
            return data
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories/list: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/v1/agents/{agent_id}/memories/search",
        response_model=SearchResponse,
        tags=["Memory Operations"],
        summary="Search memory",
        description="""
    Search memory using semantic similarity and spreading activation.

    The fact_type parameter is optional and must be one of:
    - 'world': General knowledge about people, places, events, and things that happen
    - 'agent': Memories about what the AI agent did, actions taken, and tasks performed
    - 'opinion': The agent's formed beliefs, perspectives, and viewpoints
        """,
        operation_id="search_memories"
    )
    async def api_search(agent_id: str, request: SearchRequest):
        """Run a search and return results with trace."""
        try:
            # Validate fact_type(s)
            valid_fact_types = ["world", "agent", "opinion"]

            # Default to all fact types if not specified
            if not request.fact_type:
                request.fact_type = valid_fact_types
            else:
                for ft in request.fact_type:
                    if ft not in valid_fact_types:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid fact_type '{ft}'. Must be one of: {', '.join(valid_fact_types)}"
                        )

            # Parse question_date if provided
            question_date = None
            if request.question_date:
                try:
                    question_date = datetime.fromisoformat(request.question_date.replace('Z', '+00:00'))
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid question_date format. Expected ISO format (e.g., '2023-05-30T23:40:00'): {str(e)}"
                    )

            # Run search with tracing
            core_result = await app.state.memory.search_async(
                agent_id=agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
                enable_trace=request.trace,
                fact_type=request.fact_type,
                question_date=question_date
            )

            # Convert core MemoryFact objects to API SearchResult objects (excluding internal metrics)
            search_results = [
                SearchResult(
                    id=fact.id,
                    text=fact.text,
                    type=fact.fact_type,
                    context=fact.context,
                    event_date=fact.event_date
                )
                for fact in core_result.results
            ]

            return SearchResponse(
                results=search_results,
                trace=core_result.trace
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories/search: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/v1/agents/{agent_id}/think",
        response_model=ThinkResponse,
        tags=["Reasoning"],
        summary="Think and generate answer",
        description="""
    Think and formulate an answer using agent identity, world facts, and opinions.

    This endpoint:
    1. Retrieves agent facts (agent's identity)
    2. Retrieves world facts relevant to the query
    3. Retrieves existing opinions (agent's perspectives)
    4. Uses LLM to formulate a contextual answer
    5. Extracts and stores any new opinions formed
    6. Returns plain text answer, the facts used, and new opinions
        """,
        operation_id="think"
    )
    async def api_think(agent_id: str, request: ThinkRequest):
        try:
            # Use the memory system's think_async method
            core_result = await app.state.memory.think_async(
                agent_id=agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                context=request.context
            )

            # Convert core MemoryFact objects to API ThinkFact objects (excluding internal metrics)
            based_on_facts = []
            for fact_type, facts in core_result.based_on.items():
                for fact in facts:
                    based_on_facts.append(ThinkFact(
                        id=fact.id,
                        text=fact.text,
                        type=fact.fact_type,
                        context=fact.context,
                        event_date=fact.event_date
                    ))

            return ThinkResponse(
                text=core_result.text,
                based_on=based_on_facts,
                new_opinions=core_result.new_opinions
            )

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/think: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/v1/agents",
        response_model=AgentListResponse,
        tags=["Agent Management"],
        summary="List all agents",
        description="Get a list of all agents with their profiles",
        operation_id="list_agents"
    )
    async def api_agents():
        """Get list of all agents with their profiles."""
        try:
            agents = await app.state.memory.list_agents()
            return AgentListResponse(agents=agents)
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/agents/{agent_id}/stats",
        tags=["Agent Management"],
        summary="Get memory statistics for an agent",
        description="Get statistics about nodes and links for a specific agent",
        operation_id="get_agent_stats"
    )
    async def api_stats(agent_id: str):
        """Get statistics about memory nodes and links for an agent."""
        try:
            pool = await app.state.memory._get_pool()
            async with acquire_with_retry(pool) as conn:
                # Get node counts by fact_type
                node_stats = await conn.fetch(
                    """
                    SELECT fact_type, COUNT(*) as count
                    FROM memory_units
                    WHERE agent_id = $1
                    GROUP BY fact_type
                    """,
                    agent_id
                )

                # Get link counts by link_type
                link_stats = await conn.fetch(
                    """
                    SELECT ml.link_type, COUNT(*) as count
                    FROM memory_links ml
                    JOIN memory_units mu ON ml.from_unit_id = mu.id
                    WHERE mu.agent_id = $1
                    GROUP BY ml.link_type
                    """,
                    agent_id
                )

                # Get link counts by fact_type (from nodes)
                link_fact_type_stats = await conn.fetch(
                    """
                    SELECT mu.fact_type, COUNT(*) as count
                    FROM memory_links ml
                    JOIN memory_units mu ON ml.from_unit_id = mu.id
                    WHERE mu.agent_id = $1
                    GROUP BY mu.fact_type
                    """,
                    agent_id
                )

                # Get link counts by fact_type AND link_type
                link_breakdown_stats = await conn.fetch(
                    """
                    SELECT mu.fact_type, ml.link_type, COUNT(*) as count
                    FROM memory_links ml
                    JOIN memory_units mu ON ml.from_unit_id = mu.id
                    WHERE mu.agent_id = $1
                    GROUP BY mu.fact_type, ml.link_type
                    """,
                    agent_id
                )

                # Get pending and failed operations counts
                ops_stats = await conn.fetch(
                    """
                    SELECT status, COUNT(*) as count
                    FROM async_operations
                    WHERE agent_id = $1
                    GROUP BY status
                    """,
                    agent_id
                )
                ops_by_status = {row['status']: row['count'] for row in ops_stats}
                pending_operations = ops_by_status.get('pending', 0)
                failed_operations = ops_by_status.get('failed', 0)

                # Get document count
                doc_count_result = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as count
                    FROM documents
                    WHERE agent_id = $1
                    """,
                    agent_id
                )
                total_documents = doc_count_result['count'] if doc_count_result else 0

                # Format results
                nodes_by_type = {row['fact_type']: row['count'] for row in node_stats}
                links_by_type = {row['link_type']: row['count'] for row in link_stats}
                links_by_fact_type = {row['fact_type']: row['count'] for row in link_fact_type_stats}

                # Build detailed breakdown: {fact_type: {link_type: count}}
                links_breakdown = {}
                for row in link_breakdown_stats:
                    fact_type = row['fact_type']
                    link_type = row['link_type']
                    count = row['count']
                    if fact_type not in links_breakdown:
                        links_breakdown[fact_type] = {}
                    links_breakdown[fact_type][link_type] = count

                total_nodes = sum(nodes_by_type.values())
                total_links = sum(links_by_type.values())

                return {
                    "agent_id": agent_id,
                    "total_nodes": total_nodes,
                    "total_links": total_links,
                    "total_documents": total_documents,
                    "nodes_by_fact_type": nodes_by_type,
                    "links_by_link_type": links_by_type,
                    "links_by_fact_type": links_by_fact_type,
                    "links_breakdown": links_breakdown,
                    "pending_operations": pending_operations,
                    "failed_operations": failed_operations
                }

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/stats: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/v1/agents/{agent_id}/documents",
        response_model=ListDocumentsResponse,
        tags=["Documents"],
        summary="List documents",
        description="List documents with pagination and optional search. Documents are the source content from which memory units are extracted.",
        operation_id="list_documents"
    )
    async def api_list_documents(
        agent_id: str,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List documents for an agent with optional search.

        Args:
            agent_id: Agent ID (from path)
            q: Search query (searches document ID and metadata)
            limit: Maximum number of results (default: 100)
            offset: Offset for pagination (default: 0)
        """
        try:
            data = await app.state.memory.list_documents(
                agent_id=agent_id,
                search_query=q,
                limit=limit,
                offset=offset
            )
            return data
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/documents: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/v1/agents/{agent_id}/documents/{document_id}",
        response_model=DocumentResponse,
        tags=["Documents"],
        summary="Get document details",
        description="Get a specific document including its original text",
        operation_id="get_document"
    )
    async def api_get_document(
        agent_id: str,
        document_id: str
    ):
        """
        Get a specific document with its original text.

        Args:
            agent_id: Agent ID (from path)
            document_id: Document ID (from path)
        """
        try:
            document = await app.state.memory.get_document(document_id, agent_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            return document
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/v1/agents/{agent_id}/documents/{document_id}",
        tags=["Documents"],
        summary="Delete a document",
        description="""
Delete a document and all its associated memory units and links.

This will cascade delete:
- The document itself
- All memory units extracted from this document
- All links (temporal, semantic, entity) associated with those memory units

This operation cannot be undone.
        """,
        operation_id="delete_document"
    )
    async def api_delete_document(
        agent_id: str,
        document_id: str
    ):
        """
        Delete a document and all its associated memory units and links.

        Args:
            agent_id: Agent ID (from path)
            document_id: Document ID to delete (from path)
        """
        try:
            result = await app.state.memory.delete_document(document_id, agent_id)

            if result["document_deleted"] == 0:
                raise HTTPException(status_code=404, detail="Document not found")

            return {
                "success": True,
                "message": f"Document '{document_id}' and {result['memory_units_deleted']} associated memory units deleted successfully",
                "document_id": document_id,
                "memory_units_deleted": result["memory_units_deleted"]
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/v1/agents/{agent_id}/memories",
        response_model=BatchPutResponse,
        tags=["Memory Operations"],
        summary="Store multiple memories",
        description="""
    Store multiple memory items in batch with automatic fact extraction.

    Features:
    - Efficient batch processing
    - Automatic fact extraction from natural language
    - Entity recognition and linking
    - Document tracking with automatic upsert (when document_id is provided)
    - Temporal and semantic linking

    The system automatically:
    1. Extracts semantic facts from the content
    2. Generates embeddings
    3. Deduplicates similar facts
    4. Creates temporal, semantic, and entity links
    5. Tracks document metadata

    Note: If document_id is provided and already exists, the old document and its memory units will be deleted before creating new ones (upsert behavior).
        """,
        operation_id="batch_put_memories"
    )
    async def api_batch_put(agent_id: str, request: BatchPutRequest):
        try:
            # Prepare contents for put_batch_async
            contents = []
            for item in request.items:
                content_dict = {"content": item.content}
                if item.event_date:
                    content_dict["event_date"] = item.event_date
                if item.context:
                    content_dict["context"] = item.context
                contents.append(content_dict)

            # Call put_batch_async
            result = await app.state.memory.put_batch_async(
                agent_id=agent_id,
                contents=contents,
                document_id=request.document_id
            )


            return BatchPutResponse(
                success=True,
                message=f"Successfully stored {len(contents)} memory items",
                agent_id=agent_id,
                document_id=request.document_id,
                items_count=len(contents)
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/v1/agents/{agent_id}/memories/async",
        response_model=BatchPutAsyncResponse,
        tags=["Memory Operations"],
        summary="Store multiple memories asynchronously",
        description="""
    Store multiple memory items in batch asynchronously using the task backend.

    This endpoint returns immediately after queuing the task, without waiting for completion.
    The actual processing happens in the background.

    Features:
    - Immediate response (non-blocking)
    - Background processing via task queue
    - Efficient batch processing
    - Automatic fact extraction from natural language
    - Entity recognition and linking
    - Document tracking with automatic upsert (when document_id is provided)
    - Temporal and semantic linking

    The system automatically:
    1. Queues the batch put task
    2. Returns immediately with success=True, queued=True
    3. Processes in background: extracts facts, generates embeddings, creates links

    Note: If document_id is provided and already exists, the old document and its memory units will be deleted before creating new ones (upsert behavior).
        """,
        operation_id="batch_put_async"
    )
    async def api_batch_put_async(agent_id: str, request: BatchPutRequest):
        try:
            # Prepare contents for put_batch_async
            contents = []
            for item in request.items:
                content_dict = {"content": item.content}
                if item.event_date:
                    content_dict["event_date"] = item.event_date
                if item.context:
                    content_dict["context"] = item.context
                contents.append(content_dict)

            # Generate UUID for this operation
            operation_id = uuid.uuid4()

            # Insert operation record into database BEFORE scheduling task
            pool = await app.state.memory._get_pool()
            async with acquire_with_retry(pool) as conn:
                await conn.execute(
                    """
                    INSERT INTO async_operations (id, agent_id, task_type, items_count, document_id)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    operation_id,
                    agent_id,
                    'batch_put',
                    len(contents),
                    request.document_id
                )

            # Submit task to background queue with operation_id
            await app.state.memory._task_backend.submit_task({
                'type': 'batch_put',
                'operation_id': str(operation_id),
                'agent_id': agent_id,
                'contents': contents,
                'document_id': request.document_id
            })

            logging.info(f"Batch put task queued for agent_id={agent_id}, {len(contents)} items, operation_id={operation_id}")

            return BatchPutAsyncResponse(
                success=True,
                message=f"Batch put task queued for background processing ({len(contents)} items)",
                agent_id=agent_id,
                document_id=request.document_id,
                items_count=len(contents),
                queued=True
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories/async: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/v1/agents/{agent_id}/operations",
        tags=["Memory Operations"],
        summary="List async operations",
        description="Get a list of all async operations (pending and failed) for a specific agent, including error messages for failed operations",
        operation_id="list_operations"
    )
    async def api_list_operations(agent_id: str):
        """List all async operations (pending and failed) for an agent."""
        try:
            pool = await app.state.memory._get_pool()
            async with acquire_with_retry(pool) as conn:
                operations = await conn.fetch(
                    """
                    SELECT id, agent_id, task_type, items_count, document_id, created_at, status, error_message
                    FROM async_operations
                    WHERE agent_id = $1
                    ORDER BY created_at ASC
                    """,
                    agent_id
                )

                return {
                    "agent_id": agent_id,
                    "operations": [
                        {
                            "id": str(row['id']),
                            "task_type": row['task_type'],
                            "items_count": row['items_count'],
                            "document_id": row['document_id'],
                            "created_at": row['created_at'].isoformat(),
                            "status": row['status'],
                            "error_message": row['error_message']
                        }
                        for row in operations
                    ]
                }

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/operations: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/v1/agents/{agent_id}/operations/{operation_id}",
        tags=["Memory Operations"],
        summary="Cancel a pending async operation",
        description="Cancel a pending async operation by removing it from the queue",
        operation_id="cancel_operation"
    )
    async def api_cancel_operation(agent_id: str, operation_id: str):
        """Cancel a pending async operation."""
        try:
            # Validate UUID format
            try:
                op_uuid = uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            pool = await app.state.memory._get_pool()
            async with acquire_with_retry(pool) as conn:
                # Check if operation exists and belongs to this agent
                result = await conn.fetchrow(
                    "SELECT agent_id FROM async_operations WHERE id = $1 AND agent_id = $2",
                    op_uuid,
                    agent_id
                )

                if not result:
                    raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found for agent {agent_id}")

                # Delete the operation
                await conn.execute(
                    "DELETE FROM async_operations WHERE id = $1",
                    op_uuid
                )

                return {
                    "success": True,
                    "message": f"Operation {operation_id} cancelled",
                    "operation_id": operation_id,
                    "agent_id": agent_id
                }

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/operations/{operation_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/v1/agents/{agent_id}/memories/{unit_id}",
        tags=["Memory Operations"],
        summary="Delete a memory unit",
        description="Delete a single memory unit and all its associated links (temporal, semantic, and entity links)",
        operation_id="delete_memory_unit"
    )
    async def api_delete_memory_unit(agent_id: str, unit_id: str):
        """Delete a memory unit and all its links."""
        try:
            result = await app.state.memory.delete_memory_unit(unit_id)

            if not result["success"]:
                raise HTTPException(status_code=404, detail=result["message"])

            return result
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories/{unit_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    # Agent Profile Endpoints

    @app.get(
        "/api/v1/agents/{agent_id}/profile",
        response_model=AgentProfileResponse,
        tags=["Agent Management"],
        summary="Get agent profile",
        description="Get personality traits and background for an agent. Auto-creates agent with defaults if not exists.",
        operation_id="get_agent_profile"
    )
    async def api_get_agent_profile(agent_id: str):
        """Get agent profile (personality + background)."""
        try:
            profile = await app.state.memory.get_agent_profile(agent_id)
            return AgentProfileResponse(
                agent_id=agent_id,
                name=profile["name"],
                personality=PersonalityTraits(**profile["personality"]),
                background=profile["background"]
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/profile: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.put(
        "/api/v1/agents/{agent_id}/profile",
        response_model=AgentProfileResponse,
        tags=["Agent Management"],
        summary="Update agent personality",
        description="Update agent's Big Five personality traits and bias strength",
        operation_id="update_agent_personality"
    )
    async def api_update_agent_personality(
        agent_id: str,
        request: UpdatePersonalityRequest
    ):
        """Update agent personality traits."""
        try:
            # Update personality
            await app.state.memory.update_agent_personality(
                agent_id,
                request.personality.model_dump()
            )

            # Get updated profile
            profile = await app.state.memory.get_agent_profile(agent_id)
            return AgentProfileResponse(
                agent_id=agent_id,
                name=profile["name"],
                personality=PersonalityTraits(**profile["personality"]),
                background=profile["background"]
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/profile: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/v1/agents/{agent_id}/background",
        response_model=BackgroundResponse,
        tags=["Agent Management"],
        summary="Add/merge agent background",
        description="Add new background information or merge with existing. LLM intelligently resolves conflicts, normalizes to first person, and optionally infers personality traits.",
        operation_id="add_agent_background"
    )
    async def api_add_agent_background(
        agent_id: str,
        request: AddBackgroundRequest
    ):
        """Add or merge agent background information. Optionally infer personality traits."""
        try:
            result = await app.state.memory.merge_agent_background(
                agent_id,
                request.content,
                update_personality=request.update_personality
            )

            response = BackgroundResponse(background=result["background"])
            if "personality" in result:
                response.personality = PersonalityTraits(**result["personality"])

            return response
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/background: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.put(
        "/api/v1/agents/{agent_id}",
        response_model=AgentProfileResponse,
        tags=["Agent Management"],
        summary="Create or update agent",
        description="Create a new agent or update existing agent with personality and background. Auto-fills missing fields with defaults.",
        operation_id="create_or_update_agent"
    )
    async def api_create_or_update_agent(
        agent_id: str,
        request: CreateAgentRequest
    ):
        """Create or update an agent with personality and background."""
        try:
            # Get existing profile or create with defaults
            profile = await app.state.memory.get_agent_profile(agent_id)

            # Update name if provided
            if request.name is not None:
                pool = await app.state.memory._get_pool()
                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        """
                        UPDATE agents
                        SET name = $2,
                            updated_at = NOW()
                        WHERE agent_id = $1
                        """,
                        agent_id,
                        request.name
                    )
                profile["name"] = request.name

            # Update personality if provided
            if request.personality is not None:
                await app.state.memory.update_agent_personality(
                    agent_id,
                    request.personality.model_dump()
                )
                profile["personality"] = request.personality.model_dump()

            # Update background if provided (replace, not merge)
            if request.background is not None:
                pool = await app.state.memory._get_pool()
                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        """
                        UPDATE agents
                        SET background = $2,
                            updated_at = NOW()
                        WHERE agent_id = $1
                        """,
                        agent_id,
                        request.background
                    )
                profile["background"] = request.background

            # Get final profile
            final_profile = await app.state.memory.get_agent_profile(agent_id)
            return AgentProfileResponse(
                agent_id=agent_id,
                name=final_profile["name"],
                personality=PersonalityTraits(**final_profile["personality"]),
                background=final_profile["background"]
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/v1/agents/{agent_id}/memories",
        response_model=DeleteResponse,
        tags=["Agent Management"],
        summary="Clear agent memories",
        description="Delete memory units for an agent. Optionally filter by fact_type (world, agent, opinion) to delete only specific types. This is a destructive operation that cannot be undone. The agent profile (personality and background) will be preserved.",
        operation_id="clear_agent_memories"
    )
    async def api_clear_agent_memories(
        agent_id: str,
        fact_type: Optional[str] = Query(None, description="Optional fact type filter (world, agent, opinion)")
    ):
        """Clear memories for an agent, optionally filtered by fact_type."""
        try:
            result = await app.state.memory.delete_agent(agent_id, fact_type=fact_type)

            units_deleted = result.get('memory_units_deleted', 0)
            entities_deleted = result.get('entities_deleted', 0)

            if fact_type:
                message = f"Cleared {units_deleted} {fact_type} memories for agent '{agent_id}'"
            else:
                message = f"Cleared all memories for agent '{agent_id}': {units_deleted} memory units, {entities_deleted} entities deleted"

            return DeleteResponse(
                success=True,
                message=message
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/v1/agents/{agent_id}/memories: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))
