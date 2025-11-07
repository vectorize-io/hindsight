"""
FastAPI application factory and API routes for memory system.

This module provides the create_app function to create and configure
the FastAPI application with all API endpoints.
"""
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from memora import TemporalSemanticMemory


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    fact_type: str
    agent_id: str = "default"
    thinking_budget: int = 100
    max_tokens: int = 4096
    reranker: str = "heuristic"
    trace: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What did Alice say about machine learning?",
                "fact_type": "world",
                "agent_id": "user123",
                "thinking_budget": 100,
                "max_tokens": 4096,
                "reranker": "heuristic",
                "trace": True
            }
        }


class SearchResponse(BaseModel):
    """Response model for search endpoints."""
    results: List[Dict[str, Any]]
    trace: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "text": "Alice works at Google on the AI team",
                        "score": 0.95,
                        "id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                ],
                "trace": {
                    "query": "What did Alice say about machine learning?",
                    "num_results": 1,
                    "time_seconds": 0.123
                }
            }
        }


class MemoryItem(BaseModel):
    """Single memory item for batch put."""
    content: str
    event_date: Optional[datetime] = None
    context: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Alice mentioned she's working on a new ML model",
                "event_date": "2024-01-15T10:30:00Z",
                "context": "team meeting"
            }
        }


class BatchPutRequest(BaseModel):
    """Request model for batch put endpoint."""
    agent_id: str
    items: List[MemoryItem]
    document_id: Optional[str] = None
    document_metadata: Optional[Dict[str, Any]] = None
    upsert: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "user123",
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
                "document_id": "conversation_123",
                "upsert": False
            }
        }


class BatchPutResponse(BaseModel):
    """Response model for batch put endpoint."""
    success: bool
    message: str
    agent_id: str
    document_id: Optional[str] = None
    items_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully stored 2 memory items",
                "agent_id": "user123",
                "document_id": "conversation_123",
                "items_count": 2
            }
        }


class BatchPutAsyncResponse(BaseModel):
    """Response model for async batch put endpoint."""
    success: bool
    message: str
    agent_id: str
    document_id: Optional[str] = None
    items_count: int
    queued: bool

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Batch put task queued for background processing",
                "agent_id": "user123",
                "document_id": "conversation_123",
                "items_count": 2,
                "queued": True
            }
        }


class ThinkRequest(BaseModel):
    """Request model for think endpoint."""
    query: str
    agent_id: str = "default"
    thinking_budget: int = 50

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What do you think about artificial intelligence?",
                "agent_id": "user123",
                "thinking_budget": 50
            }
        }


class OpinionItem(BaseModel):
    """Model for an opinion with confidence score."""
    text: str
    confidence: float


class ThinkResponse(BaseModel):
    """Response model for think endpoint."""
    text: str
    based_on: Dict[str, List[Dict[str, Any]]]  # {"world": [...], "agent": [...], "opinion": [...]}
    new_opinions: List[OpinionItem] = []  # List of newly formed opinions with confidence

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Based on my understanding, AI is a transformative technology...",
                "based_on": {
                    "world": [{"text": "AI is used in healthcare", "score": 0.9}],
                    "agent": [{"text": "I discussed AI applications last week", "score": 0.85}],
                    "opinion": [{"text": "I believe AI should be used ethically", "score": 0.8}]
                },
                "new_opinions": [
                    {"text": "AI has great potential when used responsibly", "confidence": 0.95}
                ]
            }
        }


class AgentsResponse(BaseModel):
    """Response model for agents list endpoint."""
    agents: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "agents": ["user123", "agent_alice", "agent_bob"]
            }
        }


class GraphDataResponse(BaseModel):
    """Response model for graph data endpoint."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    table_rows: List[Dict[str, Any]]
    total_units: int

    class Config:
        json_schema_extra = {
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
        }


class ListMemoryUnitsResponse(BaseModel):
    """Response model for list memory units endpoint."""
    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int

    class Config:
        json_schema_extra = {
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
        }


class ListDocumentsResponse(BaseModel):
    """Response model for list documents endpoint."""
    items: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "session_1",
                        "agent_id": "user123",
                        "content_hash": "abc123",
                        "metadata": {"source": "conversation"},
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
        }


class DocumentResponse(BaseModel):
    """Response model for get document endpoint."""
    id: str
    agent_id: str
    original_text: str
    content_hash: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    memory_unit_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "id": "session_1",
                "agent_id": "user123",
                "original_text": "Full document text here...",
                "content_hash": "abc123",
                "metadata": {"source": "conversation"},
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "memory_unit_count": 15
            }
        }


def create_app(memory: TemporalSemanticMemory) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        memory: TemporalSemanticMemory instance (already initialized with required parameters)

    Returns:
        Configured FastAPI application
    """
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
        }
    )

    # Mount static files (web directory is sibling to this file)
    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

    @app.on_event("startup")
    async def startup_event():
        """Initialize memory system on startup."""
        await memory.initialize()
        logging.info("Memory system initialized")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup memory system on shutdown."""
        await memory.close()
        logging.info("Memory system closed")

    # Store memory instance on app for route handlers to access
    app.state.memory = memory

    # Register all routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI):
    """Register all API routes on the given app instance."""

    @app.get("/", include_in_schema=False)
    async def index():
        """Serve the visualization page."""
        web_dir = Path(__file__).parent / "web"
        return FileResponse(str(web_dir / "templates" / "index.html"))


    @app.get(
        "/api/graph",
        response_model=GraphDataResponse,
        tags=["Visualization"],
        summary="Get memory graph data",
        description="Retrieve graph data for visualization, optionally filtered by agent_id and fact_type (world/agent/opinion). Limited to 1000 most recent items."
    )
    async def api_graph(
        agent_id: Optional[str] = None,
        fact_type: Optional[str] = None
    ):
        """Get graph data from database, optionally filtered by agent_id and fact_type."""
        try:
            data = await app.state.memory.get_graph_data(agent_id, fact_type)
            return data
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/graph: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/list",
        response_model=ListMemoryUnitsResponse,
        tags=["Visualization"],
        summary="List memory units",
        description="List memory units with pagination and optional full-text search. Supports filtering by agent_id and fact_type."
    )
    async def api_list(
        agent_id: Optional[str] = None,
        fact_type: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List memory units for table view with optional full-text search.

        Args:
            agent_id: Filter by agent ID
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
            print(f"Error in /api/list: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/search",
        response_model=SearchResponse,
        tags=["Search"],
        summary="Search memory",
        description="""
    Search memory using semantic similarity and spreading activation.

    The fact_type parameter is required and must be one of:
    - 'world': General knowledge about people, places, events, and things that happen
    - 'agent': Memories about what the AI agent did, actions taken, and tasks performed
    - 'opinion': The agent's formed beliefs, perspectives, and viewpoints
        """
    )
    async def api_search(request: SearchRequest):
        """Run a search and return results with trace."""
        try:
            # Validate fact_type
            valid_fact_types = ["world", "agent", "opinion"]
            if request.fact_type not in valid_fact_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid fact_type '{request.fact_type}'. Must be one of: {', '.join(valid_fact_types)}"
                )

            # Run search with tracing
            results, trace = await app.state.memory.search_async(
                agent_id=request.agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
                enable_trace=request.trace,
                reranker=request.reranker,
                fact_type=request.fact_type
            )

            # Convert trace to dict
            trace_dict = trace.to_dict() if trace else None

            return SearchResponse(
                results=results,
                trace=trace_dict
            )
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/search: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/think",
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
        """
    )
    async def api_think(request: ThinkRequest):
        try:
            # Use the memory system's think_async method
            result = await app.state.memory.think_async(
                agent_id=request.agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget
            )

            return ThinkResponse(
                text=result["text"],
                based_on=result["based_on"],
                new_opinions=result.get("new_opinions", [])
            )

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/think: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/agents",
        response_model=AgentsResponse,
        tags=["Management"],
        summary="List all agents",
        description="Get a list of all agent IDs that have stored memories in the system"
    )
    async def api_agents():
        """Get list of available agents from database."""
        try:
            agent_list = await app.state.memory.list_agents()
            return AgentsResponse(agents=agent_list)
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/agents: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/stats/{agent_id}",
        tags=["Memory Statistics"],
        summary="Get memory statistics for an agent",
        description="Get statistics about nodes and links for a specific agent"
    )
    async def api_stats(agent_id: str):
        """Get statistics about memory nodes and links for an agent."""
        try:
            pool = await app.state.memory._get_pool()
            async with pool.acquire() as conn:
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

                total_nodes = sum(nodes_by_type.values())
                total_links = sum(links_by_type.values())

                return {
                    "agent_id": agent_id,
                    "total_nodes": total_nodes,
                    "total_links": total_links,
                    "total_documents": total_documents,
                    "nodes_by_type": nodes_by_type,
                    "links_by_type": links_by_type,
                    "pending_operations": pending_operations,
                    "failed_operations": failed_operations
                }

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/stats/{agent_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/api/documents",
        response_model=ListDocumentsResponse,
        tags=["Documents"],
        summary="List documents",
        description="List documents with pagination and optional search. Documents are the source content from which memory units are extracted."
    )
    async def api_list_documents(
        agent_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        List documents for an agent with optional search.

        Args:
            agent_id: Filter by agent ID
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
            print(f"Error in /api/documents: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/documents/{document_id}",
        response_model=DocumentResponse,
        tags=["Documents"],
        summary="Get document details",
        description="Get a specific document including its original text"
    )
    async def api_get_document(
        document_id: str,
        agent_id: str
    ):
        """
        Get a specific document with its original text.

        Args:
            document_id: Document ID
            agent_id: Agent ID (required as query parameter)
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
            print(f"Error in /api/documents/{document_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/memories/batch",
        response_model=BatchPutResponse,
        tags=["Memory Storage"],
        summary="Store multiple memories",
        description="""
    Store multiple memory items in batch with automatic fact extraction.

    Features:
    - Efficient batch processing
    - Automatic fact extraction from natural language
    - Entity recognition and linking
    - Document tracking with optional upsert
    - Temporal and semantic linking

    The system automatically:
    1. Extracts semantic facts from the content
    2. Generates embeddings
    3. Deduplicates similar facts
    4. Creates temporal, semantic, and entity links
    5. Tracks document metadata
        """
    )
    async def api_batch_put(request: BatchPutRequest):
        try:
            # Validate agent_id - prevent writing to reserved agents
            RESERVED_AGENT_IDS = {"locomo"}
            if request.agent_id in RESERVED_AGENT_IDS:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot write to reserved agent_id '{request.agent_id}'. Reserved agents: {', '.join(RESERVED_AGENT_IDS)}"
                )

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
                agent_id=request.agent_id,
                contents=contents,
                document_id=request.document_id,
                document_metadata=request.document_metadata,
                upsert=request.upsert
            )
            logging.info(f"Batch put result: {result}")

            return BatchPutResponse(
                success=True,
                message=f"Successfully stored {len(contents)} memory items",
                agent_id=request.agent_id,
                document_id=request.document_id,
                items_count=len(contents)
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/memories/batch: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/memories/batch_async",
        response_model=BatchPutAsyncResponse,
        tags=["Memory Storage"],
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
    - Document tracking with optional upsert
    - Temporal and semantic linking

    The system automatically:
    1. Queues the batch put task
    2. Returns immediately with success=True, queued=True
    3. Processes in background: extracts facts, generates embeddings, creates links
        """
    )
    async def api_batch_put_async(request: BatchPutRequest):
        try:
            # Validate agent_id - prevent writing to reserved agents
            RESERVED_AGENT_IDS = {"locomo"}
            if request.agent_id in RESERVED_AGENT_IDS:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cannot write to reserved agent_id '{request.agent_id}'. Reserved agents: {', '.join(RESERVED_AGENT_IDS)}"
                )

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
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO async_operations (id, agent_id, task_type, items_count, document_id)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    operation_id,
                    request.agent_id,
                    'batch_put',
                    len(contents),
                    request.document_id
                )

            # Submit task to background queue with operation_id
            await app.state.memory._task_backend.submit_task({
                'type': 'batch_put',
                'operation_id': str(operation_id),
                'agent_id': request.agent_id,
                'contents': contents,
                'document_id': request.document_id,
                'document_metadata': request.document_metadata,
                'upsert': request.upsert
            })

            logging.info(f"Batch put task queued for agent_id={request.agent_id}, {len(contents)} items, operation_id={operation_id}")

            return BatchPutAsyncResponse(
                success=True,
                message=f"Batch put task queued for background processing ({len(contents)} items)",
                agent_id=request.agent_id,
                document_id=request.document_id,
                items_count=len(contents),
                queued=True
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/memories/batch_async: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get(
        "/api/operations/{agent_id}",
        tags=["Memory Storage"],
        summary="List async operations",
        description="Get a list of all async operations (pending and failed) for a specific agent, including error messages for failed operations"
    )
    async def api_list_operations(agent_id: str):
        """List all async operations (pending and failed) for an agent."""
        try:
            pool = await app.state.memory._get_pool()
            async with pool.acquire() as conn:
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
            print(f"Error in /api/operations/{agent_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/operations/{operation_id}",
        tags=["Memory Storage"],
        summary="Cancel a pending async operation",
        description="Cancel a pending async operation by removing it from the queue"
    )
    async def api_cancel_operation(operation_id: str):
        """Cancel a pending async operation."""
        try:
            # Validate UUID format
            try:
                op_uuid = uuid.UUID(operation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid operation_id format: {operation_id}")

            pool = await app.state.memory._get_pool()
            async with pool.acquire() as conn:
                # Check if operation exists
                result = await conn.fetchrow(
                    "SELECT agent_id FROM async_operations WHERE id = $1",
                    op_uuid
                )

                if not result:
                    raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")

                # Delete the operation
                await conn.execute(
                    "DELETE FROM async_operations WHERE id = $1",
                    op_uuid
                )

                return {
                    "success": True,
                    "message": f"Operation {operation_id} cancelled",
                    "operation_id": operation_id,
                    "agent_id": result['agent_id']
                }

        except HTTPException:
            raise
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/operations/{operation_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.delete(
        "/api/memory/{unit_id}",
        tags=["Memory Storage"],
        summary="Delete a memory unit",
        description="Delete a single memory unit and all its associated links (temporal, semantic, and entity links)"
    )
    async def api_delete_memory_unit(unit_id: str):
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
            print(f"Error in /api/memory/{unit_id}: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))
