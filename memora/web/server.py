"""
FastAPI server for memory graph visualization and API.

Provides REST API endpoints for memory operations and serves
the interactive visualization interface.
"""
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import from parent memora package
from memora import TemporalSemanticMemory
from memora.embeddings import Embeddings

import logging



# Environment variables are loaded by the shell script that calls this module
# No need to load .env files here as they're sourced by start-server.sh


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

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

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


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    agent_id: str = "default"
    thinking_budget: int = 100
    max_tokens: int = 4096
    reranker: str = "heuristic"
    trace: bool = False
    fact_type: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What did Alice say about machine learning?",
                "agent_id": "user123",
                "thinking_budget": 100,
                "max_tokens": 4096,
                "reranker": "heuristic",
                "trace": True,
                "fact_type": "world"
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


def _register_routes(app: FastAPI):
    """Register all API routes on the given app instance."""

    @app.get("/", include_in_schema=False)
    async def index():
        """Serve the visualization page."""
        return FileResponse(str(Path(__file__).parent / "templates" / "index.html"))


    @app.get(
        "/api/graph",
        response_model=GraphDataResponse,
        tags=["Visualization"],
        summary="Get memory graph data",
        description="Retrieve graph data for visualization, optionally filtered by agent_id and fact_type (world/agent/opinion)"
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


    @app.post(
        "/api/search",
        response_model=SearchResponse,
        tags=["Search"],
        summary="Search memory",
        description="Search memory using semantic similarity and spreading activation. Optionally filter by fact_type (world, agent, opinion)"
    )
    async def api_search(request: SearchRequest):
        """Run a search and return results with trace."""
        try:
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
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/search: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/world_search",
        response_model=SearchResponse,
        tags=["Search"],
        summary="Search world facts",
        description="Search only world facts - general knowledge about people, places, events, and things that happen"
    )
    async def api_world_search(request: SearchRequest):
        """Search only world facts (general knowledge about the world)."""
        try:
            # Run search with fact_type filter for 'world'
            results, trace = await app.state.memory.search_async(
                agent_id=request.agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
                enable_trace=request.trace,
                reranker=request.reranker,
                fact_type='world'
            )

            # Convert trace to dict
            trace_dict = trace.to_dict() if trace else None

            return SearchResponse(
                results=results,
                trace=trace_dict
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/world_search: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/agent_search",
        response_model=SearchResponse,
        tags=["Search"],
        summary="Search agent action facts",
        description="Search only agent facts - memories about what the AI agent did, actions taken, and tasks performed"
    )
    async def api_agent_search(request: SearchRequest):
        """Search only agent facts (facts about what the agent did)."""
        try:
            # Run search with fact_type filter for 'agent'
            results, trace = await app.state.memory.search_async(
                agent_id=request.agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
                enable_trace=request.trace,
                reranker=request.reranker,
                fact_type='agent'
            )

            # Convert trace to dict
            trace_dict = trace.to_dict() if trace else None

            return SearchResponse(
                results=results,
                trace=trace_dict
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/agent_search: {error_detail}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.post(
        "/api/opinion_search",
        response_model=SearchResponse,
        tags=["Search"],
        summary="Search agent opinions",
        description="Search only opinion facts - the agent's formed beliefs, perspectives, and viewpoints"
    )
    async def api_opinion_search(request: SearchRequest):
        """Search only opinion facts (agent's formed opinions and perspectives)."""
        try:
            # Run search with fact_type filter for 'opinion'
            results, trace = await app.state.memory.search_async(
                agent_id=request.agent_id,
                query=request.query,
                thinking_budget=request.thinking_budget,
                max_tokens=request.max_tokens,
                enable_trace=request.trace,
                reranker=request.reranker,
                fact_type='opinion'
            )

            # Convert trace to dict
            trace_dict = trace.to_dict() if trace else None

            return SearchResponse(
                results=results,
                trace=trace_dict
            )
        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/opinion_search: {error_detail}")
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

                # Format results
                nodes_by_type = {row['fact_type']: row['count'] for row in node_stats}
                links_by_type = {row['link_type']: row['count'] for row in link_stats}

                total_nodes = sum(nodes_by_type.values())
                total_links = sum(links_by_type.values())

                return {
                    "agent_id": agent_id,
                    "total_nodes": total_nodes,
                    "total_links": total_links,
                    "nodes_by_type": nodes_by_type,
                    "links_by_type": links_by_type
                }

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"Error in /api/stats/{agent_id}: {error_detail}")
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




# Create app at module level (required for uvicorn import string)
_memory = TemporalSemanticMemory(
    db_url=os.getenv("DATABASE_URL"),
    memory_llm_provider=os.getenv("MEMORY_LLM_PROVIDER", "groq"),
    memory_llm_api_key=os.getenv("MEMORY_LLM_API_KEY"),
    memory_llm_model=os.getenv("MEMORY_LLM_MODEL", "openai/gpt-oss-120b"),
    memory_llm_base_url=os.getenv("MEMORY_LLM_BASE_URL") or None,
)
app = create_app(_memory)


if __name__ == "__main__":
    import uvicorn
    import argparse
    logging.basicConfig(level=logging.INFO)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Memory Graph API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"],
                        help="Log level (default: info)")
    parser.add_argument("--access-log", action="store_true", help="Enable access log")
    parser.add_argument("--no-access-log", dest="access_log", action="store_false", help="Disable access log")
    parser.add_argument("--proxy-headers", action="store_true", help="Enable X-Forwarded-Proto, X-Forwarded-For headers")
    parser.add_argument("--forwarded-allow-ips", default=None, help="Comma separated list of IPs to trust with proxy headers")
    parser.add_argument("--ssl-keyfile", default=None, help="SSL key file")
    parser.add_argument("--ssl-certfile", default=None, help="SSL certificate file")
    parser.set_defaults(access_log=False)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Memory Graph API Server")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 80 + "\n")

    # Always use import string for uvicorn (required for reload and workers)
    app_ref = "memora.web.server:app"

    # Prepare uvicorn config
    uvicorn_config = {
        "app": app_ref,
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "workers": args.workers,
        "log_level": args.log_level,
        "access_log": args.access_log,
        "proxy_headers": args.proxy_headers,
    }

    # Add optional parameters if provided
    if args.forwarded_allow_ips:
        uvicorn_config["forwarded_allow_ips"] = args.forwarded_allow_ips
    if args.ssl_keyfile:
        uvicorn_config["ssl_keyfile"] = args.ssl_keyfile
    if args.ssl_certfile:
        uvicorn_config["ssl_certfile"] = args.ssl_certfile

    uvicorn.run(**uvicorn_config)
