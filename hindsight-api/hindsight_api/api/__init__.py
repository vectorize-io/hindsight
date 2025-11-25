"""
Unified API module for Hindsight.

Provides both HTTP REST API and MCP (Model Context Protocol) server.
"""
import logging
from typing import Optional
from fastapi import FastAPI

from hindsight_api import MemoryEngine

logger = logging.getLogger(__name__)


def create_app(
    memory: MemoryEngine,
    http_api_enabled: bool = True,
    mcp_api_enabled: bool = False,
    mcp_mount_path: str = "/mcp",
    run_migrations: bool = True,
    initialize_memory: bool = True
) -> FastAPI:
    """
    Create and configure the unified Hindsight API application.

    Args:
        memory: MemoryEngine instance (already initialized with required parameters)
        http_api_enabled: Whether to enable HTTP REST API endpoints (default: True)
        mcp_api_enabled: Whether to enable MCP server (default: False)
        mcp_mount_path: Path to mount MCP server (default: /mcp)
        run_migrations: Whether to run database migrations on startup (default: True)
        initialize_memory: Whether to initialize memory system on startup (default: True)

    Returns:
        Configured FastAPI application with enabled APIs

    Example:
        # HTTP only
        app = create_app(memory)

        # MCP only
        app = create_app(memory, http_api_enabled=False, mcp_api_enabled=True)

        # Both HTTP and MCP
        app = create_app(memory, mcp_api_enabled=True)
    """

    # Import and create HTTP API if enabled
    if http_api_enabled:
        from .http import create_app as create_http_app
        app = create_http_app(
            memory=memory,
            run_migrations=run_migrations,
            initialize_memory=initialize_memory
        )
        logger.info("HTTP REST API enabled")
    else:
        # Create minimal FastAPI app
        app = FastAPI(title="Hindsight API", version="0.0.7")
        logger.info("HTTP REST API disabled")

    # Mount MCP server if enabled
    if mcp_api_enabled:
        try:
            from .mcp import create_mcp_server

            # Create MCP server with shared memory instance
            mcp_server = create_mcp_server(memory=memory)

            # Mount at specified path using http_app (modern non-SSE alternative)
            app.mount(mcp_mount_path, mcp_server.http_app())
            logger.info(f"MCP server enabled at {mcp_mount_path}")
        except ImportError as e:
            logger.error(f"MCP server requested but dependencies not available: {e}")
            logger.error("Install with: pip install hindsight-api[mcp]")
            raise

    return app


# Re-export commonly used items for backwards compatibility
from .http import (
    SearchRequest,
    SearchResult,
    SearchResponse,
    MemoryItem,
    BatchPutRequest,
    ThinkRequest,
    ThinkResponse,
    CreateAgentRequest,
    PersonalityTraits,
)

__all__ = [
    "create_app",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "MemoryItem",
    "BatchPutRequest",
    "ThinkRequest",
    "ThinkResponse",
    "CreateAgentRequest",
    "PersonalityTraits",
]
