"""
Unified API module for Hindsight.

Provides both HTTP REST API and MCP (Model Context Protocol) server.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from hindsight_api import MemoryEngine

logger = logging.getLogger(__name__)


def create_app(
    memory: MemoryEngine,
    http_api_enabled: bool = True,
    mcp_api_enabled: bool = False,
    mcp_mount_path: str = "/mcp",
    initialize_memory: bool = True,
) -> FastAPI:
    """
    Create and configure the unified Hindsight API application.

    Args:
        memory: MemoryEngine instance (already initialized with required parameters).
                Migrations are controlled by the MemoryEngine's run_migrations parameter.
        http_api_enabled: Whether to enable HTTP REST API endpoints (default: True)
        mcp_api_enabled: Whether to enable MCP server (default: False)
        mcp_mount_path: Path to mount MCP server (default: /mcp)
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
    mcp_app = None

    # Create MCP app first if enabled (we need its lifespan for chaining)
    if mcp_api_enabled:
        try:
            from .mcp import create_mcp_app

            mcp_app = create_mcp_app(memory=memory)
        except ImportError as e:
            logger.error(f"MCP server requested but dependencies not available: {e}")
            logger.error("Install with: pip install hindsight-api[mcp]")
            raise

    # Import and create HTTP API if enabled
    if http_api_enabled:
        from .http import create_app as create_http_app

        app = create_http_app(memory=memory, initialize_memory=initialize_memory)
        logger.info("HTTP REST API enabled")
    else:
        # Create minimal FastAPI app
        app = FastAPI(title="Hindsight API", version="0.0.7")
        logger.info("HTTP REST API disabled")

    # Mount MCP server and chain its lifespan if enabled
    if mcp_app is not None:
        # Get both MCP apps' underlying Starlette apps for lifespan access
        multi_bank_starlette_app = mcp_app.multi_bank_app
        single_bank_starlette_app = mcp_app.single_bank_app

        # Store the original lifespan
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def chained_lifespan(app_instance: FastAPI):
            """Chain both MCP lifespans with the main app lifespan."""
            # Start both MCP lifespans (multi-bank and single-bank)
            async with multi_bank_starlette_app.router.lifespan_context(multi_bank_starlette_app):
                async with single_bank_starlette_app.router.lifespan_context(single_bank_starlette_app):
                    logger.info("MCP lifespans started (multi-bank and single-bank)")
                    # Then start the original app lifespan
                    async with original_lifespan(app_instance):
                        yield
                logger.info("MCP lifespans stopped")

        # Replace the app's lifespan with the chained version
        app.router.lifespan_context = chained_lifespan

        # Mount the MCP middleware
        app.mount(mcp_mount_path, mcp_app)
        logger.info(f"MCP server enabled at {mcp_mount_path}/")

    return app


# Re-export commonly used items for backwards compatibility
from .http import (
    CreateBankRequest,
    DispositionTraits,
    MemoryItem,
    RecallRequest,
    RecallResponse,
    RecallResult,
    ReflectRequest,
    ReflectResponse,
    RetainRequest,
)

__all__ = [
    "create_app",
    "RecallRequest",
    "RecallResult",
    "RecallResponse",
    "MemoryItem",
    "RetainRequest",
    "ReflectRequest",
    "ReflectResponse",
    "CreateBankRequest",
    "DispositionTraits",
]
