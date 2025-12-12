"""Hindsight MCP Server implementation using FastMCP."""

import json
import logging
import os
from contextvars import ContextVar
from typing import Optional

from fastmcp import FastMCP
from hindsight_api import MemoryEngine
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES

# Configure logging from HINDSIGHT_API_LOG_LEVEL environment variable
_log_level_str = os.environ.get("HINDSIGHT_API_LOG_LEVEL", "info").lower()
_log_level_map = {"critical": logging.CRITICAL, "error": logging.ERROR, "warning": logging.WARNING,
                  "info": logging.INFO, "debug": logging.DEBUG, "trace": logging.DEBUG}
logging.basicConfig(
    level=_log_level_map.get(_log_level_str, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Context variable to hold the current bank_id from the URL path
_current_bank_id: ContextVar[Optional[str]] = ContextVar("current_bank_id", default=None)


def get_current_bank_id() -> Optional[str]:
    """Get the current bank_id from context (set from URL path)."""
    return _current_bank_id.get()


def create_mcp_server(memory: MemoryEngine) -> FastMCP:
    """
    Create and configure the Hindsight MCP server.

    Args:
        memory: MemoryEngine instance (required)

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP("hindsight-mcp-server")

    @mcp.tool()
    async def retain(content: str, context: str = "general") -> str:
        """
        Store important information to long-term memory.

        Use this tool PROACTIVELY whenever the user shares:
        - Personal facts, preferences, or interests
        - Important events or milestones
        - User history, experiences, or background
        - Decisions, opinions, or stated preferences
        - Goals, plans, or future intentions
        - Relationships or people mentioned
        - Work context, projects, or responsibilities

        Args:
            content: The fact/memory to store (be specific and include relevant details)
            context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
        """
        try:
            bank_id = get_current_bank_id()
            await memory.put_batch_async(
                bank_id=bank_id,
                contents=[{"content": content, "context": context}]
            )
            return "Memory stored successfully"
        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @mcp.tool()
    async def recall(query: str, max_results: int = 10) -> str:
        """
        Search memories to provide personalized, context-aware responses.

        Use this tool PROACTIVELY to:
        - Check user's preferences before making suggestions
        - Recall user's history to provide continuity
        - Remember user's goals and context
        - Personalize responses based on past interactions

        Args:
            query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
            max_results: Maximum number of results to return (default: 10)
        """
        try:
            bank_id = get_current_bank_id()
            from hindsight_api.engine.memory_engine import Budget
            search_result = await memory.recall_async(
                bank_id=bank_id,
                query=query,
                fact_type=list(VALID_RECALL_FACT_TYPES),
                budget=Budget.LOW
            )

            results = [
                {
                    "id": fact.id,
                    "text": fact.text,
                    "type": fact.fact_type,
                    "context": fact.context,
                    "event_date": fact.event_date,
                }
                for fact in search_result.results[:max_results]
            ]

            return json.dumps({"results": results}, indent=2)
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            return json.dumps({"error": str(e), "results": []})

    return mcp


class MCPMiddleware:
    """ASGI middleware that extracts bank_id from path and sets context."""

    def __init__(self, app, memory: MemoryEngine):
        self.app = app
        self.memory = memory
        self.mcp_server = create_mcp_server(memory)
        self.mcp_app = self.mcp_server.http_app()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.mcp_app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Strip any mount prefix (e.g., /mcp) that FastAPI might not have stripped
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path):] or "/"

        # Also handle case where mount path wasn't stripped (e.g., /mcp/...)
        if path.startswith("/mcp/"):
            path = path[4:]  # Remove /mcp prefix

        # Extract bank_id from path: /{bank_id}/ or /{bank_id}
        # http_app expects requests at /
        if not path.startswith("/") or len(path) <= 1:
            # No bank_id in path - return error
            await self._send_error(send, 400, "bank_id required in path: /mcp/{bank_id}/")
            return

        # Extract bank_id from first path segment
        parts = path[1:].split("/", 1)
        if not parts[0]:
            await self._send_error(send, 400, "bank_id required in path: /mcp/{bank_id}/")
            return

        bank_id = parts[0]
        new_path = "/" + parts[1] if len(parts) > 1 else "/"

        # Set bank_id context
        token = _current_bank_id.set(bank_id)
        try:
            new_scope = scope.copy()
            new_scope["path"] = new_path

            # Wrap send to rewrite the SSE endpoint URL to include bank_id
            # The SSE app sends "event: endpoint\ndata: /messages\n" but we need
            # the client to POST to /{bank_id}/messages instead
            async def send_wrapper(message):
                if message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if body and b"/messages" in body:
                        # Rewrite /messages to /{bank_id}/messages in SSE endpoint event
                        body = body.replace(
                            b"data: /messages",
                            f"data: /{bank_id}/messages".encode()
                        )
                        message = {**message, "body": body}
                await send(message)

            await self.mcp_app(new_scope, receive, send_wrapper)
        finally:
            _current_bank_id.reset(token)

    async def _send_error(self, send, status: int, message: str):
        """Send an error response."""
        body = json.dumps({"error": message}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })


def create_mcp_app(memory: MemoryEngine):
    """
    Create an ASGI app that handles MCP requests.

    URL pattern: /mcp/{bank_id}/

    The bank_id is extracted from the URL path and made available to tools.

    Args:
        memory: MemoryEngine instance

    Returns:
        ASGI application
    """
    return MCPMiddleware(None, memory)
