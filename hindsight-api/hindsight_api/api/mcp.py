"""Hindsight MCP Server implementation using FastMCP (HTTP transport)."""

import json
import logging
import os
from contextvars import ContextVar

from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.mcp_tools import MCPToolsConfig, register_mcp_tools

# Configure logging from HINDSIGHT_API_LOG_LEVEL environment variable
_log_level_str = os.environ.get("HINDSIGHT_API_LOG_LEVEL", "info").lower()
_log_level_map = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}
logging.basicConfig(
    level=_log_level_map.get(_log_level_str, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default bank_id from environment variable
DEFAULT_BANK_ID = os.environ.get("HINDSIGHT_MCP_BANK_ID", "default")

# MCP authentication token (optional - if set, Bearer token auth is required)
MCP_AUTH_TOKEN = os.environ.get("HINDSIGHT_API_MCP_AUTH_TOKEN")

# Context variable to hold the current bank_id
_current_bank_id: ContextVar[str | None] = ContextVar("current_bank_id", default=None)

# Context variable to hold the current API key (for tenant auth propagation)
_current_api_key: ContextVar[str | None] = ContextVar("current_api_key", default=None)


def get_current_bank_id() -> str | None:
    """Get the current bank_id from context."""
    return _current_bank_id.get()


def get_current_api_key() -> str | None:
    """Get the current API key from context."""
    return _current_api_key.get()


def create_mcp_server(memory: MemoryEngine) -> FastMCP:
    """
    Create and configure the Hindsight MCP server.

    Args:
        memory: MemoryEngine instance (required)

    Returns:
        Configured FastMCP server instance with stateless_http enabled
    """
    # Use stateless_http=True for Claude Code compatibility
    mcp = FastMCP("hindsight-mcp-server", stateless_http=True)

    # Configure and register tools using shared module
    config = MCPToolsConfig(
        bank_id_resolver=get_current_bank_id,
        api_key_resolver=get_current_api_key,  # Propagate API key for tenant auth
        include_bank_id_param=True,  # HTTP MCP supports multi-bank via parameter
        tools=None,  # All tools
        retain_fire_and_forget=False,  # HTTP MCP supports sync/async modes
    )

    register_mcp_tools(mcp, memory, config)

    return mcp


class MCPMiddleware:
    """ASGI middleware that handles authentication and extracts bank_id from header or path.

    Authentication:
        If HINDSIGHT_API_MCP_AUTH_TOKEN is set, all requests must include a valid
        Authorization header with Bearer token or direct token matching the configured value.

    Bank ID can be provided via:
    1. X-Bank-Id header (recommended for Claude Code)
    2. URL path: /mcp/{bank_id}/
    3. Environment variable HINDSIGHT_MCP_BANK_ID (fallback default)

    For Claude Code, configure with:
        claude mcp add --transport http hindsight http://localhost:8888/mcp \\
            --header "X-Bank-Id: my-bank" --header "Authorization: Bearer <token>"
    """

    def __init__(self, app, memory: MemoryEngine):
        self.app = app
        self.memory = memory
        self.mcp_server = create_mcp_server(memory)
        self.mcp_app = self.mcp_server.http_app(path="/")
        # Expose the lifespan for the parent app to chain
        self.lifespan = self.mcp_app.lifespan_handler if hasattr(self.mcp_app, "lifespan_handler") else None

    def _get_header(self, scope: dict, name: str) -> str | None:
        """Extract a header value from ASGI scope."""
        name_lower = name.lower().encode()
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == name_lower:
                return header_value.decode()
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.mcp_app(scope, receive, send)
            return

        # Extract auth token from header (for tenant auth propagation)
        auth_header = self._get_header(scope, "Authorization")
        auth_token: str | None = None
        if auth_header:
            # Support both "Bearer <token>" and direct token
            auth_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else auth_header.strip()

        # Authenticate if MCP_AUTH_TOKEN is configured
        if MCP_AUTH_TOKEN:
            if not auth_token:
                await self._send_error(send, 401, "Authorization header required")
                return
            if auth_token != MCP_AUTH_TOKEN:
                await self._send_error(send, 401, "Invalid authentication token")
                return

        path = scope.get("path", "")

        # Strip any mount prefix (e.g., /mcp) that FastAPI might not have stripped
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :] or "/"

        # Also handle case where mount path wasn't stripped (e.g., /mcp/...)
        if path.startswith("/mcp/"):
            path = path[4:]  # Remove /mcp prefix
        elif path == "/mcp":
            path = "/"

        # Try to get bank_id from header first (for Claude Code compatibility)
        bank_id = self._get_header(scope, "X-Bank-Id")

        # MCP endpoint paths that should not be treated as bank_ids
        MCP_ENDPOINTS = {"sse", "messages"}

        # If no header, try to extract from path: /{bank_id}/...
        new_path = path
        if not bank_id and path.startswith("/") and len(path) > 1:
            parts = path[1:].split("/", 1)
            # Don't treat MCP endpoints as bank_ids
            if parts[0] and parts[0] not in MCP_ENDPOINTS:
                # First segment looks like a bank_id
                bank_id = parts[0]
                new_path = "/" + parts[1] if len(parts) > 1 else "/"

        # Fall back to default bank_id
        if not bank_id:
            bank_id = DEFAULT_BANK_ID
            logger.debug(f"Using default bank_id: {bank_id}")

        # Set bank_id and api_key context
        bank_id_token = _current_bank_id.set(bank_id)
        # Store the auth token for tenant extension to validate
        api_key_token = _current_api_key.set(auth_token) if auth_token else None
        try:
            new_scope = scope.copy()
            new_scope["path"] = new_path
            # Clear root_path since we're passing directly to the app
            new_scope["root_path"] = ""

            # Wrap send to rewrite the SSE endpoint URL to include bank_id if using path-based routing
            async def send_wrapper(message):
                if message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    if body and b"/messages" in body:
                        # Rewrite /messages to /{bank_id}/messages in SSE endpoint event
                        body = body.replace(b"data: /messages", f"data: /{bank_id}/messages".encode())
                        message = {**message, "body": body}
                await send(message)

            await self.mcp_app(new_scope, receive, send_wrapper)
        finally:
            _current_bank_id.reset(bank_id_token)
            if api_key_token is not None:
                _current_api_key.reset(api_key_token)

    async def _send_error(self, send, status: int, message: str):
        """Send an error response."""
        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def create_mcp_app(memory: MemoryEngine):
    """
    Create an ASGI app that handles MCP requests.

    Authentication:
        Set HINDSIGHT_API_MCP_AUTH_TOKEN to require Bearer token authentication.
        If not set, MCP endpoint is open (for local development).

    Bank ID can be provided via:
    1. X-Bank-Id header: claude mcp add --transport http hindsight http://localhost:8888/mcp --header "X-Bank-Id: my-bank"
    2. URL path: /mcp/{bank_id}/
    3. Environment variable HINDSIGHT_MCP_BANK_ID (fallback, default: "default")

    Args:
        memory: MemoryEngine instance

    Returns:
        ASGI application
    """
    return MCPMiddleware(None, memory)
