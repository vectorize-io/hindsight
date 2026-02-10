"""Hindsight MCP Server implementation using FastMCP (HTTP transport)."""

import json
import logging
import os
from contextvars import ContextVar

from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.engine.memory_engine import _current_schema
from hindsight_api.extensions import MCPExtension, load_extension
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.mcp_tools import MCPToolsConfig, register_mcp_tools
from hindsight_api.models import RequestContext

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

# Legacy MCP authentication token (for backwards compatibility)
# If set, this token is checked first before TenantExtension auth
MCP_AUTH_TOKEN = os.environ.get("HINDSIGHT_API_MCP_AUTH_TOKEN")

# Context variable to hold the current bank_id
_current_bank_id: ContextVar[str | None] = ContextVar("current_bank_id", default=None)

# Context variable to hold the current API key (for tenant auth propagation)
_current_api_key: ContextVar[str | None] = ContextVar("current_api_key", default=None)

# Context variables for tenant_id and api_key_id (set by authenticate, used by usage metering)
_current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)
_current_api_key_id: ContextVar[str | None] = ContextVar("current_api_key_id", default=None)


def get_current_bank_id() -> str | None:
    """Get the current bank_id from context."""
    return _current_bank_id.get()


def get_current_api_key() -> str | None:
    """Get the current API key from context."""
    return _current_api_key.get()


def get_current_tenant_id() -> str | None:
    """Get the current tenant_id from context."""
    return _current_tenant_id.get()


def get_current_api_key_id() -> str | None:
    """Get the current api_key_id from context."""
    return _current_api_key_id.get()


def create_mcp_server(memory: MemoryEngine, multi_bank: bool = True) -> FastMCP:
    """
    Create and configure the Hindsight MCP server.

    Args:
        memory: MemoryEngine instance (required)
        multi_bank: If True, expose all tools with bank_id parameters (default).
                   If False, only expose bank-scoped tools without bank_id parameters.

    Returns:
        Configured FastMCP server instance with stateless_http enabled
    """
    # Use stateless_http=True for Claude Code compatibility
    mcp = FastMCP("hindsight-mcp-server", stateless_http=True)

    # Configure and register tools using shared module
    config = MCPToolsConfig(
        bank_id_resolver=get_current_bank_id,
        api_key_resolver=get_current_api_key,  # Propagate API key for tenant auth
        tenant_id_resolver=get_current_tenant_id,  # Propagate tenant_id for usage metering
        api_key_id_resolver=get_current_api_key_id,  # Propagate api_key_id for usage metering
        include_bank_id_param=multi_bank,
        tools=None if multi_bank else {"retain", "recall", "reflect"},  # Scoped tools for single-bank mode
        retain_fire_and_forget=False,  # HTTP MCP supports sync/async modes
    )

    register_mcp_tools(mcp, memory, config)

    # Load and register additional tools from MCP extension if configured
    mcp_extension = load_extension("MCP", MCPExtension)
    if mcp_extension:
        logger.info(f"Loading MCP extension: {mcp_extension.__class__.__name__}")
        mcp_extension.register_tools(mcp, memory)

    return mcp


class MCPMiddleware:
    """ASGI middleware that handles authentication and routes to appropriate MCP server.

    Authentication:
        1. If HINDSIGHT_API_MCP_AUTH_TOKEN is set (legacy), validates against that token
        2. Otherwise, uses TenantExtension.authenticate_mcp() from the MemoryEngine
           - DefaultTenantExtension: no auth required (local dev)
           - ApiKeyTenantExtension: validates against env var

    Two modes based on URL structure:

    1. Multi-bank mode (for /mcp/ root endpoint):
       - Exposes all tools: retain, recall, reflect, list_banks, create_bank
       - All tools include optional bank_id parameter for cross-bank operations
       - Bank ID from: X-Bank-Id header or HINDSIGHT_MCP_BANK_ID env var

    2. Single-bank mode (for /mcp/{bank_id}/ endpoints):
       - Exposes bank-scoped tools only: retain, recall, reflect
       - No bank_id parameter (comes from URL)
       - No bank management tools (list_banks, create_bank)
       - Recommended for agent isolation

    Examples:
        # Single-bank mode (recommended for agent isolation)
        claude mcp add --transport http my-agent http://localhost:8888/mcp/my-agent-bank/ \\
            --header "Authorization: Bearer <token>"

        # Multi-bank mode (for cross-bank operations)
        claude mcp add --transport http hindsight http://localhost:8888/mcp \\
            --header "X-Bank-Id: my-bank" --header "Authorization: Bearer <token>"
    """

    def __init__(self, app, memory: MemoryEngine):
        self.app = app
        self.memory = memory
        self.tenant_extension = memory._tenant_extension

        # Create two server instances:
        # 1. Multi-bank server (for /mcp/ root endpoint)
        self.multi_bank_server = create_mcp_server(memory, multi_bank=True)
        self.multi_bank_app = self.multi_bank_server.http_app(path="/")

        # 2. Single-bank server (for /mcp/{bank_id}/ endpoints)
        self.single_bank_server = create_mcp_server(memory, multi_bank=False)
        self.single_bank_app = self.single_bank_server.http_app(path="/")

        # Backward compatibility: expose multi_bank_app as mcp_app
        self.mcp_app = self.multi_bank_app

        # Expose the lifespan for the parent app to chain (use multi-bank as default)
        self.lifespan = (
            self.multi_bank_app.lifespan_handler if hasattr(self.multi_bank_app, "lifespan_handler") else None
        )

    def _get_header(self, scope: dict, name: str) -> str | None:
        """Extract a header value from ASGI scope."""
        name_lower = name.lower().encode()
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == name_lower:
                return header_value.decode()
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.multi_bank_app(scope, receive, send)
            return

        # Extract auth token from header (for tenant auth propagation)
        auth_header = self._get_header(scope, "Authorization")
        auth_token: str | None = None
        if auth_header:
            # Support both "Bearer <token>" and direct token
            auth_token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else auth_header.strip()

        # Authenticate: check legacy MCP_AUTH_TOKEN first, then TenantExtension
        tenant_context = None
        auth_tenant_id: str | None = None
        auth_api_key_id: str | None = None
        if MCP_AUTH_TOKEN:
            # Legacy authentication mode - validate against static token
            if not auth_token:
                await self._send_error(send, 401, "Authorization header required")
                return
            if auth_token != MCP_AUTH_TOKEN:
                await self._send_error(send, 401, "Invalid authentication token")
                return
            # Legacy mode doesn't use tenant schemas
            tenant_context = None
        else:
            # Use TenantExtension.authenticate_mcp() for auth
            try:
                auth_context = RequestContext(api_key=auth_token)
                tenant_context = await self.tenant_extension.authenticate_mcp(auth_context)
                # Capture tenant_id and api_key_id set by authenticate() for usage metering
                auth_tenant_id = auth_context.tenant_id
                auth_api_key_id = auth_context.api_key_id
            except AuthenticationError as e:
                await self._send_error(send, 401, str(e))
                return

        # Set schema from tenant context so downstream DB queries use the correct schema
        schema_token = (
            _current_schema.set(tenant_context.schema_name) if tenant_context and tenant_context.schema_name else None
        )

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

        # Ensure path has leading slash (needed after stripping mount path)
        if path and not path.startswith("/"):
            path = "/" + path

        # Try to get bank_id from header first (for Claude Code compatibility)
        bank_id = self._get_header(scope, "X-Bank-Id")
        bank_id_from_path = False

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
                bank_id_from_path = True
                new_path = "/" + parts[1] if len(parts) > 1 else "/"

        # Fall back to default bank_id
        if not bank_id:
            bank_id = DEFAULT_BANK_ID
            logger.debug(f"Using default bank_id: {bank_id}")

        # Select the appropriate MCP app based on how bank_id was provided:
        # - Path-based bank_id → single-bank app (no bank_id param, scoped tools)
        # - Header/env bank_id → multi-bank app (bank_id param, all tools)
        target_app = self.single_bank_app if bank_id_from_path else self.multi_bank_app

        # Set bank_id, api_key, tenant_id, and api_key_id context
        bank_id_token = _current_bank_id.set(bank_id)
        # Store the auth token for tenant extension to validate
        api_key_token = _current_api_key.set(auth_token) if auth_token else None
        # Store tenant_id and api_key_id from authentication for usage metering
        tenant_id_token = _current_tenant_id.set(auth_tenant_id) if auth_tenant_id else None
        api_key_id_token = _current_api_key_id.set(auth_api_key_id) if auth_api_key_id else None
        try:
            new_scope = scope.copy()
            new_scope["path"] = new_path
            # Clear root_path since we're passing directly to the app
            new_scope["root_path"] = ""

            # Wrap send to rewrite the SSE endpoint URL to include bank_id if using path-based routing
            async def send_wrapper(message):
                if message["type"] == "http.response.body" and bank_id_from_path:
                    body = message.get("body", b"")
                    if body and b"/messages" in body:
                        # Rewrite /messages to /{bank_id}/messages in SSE endpoint event
                        body = body.replace(b"data: /messages", f"data: /{bank_id}/messages".encode())
                        message = {**message, "body": body}
                await send(message)

            await target_app(new_scope, receive, send_wrapper)
        finally:
            _current_bank_id.reset(bank_id_token)
            if api_key_token is not None:
                _current_api_key.reset(api_key_token)
            if tenant_id_token is not None:
                _current_tenant_id.reset(tenant_id_token)
            if api_key_id_token is not None:
                _current_api_key_id.reset(api_key_id_token)
            if schema_token is not None:
                _current_schema.reset(schema_token)

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
    Create an ASGI app that handles MCP requests with dynamic tool exposure.

    Authentication:
        Uses the TenantExtension from the MemoryEngine (same auth as REST API).

    Two modes based on URL structure:

    1. Single-bank mode (recommended for agent isolation):
       - URL: /mcp/{bank_id}/
       - Tools: retain, recall, reflect (no bank_id parameter)
       - Example: claude mcp add --transport http my-agent http://localhost:8888/mcp/my-agent-bank/

    2. Multi-bank mode (for cross-bank operations):
       - URL: /mcp/
       - Tools: retain, recall, reflect, list_banks, create_bank (all with bank_id parameter)
       - Bank ID from: X-Bank-Id header or HINDSIGHT_MCP_BANK_ID env var (default: "default")
       - Example: claude mcp add --transport http hindsight http://localhost:8888/mcp --header "X-Bank-Id: my-bank"

    Args:
        memory: MemoryEngine instance

    Returns:
        ASGI application
    """
    return MCPMiddleware(None, memory)
