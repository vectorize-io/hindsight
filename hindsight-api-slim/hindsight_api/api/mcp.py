"""Hindsight MCP Server implementation using FastMCP (HTTP transport)."""

import json
import logging
import os
from contextvars import ContextVar

from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memory_engine import _current_schema
from hindsight_api.extensions import MCPExtension, load_extension
from hindsight_api.extensions.tenant import AuthenticationError
from hindsight_api.mcp_tools import _ALL_TOOLS, MCPToolsConfig, register_mcp_tools
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

# Context variable for MCP pre-authentication flag (set when MCP_AUTH_TOKEN validates)
_current_mcp_authenticated: ContextVar[bool] = ContextVar("current_mcp_authenticated", default=False)


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


def get_current_mcp_authenticated() -> bool:
    """Get whether the request was pre-authenticated by MCP transport auth."""
    return _current_mcp_authenticated.get()


def create_mcp_server(memory: MemoryEngine, multi_bank: bool = True) -> FastMCP:
    """
    Create and configure the Hindsight MCP server.

    Args:
        memory: MemoryEngine instance (required)
        multi_bank: If True, expose all tools with bank_id parameters (default).
                   If False, only expose bank-scoped tools without bank_id parameters.

    Returns:
        Configured FastMCP server instance
    """
    mcp = FastMCP("hindsight-mcp-server")

    global_config = _get_raw_config()

    # Tools available for this mode (multi-bank exposes all tools; single-bank excludes bank-management tools)
    _SINGLE_BANK_TOOLS: frozenset[str] = frozenset(
        {
            "retain",
            "sync_retain",
            "recall",
            "reflect",
            "list_mental_models",
            "get_mental_model",
            "create_mental_model",
            "update_mental_model",
            "delete_mental_model",
            "refresh_mental_model",
            "list_directives",
            "create_directive",
            "delete_directive",
            "list_memories",
            "get_memory",
            "delete_memory",
            "list_documents",
            "get_document",
            "delete_document",
            "list_operations",
            "get_operation",
            "cancel_operation",
            "list_tags",
            "get_bank",
            "update_bank",
            "delete_bank",
            "clear_memories",
        }
    )
    base_tools: frozenset[str] | None = None if multi_bank else _SINGLE_BANK_TOOLS

    # Apply global mcp_enabled_tools filter (env-level allowlist)
    if global_config.mcp_enabled_tools is not None:
        allowed = frozenset(global_config.mcp_enabled_tools)
        base_tools = (base_tools if base_tools is not None else _ALL_TOOLS) & allowed

    # Configure and register tools using shared module
    config = MCPToolsConfig(
        bank_id_resolver=get_current_bank_id,
        api_key_resolver=get_current_api_key,  # Propagate API key for tenant auth
        tenant_id_resolver=get_current_tenant_id,  # Propagate tenant_id for usage metering
        api_key_id_resolver=get_current_api_key_id,  # Propagate api_key_id for usage metering
        mcp_authenticated_resolver=get_current_mcp_authenticated,  # Propagate MCP pre-auth flag
        include_bank_id_param=multi_bank,
        tools=base_tools,
    )

    register_mcp_tools(mcp, memory, config)

    # Load and register additional tools from MCP extension if configured
    mcp_extension = load_extension("MCP", MCPExtension)
    if mcp_extension:
        logger.info(f"Loading MCP extension: {mcp_extension.__class__.__name__}")
        mcp_extension.register_tools(mcp, memory)

    # Make all tools tolerant of extra arguments from LLMs (e.g., "explanation")
    _make_tools_tolerant(mcp)

    return mcp


def _get_mcp_tools(mcp: FastMCP) -> dict:
    """Get tool name→object mapping, compatible with FastMCP 2.x and 3.x."""
    # FastMCP 2.x: _tool_manager._tools
    if hasattr(mcp, "_tool_manager"):
        return mcp._tool_manager._tools  # type: ignore[union-attr]
    # FastMCP 3.x: _local_provider._components with "tool:" prefix
    if hasattr(mcp, "_local_provider"):
        return {
            k.split(":")[1].split("@")[0]: v
            for k, v in mcp._local_provider._components.items()  # type: ignore[union-attr]
            if k.startswith("tool:")
        }
    msg = "Cannot locate tools on FastMCP instance"
    raise AttributeError(msg)


def _make_tools_tolerant(mcp: FastMCP) -> None:
    """Wrap all tool run methods to strip unknown arguments and coerce string-encoded JSON.

    LLMs frequently add extra fields like "explanation" or "reasoning" to tool calls.
    FastMCP's Pydantic TypeAdapter rejects these with "Unexpected keyword argument".

    LLMs also frequently serialize list/dict arguments as JSON strings instead of native
    types (e.g., tags='["a","b"]' instead of tags=["a","b"]). This auto-coerces them.

    This wraps each tool's run() to apply both fixes before validation.
    """
    try:
        tools = _get_mcp_tools(mcp)
        for name, tool in tools.items():
            if hasattr(tool, "parameters") and tool.parameters:
                properties = tool.parameters.get("properties", {})
                allowed = set(properties.keys())

                # Build sets of parameter names that expect array or object types.
                # Handles both direct types {"type": "array"} and anyOf/oneOf unions
                # like {"anyOf": [{"type": "array", ...}, {"type": "null"}]}.
                array_params: set[str] = set()
                object_params: set[str] = set()
                for param_name, param_schema in properties.items():
                    _collect_coercible_types(param_schema, param_name, array_params, object_params)

                original_run = tool.run

                async def _tolerant_run(
                    arguments,
                    _allowed=allowed,
                    _orig=original_run,
                    _array_params=array_params,
                    _object_params=object_params,
                ):
                    extra_keys = set(arguments.keys()) - _allowed
                    if extra_keys:
                        logger.debug(f"Stripping unknown arguments from tool call: {extra_keys}")
                        arguments = {k: v for k, v in arguments.items() if k in _allowed}

                    # Coerce string-encoded JSON for list/dict parameters
                    arguments = _coerce_string_json(arguments, _array_params, _object_params)

                    return await _orig(arguments)

                # FunctionTool is a Pydantic model with extra='forbid', so use
                # object.__setattr__ to bypass Pydantic's setter validation.
                object.__setattr__(tool, "run", _tolerant_run)
    except (AttributeError, KeyError) as e:
        logger.warning(f"Could not make tools tolerant of extra arguments: {e}")


def _collect_coercible_types(schema: dict, param_name: str, array_params: set[str], object_params: set[str]) -> None:
    """Check a JSON Schema property and add param_name to array_params/object_params if applicable."""
    # Direct type
    schema_type = schema.get("type")
    if schema_type == "array":
        array_params.add(param_name)
        return
    if schema_type == "object":
        object_params.add(param_name)
        return

    # anyOf / oneOf unions (e.g., list[str] | None → {"anyOf": [{"type": "array"}, {"type": "null"}]})
    for variant in schema.get("anyOf", []) + schema.get("oneOf", []):
        variant_type = variant.get("type")
        if variant_type == "array":
            array_params.add(param_name)
            return
        if variant_type == "object":
            object_params.add(param_name)
            return


def _coerce_string_json(arguments: dict, array_params: set[str], object_params: set[str]) -> dict:
    """Auto-coerce string-encoded JSON arrays/objects to native types.

    LLM agents frequently serialize list and dict tool arguments as JSON strings.
    This is backward-compatible: native arrays/objects pass through unchanged.
    """
    for param_name in array_params:
        val = arguments.get(param_name)
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    arguments = {**arguments, param_name: parsed}
                    logger.debug(f"Coerced string to list for parameter '{param_name}'")
            except (json.JSONDecodeError, TypeError):
                pass

    for param_name in object_params:
        val = arguments.get(param_name)
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    arguments = {**arguments, param_name: parsed}
                    logger.debug(f"Coerced string to dict for parameter '{param_name}'")
            except (json.JSONDecodeError, TypeError):
                pass

    return arguments


class MCPMiddleware:
    """ASGI middleware that intercepts MCP requests and routes to appropriate MCP server.

    This middleware wraps the main FastAPI app and intercepts requests matching the
    configured prefix (default: /mcp). Non-MCP requests pass through to the inner app.

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

    Bank ID resolution priority:
        1. URL path (e.g., /mcp/{bank_id}/) → single-bank mode
        2. X-Bank-Id header → multi-bank mode
        3. HINDSIGHT_MCP_BANK_ID env var → multi-bank mode (default: "default")

    Examples:
        # Single-bank mode (recommended for agent isolation)
        claude mcp add --transport http my-agent http://localhost:8888/mcp/my-agent-bank/ \\
            --header "Authorization: Bearer <token>"

        # Multi-bank mode (for cross-bank operations)
        claude mcp add --transport http hindsight http://localhost:8888/mcp \\
            --header "X-Bank-Id: my-bank" --header "Authorization: Bearer <token>"
    """

    def __init__(
        self,
        app,
        memory: MemoryEngine,
        prefix: str = "/mcp",
        multi_bank_app=None,
        single_bank_app=None,
        multi_bank_server=None,
        single_bank_server=None,
    ):
        self.app = app
        self.prefix = prefix
        self.memory = memory
        self.tenant_extension = memory._tenant_extension

        if multi_bank_app and single_bank_app:
            # Pre-created servers (used when called via add_middleware from create_app)
            self.multi_bank_app = multi_bank_app
            self.single_bank_app = single_bank_app
            self.multi_bank_server = multi_bank_server
            self.single_bank_server = single_bank_server
        else:
            # Create servers internally (for direct construction / tests)
            global_config = _get_raw_config()
            stateless = global_config.mcp_stateless
            self.multi_bank_server = create_mcp_server(memory, multi_bank=True)
            self.multi_bank_app = self.multi_bank_server.http_app(path="/", stateless_http=stateless)
            self.single_bank_server = create_mcp_server(memory, multi_bank=False)
            self.single_bank_app = self.single_bank_server.http_app(path="/", stateless_http=stateless)

    def _get_header(self, scope: dict, name: str) -> str | None:
        """Extract a header value from ASGI scope."""
        name_lower = name.lower().encode()
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == name_lower:
                return header_value.decode()
        return None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Check if this is an MCP request (matches prefix)
        if not (path == self.prefix or path.startswith(self.prefix + "/")):
            # Not an MCP request — pass through to the inner app
            await self.app(scope, receive, send)
            return

        # Handle GET-before-POST gracefully (Claude Code v2.1.84+ sends GET probe before POST initialize).
        # Without a valid Mcp-Session-Id, GET has no meaningful response — return 200 OK so
        # the client proceeds to POST initialize instead of marking the server as failed.
        method = scope.get("method", "")
        if method == "GET":
            session_id = self._get_header(scope, "Mcp-Session-Id")
            if not session_id:
                logger.debug("MCP GET without session ID (client probe) — returning 200 OK")
                await self._send_ok(send)
                return

        # Strip prefix from path
        path = path[len(self.prefix) :] or "/"

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
        mcp_pre_authenticated = False
        if MCP_AUTH_TOKEN:
            # Legacy authentication mode - validate against static token
            if not auth_token:
                await self._send_error(send, 401, "Authorization header required")
                return
            if auth_token != MCP_AUTH_TOKEN:
                await self._send_error(send, 401, "Invalid authentication token")
                return
            # Legacy mode: mark as pre-authenticated so tenant extension won't re-validate
            tenant_context = None
            mcp_pre_authenticated = True
        else:
            # Use TenantExtension.authenticate_mcp() for auth
            try:
                auth_context = RequestContext(api_key=auth_token)
                tenant_context = await self.tenant_extension.authenticate_mcp(auth_context)
                # Capture tenant_id and api_key_id set by authenticate() for usage metering
                auth_tenant_id = auth_context.tenant_id
                auth_api_key_id = auth_context.api_key_id
            except AuthenticationError as e:
                await self._send_error(send, 401, str(e), extra_headers=e.headers)
                return

        # Set schema from tenant context so downstream DB queries use the correct schema
        schema_token = (
            _current_schema.set(tenant_context.schema_name) if tenant_context and tenant_context.schema_name else None
        )

        # Resolve bank_id: path takes priority over header.
        # Path = user's explicit connection endpoint (e.g., /mcp/my-bank/).
        # X-Bank-Id header = per-request override for multi-bank mode only.
        bank_id = None
        bank_id_from_path = False
        new_path = path

        # First, try to extract from path: /{bank_id}/...
        if path.startswith("/") and len(path) > 1:
            parts = path[1:].split("/", 1)
            if parts[0]:
                bank_id = parts[0]
                bank_id_from_path = True
                new_path = "/" + parts[1] if len(parts) > 1 else "/"

        # If no path-based bank_id, try X-Bank-Id header (multi-bank mode)
        if not bank_id:
            bank_id = self._get_header(scope, "X-Bank-Id")

        # Fall back to default bank_id
        if not bank_id:
            bank_id = DEFAULT_BANK_ID
            logger.debug(f"Using default bank_id: {bank_id}")

        # Select the appropriate MCP app based on how bank_id was provided:
        # - Path-based bank_id → single-bank app (no bank_id param, scoped tools)
        # - Header/env bank_id → multi-bank app (bank_id param, all tools)
        target_app = self.single_bank_app if bank_id_from_path else self.multi_bank_app

        # Set bank_id, api_key, tenant_id, api_key_id, and mcp_authenticated context
        bank_id_token = _current_bank_id.set(bank_id)
        # Store the auth token for tenant extension to validate
        api_key_token = _current_api_key.set(auth_token) if auth_token else None
        # Store tenant_id and api_key_id from authentication for usage metering
        tenant_id_token = _current_tenant_id.set(auth_tenant_id) if auth_tenant_id else None
        api_key_id_token = _current_api_key_id.set(auth_api_key_id) if auth_api_key_id else None
        # Store MCP pre-authentication flag to skip tenant re-validation
        mcp_auth_token = _current_mcp_authenticated.set(mcp_pre_authenticated)
        try:
            new_scope = scope.copy()
            new_scope["path"] = new_path
            # Clear root_path since we're passing directly to the app
            new_scope["root_path"] = ""

            # Ensure Accept header includes required MIME types for MCP SDK.
            # Some clients (e.g., Claude Code) don't send Accept, causing
            # the SDK to reject with 406 Not Acceptable.
            accept_header = self._get_header(new_scope, "accept")
            if not accept_header or "text/event-stream" not in accept_header:
                headers = [(k, v) for k, v in new_scope.get("headers", []) if k.lower() != b"accept"]
                headers.append((b"accept", b"application/json, text/event-stream"))
                new_scope["headers"] = headers

            # Wrap send to rewrite the SSE endpoint URL to include bank_id if using path-based routing.
            # Only rewrite SSE (text/event-stream) responses to avoid corrupting tool results
            # that might contain the literal string "data: /messages".
            is_sse_response = False

            async def send_wrapper(message):
                nonlocal is_sse_response
                if message["type"] == "http.response.start":
                    for header_name, header_value in message.get("headers", []):
                        if header_name == b"content-type" and b"text/event-stream" in header_value:
                            is_sse_response = True
                            break
                if message["type"] == "http.response.body" and bank_id_from_path and is_sse_response:
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
            _current_mcp_authenticated.reset(mcp_auth_token)
            if schema_token is not None:
                _current_schema.reset(schema_token)

    async def _send_ok(self, send):
        """Send a 200 OK response with empty body (used for GET probes without session)."""
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"{}",
            }
        )

    async def _send_error(self, send, status: int, message: str, extra_headers: dict[str, str] | None = None):
        """Send an error response."""
        body = json.dumps({"error": message}).encode()
        headers = [(b"content-type", b"application/json")]
        for key, value in (extra_headers or {}).items():
            headers.append((key.encode(), value.encode()))
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
            }
        )


def create_mcp_servers(memory: MemoryEngine):
    """Create multi-bank and single-bank MCP servers and their Starlette apps.

    Returns the servers and apps separately so lifespans can be chained before
    the middleware wraps the main app.

    Returns:
        Tuple of (multi_bank_server, single_bank_server, multi_bank_app, single_bank_app)
    """
    global_config = _get_raw_config()
    stateless = global_config.mcp_stateless

    multi_bank_server = create_mcp_server(memory, multi_bank=True)
    multi_bank_app = multi_bank_server.http_app(path="/", stateless_http=stateless)

    single_bank_server = create_mcp_server(memory, multi_bank=False)
    single_bank_app = single_bank_server.http_app(path="/", stateless_http=stateless)

    logger.info(f"MCP servers created (stateless_http={stateless})")
    return multi_bank_server, single_bank_server, multi_bank_app, single_bank_app
