"""Hindsight MCP Server implementation using FastMCP."""

import json
import logging
import os
from contextvars import ContextVar

from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.api.http import BankListItem, BankListResponse, BankProfileResponse, DispositionTraits
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES
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

# Context variable to hold the current bank_id
_current_bank_id: ContextVar[str | None] = ContextVar("current_bank_id", default=None)


def get_current_bank_id() -> str | None:
    """Get the current bank_id from context."""
    return _current_bank_id.get()


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

    @mcp.tool()
    async def retain(content: str, context: str = "general", bank_id: str | None = None) -> str:
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
            bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
        """
        try:
            target_bank = bank_id or get_current_bank_id()
            if target_bank is None:
                return "Error: No bank_id configured"
            await memory.retain_batch_async(
                bank_id=target_bank,
                contents=[{"content": content, "context": context}],
                request_context=RequestContext(),
            )
            return f"Memory stored successfully in bank '{target_bank}'"
        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return f"Error: {str(e)}"

    @mcp.tool()
    async def recall(query: str, max_tokens: int = 4096, bank_id: str | None = None) -> str:
        """
        Search memories to provide personalized, context-aware responses.

        Use this tool PROACTIVELY to:
        - Check user's preferences before making suggestions
        - Recall user's history to provide continuity
        - Remember user's goals and context
        - Personalize responses based on past interactions

        Args:
            query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
            max_tokens: Maximum tokens in the response (default: 4096)
            bank_id: Optional bank to search in (defaults to session bank). Use for cross-bank operations.
        """
        try:
            target_bank = bank_id or get_current_bank_id()
            if target_bank is None:
                return "Error: No bank_id configured"
            from hindsight_api.engine.memory_engine import Budget

            recall_result = await memory.recall_async(
                bank_id=target_bank,
                query=query,
                fact_type=list(VALID_RECALL_FACT_TYPES),
                budget=Budget.HIGH,
                max_tokens=max_tokens,
                request_context=RequestContext(),
            )

            # Use model's JSON serialization
            return recall_result.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            return f'{{"error": "{e}", "results": []}}'

    @mcp.tool()
    async def reflect(query: str, context: str | None = None, budget: str = "low", bank_id: str | None = None) -> str:
        """
        Generate thoughtful analysis by synthesizing stored memories with the bank's personality.

        WHEN TO USE THIS TOOL:
        Use reflect when you need reasoned analysis, not just fact retrieval. This tool
        thinks through the question using everything the bank knows and its personality traits.

        EXAMPLES OF GOOD QUERIES:
        - "What patterns have emerged in how I approach debugging?"
        - "Based on my past decisions, what architectural style do I prefer?"
        - "What might be the best approach for this problem given what you know about me?"
        - "How should I prioritize these tasks based on my goals?"

        HOW IT DIFFERS FROM RECALL:
        - recall: Returns raw facts matching your search (fast lookup)
        - reflect: Reasons across memories to form a synthesized answer (deeper analysis)

        Use recall for "what did I say about X?" and reflect for "what should I do about X?"

        Args:
            query: The question or topic to reflect on
            context: Optional context about why this reflection is needed
            budget: Search budget - 'low', 'mid', or 'high' (default: 'low')
            bank_id: Optional bank to reflect in (defaults to session bank). Use for cross-bank operations.
        """
        try:
            target_bank = bank_id or get_current_bank_id()
            if target_bank is None:
                return "Error: No bank_id configured"
            from hindsight_api.engine.memory_engine import Budget

            # Map string budget to enum
            budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
            budget_enum = budget_map.get(budget.lower(), Budget.LOW)

            reflect_result = await memory.reflect_async(
                bank_id=target_bank,
                query=query,
                budget=budget_enum,
                context=context,
                request_context=RequestContext(),
            )

            return reflect_result.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error reflecting: {e}", exc_info=True)
            return f'{{"error": "{e}", "text": ""}}'

    @mcp.tool()
    async def list_banks() -> str:
        """
        List all available memory banks.

        Use this to discover banks for orchestration or to find
        the correct bank_id for cross-bank operations.

        Returns:
            JSON object with banks array containing bank_id, name, disposition, background, and timestamps
        """
        try:
            banks = await memory.list_banks(request_context=RequestContext())
            bank_items = [
                BankListItem(
                    bank_id=b.get("bank_id") or b.get("id"),
                    name=b.get("name"),
                    disposition=DispositionTraits(
                        **b.get("disposition", {"skepticism": 3, "literalism": 3, "empathy": 3})
                    ),
                    background=b.get("background"),
                    created_at=str(b.get("created_at")) if b.get("created_at") else None,
                    updated_at=str(b.get("updated_at")) if b.get("updated_at") else None,
                )
                for b in banks
            ]
            return BankListResponse(banks=bank_items).model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error listing banks: {e}", exc_info=True)
            return f'{{"error": "{e}", "banks": []}}'

    @mcp.tool()
    async def create_bank(bank_id: str, name: str | None = None, background: str | None = None) -> str:
        """
        Create or update a memory bank.

        Use this to create new banks for different agents, sessions, or purposes.
        Banks are isolated memory stores - each bank has its own memories and personality.

        Args:
            bank_id: Unique identifier for the bank (e.g., 'orchestrator-memory', 'agent-1')
            name: Human-readable name for the bank
            background: Context about what this bank stores or its purpose
        """
        try:
            # Get or create the bank profile (auto-creates with defaults)
            await memory.get_bank_profile(bank_id, request_context=RequestContext())

            # Update name and/or background if provided
            if name is not None or background is not None:
                await memory.update_bank(bank_id, name=name, background=background, request_context=RequestContext())

            # Get final profile and return using BankProfileResponse model
            profile = await memory.get_bank_profile(bank_id, request_context=RequestContext())
            disposition = profile.get("disposition")
            if hasattr(disposition, "model_dump"):
                disposition_traits = DispositionTraits(**disposition.model_dump())
            else:
                disposition_traits = DispositionTraits(
                    **dict(disposition or {"skepticism": 3, "literalism": 3, "empathy": 3})
                )

            response = BankProfileResponse(
                bank_id=bank_id,
                name=profile.get("name") or "",
                disposition=disposition_traits,
                background=profile.get("background") or "",
            )
            return response.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error creating bank: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

    return mcp


class MCPMiddleware:
    """ASGI middleware that extracts bank_id from header or path and sets context.

    Bank ID can be provided via:
    1. X-Bank-Id header (recommended for Claude Code)
    2. URL path: /mcp/{bank_id}/
    3. Environment variable HINDSIGHT_MCP_BANK_ID (fallback default)

    For Claude Code, configure with:
        claude mcp add --transport http hindsight http://localhost:8888/mcp \\
            --header "X-Bank-Id: my-bank"
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

        # Set bank_id context
        token = _current_bank_id.set(bank_id)
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
            _current_bank_id.reset(token)

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
