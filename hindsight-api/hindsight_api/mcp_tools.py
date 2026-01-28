"""Shared MCP tool implementations for Hindsight.

This module provides the core tool logic used by both:
- mcp_local.py (stdio transport for Claude Code)
- api/mcp.py (HTTP transport for API server)
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from fastmcp import FastMCP

from hindsight_api import MemoryEngine
from hindsight_api.config import (
    DEFAULT_MCP_RECALL_DESCRIPTION,
    DEFAULT_MCP_RETAIN_DESCRIPTION,
)
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.response_models import VALID_RECALL_FACT_TYPES
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)


@dataclass
class MCPToolsConfig:
    """Configuration for MCP tools registration."""

    # How to resolve bank_id for operations
    bank_id_resolver: Callable[[], str | None]

    # How to resolve API key for tenant auth (optional)
    api_key_resolver: Callable[[], str | None] | None = None

    # Whether to include bank_id as a parameter on tools (for multi-bank support)
    include_bank_id_param: bool = False

    # Which tools to register
    tools: set[str] | None = None  # None means all tools

    # Custom descriptions (if None, uses defaults)
    retain_description: str | None = None
    recall_description: str | None = None

    # Retain behavior
    retain_fire_and_forget: bool = False  # If True, use asyncio.create_task pattern


def _get_request_context(config: MCPToolsConfig) -> RequestContext:
    """Create RequestContext with API key from resolver if available.

    This enables tenant auth to work with MCP tools by propagating
    the Bearer token from the MCP middleware to the memory engine.
    """
    api_key = config.api_key_resolver() if config.api_key_resolver else None
    return RequestContext(api_key=api_key)


def parse_timestamp(timestamp: str) -> datetime | None:
    """Parse an ISO format timestamp string.

    Args:
        timestamp: ISO format timestamp (e.g., '2024-01-15T10:30:00Z')

    Returns:
        Parsed datetime or None if invalid

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(
            f"Invalid timestamp format '{timestamp}'. "
            "Expected ISO format like '2024-01-15T10:30:00' or '2024-01-15T10:30:00Z'"
        ) from e


def build_content_dict(
    content: str,
    context: str,
    timestamp: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build a content dict for retain operations.

    Args:
        content: The memory content
        context: Category for the memory
        timestamp: Optional ISO timestamp

    Returns:
        Tuple of (content_dict, error_message). error_message is None if successful.
    """
    content_dict: dict[str, Any] = {"content": content, "context": context}

    if timestamp:
        try:
            parsed_timestamp = parse_timestamp(timestamp)
            content_dict["event_date"] = parsed_timestamp
        except ValueError as e:
            return {}, str(e)

    return content_dict, None


def register_mcp_tools(
    mcp: FastMCP,
    memory: MemoryEngine,
    config: MCPToolsConfig,
) -> None:
    """Register MCP tools on a FastMCP server.

    Args:
        mcp: FastMCP server instance
        memory: MemoryEngine instance
        config: Tool configuration
    """
    tools_to_register = config.tools or {"retain", "recall", "reflect", "list_banks", "create_bank"}

    if "retain" in tools_to_register:
        _register_retain(mcp, memory, config)

    if "recall" in tools_to_register:
        _register_recall(mcp, memory, config)

    if "reflect" in tools_to_register:
        _register_reflect(mcp, memory, config)

    if "list_banks" in tools_to_register:
        _register_list_banks(mcp, memory, config)

    if "create_bank" in tools_to_register:
        _register_create_bank(mcp, memory, config)


def _register_retain(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the retain tool."""
    description = config.retain_description or DEFAULT_MCP_RETAIN_DESCRIPTION

    if config.include_bank_id_param:
        if config.retain_fire_and_forget:

            @mcp.tool(description=description)
            async def retain(
                content: str,
                context: str = "general",
                timestamp: str | None = None,
                bank_id: str | None = None,
            ) -> dict:
                """
                Args:
                    content: The fact/memory to store (be specific and include relevant details)
                    context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                    timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                    bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                """
                import asyncio

                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return {"status": "error", "message": "No bank_id configured"}

                content_dict, error = build_content_dict(content, context, timestamp)
                if error:
                    return {"status": "error", "message": error}

                request_context = _get_request_context(config)

                async def _retain():
                    try:
                        await memory.retain_batch_async(
                            bank_id=target_bank,
                            contents=[content_dict],
                            request_context=request_context,
                        )
                    except Exception as e:
                        logger.error(f"Error storing memory: {e}", exc_info=True)

                asyncio.create_task(_retain())
                return {"status": "accepted", "message": "Memory storage initiated"}

        else:

            @mcp.tool(description=description)
            async def retain(
                content: str,
                context: str = "general",
                timestamp: str | None = None,
                async_processing: bool = True,
                bank_id: str | None = None,
            ) -> str:
                """
                Args:
                    content: The fact/memory to store (be specific and include relevant details)
                    context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                    timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                    async_processing: If True, queue for background processing and return immediately. If False, wait for completion. Default: True
                    bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                """
                try:
                    target_bank = bank_id or config.bank_id_resolver()
                    if target_bank is None:
                        return "Error: No bank_id configured"

                    content_dict, error = build_content_dict(content, context, timestamp)
                    if error:
                        return f"Error: {error}"

                    contents = [content_dict]
                    request_context = _get_request_context(config)
                    if async_processing:
                        result = await memory.submit_async_retain(
                            bank_id=target_bank, contents=contents, request_context=request_context
                        )
                        return f"Memory queued for background processing (operation_id: {result.get('operation_id', 'N/A')})"
                    else:
                        await memory.retain_batch_async(
                            bank_id=target_bank,
                            contents=contents,
                            request_context=request_context,
                        )
                        return f"Memory stored successfully in bank '{target_bank}'"
                except Exception as e:
                    logger.error(f"Error storing memory: {e}", exc_info=True)
                    return f"Error: {str(e)}"

    else:
        # No bank_id param - use fixed bank from resolver

        @mcp.tool(description=description)
        async def retain(
            content: str,
            context: str = "general",
            timestamp: str | None = None,
        ) -> dict:
            """
            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
            """
            import asyncio

            target_bank = config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(content, context, timestamp)
            if error:
                return {"status": "error", "message": error}

            request_context = _get_request_context(config)

            async def _retain():
                try:
                    await memory.retain_batch_async(
                        bank_id=target_bank,
                        contents=[content_dict],
                        request_context=request_context,
                    )
                except Exception as e:
                    logger.error(f"Error storing memory: {e}", exc_info=True)

            asyncio.create_task(_retain())
            return {"status": "accepted", "message": "Memory storage initiated"}


def _register_recall(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the recall tool."""
    description = config.recall_description or DEFAULT_MCP_RECALL_DESCRIPTION

    if config.include_bank_id_param:

        @mcp.tool(description=description)
        async def recall(
            query: str,
            max_tokens: int = 4096,
            bank_id: str | None = None,
        ) -> str | dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
                bank_id: Optional bank to search in (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                recall_result = await memory.recall_async(
                    bank_id=target_bank,
                    query=query,
                    fact_type=list(VALID_RECALL_FACT_TYPES),
                    budget=Budget.HIGH,
                    max_tokens=max_tokens,
                    request_context=_get_request_context(config),
                )

                return recall_result.model_dump_json(indent=2)
            except Exception as e:
                logger.error(f"Error searching: {e}", exc_info=True)
                return f'{{"error": "{e}", "results": []}}'

    else:

        @mcp.tool(description=description)
        async def recall(
            query: str,
            max_tokens: int = 4096,
        ) -> dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "results": []}

                recall_result = await memory.recall_async(
                    bank_id=target_bank,
                    query=query,
                    fact_type=list(VALID_RECALL_FACT_TYPES),
                    budget=Budget.HIGH,
                    max_tokens=max_tokens,
                    request_context=_get_request_context(config),
                )

                return recall_result.model_dump()
            except Exception as e:
                logger.error(f"Error searching: {e}", exc_info=True)
                return {"error": str(e), "results": []}


def _register_reflect(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the reflect tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def reflect(
            query: str,
            context: str | None = None,
            budget: str = "low",
            bank_id: str | None = None,
        ) -> str:
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
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_result = await memory.reflect_async(
                    bank_id=target_bank,
                    query=query,
                    budget=budget_enum,
                    context=context,
                    request_context=_get_request_context(config),
                )

                return reflect_result.model_dump_json(indent=2)
            except Exception as e:
                logger.error(f"Error reflecting: {e}", exc_info=True)
                return f'{{"error": "{e}", "text": ""}}'

    else:

        @mcp.tool()
        async def reflect(
            query: str,
            context: str | None = None,
            budget: str = "low",
        ) -> dict:
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
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "text": ""}

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_result = await memory.reflect_async(
                    bank_id=target_bank,
                    query=query,
                    budget=budget_enum,
                    context=context,
                    request_context=_get_request_context(config),
                )

                return reflect_result.model_dump()
            except Exception as e:
                logger.error(f"Error reflecting: {e}", exc_info=True)
                return {"error": str(e), "text": ""}


def _register_list_banks(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_banks tool."""

    @mcp.tool()
    async def list_banks() -> str:
        """
        List all available memory banks.

        Use this tool to discover what memory banks exist in the system.
        Each bank is an isolated memory store (like a separate "brain").

        Returns:
            JSON list of banks with their IDs, names, dispositions, and missions.
        """
        try:
            banks = await memory.list_banks(request_context=_get_request_context(config))
            return json.dumps({"banks": banks}, indent=2)
        except Exception as e:
            logger.error(f"Error listing banks: {e}", exc_info=True)
            return f'{{"error": "{e}", "banks": []}}'


def _register_create_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_bank tool."""

    @mcp.tool()
    async def create_bank(bank_id: str, name: str | None = None, mission: str | None = None) -> str:
        """
        Create a new memory bank or get an existing one.

        Memory banks are isolated stores - each one is like a separate "brain" for a user/agent.
        Banks are auto-created with default settings if they don't exist.

        Args:
            bank_id: Unique identifier for the bank (e.g., 'user-123', 'agent-alpha')
            name: Optional human-friendly name for the bank
            mission: Optional mission describing who the agent is and what they're trying to accomplish
        """
        try:
            request_context = _get_request_context(config)
            # get_bank_profile auto-creates bank if it doesn't exist
            profile = await memory.get_bank_profile(bank_id, request_context=request_context)

            # Update name/mission if provided
            if name is not None or mission is not None:
                await memory.update_bank(
                    bank_id,
                    name=name,
                    mission=mission,
                    request_context=request_context,
                )
                # Fetch updated profile
                profile = await memory.get_bank_profile(bank_id, request_context=request_context)

            # Serialize disposition if it's a Pydantic model
            if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                profile["disposition"] = profile["disposition"].model_dump()
            return json.dumps(profile, indent=2)
        except Exception as e:
            logger.error(f"Error creating bank: {e}", exc_info=True)
            return f'{{"error": "{e}"}}'
