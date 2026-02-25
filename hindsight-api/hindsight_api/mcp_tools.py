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

    # How to resolve tenant_id for usage metering (set by MCP middleware after auth)
    tenant_id_resolver: Callable[[], str | None] | None = None

    # How to resolve api_key_id for usage metering (set by MCP middleware after auth)
    api_key_id_resolver: Callable[[], str | None] | None = None

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
    """Create RequestContext with auth details from resolvers.

    This enables tenant auth and usage metering to work with MCP tools by propagating
    the authentication results from the MCP middleware to the memory engine.
    """
    api_key = config.api_key_resolver() if config.api_key_resolver else None
    tenant_id = config.tenant_id_resolver() if config.tenant_id_resolver else None
    api_key_id = config.api_key_id_resolver() if config.api_key_id_resolver else None
    return RequestContext(api_key=api_key, tenant_id=tenant_id, api_key_id=api_key_id)


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
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    document_id: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Build a content dict for retain operations.

    Args:
        content: The memory content
        context: Category for the memory
        timestamp: Optional ISO timestamp
        tags: Optional tags for scoped visibility filtering
        metadata: Optional key-value metadata to attach to the memory
        document_id: Optional document ID to associate the memory with

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

    if tags is not None:
        content_dict["tags"] = tags
    if metadata is not None:
        content_dict["metadata"] = metadata
    if document_id is not None:
        content_dict["document_id"] = document_id

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
    tools_to_register = config.tools or {
        "retain",
        "recall",
        "reflect",
        "list_banks",
        "create_bank",
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
        "get_bank_stats",
        "update_bank",
        "delete_bank",
        "clear_memories",
    }

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

    # Mental model tools
    if "list_mental_models" in tools_to_register:
        _register_list_mental_models(mcp, memory, config)

    if "get_mental_model" in tools_to_register:
        _register_get_mental_model(mcp, memory, config)

    if "create_mental_model" in tools_to_register:
        _register_create_mental_model(mcp, memory, config)

    if "update_mental_model" in tools_to_register:
        _register_update_mental_model(mcp, memory, config)

    if "delete_mental_model" in tools_to_register:
        _register_delete_mental_model(mcp, memory, config)

    if "refresh_mental_model" in tools_to_register:
        _register_refresh_mental_model(mcp, memory, config)

    # Directive tools
    if "list_directives" in tools_to_register:
        _register_list_directives(mcp, memory, config)

    if "create_directive" in tools_to_register:
        _register_create_directive(mcp, memory, config)

    if "delete_directive" in tools_to_register:
        _register_delete_directive(mcp, memory, config)

    # Memory browsing tools
    if "list_memories" in tools_to_register:
        _register_list_memories(mcp, memory, config)

    if "get_memory" in tools_to_register:
        _register_get_memory(mcp, memory, config)

    if "delete_memory" in tools_to_register:
        _register_delete_memory(mcp, memory, config)

    # Document tools
    if "list_documents" in tools_to_register:
        _register_list_documents(mcp, memory, config)

    if "get_document" in tools_to_register:
        _register_get_document(mcp, memory, config)

    if "delete_document" in tools_to_register:
        _register_delete_document(mcp, memory, config)

    # Operation tools
    if "list_operations" in tools_to_register:
        _register_list_operations(mcp, memory, config)

    if "get_operation" in tools_to_register:
        _register_get_operation(mcp, memory, config)

    if "cancel_operation" in tools_to_register:
        _register_cancel_operation(mcp, memory, config)

    # Tags & bank tools
    if "list_tags" in tools_to_register:
        _register_list_tags(mcp, memory, config)

    if "get_bank" in tools_to_register:
        _register_get_bank(mcp, memory, config)

    if "get_bank_stats" in tools_to_register:
        _register_get_bank_stats(mcp, memory, config)

    if "update_bank" in tools_to_register:
        _register_update_bank(mcp, memory, config)

    if "delete_bank" in tools_to_register:
        _register_delete_bank(mcp, memory, config)

    if "clear_memories" in tools_to_register:
        _register_clear_memories(mcp, memory, config)

    _apply_bank_tool_filtering(mcp, memory, config)


def _apply_bank_tool_filtering(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Filter bank-level mcp_enabled_tools from both tools/list and tool invocation.

    Wraps _tool_manager.get_tools() so that:
    - tools/list only returns permitted tools (they are hidden, not just blocked)
    - tools/call for a disabled tool raises NotFoundError (via the manager) before run()

    tool.run wrappers are kept as defense-in-depth for any caller that bypasses the manager.
    """
    try:
        tool_manager = mcp._tool_manager
        original_get_tools = tool_manager.get_tools

        async def _filtered_get_tools():
            all_tools = await original_get_tools()
            bank_id = config.bank_id_resolver()
            if not bank_id:
                return all_tools
            request_context = _get_request_context(config)
            bank_cfg = await memory._config_resolver.get_bank_config(bank_id, request_context)
            enabled: list[str] | None = bank_cfg.get("mcp_enabled_tools")
            if enabled is None:
                return all_tools
            enabled_set = set(enabled)
            return {k: v for k, v in all_tools.items() if k in enabled_set}

        setattr(tool_manager, "get_tools", _filtered_get_tools)

        # Defense-in-depth: also wrap tool.run for any direct caller that bypasses the manager
        for name, tool in tool_manager._tools.items():
            original_run = tool.run

            async def _filtered_run(arguments, _name=name, _orig=original_run):
                bank_id = config.bank_id_resolver()
                if bank_id:
                    request_context = _get_request_context(config)
                    bank_cfg = await memory._config_resolver.get_bank_config(bank_id, request_context)
                    enabled: list[str] | None = bank_cfg.get("mcp_enabled_tools")
                    if enabled is not None and _name not in enabled:
                        raise ValueError(f"Tool '{_name}' is not enabled for bank '{bank_id}'")
                return await _orig(arguments)

            object.__setattr__(tool, "run", _filtered_run)
    except (AttributeError, KeyError) as e:
        logger.warning(f"Could not apply bank tool filtering: {e}")


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
                tags: list[str] | None = None,
                metadata: dict[str, str] | None = None,
                document_id: str | None = None,
                bank_id: str | None = None,
            ) -> dict:
                """
                Args:
                    content: The fact/memory to store (be specific and include relevant details)
                    context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                    timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                    tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                    metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                    document_id: Optional document ID to associate this memory with
                    bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                """
                import asyncio

                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return {"status": "error", "message": "No bank_id configured"}

                content_dict, error = build_content_dict(content, context, timestamp, tags, metadata, document_id)
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
                tags: list[str] | None = None,
                metadata: dict[str, str] | None = None,
                document_id: str | None = None,
                async_processing: bool = True,
                bank_id: str | None = None,
            ) -> str:
                """
                Args:
                    content: The fact/memory to store (be specific and include relevant details)
                    context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                    timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                    tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                    metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                    document_id: Optional document ID to associate this memory with
                    async_processing: If True, queue for background processing and return immediately. If False, wait for completion. Default: True
                    bank_id: Optional bank to store in (defaults to session bank). Use for cross-bank operations.
                """
                try:
                    target_bank = bank_id or config.bank_id_resolver()
                    if target_bank is None:
                        return "Error: No bank_id configured"

                    content_dict, error = build_content_dict(content, context, timestamp, tags, metadata, document_id)
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
            tags: list[str] | None = None,
            metadata: dict[str, str] | None = None,
            document_id: str | None = None,
        ) -> dict:
            """
            Args:
                content: The fact/memory to store (be specific and include relevant details)
                context: Category for the memory (e.g., 'preferences', 'work', 'hobbies', 'family'). Default: 'general'
                timestamp: When this event/fact occurred (ISO format, e.g., '2024-01-15T10:30:00Z'). Useful for timeline tracking.
                tags: Optional tags for scoped visibility filtering (e.g., ['project:alpha', 'user:123'])
                metadata: Optional key-value metadata to attach (e.g., {'source': 'slack', 'channel': 'general'})
                document_id: Optional document ID to associate this memory with
            """
            import asyncio

            target_bank = config.bank_id_resolver()
            if target_bank is None:
                return {"status": "error", "message": "No bank_id configured"}

            content_dict, error = build_content_dict(content, context, timestamp, tags, metadata, document_id)
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
            budget: str = "high",
            types: list[str] | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
            query_timestamp: str | None = None,
            bank_id: str | None = None,
        ) -> str | dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
                budget: Search budget - 'low', 'mid', or 'high' (default: 'high'). Higher budgets search more thoroughly.
                types: Fact types to include (e.g., ['world', 'experience']). Default: all types.
                tags: Optional tags to filter results by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                query_timestamp: Temporal context for the query (ISO format, e.g., '2024-01-15T10:30:00Z'). Helps retrieve time-relevant memories.
                bank_id: Optional bank to search in (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.HIGH)
                fact_types = types if types is not None else list(VALID_RECALL_FACT_TYPES)

                recall_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "fact_type": fact_types,
                    "budget": budget_enum,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if tags is not None:
                    recall_kwargs["tags"] = tags
                    recall_kwargs["tags_match"] = tags_match
                if query_timestamp is not None:
                    recall_kwargs["question_date"] = parse_timestamp(query_timestamp)

                recall_result = await memory.recall_async(**recall_kwargs)

                return recall_result.model_dump_json(indent=2)
            except ValueError as e:
                return f'{{"error": "{e}", "results": []}}'
            except Exception as e:
                logger.error(f"Error searching: {e}", exc_info=True)
                return f'{{"error": "{e}", "results": []}}'

    else:

        @mcp.tool(description=description)
        async def recall(
            query: str,
            max_tokens: int = 4096,
            budget: str = "high",
            types: list[str] | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
            query_timestamp: str | None = None,
        ) -> dict:
            """
            Args:
                query: Natural language search query (e.g., "user's food preferences", "what projects is user working on")
                max_tokens: Maximum tokens to return in results (default: 4096)
                budget: Search budget - 'low', 'mid', or 'high' (default: 'high'). Higher budgets search more thoroughly.
                types: Fact types to include (e.g., ['world', 'experience']). Default: all types.
                tags: Optional tags to filter results by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                query_timestamp: Temporal context for the query (ISO format, e.g., '2024-01-15T10:30:00Z'). Helps retrieve time-relevant memories.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "results": []}

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.HIGH)
                fact_types = types if types is not None else list(VALID_RECALL_FACT_TYPES)

                recall_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "fact_type": fact_types,
                    "budget": budget_enum,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if tags is not None:
                    recall_kwargs["tags"] = tags
                    recall_kwargs["tags_match"] = tags_match
                if query_timestamp is not None:
                    recall_kwargs["question_date"] = parse_timestamp(query_timestamp)

                recall_result = await memory.recall_async(**recall_kwargs)

                return recall_result.model_dump()
            except ValueError as e:
                return {"error": str(e), "results": []}
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
            max_tokens: int = 4096,
            response_schema: dict | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
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
                max_tokens: Maximum tokens for the response (default: 4096)
                response_schema: Optional JSON schema for structured output. When provided, the response includes a 'structured_output' field.
                tags: Optional tags to filter memories by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
                bank_id: Optional bank to reflect in (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return "Error: No bank_id configured"

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "budget": budget_enum,
                    "context": context,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if response_schema is not None:
                    reflect_kwargs["response_schema"] = response_schema
                if tags is not None:
                    reflect_kwargs["tags"] = tags
                    reflect_kwargs["tags_match"] = tags_match

                reflect_result = await memory.reflect_async(**reflect_kwargs)

                result_data = json.loads(reflect_result.model_dump_json(indent=2))
                if response_schema is not None and hasattr(reflect_result, "structured_output"):
                    result_data["structured_output"] = reflect_result.structured_output
                return json.dumps(result_data, indent=2)
            except Exception as e:
                logger.error(f"Error reflecting: {e}", exc_info=True)
                return f'{{"error": "{e}", "text": ""}}'

    else:

        @mcp.tool()
        async def reflect(
            query: str,
            context: str | None = None,
            budget: str = "low",
            max_tokens: int = 4096,
            response_schema: dict | None = None,
            tags: list[str] | None = None,
            tags_match: str = "any",
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
                max_tokens: Maximum tokens for the response (default: 4096)
                response_schema: Optional JSON schema for structured output. When provided, the response includes a 'structured_output' field.
                tags: Optional tags to filter memories by (e.g., ['project:alpha'])
                tags_match: How to match tags - 'any' (match any tag) or 'all' (match all tags). Default: 'any'
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "text": ""}

                budget_map = {"low": Budget.LOW, "mid": Budget.MID, "high": Budget.HIGH}
                budget_enum = budget_map.get(budget.lower(), Budget.LOW)

                reflect_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "query": query,
                    "budget": budget_enum,
                    "context": context,
                    "max_tokens": max_tokens,
                    "request_context": _get_request_context(config),
                }
                if response_schema is not None:
                    reflect_kwargs["response_schema"] = response_schema
                if tags is not None:
                    reflect_kwargs["tags"] = tags
                    reflect_kwargs["tags_match"] = tags_match

                reflect_result = await memory.reflect_async(**reflect_kwargs)

                result_data = reflect_result.model_dump()
                if response_schema is not None and hasattr(reflect_result, "structured_output"):
                    result_data["structured_output"] = reflect_result.structured_output
                return result_data
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


def _validate_mental_model_inputs(
    name: str | None = None, source_query: str | None = None, max_tokens: int | None = None
) -> str | None:
    """Validate mental model inputs, returning an error message or None if valid."""
    if name is not None and not name.strip():
        return "name cannot be empty"
    if source_query is not None and not source_query.strip():
        return "source_query cannot be empty"
    if max_tokens is not None and (max_tokens < 256 or max_tokens > 8192):
        return f"max_tokens must be between 256 and 8192, got {max_tokens}"
    return None


# =========================================================================
# MENTAL MODEL TOOLS
# =========================================================================


def _register_list_mental_models(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_mental_models tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_mental_models(
            tags: list[str] | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            List mental models (pinned reflections) for a memory bank.

            Mental models are living documents that stay current by periodically re-running
            a source query through reflect. Use them to maintain up-to-date summaries,
            preferences, or synthesized knowledge.

            Args:
                tags: Optional tags to filter by (returns models matching any tag)
                bank_id: Optional bank to list from (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured", "items": []}'

                models = await memory.list_mental_models(
                    bank_id=target_bank,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"items": models}, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing mental models: {e}", exc_info=True)
                return f'{{"error": "{e}", "items": []}}'

    else:

        @mcp.tool()
        async def list_mental_models(
            tags: list[str] | None = None,
        ) -> dict:
            """
            List mental models (pinned reflections) for this memory bank.

            Mental models are living documents that stay current by periodically re-running
            a source query through reflect. Use them to maintain up-to-date summaries,
            preferences, or synthesized knowledge.

            Args:
                tags: Optional tags to filter by (returns models matching any tag)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "items": []}

                models = await memory.list_mental_models(
                    bank_id=target_bank,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return {"items": models}
            except Exception as e:
                logger.error(f"Error listing mental models: {e}", exc_info=True)
                return {"error": str(e), "items": []}


def _register_get_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def get_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific mental model by ID.

            Returns the full mental model including its generated content, source query,
            and metadata. Use list_mental_models first to discover available model IDs.

            Args:
                mental_model_id: The ID of the mental model to retrieve
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                model = await memory.get_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if model is None:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps(model, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error getting mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def get_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Get a specific mental model by ID.

            Returns the full mental model including its generated content, source query,
            and metadata. Use list_mental_models first to discover available model IDs.

            Args:
                mental_model_id: The ID of the mental model to retrieve
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                model = await memory.get_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if model is None:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return model
            except Exception as e:
                logger.error(f"Error getting mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_create_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def create_mental_model(
            name: str,
            source_query: str,
            mental_model_id: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int = 2048,
            trigger_refresh_after_consolidation: bool = False,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a new mental model (pinned reflection).

            A mental model is a living document generated by running the source_query through
            reflect. The content is auto-generated asynchronously - use the returned operation_id
            to track progress.

            EXAMPLES:
            - name="Coding Preferences", source_query="What coding patterns and tools does the user prefer?"
            - name="Project Goals", source_query="What are the user's current project goals and priorities?"
            - name="Communication Style", source_query="How does the user prefer to communicate?"

            Args:
                name: Human-readable name for the mental model
                source_query: The query to run through reflect to generate content
                mental_model_id: Optional custom ID (alphanumeric lowercase with hyphens). Auto-generated if not provided.
                tags: Optional tags for scoped visibility filtering
                max_tokens: Maximum tokens for generated content (256-8192, default: 2048)
                trigger_refresh_after_consolidation: If True, automatically refresh this model after memory consolidation. Default: False
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens
                )
                if validation_error:
                    return json.dumps({"error": validation_error})

                request_context = _get_request_context(config)
                trigger = {"refresh_after_consolidation": trigger_refresh_after_consolidation}

                # Create with placeholder content
                model = await memory.create_mental_model(
                    bank_id=target_bank,
                    name=name,
                    source_query=source_query,
                    content="Generating content...",
                    mental_model_id=mental_model_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    request_context=request_context,
                )

                # Schedule async refresh to generate actual content
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=model["id"],
                    request_context=request_context,
                )

                return json.dumps(
                    {
                        "mental_model_id": model["id"],
                        "operation_id": result["operation_id"],
                        "status": "created",
                        "message": f"Mental model '{name}' created. Content is being generated asynchronously.",
                    }
                )
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error creating mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def create_mental_model(
            name: str,
            source_query: str,
            mental_model_id: str | None = None,
            tags: list[str] | None = None,
            max_tokens: int = 2048,
            trigger_refresh_after_consolidation: bool = False,
        ) -> dict:
            """
            Create a new mental model (pinned reflection).

            A mental model is a living document generated by running the source_query through
            reflect. The content is auto-generated asynchronously - use the returned operation_id
            to track progress.

            EXAMPLES:
            - name="Coding Preferences", source_query="What coding patterns and tools does the user prefer?"
            - name="Project Goals", source_query="What are the user's current project goals and priorities?"
            - name="Communication Style", source_query="How does the user prefer to communicate?"

            Args:
                name: Human-readable name for the mental model
                source_query: The query to run through reflect to generate content
                mental_model_id: Optional custom ID (alphanumeric lowercase with hyphens). Auto-generated if not provided.
                tags: Optional tags for scoped visibility filtering
                max_tokens: Maximum tokens for generated content (256-8192, default: 2048)
                trigger_refresh_after_consolidation: If True, automatically refresh this model after memory consolidation. Default: False
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens
                )
                if validation_error:
                    return {"error": validation_error}

                request_context = _get_request_context(config)
                trigger = {"refresh_after_consolidation": trigger_refresh_after_consolidation}

                model = await memory.create_mental_model(
                    bank_id=target_bank,
                    name=name,
                    source_query=source_query,
                    content="Generating content...",
                    mental_model_id=mental_model_id,
                    tags=tags,
                    max_tokens=max_tokens,
                    trigger=trigger,
                    request_context=request_context,
                )

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=model["id"],
                    request_context=request_context,
                )

                return {
                    "mental_model_id": model["id"],
                    "operation_id": result["operation_id"],
                    "status": "created",
                    "message": f"Mental model '{name}' created. Content is being generated asynchronously.",
                }
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error creating mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_update_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def update_mental_model(
            mental_model_id: str,
            name: str | None = None,
            source_query: str | None = None,
            max_tokens: int | None = None,
            tags: list[str] | None = None,
            trigger_refresh_after_consolidation: bool | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Update a mental model's metadata.

            Changes the name, source query, or tags of an existing mental model.
            To regenerate the content, use refresh_mental_model after updating the source query.

            Args:
                mental_model_id: The ID of the mental model to update
                name: New name (leave None to keep current)
                source_query: New source query (leave None to keep current)
                max_tokens: New max tokens for content generation (256-8192, leave None to keep current)
                tags: New tags (leave None to keep current)
                trigger_refresh_after_consolidation: If set, update whether this model auto-refreshes after consolidation
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens
                )
                if validation_error:
                    return json.dumps({"error": validation_error})

                update_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "mental_model_id": mental_model_id,
                    "name": name,
                    "source_query": source_query,
                    "max_tokens": max_tokens,
                    "tags": tags,
                    "request_context": _get_request_context(config),
                }
                if trigger_refresh_after_consolidation is not None:
                    update_kwargs["trigger"] = {"refresh_after_consolidation": trigger_refresh_after_consolidation}

                model = await memory.update_mental_model(**update_kwargs)
                if model is None:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps(model, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error updating mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def update_mental_model(
            mental_model_id: str,
            name: str | None = None,
            source_query: str | None = None,
            max_tokens: int | None = None,
            tags: list[str] | None = None,
            trigger_refresh_after_consolidation: bool | None = None,
        ) -> dict:
            """
            Update a mental model's metadata.

            Changes the name, source query, or tags of an existing mental model.
            To regenerate the content, use refresh_mental_model after updating the source query.

            Args:
                mental_model_id: The ID of the mental model to update
                name: New name (leave None to keep current)
                source_query: New source query (leave None to keep current)
                max_tokens: New max tokens for content generation (256-8192, leave None to keep current)
                tags: New tags (leave None to keep current)
                trigger_refresh_after_consolidation: If set, update whether this model auto-refreshes after consolidation
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                validation_error = _validate_mental_model_inputs(
                    name=name, source_query=source_query, max_tokens=max_tokens
                )
                if validation_error:
                    return {"error": validation_error}

                update_kwargs: dict[str, Any] = {
                    "bank_id": target_bank,
                    "mental_model_id": mental_model_id,
                    "name": name,
                    "source_query": source_query,
                    "max_tokens": max_tokens,
                    "tags": tags,
                    "request_context": _get_request_context(config),
                }
                if trigger_refresh_after_consolidation is not None:
                    update_kwargs["trigger"] = {"refresh_after_consolidation": trigger_refresh_after_consolidation}

                model = await memory.update_mental_model(**update_kwargs)
                if model is None:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return model
            except Exception as e:
                logger.error(f"Error updating mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def delete_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a mental model.

            Permanently removes a mental model and its generated content.

            Args:
                mental_model_id: The ID of the mental model to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                deleted = await memory.delete_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return json.dumps({"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"})
                return json.dumps({"status": "deleted", "mental_model_id": mental_model_id})
            except Exception as e:
                logger.error(f"Error deleting mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def delete_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Delete a mental model.

            Permanently removes a mental model and its generated content.

            Args:
                mental_model_id: The ID of the mental model to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                deleted = await memory.delete_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return {"error": f"Mental model '{mental_model_id}' not found in bank '{target_bank}'"}
                return {"status": "deleted", "mental_model_id": mental_model_id}
            except Exception as e:
                logger.error(f"Error deleting mental model: {e}", exc_info=True)
                return {"error": str(e)}


def _register_refresh_mental_model(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the refresh_mental_model tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def refresh_mental_model(
            mental_model_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Refresh a mental model by re-running its source query.

            Schedules an async task to re-run the source query through reflect and update the
            mental model's content with fresh results. Use this after adding new memories or
            when the mental model's content may be stale.

            Args:
                mental_model_id: The ID of the mental model to refresh
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(
                    {
                        "operation_id": result["operation_id"],
                        "status": "queued",
                        "message": f"Refresh queued for mental model '{mental_model_id}'.",
                    }
                )
            except ValueError as e:
                return json.dumps({"error": str(e)})
            except Exception as e:
                logger.error(f"Error refreshing mental model: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def refresh_mental_model(
            mental_model_id: str,
        ) -> dict:
            """
            Refresh a mental model by re-running its source query.

            Schedules an async task to re-run the source query through reflect and update the
            mental model's content with fresh results. Use this after adding new memories or
            when the mental model's content may be stale.

            Args:
                mental_model_id: The ID of the mental model to refresh
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.submit_async_refresh_mental_model(
                    bank_id=target_bank,
                    mental_model_id=mental_model_id,
                    request_context=_get_request_context(config),
                )
                return {
                    "operation_id": result["operation_id"],
                    "status": "queued",
                    "message": f"Refresh queued for mental model '{mental_model_id}'.",
                }
            except ValueError as e:
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"Error refreshing mental model: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# DIRECTIVE TOOLS
# =========================================================================


def _register_list_directives(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_directives tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_directives(
            tags: list[str] | None = None,
            active_only: bool = True,
            bank_id: str | None = None,
        ) -> str:
            """
            List directives for a memory bank.

            Directives are instructions that guide how the memory engine processes and
            responds to queries. They influence reflect behavior and memory organization.

            Args:
                tags: Optional tags to filter by
                active_only: If True, only return active directives (default: True)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                directives = await memory.list_directives(
                    target_bank,
                    tags=tags,
                    active_only=active_only,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"items": directives}, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing directives: {e}", exc_info=True)
                return f'{{"error": "{e}", "items": []}}'

    else:

        @mcp.tool()
        async def list_directives(
            tags: list[str] | None = None,
            active_only: bool = True,
        ) -> dict:
            """
            List directives for this memory bank.

            Directives are instructions that guide how the memory engine processes and
            responds to queries. They influence reflect behavior and memory organization.

            Args:
                tags: Optional tags to filter by
                active_only: If True, only return active directives (default: True)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured", "items": []}

                directives = await memory.list_directives(
                    target_bank,
                    tags=tags,
                    active_only=active_only,
                    request_context=_get_request_context(config),
                )
                return {"items": directives}
            except Exception as e:
                logger.error(f"Error listing directives: {e}", exc_info=True)
                return {"error": str(e), "items": []}


def _register_create_directive(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the create_directive tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def create_directive(
            name: str,
            content: str,
            priority: int = 0,
            is_active: bool = True,
            tags: list[str] | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Create a new directive for a memory bank.

            Directives guide how the memory engine processes queries and generates reflections.

            Args:
                name: Human-readable name for the directive
                content: The directive content/instructions
                priority: Priority level (higher = more important, default: 0)
                is_active: Whether the directive is active (default: True)
                tags: Optional tags for filtering
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                directive = await memory.create_directive(
                    target_bank,
                    name=name,
                    content=content,
                    priority=priority,
                    is_active=is_active,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return json.dumps(directive, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error creating directive: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def create_directive(
            name: str,
            content: str,
            priority: int = 0,
            is_active: bool = True,
            tags: list[str] | None = None,
        ) -> dict:
            """
            Create a new directive for this memory bank.

            Directives guide how the memory engine processes queries and generates reflections.

            Args:
                name: Human-readable name for the directive
                content: The directive content/instructions
                priority: Priority level (higher = more important, default: 0)
                is_active: Whether the directive is active (default: True)
                tags: Optional tags for filtering
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                directive = await memory.create_directive(
                    target_bank,
                    name=name,
                    content=content,
                    priority=priority,
                    is_active=is_active,
                    tags=tags,
                    request_context=_get_request_context(config),
                )
                return directive
            except Exception as e:
                logger.error(f"Error creating directive: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_directive(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_directive tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def delete_directive(
            directive_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a directive.

            Permanently removes a directive from the memory bank.

            Args:
                directive_id: The ID of the directive to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                deleted = await memory.delete_directive(
                    target_bank,
                    directive_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return json.dumps({"error": f"Directive '{directive_id}' not found"})
                return json.dumps({"status": "deleted", "directive_id": directive_id})
            except Exception as e:
                logger.error(f"Error deleting directive: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def delete_directive(
            directive_id: str,
        ) -> dict:
            """
            Delete a directive.

            Permanently removes a directive from this memory bank.

            Args:
                directive_id: The ID of the directive to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                deleted = await memory.delete_directive(
                    target_bank,
                    directive_id,
                    request_context=_get_request_context(config),
                )
                if not deleted:
                    return {"error": f"Directive '{directive_id}' not found"}
                return {"status": "deleted", "directive_id": directive_id}
            except Exception as e:
                logger.error(f"Error deleting directive: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# MEMORY BROWSING TOOLS
# =========================================================================


def _register_list_memories(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_memories tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_memories(
            type: str | None = None,
            q: str | None = None,
            limit: int = 100,
            offset: int = 0,
            bank_id: str | None = None,
        ) -> str:
            """
            Browse stored memories with optional filtering.

            Lists memory units (facts) stored in the bank. Unlike recall, this is a direct
            browse/search without relevance ranking.

            Args:
                type: Filter by fact type: 'world', 'experience', or 'opinion'
                q: Optional text search query to filter memories
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_memory_units(
                    target_bank,
                    fact_type=type,
                    search_query=q,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing memories: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def list_memories(
            type: str | None = None,
            q: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> dict:
            """
            Browse stored memories with optional filtering.

            Lists memory units (facts) stored in the bank. Unlike recall, this is a direct
            browse/search without relevance ranking.

            Args:
                type: Filter by fact type: 'world', 'experience', or 'opinion'
                q: Optional text search query to filter memories
                limit: Maximum number of results (default: 100)
                offset: Pagination offset (default: 0)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_memory_units(
                    target_bank,
                    fact_type=type,
                    search_query=q,
                    limit=limit,
                    offset=offset,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error listing memories: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_memory(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_memory tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def get_memory(
            memory_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific memory by ID.

            Returns the full memory unit including content, metadata, and timestamps.

            Args:
                memory_id: The ID of the memory to retrieve
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_memory_unit(
                    target_bank,
                    memory_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Memory '{memory_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error getting memory: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def get_memory(
            memory_id: str,
        ) -> dict:
            """
            Get a specific memory by ID.

            Returns the full memory unit including content, metadata, and timestamps.

            Args:
                memory_id: The ID of the memory to retrieve
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_memory_unit(
                    target_bank,
                    memory_id,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Memory '{memory_id}' not found"}
                return result
            except Exception as e:
                logger.error(f"Error getting memory: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_memory(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_memory tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def delete_memory(
            memory_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a specific memory by ID.

            Permanently removes a memory unit and its associated data.

            Args:
                memory_id: The ID of the memory to delete
                bank_id: Optional bank (accepted for consistency, not used in deletion).
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_memory_unit(
                    unit_id=memory_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "deleted", "memory_id": memory_id, **result}, default=str)
            except Exception as e:
                logger.error(f"Error deleting memory: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def delete_memory(
            memory_id: str,
        ) -> dict:
            """
            Delete a specific memory by ID.

            Permanently removes a memory unit and its associated data.

            Args:
                memory_id: The ID of the memory to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_memory_unit(
                    unit_id=memory_id,
                    request_context=_get_request_context(config),
                )
                return {"status": "deleted", "memory_id": memory_id, **result}
            except Exception as e:
                logger.error(f"Error deleting memory: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# DOCUMENT TOOLS
# =========================================================================


def _register_list_documents(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_documents tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_documents(
            q: str | None = None,
            limit: int = 100,
            bank_id: str | None = None,
        ) -> str:
            """
            List documents in a memory bank.

            Documents are containers for related memories (e.g., a conversation transcript,
            a meeting notes file). Memories created with a document_id are grouped under that document.

            Args:
                q: Optional search query to filter documents
                limit: Maximum number of results (default: 100)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_documents(
                    target_bank,
                    search_query=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing documents: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def list_documents(
            q: str | None = None,
            limit: int = 100,
        ) -> dict:
            """
            List documents in this memory bank.

            Documents are containers for related memories (e.g., a conversation transcript,
            a meeting notes file). Memories created with a document_id are grouped under that document.

            Args:
                q: Optional search query to filter documents
                limit: Maximum number of results (default: 100)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_documents(
                    target_bank,
                    search_query=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error listing documents: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_document(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_document tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def get_document(
            document_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get a specific document by ID.

            Returns document metadata and associated memory information.

            Args:
                document_id: The ID of the document to retrieve
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return json.dumps({"error": f"Document '{document_id}' not found"})
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error getting document: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def get_document(
            document_id: str,
        ) -> dict:
            """
            Get a specific document by ID.

            Returns document metadata and associated memory information.

            Args:
                document_id: The ID of the document to retrieve
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if result is None:
                    return {"error": f"Document '{document_id}' not found"}
                return result
            except Exception as e:
                logger.error(f"Error getting document: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_document(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_document tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def delete_document(
            document_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a document and its associated memories.

            Permanently removes a document and all memories linked to it.

            Args:
                document_id: The ID of the document to delete
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "deleted", "document_id": document_id, **result}, default=str)
            except Exception as e:
                logger.error(f"Error deleting document: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def delete_document(
            document_id: str,
        ) -> dict:
            """
            Delete a document and its associated memories.

            Permanently removes a document and all memories linked to it.

            Args:
                document_id: The ID of the document to delete
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_document(
                    document_id,
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return {"status": "deleted", "document_id": document_id, **result}
            except Exception as e:
                logger.error(f"Error deleting document: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# OPERATION TOOLS
# =========================================================================


def _register_list_operations(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_operations tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_operations(
            status: str | None = None,
            limit: int = 20,
            bank_id: str | None = None,
        ) -> str:
            """
            List async operations for a memory bank.

            Operations track background tasks like retain processing, mental model refresh, etc.

            Args:
                status: Filter by status: 'pending', 'running', 'completed', 'failed', 'cancelled'
                limit: Maximum number of results (default: 20)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_operations(
                    target_bank,
                    status=status,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing operations: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def list_operations(
            status: str | None = None,
            limit: int = 20,
        ) -> dict:
            """
            List async operations for this memory bank.

            Operations track background tasks like retain processing, mental model refresh, etc.

            Args:
                status: Filter by status: 'pending', 'running', 'completed', 'failed', 'cancelled'
                limit: Maximum number of results (default: 20)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_operations(
                    target_bank,
                    status=status,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error listing operations: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_operation(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_operation tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def get_operation(
            operation_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Get the status of an async operation.

            Check progress of background tasks like retain processing or mental model refresh.

            Args:
                operation_id: The ID of the operation to check
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.get_operation_status(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error getting operation: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def get_operation(
            operation_id: str,
        ) -> dict:
            """
            Get the status of an async operation.

            Check progress of background tasks like retain processing or mental model refresh.

            Args:
                operation_id: The ID of the operation to check
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.get_operation_status(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error getting operation: {e}", exc_info=True)
                return {"error": str(e)}


def _register_cancel_operation(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the cancel_operation tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def cancel_operation(
            operation_id: str,
            bank_id: str | None = None,
        ) -> str:
            """
            Cancel a pending or running async operation.

            Args:
                operation_id: The ID of the operation to cancel
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.cancel_operation(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error cancelling operation: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def cancel_operation(
            operation_id: str,
        ) -> dict:
            """
            Cancel a pending or running async operation.

            Args:
                operation_id: The ID of the operation to cancel
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.cancel_operation(
                    target_bank,
                    operation_id,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error cancelling operation: {e}", exc_info=True)
                return {"error": str(e)}


# =========================================================================
# TAGS & BANK TOOLS
# =========================================================================


def _register_list_tags(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the list_tags tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def list_tags(
            q: str | None = None,
            limit: int = 100,
            bank_id: str | None = None,
        ) -> str:
            """
            List tags used in a memory bank.

            Tags are used to organize and filter memories, directives, and mental models.

            Args:
                q: Optional pattern to filter tags (e.g., 'project:*')
                limit: Maximum number of results (default: 100)
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.list_tags(
                    target_bank,
                    pattern=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error listing tags: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def list_tags(
            q: str | None = None,
            limit: int = 100,
        ) -> dict:
            """
            List tags used in this memory bank.

            Tags are used to organize and filter memories, directives, and mental models.

            Args:
                q: Optional pattern to filter tags (e.g., 'project:*')
                limit: Maximum number of results (default: 100)
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.list_tags(
                    target_bank,
                    pattern=q,
                    limit=limit,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error listing tags: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def get_bank(
            bank_id: str | None = None,
        ) -> str:
            """
            Get the profile of a memory bank.

            Returns bank metadata including name, disposition, and mission.

            Args:
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                profile = await memory.get_bank_profile(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                    profile["disposition"] = profile["disposition"].model_dump()
                return json.dumps(profile, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error getting bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def get_bank() -> dict:
            """
            Get the profile of this memory bank.

            Returns bank metadata including name, disposition, and mission.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                profile = await memory.get_bank_profile(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                if "disposition" in profile and hasattr(profile["disposition"], "model_dump"):
                    profile["disposition"] = profile["disposition"].model_dump()
                return profile
            except Exception as e:
                logger.error(f"Error getting bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_get_bank_stats(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the get_bank_stats tool (multi-bank only)."""

    @mcp.tool()
    async def get_bank_stats(
        bank_id: str | None = None,
    ) -> str:
        """
        Get statistics for a memory bank.

        Returns counts of nodes, links, and other metrics.

        Args:
            bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
        """
        try:
            target_bank = bank_id or config.bank_id_resolver()
            if target_bank is None:
                return '{"error": "No bank_id configured"}'

            result = await memory.get_bank_stats(
                target_bank,
                request_context=_get_request_context(config),
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error getting bank stats: {e}", exc_info=True)
            return f'{{"error": "{e}"}}'


def _register_update_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the update_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def update_bank(
            name: str | None = None,
            mission: str | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Update a memory bank's metadata.

            Changes the name or mission of an existing bank.

            Args:
                name: New human-friendly name for the bank
                mission: New mission describing who the agent is and what they're trying to accomplish
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.update_bank(
                    target_bank,
                    name=name,
                    mission=mission,
                    request_context=_get_request_context(config),
                )
                return json.dumps(result, indent=2, default=str)
            except Exception as e:
                logger.error(f"Error updating bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def update_bank(
            name: str | None = None,
            mission: str | None = None,
        ) -> dict:
            """
            Update this memory bank's metadata.

            Changes the name or mission of the bank.

            Args:
                name: New human-friendly name for the bank
                mission: New mission describing who the agent is and what they're trying to accomplish
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.update_bank(
                    target_bank,
                    name=name,
                    mission=mission,
                    request_context=_get_request_context(config),
                )
                return result
            except Exception as e:
                logger.error(f"Error updating bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_delete_bank(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the delete_bank tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def delete_bank(
            bank_id: str | None = None,
        ) -> str:
            """
            Delete a memory bank and all its data.

            WARNING: This permanently deletes the bank and all its memories, documents,
            mental models, directives, and other data. This action cannot be undone.

            Args:
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_bank(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "deleted", "bank_id": target_bank, **result}, default=str)
            except Exception as e:
                logger.error(f"Error deleting bank: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def delete_bank() -> dict:
            """
            Delete this memory bank and all its data.

            WARNING: This permanently deletes the bank and all its memories, documents,
            mental models, directives, and other data. This action cannot be undone.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_bank(
                    target_bank,
                    request_context=_get_request_context(config),
                )
                return {"status": "deleted", "bank_id": target_bank, **result}
            except Exception as e:
                logger.error(f"Error deleting bank: {e}", exc_info=True)
                return {"error": str(e)}


def _register_clear_memories(mcp: FastMCP, memory: MemoryEngine, config: MCPToolsConfig) -> None:
    """Register the clear_memories tool."""

    if config.include_bank_id_param:

        @mcp.tool()
        async def clear_memories(
            type: str | None = None,
            bank_id: str | None = None,
        ) -> str:
            """
            Clear all memories from a bank without deleting the bank itself.

            Optionally filter by fact type to only clear specific kinds of memories.

            Args:
                type: Optional fact type filter: 'world', 'experience', or 'opinion'. If not specified, clears all.
                bank_id: Optional bank (defaults to session bank). Use for cross-bank operations.
            """
            try:
                target_bank = bank_id or config.bank_id_resolver()
                if target_bank is None:
                    return '{"error": "No bank_id configured"}'

                result = await memory.delete_bank(
                    target_bank,
                    fact_type=type,
                    request_context=_get_request_context(config),
                )
                return json.dumps({"status": "cleared", "bank_id": target_bank, **result}, default=str)
            except Exception as e:
                logger.error(f"Error clearing memories: {e}", exc_info=True)
                return f'{{"error": "{e}"}}'

    else:

        @mcp.tool()
        async def clear_memories(
            type: str | None = None,
        ) -> dict:
            """
            Clear all memories from this bank without deleting the bank itself.

            Optionally filter by fact type to only clear specific kinds of memories.

            Args:
                type: Optional fact type filter: 'world', 'experience', or 'opinion'. If not specified, clears all.
            """
            try:
                target_bank = config.bank_id_resolver()
                if target_bank is None:
                    return {"error": "No bank_id configured"}

                result = await memory.delete_bank(
                    target_bank,
                    fact_type=type,
                    request_context=_get_request_context(config),
                )
                return {"status": "cleared", "bank_id": target_bank, **result}
            except Exception as e:
                logger.error(f"Error clearing memories: {e}", exc_info=True)
                return {"error": str(e)}
