"""Claude Agent SDK tool definitions for Hindsight memory operations.

Provides factory functions that create Hindsight retain/recall/reflect
tools as ``SdkMcpTool`` instances and an in-process MCP server for use
with ``ClaudeAgentOptions(mcp_servers={...})``.
"""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, ToolAnnotations, create_sdk_mcp_server, tool
from hindsight_client import Hindsight

from ._client import resolve_client
from ._version import __version__
from .config import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RECALL_TAGS_MATCH,
    Budget,
    TagsMatch,
    get_config,
)

logger = logging.getLogger(__name__)


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch | None = None,
    # Retain options
    retain_metadata: dict[str, str] | None = None,
    retain_document_id: str | None = None,
    # Recall options
    recall_types: list[str] | None = None,
    recall_include_entities: bool = False,
    # Reflect options
    reflect_context: str | None = None,
    reflect_max_tokens: int | None = None,
    reflect_response_schema: dict[str, Any] | None = None,
    reflect_tags: list[str] | None = None,
    reflect_tags_match: TagsMatch | None = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[SdkMcpTool]:
    """Create Hindsight memory tools as Claude Agent SDK ``SdkMcpTool`` instances.

    Returns a list of tools that can be wrapped into an MCP server via
    ``create_hindsight_server``, or used directly if needed.

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        retain_metadata: Default metadata dict for retain operations.
        retain_document_id: Default document_id for retain (groups/upserts memories).
        recall_types: Fact types to filter (world, experience, opinion, observation).
        recall_include_entities: Include entity information in recall results.
        reflect_context: Additional context for reflect operations.
        reflect_max_tokens: Max tokens for reflect results (defaults to max_tokens).
        reflect_response_schema: JSON schema to constrain reflect output format.
        reflect_tags: Tags to filter memories used in reflect (defaults to recall_tags).
        reflect_tags_match: Tag matching for reflect (defaults to recall_tags_match).
        include_retain: Include the retain (store) tool.
        include_recall: Include the recall (search) tool.
        include_reflect: Include the reflect (synthesize) tool.

    Returns:
        List of ``SdkMcpTool`` instances.

    Raises:
        HindsightError: If no client or API URL can be resolved.
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)

    config = get_config()
    effective_tags = tags if tags is not None else (config.tags if config else None)
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = (
        recall_tags_match
        if recall_tags_match is not None
        else (config.recall_tags_match if config else DEFAULT_RECALL_TAGS_MATCH)
    )
    effective_budget = budget if budget is not None else (config.budget if config else DEFAULT_BUDGET)
    effective_max_tokens = (
        max_tokens if max_tokens is not None else (config.max_tokens if config else DEFAULT_MAX_TOKENS)
    )

    tools: list[SdkMcpTool] = []

    if include_retain:

        @tool(
            "hindsight_retain",
            "Store information to long-term memory for later retrieval. "
            "Use this to save important facts, user preferences, decisions, "
            "or any information that should be remembered across conversations.",
            {"content": str},
            annotations=ToolAnnotations(
                readOnlyHint=False,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )
        async def hindsight_retain(args: dict[str, Any]) -> dict[str, Any]:
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": args["content"]}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                if retain_metadata:
                    retain_kwargs["metadata"] = retain_metadata
                if retain_document_id:
                    retain_kwargs["document_id"] = retain_document_id
                await resolved_client.aretain(**retain_kwargs)
                return {"content": [{"type": "text", "text": "Memory stored successfully."}]}
            except Exception as e:
                logger.error("Retain failed: %s", e)
                return {"content": [{"type": "text", "text": f"Retain failed: {e}"}], "is_error": True}

        tools.append(hindsight_retain)

    if include_recall:

        @tool(
            "hindsight_recall",
            "Search long-term memory for relevant information. "
            "Use this to find previously stored facts, preferences, or context. "
            "Returns a numbered list of matching memories.",
            {"query": str},
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=True,
                openWorldHint=True,
            ),
        )
        async def hindsight_recall(args: dict[str, Any]) -> dict[str, Any]:
            try:
                recall_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": args["query"],
                    "budget": effective_budget,
                    "max_tokens": effective_max_tokens,
                }
                if effective_recall_tags:
                    recall_kwargs["tags"] = effective_recall_tags
                    recall_kwargs["tags_match"] = effective_recall_tags_match
                if recall_types:
                    recall_kwargs["types"] = recall_types
                if recall_include_entities:
                    recall_kwargs["include_entities"] = True
                response = await resolved_client.arecall(**recall_kwargs)
                if not response.results:
                    return {"content": [{"type": "text", "text": "No relevant memories found."}]}
                lines = []
                for i, result in enumerate(response.results, 1):
                    line = f"{i}. {result.text}"
                    if recall_include_entities and getattr(result, "entities", None):
                        entity_names = [e.name if hasattr(e, "name") else str(e) for e in result.entities]
                        line += f" [entities: {', '.join(entity_names)}]"
                    lines.append(line)
                return {"content": [{"type": "text", "text": "\n".join(lines)}]}
            except Exception as e:
                logger.error("Recall failed: %s", e)
                return {"content": [{"type": "text", "text": f"Recall failed: {e}"}], "is_error": True}

        tools.append(hindsight_recall)

    if include_reflect:

        @tool(
            "hindsight_reflect",
            "Synthesize a thoughtful answer from long-term memories. "
            "Use this when you need a coherent summary or reasoned response "
            "about what you know, rather than raw memory facts.",
            {"query": str},
            annotations=ToolAnnotations(
                readOnlyHint=True,
                destructiveHint=False,
                idempotentHint=False,
                openWorldHint=True,
            ),
        )
        async def hindsight_reflect(args: dict[str, Any]) -> dict[str, Any]:
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": args["query"],
                    "budget": effective_budget,
                }
                if reflect_context:
                    reflect_kwargs["context"] = reflect_context
                effective_reflect_max = reflect_max_tokens if reflect_max_tokens is not None else effective_max_tokens
                if effective_reflect_max:
                    reflect_kwargs["max_tokens"] = effective_reflect_max
                if reflect_response_schema:
                    reflect_kwargs["response_schema"] = reflect_response_schema
                effective_reflect_tags = reflect_tags if reflect_tags is not None else effective_recall_tags
                effective_reflect_tags_match = (
                    reflect_tags_match if reflect_tags_match is not None else effective_recall_tags_match
                )
                if effective_reflect_tags:
                    reflect_kwargs["tags"] = effective_reflect_tags
                    reflect_kwargs["tags_match"] = effective_reflect_tags_match
                response = await resolved_client.areflect(**reflect_kwargs)
                if response.text is not None and response.text != "":
                    return {"content": [{"type": "text", "text": response.text}]}
                return {"content": [{"type": "text", "text": "No relevant memories found."}]}
            except Exception as e:
                logger.error("Reflect failed: %s", e)
                return {"content": [{"type": "text", "text": f"Reflect failed: {e}"}], "is_error": True}

        tools.append(hindsight_reflect)

    return tools


def create_hindsight_server(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch | None = None,
    retain_metadata: dict[str, str] | None = None,
    retain_document_id: str | None = None,
    recall_types: list[str] | None = None,
    recall_include_entities: bool = False,
    reflect_context: str | None = None,
    reflect_max_tokens: int | None = None,
    reflect_response_schema: dict[str, Any] | None = None,
    reflect_tags: list[str] | None = None,
    reflect_tags_match: TagsMatch | None = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> McpSdkServerConfig:
    """Create an in-process MCP server with Hindsight memory tools.

    Returns a server config dict for use with Claude Agent SDK's
    ``ClaudeAgentOptions(mcp_servers={"hindsight": server})``.

    All arguments are forwarded to ``create_hindsight_tools``. See its
    docstring for parameter descriptions.

    Example::

        from hindsight_claude_agent_sdk import create_hindsight_server
        from claude_agent_sdk import query, ClaudeAgentOptions

        server = create_hindsight_server(
            bank_id="my-agent",
            hindsight_api_url="http://localhost:8888",
        )

        async for msg in query(
            prompt="Check memory for past decisions, then help me plan.",
            options=ClaudeAgentOptions(
                mcp_servers={"hindsight": server},
                allowed_tools=["mcp__hindsight__*"],
            ),
        ):
            print(msg)
    """
    hindsight_tools = create_hindsight_tools(
        bank_id=bank_id,
        client=client,
        hindsight_api_url=hindsight_api_url,
        api_key=api_key,
        budget=budget,
        max_tokens=max_tokens,
        tags=tags,
        recall_tags=recall_tags,
        recall_tags_match=recall_tags_match,
        retain_metadata=retain_metadata,
        retain_document_id=retain_document_id,
        recall_types=recall_types,
        recall_include_entities=recall_include_entities,
        reflect_context=reflect_context,
        reflect_max_tokens=reflect_max_tokens,
        reflect_response_schema=reflect_response_schema,
        reflect_tags=reflect_tags,
        reflect_tags_match=reflect_tags_match,
        include_retain=include_retain,
        include_recall=include_recall,
        include_reflect=include_reflect,
    )

    return create_sdk_mcp_server(
        name="hindsight",
        version=__version__,
        tools=hindsight_tools,
    )
