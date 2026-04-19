"""OpenAI Agents SDK tool definitions for Hindsight memory operations.

Provides factory functions that create OpenAI Agents SDK-compatible
``FunctionTool`` instances and dynamic instructions backed by Hindsight's
retain/recall/reflect APIs. Tools can be passed directly to
``Agent(tools=[...])`` and instructions to ``Agent(instructions=[...])``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agents import function_tool
from agents.tool import FunctionTool
from hindsight_client import Hindsight

from ._client import resolve_client
from .config import (
    DEFAULT_BUDGET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RECALL_TAGS_MATCH,
    Budget,
    TagsMatch,
    get_config,
)
from .errors import HindsightError

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
) -> list[FunctionTool]:
    """Create Hindsight memory tools for an OpenAI Agents SDK agent.

    Returns a list of ``FunctionTool`` instances compatible with OpenAI Agents
    SDK's ``Agent(tools=[...])``.

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
        List of OpenAI Agents SDK FunctionTool instances.

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

    tools: list[FunctionTool] = []

    if include_retain:

        @function_tool
        async def hindsight_retain(content: str) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or any information that should be remembered across conversations.

            Args:
                content: The information to store in memory.
            """
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                if retain_metadata:
                    retain_kwargs["metadata"] = retain_metadata
                if retain_document_id:
                    retain_kwargs["document_id"] = retain_document_id
                await resolved_client.aretain(**retain_kwargs)
                return "Memory stored successfully."
            except Exception as e:
                logger.error("Retain failed: %s", e)
                raise HindsightError(f"Retain failed: {e}") from e

        tools.append(hindsight_retain)

    if include_recall:

        @function_tool
        async def hindsight_recall(query: str) -> str:
            """Search long-term memory for relevant information.

            Use this to find previously stored facts, preferences, or context.
            Returns a numbered list of matching memories.

            Args:
                query: What to search for in memory.
            """
            try:
                recall_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
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
                    return "No relevant memories found."
                lines = []
                for i, result in enumerate(response.results, 1):
                    line = f"{i}. {result.text}"
                    if recall_include_entities and getattr(result, "entities", None):
                        entity_names = [e.name if hasattr(e, "name") else str(e) for e in result.entities]
                        line += f" [entities: {', '.join(entity_names)}]"
                    lines.append(line)
                return "\n".join(lines)
            except Exception as e:
                logger.error("Recall failed: %s", e)
                raise HindsightError(f"Recall failed: {e}") from e

        tools.append(hindsight_recall)

    if include_reflect:

        @function_tool
        async def hindsight_reflect(query: str) -> str:
            """Synthesize a thoughtful answer from long-term memories.

            Use this when you need a coherent summary or reasoned response
            about what you know, rather than raw memory facts.

            Args:
                query: The question to reflect on using stored memories.
            """
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                }
                if reflect_context:
                    reflect_kwargs["context"] = reflect_context
                effective_reflect_max = reflect_max_tokens if reflect_max_tokens is not None else effective_max_tokens
                if effective_reflect_max:
                    reflect_kwargs["max_tokens"] = effective_reflect_max
                if reflect_response_schema:
                    reflect_kwargs["response_schema"] = reflect_response_schema
                # Reflect tags: use reflect-specific or fall back to recall tags
                effective_reflect_tags = reflect_tags if reflect_tags is not None else effective_recall_tags
                effective_reflect_tags_match = (
                    reflect_tags_match if reflect_tags_match is not None else effective_recall_tags_match
                )
                if effective_reflect_tags:
                    reflect_kwargs["tags"] = effective_reflect_tags
                    reflect_kwargs["tags_match"] = effective_reflect_tags_match
                response = await resolved_client.areflect(**reflect_kwargs)
                if response.text is not None and response.text != "":
                    return response.text
                return "No relevant memories found."
            except Exception as e:
                logger.error("Reflect failed: %s", e)
                raise HindsightError(f"Reflect failed: {e}") from e

        tools.append(hindsight_reflect)

    return tools


def memory_instructions(
    *,
    bank_id: str,
    base_instructions: str = "",
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    query: str = "relevant context about the user",
    budget: Budget | None = None,
    max_results: int = 5,
    max_tokens: int | None = None,
    prefix: str = "\n\nRelevant memories:\n",
    tags: list[str] | None = None,
    tags_match: TagsMatch | None = None,
) -> Callable[..., Awaitable[str]]:
    """Create an instructions function that auto-injects relevant memories.

    Returns an async callable compatible with the OpenAI Agents SDK's
    ``Agent(instructions=...)`` parameter. On each agent run, it
    automatically recalls relevant memories from Hindsight and appends
    them to the base instructions — no explicit tool call needed.

    The OpenAI Agents SDK accepts ``str | Callable | None`` for instructions
    (not a list). This function returns a single callable that composes your
    static instructions with dynamically recalled memories.

    Args:
        bank_id: The Hindsight memory bank to recall from.
        base_instructions: Static instructions prepended before memories.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        query: The recall query to find relevant memories.
        budget: Recall budget level (low/mid/high).
        max_results: Maximum number of memories to include.
        max_tokens: Maximum tokens for recall results (falls back to global config).
        prefix: Text prepended before the memory list.
        tags: Tags to filter when searching memories.
        tags_match: Tag matching mode (any/all/any_strict/all_strict).

    Returns:
        An async callable suitable for ``Agent(instructions=...)``.

    Example::

        from hindsight_openai_agents import memory_instructions, create_hindsight_tools

        agent = Agent(
            name="assistant",
            instructions=memory_instructions(
                client=client,
                bank_id="user-123",
                base_instructions="You are a helpful assistant.",
            ),
            tools=create_hindsight_tools(client=client, bank_id="user-123"),
        )
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)

    config = get_config()
    effective_budget = budget if budget is not None else (config.budget if config else DEFAULT_BUDGET)
    effective_max_tokens = (
        max_tokens if max_tokens is not None else (config.max_tokens if config else DEFAULT_MAX_TOKENS)
    )
    effective_tags = tags if tags is not None else (config.recall_tags if config else None)
    effective_tags_match = (
        tags_match if tags_match is not None else (config.recall_tags_match if config else DEFAULT_RECALL_TAGS_MATCH)
    )

    async def _instructions(ctx: Any, agent: Any) -> str:
        """Recall memories and format as instructions text."""
        try:
            recall_kwargs: dict[str, Any] = {
                "bank_id": bank_id,
                "query": query,
                "budget": effective_budget,
                "max_tokens": effective_max_tokens,
            }
            if effective_tags:
                recall_kwargs["tags"] = effective_tags
                recall_kwargs["tags_match"] = effective_tags_match
            response = await resolved_client.arecall(**recall_kwargs)
            if not response.results:
                return base_instructions
            lines = []
            for i, result in enumerate(response.results[:max_results], 1):
                lines.append(f"{i}. {result.text}")
            return base_instructions + prefix + "\n".join(lines)
        except Exception as e:
            logger.error("memory_instructions recall failed: %s", e)
            return base_instructions

    return _instructions
