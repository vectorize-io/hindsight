"""Pydantic AI tools for Hindsight memory operations.

Provides factory functions that create Pydantic AI ``Tool`` instances
backed by Hindsight's retain/recall/reflect APIs.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from importlib import metadata
from typing import Any

from hindsight_client import Hindsight
from pydantic_ai import RunContext, Tool

from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)

try:
    _VERSION = metadata.version("hindsight-pydantic-ai")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-pydantic-ai/{_VERSION}"


def _resolve_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config."""
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else None)
    key = api_key or (config.api_key if config else None)

    if url is None:
        raise HindsightError(
            "No Hindsight API URL configured. Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: str = "any",
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[Tool]:
    """Create Hindsight memory tools for a Pydantic AI agent.

    Returns a list of ``Tool`` instances that can be passed directly to
    ``Agent(tools=...)``. Each tool is an async closure that captures
    the Hindsight client — no ``RunContext`` or deps modification needed.

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
        include_retain: Include the retain (store) tool.
        include_recall: Include the recall (search) tool.
        include_reflect: Include the reflect (synthesize) tool.

    Returns:
        List of Pydantic AI Tool instances.

    Raises:
        HindsightError: If no client or API URL can be resolved.
    """
    resolved_client = _resolve_client(client, hindsight_api_url, api_key)

    # Resolve defaults from global config
    config = get_config()
    effective_tags = tags if tags is not None else (config.tags if config else None)
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = recall_tags_match or (config.recall_tags_match if config else "any")
    effective_budget = budget or (config.budget if config else "mid")
    effective_max_tokens = max_tokens or (config.max_tokens if config else 4096)

    tools: list[Tool] = []

    if include_retain:

        async def hindsight_retain(content: str) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or any information that should be remembered across conversations.
            """
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                await resolved_client.aretain(**retain_kwargs)
                return "Memory stored successfully."
            except Exception as e:
                logger.error(f"Retain failed: {e}")
                raise HindsightError(f"Retain failed: {e}") from e

        tools.append(Tool(hindsight_retain, takes_ctx=False))

    if include_recall:

        async def hindsight_recall(query: str) -> str:
            """Search long-term memory for relevant information.

            Use this to find previously stored facts, preferences, or context.
            Returns a numbered list of matching memories.
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
                response = await resolved_client.arecall(**recall_kwargs)
                if not response.results:
                    return "No relevant memories found."
                lines = []
                for i, result in enumerate(response.results, 1):
                    lines.append(f"{i}. {result.text}")
                return "\n".join(lines)
            except Exception as e:
                logger.error(f"Recall failed: {e}")
                raise HindsightError(f"Recall failed: {e}") from e

        tools.append(Tool(hindsight_recall, takes_ctx=False))

    if include_reflect:

        async def hindsight_reflect(query: str) -> str:
            """Synthesize a thoughtful answer from long-term memories.

            Use this when you need a coherent summary or reasoned response
            about what you know, rather than raw memory facts.
            """
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                }
                response = await resolved_client.areflect(**reflect_kwargs)
                return response.text or "No relevant memories found."
            except Exception as e:
                logger.error(f"Reflect failed: {e}")
                raise HindsightError(f"Reflect failed: {e}") from e

        tools.append(Tool(hindsight_reflect, takes_ctx=False))

    return tools


def memory_instructions(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    query: str = "relevant context about the user",
    budget: str = "low",
    max_results: int = 5,
    max_tokens: int = 4096,
    prefix: str = "Relevant memories:\n",
    tags: list[str] | None = None,
    tags_match: str = "any",
) -> Callable[[RunContext[Any]], Awaitable[str]]:
    """Create an instructions function that auto-injects relevant memories.

    Returns an async callable suitable for use with Pydantic AI's
    ``instructions`` parameter. Because instructions are re-evaluated
    on every run, memories stay fresh even when ``message_history``
    is reused.

    Args:
        bank_id: The Hindsight memory bank to recall from.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        query: The recall query to find relevant memories.
        budget: Recall budget level (low/mid/high).
        max_results: Maximum number of memories to include.
        max_tokens: Maximum tokens for recall results.
        prefix: Text prepended before the memory list.
        tags: Tags to filter recall results.
        tags_match: Tag matching mode (any/all/any_strict/all_strict).

    Returns:
        An async function compatible with ``Agent(instructions=[...])``.

    Raises:
        HindsightError: If no client or API URL can be resolved.
    """
    resolved_client = _resolve_client(client, hindsight_api_url, api_key)

    async def _instructions(ctx: RunContext[Any]) -> str:
        try:
            recall_kwargs: dict[str, Any] = {
                "bank_id": bank_id,
                "query": query,
                "budget": budget,
                "max_tokens": max_tokens,
            }
            if tags:
                recall_kwargs["tags"] = tags
                recall_kwargs["tags_match"] = tags_match
            response = await resolved_client.arecall(**recall_kwargs)
            results = response.results[:max_results] if response.results else []
            if not results:
                return ""
            lines = [prefix]
            for i, result in enumerate(results, 1):
                lines.append(f"{i}. {result.text}")
            return "\n".join(lines)
        except Exception:
            # Silently return empty — instructions failures shouldn't block the agent
            return ""

    return _instructions
