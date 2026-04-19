"""Strands Agents tool factory for Hindsight memory operations.

Provides a factory function that creates Strands-compatible tool functions
backed by Hindsight's retain/recall/reflect APIs. Tools are plain Python
functions decorated with ``@tool`` — bank_id and client are captured in
the closure at construction time.
"""

from __future__ import annotations

import concurrent.futures
import logging
from importlib import metadata
from typing import Any

try:
    _VERSION = metadata.version("hindsight-strands")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-strands/{_VERSION}"

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_in_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a callable in a dedicated thread with a clean event loop.

    Strands runs tools inside its own asyncio event loop. The hindsight client
    uses asyncio internally (including asyncio.timeout), which conflicts with
    an already-running loop. Running in a separate thread gives a fresh loop.
    """
    return _executor.submit(fn, *args, **kwargs).result()


from hindsight_client import Hindsight
from strands import tool

from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


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
    enable_retain: bool = True,
    enable_recall: bool = True,
    enable_reflect: bool = True,
) -> list:
    """Create Hindsight memory tools for a Strands agent.

    Returns a list of ``@tool``-decorated functions that can be passed
    directly to ``Agent(tools=...)``. Each function captures ``bank_id``
    and the Hindsight client in its closure — no context modification needed.

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
        enable_retain: Include the retain (store) tool.
        enable_recall: Include the recall (search) tool.
        enable_reflect: Include the reflect (synthesize) tool.

    Returns:
        A list of Strands tool functions.

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

    created_banks: set[str] = set()

    def _ensure_bank(bid: str) -> None:
        if bid in created_banks:
            return
        try:
            resolved_client.create_bank(bank_id=bid, name=bid)
            created_banks.add(bid)
        except Exception:
            created_banks.add(bid)

    tools = []

    if enable_retain:

        @tool
        def hindsight_retain(content: str) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or any information that should be remembered across conversations.
            """
            try:
                _ensure_bank(bank_id)
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                _run_in_thread(resolved_client.retain, **retain_kwargs)
                return "Memory stored successfully."
            except HindsightError:
                raise
            except Exception as e:
                logger.error(f"Retain failed: {e}")
                raise HindsightError(f"Retain failed: {e}") from e

        tools.append(hindsight_retain)

    if enable_recall:

        @tool
        def hindsight_recall(query: str) -> str:
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
                response = _run_in_thread(resolved_client.recall, **recall_kwargs)
                if not response.results:
                    return "No relevant memories found."
                lines = []
                for i, result in enumerate(response.results, 1):
                    lines.append(f"{i}. {result.text}")
                return "\n".join(lines)
            except HindsightError:
                raise
            except Exception as e:
                logger.error(f"Recall failed: {e}")
                raise HindsightError(f"Recall failed: {e}") from e

        tools.append(hindsight_recall)

    if enable_reflect:

        @tool
        def hindsight_reflect(query: str) -> str:
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
                response = _run_in_thread(resolved_client.reflect, **reflect_kwargs)
                return response.text or "No relevant memories found."
            except HindsightError:
                raise
            except Exception as e:
                logger.error(f"Reflect failed: {e}")
                raise HindsightError(f"Reflect failed: {e}") from e

        tools.append(hindsight_reflect)

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
) -> str:
    """Pre-recall memories for injection into agent system prompt.

    Performs a sync recall and returns a formatted string of memories.
    Pass the result to ``Agent(system_prompt=...)`` or prepend it to
    your system prompt string.

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
        A formatted string of memories, or empty string if none found.

    Raises:
        HindsightError: If no client or API URL can be resolved.
    """
    resolved_client = _resolve_client(client, hindsight_api_url, api_key)

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
        response = _run_in_thread(resolved_client.recall, **recall_kwargs)
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
