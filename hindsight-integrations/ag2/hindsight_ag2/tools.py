"""AG2 tool definitions for Hindsight memory operations.

Provides factory functions that create AG2-compatible tool functions
backed by Hindsight's retain/recall/reflect APIs. Tools are plain Python
functions with ``Annotated`` type hints, compatible with AG2's
``@register_for_llm`` / ``@register_for_execution`` pattern.
"""

import logging
from collections.abc import Callable
from typing import Annotated, Any, Optional

from hindsight_client import Hindsight

from ._client import resolve_client
from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Optional[Hindsight] = None,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    tags: Optional[list[str]] = None,
    recall_tags: Optional[list[str]] = None,
    recall_tags_match: Optional[str] = None,
    # Retain options
    retain_metadata: Optional[dict[str, str]] = None,
    retain_document_id: Optional[str] = None,
    # Recall options
    recall_types: Optional[list[str]] = None,
    recall_include_entities: bool = False,
    # Reflect options
    reflect_context: Optional[str] = None,
    reflect_max_tokens: Optional[int] = None,
    reflect_response_schema: Optional[dict[str, Any]] = None,
    reflect_tags: Optional[list[str]] = None,
    reflect_tags_match: Optional[str] = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[Callable]:
    """Create Hindsight memory tools for AG2 agents.

    Returns a list of plain Python functions compatible with AG2's
    ``@register_for_llm`` / ``@register_for_execution`` pattern.
    Each function uses ``Annotated`` type hints for parameter descriptions.

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
        List of callable tool functions.

    Raises:
        HindsightError: If no client or API URL can be resolved.

    Usage::

        tools = create_hindsight_tools(bank_id="my-bank", client=client)
        for tool_fn in tools:
            agent.register_for_llm(description=tool_fn.__doc__)(tool_fn)
            executor.register_for_execution()(tool_fn)
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)

    config = get_config()
    effective_tags = tags if tags is not None else (config.tags if config else None)
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = (
        recall_tags_match if recall_tags_match is not None else (config.recall_tags_match if config else "any")
    )
    effective_budget = budget if budget is not None else (config.budget if config else "mid")
    effective_max_tokens = max_tokens if max_tokens is not None else (config.max_tokens if config else 4096)

    tools: list[Callable] = []

    if include_retain:

        def hindsight_retain(
            content: Annotated[
                str,
                "The information to store in long-term memory. Include important facts, "
                "user preferences, decisions, or anything that should be remembered across conversations.",
            ],
        ) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or any information that should be remembered across conversations.
            """
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                if retain_metadata:
                    retain_kwargs["metadata"] = retain_metadata
                if retain_document_id:
                    retain_kwargs["document_id"] = retain_document_id
                resolved_client.retain(**retain_kwargs)
                return "Memory stored successfully."
            except Exception as e:
                logger.error("Retain failed: %s", e)
                raise HindsightError(f"Retain failed: {e}") from e

        tools.append(hindsight_retain)

    if include_recall:

        def hindsight_recall(
            query: Annotated[
                str,
                "The search query to find relevant memories. Be specific about what information you're looking for.",
            ],
        ) -> str:
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
                if recall_types:
                    recall_kwargs["types"] = recall_types
                if recall_include_entities:
                    recall_kwargs["include_entities"] = True
                response = resolved_client.recall(**recall_kwargs)
                if not response.results:
                    return "No relevant memories found."
                lines = []
                for i, result in enumerate(response.results, 1):
                    lines.append(f"{i}. {result.text}")
                return "\n".join(lines)
            except Exception as e:
                logger.error("Recall failed: %s", e)
                raise HindsightError(f"Recall failed: {e}") from e

        tools.append(hindsight_recall)

    if include_reflect:

        def hindsight_reflect(
            query: Annotated[
                str,
                "The question or topic to synthesize a thoughtful answer about from long-term memories.",
            ],
        ) -> str:
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
                if reflect_context:
                    reflect_kwargs["context"] = reflect_context
                effective_reflect_max = reflect_max_tokens or effective_max_tokens
                if effective_reflect_max:
                    reflect_kwargs["max_tokens"] = effective_reflect_max
                if reflect_response_schema:
                    reflect_kwargs["response_schema"] = reflect_response_schema
                # Reflect tags: use reflect-specific or fall back to recall tags
                effective_reflect_tags = reflect_tags if reflect_tags is not None else effective_recall_tags
                effective_reflect_tags_match = reflect_tags_match or effective_recall_tags_match
                if effective_reflect_tags:
                    reflect_kwargs["tags"] = effective_reflect_tags
                    reflect_kwargs["tags_match"] = effective_reflect_tags_match
                response = resolved_client.reflect(**reflect_kwargs)
                return response.text or "No relevant memories found."
            except Exception as e:
                logger.error("Reflect failed: %s", e)
                raise HindsightError(f"Reflect failed: {e}") from e

        tools.append(hindsight_reflect)

    return tools


def register_hindsight_tools(
    agent,
    executor,
    *,
    bank_id: str,
    **kwargs,
) -> list[Callable]:
    """Convenience: create tools AND register them on AG2 agents.

    Creates Hindsight memory tools and registers them on the given AG2
    agents using ``register_for_llm`` and ``register_for_execution``.

    Args:
        agent: AG2 agent to register tools for LLM calling.
        executor: AG2 agent to register tools for execution.
        bank_id: Hindsight memory bank ID.
        **kwargs: All other args passed to ``create_hindsight_tools()``.

    Returns:
        List of registered tool functions.

    Usage::

        tools = register_hindsight_tools(
            assistant, user_proxy,
            bank_id="my-bank",
            hindsight_api_url="http://localhost:8888",
        )
    """
    tools = create_hindsight_tools(bank_id=bank_id, **kwargs)
    for tool_fn in tools:
        agent.register_for_llm(description=tool_fn.__doc__)(tool_fn)
        executor.register_for_execution()(tool_fn)
    return tools
