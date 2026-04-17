"""Hermes tool definitions and registration for Hindsight memory operations.

Provides retain/recall/reflect as native Hermes tools via the plugin system
or manual ``register_tools()`` call.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

from hindsight_client import Hindsight

from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)

_TOOL_INSTRUCTIONS = """\
You have access to long-term memory via Hindsight tools.

- Use `hindsight_retain` to save important facts, user preferences, decisions, \
or any information that should be remembered across conversations.
- Use `hindsight_recall` to search for previously stored facts, preferences, or context.
- Use `hindsight_reflect` to synthesize a thoughtful, reasoned answer from \
what you know, rather than raw memory facts.

Proactively store information the user shares that may be useful later. \
When answering questions, check memory first for relevant context.\
"""

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory for later retrieval. "
        "Use this to save important facts, user preferences, decisions, "
        "or any information that should be remembered across conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to store in memory.",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory for relevant information. "
        "Use this to find previously stored facts, preferences, or context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant memories.",
            },
        },
        "required": ["query"],
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a thoughtful answer from long-term memories. "
        "Use this when you need a coherent summary or reasoned response "
        "about what you know, rather than raw memory facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question to reflect on using stored memories.",
            },
        },
        "required": ["query"],
    },
}


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
            "No Hindsight API URL configured. "
            "Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


def _resolve_bank_id(
    args: dict[str, Any],
    bank_id: str | None,
    bank_resolver: Callable[[dict[str, Any]], str] | None,
) -> str:
    """Resolve the effective bank_id for an operation.

    Resolution order:
    1. bank_resolver(args) if set
    2. Static bank_id if set
    3. HINDSIGHT_BANK_ID env var
    4. Raise HindsightError
    """
    if bank_resolver is not None:
        return bank_resolver(args)

    if bank_id is not None:
        return bank_id

    env_bank = os.environ.get("HINDSIGHT_BANK_ID")
    if env_bank:
        return env_bank

    raise HindsightError(
        "No bank_id available. Provide bank_id=, bank_resolver=, "
        "or set the HINDSIGHT_BANK_ID environment variable."
    )


def register_tools(
    *,
    bank_id: str | None = None,
    bank_resolver: Callable[[dict[str, Any]], str] | None = None,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: str = "any",
    toolset: str = "hindsight",
) -> None:
    """Register Hindsight memory tools into the Hermes tool registry.

    This imports ``tools.registry`` lazily so that hermes-agent is not a
    hard dependency of the package.

    Args:
        bank_id: Static memory bank ID.
        bank_resolver: Callable that resolves bank_id from tool args dict.
        client: Pre-configured Hindsight client.
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        toolset: Hermes toolset name for grouping.
    """
    from tools.registry import registry  # type: ignore[import-untyped]

    resolved_client = _resolve_client(client, hindsight_api_url, api_key)
    created_banks: set[str] = set()

    async def _ensure_bank(bid: str) -> None:
        if bid in created_banks:
            return
        try:
            await resolved_client.acreate_bank(bank_id=bid, name=bid)
            created_banks.add(bid)
        except Exception:
            created_banks.add(bid)

    async def handle_retain(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, bank_resolver)
            await _ensure_bank(bid)
            retain_kwargs: dict[str, Any] = {"bank_id": bid, "content": args["content"]}
            if tags:
                retain_kwargs["tags"] = tags
            await resolved_client.aretain(**retain_kwargs)
            return json.dumps({"result": "Memory stored successfully."})
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            return json.dumps({"error": str(e)})

    async def handle_recall(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, bank_resolver)
            recall_kwargs: dict[str, Any] = {
                "bank_id": bid,
                "query": args["query"],
                "budget": budget,
                "max_tokens": max_tokens,
            }
            if recall_tags:
                recall_kwargs["tags"] = recall_tags
                recall_kwargs["tags_match"] = recall_tags_match
            response = await resolved_client.arecall(**recall_kwargs)
            if not response.results:
                return json.dumps({"result": "No relevant memories found."})
            lines = []
            for i, result in enumerate(response.results, 1):
                lines.append(f"{i}. {result.text}")
            return json.dumps({"result": "\n".join(lines)})
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return json.dumps({"error": str(e)})

    async def handle_reflect(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, bank_resolver)
            reflect_kwargs: dict[str, Any] = {
                "bank_id": bid,
                "query": args["query"],
                "budget": budget,
            }
            response = await resolved_client.areflect(**reflect_kwargs)
            return json.dumps(
                {"result": response.text or "No relevant memories found."}
            )
        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            return json.dumps({"error": str(e)})

    registry.register(
        name="hindsight_retain",
        toolset=toolset,
        schema=RETAIN_SCHEMA,
        handler=handle_retain,
        is_async=True,
    )
    registry.register(
        name="hindsight_recall",
        toolset=toolset,
        schema=RECALL_SCHEMA,
        handler=handle_recall,
        is_async=True,
    )
    registry.register(
        name="hindsight_reflect",
        toolset=toolset,
        schema=REFLECT_SCHEMA,
        handler=handle_reflect,
        is_async=True,
    )


def register(ctx: Any) -> None:
    """Hermes plugin entry point — called via ``hermes_agent.plugins`` entry point.

    Reads configuration from environment variables and registers tools
    using ``ctx.register_tool()``.

    Args:
        ctx: Hermes PluginContext.
    """
    hindsight_api_url = os.environ.get("HINDSIGHT_API_URL")
    api_key = os.environ.get("HINDSIGHT_API_KEY")
    bank_id = os.environ.get("HINDSIGHT_BANK_ID")
    budget = os.environ.get("HINDSIGHT_BUDGET", "mid")

    if not hindsight_api_url and not api_key:
        logger.debug(
            "Hindsight plugin: no API URL or key configured, skipping registration"
        )
        return

    resolved_client = _resolve_client(None, hindsight_api_url, api_key)
    created_banks: set[str] = set()

    async def _ensure_bank(bid: str) -> None:
        if bid in created_banks:
            return
        try:
            await resolved_client.acreate_bank(bank_id=bid, name=bid)
            created_banks.add(bid)
        except Exception:
            created_banks.add(bid)

    async def handle_retain(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, None)
            await _ensure_bank(bid)
            await resolved_client.aretain(bank_id=bid, content=args["content"])
            return json.dumps({"result": "Memory stored successfully."})
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            return json.dumps({"error": str(e)})

    async def handle_recall(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, None)
            response = await resolved_client.arecall(
                bank_id=bid, query=args["query"], budget=budget
            )
            if not response.results:
                return json.dumps({"result": "No relevant memories found."})
            lines = [f"{i}. {r.text}" for i, r in enumerate(response.results, 1)]
            return json.dumps({"result": "\n".join(lines)})
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return json.dumps({"error": str(e)})

    async def handle_reflect(args: dict[str, Any], **kwargs: Any) -> str:
        try:
            bid = _resolve_bank_id(args, bank_id, None)
            response = await resolved_client.areflect(
                bank_id=bid, query=args["query"], budget=budget
            )
            return json.dumps(
                {"result": response.text or "No relevant memories found."}
            )
        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            return json.dumps({"error": str(e)})

    ctx.register_tool(
        name="hindsight_retain",
        toolset="hindsight",
        schema=RETAIN_SCHEMA,
        handler=handle_retain,
        is_async=True,
    )
    ctx.register_tool(
        name="hindsight_recall",
        toolset="hindsight",
        schema=RECALL_SCHEMA,
        handler=handle_recall,
        is_async=True,
    )
    ctx.register_tool(
        name="hindsight_reflect",
        toolset="hindsight",
        schema=REFLECT_SCHEMA,
        handler=handle_reflect,
        is_async=True,
    )

    # ── Lifecycle hooks ──────────────────────────────────────────────────
    # These require hermes-agent ≥ the version that invokes pre/post_llm_call.
    # When running on an older hermes-agent the hooks are simply never called,
    # so registering them is always safe.

    recall_budget = os.environ.get("HINDSIGHT_RECALL_BUDGET", budget)
    recall_max_tokens = int(os.environ.get("HINDSIGHT_RECALL_MAX_TOKENS", "4096"))
    retain_enabled = os.environ.get("HINDSIGHT_AUTO_RETAIN", "true").lower() in {"1", "true", "yes", "on"}

    async def _on_pre_llm_call(
        *,
        session_id: str = "",
        user_message: str = "",
        conversation_history: list | None = None,
        is_first_turn: bool = False,
        model: str = "",
        **kwargs: Any,
    ) -> dict[str, str] | None:
        """Recall relevant memories and inject them as system prompt context."""
        if not user_message or not bank_id:
            return None
        try:
            await _ensure_bank(bank_id)
            response = await resolved_client.arecall(
                bank_id=bank_id,
                query=user_message,
                budget=recall_budget,
                max_tokens=recall_max_tokens,
            )
            if not response.results:
                return None
            lines = [f"- {r.text}" for r in response.results]
            context = (
                "# Hindsight Memory (persistent cross-session context)\n"
                "Use this to answer questions about the user and prior sessions. "
                "Do not call tools to look up information that is already present here.\n\n"
                + "\n".join(lines)
            )
            return {"context": context}
        except Exception as exc:
            logger.warning("Hindsight pre_llm_call recall failed: %s", exc)
            return None

    async def _on_post_llm_call(
        *,
        session_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        model: str = "",
        **kwargs: Any,
    ) -> None:
        """Retain the conversation turn so it can be recalled in future sessions."""
        if not retain_enabled or not bank_id:
            return
        if not user_message or not assistant_response:
            return
        try:
            await _ensure_bank(bank_id)
            content = f"User: {user_message}\nAssistant: {assistant_response}"
            await resolved_client.aretain(bank_id=bank_id, content=content)
        except Exception as exc:
            logger.warning("Hindsight post_llm_call retain failed: %s", exc)

    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)


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
    """Pre-recall memories for injection into system prompt.

    Performs a sync recall and returns a formatted string of memories.
    Silently returns empty string on failure so it never blocks the agent.

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
    """
    try:
        resolved_client = _resolve_client(client, hindsight_api_url, api_key)
    except HindsightError:
        return ""

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
        response = resolved_client.recall(**recall_kwargs)
        results = response.results[:max_results] if response.results else []
        if not results:
            return ""
        lines = [prefix]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.text}")
        return "\n".join(lines)
    except Exception:
        return ""


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return tool schema dicts without registering them.

    Useful for inspection or manual integration without importing Hermes.

    Returns:
        List of OpenAI function-calling format schema dicts.
    """
    return [RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA]
