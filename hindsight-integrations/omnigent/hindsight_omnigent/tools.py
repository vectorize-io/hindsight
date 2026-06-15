"""Hindsight memory tools for Omnigent agents.

Omnigent (https://github.com/omnigent-ai/omnigent) runs ``type: function`` tools
as plain, in-process Python callables resolved from a dotted import path. It
invokes them with the LLM's JSON arguments parsed into keyword args and coerces
the return value to a string (see ``omnigent/tools/local_callable.py``).

This module exposes Hindsight's retain / recall / reflect as exactly that shape:

    tools:
      hindsight_recall:
        type: function
        callable: hindsight_omnigent.tools.recall
        parameters: {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

Because Omnigent passes the callable **no session context**, the Hindsight bank
and connection come from :func:`hindsight_omnigent.configure` (or ``HINDSIGHT_*``
env vars) — one bank per agent process. Use :func:`tools_yaml` to emit a ready
``tools:`` block for an agent YAML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .config import get_config
from .errors import HindsightError

if TYPE_CHECKING:
    from hindsight_client import Hindsight

logger = logging.getLogger(__name__)

# Banks we've already ensured exist this process, so retain doesn't issue a
# redundant create_bank on every call. Keyed by bank id.
_created_banks: set[str] = set()


# ---------------------------------------------------------------------------
# Connection / bank resolution
# ---------------------------------------------------------------------------


def _client() -> Hindsight:
    """Resolve a Hindsight client from the global config."""
    config = get_config()
    if config.client is not None:
        return config.client

    from hindsight_client import Hindsight

    kwargs: dict[str, Any] = {"base_url": config.hindsight_api_url, "timeout": 30.0}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return Hindsight(**kwargs)


def _bank() -> str:
    """Resolve the configured Hindsight bank, or raise if none is set."""
    bank = get_config().bank_id
    if not bank:
        raise HindsightError(
            "No Hindsight bank configured. Call configure(bank_id=...) or set "
            "the HINDSIGHT_BANK_ID environment variable."
        )
    return bank


def _ensure_bank(client: Hindsight, bank: str) -> None:
    """Create the bank once per process; tolerate it already existing."""
    if bank in _created_banks:
        return
    try:
        client.create_bank(bank_id=bank, name=bank)
    except Exception as e:
        # Bank likely already exists; treat as created either way. Logged at
        # debug so a real auth/network failure is visible here rather than only
        # surfacing later on the retain call.
        logger.debug(f"create_bank({bank!r}) failed (assuming it exists): {e}")
    _created_banks.add(bank)


def _reset_created_banks() -> None:
    """Clear the per-process bank cache (used by tests)."""
    _created_banks.clear()


# ---------------------------------------------------------------------------
# Tool callables — referenced by dotted path from an Omnigent agent YAML
# ---------------------------------------------------------------------------


def retain(content: str) -> str:
    """Store information in long-term memory for later retrieval.

    Use this to save important facts, user preferences, decisions, or anything
    that should be remembered across conversations.
    """
    config = get_config()
    try:
        client = _client()
        bank = _bank()
        _ensure_bank(client, bank)
        kwargs: dict[str, Any] = {"bank_id": bank, "content": content}
        if config.tags:
            kwargs["tags"] = config.tags
        client.retain(**kwargs)
        return "Stored to long-term memory."
    except HindsightError:
        raise
    except Exception as e:
        logger.error(f"Retain failed: {e}")
        raise HindsightError(f"Retain failed: {e}") from e


def recall(query: str) -> str:
    """Search long-term memory for relevant information.

    Use this to find previously stored facts, preferences, or context. Returns
    the matching memories as a bullet list, or a note that none were found.
    """
    config = get_config()
    try:
        client = _client()
        bank = _bank()
        kwargs: dict[str, Any] = {
            "bank_id": bank,
            "query": query,
            "budget": config.budget,
            "max_tokens": config.max_tokens,
        }
        if config.recall_tags:
            kwargs["tags"] = config.recall_tags
            kwargs["tags_match"] = config.recall_tags_match
        response = client.recall(**kwargs)
        results = response.results or []
        memories = [r.text for r in results]
        if not memories:
            return "No relevant memories found."
        return "\n".join(f"- {m}" for m in memories)
    except HindsightError:
        raise
    except Exception as e:
        logger.error(f"Recall failed: {e}")
        raise HindsightError(f"Recall failed: {e}") from e


def reflect(query: str) -> str:
    """Synthesize a reasoned answer from long-term memories.

    Use this when you need a coherent summary or reasoned response about what
    you know, rather than raw memory facts.
    """
    config = get_config()
    try:
        client = _client()
        bank = _bank()
        response = client.reflect(bank_id=bank, query=query, budget=config.budget)
        return response.text or "No relevant memories found."
    except HindsightError:
        raise
    except Exception as e:
        logger.error(f"Reflect failed: {e}")
        raise HindsightError(f"Reflect failed: {e}") from e


# ---------------------------------------------------------------------------
# System-prompt pre-injection (Omnigent has no pre-turn context hook)
# ---------------------------------------------------------------------------


def memory_instructions(
    *,
    query: str = "relevant context about the user",
    bank_id: str | None = None,
    max_results: int = 5,
    prefix: str = "Relevant memories:\n",
) -> str:
    """Pre-recall memories for injection into an agent's ``prompt``.

    Omnigent agents have a static YAML ``prompt`` and no pre-turn context hook,
    so to seed an agent with what Hindsight already knows, call this and splice
    the result into the prompt yourself. Returns an empty string if no bank is
    configured or nothing is found — never raises.

    Args:
        query: The recall query used to find relevant memories.
        bank_id: Bank to recall from (defaults to the configured bank).
        max_results: Maximum number of memories to include.
        prefix: Text prepended before the memory list.
    """
    config = get_config()
    bank = bank_id or config.bank_id
    if not bank:
        return ""
    try:
        client = _client()
        kwargs: dict[str, Any] = {
            "bank_id": bank,
            "query": query,
            "budget": "low",
            "max_tokens": config.max_tokens,
        }
        if config.recall_tags:
            kwargs["tags"] = config.recall_tags
            kwargs["tags_match"] = config.recall_tags_match
        response = client.recall(**kwargs)
        results = response.results[:max_results] if response.results else []
        if not results:
            return ""
        lines = [prefix]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.text}")
        return "\n".join(lines)
    except Exception:
        # Silently return empty — instructions failures shouldn't block the agent.
        return ""


# ---------------------------------------------------------------------------
# Agent-YAML helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmnigentToolSpec:
    """A Hindsight tool declaration for an Omnigent agent YAML.

    Attributes:
        name: Tool name advertised to the LLM (the YAML key).
        callable_path: Dotted import path Omnigent resolves and calls.
        description: One-line description shown to the LLM.
        parameters: OpenAI/JSON-Schema parameter block for the tool.
    """

    name: str
    callable_path: str
    description: str
    parameters: dict[str, Any]


def _string_param(field: str, description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {field: {"type": "string", "description": description}},
        "required": [field],
    }


_RETAIN_SPEC = OmnigentToolSpec(
    name="hindsight_retain",
    callable_path="hindsight_omnigent.tools.retain",
    description="Store information in long-term memory for later retrieval.",
    parameters=_string_param("content", "The information to store in long-term memory."),
)
_RECALL_SPEC = OmnigentToolSpec(
    name="hindsight_recall",
    callable_path="hindsight_omnigent.tools.recall",
    description="Search long-term memory for relevant information.",
    parameters=_string_param("query", "The search query to find relevant memories."),
)
_REFLECT_SPEC = OmnigentToolSpec(
    name="hindsight_reflect",
    callable_path="hindsight_omnigent.tools.reflect",
    description="Synthesize a reasoned answer from long-term memories.",
    parameters=_string_param("query", "The question to reflect on using stored memories."),
)


def tool_specs(
    *,
    enable_retain: bool = True,
    enable_recall: bool = True,
    enable_reflect: bool = True,
) -> list[OmnigentToolSpec]:
    """Return the Hindsight tool specs to declare on an Omnigent agent.

    Args:
        enable_retain: Include the retain (store) tool.
        enable_recall: Include the recall (search) tool.
        enable_reflect: Include the reflect (synthesize) tool.
    """
    specs: list[OmnigentToolSpec] = []
    if enable_retain:
        specs.append(_RETAIN_SPEC)
    if enable_recall:
        specs.append(_RECALL_SPEC)
    if enable_reflect:
        specs.append(_REFLECT_SPEC)
    return specs


def tools_yaml(
    *,
    enable_retain: bool = True,
    enable_recall: bool = True,
    enable_reflect: bool = True,
    indent: int = 2,
) -> str:
    """Render a ``tools:`` block for an Omnigent agent YAML.

    Each tool's ``parameters`` is emitted as inline JSON (valid YAML flow style),
    so the schema round-trips exactly. Drop the result into your ``agent.yaml``.

    Args:
        enable_retain: Include the retain (store) tool.
        enable_recall: Include the recall (search) tool.
        enable_reflect: Include the reflect (synthesize) tool.
        indent: Spaces per indent level.
    """
    specs = tool_specs(
        enable_retain=enable_retain,
        enable_recall=enable_recall,
        enable_reflect=enable_reflect,
    )
    pad = " " * indent
    lines = ["tools:"]
    for spec in specs:
        lines.append(f"{pad}{spec.name}:")
        lines.append(f"{pad * 2}type: function")
        lines.append(f"{pad * 2}description: {spec.description}")
        lines.append(f"{pad * 2}callable: {spec.callable_path}")
        lines.append(f"{pad * 2}parameters: {json.dumps(spec.parameters)}")
    return "\n".join(lines) + "\n"
