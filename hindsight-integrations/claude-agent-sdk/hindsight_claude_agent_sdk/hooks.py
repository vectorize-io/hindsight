"""Claude Agent SDK hooks for automatic Hindsight memory operations.

Provides hook factories that auto-inject recalled memories into prompts
and auto-retain agent results, without the agent needing to explicitly
call memory tools.

Hooks complement the explicit tools from ``create_hindsight_server`` —
use tools when the agent should decide when to remember/recall, and
hooks when you want memory to happen automatically.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    HookContext,
    HookJSONOutput,
    HookMatcher,
    PostToolUseHookInput,
    StopHookInput,
    UserPromptSubmitHookInput,
)
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

logger = logging.getLogger(__name__)


def _extract_result_from_transcript(transcript_path: str) -> str:
    """Extract the final result text from a Claude session transcript."""
    path = Path(transcript_path)
    if not path.exists():
        return ""

    fallback_assistant_text = ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if entry.get("type") == "result":
            result_text = entry.get("result")
            if isinstance(result_text, str) and result_text.strip():
                return result_text.strip()

        if not fallback_assistant_text and entry.get("type") == "assistant":
            content = entry.get("message", {}).get("content", [])
            if isinstance(content, list):
                text_blocks = [
                    block.get("text", "").strip()
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                fallback_assistant_text = "\n".join(block for block in text_blocks if block)

    return fallback_assistant_text


@dataclass
class MemoryHookConfig:
    """Configuration for automatic memory hooks.

    Attributes:
        auto_recall: Inject relevant memories into the system prompt
            on each ``UserPromptSubmit`` event.
        auto_retain: Store agent results on ``Stop`` events.
        retain_on_tools: Tool name regex patterns whose results should
            be auto-retained (e.g. ``"Bash"`` to remember command outputs).
        recall_query: Query used for auto-recall. If ``"$prompt"``,
            uses the user's actual prompt text.
        recall_max_results: Maximum memories to inject.
        recall_prefix: Text prepended before the memory list in the
            system prompt injection.
        retain_tags: Tags applied to auto-retained memories.
        retain_prefix: Text prepended to auto-retained content.
    """

    auto_recall: bool = True
    auto_retain: bool = True
    retain_on_tools: list[str] = field(default_factory=list)
    recall_query: str = "$prompt"
    recall_max_results: int = 5
    recall_prefix: str = "\n\nRelevant memories from previous sessions:\n"
    retain_tags: list[str] = field(default_factory=lambda: ["source:claude-agent-sdk"])
    retain_prefix: str = "Agent session result: "


def create_memory_hooks(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch | None = None,
    hook_config: MemoryHookConfig | None = None,
) -> dict[str, list[HookMatcher]]:
    """Create Claude Agent SDK hooks for automatic memory operations.

    Returns a dict suitable for ``ClaudeAgentOptions(hooks=...)``.

    Up to three hook types are created:

    - **UserPromptSubmit**: Before each agent turn, recalls relevant
      memories from Hindsight and injects them as a system message.
    - **Stop**: When the agent finishes, retains a summary of the
      result into Hindsight for future sessions.
    - **PostToolUse** (optional): Retains notable tool results based
      on ``hook_config.retain_on_tools`` patterns.

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        recall_tags: Tags to filter when recalling memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        hook_config: Fine-grained hook behavior configuration.

    Returns:
        A dict mapping hook event names to lists of HookMatcher,
        for use with ``ClaudeAgentOptions(hooks=...)``.

    Raises:
        HindsightError: If no client or API URL can be resolved.

    Example::

        from hindsight_claude_agent_sdk import create_memory_hooks
        from claude_agent_sdk import query, ClaudeAgentOptions

        hooks = create_memory_hooks(
            bank_id="my-agent",
            hindsight_api_url="http://localhost:8888",
        )

        async for msg in query(
            prompt="What did we decide about the auth module?",
            options=ClaudeAgentOptions(hooks=hooks),
        ):
            print(msg)
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)
    cfg = hook_config or MemoryHookConfig()

    config = get_config()
    effective_budget = budget if budget is not None else (config.budget if config else DEFAULT_BUDGET)
    effective_max_tokens = (
        max_tokens if max_tokens is not None else (config.max_tokens if config else DEFAULT_MAX_TOKENS)
    )
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = (
        recall_tags_match
        if recall_tags_match is not None
        else (config.recall_tags_match if config else DEFAULT_RECALL_TAGS_MATCH)
    )

    hooks: dict[str, list[HookMatcher]] = {}

    if cfg.auto_recall:

        async def _recall_hook(
            input_data: UserPromptSubmitHookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> HookJSONOutput:
            """Recall relevant memories and inject as system context."""
            try:
                query_text = input_data.get("prompt", "")
                if not query_text:
                    return {}

                if cfg.recall_query != "$prompt":
                    query_text = cfg.recall_query

                recall_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query_text,
                    "budget": effective_budget,
                    "max_tokens": effective_max_tokens,
                }
                if effective_recall_tags:
                    recall_kwargs["tags"] = effective_recall_tags
                    recall_kwargs["tags_match"] = effective_recall_tags_match

                response = await resolved_client.arecall(**recall_kwargs)
                if not response.results:
                    return {}

                lines = []
                for i, result in enumerate(response.results[: cfg.recall_max_results], 1):
                    lines.append(f"{i}. {result.text}")

                memory_text = cfg.recall_prefix + "\n".join(lines)
                return {"systemMessage": memory_text}

            except Exception as e:
                logger.warning("Auto-recall hook failed (non-fatal): %s", e)
                return {}

        hooks["UserPromptSubmit"] = [HookMatcher(hooks=[_recall_hook])]

    if cfg.auto_retain:

        async def _retain_hook(
            input_data: StopHookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> HookJSONOutput:
            """Retain agent result summary into memory."""
            try:
                result_text = _extract_result_from_transcript(input_data["transcript_path"])
                if not result_text or len(result_text) < 20:
                    return {}

                # Truncate very long results to avoid storing noise
                content = result_text[:4000] if len(result_text) > 4000 else result_text
                content = cfg.retain_prefix + content

                retain_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "content": content,
                }
                if cfg.retain_tags:
                    retain_kwargs["tags"] = cfg.retain_tags

                await resolved_client.aretain(**retain_kwargs)
                logger.debug("Auto-retained agent result to bank %s", bank_id)

            except Exception as e:
                logger.warning("Auto-retain hook failed (non-fatal): %s", e)

            return {}

        hooks["Stop"] = [HookMatcher(hooks=[_retain_hook])]

    if cfg.retain_on_tools:
        matcher_pattern = "|".join(re.escape(t) for t in cfg.retain_on_tools)

        async def _tool_retain_hook(
            input_data: PostToolUseHookInput,
            tool_use_id: str | None,
            context: HookContext,
        ) -> HookJSONOutput:
            """Retain notable tool results into memory."""
            try:
                tool_name = input_data.get("tool_name", "")
                tool_input = input_data.get("tool_input", {})
                tool_response = input_data.get("tool_response", "")

                response_str = str(tool_response)
                if not response_str or len(response_str) < 20:
                    return {}

                # Build a concise summary of the tool invocation
                input_summary = str(tool_input)[:500]
                response_summary = response_str[:2000]
                content = f"Tool {tool_name} called with: {input_summary}\nResult: {response_summary}"

                retain_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "content": content,
                    "tags": [*cfg.retain_tags, f"tool:{tool_name}"],
                }

                await resolved_client.aretain(**retain_kwargs)
                logger.debug("Auto-retained tool result from %s to bank %s", tool_name, bank_id)

            except Exception as e:
                logger.warning("Tool retain hook failed (non-fatal): %s", e)

            return {}

        hooks["PostToolUse"] = [HookMatcher(matcher=matcher_pattern, hooks=[_tool_retain_hook])]

    return hooks
