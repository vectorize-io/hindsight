#!/usr/bin/env python3
"""Auto-recall hook for Cursor's beforeSubmitPrompt event.

Fires before every user prompt, retrieves relevant memories, injects them
through Cursor's native ``additional_context`` field, and refreshes the
workspace rules-file fallback for older Cursor versions.
"""

import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    compose_recall_query,
    format_memories,
    truncate_recall_query,
)
from lib.daemon import get_api_url
from lib.hook_io import read_hook_input
from lib.rules_file import (
    ensure_gitignored,
    format_rule_content,
    write_session_rules,
)
from lib.state import write_state

LAST_RECALL_STATE = "last_recall.json"


def _workspace_root(hook_input: Mapping[str, Any]) -> str:
    """Resolve the workspace root from Cursor's common hook fields."""
    project_dir = os.environ.get("CURSOR_PROJECT_DIR", "").strip()
    if project_dir:
        return project_dir

    roots = hook_input.get("workspace_roots")
    if isinstance(roots, list) and roots and isinstance(roots[0], str):
        return roots[0]

    return hook_input.get("cwd", "") or os.getcwd()


def _normalize_content(content: Any, include_tools: bool = False) -> str:
    """Flatten Cursor text blocks for multi-turn recall context."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif include_tools and block_type == "tool_use":
            parts.append(f"[tool_use:{block.get('name', 'tool')}]")
        elif include_tools and block_type == "tool_result":
            parts.append("[tool_result]")
    return "\n".join(parts)


def read_transcript_messages(transcript_path: str, include_tools: bool = False) -> list[Mapping[str, str]]:
    """Read user/assistant messages from Cursor's JSONL transcript formats."""
    if not transcript_path or not os.path.isfile(transcript_path):
        return []

    messages: list[Mapping[str, str]] = []
    try:
        with open(transcript_path, encoding="utf-8") as transcript_file:
            for line in transcript_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue

                role = entry.get("role")
                message = entry.get("message")
                if role in ("user", "assistant") and isinstance(message, dict):
                    content = _normalize_content(message.get("content"), include_tools)
                elif entry.get("type") in ("user", "assistant") and isinstance(message, dict):
                    role = entry["type"]
                    content = _normalize_content(message.get("content"), include_tools)
                elif role in ("user", "assistant") and "content" in entry:
                    content = _normalize_content(entry.get("content"), include_tools)
                else:
                    continue

                if content.strip():
                    messages.append({"role": role, "content": content.strip()})
    except OSError:
        pass
    return messages


def _write_recall_status(status: str, **extra) -> None:
    """Write recall diagnostics on every invocation."""
    data = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "plugin",
        "hook": "beforeSubmitPrompt",
        "status": status,
    }
    data.update(extra)
    try:
        write_state(LAST_RECALL_STATE, data)
    except Exception:
        pass


def filter_by_min_scores(
    results: list[Mapping[str, Any]],
    min_scores: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Drop recall results whose numeric scores are below configured floors."""
    if not min_scores:
        return results

    floors = {}
    for field, floor in min_scores.items():
        try:
            floors[field] = float(floor)
        except (TypeError, ValueError):
            debug_log(config, f"Ignoring invalid recallMinScores floor for '{field}': {floor!r}")

    if not floors:
        return results

    def passes_floors(result: Mapping[str, Any]) -> bool:
        scores = result.get("scores") or {}
        return all(
            not isinstance(scores.get(field), (int, float)) or scores[field] >= floor
            for field, floor in floors.items()
        )

    filtered = [result for result in results if passes_floors(result)]
    debug_log(config, f"Score floors dropped {len(results) - len(filtered)}/{len(results)} results")
    return filtered


def main() -> None:
    config = load_config()
    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, exiting")
        _write_recall_status("skipped", reason="disabled")
        return

    try:
        hook_input = read_hook_input()
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        _write_recall_status("error", reason="bad_stdin")
        return

    debug_log(config, f"Hook input keys: {list(hook_input.keys())}")
    workspace_root = _workspace_root(hook_input)

    prompt = (hook_input.get("prompt") or hook_input.get("user_prompt") or "").strip()
    if not prompt or len(prompt) < 5:
        debug_log(config, "Prompt too short for recall, skipping")
        _write_recall_status("skipped", reason="short_prompt")
        return

    def _dbg(*args):
        debug_log(config, *args)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
        client = HindsightClient(api_url, config.get("hindsightApiToken"))
    except (RuntimeError, ValueError) as error:
        print(f"[Hindsight] Recall setup failed: {error}", file=sys.stderr)
        _write_recall_status("error", reason=str(error)[:200])
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    recall_context_turns = config.get("recallContextTurns", 1)
    recall_max_query_chars = config.get("recallMaxQueryChars", 800)
    if recall_context_turns > 1:
        transcript_path = hook_input.get("transcript_path") or os.environ.get("CURSOR_TRANSCRIPT_PATH", "")
        messages = read_transcript_messages(
            transcript_path,
            include_tools=config.get("includeTools", False),
        )
        query = compose_recall_query(
            prompt,
            messages,
            recall_context_turns,
            config.get("recallRoles", ["user", "assistant"]),
        )
        debug_log(config, f"Multi-turn context: {recall_context_turns} turns, {len(messages)} messages")
    else:
        query = prompt

    query = truncate_recall_query(query, prompt, recall_max_query_chars)
    query = query.encode("utf-8", errors="ignore").decode("utf-8")
    recall_timeout = config.get("recallTimeout", 10)

    debug_log(config, f"Recalling from bank '{bank_id}', query length: {len(query)}, timeout: {recall_timeout}")
    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            max_tokens=config.get("recallMaxTokens", 1024),
            budget=config.get("recallBudget", "mid"),
            types=config.get("recallTypes"),
            timeout=recall_timeout,
        )
    except Exception as error:
        print(f"[Hindsight] Recall failed: {error}", file=sys.stderr)
        _write_recall_status("error", reason=str(error)[:200], bank_id=bank_id)
        return

    results = filter_by_min_scores(response.get("results", []), config.get("recallMinScores") or {}, config)
    if not results:
        debug_log(config, "No memories found")
        _write_recall_status("empty", bank_id=bank_id, query_length=len(query))
        return

    memories_formatted = format_memories(results)
    preamble = config.get("recallPromptPreamble", "")
    context_message = (
        f"<hindsight_memories>\n"
        f"{preamble}\n"
        f"\n"
        f"{memories_formatted}\n"
        f"</hindsight_memories>"
    )

    _write_recall_status(
        "success",
        bank_id=bank_id,
        result_count=len(results),
        query_length=len(query),
        context=context_message,
    )

    if workspace_root and config.get("useRulesFileFallback", True):
        rule_content = format_rule_content(memories_formatted, preamble)
        if write_session_rules(workspace_root, rule_content, debug_fn=lambda m: debug_log(config, m)):
            if config.get("appendToGitignore", True):
                ensure_gitignored(workspace_root, debug_fn=lambda m: debug_log(config, m))

    # Current Cursor builds consume additional_context from beforeSubmitPrompt
    # and inject it into the prompt's context. Keep the rules-file fallback for
    # older builds that silently ignore this field.
    json.dump({"continue": True, "additional_context": context_message}, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[Hindsight] Unexpected error in recall: {error}", file=sys.stderr)
        try:
            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)
