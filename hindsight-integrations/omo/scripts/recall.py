#!/usr/bin/env python3
"""Auto-recall hook for UserPromptSubmit.

Fires before each user prompt. Queries Hindsight for relevant memories
and injects them as additionalContext.

Exit codes:
  0 — always (graceful degradation on any error)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    compose_recall_query,
    format_current_time,
    format_memories,
    truncate_recall_query,
)
from lib.state import write_state


def read_transcript_messages(transcript_path: str) -> list:
    """Read messages from a JSONL transcript file for multi-turn context."""
    if not transcript_path or not os.path.isfile(transcript_path):
        return []
    messages = []
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") in ("user", "assistant"):
                        msg = entry.get("message", {})
                        if isinstance(msg, dict) and msg.get("role"):
                            messages.append(msg)
                    elif "role" in entry and "content" in entry:
                        messages.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return messages


def main():
    config = load_config()

    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, exiting")
        return

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    debug_log(config, f"Hook input keys: {list(hook_input.keys())}")

    prompt = (hook_input.get("prompt") or hook_input.get("user_prompt") or "").strip()
    if not prompt or len(prompt) < 5:
        debug_log(config, "Prompt too short for recall, skipping")
        return

    api_url = config.get("hindsightApiUrl")
    if not api_url:
        debug_log(config, "No hindsightApiUrl configured")
        return

    api_token = config.get("hindsightApiToken")
    if not api_token and "api.hindsight.vectorize.io" in api_url:
        return  # Cloud mode without token — silently skip

    try:
        client = HindsightClient(
            api_url,
            api_token,
            request_timeout_override=config.get("requestTimeoutSeconds"),
        )
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=lambda *a: debug_log(config, *a))

    recall_context_turns = config.get("recallContextTurns", 1)
    recall_max_query_chars = config.get("recallMaxQueryChars", 800)
    recall_roles = config.get("recallRoles", ["user", "assistant"])

    if recall_context_turns > 1:
        transcript_path = hook_input.get("transcript_path", "")
        messages = read_transcript_messages(transcript_path)
        query = compose_recall_query(prompt, messages, recall_context_turns, recall_roles)
    else:
        query = prompt

    query = truncate_recall_query(query, prompt, recall_max_query_chars)
    if len(query) > recall_max_query_chars:
        query = query[:recall_max_query_chars]

    debug_log(config, f"Recalling from bank '{bank_id}', query length: {len(query)}")

    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            max_tokens=config.get("recallMaxTokens", 1024),
            budget=config.get("recallBudget", "mid"),
            types=config.get("recallTypes"),
            timeout=10,
        )
    except Exception as e:
        print(f"[Hindsight] Recall failed: {e}", file=sys.stderr)
        return

    results = response.get("results", [])

    additional_banks = config.get("recallAdditionalBanks", [])
    for extra_bank_id in additional_banks:
        try:
            extra_response = client.recall(
                bank_id=extra_bank_id,
                query=query,
                max_tokens=config.get("recallMaxTokens", 1024),
                budget=config.get("recallBudget", "mid"),
                types=config.get("recallTypes"),
                timeout=10,
            )
            extra_results = extra_response.get("results", [])
            if extra_results:
                results = results + extra_results
        except Exception as e:
            debug_log(config, f"Recall from additional bank '{extra_bank_id}' failed: {e}")

    if not results:
        debug_log(config, "No memories found")
        return

    debug_log(config, f"Injecting {len(results)} memories")

    memories_formatted = format_memories(results)
    preamble = config.get("recallPromptPreamble", "")
    current_time = format_current_time()

    context_message = (
        f"<hindsight_memories>\n"
        f"{preamble}\n"
        f"Current time - {current_time}\n\n"
        f"{memories_formatted}\n"
        f"</hindsight_memories>"
    )

    write_state(
        "last_recall.json",
        {
            "context": context_message,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bank_id": bank_id,
            "result_count": len(results),
        },
    )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context_message,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in recall: {e}", file=sys.stderr)
        try:
            from lib.config import load_config

            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)
