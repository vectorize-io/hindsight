#!/usr/bin/env python3
"""PreCompact hook for Cursor CLI.

Fires before the agent's context window is compacted/summarized.
Recalls relevant memories and surfaces them via the optional
`user_message` field so the user sees what context is being preserved
through compaction.

Cursor's `preCompact` is documented as observational — the hook
output cannot influence the compaction itself. The actual mechanism
that lets memories survive trimming is the `beforeSubmitPrompt`
recall that runs after compaction. This hook is a useful
companion: it pre-warms the recall and gives the user a chance to
glance at the memories that will be re-injected.

Flow:
  1. Read hook input from stdin (transcript_path, message_count, ...)
  2. Build a recall query from the last user message + recent context
  3. Call Hindsight recall
  4. Emit the count of recalled memories in `user_message`
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    compose_recall_query,
    format_memories,
    read_transcript,
    truncate_recall_query,
)
from lib.daemon import get_api_url


def main():
    if sys.platform == "win32":
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    config = load_config()

    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, skipping preCompact")
        return

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"preCompact hook input keys: {list(hook_input.keys())}")

    transcript_path = hook_input.get("transcript_path", "")
    messages = read_transcript(transcript_path)

    last_user_msg = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    if not last_user_msg or len(last_user_msg.strip()) < 5:
        debug_log(config, "No usable last user message for preCompact recall")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
    except RuntimeError as e:
        debug_log(config, f"Skipping preCompact recall: {e}")
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(api_url, api_token)
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    recall_context_turns = config.get("recallContextTurns", 1)
    recall_max_query_chars = config.get("recallMaxQueryChars", 800)
    recall_roles = config.get("recallRoles", ["user", "assistant"])

    if recall_context_turns > 1 and messages:
        query = compose_recall_query(
            last_user_msg, messages, recall_context_turns, recall_roles
        )
    else:
        query = last_user_msg
    query = truncate_recall_query(query, last_user_msg, recall_max_query_chars)
    if len(query) > recall_max_query_chars:
        query = query[:recall_max_query_chars]
    query = query.encode("utf-8", errors="ignore").decode("utf-8")

    recall_timeout = config.get("recallTimeout", 10)
    try:
        response = client.recall(
            bank_id=bank_id,
            query=query,
            max_tokens=config.get("recallMaxTokens", 1024),
            budget=config.get("recallBudget", "mid"),
            types=config.get("recallTypes"),
            timeout=recall_timeout,
        )
    except Exception as e:
        print(f"[Hindsight] preCompact recall failed: {e}", file=sys.stderr)
        return

    results = response.get("results", [])
    if not results:
        debug_log(config, "No memories found during preCompact")
        return

    # preCompact is observational, but we can still surface a small note so
    # the user sees what context is being held in long-term memory.
    note = (
        f"Hindsight preserved {len(results)} relevant memor{'y' if len(results) == 1 else 'ies'} "
        f"through this compaction. They will be re-injected on the next prompt."
    )
    output = {
        "user_message": note,
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in pre_compact: {e}", file=sys.stderr)
        try:
            from lib.config import load_config
            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)
