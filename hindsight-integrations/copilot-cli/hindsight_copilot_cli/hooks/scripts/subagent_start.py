#!/usr/bin/env python3
"""subagentStart hook for GitHub Copilot CLI.

Fires when a subagent (`explore`, `task`, `research`, `code-review`,
`rubber-duck`, `security-review`, or a custom agent) is spawned — but *not*
for the built-in `general-purpose` agent, which never emits this event
(see https://docs.github.com/en/copilot/reference/hooks-reference).

Subagents run in their own isolated context and get none of the memory
injected at the parent session's `sessionStart` — this hook is the only way
to give them any persistent-memory continuity at all.

The payload never carries the specific task/prompt text given to the
subagent instance (only the static `agentName`/`agentDisplayName`/
`agentDescription`), so recall always uses the generic cwd-derived fallback
query — the same one `session_start.py` falls back to when there's no
`initialPrompt`.

Output `additionalContext` is prepended to the subagent's prompt. It cannot
block subagent creation, and this hook must never do so either.

Exit codes:
  0 — always (graceful degradation on any error).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.daemon import get_api_url
from lib.hook_io import field as hook_field
from lib.recall import build_recall_context, fallback_query


def main():
    config = load_config()

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    agent_name = hook_field(hook_input, "agentName", default="unknown")
    cwd = hook_field(hook_input, "cwd", default="")

    debug_log(config, f"subagentStart hook, agent: {agent_name}")

    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, skipping subagentStart recall")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
    except RuntimeError as e:
        debug_log(config, f"Hindsight not reachable, skipping subagent recall: {e}")
        return

    try:
        client = HindsightClient(api_url, config.get("hindsightApiToken"))
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    query = fallback_query(cwd, config)
    context_message = build_recall_context(client, bank_id, query, config, debug_fn=_dbg)
    if not context_message:
        return

    # subagentStart output schema: { additionalContext?: string }
    output = {"additionalContext": context_message}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] subagentStart error: {e}", file=sys.stderr)
        sys.exit(0)
