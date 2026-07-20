#!/usr/bin/env python3
"""SessionStart hook for GitHub Copilot CLI.

Fires once when a new or resumed session begins
(`sessionStart` — see https://docs.github.com/en/copilot/reference/hooks-reference).
Emits `additionalContext` with relevant memories recalled from Hindsight.

Recall query:
  - Uses `initialPrompt` when the session started with a queued prompt.
  - Falls back to a generic project-context query derived from `cwd`
    otherwise (the common case for an interactive `copilot` session, which
    rarely has an `initialPrompt`).

This is the *only* recall point in this integration — Copilot CLI's
`userPromptSubmitted` and `preToolUse` hooks cannot inject
`additionalContext` (see README's "Limitations" section), so there is no
per-turn recall refresh after the session starts.

Exit codes:
  0 — always (graceful degradation on any error; a memory hook must never
      block a session from starting).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.daemon import get_api_url, prestart_daemon_background
from lib.hook_io import field as hook_field
from lib.recall import build_recall_context, fallback_query


def main():
    config = load_config()

    # Consume stdin regardless of config, so the CLI never sees a hung pipe.
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    session_id = hook_field(hook_input, "sessionId", default="unknown")
    cwd = hook_field(hook_input, "cwd", default="")
    initial_prompt = (hook_field(hook_input, "initialPrompt", default="") or "").strip()

    debug_log(config, f"SessionStart hook, session: {session_id}, source: {hook_field(hook_input, 'source')}")

    if not config.get("autoRecall") and not config.get("autoRetain"):
        debug_log(config, "Both autoRecall and autoRetain disabled, skipping session start")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
    except RuntimeError as e:
        debug_log(config, f"Hindsight not running, initiating background pre-start: {e}")
        prestart_daemon_background(config, debug_fn=_dbg)
        return

    try:
        client = HindsightClient(api_url, config.get("hindsightApiToken"))
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    debug_log(config, f"Hindsight server reachable at {api_url}")

    if not config.get("autoRecall"):
        debug_log(config, "Auto-recall disabled, skipping recall")
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    query = initial_prompt if initial_prompt else fallback_query(cwd, config)
    context_message = build_recall_context(client, bank_id, query, config, debug_fn=_dbg)
    if not context_message:
        return

    # sessionStart output schema: { additionalContext?: string }
    output = {"additionalContext": context_message}
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)
