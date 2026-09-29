#!/usr/bin/env python3
"""sessionEnd hook for GitHub Copilot CLI.

Fires once when a session terminates (`sessionEnd` — see
https://docs.github.com/en/copilot/reference/hooks-reference). Its payload
has no `transcriptPath`, unlike `agentStop`, so this hook forces a final
retain using the transcript path cached from the most recent `agentStop`
call for this session (see `lib/state.cache_session_transcript`).

This ensures short sessions still get retained even when
`retainEveryNTurns` would otherwise have skipped the per-turn retain.

Exit codes:
  0 — always (graceful degradation on any error). `sessionEnd` output is
      not processed by Copilot CLI regardless.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.hook_io import field as hook_field
from lib.state import clear_cached_session_transcript, get_cached_session_transcript


def main():
    config = load_config()

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    session_id = hook_field(hook_input, "sessionId", default="unknown")
    reason = hook_field(hook_input, "reason", default="unknown")
    debug_log(config, f"sessionEnd hook, session: {session_id}, reason: {reason}")

    try:
        if not config.get("autoRetain"):
            debug_log(config, "Auto-retain disabled, skipping final retain")
            return

        cached = get_cached_session_transcript(session_id)
        if not cached or not cached.get("transcript_path"):
            debug_log(config, "No cached transcript for this session, nothing to retain")
            return

        forced_input = dict(hook_input)
        forced_input["transcriptPath"] = cached["transcript_path"]
        if cached.get("cwd") and not forced_input.get("cwd"):
            forced_input["cwd"] = cached["cwd"]

        try:
            from agent_stop import run_retain

            run_retain(forced_input, force=True)
        except Exception as e:
            print(f"[Hindsight] sessionEnd final retain error: {e}", file=sys.stderr)
    finally:
        # Always clear the cache entry for this session, regardless of
        # whether autoRetain was disabled, no cache existed, or retain
        # failed — the entry is single-use and must not leak indefinitely.
        clear_cached_session_transcript(session_id)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] sessionEnd error: {e}", file=sys.stderr)
        sys.exit(0)
