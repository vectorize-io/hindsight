#!/usr/bin/env python3
"""SessionEnd hook: final retain + daemon cleanup.

Fires once when a Devin CLI session terminates. Forces a final retain (so
short sessions — fewer turns than retainEveryNTurns — still land on disk),
then stops the auto-started daemon, if this plugin started one.

Port of the Claude Code plugin's session_end.py. That version only forced a
final retain when `transcript_path` was present on stdin; Devin CLI's
SessionEnd stdin never had one, so this version keys off `session_id` instead
(present on every event as of Devin CLI 3000.3.22) — see retain.py's module
docstring for how that reads the conversation.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.daemon import stop_daemon


def main():
    config = load_config()

    try:
        hook_input = json.load(sys.stdin)
    # UnicodeDecodeError is what invalid UTF-8 on stdin raises; it subclasses
    # ValueError, not either of the others, so it would escape as a traceback.
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError):
        hook_input = {}

    debug_log(config, f"SessionEnd hook, reason: {hook_input.get('reason', 'unknown')}")

    if config.get("autoRetain") and hook_input.get("session_id"):
        try:
            from retain import run_retain

            run_retain(hook_input, force=True)
        except Exception as e:
            print(f"[Hindsight] SessionEnd final retain error: {e}", file=sys.stderr)

    def _dbg(*a):
        debug_log(config, *a)

    # Passing the session id deregisters it; the daemon is shared, so it is only
    # stopped once no other session is still registered against it.
    stop_daemon(config, session_id=hook_input.get("session_id", ""), debug_fn=_dbg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionEnd error: {e}", file=sys.stderr)
        sys.exit(0)
