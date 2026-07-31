#!/usr/bin/env python3
"""SessionStart hook: health check + daemon pre-warm.

Fires once when a Devin CLI session begins. Verifies the Hindsight server is
reachable early, before the first prompt, and kicks off a background daemon
pre-start if it's not.

Port of the Claude Code plugin's session_start.py — unchanged logic, since
this hook doesn't need anything Devin CLI's sparser stdin is missing.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.daemon import get_api_url, prestart_daemon_background, register_session


def main():
    config = load_config()

    if not config.get("autoRecall") and not config.get("autoRetain"):
        debug_log(config, "Both autoRecall and autoRetain disabled, skipping session start")
        return

    try:
        hook_input = json.load(sys.stdin)
    # UnicodeDecodeError is what invalid UTF-8 on stdin raises; it subclasses
    # ValueError, not either of the others, so it would escape as a traceback.
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError):
        hook_input = {}

    debug_log(config, f"SessionStart hook, source: {hook_input.get('source', 'unknown')}")

    def _dbg(*a):
        debug_log(config, *a)

    session_id = hook_input.get("session_id", "")

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
        debug_log(config, f"Hindsight server reachable at {api_url}")
        # Already up — join the registry so this session counts towards keeping
        # the daemon alive until its own SessionEnd. No-op when the running
        # daemon isn't the plugin's.
        register_session(session_id, debug_fn=_dbg)
    except (RuntimeError, ValueError) as e:
        debug_log(config, f"Hindsight not running, initiating background pre-start: {e}")
        prestart_daemon_background(config, session_id=session_id, debug_fn=_dbg)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)
