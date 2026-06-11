#!/usr/bin/env python3
"""SessionStart hook: health check + session logging.

Fires once when a Claude Code session begins. Uses additionalContext
(supported on SessionStart) to inject an initial system note if
Hindsight is available.

This is the Claude Code equivalent of Openclaw's service.start() —
verify the server is reachable early, before the first prompt.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.daemon import get_api_url, prestart_daemon_background


def main():
    config = load_config()

    if not config.get("autoRecall") and not config.get("autoRetain"):
        debug_log(config, "Both autoRecall and autoRetain disabled, skipping session start")
        return

    # Consume stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"SessionStart hook, source: {hook_input.get('source', 'unknown')}")

    # Try to resolve API URL (health check). Don't start daemon here —
    # that's too slow for session start. Just check if server is reachable.
    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
        debug_log(config, f"Hindsight server reachable at {api_url}")
    except (RuntimeError, ValueError) as e:
        # Server not running — kick off background pre-start so it's ready
        # by the time the first recall or retain hook fires.
        debug_log(config, f"Hindsight not running, initiating background pre-start: {e}")
        prestart_daemon_background(config, debug_fn=_dbg)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)
