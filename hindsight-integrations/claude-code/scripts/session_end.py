#!/usr/bin/env python3
"""SessionEnd hook: daemon cleanup.

Fires once when a Claude Code session terminates. If the plugin
auto-started a hindsight-embed daemon, this is where we stop it.

Port of: Openclaw's service.stop() in index.js
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.daemon import stop_daemon


def main():
    config = load_config()

    # Consume stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"SessionEnd hook, reason: {hook_input.get('reason', 'unknown')}")

    # Stop daemon if we started it
    def _dbg(*a):
        debug_log(config, *a)

    stop_daemon(config, debug_fn=_dbg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionEnd error: {e}", file=sys.stderr)
        sys.exit(0)
