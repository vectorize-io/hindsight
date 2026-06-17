#!/usr/bin/env python3
"""SessionStart hook for Cursor CLI.

Fires once when a Cursor composer conversation begins. Verifies the
Hindsight server is reachable, and kicks off a background daemon
pre-start if not — so it's ready by the first recall or retain hook.

Cursor's `sessionStart` is documented as fire-and-forget. The full
recall happens on `beforeSubmitPrompt`.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.client import HindsightClient
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

    debug_log(
        config,
        f"SessionStart hook, conversation: {hook_input.get('conversation_id', 'unknown')}",
    )

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
        HindsightClient(api_url, config.get("hindsightApiToken"))
        debug_log(config, f"Hindsight server reachable at {api_url}")
    except (RuntimeError, ValueError) as e:
        debug_log(config, f"Hindsight not running, initiating background pre-start: {e}")
        prestart_daemon_background(config, debug_fn=_dbg)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] SessionStart error: {e}", file=sys.stderr)
        sys.exit(0)
