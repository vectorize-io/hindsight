#!/usr/bin/env python3
"""Detached entrypoint for SessionStart daemon warmup.

Invoked by prestart_daemon_background() as a fresh Python process so the
codex hook that fired it can exit immediately. Takes a single argv: the
JSON-encoded plugin config. Silent by design — failures are recorded in
~/.hindsight/codex/state/daemon-start.log by background_start_daemon().
"""

import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        return 2

    try:
        config = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        return 2

    # Resolve lib/ relative to this file so the helper works regardless of
    # how Python was launched (cwd, -m, absolute path, etc.).
    scripts_dir = os.path.abspath(os.path.dirname(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from lib.daemon import background_start_daemon

    return background_start_daemon(config)


if __name__ == "__main__":
    sys.exit(main())
