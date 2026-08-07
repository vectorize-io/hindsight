#!/usr/bin/env python3
"""SessionEnd hook: final retain + daemon cleanup, detached from the CLI.

Fires once when a Claude Code session terminates. Claude Code cancels
SessionEnd hooks that are still running when its shutdown teardown finishes
(reliably reproducible with Ctrl+C Ctrl+C exits in MCP-heavy projects:
"SessionEnd hook [...] failed: Hook cancelled" — see anthropics/claude-code
issues #32712 and #41577), which silently loses the forced final retain.

The hook entry therefore does no real work itself: it re-execs this script as
a detached child (own session, so it survives the CLI's process-group
teardown) and returns within milliseconds. The child performs the final
retain and stops the auto-managed daemon if we started one.

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

    if os.environ.get("HINDSIGHT_SESSION_END_DETACHED"):
        # Detached child: do the real work, immune to the parent's fate.
        try:
            hook_input = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
        except json.JSONDecodeError:
            hook_input = {}
        _do_session_end(config, hook_input)
        return

    # Hook entry: hand off to a detached child and return immediately, so the
    # CLI's shutdown cancellation window has nothing left to kill.
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"SessionEnd hook, reason: {hook_input.get('reason', 'unknown')}")

    import subprocess

    env = dict(os.environ)
    env["HINDSIGHT_SESSION_END_DETACHED"] = "1"
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), json.dumps(hook_input)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _do_session_end(config: dict, hook_input: dict) -> None:
    # Force a final retain before stopping the daemon — guarantees short sessions
    # (fewer turns than retainEveryNTurns) still land on disk.
    if config.get("autoRetain") and hook_input.get("transcript_path"):
        try:
            from retain import run_retain
            run_retain(hook_input, force=True)
        except Exception as e:
            print(f"[Hindsight] SessionEnd final retain error: {e}", file=sys.stderr)

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
