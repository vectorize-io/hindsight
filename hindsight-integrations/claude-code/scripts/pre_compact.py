#!/usr/bin/env python3
"""PreCompact hook: retain and mark the boundary before Claude Code compacts.

Claude Code keeps transcript JSONL files append-only. Compaction appends a
boundary plus summary and preserved tail; it does not shrink the file. This hook
forces a retain of the current pre-compact transcript, then records the
pre-compact message count only if that retain succeeds so the next retain writes
only the appended summary/tail/new messages into a fresh ``session_id-cN``
document.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.config import debug_log, load_config
from lib.state import mark_precompact
from retain import read_transcript, run_retain


def main() -> None:
    config = load_config()

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")
    trigger = hook_input.get("trigger", "")

    if not config.get("autoRetain"):
        debug_log(config, f"Auto-retain disabled; skipping PreCompact retain for session {session_id}")
        return

    message_count = len(read_transcript(transcript_path))
    retained = run_retain(hook_input, force=True)
    if not retained:
        debug_log(config, f"PreCompact retain did not complete for session {session_id}; checkpoint not marked")
        return

    if config.get("retainMode", "full-session") == "chunked":
        debug_log(config, f"PreCompact retained chunked session {session_id}; no full-session checkpoint needed")
        return

    chunk, start_index = mark_precompact(session_id, message_count)
    debug_log(
        config,
        f"PreCompact marked session {session_id}: chunk={chunk}, start_index={start_index}, trigger={trigger}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] PreCompact error: {e}", file=sys.stderr)
        sys.exit(0)
