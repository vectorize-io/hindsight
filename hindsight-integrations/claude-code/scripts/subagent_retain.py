#!/usr/bin/env python3
"""Stop hook for subagent: retain conversation to the agent's bank.

Called from a subagent's Stop hook with HINDSIGHT_BANK_ID set in the
environment. Reads the transcript, formats it, and posts to the
agent-specific bank.

This is separate from the main plugin's retain.py which retains to
the global bank derived from config. This script always retains to
the bank specified in HINDSIGHT_BANK_ID.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import prepare_retention_transcript
from lib.daemon import get_api_url


def read_transcript(transcript_path: str) -> list:
    if not transcript_path or not os.path.isfile(transcript_path):
        return []
    messages = []
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") in ("user", "assistant"):
                        msg = entry.get("message", {})
                        if isinstance(msg, dict) and msg.get("role"):
                            messages.append(msg)
                    elif "role" in entry and "content" in entry:
                        messages.append(entry)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return messages


def main():
    bank_id = os.environ.get("HINDSIGHT_BANK_ID")
    if not bank_id:
        return

    config = load_config()

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    transcript_path = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "unknown")

    messages = read_transcript(transcript_path)
    if not messages:
        debug_log(config, "Subagent retain: no messages, skipping")
        return

    transcript, message_count = prepare_retention_transcript(
        messages,
        retain_roles=["user", "assistant"],
        retain_full_window=True,
        include_tool_calls=config.get("retainToolCalls", True),
    )

    if not transcript:
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] Subagent retain: {e}", file=sys.stderr)
        return

    try:
        client = HindsightClient(api_url, config.get("hindsightApiToken"))
    except ValueError as e:
        print(f"[Hindsight] Subagent retain: {e}", file=sys.stderr)
        return

    document_id = f"{session_id}-{int(time.time() * 1000)}"

    debug_log(
        config,
        f"Subagent retaining to bank '{bank_id}', doc '{document_id}', {message_count} messages",
    )

    try:
        client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context="claude-code-subagent",
            metadata={
                "retained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message_count": str(message_count),
                "session_id": session_id,
            },
        )
    except Exception as e:
        print(f"[Hindsight] Subagent retain failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Subagent retain error: {e}", file=sys.stderr)
        sys.exit(0)
