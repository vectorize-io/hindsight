#!/usr/bin/env python3
"""SessionEnd hook: final retain flush + daemon cleanup.

Fires once when a Claude Code session terminates. Two responsibilities:
1. Flush any un-retained turns that were skipped by retainEveryNTurns throttle.
2. Stop the auto-started hindsight-embed daemon (if any).

Port of: Openclaw's service.stop() in index.js
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import prepare_retention_transcript
from lib.daemon import get_api_url, stop_daemon
from lib.state import get_turn_count


def _read_transcript(transcript_path: str) -> list:
    """Read a JSONL transcript file and return list of message dicts."""
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


def _flush_remaining_turns(config: dict, hook_input: dict) -> None:
    """Retain the full session transcript, bypassing retainEveryNTurns throttle.

    Called by SessionEnd to ensure turns between the last throttled retain
    and session end are not silently dropped.
    """
    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")

    retain_every_n = max(1, config.get("retainEveryNTurns", 1))

    # If retainEveryNTurns == 1, the Stop hook retains every turn — no flush needed.
    # Also skip if turn_count is already a multiple (last Stop hook already flushed).
    if retain_every_n <= 1:
        debug_log(config, "SessionEnd flush: retainEveryNTurns=1, Stop hook covers all turns — skipping")
        return
    turn_count = get_turn_count(session_id)
    if turn_count % retain_every_n == 0:
        debug_log(config, f"SessionEnd flush: turn {turn_count} is a multiple of {retain_every_n} — already retained by Stop hook")
        return

    debug_log(config, f"SessionEnd flush: turn {turn_count} not a multiple of {retain_every_n} — flushing remaining turns")

    all_messages = _read_transcript(transcript_path)
    if not all_messages:
        debug_log(config, "SessionEnd flush: no messages in transcript, skipping")
        return

    retain_roles = config.get("retainRoles", ["user", "assistant"])
    include_tool_calls = config.get("retainToolCalls", True)
    transcript, message_count = prepare_retention_transcript(
        all_messages, retain_roles, True, include_tool_calls=include_tool_calls
    )
    if not transcript:
        debug_log(config, "SessionEnd flush: empty transcript after formatting, skipping")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=False)
    except RuntimeError as e:
        print(f"[Hindsight] SessionEnd flush: cannot reach API — {e}", file=sys.stderr)
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(api_url, api_token)
    except ValueError as e:
        print(f"[Hindsight] SessionEnd flush: invalid API URL — {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    # Use session_id as document_id so this upserts the same document as prior
    # full-session retains (deduplicates; last write wins).
    document_id = session_id

    template_vars = {
        "session_id": session_id,
        "bank_id": bank_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_id": os.environ.get("HINDSIGHT_USER_ID", ""),
    }

    def _resolve_template(value: str) -> str:
        for k, v in template_vars.items():
            value = value.replace(f"{{{k}}}", v)
        return value

    raw_tags = config.get("retainTags", [])
    tags = None
    if raw_tags:
        resolved = []
        for original in raw_tags:
            tag = _resolve_template(original)
            if ":" in tag and tag.split(":", 1)[1] == "":
                continue
            resolved.append(tag)
        tags = resolved or None

    metadata = {
        "retained_at": template_vars["timestamp"],
        "message_count": str(message_count),
        "session_id": session_id,
        "flush_reason": "session_end",
    }
    for k, v in config.get("retainMetadata", {}).items():
        metadata[k] = _resolve_template(str(v))

    debug_log(config, f"SessionEnd flush: retaining bank '{bank_id}', {message_count} messages, {len(transcript)} chars")

    try:
        response = client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context=config.get("retainContext", "claude-code"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        debug_log(config, f"SessionEnd flush: retain response {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] SessionEnd flush failed: {e}", file=sys.stderr)


def main():
    config = load_config()

    # Consume stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    debug_log(config, f"SessionEnd hook, reason: {hook_input.get('reason', 'unknown')}")

    # Flush any turns not captured by the Stop-hook throttle before stopping daemon
    if config.get("autoRetain"):
        try:
            _flush_remaining_turns(config, hook_input)
        except Exception as e:
            print(f"[Hindsight] SessionEnd flush error: {e}", file=sys.stderr)

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
