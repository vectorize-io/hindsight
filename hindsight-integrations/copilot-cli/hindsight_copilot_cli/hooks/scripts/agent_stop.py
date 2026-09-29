#!/usr/bin/env python3
"""agentStop hook for GitHub Copilot CLI.

Fires each time the main agent finishes a turn (`agentStop` — see
https://docs.github.com/en/copilot/reference/hooks-reference). Reads the
session transcript from `transcriptPath` and retains it into Hindsight.

Also caches `transcriptPath` (and `cwd`) for the session, since
`sessionEnd`'s own payload has no `transcriptPath` field — `session_end.py`
reuses this cache to force a final retain when the session terminates.

Exit codes:
  0 — always (graceful degradation on any error). `agentStop` output can
      force another turn via `{"decision": "block", ...}`, but this
      integration never uses that — a memory hook should never insert
      itself into the conversation loop.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import prepare_retention_transcript, read_transcript, slice_last_turns_by_user_boundary
from lib.daemon import get_api_url
from lib.hook_io import field as hook_field
from lib.state import cache_session_transcript, increment_turn_count


def run_retain(hook_input: dict, force: bool = False) -> None:
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        return

    debug_log(config, f"Retain hook input keys: {list(hook_input.keys())} force={force}")

    session_id = hook_field(hook_input, "sessionId", default="unknown")
    transcript_path = hook_field(hook_input, "transcriptPath", default="")

    include_tool_calls = config.get("retainToolCalls", True)
    all_messages = read_transcript(
        transcript_path,
        include_tool_calls=include_tool_calls,
        include_tools=config.get("includeTools", False),
    )
    if not all_messages:
        debug_log(config, "No messages in transcript, skipping retain")
        return

    debug_log(config, f"Read {len(all_messages)} messages from transcript")

    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 1))
    retain_full_window = False
    messages_to_retain = all_messages

    # Respect retainEveryNTurns in both modes, unless force=True (sessionEnd final retain).
    if retain_every_n > 1 and not force:
        turn_count = increment_turn_count(session_id)
        if turn_count % retain_every_n != 0:
            next_at = ((turn_count // retain_every_n) + 1) * retain_every_n
            debug_log(config, f"Turn {turn_count}/{retain_every_n}, skipping retain (next at turn {next_at})")
            return

    if retain_mode == "chunked" and retain_every_n > 1:
        overlap_turns = config.get("retainOverlapTurns", 0)
        window_turns = retain_every_n + overlap_turns
        messages_to_retain = slice_last_turns_by_user_boundary(all_messages, window_turns)
        retain_full_window = True
        debug_log(
            config,
            f"Chunked retain firing (window: {window_turns} turns, {len(messages_to_retain)} messages)",
        )
    else:
        retain_full_window = True
        debug_log(config, f"Full session retain: {len(all_messages)} messages")

    retain_roles = config.get("retainRoles", ["user", "assistant"])
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, retain_roles, retain_full_window, include_tool_calls=include_tool_calls
    )

    if not transcript:
        debug_log(config, "Empty transcript after formatting, skipping retain")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        return

    try:
        client = HindsightClient(api_url, config.get("hindsightApiToken"))
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    if retain_mode == "chunked" and retain_every_n > 1:
        document_id = f"{session_id}-{int(time.time() * 1000)}"
    else:
        document_id = session_id

    template_vars = {
        "session_id": session_id,
        "bank_id": bank_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def _resolve_template(value: str) -> str:
        for k, v in template_vars.items():
            value = value.replace(f"{{{k}}}", v)
        return value

    raw_tags = config.get("retainTags", [])
    tags = [_resolve_template(t) for t in raw_tags] if raw_tags else None

    metadata = {
        "retained_at": template_vars["timestamp"],
        "message_count": str(message_count),
        "session_id": session_id,
    }
    for k, v in config.get("retainMetadata", {}).items():
        metadata[k] = _resolve_template(str(v))

    debug_log(
        config, f"Retaining to bank '{bank_id}', doc '{document_id}', {message_count} messages, {len(transcript)} chars"
    )
    if tags:
        debug_log(config, f"Tags: {tags}")

    try:
        response = client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context=config.get("retainContext", "copilot-cli"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    session_id = hook_field(hook_input, "sessionId", default="unknown")
    transcript_path = hook_field(hook_input, "transcriptPath", default="")
    cwd = hook_field(hook_input, "cwd", default="")

    # sessionEnd's payload has no transcriptPath — cache it here so
    # session_end.py can still force a final retain.
    if transcript_path:
        cache_session_transcript(session_id, transcript_path, cwd)

    run_retain(hook_input, force=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in agentStop: {e}", file=sys.stderr)
        try:
            from lib.config import load_config

            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)
