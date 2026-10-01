#!/usr/bin/env python3
"""Auto-retain hook for Stop event.

Fires after each agent turn. Reads the Codex session transcript and stores
the conversation into Hindsight memory for future recall.

Flow:
  1. Read hook input from stdin (session_id, transcript_path, cwd)
  2. Read conversation transcript from transcript_path
  3. Apply chunked retention logic (retainEveryNTurns + overlap window)
  4. Resolve API URL (external, existing local, or auto-start daemon)
  5. Derive bank ID and ensure mission
  6. Format transcript (strip memory tags, filter roles)
  7. POST to Hindsight retain API

Exit codes:
  0 — always (graceful degradation on any error)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id, ensure_bank_mission
from lib.client import HindsightClient
from lib.config import debug_log, load_config
from lib.content import (
    prepare_retention_transcript,
    read_transcript,
    slice_last_turns_by_user_boundary,
)
from lib.daemon import get_api_url
from lib.state import increment_turn_count

# This is source interpretation context, not another bank mission definition.
# Only explicit agent-session requests receive it. Other strategies and banks
# retain their configured policy, including collector-specific source safeguards.
AGENT_SESSION_CONTEXT = (
    "This source is a Codex agent-session transcript, not a verified operational ledger. "
    "Outer message roles identify the message sender, not necessarily the speaker in quoted or embedded material. "
    "Preserve explicit speaker labels and unknown or ambiguous speakers. Never infer identity from bank ownership, "
    "the surrounding message role, a recording owner, or first-person wording in a quotation. "
    "Keep lead-agent and subagent actions distinct. "
    "Assistant guidance is not the user's preference or accepted decision unless the user explicitly adopts it. "
    "Distinguish proposed, requested, open, merged, installed and deployed states. Preserve the final status "
    "supported by this source at its own time, including negations, uncertainty and rejected alternatives. "
    "When an outcome is supported only by an agent's claim, preserve it as agent-reported, not independently "
    "verified. A tool-call envelope, command name, filename, plan or generated summary do not prove success. "
    "Do not infer that a requested or planned action happened. Retention time is not event time. "
    "Use an event date only when the source supports it. Preserve the source language and legitimate quotations. "
    "Ignore routine progress narration, tool metadata and one-off debugging chatter without durable knowledge."
)


def main():
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        return

    retain_strategy = config.get("retainStrategy")
    if retain_strategy is not None and not isinstance(retain_strategy, str):
        print("[Hindsight] Invalid retainStrategy: expected a string or null, skipping retain", file=sys.stderr)
        return
    if retain_strategy is not None:
        retain_strategy = retain_strategy.strip() or None

    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    debug_log(config, f"Stop hook input keys: {list(hook_input.keys())}")

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")

    # Read full transcript
    include_tool_calls = config.get("retainToolCalls", True)
    all_messages = read_transcript(transcript_path, include_tool_calls=include_tool_calls)
    if not all_messages:
        debug_log(config, "No messages in transcript, skipping retain")
        return

    debug_log(config, f"Read {len(all_messages)} messages from transcript")

    # Retention mode: full session (default) or chunked (legacy)
    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 1))
    retain_full_window = False
    messages_to_retain = all_messages

    # Respect retainEveryNTurns in both modes
    if retain_every_n > 1:
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

    # Format transcript
    retain_roles = config.get("retainRoles", ["user", "assistant"])
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, retain_roles, retain_full_window, include_tool_calls=include_tool_calls
    )

    if not transcript:
        debug_log(config, "Empty transcript after formatting, skipping retain")
        return

    # Resolve API URL
    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(api_url, api_token)
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    # Derive bank ID and ensure legacy mission only if no strategy is selected.
    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    retain_context = config.get("retainContext", "codex")
    if retain_strategy == "agent-session":
        retain_context = f"{retain_context}\n\n{AGENT_SESSION_CONTEXT}" if retain_context else AGENT_SESSION_CONTEXT

    # Document ID: use session_id so the same session always upserts.
    # In chunked mode, append timestamp to create distinct documents per chunk.
    if retain_mode == "chunked" and retain_every_n > 1:
        document_id = f"{session_id}-{int(time.time() * 1000)}"
    else:
        document_id = session_id

    # Resolve template variables in tags and metadata
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

    # POST to Hindsight retain API
    try:
        response = client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context=retain_context,
            metadata=metadata,
            tags=tags,
            timeout=15,
            strategy=retain_strategy,
        )
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in retain: {e}", file=sys.stderr)
        try:
            from lib.config import load_config

            sys.exit(2 if load_config().get("debug") else 0)
        except Exception:
            sys.exit(0)
