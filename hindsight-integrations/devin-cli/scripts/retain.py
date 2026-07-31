#!/usr/bin/env python3
"""Auto-retain hook for Stop event.

Port of the Claude Code plugin's retain.py, adapted for Devin CLI hooks.

This is the piece that was previously blocked: Devin CLI's Stop hook stdin
carries only `stop_hook_active` (plus the universal `session_id`/`prompt_id`)
— no `transcript_path`. Devin CLI does, however, persist every session to a
local SQLite database keyed by the same `session_id` (see
lib/devin_transcript.py for the read path and its caveats). That's what this
script reads instead of a transcript file.

Flow:
  1. Read hook input from stdin (session_id)
  2. Read the session's messages from Devin CLI's session database
  3. Apply chunked retention logic (retainEveryNTurns + overlap window)
  4. Resolve API URL (external, existing local, or auto-start daemon)
  5. Derive bank ID and ensure mission
  6. Format transcript (strip memory tags, filter roles)
  7. POST to Hindsight retain API (async)

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
    slice_last_turns_by_user_boundary,
)
from lib.daemon import get_api_url
from lib.devin_transcript import fold_tool_calls_into_content, read_session_messages
from lib.state import commit_retention, increment_turn_count, plan_retention


def _apply_tool_call_folding(messages: list, include_tool_calls: bool) -> list:
    """Fold Devin CLI's separate `tool_calls` field into content.py's block shape.

    Only needed when retainToolCalls is enabled — the default (False) never
    looks at message["tool_calls"], so plain string content passes through
    lib/content.py unmodified either way.
    """
    if not include_tool_calls:
        return messages
    folded = []
    for msg in messages:
        if msg.get("tool_calls"):
            folded.append({"role": msg.get("role"), "content": fold_tool_calls_into_content(msg)})
        else:
            folded.append(msg)
    return folded


def run_retain(hook_input: dict, force: bool = False) -> None:
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        return

    debug_log(config, f"Retain hook_input keys: {list(hook_input.keys())} force={force}")

    session_id = hook_input.get("session_id", "unknown")

    all_messages = read_session_messages(session_id)
    if not all_messages:
        debug_log(config, "No messages found for session, skipping retain")
        return

    debug_log(config, f"Read {len(all_messages)} messages from session db")

    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 1))
    messages_to_retain = all_messages
    document_id = session_id
    retention_progress = None

    if retain_every_n > 1 and not force:
        turn_count = increment_turn_count(session_id)
        if turn_count % retain_every_n != 0:
            next_at = ((turn_count // retain_every_n) + 1) * retain_every_n
            debug_log(config, f"Turn {turn_count}/{retain_every_n}, skipping retain (next at turn {next_at})")
            return

    # Document ID strategy: see the Claude Code plugin's retain.py for the
    # full rationale — this mirrors it exactly, just against session-db reads
    # instead of a transcript file.
    if retain_mode == "chunked" and retain_every_n > 1:
        # Clamped, not just type-checked. load_config() guarantees this is an
        # int, but not a sensible one, and the window it feeds is passed to
        # slice_last_turns_by_user_boundary(), which returns [] for turns <= 0.
        # A negative overlap therefore empties the transcript, and an empty
        # transcript is skipped — so retention silently stops, including on the
        # forced session-end retain. Overlap can only ever widen the window.
        overlap_turns = max(0, config.get("retainOverlapTurns", 0))
        window_turns = retain_every_n + overlap_turns
        messages_to_retain = slice_last_turns_by_user_boundary(all_messages, window_turns)
        retain_full_window = True
        document_id = f"{session_id}-{int(time.time() * 1000)}"
        debug_log(
            config,
            f"Chunked retain firing (window: {window_turns} turns, {len(messages_to_retain)} messages)",
        )
    else:
        retention_progress = plan_retention(session_id, len(all_messages))
        if retention_progress.start_index >= len(all_messages):
            debug_log(config, f"No new messages for session {session_id}, skipping retain")
            return
        messages_to_retain = all_messages[retention_progress.start_index :]
        # Always the full slice. The checkpoint has already narrowed this to
        # exactly the unretained messages, so asking the formatter to narrow it
        # again to "the last turn" discards the rest of them — and the
        # empty-transcript path below then commits the full count, so they are
        # never retained by any later run either. Two ways that bit:
        # a slice with no user message at all (an assistant-only tail) came back
        # empty, and a slice spanning more than one turn (after a retain failed
        # once) silently lost every turn but the last.
        retain_full_window = True
        if retention_progress.compacted:
            debug_log(
                config,
                f"Message count shrank for session {session_id}: retaining as chunk "
                f"{retention_progress.chunk_index} (compaction or /clear)",
            )
        document_id = (
            session_id if retention_progress.chunk_index == 0 else f"{session_id}-c{retention_progress.chunk_index}"
        )
        debug_log(
            config,
            f"Full session retain: {len(messages_to_retain)} new messages "
            f"from {len(all_messages)} total into chunk {retention_progress.chunk_index}",
        )

    retain_roles = config.get("retainRoles", ["user", "assistant"])
    include_tool_calls = config.get("retainToolCalls", False)
    messages_to_retain = _apply_tool_call_folding(messages_to_retain, include_tool_calls)
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, retain_roles, retain_full_window, include_tool_calls=include_tool_calls
    )

    if not transcript:
        if retention_progress is not None:
            commit_retention(session_id, len(all_messages), retention_progress)
        debug_log(config, "Empty transcript after formatting, skipping retain")
        return

    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(
            api_url,
            api_token,
            request_timeout_override=config.get("requestTimeoutSeconds"),
        )
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        return

    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

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
    if raw_tags:
        tags = []
        for original in raw_tags:
            resolved = _resolve_template(original)
            if ":" in resolved and resolved.split(":", 1)[1] == "":
                debug_log(config, f"Dropping tag '{original}' -> '{resolved}' (empty content after ':')")
                continue
            tags.append(resolved)
        if not tags:
            tags = None
    else:
        tags = None

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
            context=config.get("retainContext", "devin-cli"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        if retention_progress is not None:
            commit_retention(session_id, len(all_messages), retention_progress)
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)


def main():
    try:
        hook_input = json.load(sys.stdin)
    # UnicodeDecodeError is what invalid UTF-8 on stdin raises; it subclasses
    # ValueError, not either of the others, so it would escape as a traceback.
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return
    run_retain(hook_input, force=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] Unexpected error in retain: {e}", file=sys.stderr)
        # Always 0 — same contract, same reasoning as recall.py. Not flagged in
        # review, but it is the identical code path; fixing one and leaving the
        # other is how a contract drifts back apart.
        sys.exit(0)
