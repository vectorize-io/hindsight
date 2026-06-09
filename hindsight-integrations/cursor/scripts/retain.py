#!/usr/bin/env python3
"""Auto-retain hook for Cursor's stop event.

Flow:
  1. Read hook input from stdin (conversation_id, transcript_path, status)
  2. Read conversation transcript from transcript_path
  3. Apply retention logic (full-session or chunked with overlap)
  4. Resolve API URL (external, existing local, or auto-start daemon)
  5. Derive bank ID and ensure mission
  6. Format transcript (strip memory tags)
  7. POST to Hindsight retain API (async)

Exit codes:
  0 -- always (graceful degradation on any error)
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
from lib.state import increment_turn_count, write_state

LAST_RETAIN_STATE = "last_retain.json"


def _write_retain_status(status: str, **extra):
    """Write retain diagnostics on every invocation."""
    data = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "plugin",
        "status": status,
    }
    data.update(extra)
    try:
        write_state(LAST_RETAIN_STATE, data)
    except Exception:
        pass


def _normalize_blocks_to_text(content) -> str:
    """Flatten a content payload to a single text string.

    Cursor 3 emits content as a list of typed blocks; we keep text blocks and
    inline a compact tool_use marker so retain.py's downstream Answer:/Thought:
    handling still sees recognizable structure.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "tool")
            parts.append(f"[tool_use:{name}]")
        elif btype == "tool_result":
            parts.append("[tool_result]")
    return "\n".join(parts)


def read_transcript(transcript_path: str) -> list:
    """Read a JSONL transcript file and return list of message dicts.

    Supports three shapes seen in the wild:
      Flat:        {role: "user", content: "..."}
      Type-nested: {type: "user", message: {role: "user", content: "..."}}
      Role-nested: {role: "user", message: {content: [...blocks...]}}  (Cursor 3.x)

    The role-nested form is what Cursor 3.6.31 writes to
    ``~/.cursor/projects/<workspace>/agent-transcripts/<conv>/<conv>.jsonl``:
    a top-level ``role`` plus a ``message.content`` list of typed text/tool
    blocks. The earlier two-branch parser silently dropped every line of those
    transcripts because the top-level didn't have ``content`` and the
    ``type`` key wasn't present, so retain.py bailed with
    ``empty_transcript`` on every Cursor 3 stop hook.
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return []
    messages = []
    try:
        with open(transcript_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not isinstance(entry, dict):
                    continue

                # Type-nested: {type: "user", message: {role, content, ...}}
                if entry.get("type") in ("user", "assistant"):
                    msg = entry.get("message", {})
                    if isinstance(msg, dict) and msg.get("role"):
                        if "content" in msg:
                            msg = {**msg, "content": _normalize_blocks_to_text(msg.get("content"))}
                        messages.append(msg)
                    continue

                # Role-nested (Cursor 3.x):
                #   {role: "user", message: {content: [...blocks...]}}
                role = entry.get("role")
                if role in ("user", "assistant") and isinstance(entry.get("message"), dict):
                    msg_obj = entry["message"]
                    content = _normalize_blocks_to_text(msg_obj.get("content"))
                    messages.append({"role": role, "content": content})
                    continue

                # Flat: {role, content}
                if role in ("user", "assistant") and "content" in entry:
                    messages.append({"role": role, "content": _normalize_blocks_to_text(entry.get("content"))})
                    continue
    except OSError:
        pass
    return messages


def main():
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        _write_retain_status("skipped", reason="disabled")
        return

    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    debug_log(config, f"Stop hook input keys: {list(hook_input.keys())}")

    # Cursor stop hook provides conversation_id and transcript_path
    session_id = hook_input.get("conversation_id") or hook_input.get("session_id") or "unknown"
    transcript_path = hook_input.get("transcript_path", "")

    # Read full transcript
    all_messages = read_transcript(transcript_path)
    if not all_messages:
        debug_log(config, "No messages in transcript, skipping retain")
        _write_retain_status("skipped", reason="empty_transcript")
        return

    debug_log(config, f"Read {len(all_messages)} messages from transcript")

    # Retention mode: full session (default) or chunked
    retain_mode = config.get("retainMode", "full-session")
    retain_every_n = max(1, config.get("retainEveryNTurns", 10))
    retain_full_window = False
    messages_to_retain = all_messages

    if retain_every_n > 1:
        turn_count = increment_turn_count(session_id)
        if turn_count % retain_every_n != 0:
            next_at = ((turn_count // retain_every_n) + 1) * retain_every_n
            debug_log(config, f"Turn {turn_count}/{retain_every_n}, skipping retain (next at turn {next_at})")
            _write_retain_status("skipped", reason="turn_window", turn=turn_count, next_at=next_at)
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
    include_tool_calls = config.get("retainToolCalls", False)
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, ["user", "assistant"], retain_full_window, include_tool_calls=include_tool_calls
    )

    if not transcript:
        debug_log(config, "Empty transcript after formatting, skipping retain")
        _write_retain_status("skipped", reason="empty_after_format")
        return

    # Resolve API URL
    def _dbg(*a):
        debug_log(config, *a)

    try:
        api_url = get_api_url(config, debug_fn=_dbg, allow_daemon_start=True)
    except RuntimeError as e:
        print(f"[Hindsight] {e}", file=sys.stderr)
        _write_retain_status("error", reason=f"api_url: {e}"[:200])
        return

    api_token = config.get("hindsightApiToken")
    try:
        client = HindsightClient(api_url, api_token)
    except ValueError as e:
        print(f"[Hindsight] Invalid API URL: {e}", file=sys.stderr)
        _write_retain_status("error", reason=f"invalid_url: {e}"[:200])
        return

    # Derive bank ID and ensure mission
    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    # Document ID: unique per retain so successive retains within one session
    # accumulate across distinct documents instead of upserting a single one.
    # The previous design used document_id=session_id in full-session mode,
    # which silently dropped earlier turns whenever a multi-turn session
    # re-retained — the V2 audit's 5-turn driver surfaced this as the
    # "5-turn → 1 topic" failure mode. Timestamps are millisecond-grained
    # so back-to-back retains in the same wall-clock millisecond stay distinct
    # via session_id.
    document_id = f"{session_id}-{int(time.time() * 1000)}"

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

    # POST to Hindsight retain API
    try:
        response = client.retain(
            bank_id=bank_id,
            content=transcript,
            document_id=document_id,
            context=config.get("retainContext", "cursor"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
        _write_retain_status(
            "success",
            bank_id=bank_id,
            document_id=document_id,
            message_count=message_count,
            transcript_chars=len(transcript),
        )
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)
        _write_retain_status("error", reason=str(e)[:200], bank_id=bank_id)


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
