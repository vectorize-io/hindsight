#!/usr/bin/env python3
"""Auto-retain hook for ZCode's `Stop` event.

Fires after the agent loop ends. ZCode's Stop payload carries the full assistant
reply in `responseText` plus an ephemeral `transcript_path` (an assistant-only
temp file that is deleted right after the hook). It does NOT carry the user
prompt, so we pair the assistant reply with the prompt the recall hook stashed
under `last_prompt_<session_id>.json` and retain that turn.

ZCode runs hooks inline (no async), so retain runs synchronously. We keep the
timeout tight and degrade gracefully — if retain fails we log to stderr and
exit 0 so the agent is never blocked.

Assistant text resolution, in order:
  1. hook_input["responseText"]         (full reply — preferred)
  2. transcript_path                     (parse ZCode {"message": {...}} lines)
  3. hook_input["responsePreview"]       (truncated fallback)

Flow:
  1. Read hook input from stdin (session_id/sessionId, responseText, transcript_path, ...)
  2. Resolve assistant text + stashed user prompt
  3. Build a [user, assistant] messages list for the turn
  4. Apply retainEveryNTurns gating
  5. Resolve API URL (external, existing local, or auto-start daemon)
  6. Derive bank ID and ensure mission
  7. Format transcript (strip memory tags, filter roles)
  8. POST to Hindsight retain API (distinct document_id per turn)

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
from lib.content import prepare_retention_transcript, read_transcript
from lib.daemon import get_api_url
from lib.state import increment_turn_count, read_state


def _resolve_assistant_text(hook_input: dict) -> str:
    """Resolve the assistant reply for this turn.

    Prefer `responseText`; fall back to parsing the ephemeral transcript for the
    last assistant message; final fallback is `responsePreview`.
    """
    response_text = (hook_input.get("responseText") or "").strip()
    if response_text:
        return response_text

    transcript_path = hook_input.get("transcript_path", "")
    messages = read_transcript(transcript_path, include_tool_calls=False)
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

    return (hook_input.get("responsePreview") or "").strip()


def run_retain(hook_input: dict) -> None:
    config = load_config()

    if not config.get("autoRetain"):
        debug_log(config, "Auto-retain disabled, exiting")
        return

    debug_log(config, f"Retain hook input keys: {list(hook_input.keys())}")

    # ZCode sends both "session_id" and "sessionId"; accept either, plus
    # conversation_id for alternative payload shapes.
    session_id = (
        hook_input.get("session_id") or hook_input.get("sessionId") or hook_input.get("conversation_id") or "unknown"
    )

    # Assemble the turn from reliable payload fields: the recall hook stashed the
    # user prompt, and the Stop payload carries the assistant reply.
    assistant_text = _resolve_assistant_text(hook_input)
    stashed = read_state(f"last_prompt_{session_id}.json", {}) or {}
    prompt = (stashed.get("prompt") or "").strip()

    if not assistant_text and not prompt:
        debug_log(config, "No assistant text and no stashed prompt, skipping retain")
        return

    messages_to_retain = []
    if prompt:
        messages_to_retain.append({"role": "user", "content": prompt})
    if assistant_text:
        messages_to_retain.append({"role": "assistant", "content": assistant_text})

    debug_log(config, f"Assembled turn: {len(messages_to_retain)} messages (prompt={bool(prompt)})")

    retain_every_n = max(1, config.get("retainEveryNTurns", 1))

    # Gate retain frequency: only every Nth turn is stored when configured.
    if retain_every_n > 1:
        turn_count = increment_turn_count(session_id)
        if turn_count % retain_every_n != 0:
            next_at = ((turn_count // retain_every_n) + 1) * retain_every_n
            debug_log(config, f"Turn {turn_count}/{retain_every_n}, skipping retain (next at turn {next_at})")
            return

    # Format transcript. Turns are plain text (no tool-call structure), so the
    # text transcript path is used.
    retain_roles = config.get("retainRoles", ["user", "assistant"])
    transcript, message_count = prepare_retention_transcript(
        messages_to_retain, retain_roles, retain_full_window=True, include_tool_calls=False
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

    # Derive bank ID and ensure mission
    bank_id = derive_bank_id(hook_input, config)
    ensure_bank_mission(client, bank_id, config, debug_fn=_dbg)

    # Distinct document_id per turn so each turn is its own memory and prior
    # turns are never overwritten.
    document_id = f"{session_id}-{int(time.time() * 1000)}"

    # Resolve template variables in tags and metadata
    template_vars = {
        "session_id": session_id,
        "conversation_id": session_id,
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
            context=config.get("retainContext", "zcode"),
            metadata=metadata,
            tags=tags,
            timeout=15,
        )
        debug_log(config, f"Retain response: {json.dumps(response)[:200]}")
    except Exception as e:
        print(f"[Hindsight] Retain failed: {e}", file=sys.stderr)


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        print("[Hindsight] Failed to read hook input", file=sys.stderr)
        return

    run_retain(hook_input)


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
