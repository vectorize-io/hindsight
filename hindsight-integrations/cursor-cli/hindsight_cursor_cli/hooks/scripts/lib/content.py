"""Content processing utilities for the Cursor CLI integration.

Adapts `hindsight-integrations/codex/scripts/lib/content.py` for Cursor's
on-disk transcript format.

Cursor transcript format (JSONL):
  Each line is an event from the agent's SDK stream. We care about:

  - {"type": "user",      "message": {"role": "user",      "content": [TextBlock...]}}
  - {"type": "assistant", "message": {"role": "assistant", "content": [TextBlock|ToolUseBlock...]}}
  - {"type": "system",    ...}                              (init metadata)
  - {"type": "thinking",  "text": "..."}                    (reasoning)
  - {"type": "tool_call", "name": "...", "args": ..., "result": ...}  (tool lifecycle)
  - {"type": "status",    "status": "..."}                  (lifecycle transitions)
  - {"type": "task",      ...}                              (task milestones)
  - {"type": "request",   ...}                              (awaiting user input)

  TextBlock is always `{"type": "text", "text": "..."}`. ToolUseBlock
  can vary; the docs warn that tool args/result shape is unstable.

For testing and future compatibility we also accept a flat shape:
  - {"role": "user", "content": "..."}
  - {"role": "user", "content": [{"type": "text", "text": "..."}]}
"""

import json
import os
import re
from datetime import datetime, timezone

_MAX_TOOL_OUTPUT_CHARS = 2000


# ---------------------------------------------------------------------------
# Memory tag stripping (anti-feedback-loop)
# ---------------------------------------------------------------------------


def strip_memory_tags(content):
    """Remove <hindsight_memories> and <relevant_memories> blocks.

    Prevents retain feedback loop — these were injected during recall and
    should not be re-stored.
    """
    if not isinstance(content, str):
        return content
    content = re.sub(r"<hindsight_memories>[\s\S]*?</hindsight_memories>", "", content)
    content = re.sub(r"<relevant_memories>[\s\S]*?</relevant_memories>", "", content)
    return content


# ---------------------------------------------------------------------------
# Transcript reading
# ---------------------------------------------------------------------------


def read_transcript(transcript_path, include_tool_calls=False):
    """Read a Cursor JSONL transcript and return list of message dicts.

    When `include_tool_calls` is False (default for retention), we keep
    the transcript light: only text from user/assistant messages is
    preserved, with the [role:]...[role:end] markers downstream.

    When `include_tool_calls` is True, we project tool_call events into
    structured content blocks (matching Claude Code's JSON format):
      - {"role": "user",      "content": [{"type": "text", "text": "..."}]}
      - {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "name": "shell", "input": {...}},
            {"type": "tool_result", "content": "..."},
        ]}

    Flat format for testing:
      - {"role": "user", "content": "..."}
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return []

    if include_tool_calls:
        return _read_transcript_rich(transcript_path)
    return _read_transcript_text(transcript_path)


def _read_transcript_text(transcript_path):
    """Light text-only transcript reader — user/assistant text only."""
    messages = []
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Flat test format
                if "role" in entry and "content" in entry:
                    role = entry.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    text = entry["content"]
                    if isinstance(text, list):
                        text = _extract_text_from_blocks(text)
                    if isinstance(text, str) and text.strip():
                        messages.append({"role": role, "content": text.strip()})
                    continue

                # Cursor SDK envelope
                if entry.get("type") in ("user", "assistant"):
                    msg = entry.get("message", {})
                    role = msg.get("role") or entry["type"]
                    if role not in ("user", "assistant"):
                        continue
                    text = _extract_text_from_blocks(msg.get("content", []))
                    if text.strip():
                        messages.append({"role": role, "content": text.strip()})
    except OSError:
        pass
    return messages


def _read_transcript_rich(transcript_path):
    """Rich transcript reader that preserves tool calls as structured blocks.

    Collects all assistant-side events between user messages into a
    single assistant message with structured content blocks.
    """
    messages = []
    assistant_blocks = []

    def _flush_assistant():
        if assistant_blocks:
            # Pass a snapshot — `assistant_blocks` is reused for the next
            # turn and we don't want later clears() to wipe this one.
            messages.append({"role": "assistant", "content": list(assistant_blocks)})
            assistant_blocks.clear()

    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Flat test format
                if "role" in entry and "content" in entry:
                    role = entry["role"]
                    content = entry["content"]
                    if role == "user":
                        _flush_assistant()
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        messages.append({"role": "user", "content": content})
                    elif role == "assistant":
                        if isinstance(content, str):
                            content = [{"type": "text", "text": content}]
                        if isinstance(content, list):
                            assistant_blocks.extend(content)
                        else:
                            assistant_blocks.append({"type": "text", "text": str(content)})
                    continue

                event_type = entry.get("type")

                if event_type == "user":
                    _flush_assistant()
                    msg = entry.get("message", {})
                    text = _extract_text_from_blocks(msg.get("content", []))
                    if text.strip():
                        messages.append({"role": "user", "content": [{"type": "text", "text": text.strip()}]})

                elif event_type == "assistant":
                    msg = entry.get("message", {})
                    text = _extract_text_from_blocks(msg.get("content", []))
                    if text.strip():
                        assistant_blocks.append({"type": "text", "text": text.strip()})

                elif event_type == "thinking":
                    text = entry.get("text", "")
                    if text:
                        assistant_blocks.append({"type": "text", "text": f"[thinking] {text.strip()}"})

                elif event_type == "tool_call":
                    name = entry.get("name", "unknown")
                    status = entry.get("status", "running")
                    args = entry.get("args")
                    result = entry.get("result")
                    truncated = entry.get("truncated") or {}
                    if args is not None:
                        assistant_blocks.append(
                            {
                                "type": "tool_use",
                                "name": name,
                                "input": _maybe_json_loads(args),
                                "truncated": bool(truncated.get("args")),
                            }
                        )
                    if status in ("completed", "error") and result is not None:
                        result_text = _coerce_result_text(result)
                        assistant_blocks.append(
                            {
                                "type": "tool_result",
                                "name": name,
                                "content": _truncate(result_text),
                                "truncated": bool(truncated.get("result")),
                                "status": status,
                            }
                        )

                elif event_type == "status":
                    # Mostly lifecycle telemetry; skip but keep in transcript.
                    continue

                elif event_type == "task":
                    text = entry.get("text") or entry.get("summary") or ""
                    if text:
                        assistant_blocks.append({"type": "text", "text": f"[task] {text.strip()}"})

                elif event_type == "request":
                    # Awaiting user input — nothing to retain.
                    continue

                elif event_type == "system":
                    # Init metadata.
                    continue

    except OSError:
        pass

    _flush_assistant()
    return messages


def _extract_text_from_blocks(blocks):
    """Extract plain text from a list of Cursor content blocks."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _coerce_result_text(result):
    """Coerce a tool result (unknown shape) into a string."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _maybe_json_loads(value):
    """If `value` looks like a JSON string, parse it; else return as-is."""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
    return value


def _truncate(text):
    if len(text) > _MAX_TOOL_OUTPUT_CHARS:
        return text[:_MAX_TOOL_OUTPUT_CHARS] + "... (truncated)"
    return text


# ---------------------------------------------------------------------------
# Recall: query composition and truncation
# ---------------------------------------------------------------------------


def compose_recall_query(latest_query, messages, recall_context_turns, recall_roles=None):
    """Compose a multi-turn recall query from conversation history."""
    latest = latest_query.strip()
    if recall_context_turns <= 1 or not isinstance(messages, list) or not messages:
        return latest

    allowed_roles = set(recall_roles or ["user", "assistant"])
    contextual_messages = slice_last_turns_by_user_boundary(messages, recall_context_turns)

    context_lines = []
    for msg in contextual_messages:
        role = msg.get("role")
        if role not in allowed_roles:
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = strip_memory_tags(content).strip()
        if not content:
            continue

        if role == "user" and content == latest:
            continue

        context_lines.append(f"{role}: {content}")

    if not context_lines:
        return latest

    return "\n\n".join(
        [
            "Prior context:",
            "\n".join(context_lines),
            latest,
        ]
    )


def truncate_recall_query(query, latest_query, max_chars):
    """Truncate a composed recall query to max_chars. Preserves the latest user message."""
    if max_chars <= 0:
        return query

    latest = latest_query.strip()
    if len(query) <= max_chars:
        return query

    latest_only = latest[:max_chars] if len(latest) > max_chars else latest

    if "Prior context:" not in query:
        return latest_only

    context_marker = "Prior context:\n\n"
    marker_index = query.find(context_marker)
    if marker_index == -1:
        return latest_only

    suffix_marker = "\n\n" + latest
    suffix_index = query.rfind(suffix_marker)
    if suffix_index == -1:
        return latest_only

    suffix = query[suffix_index:]
    if len(suffix) >= max_chars:
        return latest_only

    context_body = query[marker_index + len(context_marker) : suffix_index]
    context_lines = [line for line in context_body.split("\n") if line]

    kept = []
    for i in range(len(context_lines) - 1, -1, -1):
        kept.insert(0, context_lines[i])
        candidate = f"{context_marker}{chr(10).join(kept)}{suffix}"
        if len(candidate) > max_chars:
            kept.pop(0)
            break

    if kept:
        return f"{context_marker}{chr(10).join(kept)}{suffix}"
    return latest_only


# ---------------------------------------------------------------------------
# Turn slicing
# ---------------------------------------------------------------------------


def slice_last_turns_by_user_boundary(messages, turns):
    """Slice messages to the last N turns, where a turn starts at a user message."""
    if not isinstance(messages, list) or not messages or turns <= 0:
        return []

    user_turns_seen = 0
    start_index = -1

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_turns_seen += 1
            if user_turns_seen >= turns:
                start_index = i
                break

    if start_index == -1:
        return list(messages)

    return messages[start_index:]


# ---------------------------------------------------------------------------
# Memory formatting (recall results → context string)
# ---------------------------------------------------------------------------


def format_memories(results):
    """Format recall results into human-readable text."""
    if not results:
        return ""
    lines = []
    for r in results:
        text = r.get("text", "")
        mem_type = r.get("type", "")
        mentioned_at = r.get("mentioned_at", "")
        type_str = f" [{mem_type}]" if mem_type else ""
        date_str = f" ({mentioned_at})" if mentioned_at else ""
        lines.append(f"- {text}{type_str}{date_str}")
    return "\n\n".join(lines)


def format_current_time():
    """Format current UTC time for recall context."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Retention transcript formatting
# ---------------------------------------------------------------------------


def prepare_retention_transcript(
    messages,
    retain_roles=None,
    retain_full_window=False,
    include_tool_calls=False,
):
    """Format messages into a retention transcript.

    When `include_tool_calls` is True, output JSON with full message
    structure including tool calls and their inputs (matching Claude
    Code's format). Otherwise output the legacy text format with
    [role: ...]...[role:end] markers.

    Returns:
        (transcript_text, message_count) or (None, 0) if nothing to retain.
    """
    if not messages:
        return None, 0

    if retain_full_window:
        target_messages = messages
    else:
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx == -1:
            return None, 0
        target_messages = messages[last_user_idx:]

    allowed_roles = set(retain_roles or ["user", "assistant"])

    if include_tool_calls:
        return _prepare_json_transcript(target_messages, allowed_roles)
    return _prepare_text_transcript(target_messages, allowed_roles)


def _prepare_json_transcript(messages, allowed_roles):
    """Format messages as JSON with full tool call data."""
    structured_messages = []
    for msg in messages:
        role = msg.get("role", "unknown")
        if role not in allowed_roles:
            continue

        content = msg.get("content", "")
        blocks = _strip_memory_tags_from_blocks(content)
        if not blocks:
            continue

        structured_messages.append({"role": role, "content": blocks})

    if not structured_messages:
        return None, 0

    transcript = json.dumps(structured_messages, ensure_ascii=False)
    if len(transcript.strip()) < 10:
        return None, 0

    return transcript, len(structured_messages)


def _prepare_text_transcript(messages, allowed_roles):
    """Format messages as legacy text with [role:]...[role:end] markers."""
    parts = []

    for msg in messages:
        role = msg.get("role", "unknown")
        if role not in allowed_roles:
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = strip_memory_tags(content).strip()

        if not content:
            continue

        parts.append(f"[role: {role}]\n{content}\n[{role}:end]")

    if not parts:
        return None, 0

    transcript = "\n\n".join(parts)
    if len(transcript.strip()) < 10:
        return None, 0

    return transcript, len(parts)


def _strip_memory_tags_from_blocks(content):
    """Strip memory tags from content, handling both string and list formats."""
    if isinstance(content, str):
        cleaned = strip_memory_tags(content).strip()
        return [{"type": "text", "text": cleaned}] if cleaned else []

    if not isinstance(content, list):
        return []

    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            text = strip_memory_tags(block.get("text", "")).strip()
            if text:
                blocks.append({"type": "text", "text": text})
        elif block_type in ("tool_use", "tool_result"):
            # Pass through tool blocks as-is
            blocks.append(block)

    return blocks
