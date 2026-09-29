"""Content helpers: recall-query composition, memory formatting, and the
per-task transcript accumulator.

Cline does not hand hooks a conversation transcript, so we build one
ourselves: UserPromptSubmit/TaskStart append turns to per-task state, and the
end-of-task hooks read it back to retain. The recall-query/formatting helpers
are reused from the Codex integration (same behavior, same tests).
"""

import re
from datetime import datetime, timezone

from .state import clear_state, read_state, write_state

# ── Transcript accumulator (Cline-specific) ──────────────────────────────────


def _transcript_key(task_id: str) -> str:
    return f"transcript_{task_id or 'unknown'}.json"


def append_turn(task_id: str, role: str, content: str) -> None:
    """Append one {role, content} turn to a task's accumulated transcript."""
    content = (content or "").strip()
    if not content:
        return
    key = _transcript_key(task_id)
    messages = read_state(key, [])
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": role, "content": content})
    # Cap to keep state files bounded on very long tasks.
    if len(messages) > 500:
        messages = messages[-500:]
    write_state(key, messages)


def read_transcript(task_id: str) -> list:
    """Return the accumulated transcript for a task (list of {role, content})."""
    messages = read_state(_transcript_key(task_id), [])
    return messages if isinstance(messages, list) else []


def clear_transcript(task_id: str) -> None:
    """Drop a task's accumulated transcript (called after a successful retain)."""
    clear_state(_transcript_key(task_id))


def format_retention(messages: list) -> str:
    """Render accumulated turns into a plain-text transcript for retain."""
    blocks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = strip_memory_tags(str(msg.get("content", ""))).strip()
        if content:
            blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


# ── Reused generic helpers (from the Codex integration) ──────────────────────


def strip_memory_tags(content: str) -> str:
    """Remove <hindsight_memories>/<relevant_memories> blocks.

    Prevents a retain feedback loop — these were injected during recall and
    must not be re-stored.
    """
    content = re.sub(r"<hindsight_memories>[\s\S]*?</hindsight_memories>", "", content)
    content = re.sub(r"<relevant_memories>[\s\S]*?</relevant_memories>", "", content)
    return content


def slice_last_turns_by_user_boundary(messages: list, turns: int) -> list:
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


def compose_recall_query(latest_query: str, messages: list, recall_context_turns: int) -> str:
    """Compose a multi-turn recall query from accumulated history.

    With recallContextTurns <= 1 (the default), returns just the latest query.
    """
    latest = latest_query.strip()
    if recall_context_turns <= 1 or not isinstance(messages, list) or not messages:
        return latest

    contextual = slice_last_turns_by_user_boundary(messages, recall_context_turns)
    context_lines = []
    for msg in contextual:
        role = msg.get("role")
        content = strip_memory_tags(str(msg.get("content", ""))).strip()
        if not content:
            continue
        if role == "user" and content == latest:
            continue
        context_lines.append(f"{role}: {content}")

    if not context_lines:
        return latest
    return "\n\n".join(["Prior context:", "\n".join(context_lines), latest])


def truncate_recall_query(query: str, latest_query: str, max_chars: int) -> str:
    """Truncate a composed query to max_chars, preserving the latest message.

    Drops oldest context lines first.
    """
    if max_chars <= 0:
        return query

    latest = latest_query.strip()
    if len(query) <= max_chars:
        return query

    latest_only = latest[:max_chars] if len(latest) > max_chars else latest

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


def format_memories(results: list) -> str:
    """Format recall results into human-readable bullet lines."""
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


def format_current_time() -> str:
    """UTC time for the recall context block."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
