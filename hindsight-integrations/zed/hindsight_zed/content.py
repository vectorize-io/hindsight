"""Format Zed threads for retain, and recall results for injection."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .threads_db import ThreadMessage, ZedThread


def compose_recall_query(messages: "list[ThreadMessage]", max_chars: int = 800) -> str:
    """Build a recall query from the most recent user message(s).

    Recall is keyed on what the user is currently asking, so we use the last
    user turn (truncated). Falls back to the last message of any role.
    """
    last_user = next((m.text for m in reversed(messages) if m.role == "user" and m.text.strip()), "")
    query = last_user or (messages[-1].text if messages else "")
    query = query.strip()
    if len(query) > max_chars:
        query = query[:max_chars]
    return query


def format_memory_block(results: list) -> str:
    """Format recall results into the memory block body (markdown list).

    ``results`` is the ``results`` array from the recall API response.
    """
    lines = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        mem_type = r.get("type") or ""
        when = r.get("mentioned_at") or ""
        suffix = ""
        if mem_type:
            suffix += f" [{mem_type}]"
        if when:
            suffix += f" ({when})"
        lines.append(f"- {text}{suffix}")
    return "\n".join(lines)


def format_transcript(thread: "ZedThread") -> str:
    """Render a thread's messages as a ``[role]\\ntext`` transcript for retain."""
    blocks = []
    for m in thread.messages:
        text = m.text.strip()
        if text:
            blocks.append(f"[{m.role}]\n{text}")
    return "\n\n".join(blocks)
