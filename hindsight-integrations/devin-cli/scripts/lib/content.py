"""Content processing utilities.

Faithful port of Openclaw plugin's content processing: memory tag stripping,
query composition/truncation, transcript formatting, and memory formatting.

Source: reference/openclaw-source/index.js — stripMemoryTags, composeRecallQuery,
truncateRecallQuery, sliceLastTurnsByUserBoundary, prepareRetentionTranscript,
formatMemories.
"""

import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Memory tag stripping (anti-feedback-loop)
# ---------------------------------------------------------------------------


def strip_channel_envelope(content: str) -> str:
    """Strip Claude Code channel XML wrappers from user messages.

    Claude Code wraps incoming channel messages in XML:
      <channel source="plugin:telegram:telegram" chat_id="..." ...>
      actual message text
      </channel>

    This is the Claude Code equivalent of Openclaw's stripMetadataEnvelopes().
    Extracts the inner text, preserving the actual user message while removing
    transport metadata that Hindsight doesn't need.
    """
    # fullmatch, not search: the envelope is the whole message or it is not an
    # envelope at all. An unanchored search reduced any message that merely
    # *mentioned* one to its inner text — "please explain
    # <channel source="x">hello</channel> thanks" was retained as "hello",
    # silently discarding what the user actually wrote. Surrounding whitespace
    # is allowed because the real wrapper puts the tags on their own lines.
    match = re.fullmatch(r"\s*<channel\b[^>]*>([\s\S]*?)</channel>\s*", content)
    if match:
        return match.group(1).strip()
    return content


def strip_memory_tags(content: str) -> str:
    """Remove <hindsight_memories> and <relevant_memories> blocks.

    Prevents retain feedback loop — these were injected during recall and
    should not be re-stored.

    Port of: stripMemoryTags() in index.js
    """
    content = re.sub(r"<hindsight_memories>[\s\S]*?</hindsight_memories>", "", content)
    content = re.sub(r"<relevant_memories>[\s\S]*?</relevant_memories>", "", content)
    return content


# ---------------------------------------------------------------------------
# Recall: query composition and truncation
# ---------------------------------------------------------------------------


def compose_recall_query(
    latest_query: str,
    messages: list,
    recall_context_turns: int,
    recall_roles: list = None,
) -> str:
    """Compose a multi-turn recall query from conversation history.

    Port of: composeRecallQuery() in index.js

    When recallContextTurns > 1, includes prior context from the transcript
    above the latest user query. Format:

        Prior context:

        user: ...
        assistant: ...

        <latest query>
    """
    # Guarded here as well as at the hook boundary: these are library
    # functions with their own callers, and an empty query is the same
    # degradation the rest of the recall path already applies.
    latest = latest_query.strip() if isinstance(latest_query, str) else ""
    if recall_context_turns <= 1 or not isinstance(messages, list) or not messages:
        return latest

    allowed_roles = set(recall_roles or ["user", "assistant"])
    contextual_messages = slice_last_turns_by_user_boundary(messages, recall_context_turns)

    context_lines = []
    for msg in contextual_messages:
        role = msg.get("role")
        if role not in allowed_roles:
            continue

        content = _extract_text_content(msg.get("content", ""), role=role)
        content = strip_channel_envelope(content)
        content = strip_memory_tags(content).strip()
        if not content:
            continue

        # Skip if this is the same as the latest query (avoid duplication)
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


def truncate_recall_query(query: str, latest_query: str, max_chars: int) -> str:
    """Truncate a composed recall query to max_chars.

    Port of: truncateRecallQuery() in index.js

    Preserves the latest user message. When the query contains "Prior context:",
    drops oldest context lines first (from the top) to fit within the limit.
    """
    if max_chars <= 0:
        return query

    # Guarded here as well as at the hook boundary: these are library
    # functions with their own callers, and an empty query is the same
    # degradation the rest of the recall path already applies.
    latest = latest_query.strip() if isinstance(latest_query, str) else ""
    if len(query) <= max_chars:
        return query

    # If even the latest alone is too long, hard-truncate it
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

    suffix = query[suffix_index:]  # \n\n<latest>
    if len(suffix) >= max_chars:
        return latest_only

    context_body = query[marker_index + len(context_marker) : suffix_index]
    context_lines = [line for line in context_body.split("\n") if line]

    # Add context lines from newest (bottom) to oldest (top), stop when exceeding
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


def _is_real_user_turn(msg) -> bool:
    """True if `msg` is a user message carrying actual text the user wrote.

    A tool result is delivered as a `role: "user"` message whose content is a
    tool_result block, so counting every `role: "user"` message as a turn
    boundary puts the boundary on the tool result instead of the prompt that
    caused it — and the window then starts mid-turn, dropping the user input it
    was supposed to include. The sibling openclaw integration filters these the
    same way (see its sliceLastTurnsByUserBoundary).
    """
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip()
            for b in content
        )
    return False


def slice_last_turns_by_user_boundary(messages: list, turns: int) -> list:
    """Slice messages to the last N turns, where a turn starts at a user message.

    Port of: sliceLastTurnsByUserBoundary() in index.js

    Walks backward counting user messages as turn boundaries. Returns
    messages from the Nth user boundary to the end. Synthetic tool-result
    messages do not count — see _is_real_user_turn().
    """
    if not isinstance(messages, list) or not messages or turns <= 0:
        return []

    user_turns_seen = 0
    start_index = -1

    for i in range(len(messages) - 1, -1, -1):
        if _is_real_user_turn(messages[i]):
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


def format_memories(results: list) -> str:
    """Format recall results into human-readable text.

    Port of: formatMemories() in index.js
    Format: - <text> [<type>] (<mentioned_at>)
    """
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
    """Format current UTC time for recall context.

    The "UTC" suffix is explicit so client LLMs do not misread the
    value as local time when reasoning about wall-clock context.

    Port of: formatCurrentTimeForRecall() in index.js
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Retention transcript formatting
# ---------------------------------------------------------------------------


def _extract_message_blocks(content, role: str = "", skipped_tool_use_ids=None) -> list:
    """Extract structured content blocks from a message for JSON retention.

    Returns a list of dicts, each representing a content block:
      - {"type": "text", "text": "..."} for text blocks
      - {"type": "tool_use", "name": "...", "input": {...}} for tool calls
      - Channel message tool_use blocks get their text extracted inline.

    `skipped_tool_use_ids` accumulates the ids of operational Hindsight tool_use
    blocks that were dropped, so their tool_result — which arrives in a later
    message, and carries the recalled memories verbatim — can be dropped too.
    Pass the same set across every message of a transcript.
    """
    if skipped_tool_use_ids is None:
        skipped_tool_use_ids = set()
    if isinstance(content, str):
        cleaned = strip_channel_envelope(strip_memory_tags(content)).strip()
        return [{"type": "text", "text": cleaned}] if cleaned else []

    if not isinstance(content, list):
        return []

    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")

        if block_type == "text":
            # Skipped rather than str()-coerced: a null `text` is a truncated
            # block, and coercing would retain the literal "None" as if the
            # model had said it. The blocks reaching here are decoded from
            # Devin CLI's own SQLite payload, so their shape is not ours to
            # guarantee — and strip_memory_tags() runs re.sub() on the value,
            # which raises on anything but a string.
            raw_text = block.get("text")
            if not isinstance(raw_text, str):
                continue
            text = strip_channel_envelope(strip_memory_tags(raw_text)).strip()
            if text:
                blocks.append({"type": "text", "text": text})

        elif block_type == "tool_use" and role == "assistant":
            if _is_channel_message_tool(block):
                # Channel messages: extract the outgoing text
                tool_input = block.get("input", {})
                for field in _MESSAGE_TEXT_FIELDS:
                    val = tool_input.get(field)
                    if isinstance(val, str) and val.strip():
                        blocks.append({"type": "text", "text": val.strip()})
                        break
            else:
                # `.get(k, default)` only fills in for a *missing* key, so an
                # explicit "name": null reached .startswith() and raised.
                name = block.get("name")
                if not isinstance(name, str):
                    name = "unknown"
                inp = block.get("input", {})
                # Skip Hindsight MCP tools to avoid feedback loops
                if name.startswith("mcp__") and _OPERATIONAL_TOOL_PATTERN.search(name.split("__")[-1]):
                    block_id = block.get("id")
                    # `None` is recorded deliberately when the block carries no
                    # usable id. The matching tool_result cannot carry a
                    # tool_use_id either — both sides are decoded from the same
                    # payload — so its own *missing* id is the only handle left
                    # to correlate on, and block.get() below yields None for it
                    # too. Dropping id-less results once an id-less operational
                    # call was dropped is the conservative direction: a false
                    # positive costs one tool result missing from the
                    # transcript, a false negative writes recalled memories
                    # back into the bank, which then compounds on every retain.
                    skipped_tool_use_ids.add(block_id if isinstance(block_id, str) and block_id else None)
                    continue
                blocks.append({"type": "tool_use", "name": name, "input": inp})

        elif block_type == "tool_result":
            # Dropping the operational tool_use above is only half the job: its
            # result is what actually carries the injected memories back into the
            # transcript, which is the feedback loop we are avoiding.
            # An unhashable tool_use_id would raise on the set lookup, and it
            # cannot have been put there by the branch above in any case, so it
            # is simply not one of ours.
            tool_use_id = block.get("tool_use_id")
            if (tool_use_id is None or isinstance(tool_use_id, str)) and tool_use_id in skipped_tool_use_ids:
                continue
            # Include tool results for context.
            # content can be a plain string or a list of content blocks
            # (e.g. [{"type": "text", "text": "..."}] for Agent results).
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                # Extract text from content blocks
                parts = []
                for item in result_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        # An explicit "text": null reaches .strip() as None —
                        # dict.get(k, default) only fills in for a *missing*
                        # key. Same defect as the "name": null tool_use block
                        # fixed above; one malformed block would abort the
                        # whole transcript.
                        t = item.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                result_content = "\n".join(parts)
            if isinstance(result_content, str) and result_content.strip():
                text = result_content.strip()
                # Truncate very long results
                if len(text) > 2000:
                    text = text[:2000] + "... (truncated)"
                blocks.append({"type": "tool_result", "tool_use_id": block.get("tool_use_id", ""), "content": text})

    return blocks


def prepare_retention_transcript(
    messages: list,
    retain_roles: list = None,
    retain_full_window: bool = False,
    include_tool_calls: bool = False,
) -> tuple:
    """Format messages into a retention transcript.

    When include_tool_calls is True, outputs JSON with full message structure
    including tool calls and their inputs. Otherwise outputs the legacy
    text format with [role: ...]...[role:end] markers.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        retain_roles: Roles to include (default: ['user', 'assistant']).
        retain_full_window: If True, retain all messages (chunked mode).
            If False, retain only the last turn (last user msg + responses).
        include_tool_calls: If True, output JSON format with full tool call data.

    Returns:
        (transcript_text, message_count) or (None, 0) if nothing to retain.
    """
    if not messages:
        return None, 0

    if retain_full_window:
        target_messages = messages
    else:
        # Default: retain only the last turn
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


def _prepare_json_transcript(messages: list, allowed_roles: set) -> tuple:
    """Format messages as JSON with full tool call data."""
    import json

    structured_messages = []
    # Shared across messages: an operational tool_use and its tool_result sit in
    # different messages, so the ids have to outlive a single call.
    skipped_tool_use_ids = set()
    for msg in messages:
        role = msg.get("role", "unknown")
        if role not in allowed_roles:
            continue

        blocks = _extract_message_blocks(msg.get("content", ""), role=role, skipped_tool_use_ids=skipped_tool_use_ids)
        if not blocks:
            continue

        structured_messages.append({"role": role, "content": blocks})

    if not structured_messages:
        return None, 0

    transcript = json.dumps(structured_messages, indent=None, ensure_ascii=False)
    if len(transcript.strip()) < 10:
        return None, 0

    return transcript, len(structured_messages)


def _prepare_text_transcript(messages: list, allowed_roles: set) -> tuple:
    """Format messages as legacy text with [role:]...[role:end] markers."""
    parts = []

    for msg in messages:
        role = msg.get("role", "unknown")
        if role not in allowed_roles:
            continue

        content = _extract_text_content(msg.get("content", ""), role=role)
        content = strip_channel_envelope(content)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fields in tool_use input that carry the outgoing message text.
# Ordered by likelihood — first match wins.
_MESSAGE_TEXT_FIELDS = ("text", "body", "message", "content")

# MCP tool name suffixes that are operational, not conversational.
# Checked against the last segment of the tool name (after the last __).
import re as _re

_OPERATIONAL_TOOL_PATTERN = _re.compile(
    r"(?:recall|retain|reflect|search|extract|create_|delete_|update_|get_|list_)",
    _re.IGNORECASE,
)


def _is_channel_message_tool(block: dict) -> bool:
    """Detect if a tool_use block is a channel message (reply/send).

    Uses a structural approach rather than name-matching for robustness:
      1. Must be an MCP tool (name starts with "mcp__")
      2. Must NOT match known operational patterns (recall, search, CRUD)
      3. Must have a text-like field in input (text, body, message, content)

    This catches any channel plugin (Telegram, Slack, Discord, Matrix,
    future channels) without hardcoding tool names. Built-in tools (Bash,
    Read, Write) don't start with mcp__. MCP tools for non-messaging
    purposes (hindsight recall, search) are excluded by pattern and by
    lacking text/body fields.
    """
    # Not `.get("name", "")` — that only fills in for a *missing* key, so an
    # explicit "name": null reached .startswith() and raised. This runs before
    # the caller's own name handling, so guarding only there was not enough.
    name = block.get("name")
    if not isinstance(name, str) or not name.startswith("mcp__"):
        return False

    # Exclude operational MCP tools (check only the tool suffix, not server name)
    tool_suffix = name.split("__")[-1]
    if _OPERATIONAL_TOOL_PATTERN.search(tool_suffix):
        return False

    tool_input = block.get("input", {})
    if not isinstance(tool_input, dict):
        return False

    # Must have a text-carrying field with actual content
    return any(isinstance(tool_input.get(f), str) and tool_input[f].strip() for f in _MESSAGE_TEXT_FIELDS)


def _extract_text_content(content, role: str = "") -> str:
    """Extract text from message content (string or content blocks array).

    For user messages: extracts from plain strings (channel XML wrappers
    are stripped separately by strip_channel_envelope).

    For assistant messages: extracts from:
      - {type: "text"} blocks — terminal output/narration
      - {type: "tool_use"} blocks detected as channel messages — the agent's
        actual responses to the user. Detection is structural (MCP tool with
        text-like input field), not name-based, for channel-agnosticism.

    Excludes:
      - {type: "thinking"} — internal reasoning
      - {type: "tool_use"} for operational tools — Bash, Read, Write, recall, etc.
      - {type: "tool_result"} — operational results, not conversation
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            # Text blocks: terminal output / narration
            if block_type == "text":
                raw_text = block.get("text")
                if not isinstance(raw_text, str):
                    continue
                text = raw_text.strip()
                if text:
                    texts.append(text)

            # Tool use blocks: extract channel messages
            elif block_type == "tool_use" and role == "assistant":
                if _is_channel_message_tool(block):
                    tool_input = block.get("input", {})
                    for field in _MESSAGE_TEXT_FIELDS:
                        val = tool_input.get(field)
                        if isinstance(val, str) and val.strip():
                            texts.append(val.strip())
                            break

        return "\n".join(texts)
    return ""
