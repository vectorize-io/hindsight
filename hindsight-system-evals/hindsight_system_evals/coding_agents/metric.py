"""The number this harness exists to produce: searches per user turn.

Read from the plugin's own ``usage.jsonl`` — one line per finished turn,
``{harness, session, bank, turn, calls, credited}`` — rather than re-derived from
the transcript. Two reasons: it is the same counter the shipped ``stats`` command
reports, so a harness result and a field report mean the same thing; and the
plugin already solves "which tool name is a Hindsight tool" across host prefixes.

A turn with no Hindsight call at all is still a line in that file, so the
denominator is every turn the agent took, which is the rate that matters.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SEARCH_TOOL = "hindsight_search_knowledge_pages"

#: The tools that hand memory back, mirroring core/usage.ts. Writes
#: (ingest_document, capture_initiative) are not retrieval and never count.
RETRIEVAL_TOOLS = frozenset(
    {
        "hindsight_search_knowledge_pages",
        "hindsight_list_knowledge_pages",
        "hindsight_read_knowledge_page",
        "hindsight_reflect",
    }
)


@dataclass
class UsageStats:
    turns: int = 0
    searches: int = 0
    turns_with_search: int = 0
    retrieval_calls: int = 0
    turns_with_retrieval: int = 0
    turns_with_any_call: int = 0
    credited_turns: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    sessions: int = 0

    @property
    def searches_per_turn(self) -> float:
        return self.searches / self.turns if self.turns else 0.0

    @property
    def search_turn_rate(self) -> float:
        """Share of turns that searched at least once — the headline number."""
        return self.turns_with_search / self.turns if self.turns else 0.0

    @property
    def retrieval_turn_rate(self) -> float:
        return self.turns_with_retrieval / self.turns if self.turns else 0.0

    @property
    def credit_rate(self) -> float | None:
        """Of the turns that retrieved, how many credited memory in the reply.

        ``None`` when nothing retrieved: 0/0 is not 0%, and printing 0% there
        reads as "retrieval was useless" when it means "retrieval never ran".
        """
        if not self.turns_with_retrieval:
            return None
        return self.credited_turns / self.turns_with_retrieval

    def to_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "turns": self.turns,
            "searches": self.searches,
            "searches_per_turn": round(self.searches_per_turn, 4),
            "turns_with_search": self.turns_with_search,
            "search_turn_rate": round(self.search_turn_rate, 4),
            "retrieval_calls": self.retrieval_calls,
            "turns_with_retrieval": self.turns_with_retrieval,
            "retrieval_turn_rate": round(self.retrieval_turn_rate, 4),
            "turns_with_any_call": self.turns_with_any_call,
            "credit_rate": round(self.credit_rate, 4) if self.credit_rate is not None else None,
            "calls_by_tool": dict(sorted(self.calls_by_tool.items(), key=lambda kv: -kv[1])),
        }


#: The credit blockquote the injected tool guide prescribes.
CREDIT_MARKER = "from hindsight memory"


def transcript_credits(transcript: Path) -> set[int]:
    """Turn numbers whose reply credited memory, read from the finished transcript.

    NOT taken from the plugin's live `credited` flag. The Stop hook evaluates a
    turn before the host has flushed that turn's final assistant message — the
    one carrying the credit line — so the live log understates attribution badly
    (13 recorded against 31 actually present, on the run that exposed it). The
    transcript on disk after the session is complete, so this counts what was
    really said.
    """
    credited: set[int] = set()
    if not transcript.exists():
        return credited

    turn = 0
    for line in transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message") or {}
        content = message.get("content")
        role = message.get("role")

        if role == "user":
            # A tool result arrives as a user message; it does not open a turn.
            if isinstance(content, list) and all(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content
            ):
                continue
            turn += 1
            continue
        if role not in ("assistant", "model") or not isinstance(content, list) or not turn:
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            if CREDIT_MARKER in (block.get("text") or "").lower():
                credited.add(turn)
    return credited


def read_usage_stats(
    path: Path,
    *,
    sessions: set[str] | None = None,
    credits: dict[str, set[int]] | None = None,
) -> UsageStats:
    """Aggregate ``usage.jsonl``, optionally restricted to the run's own sessions.

    Deduped by (session, turn) the way the shipped report is: losing the cursor
    that tracks how much of a session is already written re-records its turns.
    """
    stats = UsageStats()
    if not path.exists():
        return stats

    seen: set[tuple[str, int]] = set()
    tools: Counter[str] = Counter()
    seen_sessions: set[str] = set()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        session, turn = str(row.get("session", "")), row.get("turn")
        if not isinstance(turn, int):
            continue
        if sessions is not None and session not in sessions:
            continue
        if (session, turn) in seen:
            continue
        seen.add((session, turn))
        seen_sessions.add(session)

        calls = [c for c in row.get("calls", []) if isinstance(c, str)]
        tools.update(calls)
        unique = set(calls)
        stats.turns += 1
        stats.searches += calls.count(SEARCH_TOOL)
        stats.turns_with_search += SEARCH_TOOL in unique
        retrieval = unique & RETRIEVAL_TOOLS
        stats.retrieval_calls += sum(1 for c in calls if c in RETRIEVAL_TOOLS)
        stats.turns_with_retrieval += bool(retrieval)
        stats.turns_with_any_call += bool(unique)
        credited = turn in credits.get(session, set()) if credits is not None else bool(row.get("credited"))
        if retrieval and credited:
            stats.credited_turns += 1

    stats.calls_by_tool = dict(tools)
    stats.sessions = len(seen_sessions)
    return stats
