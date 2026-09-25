"""The one deterministic piece of the coding-agents harness: the metric.

Everything else in that package needs a real model and a real agent. This does
not, so it is asserted exactly rather than judged — and it is the piece a wrong
answer would be silent about, because a rate is plausible whatever it says.
"""

from __future__ import annotations

import json
from pathlib import Path

from hindsight_system_evals.coding_agents.metric import read_usage_stats, transcript_credits

SEARCH = "hindsight_search_knowledge_pages"
WRITE = "hindsight_ingest_document"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_rate_counts_every_turn_not_just_the_ones_that_called(tmp_path: Path) -> None:
    usage = _write(
        tmp_path / "usage.jsonl",
        [
            {"session": "a", "turn": 1, "calls": [], "credited": False},
            {"session": "a", "turn": 2, "calls": [SEARCH], "credited": True},
            {"session": "a", "turn": 3, "calls": [SEARCH, SEARCH], "credited": False},
            {"session": "a", "turn": 4, "calls": [WRITE], "credited": False},
        ],
    )
    stats = read_usage_stats(usage)

    assert stats.turns == 4
    assert stats.searches == 3
    assert stats.turns_with_search == 2
    assert stats.search_turn_rate == 0.5
    # The write tool is a call, but never retrieval — counting it in the credit
    # denominator is what made the shipped rate read ~5% on real sessions.
    assert stats.turns_with_any_call == 3
    assert stats.turns_with_retrieval == 2
    assert stats.credit_rate == 0.5


def test_replayed_turns_are_deduped_and_other_runs_excluded(tmp_path: Path) -> None:
    usage = _write(
        tmp_path / "usage.jsonl",
        [
            {"session": "a", "turn": 1, "calls": [SEARCH], "credited": False},
            # The Stop hook re-reads the whole transcript; a lost cursor rewrites
            # turns that are already recorded.
            {"session": "a", "turn": 1, "calls": [SEARCH], "credited": False},
            {"session": "other-run", "turn": 1, "calls": [SEARCH], "credited": False},
        ],
    )

    assert read_usage_stats(usage).turns == 2
    assert read_usage_stats(usage, sessions={"a"}).turns == 1
    assert read_usage_stats(usage, sessions={"a"}).searches == 1


def test_nothing_retrieved_is_not_a_zero_percent_credit_rate(tmp_path: Path) -> None:
    usage = _write(tmp_path / "usage.jsonl", [{"session": "a", "turn": 1, "calls": [], "credited": False}])
    stats = read_usage_stats(usage)

    assert stats.credit_rate is None
    assert stats.to_dict()["credit_rate"] is None


def test_a_missing_usage_log_reads_as_no_turns(tmp_path: Path) -> None:
    assert read_usage_stats(tmp_path / "absent.jsonl").turns == 0


def _transcript(path: Path, entries: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return path


def _user(text: str) -> dict:
    return {"message": {"role": "user", "content": [{"type": "text", "text": text}]}}


def _reply(text: str) -> dict:
    return {"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def test_credit_is_counted_from_the_transcript_not_the_live_flag(tmp_path: Path) -> None:
    """The live flag misses a credit the host flushed after the Stop hook read."""
    transcript = _transcript(
        tmp_path / "s.jsonl",
        [
            _user("add a discount"),
            {"message": {"role": "assistant", "content": [{"type": "tool_use", "name": SEARCH, "input": {}}]}},
            # A tool result is a user message, but it does not open a new turn.
            {"message": {"role": "user", "content": [{"type": "tool_result", "content": "hits"}]}},
            _reply("> 🧠 **From Hindsight memory (Pricing decisions)** — post-discount threshold"),
            _user("ok commit this"),
            _reply("done, no memory involved"),
        ],
    )
    assert transcript_credits(transcript) == {1}

    usage = _write(
        tmp_path / "usage.jsonl",
        [
            {"session": "s", "turn": 1, "calls": [SEARCH], "credited": False},
            {"session": "s", "turn": 2, "calls": [], "credited": False},
        ],
    )
    assert read_usage_stats(usage).credited_turns == 0
    assert read_usage_stats(usage, credits={"s": transcript_credits(transcript)}).credited_turns == 1


def test_a_missing_transcript_credits_nothing(tmp_path: Path) -> None:
    assert transcript_credits(tmp_path / "absent.jsonl") == set()
