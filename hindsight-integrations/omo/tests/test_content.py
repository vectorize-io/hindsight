"""Tests for lib/content.py."""

from lib.content import filter_memories_by_score


def test_filter_memories_by_score_keeps_threshold_and_above():
    results = [
        {"text": "weak", "score": 0.24},
        {"text": "edge", "score": 0.25},
        {"text": "strong", "score": 0.9},
    ]
    out = filter_memories_by_score(results, 0.25)
    assert [r["text"] for r in out] == ["edge", "strong"]


def test_filter_memories_by_score_passes_missing_score_through():
    results = [{"text": "missing"}, {"text": "zero", "score": 0.0}]
    assert [r["text"] for r in filter_memories_by_score(results, 0.25)] == ["missing"]
    assert [r["text"] for r in filter_memories_by_score(results, 0.0)] == ["missing", "zero"]
