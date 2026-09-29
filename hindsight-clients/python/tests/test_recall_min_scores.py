"""The maintained wrapper threads min_scores into the recall request."""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_min_scores(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall(
        "test-bank",
        "q",
        min_scores={"semantic": 0.2, "final": 0.5},
    )

    min_scores = captured["request"].min_scores
    assert min_scores is not None
    assert min_scores.semantic == 0.2
    assert min_scores.final == 0.5
    # Unspecified stages impose no floor.
    assert min_scores.keyword is None
    assert min_scores.reranker is None


def test_recall_min_scores_defaults_none(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall("test-bank", "q")

    assert captured["request"].min_scores is None


def test_recall_min_scores_rejects_unknown_key(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    # A typo must fail loud, not silently apply no floor.
    with pytest.raises(ValueError, match="sematic"):
        client.recall("test-bank", "q", min_scores={"sematic": 0.8})

    assert "request" not in captured
