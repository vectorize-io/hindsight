"""The maintained wrapper threads rerank (prune / dedup) into the recall request.

Mirrors the TypeScript wrapper's ``recall_rerank_mapping.test.ts``.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_rerank(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall("test-bank", "q", rerank={"prune": True, "dedup": False})

    rerank = captured["request"].rerank
    assert rerank is not None
    assert rerank.prune is True
    assert rerank.dedup is False


def test_recall_rerank_defaults_none(monkeypatch):
    """Omitted, the bank's recall_decision_model_prune / _dedup decide."""
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall("test-bank", "q")

    assert captured["request"].rerank is None


def test_recall_rerank_rejects_unknown_key(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    # A typo must fail loud, not silently leave the bank's default in charge.
    with pytest.raises(ValueError, match="dedupe"):
        client.recall("test-bank", "q", rerank={"dedupe": True})

    assert "request" not in captured
