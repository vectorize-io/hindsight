"""The maintained wrapper threads prefer_observations into the recall request."""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_prefer_observations(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall(
        "test-bank",
        "q",
        types=["world", "experience", "observation"],
        prefer_observations=True,
    )

    assert captured["request"].prefer_observations is True


def test_recall_prefer_observations_defaults_false(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall("test-bank", "q")

    assert captured["request"].prefer_observations is False
