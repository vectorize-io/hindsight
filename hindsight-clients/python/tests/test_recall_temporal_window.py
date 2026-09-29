"""The maintained wrapper threads temporal_window into the recall request.

Kept in step with the TypeScript wrapper's equivalent test: a parameter that
only one wrapper forwards is silently dropped for every consumer of the other.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_recall(monkeypatch, client, captured):
    async def fake_recall(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(results=[])

    monkeypatch.setattr(client._memory_api, "recall_memories", fake_recall)


def test_recall_threads_temporal_window(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall(
        "test-bank",
        "q",
        temporal_window={"start": "2023-04-01T00:00:00Z", "end": "2023-06-30T23:59:59Z"},
    )

    window = captured["request"].temporal_window
    assert window is not None
    assert window.start.isoformat() == "2023-04-01T00:00:00+00:00"
    assert window.end.isoformat() == "2023-06-30T23:59:59+00:00"


def test_recall_temporal_window_defaults_none(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_recall(monkeypatch, client, captured)

    client.recall("test-bank", "q")

    assert captured["request"].temporal_window is None
