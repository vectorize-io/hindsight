"""The maintained wrapper threads resolve_entities into the retain request.

A field the generated SDK accepts but the wrapper drops is invisible to callers of the
wrapper — the exact gap #2975/#3042 closed for the mental-model methods. Here dropping it
would silently restore the entity substitution the flag exists to prevent (#3479).
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_retain(monkeypatch, client, captured):
    async def fake_retain(bank_id, request_obj, _request_timeout=None):
        captured["request"] = request_obj
        return MagicMock(success=True)

    monkeypatch.setattr(client._memory_api, "retain_memories", fake_retain)


def test_retain_threads_resolve_entities(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain(
        "test-bank",
        "The patient saw a specialist.",
        entities=[{"text": "Dr. Waller", "type": "PERSON"}],
        resolve_entities=False,
    )

    assert captured["request"].items[0].resolve_entities is False


def test_retain_resolve_entities_defaults_to_resolving(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain("test-bank", "A fact.", entities=[{"text": "Alice"}])

    # Omitted by the wrapper, so the server default (resolve) applies.
    assert captured["request"].items[0].resolve_entities is not False


def test_retain_batch_passes_resolve_entities_through(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_retain(monkeypatch, client, captured)

    client.retain_batch(
        "test-bank",
        items=[{"content": "A fact.", "entities": [{"text": "Alice"}], "resolve_entities": False}],
    )

    assert captured["request"].items[0].resolve_entities is False
