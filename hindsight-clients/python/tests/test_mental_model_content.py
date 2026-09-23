"""Authored mental-model content reaches the wire from the wrapper's own methods.

The generated request models grew a ``content`` field, but the wrapper is the
surface callers actually use — a keyword the wrapper forgets to forward is a
feature that exists only in the spec. These assert on the serialized body rather
than on attributes, because the body is what the server reads: a wrapper that
sets ``content`` on the request object but drops it from ``__properties`` would
pass an attribute check and still send nothing.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight

AUTHORED = "## Preferences\n\n- Prefer typed APIs\n"


def _capture(monkeypatch, api, method, captured):
    async def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(api, method, fake)


def _client() -> Hindsight:
    return Hindsight(base_url="http://example.invalid")


def test_create_mental_model_sends_authored_content(monkeypatch):
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "create_mental_model", captured)

    client.create_mental_model("bank-1", "Team Preferences", "What are the team's preferences?", content=AUTHORED)

    _, request = captured["args"]
    assert request.to_dict()["content"] == AUTHORED
    assert request.to_dict()["source_query"] == "What are the team's preferences?"


def test_create_mental_model_omits_content_when_not_authored(monkeypatch):
    """No content means the server generates it.

    The wrapper passes every parameter explicitly, so the generated model marks
    ``content`` as set and serializes it as ``null`` rather than dropping the key.
    The server reads null exactly as it reads an absent field — ``body.content is
    None`` takes the placeholder-and-refresh branch — so what matters is that no
    *value* travels, not that the key is missing.
    """
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "create_mental_model", captured)

    client.create_mental_model("bank-1", "Team Preferences", "What are the team's preferences?")

    _, request = captured["args"]
    assert request.to_dict().get("content") is None
    assert "source_query" in request.to_dict()


async def test_acreate_mental_model_sends_authored_content(monkeypatch):
    """The async half is the one async frameworks call; it must forward content too."""
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "create_mental_model", captured)

    await client.acreate_mental_model("bank-1", "Team Preferences", "q", content=AUTHORED)

    _, request = captured["args"]
    assert request.to_dict()["content"] == AUTHORED


def test_update_mental_model_sends_authored_content(monkeypatch):
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", content=AUTHORED)

    _, _, request = captured["args"]
    assert request.to_dict()["content"] == AUTHORED


def test_update_mental_model_omits_content_when_not_provided(monkeypatch):
    """Updating a name must not blank the stored content.

    Same null-vs-absent equivalence as create: the engine's ``content=None`` means
    "leave it alone", so an unnamed content field is a no-op either way.
    """
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", name="Renamed")

    _, _, request = captured["args"]
    assert request.to_dict().get("content") is None
    assert request.to_dict()["name"] == "Renamed"


async def test_aupdate_mental_model_sends_authored_content(monkeypatch):
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    await client.aupdate_mental_model("bank-1", "mm-1", content=AUTHORED)

    _, _, request = captured["args"]
    assert request.to_dict()["content"] == AUTHORED


def test_content_and_delta_trigger_travel_together(monkeypatch):
    """The documented way to keep authored content: inject it *and* ask for delta.

    Both halves must survive the same request — content alone is overwritten by the
    next full refresh, and a delta trigger without content edits a placeholder.
    """
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "create_mental_model", captured)

    client.create_mental_model(
        "bank-1",
        "Team Preferences",
        "What are the team's preferences?",
        content=AUTHORED,
        trigger={"mode": "delta"},
    )

    _, request = captured["args"]
    assert request.to_dict()["content"] == AUTHORED
    assert request.trigger.to_dict() == {"mode": "delta"}


def test_create_keeps_the_existing_positional_parameter_order(monkeypatch):
    """Adding ``content`` must not shift the parameters callers already pass positionally.

    ``create_mental_model(bank, name, source_query, ["tag"])`` used to put the list in
    ``tags``. Inserting ``content`` before ``tags`` would silently send the list as the
    authored document instead — a 422 at best, and a wrong mental model at worst.
    """
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "create_mental_model", captured)

    client.create_mental_model("bank-1", "Team Preferences", "q", ["legacy-tag"])

    _, request = captured["args"]
    body = request.to_dict()
    assert body["tags"] == ["legacy-tag"]
    assert body.get("content") is None


def test_update_keeps_the_existing_positional_parameter_order(monkeypatch):
    """Same guard for ``update_mental_model``, whose tail is longer."""
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", "Renamed", "new query", ["tag"], 4096, {"mode": "delta"})

    _, _, request = captured["args"]
    body = request.to_dict()
    assert body["name"] == "Renamed"
    assert body["source_query"] == "new query"
    assert body["tags"] == ["tag"]
    assert body["max_tokens"] == 4096
    assert request.trigger.to_dict() == {"mode": "delta"}
    assert body.get("content") is None

