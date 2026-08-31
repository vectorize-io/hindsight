"""The maintained wrapper builds correct knowledge-base requests.

Mirrors the TypeScript wrapper's ``knowledge_base_mapping`` tests so the two
hand-written clients stay at parity. The interesting case is
``update_knowledge_node``: the server distinguishes "field not provided" from
``parent_id: null`` (which means "move to the root"), so an omitted argument
must not be serialized at all.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture(monkeypatch, client, method, captured):
    async def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(client._knowledge_base_api, method, fake)


def test_create_page_maps_every_option(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "create_knowledge_page", captured)

    client.create_knowledge_page(
        "bank-1",
        name="Deploying the API",
        source_query="How is the API deployed?",
        parent_id="kf-1",
        tags=["ops", "type:runbook"],
        max_tokens=8192,
    )

    bank_id, request = captured["args"]
    assert bank_id == "bank-1"
    assert request.name == "Deploying the API"
    assert request.source_query == "How is the API deployed?"
    assert request.parent_id == "kf-1"
    assert request.tags == ["ops", "type:runbook"]
    assert request.max_tokens == 8192
    # Omitting trigger must leave the server-side page defaults in place.
    assert request.trigger is None


def test_create_page_threads_trigger_fields(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "create_knowledge_page", captured)

    client.create_knowledge_page(
        "bank-1",
        name="Page",
        source_query="q",
        trigger={
            "mode": "delta",
            "refresh_after_consolidation": True,
            "fact_types": ["observation"],
            "exclude_mental_models": True,
        },
    )

    _, request = captured["args"]
    assert request.trigger.mode == "delta"
    assert request.trigger.refresh_after_consolidation is True
    assert request.trigger.fact_types == ["observation"]
    assert request.trigger.exclude_mental_models is True


def test_update_node_sends_only_provided_fields(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "update_knowledge_node", captured)

    client.update_knowledge_node("bank-1", "kp-1", name="Renamed")

    _, node_id, request = captured["args"]
    assert node_id == "kp-1"
    assert request.name == "Renamed"
    assert "parent_id" not in request.model_fields_set


def test_update_node_sends_explicit_null_parent(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "update_knowledge_node", captured)

    client.update_knowledge_node("bank-1", "kp-1", parent_id=None)

    _, _, request = captured["args"]
    assert "parent_id" in request.model_fields_set
    assert request.parent_id is None


def test_update_node_maps_page_options(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "update_knowledge_node", captured)

    client.update_knowledge_node("bank-1", "kp-1", source_query="New question?", tags=[], max_tokens=2048)

    _, _, request = captured["args"]
    assert request.source_query == "New question?"
    assert request.tags == []
    assert request.max_tokens == 2048
    assert "trigger" not in request.model_fields_set


def test_update_node_forwards_a_partial_trigger(monkeypatch):
    """The trigger is a PATCH server-side: unsent fields keep the page's current values,
    so the wrapper must pass exactly what the caller gave it — and a wrapper that dropped
    the field would leave callers unable to change a refresh policy at all.

    Asserted on the serialized body, not on ``model_fields_set``: the server reads
    which fields the REQUEST carried, and the generated model's ``to_dict`` keeps a
    non-None default even for a field nobody set. Passing the dict straight through
    therefore shipped ``mode="full"``, ``refresh_after_consolidation``,
    ``exclude_mental_models`` and ``keep_trace`` alongside the one setting asked for,
    resetting the page to a from-scratch rebuild over its sibling pages — #3506's bug,
    one layer down.
    """
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "update_knowledge_node", captured)

    client.update_knowledge_node("bank-1", "kp-1", trigger={"refresh_cron": "0 3 * * *"})

    _, _, request = captured["args"]
    assert request.trigger.refresh_cron == "0 3 * * *"
    assert request.trigger.to_dict() == {"refresh_cron": "0 3 * * *"}


def test_search_forwards_query_and_limit(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "search_knowledge_base", captured)

    client.search_knowledge_base("bank-1", "how do we deploy", limit=5)

    bank_id, query = captured["args"]
    assert bank_id == "bank-1"
    assert query == "how do we deploy"
    assert captured["kwargs"]["limit"] == 5


def test_create_folder_maps_parent(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture(monkeypatch, client, "create_knowledge_folder", captured)

    client.create_knowledge_folder("bank-1", "Operations", parent_id=None)

    bank_id, request = captured["args"]
    assert bank_id == "bank-1"
    assert request.name == "Operations"
    assert request.parent_id is None
