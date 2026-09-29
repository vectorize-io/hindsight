"""A wrapper-supplied trigger carries only the settings the caller named.

Every route that takes a trigger patches it over what is already stored — the
page defaults on page creation, the model's current trigger on either update —
and decides what "named" means from the fields the request actually carried
(``model_dump(exclude_unset=True)``, #3506/#3549). The generated model undoes
that by itself: ``to_dict`` drops ``None`` but keeps a non-None default, so a
caller asking for one setting also shipped ``mode="full"``,
``refresh_after_consolidation=False``, ``exclude_mental_models=False`` and
``keep_trace=False``, silently resetting the four settings they never mentioned.

These assert on the serialized body rather than on attributes: an attribute is
right either way, and it is the wire payload the server reads.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight

# The fields whose model default is not None — the ones that leaked into every
# request before the wrapper started blanking them.
DEFAULTED_FIELDS = {"mode", "refresh_after_consolidation", "exclude_mental_models", "keep_trace"}


def _capture(monkeypatch, api, method, captured):
    async def fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(api, method, fake)


def _client() -> Hindsight:
    return Hindsight(base_url="http://example.invalid")


def test_update_mental_model_sends_only_the_named_setting(monkeypatch):
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", trigger={"mode": "delta"})

    _, _, request = captured["args"]
    assert request.trigger.to_dict() == {"mode": "delta"}


def test_update_mental_model_keeps_an_explicit_false(monkeypatch):
    """A caller turning a setting OFF must still be heard: False is a value, not an omission."""
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", trigger={"refresh_after_consolidation": False})

    _, _, request = captured["args"]
    assert request.trigger.to_dict() == {"refresh_after_consolidation": False}


def test_update_mental_model_clears_a_nullable_field(monkeypatch):
    """An explicit None on a nullable field still serializes as null, which is how a
    stored cron schedule is removed — blanking the unnamed defaults must not break it."""
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", trigger={"refresh_cron": None, "keep_trace": True})

    _, _, request = captured["args"]
    assert request.trigger.to_dict() == {"refresh_cron": None, "keep_trace": True}


def test_create_knowledge_page_keeps_the_page_defaults(monkeypatch):
    """Page creation patches over KNOWLEDGE_PAGE_DEFAULT_TRIGGER (delta, observation-only,
    exclude_mental_models). A trigger naming one setting used to carry mode="full" with it
    and turn a page into a from-scratch rebuild that reflected over its siblings."""
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._knowledge_base_api, "create_knowledge_page", captured)

    client.create_knowledge_page("bank-1", name="Page", source_query="q", trigger={"tags_match": "all"})

    _, request = captured["args"]
    assert request.trigger.to_dict() == {"tags_match": "all"}


@pytest.mark.parametrize("field", sorted(DEFAULTED_FIELDS))
def test_a_defaulted_field_is_sent_only_when_named(monkeypatch, field):
    """Guards the whole family: each defaulted field must be absent unless asked for.

    Parametrized rather than spelled out per field so a regenerated client that adds
    another defaulted setting is covered by extending DEFAULTED_FIELDS alone.
    """
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", trigger={"tags_match": "any"})
    _, _, request = captured["args"]
    assert field not in request.trigger.to_dict()


def test_no_trigger_sends_no_trigger(monkeypatch):
    client = _client()
    captured: dict[str, object] = {}
    _capture(monkeypatch, client._mental_models_api, "update_mental_model", captured)

    client.update_mental_model("bank-1", "mm-1", name="Renamed")

    _, _, request = captured["args"]
    assert request.trigger is None
