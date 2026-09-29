"""The maintained wrapper forwards mental-model query controls to the SDK.

Mirrors the TypeScript wrapper's ``mental_model_query_mapping`` regression tests
(#2975 / #3042) so a refactor cannot silently restore the server's ``detail=full``
default or drop list pagination / tag-match controls.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_list(monkeypatch, client, captured):
    async def fake_list(bank_id, **kwargs):
        captured["bank_id"] = bank_id
        captured["kwargs"] = kwargs
        return MagicMock(items=[])

    monkeypatch.setattr(client._mental_models_api, "list_mental_models", fake_list)


def _capture_get(monkeypatch, client, captured):
    async def fake_get(bank_id, mental_model_id, **kwargs):
        captured["bank_id"] = bank_id
        captured["mental_model_id"] = mental_model_id
        captured["kwargs"] = kwargs
        return MagicMock()

    monkeypatch.setattr(client._mental_models_api, "get_mental_model", fake_get)


def test_list_forwards_every_supported_query_option(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_list(monkeypatch, client, captured)

    client.list_mental_models(
        "bank-1",
        tags=["project"],
        tags_match="exact",
        detail="metadata",
        limit=25,
        offset=50,
    )

    assert captured["bank_id"] == "bank-1"
    kwargs = captured["kwargs"]
    assert kwargs["tags"] == ["project"]
    assert kwargs["tags_match"] == "exact"
    assert kwargs["detail"] == "metadata"
    assert kwargs["limit"] == 25
    assert kwargs["offset"] == 50


def test_list_defaults_leave_controls_unset(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_list(monkeypatch, client, captured)

    client.list_mental_models("bank-1")

    kwargs = captured["kwargs"]
    # Nothing forced on: the server keeps its own defaults for every control.
    assert kwargs["tags"] is None
    assert kwargs["tags_match"] is None
    assert kwargs["detail"] is None
    assert kwargs["limit"] is None
    assert kwargs["offset"] is None


def test_list_tags_only_shape_preserved(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_list(monkeypatch, client, captured)

    client.list_mental_models("bank-1", tags=["project"])

    kwargs = captured["kwargs"]
    assert kwargs["tags"] == ["project"]
    assert kwargs["detail"] is None


def test_get_forwards_detail(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_get(monkeypatch, client, captured)

    client.get_mental_model("bank-1", "model-1", detail="content")

    assert captured["bank_id"] == "bank-1"
    assert captured["mental_model_id"] == "model-1"
    assert captured["kwargs"]["detail"] == "content"


def test_get_defaults_leave_detail_unset(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_get(monkeypatch, client, captured)

    client.get_mental_model("bank-1", "model-1")

    assert captured["kwargs"]["detail"] is None
