"""The maintained wrapper forwards directive list pagination to the SDK.

Mirrors the TypeScript wrapper's ``directive_query_mapping`` regression tests.
The directive list endpoint reports a ``total``, so dropping ``limit``/``offset``
in the wrapper would leave SDK callers stuck on the server's first page.
"""

from unittest.mock import MagicMock

from hindsight_client import Hindsight


def _capture_list(monkeypatch, client, captured):
    async def fake_list(bank_id, **kwargs):
        captured["bank_id"] = bank_id
        captured["kwargs"] = kwargs
        return MagicMock(items=[], total=0)

    monkeypatch.setattr(client._directives_api, "list_directives", fake_list)


def test_list_forwards_every_supported_query_option(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_list(monkeypatch, client, captured)

    client.list_directives("bank-1", tags=["project"], limit=25, offset=50)

    assert captured["bank_id"] == "bank-1"
    kwargs = captured["kwargs"]
    assert kwargs["tags"] == ["project"]
    assert kwargs["limit"] == 25
    assert kwargs["offset"] == 50


def test_list_defaults_leave_controls_unset(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    captured: dict[str, object] = {}
    _capture_list(monkeypatch, client, captured)

    client.list_directives("bank-1")

    kwargs = captured["kwargs"]
    # Nothing forced on: the server keeps its own defaults for every control.
    assert kwargs["tags"] is None
    assert kwargs["limit"] is None
    assert kwargs["offset"] is None
