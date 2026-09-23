"""The `hindsight-all` namespaces forward mental-model content to the real client.

Both namespaces in this package wrap ``hindsight_client.Hindsight`` and used to be
broken in opposite ways: they passed ``content`` to a method that did not accept
it (``TypeError: unexpected keyword argument``) while never passing the required
``source_query``. Nothing exercised those paths, so nothing noticed.

These tests pin two things at once — that the arguments are forwarded, and that
the method they are forwarded to actually accepts them. Asserting only the former
would pass against a ``Mock`` even if the real signature still rejected them,
which is exactly the bug being guarded.
"""

from __future__ import annotations

import inspect
from unittest.mock import Mock, patch

import pytest
from hindsight_client import Hindsight

from hindsight import HindsightClient, HindsightEmbedded

AUTHORED = "## Preferences\n\n- Prefer typed APIs\n"
SOURCE_QUERY = "What are the team's preferences?"


def _assert_real_signature_accepts(method, kwargs) -> None:
    """Bind the forwarded kwargs against the real method the namespace calls.

    ``self`` is bound as None: binding is a static check of names and
    requirement, which is the part that a stale wrapper signature breaks.
    """
    inspect.signature(method).bind(None, **kwargs)


@pytest.fixture
def embedded() -> HindsightEmbedded:
    client = HindsightEmbedded(profile="test", llm_provider="openai", llm_api_key="test-key")
    # The daemon is not part of what is under test; the namespace calls this first.
    client._ensure_started = Mock()  # type: ignore[method-assign]
    client._client = Mock()
    return client


@pytest.fixture
def manual() -> HindsightClient:
    return HindsightClient(base_url="http://example.invalid")


def test_embedded_create_forwards_source_query_and_content(embedded):
    embedded.mental_models.create(
        bank_id="bank",
        name="Team preferences",
        source_query=SOURCE_QUERY,
        content=AUTHORED,
        tags=["team"],
    )

    kwargs = embedded._client.create_mental_model.call_args.kwargs
    assert kwargs["source_query"] == SOURCE_QUERY
    assert kwargs["content"] == AUTHORED
    _assert_real_signature_accepts(Hindsight.create_mental_model, kwargs)


def test_embedded_create_forwards_a_delta_trigger(embedded):
    """Delta is how a caller keeps authored content; the wrapper must be able to ask for it."""
    embedded.mental_models.create(
        bank_id="bank",
        name="Team preferences",
        source_query=SOURCE_QUERY,
        content=AUTHORED,
        trigger={"mode": "delta"},
    )

    kwargs = embedded._client.create_mental_model.call_args.kwargs
    assert kwargs["trigger"] == {"mode": "delta"}
    _assert_real_signature_accepts(Hindsight.create_mental_model, kwargs)


def test_embedded_create_without_content_still_works(embedded):
    embedded.mental_models.create(bank_id="bank", name="Team preferences", source_query=SOURCE_QUERY)

    kwargs = embedded._client.create_mental_model.call_args.kwargs
    assert kwargs["content"] is None
    _assert_real_signature_accepts(Hindsight.create_mental_model, kwargs)


def test_embedded_update_forwards_content(embedded):
    embedded.mental_models.update(bank_id="bank", mental_model_id="mm-1", content=AUTHORED)

    kwargs = embedded._client.update_mental_model.call_args.kwargs
    assert kwargs["content"] == AUTHORED
    _assert_real_signature_accepts(Hindsight.update_mental_model, kwargs)


def test_embedded_update_forwards_a_delta_trigger(embedded):
    embedded.mental_models.update(
        bank_id="bank", mental_model_id="mm-1", content=AUTHORED, trigger={"mode": "delta"}
    )

    kwargs = embedded._client.update_mental_model.call_args.kwargs
    assert kwargs["trigger"] == {"mode": "delta"}
    _assert_real_signature_accepts(Hindsight.update_mental_model, kwargs)


def test_manual_client_create_forwards_source_query_and_content(manual):
    with patch.object(manual, "create_mental_model") as create:
        manual.mental_models.create(
            bank_id="bank",
            name="Team preferences",
            source_query=SOURCE_QUERY,
            content=AUTHORED,
        )

    kwargs = create.call_args.kwargs
    assert kwargs["source_query"] == SOURCE_QUERY
    assert kwargs["content"] == AUTHORED
    _assert_real_signature_accepts(Hindsight.create_mental_model, kwargs)


def test_manual_client_update_forwards_content(manual):
    with patch.object(manual, "update_mental_model") as update:
        manual.mental_models.update(bank_id="bank", mental_model_id="mm-1", content=AUTHORED)

    kwargs = update.call_args.kwargs
    assert kwargs["content"] == AUTHORED
    _assert_real_signature_accepts(Hindsight.update_mental_model, kwargs)


def test_creating_a_mental_model_no_longer_raises_type_error(embedded):
    """The regression itself, against the real client rather than a Mock.

    The namespace is wired to a genuine ``Hindsight`` whose HTTP layer is stubbed
    out, so a keyword the real method does not accept fails here as a TypeError
    instead of at a user's first call.
    """
    real = Hindsight(base_url="http://example.invalid")
    recorded: dict[str, object] = {}

    async def fake_api(bank_id, request, **kwargs):
        recorded["request"] = request
        return Mock()

    real._mental_models_api.create_mental_model = fake_api  # type: ignore[method-assign]
    embedded._client = real

    embedded.mental_models.create(
        bank_id="bank",
        name="Team preferences",
        source_query=SOURCE_QUERY,
        content=AUTHORED,
    )

    request = recorded["request"]
    assert request.to_dict()["content"] == AUTHORED
    assert request.to_dict()["source_query"] == SOURCE_QUERY
