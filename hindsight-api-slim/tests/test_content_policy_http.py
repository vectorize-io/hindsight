"""A provider's content-policy refusal answers 422 on every route, never a 500.

A refusal is a deterministic verdict on the content the caller sent: the same input earns the same
refusal on every attempt (#3690). Every route's catch-all answered it with a 500, which tells a client
the server is broken and invites a retry that can only fail again; it also counts toward 5xx error-rate
alerting, so one caller sending refused content looks like an outage.
"""

import logging

import httpx
import pytest

from hindsight_api.api.http import _content_refused, _expected_failure, _internal_error
from hindsight_api.engine.llm_interface import ProviderContentPolicyError


def _wrapped() -> RuntimeError:
    """The shape a route sees: the provider's refusal, wrapped by the engine on the way up."""
    try:
        try:
            raise ProviderContentPolicyError("Gemini blocked the request (PROHIBITED_CONTENT); not retrying")
        except ProviderContentPolicyError as inner:
            raise ProviderContentPolicyError(
                "Fact extraction refused by provider content policy: 1 of 1 failed chunks (1 total) were refused"
            ) from inner
    except ProviderContentPolicyError as outer:
        try:
            raise RuntimeError("retain failed") from outer
        except RuntimeError as top:
            return top


def test_a_wrapped_refusal_is_a_422_naming_the_refusal():
    err = _content_refused(_wrapped())
    assert err is not None
    assert err.status_code == 422
    assert "refused by provider content policy" in err.detail


def test_any_other_failure_is_not_a_refusal():
    assert _content_refused(RuntimeError("boom")) is None


def test_the_shared_catch_all_answers_422_and_does_not_log_an_error(caplog):
    with caplog.at_level(logging.INFO, logger="hindsight_api.api.http"):
        try:
            raise _wrapped()
        except RuntimeError as e:
            result = _internal_error(e, "POST /v1/default/banks/b/memories (retain)")
    assert result.status_code == 422
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], "a refusal is not a server error"
    assert not [r for r in caplog.records if "Traceback" in r.getMessage()]


def test_catch_alls_that_build_their_own_500_also_answer_422():
    err = _expected_failure(_wrapped())
    assert err is not None and err.status_code == 422


@pytest.mark.asyncio
async def test_retain_and_reflect_answer_422_when_the_provider_refuses(memory, monkeypatch):
    """Through the real routes."""
    from hindsight_api.api import create_app

    async def refused(*args, **kwargs):
        raise _wrapped()

    monkeypatch.setattr(memory, "retain_batch_async", refused)
    monkeypatch.setattr(memory, "reflect_async", refused)
    app = create_app(memory, initialize_memory=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        retain = await client.post(
            "/v1/default/banks/refused-test/memories",
            json={"items": [{"content": "a fact"}], "async": False},
        )
        reflect = await client.post(
            "/v1/default/banks/refused-test/reflect",
            json={"query": "anything"},
        )

    for response in (retain, reflect):
        assert response.status_code == 422, response.text
        assert "content policy" in response.json()["detail"]
