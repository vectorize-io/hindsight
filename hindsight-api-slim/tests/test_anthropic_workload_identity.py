"""Issue #5502: with Workload Identity Federation configured, the Anthropic provider
runs without a static key and the SDK swaps the platform identity token for a
short-lived access token at /v1/oauth/token."""

import pytest
from aiohttp import web

from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key

from tests.aiohttp_stub import stub_server

_WIF_ENV = ("ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE")


@pytest.fixture
def no_anthropic_env(monkeypatch):
    for name in (*_WIF_ENV, "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def test_key_still_required_without_federation(no_anthropic_env):
    assert requires_api_key("anthropic") is True
    with pytest.raises(ValueError, match="API key is required"):
        create_llm_provider(
            provider="anthropic", api_key="", base_url="", model="claude-haiku-4-5", reasoning_effort=None
        )


@pytest.mark.asyncio
async def test_federation_exchanges_identity_token(no_anthropic_env, monkeypatch, tmp_path):
    seen: dict = {}

    async def handler(request: web.Request) -> web.Response:
        if request.path == "/v1/oauth/token":
            seen["exchange"] = await request.json()
            return web.json_response({"access_token": "short-lived", "token_type": "Bearer", "expires_in": 600})
        seen["authorization"] = request.headers.get("Authorization")
        seen["x_api_key"] = request.headers.get("x-api-key")
        return web.json_response(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    token_file = tmp_path / "token"
    token_file.write_text("pod-identity-jwt")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_1")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_1")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))
    assert requires_api_key("anthropic") is False

    async with stub_server(handler) as base_url:
        llm = create_llm_provider(
            provider="anthropic", api_key="", base_url=base_url, model="claude-haiku-4-5", reasoning_effort=None
        )
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_completion_tokens=10, scope="test")

    assert result.content == "ok"
    assert seen["exchange"]["assertion"] == "pod-identity-jwt"
    assert seen["exchange"]["federation_rule_id"] == "fdrl_1"
    assert seen["authorization"] == "Bearer short-lived"
    assert seen["x_api_key"] is None
