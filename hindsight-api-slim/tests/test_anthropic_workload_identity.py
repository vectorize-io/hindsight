"""Issue #5502: with Workload Identity Federation configured, the Anthropic provider
runs without a static key and the SDK swaps the platform identity token for a
short-lived access token at /v1/oauth/token."""

import pytest
from aiohttp import web

from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key

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

    async def token(request: web.Request) -> web.Response:
        seen["exchange"] = await request.json()
        return web.json_response({"access_token": "short-lived", "token_type": "Bearer", "expires_in": 600})

    async def messages(request: web.Request) -> web.Response:
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

    app = web.Application()
    app.router.add_post("/v1/oauth/token", token)
    app.router.add_post("/v1/messages", messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    token_file = tmp_path / "token"
    token_file.write_text("pod-identity-jwt")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_1")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_1")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(token_file))

    try:
        assert requires_api_key("anthropic") is False
        llm = create_llm_provider(
            provider="anthropic",
            api_key="",
            base_url=f"http://127.0.0.1:{port}",
            model="claude-haiku-4-5",
            reasoning_effort=None,
        )
        result = await llm.call(messages=[{"role": "user", "content": "hi"}], max_completion_tokens=10, scope="test")
    finally:
        await runner.cleanup()

    assert result.content == "ok"
    assert seen["exchange"]["assertion"] == "pod-identity-jwt"
    assert seen["exchange"]["federation_rule_id"] == "fdrl_1"
    assert seen["authorization"] == "Bearer short-lived"
    assert seen["x_api_key"] is None
