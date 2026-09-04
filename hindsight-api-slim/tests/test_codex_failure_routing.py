"""Real Codex/wrapper/router calls with synthetic auth files and mocked HTTP only."""

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from hindsight_api.config import LLMStrategyConfig
from hindsight_api.engine.llm_interface import ProviderReauthenticationRequiredError
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.providers.codex_llm import CodexLLM


def write_auth(path: Path, *, expired: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + (-120 if expired else 3600)}).encode())
    (path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": f"header.{payload.decode().rstrip('=')}.signature",
                    "refresh_token": "synthetic-refresh",
                    "account_id": "synthetic-account",
                },
            }
        )
    )
    (path / "auth.json").chmod(0o600)


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
@pytest.mark.parametrize("expired", [False, True])
@pytest.mark.parametrize("code", ["refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated"])
async def test_confirmed_refresh_failure_stops_direct_and_routed_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, expired: bool, code: str
) -> None:
    auth = tmp_path / "synthetic-profile"
    write_auth(auth, expired=expired)
    attempts: list[str] = []

    def refresh(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        attempts.append("refresh")
        return httpx.Response(401, json={"error": {"code": code}}, request=request)

    async def backend(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        attempts.append("backend")
        return httpx.Response(401, json={"error": "rejected"}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", backend)
    monkeypatch.setattr(httpx.Client, "send", refresh)
    primary = LLMProvider(
        provider="openai-codex", api_key="", base_url="", model="test", codex_home=str(auth), member_label="work"
    )
    fallback = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "synthetic"}], "max_retries": 3}
    if method == "call_with_tools":
        kwargs["tools"] = []
    try:
        for provider in (primary, router):
            attempts.clear()
            with pytest.raises(ProviderReauthenticationRequiredError, match="work") as caught:
                await getattr(provider, method)(**kwargs)
            assert str(auth) not in str(caught.value)
            assert attempts == (["refresh"] if expired else ["backend", "refresh"])
    finally:
        await router.cleanup()


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
@pytest.mark.parametrize("refresh_status", [200, 401, 503])
async def test_unconfirmed_auth_failure_keeps_generic_fallback_without_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, refresh_status: int
) -> None:
    write_auth(tmp_path)
    backend_attempts = 0

    async def backend(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        nonlocal backend_attempts
        backend_attempts += 1
        return httpx.Response(401, json={"error": "access rejected"}, request=request)

    def refresh(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        body = {"access_token": "synthetic-replacement"} if refresh_status == 200 else {"error": {"code": "unknown"}}
        return httpx.Response(refresh_status, json=body, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", backend)
    monkeypatch.setattr(httpx.Client, "send", refresh)
    primary = LLMProvider(provider="openai-codex", api_key="", base_url="", model="test", codex_home=str(tmp_path))
    fallback = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    fallback.set_mock_response("fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "synthetic"}], "max_retries": 0}
    if method == "call_with_tools":
        kwargs["tools"] = []
    try:
        for _ in range(2):
            result = await getattr(router, method)(**kwargs)
            assert (result if isinstance(result, str) else result.content) == (
                "fallback" if method == "call" else "mock response"
            )
        assert backend_attempts == (4 if refresh_status == 200 else 2)
    finally:
        await router.cleanup()


@pytest.mark.parametrize("method,expected_attempts", [("call", 2), ("call_with_tools", 1)])
async def test_real_codex_quota_preserves_retry_budget_then_cools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, expected_attempts: int
) -> None:
    write_auth(tmp_path)
    now = 100.0
    attempts = 0

    async def backend(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if now < 110:
            return httpx.Response(429, headers={"Retry-After": "10"}, request=request)
        return httpx.Response(200, text='event: response.text.delta\ndata: {"delta":"primary"}\n\n', request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", backend)
    primary = LLMProvider(provider="openai-codex", api_key="", base_url="", model="test", codex_home=str(tmp_path))
    fallback = LLMProvider(provider="mock", api_key="", base_url="", model="mock")
    fallback.set_mock_response("fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": "synthetic"}],
        "max_retries": 1,
        "initial_backoff": 0,
    }
    if method == "call_with_tools":
        kwargs["tools"] = []
    try:
        for _ in range(2):
            result = await getattr(router, method)(**kwargs)
            assert (result if isinstance(result, str) else result.content) == (
                "fallback" if method == "call" else "mock response"
            )
        assert attempts == expected_attempts
        now = 110.0
        result = await getattr(router, method)(**kwargs)
        assert (result if isinstance(result, str) else result.content) == "primary"
        assert attempts == expected_attempts + 1
    finally:
        await router.cleanup()


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_terminal_refresh_response_adopts_newer_fresh_auth_before_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    write_auth(tmp_path)
    attempts: list[str] = []

    async def backend(self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        attempts.append("backend")
        if len(attempts) == 1:
            return httpx.Response(401, request=request)
        assert "replacement" in request.headers["Authorization"]
        return httpx.Response(200, text='event: response.text.delta\ndata: {"delta":"recovered"}\n\n', request=request)

    def refresh(self: httpx.Client, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        attempts.append("refresh")
        body = json.loads((tmp_path / "auth.json").read_text())
        body["tokens"]["access_token"] += "replacement"
        body["tokens"]["refresh_token"] = "synthetic-new-refresh"
        (tmp_path / "auth.json").write_text(json.dumps(body))
        return httpx.Response(401, json={"error": {"code": "refresh_token_reused"}}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", backend)
    monkeypatch.setattr(httpx.Client, "send", refresh)
    provider = CodexLLM(provider="openai-codex", api_key="", base_url="", model="test", codex_home=str(tmp_path))
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "synthetic"}], "max_retries": 0}
    if method == "call_with_tools":
        kwargs["tools"] = []
    try:
        result = await getattr(provider, method)(**kwargs)
        assert (result if isinstance(result, str) else result.content) == "recovered"
        assert attempts == ["backend", "refresh", "backend"]
    finally:
        await provider.cleanup()
