"""Tests for Nous provider wiring into the LLM factory and the NousLLM subclass.

These exercise the integration surface without a Hermes install: the auth
manager is stubbed so we can assert base-url handling, the no-api-key contract,
provider validation, and the proactive/reactive token-refresh plumbing on
``NousLLM`` (rebuild client on rotation; one reactive refresh on a 401).
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError

from hindsight_api.engine.llm_wrapper import requires_api_key
from hindsight_api.engine.providers.nous_auth import NousAuthManager
from hindsight_api.engine.providers.nous_llm import NousLLM


class _FakeAuth:
    """Stand-in for NousAuthManager with a controllable token + refresh."""

    def __init__(self, token: str = "tok-1") -> None:
        self.access_token = token
        self.base_url = "https://inference-api.nousresearch.com/v1"
        self.stale = False
        self.refresh_calls: list[tuple[str, bool]] = []
        self.next_token = token

    def _token_is_stale(self) -> bool:
        return self.stale

    def refresh_tokens(self, reason: str = "", *, force: bool = False) -> None:
        self.refresh_calls.append((reason, force))
        self.access_token = self.next_token
        self.stale = False

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _make(base_url: str = "", auth: _FakeAuth | None = None) -> NousLLM:
    fake = auth or _FakeAuth()
    with patch.object(NousAuthManager, "from_file", return_value=fake):
        return NousLLM(provider="nous", api_key="ignored", base_url=base_url, model="deepseek/deepseek-v4-flash")


# ---------------------------------------------------------------------------
# Factory contract
# ---------------------------------------------------------------------------


def test_nous_does_not_require_api_key() -> None:
    assert requires_api_key("nous") is False


def test_nous_default_base_url_when_empty() -> None:
    assert _make("").base_url == "https://inference-api.nousresearch.com/v1"


def test_nous_respects_explicit_base_url() -> None:
    llm = _make("https://staging-inference.example.com/v1")
    assert llm.base_url == "https://staging-inference.example.com/v1"


def test_nous_uses_loaded_token_as_api_key() -> None:
    llm = _make(auth=_FakeAuth(token="tok-loaded"))
    assert llm.api_key == "tok-loaded"


# ---------------------------------------------------------------------------
# Token refresh plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_fresh_token_rebuilds_client_on_rotation() -> None:
    auth = _FakeAuth(token="tok-1")
    llm = _make(auth=auth)
    original_client = llm._client

    auth.stale = True
    auth.next_token = "tok-2"
    await llm._ensure_fresh_token()

    assert auth.refresh_calls == [("proactive (token near expiry)", False)]
    assert llm.api_key == "tok-2"
    assert llm._client is not original_client  # client rebuilt with the new token


@pytest.mark.asyncio
async def test_fresh_token_skips_refresh() -> None:
    auth = _FakeAuth(token="tok-1")
    llm = _make(auth=auth)
    await llm._ensure_fresh_token()
    assert auth.refresh_calls == []


@pytest.mark.asyncio
async def test_call_refreshes_once_on_401_then_retries() -> None:
    auth = _FakeAuth(token="tok-1")
    llm = _make(auth=auth)
    auth.next_token = "tok-2"

    calls = {"n": 0}

    async def fake_super_call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIStatusError(
                "unauthorized",
                response=httpx.Response(401, request=httpx.Request("POST", "http://x")),
                body=None,
            )
        return "ok"

    with patch.object(NousLLM.__bases__[0], "call", side_effect=fake_super_call, autospec=False):
        result = await llm.call(messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert calls["n"] == 2  # original + one retry
    assert auth.refresh_calls[-1][1] is True  # forced refresh on the 401 path
    assert llm.api_key == "tok-2"
