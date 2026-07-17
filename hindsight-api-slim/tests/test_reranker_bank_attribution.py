"""
Tests for opt-in per-bank attribution on Cohere-compatible remote rerank requests.

Covers the shared `_CohereCompatibleRerankClient` path used by Cohere (base_url),
OpenRouter, ZeroEntropy, SiliconFlow, and Alibaba, plus the `LiteLLMCrossEncoder`
proxy path (httpx POST to a LiteLLM proxy /rerank). Opt-in via
`HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER`; when enabled, outbound rerank posts
carry `X-Hindsight-Bank-Id: <bank_id>`.

All deterministic — no network, stdlib/pytest/httpx only.
"""

import os
from unittest.mock import patch

import httpx
import pytest

from hindsight_api.engine.cross_encoder import LiteLLMCrossEncoder, _CohereCompatibleRerankClient
from hindsight_api.engine.memory_engine import _current_bank_id


@pytest.fixture(autouse=True)
def restore_reranker_send_bank_env():
    """Save/restore the reranker attribution env var and clear the cached config."""
    from hindsight_api.config import clear_config_cache

    original = os.environ.get("HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER")
    clear_config_cache()
    yield
    if original is None:
        os.environ.pop("HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER", None)
    else:
        os.environ["HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER"] = original
    clear_config_cache()


def _set_flag(enabled: bool) -> None:
    from hindsight_api.config import clear_config_cache

    os.environ["HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER"] = "true" if enabled else "false"
    clear_config_cache()


async def _make_initialized_client(transport: httpx.MockTransport) -> _CohereCompatibleRerankClient:
    client = _CohereCompatibleRerankClient(
        api_key="k",
        model="m",
        rerank_url="https://gw.example/rerank",
    )
    # Capture the real class BEFORE patching: the lambda body resolves
    # httpx.AsyncClient at CALL time, so referencing it directly would re-enter
    # the mock and recurse.
    real_async_client = httpx.AsyncClient
    # side_effect (NOT return_value) so initialize()'s real headers=
    # {Authorization, Content-Type} kwarg survives.
    with patch.object(
        httpx,
        "AsyncClient",
        side_effect=lambda **kw: real_async_client(**kw, transport=transport),
    ):
        await client.initialize()
    return client


@pytest.mark.asyncio
async def test_bank_header_sent_when_enabled():
    _set_flag(True)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    token = _current_bank_id.set("bank-42")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert captured["headers"]["X-Hindsight-Bank-Id"] == "bank-42"


@pytest.mark.asyncio
async def test_bank_header_absent_when_flag_off():
    _set_flag(False)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    token = _current_bank_id.set("bank-42")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert "X-Hindsight-Bank-Id" not in captured["headers"]


@pytest.mark.asyncio
async def test_bank_header_absent_when_no_bank_bound():
    _set_flag(True)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    await client.predict([("q", "d")])

    assert "X-Hindsight-Bank-Id" not in captured["headers"]


@pytest.mark.asyncio
async def test_auth_and_content_type_headers_survive_when_enabled():
    _set_flag(True)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    token = _current_bank_id.set("bank-42")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["X-Hindsight-Bank-Id"] == "bank-42"


@pytest.mark.asyncio
async def test_header_is_per_request_not_client_default():
    """Two predict() calls on the same client must each carry their own bank.

    Catches an implementation that pins the bank into client defaults at
    initialize() time and mis-attributes every later bank.
    """
    _set_flag(True)
    seen: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-Hindsight-Bank-Id"))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    token_a = _current_bank_id.set("bank-a")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token_a)

    token_b = _current_bank_id.set("bank-b")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token_b)

    assert seen == ["bank-a", "bank-b"]


@pytest.mark.asyncio
async def test_bank_header_skipped_for_non_ascii_bank_id():
    """Non-ASCII bank ids are skipped (httpx header values must be ASCII) — the
    request still succeeds, just without attribution."""
    _set_flag(True)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_client(transport)

    token = _current_bank_id.set("bänk-中文")
    try:
        scores = await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert scores == [0.9]
    assert "X-Hindsight-Bank-Id" not in captured["headers"]


async def _make_initialized_litellm_client(transport: httpx.MockTransport) -> LiteLLMCrossEncoder:
    client = LiteLLMCrossEncoder(api_base="https://proxy.example", api_key="k", model="m")
    # Same capture-before-patch dance as _make_initialized_client above.
    real_async_client = httpx.AsyncClient
    with patch.object(
        httpx,
        "AsyncClient",
        side_effect=lambda **kw: real_async_client(**kw, transport=transport),
    ):
        await client.initialize()
    return client


@pytest.mark.asyncio
async def test_litellm_proxy_bank_header_sent_when_enabled():
    _set_flag(True)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_litellm_client(transport)

    token = _current_bank_id.set("bank-42")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert captured["headers"]["X-Hindsight-Bank-Id"] == "bank-42"
    # Client-default headers from initialize() must survive the per-request merge.
    assert captured["headers"]["Authorization"] == "Bearer k"


@pytest.mark.asyncio
async def test_litellm_proxy_bank_header_absent_when_flag_off():
    _set_flag(False)
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    transport = httpx.MockTransport(handler)
    client = await _make_initialized_litellm_client(transport)

    token = _current_bank_id.set("bank-42")
    try:
        await client.predict([("q", "d")])
    finally:
        _current_bank_id.reset(token)

    assert "X-Hindsight-Bank-Id" not in captured["headers"]
