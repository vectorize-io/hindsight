"""Unit tests for GeminiCacheManager.

The SDK's caches.create call is replaced with a fake throughout — no
network, no real Gemini calls. We assert:

* Identical prefixes return identical fingerprints (cache hits).
* Different prefixes return different fingerprints.
* The first get_or_create for a fingerprint creates; the second within
  the TTL window reuses without calling the SDK again.
* "minimum token count" errors from Gemini surface as ``None`` (soft
  fallback), not exceptions.
* Other SDK errors also surface as ``None`` so callers don't crash on
  transient creation failures.
* The TTL refresh boundary recreates after the safety margin elapses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager


def _make_client(create_side_effect=None):
    """Build a fake Gemini client whose ``aio.caches.create`` returns
    a SimpleNamespace with ``.name`` (or raises the given exception)."""
    create_mock = AsyncMock()
    if isinstance(create_side_effect, Exception):
        create_mock.side_effect = create_side_effect
    elif callable(create_side_effect):
        create_mock.side_effect = create_side_effect
    else:
        create_mock.return_value = SimpleNamespace(name="cachedContents/test-cache-name-001")

    client = MagicMock()
    client.aio = MagicMock()
    client.aio.caches = MagicMock()
    client.aio.caches.create = create_mock
    return client, create_mock


# ---- Fingerprint properties ----------------------------------------------


def test_fingerprint_is_stable():
    fp1 = GeminiCacheManager.fingerprint(
        model="gemini-3.1-flash-lite",
        system_instruction="Extract facts.",
        response_schema=None,
    )
    fp2 = GeminiCacheManager.fingerprint(
        model="gemini-3.1-flash-lite",
        system_instruction="Extract facts.",
        response_schema=None,
    )
    assert fp1 == fp2


def test_fingerprint_changes_with_model():
    fp1 = GeminiCacheManager.fingerprint("gemini-3.1-flash-lite", "X", None)
    fp2 = GeminiCacheManager.fingerprint("gemini-3.1-flash", "X", None)
    assert fp1 != fp2


def test_fingerprint_changes_with_system_instruction():
    fp1 = GeminiCacheManager.fingerprint("m", "Extract facts.", None)
    fp2 = GeminiCacheManager.fingerprint("m", "Extract entities.", None)
    assert fp1 != fp2


def test_fingerprint_handles_pydantic_schema():
    """Two equivalent Pydantic schemas should fingerprint identically;
    a different shape should not."""
    from pydantic import BaseModel

    class A(BaseModel):
        x: int
        y: str

    class A_dup(BaseModel):
        x: int
        y: str

    class B(BaseModel):
        x: int
        y: int  # different type

    fp_a = GeminiCacheManager.fingerprint("m", "p", A)
    fp_dup = GeminiCacheManager.fingerprint("m", "p", A_dup)
    fp_b = GeminiCacheManager.fingerprint("m", "p", B)

    # A and A_dup have the same JSON schema, even though they're distinct classes.
    assert fp_a == fp_dup
    assert fp_a != fp_b


# ---- get_or_create lifecycle ---------------------------------------------


@pytest.mark.asyncio
async def test_first_call_creates_subsequent_reuses():
    client, create_mock = _make_client()
    mgr = GeminiCacheManager(client)

    name1 = await mgr.get_or_create(
        model="gemini-3.1-flash-lite",
        system_instruction="Extract facts.",
        response_schema=None,
    )
    name2 = await mgr.get_or_create(
        model="gemini-3.1-flash-lite",
        system_instruction="Extract facts.",
        response_schema=None,
    )

    assert name1 == "cachedContents/test-cache-name-001"
    assert name2 == name1
    # Only ONE underlying create call — second was served from in-memory cache.
    assert create_mock.call_count == 1


@pytest.mark.asyncio
async def test_different_prefixes_create_separately():
    client, create_mock = _make_client(
        create_side_effect=lambda *a, **kw: SimpleNamespace(name=f"cachedContents/created-{create_mock.call_count}")
    )
    mgr = GeminiCacheManager(client)

    name_a = await mgr.get_or_create(model="m", system_instruction="A", response_schema=None)
    name_b = await mgr.get_or_create(model="m", system_instruction="B", response_schema=None)
    assert name_a != name_b
    assert create_mock.call_count == 2


# ---- Failure / fallback handling -----------------------------------------


@pytest.mark.asyncio
async def test_minimum_token_count_error_returns_none():
    """When the prefix is too short, Gemini rejects with a 'minimum
    token count' style message. Manager must surface this as None so
    the caller transparently falls back to a non-cached call."""
    err = Exception("Cached content must have at least 1024 input tokens (minimum)")
    client, _ = _make_client(create_side_effect=err)
    mgr = GeminiCacheManager(client)

    result = await mgr.get_or_create(model="m", system_instruction="tiny", response_schema=None)
    assert result is None


@pytest.mark.asyncio
async def test_other_sdk_errors_also_return_none():
    """Transient errors (rate limits, 5xx, etc.) should fail soft so
    a single bad create doesn't crash every retain call."""
    err = RuntimeError("transient backend error 503")
    client, _ = _make_client(create_side_effect=err)
    mgr = GeminiCacheManager(client)

    result = await mgr.get_or_create(model="m", system_instruction="ok-sized prefix", response_schema=None)
    assert result is None


@pytest.mark.asyncio
async def test_failed_create_does_not_poison_cache():
    """If create fails on first attempt, a retry should call create
    again instead of returning a stale/None entry."""
    call_log = []

    async def maybe_fail(*args, **kwargs):
        call_log.append(1)
        if len(call_log) == 1:
            raise RuntimeError("first call fails")
        return SimpleNamespace(name="cachedContents/recovered")

    client = MagicMock()
    client.aio = MagicMock()
    client.aio.caches = MagicMock()
    client.aio.caches.create = maybe_fail

    mgr = GeminiCacheManager(client)

    first = await mgr.get_or_create(model="m", system_instruction="prefix", response_schema=None)
    second = await mgr.get_or_create(model="m", system_instruction="prefix", response_schema=None)

    assert first is None
    assert second == "cachedContents/recovered"
    assert len(call_log) == 2


# ---- TTL behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_refreshes_after_ttl_margin(monkeypatch):
    """An entry created at t=0 with ttl=10 and margin=2 should be
    treated as stale at t>=8 and trigger a recreate."""
    client, create_mock = _make_client(
        create_side_effect=lambda *a, **kw: SimpleNamespace(name=f"cachedContents/v{create_mock.call_count}")
    )
    mgr = GeminiCacheManager(client, ttl_seconds=10, refresh_margin_seconds=2)

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(
        "hindsight_api.engine.providers.gemini_cache.time.monotonic",
        lambda: fake_now["t"],
    )

    first = await mgr.get_or_create(model="m", system_instruction="p", response_schema=None)
    assert first == "cachedContents/v1"

    # Advance to just before the refresh boundary — should reuse.
    fake_now["t"] = 1000.0 + 7.0
    again = await mgr.get_or_create(model="m", system_instruction="p", response_schema=None)
    assert again == "cachedContents/v1"
    assert create_mock.call_count == 1

    # Advance past the refresh boundary — should recreate.
    fake_now["t"] = 1000.0 + 9.0
    refreshed = await mgr.get_or_create(model="m", system_instruction="p", response_schema=None)
    assert refreshed == "cachedContents/v2"
    assert create_mock.call_count == 2


# ---- Integration: feature flag + GeminiLLM accessor ----------------------


@pytest.mark.asyncio
async def test_gemini_llm_returns_none_when_cache_disabled():
    """A directly-constructed GeminiLLM (no prompt_cache_enabled kwarg) does not
    cache: ``get_or_create_cached_prefix`` returns None without ever building a
    cache manager. The server-level default-on flows in via the kwarg (resolved
    from config in LLMProvider), not via this constructor default."""
    from hindsight_api.engine.providers.gemini_llm import GeminiLLM

    llm = GeminiLLM(
        provider="gemini",
        api_key="not-real-key",
        base_url="",
        model="gemini-test",
    )
    # Constructor default is off; even with a stable prefix the cache stays disabled.
    result = await llm.get_or_create_cached_prefix(
        system_instruction="A reasonably long system prompt " * 50,
        response_schema=None,
    )
    assert result is None
    assert llm._cache_manager is None  # never built


@pytest.mark.asyncio
async def test_gemini_llm_uses_cache_when_enabled(monkeypatch):
    """When the flag is on, the manager is constructed lazily and its
    get_or_create is delegated to. We don't hit the real SDK; we replace
    the client's caches.create with a fake."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from hindsight_api.engine.providers.gemini_llm import GeminiLLM

    llm = GeminiLLM(
        provider="gemini",
        api_key="not-real-key",
        base_url="",
        model="gemini-test",
        prompt_cache_enabled=True,
    )

    # Replace the SDK-shaped client with a fake whose caches.create returns
    # a predictable name. The lazy import inside get_or_create_cached_prefix
    # picks up the patched module-level GeminiCacheManager naturally.
    fake_create = AsyncMock(return_value=SimpleNamespace(name="cachedContents/from-llm-test"))
    llm._client = MagicMock()
    llm._client.aio = MagicMock()
    llm._client.aio.caches = MagicMock()
    llm._client.aio.caches.create = fake_create

    name = await llm.get_or_create_cached_prefix(
        system_instruction="A long enough system prompt for caching",
        response_schema=None,
    )
    assert name == "cachedContents/from-llm-test"
    # The manager was lazy-built on first use.
    assert llm._cache_manager is not None
    # Second call within TTL → no new SDK call.
    again = await llm.get_or_create_cached_prefix(
        system_instruction="A long enough system prompt for caching",
        response_schema=None,
    )
    assert again == name
    assert fake_create.call_count == 1


@pytest.mark.asyncio
async def test_call_falls_back_to_uncached_when_cache_400s():
    """A stale/invalid CachedContent makes the generate call 400. The provider
    must drop the cache, invalidate the entry, and retry the SAME call inline
    (prefix re-sent) so caching never breaks a request."""
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from google.genai import errors as genai_errors

    from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager, _CacheEntry
    from hindsight_api.engine.providers.gemini_llm import GeminiLLM

    llm = GeminiLLM(
        provider="gemini", api_key="not-real-key", base_url="", model="gemini-test", prompt_cache_enabled=True
    )

    # Seed a cache manager entry that maps to the (now invalid) cache name.
    mgr = GeminiCacheManager(client=MagicMock())
    mgr._entries["fp"] = _CacheEntry(name="cachedContents/stale", created_at=time.monotonic(), ttl_seconds=3300)
    llm._cache_manager = mgr

    captured = []

    def _gen(*, model, contents, config):
        captured.append(config)
        if len(captured) == 1:
            # First (cached) attempt — Gemini rejects the dead cache.
            raise genai_errors.ClientError(
                400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "CachedContent not found"}}
            )
        # Retry without the cache succeeds.
        return SimpleNamespace(
            text="extracted",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=2, cached_content_token_count=0, thoughts_token_count=0
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    llm._client = MagicMock()
    llm._client.aio = MagicMock()
    llm._client.aio.models = MagicMock()
    llm._client.aio.models.generate_content = AsyncMock(side_effect=_gen)

    result = await llm.call(
        messages=[{"role": "system", "content": "SYSTEM PREFIX"}, {"role": "user", "content": "doc"}],
        cached_prefix="cachedContents/stale",
        max_retries=2,
        temperature=0.1,
    )

    # The request succeeded via the uncached retry.
    assert result == "extracted"
    assert len(captured) == 2
    # First attempt referenced the cache; the retry inlined the prefix instead.
    assert captured[0].cached_content == "cachedContents/stale"
    assert captured[1].cached_content is None
    assert captured[1].system_instruction == "SYSTEM PREFIX"
    # The dead entry was invalidated so the next operation recreates it.
    assert mgr._entries == {}


@pytest.mark.asyncio
async def test_create_cache_times_out_and_falls_back():
    """The create runs under the manager lock, so a hung caches.create would block
    every concurrent caller (e.g. all chunks of a retain batch). It must time out
    and return None so callers proceed uncached instead of stalling."""
    import asyncio
    from types import SimpleNamespace

    from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager

    async def _hang(*args, **kwargs):
        await asyncio.sleep(5)
        return SimpleNamespace(name="never")

    client = MagicMock()
    client.aio = MagicMock()
    client.aio.caches = MagicMock()
    client.aio.caches.create = _hang

    mgr = GeminiCacheManager(client, create_timeout_seconds=0.05)
    result = await mgr.get_or_create(model="m", system_instruction="long enough prefix " * 20)
    assert result is None
    assert mgr._entries == {}


# ---- Tools: cache key + create wiring -----------------------------------


def test_fingerprint_changes_with_tools():
    """Two prefixes that differ ONLY in tools must hash differently —
    otherwise a loop that adds a tool would silently reuse a stale
    cache that doesn't know about it."""
    tools_a = [{"type": "function", "function": {"name": "search", "description": "search", "parameters": {}}}]
    tools_b = [
        {"type": "function", "function": {"name": "search", "description": "search", "parameters": {}}},
        {"type": "function", "function": {"name": "fetch", "description": "fetch", "parameters": {}}},
    ]
    fp_a = GeminiCacheManager.fingerprint("m", "sys", None, tools=tools_a)
    fp_b = GeminiCacheManager.fingerprint("m", "sys", None, tools=tools_b)
    assert fp_a != fp_b


def test_fingerprint_stable_under_dict_reordering():
    """The tools list contains dicts; iteration order of dict keys
    must not affect the fingerprint (otherwise upstream re-serialisation
    would produce phantom cache misses)."""
    tools_1 = [{"type": "function", "function": {"description": "d", "name": "n", "parameters": {"a": 1, "b": 2}}}]
    tools_2 = [{"function": {"parameters": {"b": 2, "a": 1}, "name": "n", "description": "d"}, "type": "function"}]
    fp_1 = GeminiCacheManager.fingerprint("m", "sys", None, tools=tools_1)
    fp_2 = GeminiCacheManager.fingerprint("m", "sys", None, tools=tools_2)
    assert fp_1 == fp_2


@pytest.mark.asyncio
async def test_get_or_create_passes_tools_to_create():
    """When tools are provided, the underlying caches.create call
    must include them so the cached prefix actually contains the tool
    definitions."""
    captured = {}

    async def fake_create(*, model, config):
        captured["model"] = model
        captured["config_dict"] = config.__dict__ if hasattr(config, "__dict__") else dict(config)
        return SimpleNamespace(name="cachedContents/with-tools")

    client = MagicMock()
    client.aio = MagicMock()
    client.aio.caches = MagicMock()
    client.aio.caches.create = fake_create

    mgr = GeminiCacheManager(client)
    tools = [
        {
            "type": "function",
            "function": {"name": "search", "description": "do a search", "parameters": {"type": "object"}},
        }
    ]
    name = await mgr.get_or_create(
        model="gemini-3.1-flash-lite",
        system_instruction="You are a helpful tool-using assistant.",
        tools=tools,
    )
    assert name == "cachedContents/with-tools"
    # The Gemini SDK's CreateCachedContentConfig accepted a `tools` list.
    cfg = captured["config_dict"]
    assert "tools" in cfg, f"tools should be in cache config; got keys: {list(cfg.keys())}"
    assert cfg["tools"], "tools list should be non-empty"
