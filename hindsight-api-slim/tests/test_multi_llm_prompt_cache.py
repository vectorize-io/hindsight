"""Prompt-prefix caching in a multi-LLM chain is member-aware (#5123).

On a chain, ``_provider_impl`` resolves to the primary member only. Retain and
consolidation used to ask that primary for a cache handle and send it to
whichever member served the request, so a caching-capable failover /
round-robin member never got a handle, and a caching primary's handle could be
sent to a member it does not belong to. Each member must get its own handle.

Members are lightweight fakes; no real providers or network.
"""

import dataclasses
from typing import Any

from hindsight_api.config import LLMMetadataRoute, LLMStrategyConfig
from hindsight_api.engine.llm_wrapper import ConfiguredLLMProvider
from hindsight_api.engine.multi_llm import MemberCachedPrefixes, MultiLLMProvider, get_or_create_cached_prefix
from hindsight_api.engine.response_models import LLMCallResult, TokenUsage


class _FakeImpl:
    def __init__(self, name: str, cache_handle: str | None, lookup_error: Exception | None = None):
        self._name = name
        self._cache_handle = cache_handle
        self._lookup_error = lookup_error
        self.cache_requests: list[str] = []

    def supports_prompt_caching(self) -> bool:
        return self._cache_handle is not None or self._lookup_error is not None

    async def get_or_create_cached_prefix(
        self, *, system_instruction: str, response_schema: Any | None = None, tools: Any | None = None
    ) -> str | None:
        self.cache_requests.append(system_instruction)
        if self._lookup_error is not None:
            raise self._lookup_error
        return self._cache_handle


class _FakeMember:
    """Stands in for an LLMProvider member and records the kwargs of every call."""

    def __init__(
        self,
        name: str,
        *,
        cache_handle: str | None = None,
        fail: bool = False,
        lookup_error: Exception | None = None,
    ):
        self.provider = name
        self.model = f"{name}-model"
        self._provider_impl = _FakeImpl(name, cache_handle, lookup_error)
        self._fail = fail
        self.calls: list[dict[str, Any]] = []

    async def call(self, **kwargs: Any) -> LLMCallResult:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError(f"{self.provider} down")
        return LLMCallResult(content={"facts": []}, usage=TokenUsage())

    async def call_with_tools(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError(f"{self.provider} down")
        return "tool-result"


def _failover(*members: _FakeMember) -> MultiLLMProvider:
    return MultiLLMProvider(list(members), LLMStrategyConfig(mode="failover"))


def _configured(chain: MultiLLMProvider) -> ConfiguredLLMProvider:
    return ConfiguredLLMProvider(chain, gemini_safety_settings=None)


async def test_caching_failover_member_gets_its_own_handle():
    primary = _FakeMember("openai", fail=True)
    gemini = _FakeMember("gemini", cache_handle="cachedContents/gemini-1")
    llm = _configured(_failover(primary, gemini))

    handle = await get_or_create_cached_prefix(llm, system_instruction="SYSTEM")
    assert handle is not None
    await llm.call(messages=[{"role": "user", "content": "x"}], cached_prefix=handle)

    assert gemini._provider_impl.cache_requests == ["SYSTEM"]
    assert "cached_prefix" not in primary.calls[0]
    assert gemini.calls[0]["cached_prefix"] == "cachedContents/gemini-1"


async def test_caching_primary_handle_is_not_sent_to_other_members():
    gemini = _FakeMember("gemini", cache_handle="cachedContents/gemini-1", fail=True)
    fallback = _FakeMember("anthropic")
    llm = _configured(_failover(gemini, fallback))

    handle = await get_or_create_cached_prefix(llm, system_instruction="SYSTEM")
    await llm.call_with_tools(
        messages=[{"role": "user", "content": "x"}],
        tools=[],
        cached_prefix=handle,
        cached_prefix_message_count=1,
    )

    assert gemini.calls[0]["cached_prefix"] == "cachedContents/gemini-1"
    assert gemini.calls[0]["cached_prefix_message_count"] == 1
    assert "cached_prefix" not in fallback.calls[0]
    assert "cached_prefix_message_count" not in fallback.calls[0]


async def test_each_caching_member_gets_its_own_handle():
    a = _FakeMember("gemini-a", cache_handle="cachedContents/a", fail=True)
    b = _FakeMember("gemini-b", cache_handle="cachedContents/b")
    chain = _failover(a, b)

    handle = await get_or_create_cached_prefix(chain, system_instruction="SYSTEM")
    await chain.call(messages=[], cached_prefix=handle)

    assert a.calls[0]["cached_prefix"] == "cachedContents/a"
    assert b.calls[0]["cached_prefix"] == "cachedContents/b"


async def test_no_handle_when_no_member_caches():
    chain = _failover(_FakeMember("openai"), _FakeMember("anthropic"))
    assert not chain.supports_prompt_caching()
    assert await get_or_create_cached_prefix(chain, system_instruction="SYSTEM") is None


async def test_one_member_lookup_failure_keeps_the_other_members_cache():
    broken = _FakeMember("gemini-a", lookup_error=RuntimeError("cache quota"), fail=True)
    healthy = _FakeMember("gemini-b", cache_handle="cachedContents/b")
    chain = _failover(broken, healthy)

    handle = await get_or_create_cached_prefix(chain, system_instruction="SYSTEM")
    await chain.call(messages=[], cached_prefix=handle)

    assert "cached_prefix" not in broken.calls[0]
    assert healthy.calls[0]["cached_prefix"] == "cachedContents/b"


async def test_metadata_chain_only_caches_for_the_primary():
    # Chain-level calls in metadata mode stay on the primary; routed items call
    # their member directly (and resolve that member's own cache).
    primary = _FakeMember("gemini-a", cache_handle="cachedContents/a")
    routed = _FakeMember("gemini-b", cache_handle="cachedContents/b")
    chain = MultiLLMProvider(
        [primary, routed],
        LLMStrategyConfig(mode="metadata", routes=[LLMMetadataRoute(key="tier", value="eu", member=1)]),
    )

    handle = await get_or_create_cached_prefix(chain, system_instruction="SYSTEM")

    assert isinstance(handle, MemberCachedPrefixes)
    assert handle.for_member(0) == "cachedContents/a"
    assert routed._provider_impl.cache_requests == []


async def test_single_provider_still_uses_its_own_impl():
    member = _FakeMember("gemini", cache_handle="cachedContents/solo")
    assert await get_or_create_cached_prefix(member, system_instruction="SYSTEM") == "cachedContents/solo"


async def test_retain_extraction_sends_the_serving_member_its_own_handle():
    """End to end through the retain chunk extractor with a failover chain."""
    from hindsight_api.config import _get_raw_config
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    primary = _FakeMember("openai", fail=True)
    gemini = _FakeMember("gemini", cache_handle="cachedContents/gemini-1")
    config = dataclasses.replace(
        _get_raw_config(),
        retain_llm_max_retries=0,
        llm_max_retries=0,
        retain_llm_initial_backoff=0.0,
        llm_initial_backoff=0.0,
        retain_llm_max_backoff=0.0,
        llm_max_backoff=0.0,
        retain_extraction_mode="concise",
        retain_extract_causal_links=False,
        retain_mission=None,
    )

    await _extract_facts_from_chunk(
        chunk="some text",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="",
        llm_config=_configured(_failover(primary, gemini)),
        config=config,
    )

    assert len(gemini._provider_impl.cache_requests) == 1
    assert gemini.calls[0]["cached_prefix"] == "cachedContents/gemini-1"
    assert "cached_prefix" not in primary.calls[0]
