"""Every path that writes the bank row invalidates the cached copy of it.

``engine/bank_info_cache`` keeps the two rows a retain reads on every call out of the pool. Across
processes it is TTL-only and that is the documented trade. Within the writing process it is not a
trade: a caller that writes a bank and reads it back must see what it wrote, and the write paths
call ``invalidate`` to keep that true.

These assert the CONTRACT (write, then read, see it) rather than that ``invalidate`` was called, so
a new write path that forgets it fails here even though it never touches this file -- which is the
whole reason the cache is allowed to have a TTL at all. Each test warms the cache with a read
first: without that the read-back would pass on an empty cache and prove nothing.
"""

import uuid

import pytest

from hindsight_api import MemoryEngine


def _bank(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_a_disposition_update_is_visible_to_the_next_read(memory: MemoryEngine, request_context):
    bank_id = _bank("cache_disposition")
    warm = await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)
    assert warm["disposition"]["skepticism"] == 3

    await memory.update_bank_disposition(
        bank_id, {"skepticism": 5, "literalism": 4, "empathy": 2}, request_context=request_context
    )

    after = await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False)
    assert after["disposition"]["skepticism"] == 5, "the cached profile survived a disposition write"


@pytest.mark.asyncio
async def test_a_mission_update_is_visible_to_the_next_read(memory: MemoryEngine, request_context):
    bank_id = _bank("cache_mission")
    await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)

    await memory.set_bank_mission(bank_id, "the new mission", request_context=request_context)

    after = await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=False)
    assert after["mission"] == "the new mission", "the cached profile survived a mission write"


@pytest.mark.asyncio
async def test_update_bank_returns_what_it_wrote(memory: MemoryEngine, request_context):
    """`update_bank` reads the profile back to return it, so a stale entry makes a successful
    update answer with the values it just replaced."""
    bank_id = _bank("cache_update_bank")
    await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)

    returned = await memory.update_bank(bank_id, name="renamed", request_context=request_context)
    assert returned["name"] == "renamed", "update_bank answered with the pre-update profile"


@pytest.mark.asyncio
async def test_a_config_override_is_visible_to_the_next_resolve(memory: MemoryEngine, request_context):
    """The config row is cached under its own key, so it needs its own invalidation -- and the
    resolved config is what a retain reads, which is the path the cache exists to speed up."""
    bank_id = _bank("cache_config")
    await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)
    resolver = memory._config_resolver
    await resolver._load_bank_config(bank_id)

    await memory.update_bank_config(bank_id, {"retain_chunk_size": 1234}, request_context=request_context)

    after = await resolver._load_bank_config(bank_id)
    assert after.get("retain_chunk_size") == 1234, "the cached config row survived a config write"


@pytest.mark.asyncio
async def test_a_config_read_back_does_not_go_through_the_cache(memory: MemoryEngine, request_context, monkeypatch):
    """The cache is per PROCESS and a deployment runs several API pods, so invalidating on write
    only ever fixes the pod that served the write. The endpoint a caller reads back through must
    therefore not consult the cache at all.

    The other pods are modelled by disabling invalidation entirely: that leaves this process in
    exactly their state -- an entry cached before the write, and no notification that it moved.
    A read that still sees the new value is one that did not come from the cache.
    """
    from hindsight_api.engine import bank_info_cache

    bank_id = _bank("cache_bypass")
    await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)
    # Warm the config entry, then take invalidation away before the write.
    await memory.get_bank_config(bank_id, request_context=request_context)

    async def _no_invalidation(*_a, **_kw):
        return None

    monkeypatch.setattr(bank_info_cache, "invalidate", _no_invalidation)
    await memory.update_bank_config(bank_id, {"retain_chunk_size": 4321}, request_context=request_context)

    state = await memory.get_bank_config(bank_id, request_context=request_context)
    assert state.overrides.get("retain_chunk_size") == 4321, (
        "get_bank_config served a cached config row; a caller reading back its own edit sees the "
        "value it replaced on any pod that did not serve the write"
    )
    assert state.config.get("retain_chunk_size") == 4321, "the resolved config came from the cache too"


@pytest.mark.asyncio
async def test_recall_reads_its_config_through_the_cache(memory: MemoryEngine, request_context):
    """Freshness belongs to the caller that needs it, not to `get_bank_config` itself.

    `recall_async` and `retain_batch_async` resolve the bank's config per request. Making the
    method itself uncached to fix read-your-writes on the CONFIG ENDPOINT put a pool acquire on
    both hot paths -- and an acquire costs more than the query it carries, because the pool runs
    five `set_config` calls on checkout and a `RESET ALL` on release.

    Asserted as the property (a warm second read issues no query) rather than by counting call
    sites, so a new hot-path caller that forces a read fails here.
    """
    bank_id = _bank("cache_hot_path")
    await memory.get_bank_profile(bank_id, request_context=request_context, create_if_missing=True)

    resolver = memory._config_resolver
    await resolver.get_bank_config(bank_id, request_context)  # warm

    reads = 0
    original = resolver._load_bank_config

    async def _counting(bank, *, cached=True):
        nonlocal reads
        if not cached:
            reads += 1
        return await original(bank, cached=cached)

    resolver._load_bank_config = _counting
    try:
        await resolver.get_bank_config(bank_id, request_context)
    finally:
        resolver._load_bank_config = original

    assert reads == 0, (
        "get_bank_config forced an uncached bank-config read; recall and retain call this per "
        "request, so that is a pool acquire on every one of them"
    )
