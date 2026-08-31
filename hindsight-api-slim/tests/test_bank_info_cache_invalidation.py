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
