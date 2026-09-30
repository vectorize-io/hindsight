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
    warm = await memory.ensure_bank_profile(bank_id, request_context=request_context)
    assert warm["disposition"]["skepticism"] == 3

    await memory.update_bank_disposition(
        bank_id, {"skepticism": 5, "literalism": 4, "empathy": 2}, request_context=request_context
    )

    after = await memory.get_bank_profile(bank_id, request_context=request_context)
    assert after["disposition"]["skepticism"] == 5, "the cached profile survived a disposition write"


@pytest.mark.asyncio
async def test_a_mission_update_is_visible_to_the_next_read(memory: MemoryEngine, request_context):
    bank_id = _bank("cache_mission")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

    await memory.set_bank_mission(bank_id, "the new mission", request_context=request_context)

    after = await memory.get_bank_profile(bank_id, request_context=request_context)
    assert after["mission"] == "the new mission", "the cached profile survived a mission write"


@pytest.mark.asyncio
async def test_update_bank_returns_what_it_wrote(memory: MemoryEngine, request_context):
    """`update_bank` reads the profile back to return it, so a stale entry makes a successful
    update answer with the values it just replaced."""
    bank_id = _bank("cache_update_bank")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

    returned = await memory.update_bank(bank_id, name="renamed", request_context=request_context)
    assert returned["name"] == "renamed", "update_bank answered with the pre-update profile"


@pytest.mark.asyncio
async def test_a_config_override_is_visible_to_the_next_resolve(memory: MemoryEngine, request_context):
    """The config row is cached under its own key, so it needs its own invalidation -- and the
    resolved config is what a retain reads, which is the path the cache exists to speed up."""
    bank_id = _bank("cache_config")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
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
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
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
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

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


@pytest.mark.asyncio
async def test_a_deleted_bank_stops_reading_as_existing(memory: MemoryEngine, request_context):
    """A restore or clone into a freshly deleted id checks existence through the cache, so a stale
    entry refuses it with "already exists" for the whole TTL."""
    from hindsight_api.engine.retain import bank_utils

    bank_id = _bank("cache_delete")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    backend = await memory._get_backend()
    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is not None

    await memory.delete_bank(bank_id, request_context=request_context)

    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is None, (
        "the cached profile survived the bank's deletion"
    )


async def _delete_as_another_process(memory: MemoryEngine, bank_id: str, request_context, monkeypatch) -> None:
    """Delete the bank with invalidation disabled, then restore it.

    That leaves this process holding the entry it cached before the delete, with no notice that it
    moved -- exactly the state of every process that did not serve the delete.
    """
    from hindsight_api.engine import bank_info_cache
    from hindsight_api.engine.retain import bank_utils

    backend = await memory._get_backend()
    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is not None  # warm

    async def _no_invalidation(*_a, **_kw):
        return None

    with monkeypatch.context() as m:
        m.setattr(bank_info_cache, "invalidate", _no_invalidation)
        await memory.delete_bank(bank_id, request_context=request_context)

    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is not None, (
        "the setup no longer models another process: the entry did not survive the delete"
    )


@pytest.mark.asyncio
async def test_a_recall_of_a_bank_deleted_by_another_process_404s(memory: MemoryEngine, request_context, monkeypatch):
    """The existence guard answers from the cache, so on a process that did not serve the delete a
    recall of the deleted bank gets past it and fails in the store instead. A store that owns its
    storage has already dropped the bank's, so that failure is an opaque store error (a 500) rather
    than the 404 every other process answers. The store failure is simulated here: the SQL store
    answers a deleted bank with an empty result rather than an error."""
    from hindsight_api.engine.retain import bank_utils
    from hindsight_api.extensions import OperationValidationError

    bank_id = _bank("cache_recall_deleted")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    await _delete_as_another_process(memory, bank_id, request_context, monkeypatch)

    async def _storage_gone(*_a, **_kw):
        raise RuntimeError("the bank's storage no longer exists")

    monkeypatch.setattr(memory, "_search_with_retries", _storage_gone)

    with pytest.raises(OperationValidationError) as exc_info:
        await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert exc_info.value.status_code == 404

    backend = await memory._get_backend()
    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is None, (
        "the stale entry survived, so every later read on this process fails in the store again"
    )


@pytest.mark.asyncio
async def test_a_recall_failure_on_an_existing_bank_keeps_its_error(memory: MemoryEngine, request_context, monkeypatch):
    """The failure-path re-check must not turn a real store fault into a 404."""
    from hindsight_api.extensions import OperationValidationError

    bank_id = _bank("cache_recall_fault")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

    async def _fault(*_a, **_kw):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(memory, "_search_with_retries", _fault)

    with pytest.raises(Exception) as exc_info:
        await memory.recall_async(bank_id=bank_id, query="anything", request_context=request_context)
    assert not isinstance(exc_info.value, OperationValidationError)
    assert "store unavailable" in str(exc_info.value)


_QUERY = "Where does Alice work?"


async def _retain_one(memory: MemoryEngine, bank_id: str, request_context) -> None:
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "Alice works at Acme as an engineer."}],
        request_context=request_context,
    )


@pytest.mark.asyncio
async def test_a_recall_with_results_pays_no_uncached_existence_read(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The deleted-bank re-check runs only on a failed or empty recall. A recall that returns
    results must not take the pool acquire an uncached existence probe costs -- recall is a hot
    path."""
    from hindsight_api.engine.retain import bank_utils

    bank_id = _bank("cache_recall_hot")
    await _retain_one(memory, bank_id, request_context)
    warm = await memory.recall_async(bank_id=bank_id, query=_QUERY, request_context=request_context)
    assert warm.results, "the setup needs a recall that returns results"

    probes = 0
    original = bank_utils.bank_exists

    async def _counting(*a, **kw):
        nonlocal probes
        probes += 1
        return await original(*a, **kw)

    monkeypatch.setattr(bank_utils, "bank_exists", _counting)
    result = await memory.recall_async(bank_id=bank_id, query=_QUERY, request_context=request_context)

    assert result.results
    assert probes == 0, "a recall that returned results ran the uncached existence probe"


@pytest.mark.asyncio
async def test_a_sql_recall_of_a_bank_deleted_by_another_process_404s(
    memory: MemoryEngine, request_context, monkeypatch
):
    """The SQL store does not fail for a deleted bank: its rows are gone, so a recall that slips
    past the stale guard answers 200 with no results -- byte-identical to a healthy empty bank,
    which is what the guard exists to rule out (#4175). Unlike the store-failure case, nothing is
    simulated here: this is the real SQL recall path."""
    from hindsight_api.engine.retain import bank_utils
    from hindsight_api.extensions import OperationValidationError

    bank_id = _bank("cache_sql_deleted")
    await _retain_one(memory, bank_id, request_context)
    before = await memory.recall_async(bank_id=bank_id, query=_QUERY, request_context=request_context)
    assert before.results, "the setup needs a bank whose recall returns results before the delete"

    await _delete_as_another_process(memory, bank_id, request_context, monkeypatch)

    with pytest.raises(OperationValidationError) as exc_info:
        await memory.recall_async(bank_id=bank_id, query=_QUERY, request_context=request_context)
    assert exc_info.value.status_code == 404

    backend = await memory._get_backend()
    assert await bank_utils.get_bank_profile_if_exists(backend, bank_id) is None, "the stale entry survived the 404"


@pytest.mark.asyncio
async def test_a_recall_of_an_existing_empty_bank_still_answers_empty(memory: MemoryEngine, request_context):
    """The empty-result re-check must not turn a healthy empty bank into a 404."""
    bank_id = _bank("cache_sql_empty")
    await memory.ensure_bank_profile(bank_id, request_context=request_context)

    result = await memory.recall_async(bank_id=bank_id, query=_QUERY, request_context=request_context)
    assert result.results == []
