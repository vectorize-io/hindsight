"""
Tests for the bank stats endpoint and the memories-timeseries endpoint.

Covers the new fields exposed by GET /v1/default/banks/{bank_id}/stats
(operations_by_status) and the new endpoint
GET /v1/default/banks/{bank_id}/stats/memories-timeseries.
"""

import uuid
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_bank_id():
    return f"stats_test_{datetime.now().timestamp()}"


async def _insert_memory(memory, bank_id: str, text: str, *, failed: bool = False) -> str:
    """Insert a single experience memory, optionally marked as consolidation-failed."""
    mem_id = uuid.uuid4()
    async with memory._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, created_at, consolidation_failed_at)
            VALUES ($1, $2, $3, 'experience', now(), CASE WHEN $4 THEN now() ELSE NULL END)
            """,
            mem_id,
            bank_id,
            text,
            failed,
        )
    return str(mem_id)


@pytest.mark.asyncio
async def test_bank_stats_exposes_operations_by_status(api_client, test_bank_id):
    """/stats should return operations_by_status with all finished operations."""
    try:
        # Kick off a retain so at least one completed operation exists.
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Alice is a software engineer.", "context": "team"}]},
        )
        assert response.status_code == 200

        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
        assert response.status_code == 200
        stats = response.json()

        assert "operations_by_status" in stats
        assert isinstance(stats["operations_by_status"], dict)
        # A synchronous retain finishes as "completed".
        assert stats["operations_by_status"].get("completed", 0) >= 1
        # pending/failed counters should still be present as scalar mirrors.
        assert stats["pending_operations"] == stats["operations_by_status"].get("pending", 0)
        assert stats["failed_operations"] == stats["operations_by_status"].get("failed", 0)
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "period,expected_count,expected_trunc",
    [
        ("1h", 60, "minute"),
        ("12h", 12, "hour"),
        ("1d", 24, "hour"),
        ("7d", 7, "day"),
        ("30d", 30, "day"),
        ("90d", 90, "day"),
    ],
)
async def test_memories_timeseries_periods(api_client, test_bank_id, period, expected_count, expected_trunc):
    """Every period must return the full expected bucket count and trunc."""
    try:
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Bob works on infrastructure.", "context": "team"}]},
        )
        assert response.status_code == 200

        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/stats/memories-timeseries",
            params={"period": period},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["bank_id"] == test_bank_id
        assert body["period"] == period
        assert body["trunc"] == expected_trunc
        assert len(body["buckets"]) == expected_count

        for bucket in body["buckets"]:
            assert "time" in bucket
            # Bucket `time` must serialize as a tz-aware ISO (ending in `+00:00` or `Z`).
            # A naive ISO (`2026-04-18T00:00:00`) would be parsed as local time by
            # `new Date()` per ECMA-262, shifting the chart by the browser's timezone.
            assert bucket["time"].endswith("+00:00") or bucket["time"].endswith("Z"), (
                f"bucket time must include UTC offset, got {bucket['time']!r}"
            )
            assert bucket["world"] >= 0
            assert bucket["experience"] >= 0
            assert bucket["observation"] >= 0
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_memories_timeseries_invalid_period_falls_back(api_client, test_bank_id):
    """An unknown period must fall back to the 7d default."""
    try:
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/stats/memories-timeseries",
            params={"period": "nonsense"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["period"] == "7d"
        assert body["trunc"] == "day"
        assert len(body["buckets"]) == 7
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_memories_timeseries_empty_bank_returns_zero_filled_buckets(api_client, test_bank_id):
    """A bank with no memories must still return the full zero-filled bucket set."""
    try:
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/stats/memories-timeseries",
            params={"period": "7d"},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["buckets"]) == 7
        for bucket in body["buckets"]:
            assert bucket["world"] == 0
            assert bucket["experience"] == 0
            assert bucket["observation"] == 0
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_memories_timeseries_reflects_retained_memories(api_client, test_bank_id):
    """Freshly-retained memories must show up in today's bucket counts."""
    try:
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={
                "items": [
                    {"content": "Alice is a software engineer.", "context": "team"},
                    {"content": "Bob works on infrastructure.", "context": "team"},
                ]
            },
        )
        assert response.status_code == 200

        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/stats/memories-timeseries",
            params={"period": "7d"},
        )
        assert response.status_code == 200
        body = response.json()
        totals = sum(b["world"] + b["experience"] + b["observation"] for b in body["buckets"])
        assert totals >= 2, "expected at least two memories across all buckets"

        # Those memories should land in the most-recent bucket.
        latest = body["buckets"][-1]
        assert latest["world"] + latest["experience"] + latest["observation"] >= 2
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_bank_stats_reports_failed_consolidation(api_client, memory, test_bank_id):
    """/stats must surface the count of memories with consolidation_failed_at set."""
    try:
        await _insert_memory(memory, test_bank_id, "Alice failed 1.", failed=True)
        await _insert_memory(memory, test_bank_id, "Alice failed 2.", failed=True)
        await _insert_memory(memory, test_bank_id, "Alice pending.", failed=False)

        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
        assert response.status_code == 200
        stats = response.json()

        assert stats["failed_consolidation"] == 2
        # The two failed memories also count as "not-yet-consolidated".
        assert stats["pending_consolidation"] >= 3
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_list_memories_filter_by_consolidation_state_failed(api_client, memory, test_bank_id):
    """?consolidation_state=failed returns only memories with consolidation_failed_at set."""
    try:
        failed_id = await _insert_memory(memory, test_bank_id, "Broken item.", failed=True)
        await _insert_memory(memory, test_bank_id, "Healthy item.", failed=False)

        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/memories/list",
            params={"consolidation_state": "failed"},
        )
        assert response.status_code == 200
        body = response.json()

        ids = [item["id"] for item in body["items"]]
        assert failed_id in ids
        assert body["total"] == 1
        assert body["items"][0]["consolidation_failed_at"] is not None
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_list_memories_filter_by_consolidation_state_rejects_unknown(api_client, test_bank_id):
    """An invalid consolidation_state value must return a 400 (not 500)."""
    try:
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/memories/list",
            params={"consolidation_state": "bogus"},
        )
        assert response.status_code == 400
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_bank_stats_link_counts_have_no_join(api_client, test_bank_id):
    """link_counts must be populated; the deprecated breakdown fields must be empty.

    Confirms the simplified single-table aggregation still produces the totals
    the UI reads (`links_by_link_type`) without the historical
    memory_links⇒memory_units join that powered the 2D `links_breakdown` no
    consumer reads.
    """
    try:
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Carol leads platform engineering.", "context": "team"}]},
        )
        assert response.status_code == 200

        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
        assert response.status_code == 200
        stats = response.json()

        # link totals must still come back so the UI overview cards render.
        assert isinstance(stats["links_by_link_type"], dict)
        assert stats["total_links"] >= 0

        # Deprecated breakdown fields stay in the response shape but are empty.
        assert stats["links_breakdown"] == {}
        assert stats["links_by_fact_type"] == {}
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")


@pytest.mark.asyncio
async def test_get_bank_freshness_returns_only_consolidation_fields(memory, test_bank_id):
    """get_bank_freshness must return just the freshness keys, no link aggregation."""
    from hindsight_api.extensions import RequestContext

    try:
        await _insert_memory(memory, test_bank_id, "Headed for consolidation.", failed=False)
        await _insert_memory(memory, test_bank_id, "Also pending.", failed=True)

        freshness = await memory.get_bank_freshness(
            test_bank_id,
            request_context=RequestContext(internal=True),
        )

        assert set(freshness.keys()) == {
            "last_consolidated_at",
            "pending_consolidation",
            "failed_consolidation",
        }
        assert freshness["pending_consolidation"] >= 2
        assert freshness["failed_consolidation"] >= 1
    finally:
        await memory._bank_stats_cache.clear()
        async with memory._pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", test_bank_id)


@pytest.mark.asyncio
async def test_reflect_uses_freshness_not_bank_stats(memory, test_bank_id):
    """reflect() must call the cheap freshness query, not get_bank_stats.

    Counts calls to `_compute_bank_stats` (the heavy loader) during a reflect
    invocation; it must stay at zero — reflect should route through
    `get_bank_freshness` instead.
    """
    from hindsight_api.extensions import RequestContext

    try:
        # Seed a single memory so reflect has something to inspect.
        await _insert_memory(memory, test_bank_id, "Reflect seed.", failed=False)

        compute_calls = 0
        original_compute = memory._compute_bank_stats

        async def counting_compute(bank_id: str):
            nonlocal compute_calls
            compute_calls += 1
            return await original_compute(bank_id)

        memory._compute_bank_stats = counting_compute  # type: ignore[method-assign]
        try:
            await memory._bank_stats_cache.clear()
            try:
                await memory.reflect(
                    test_bank_id,
                    "What do you know about this bank?",
                    request_context=RequestContext(internal=True),
                )
            except Exception:
                # reflect may fail without a configured LLM in this test env;
                # we only care that it did not invoke the heavy stats loader
                # before failing.
                pass
            assert compute_calls == 0
        finally:
            memory._compute_bank_stats = original_compute  # type: ignore[method-assign]
    finally:
        await memory._bank_stats_cache.clear()
        async with memory._pool.acquire() as conn:
            await conn.execute("DELETE FROM memory_units WHERE bank_id = $1", test_bank_id)


@pytest.mark.asyncio
async def test_bank_stats_served_from_cache_on_repeat_call(api_client, memory, test_bank_id):
    """A second /stats call within the TTL must not re-run the aggregations.

    The cache layer wraps the DB-heavy `_compute_bank_stats` body; counting
    its invocations is the cleanest way to prove the wiring works without
    relying on timing.
    """
    try:
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Bob is a project manager.", "context": "team"}]},
        )
        assert response.status_code == 200

        original = memory._compute_bank_stats
        call_count = 0

        async def counting_compute(bank_id: str):
            nonlocal call_count
            call_count += 1
            return await original(bank_id)

        # Make sure no stale entry exists from prior test ordering.
        await memory._bank_stats_cache.clear()
        memory._compute_bank_stats = counting_compute  # type: ignore[method-assign]
        try:
            first = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
            second = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats")
            assert first.status_code == 200
            assert second.status_code == 200
            assert first.json() == second.json()
            assert call_count == 1
        finally:
            memory._compute_bank_stats = original  # type: ignore[method-assign]
            await memory._bank_stats_cache.clear()
    finally:
        await api_client.delete(f"/v1/default/banks/{test_bank_id}")
