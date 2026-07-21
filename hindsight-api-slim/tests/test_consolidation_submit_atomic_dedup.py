"""Regression tests for issue #1842: submitting consolidation is atomic per bank.

`submit_async_consolidation` promises "at most one pending full-bank consolidation
per bank", but the dedup was a check-then-INSERT that raced under READ COMMITTED —
concurrent submits (a manual `/consolidate` loop racing retain-driven submits and
the round-limit re-queue) each saw no pending row and each inserted, leaking
duplicate pending ops that then piled up as `retry_blocked` and starved the bank.

The submit now serializes per bank (SELECT ... FOR UPDATE on the bank row) so the
check-and-insert is atomic. Runs targeted by observation scopes or source-tag
filters are exempt and may have multiple pending ops.
"""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.search.tags import TagGroupLeaf, TagGroupOr


@pytest.fixture
def no_inline_execution(memory):
    """Stop the SyncTaskBackend from executing submitted ops inline so the pending
    rows survive for inspection (we're testing the submit/dedup path, not execution)."""

    async def _noop(_payload):
        return None

    original = memory._task_backend.submit_task
    memory._task_backend.submit_task = _noop
    yield
    memory._task_backend.submit_task = original


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _count_pending(pool, bank_id: str) -> int:
    return await pool.fetchval(
        """
        SELECT COUNT(*) FROM async_operations
        WHERE bank_id = $1 AND operation_type = 'consolidation' AND status = 'pending'
        """,
        bank_id,
    )


async def _cleanup(pool, bank_id: str) -> None:
    await pool.execute("DELETE FROM async_operations WHERE bank_id = $1", bank_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


@pytest.mark.asyncio
async def test_source_filters_are_serialized_into_targeted_task_payload(
    memory,
    request_context,
    no_inline_execution,
):
    bank_id = f"test-filter-payload-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        result = await memory.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            observation_scopes=[["user:alice"]],
            tags_match="exact",
            tag_groups=[
                TagGroupOr(
                    **{
                        "or": [
                            TagGroupLeaf(tags=["source:app"], match="exact"),
                            TagGroupLeaf(tags=["source:web"], match="all_strict"),
                        ]
                    }
                )
            ],
        )

        raw_payload = await pool.fetchval(
            "SELECT task_payload FROM async_operations WHERE operation_id = $1",
            uuid.UUID(result["operation_id"]),
        )
        payload = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
        assert payload["observation_scopes"] == [["user:alice"]]
        assert payload["tags_match"] == "exact"
        assert payload["tag_groups"] == [
            {
                "or": [
                    {"tags": ["source:app"], "match": "exact"},
                    {"tags": ["source:web"], "match": "all_strict"},
                ]
            }
        ]
        assert not result.get("deduplicated")
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_source_filters_are_deserialized_by_consolidation_task_handler(memory):
    run_job = AsyncMock(return_value={"memories_processed": 0})
    with patch("hindsight_api.engine.consolidation.run_consolidation_job", new=run_job):
        await memory._handle_consolidation(
            {
                "bank_id": "test-filter-handler",
                "observation_scopes": [["user:alice"]],
                "tags_match": "exact",
                "tag_groups": [
                    {
                        "or": [
                            {"tags": ["source:app"], "match": "exact"},
                            {"tags": ["source:web"], "match": "all_strict"},
                        ]
                    }
                ],
            }
        )

    kwargs = run_job.await_args.kwargs
    assert kwargs["observation_scopes"] == [["user:alice"]]
    assert kwargs["tags_match"] == "exact"
    assert [group.model_dump(by_alias=True) for group in kwargs["tag_groups"]] == [
        {
            "or": [
                {"tags": ["source:app"], "match": "exact"},
                {"tags": ["source:web"], "match": "all_strict"},
            ]
        }
    ]


@pytest.mark.asyncio
async def test_concurrent_submits_leave_one_pending(memory, request_context, no_inline_execution):
    """Five concurrent unscoped submits on a bank with no pending op must end with
    exactly one pending row, and all calls must resolve to that one operation_id."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        results = await asyncio.gather(
            *(memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context) for _ in range(5))
        )
        assert await _count_pending(pool, bank_id) == 1
        op_ids = {r["operation_id"] for r in results}
        assert len(op_ids) == 1, f"all submits should share one op, got {op_ids}"
        # Exactly one call created the op; the other four were deduplicated.
        assert sum(1 for r in results if r.get("deduplicated")) == 4
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_sequential_submit_dedups(memory, request_context, no_inline_execution):
    """A second submit while one is already pending returns the existing op."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        first = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        second = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert second["operation_id"] == first["operation_id"]
        assert second.get("deduplicated") is True
        assert await _count_pending(pool, bank_id) == 1
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_filter_kwargs",
    [
        {"tags": []},
        {"tag_groups": []},
    ],
    ids=["empty-tags", "empty-tag-groups"],
)
async def test_effectively_empty_tag_filters_use_full_bank_dedup(
    memory,
    request_context,
    no_inline_execution,
    empty_filter_kwargs,
):
    """Empty ordinary-tag filters execute full-bank work and share its dedupe class."""
    bank_id = f"test-atomic-empty-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        first = await memory.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            **empty_filter_kwargs,
        )
        second = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)

        assert second["operation_id"] == first["operation_id"]
        assert second.get("deduplicated") is True
        assert await _count_pending(pool, bank_id) == 1
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "targeted_kwargs",
    [
        {"observation_scopes": [["proj-a"]]},
        {"tags": ["source:app"]},
        {"tag_groups": [TagGroupLeaf(tags=["source:web"], match="all_strict")]},
        {"tags_match": "exact"},
    ],
    ids=["observation-scopes", "tags", "tag-groups", "exact-untagged"],
)
async def test_full_bank_not_deduped_against_targeted_pending(
    memory,
    request_context,
    no_inline_execution,
    targeted_kwargs,
):
    """A targeted pending task must not swallow a later full-bank sweep."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        targeted = await memory.submit_async_consolidation(
            bank_id=bank_id,
            request_context=request_context,
            **targeted_kwargs,
        )
        full_bank = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert not full_bank.get("deduplicated"), "full-bank submit must not dedup against a targeted pending op"
        assert full_bank["operation_id"] != targeted["operation_id"]
        assert await _count_pending(pool, bank_id) == 2

        # A second full-bank submit still dedups against the full-bank pending op.
        full_bank_2 = await memory.submit_async_consolidation(bank_id=bank_id, request_context=request_context)
        assert full_bank_2["operation_id"] == full_bank["operation_id"]
        assert full_bank_2.get("deduplicated") is True
        assert await _count_pending(pool, bank_id) == 2
    finally:
        await _cleanup(pool, bank_id)


@pytest.mark.asyncio
async def test_scoped_submits_are_not_deduped(memory, request_context, no_inline_execution):
    """Scoped runs are targeted and intentionally exempt — they may pile up pending."""
    bank_id = f"test-atomic-{uuid.uuid4().hex[:8]}"
    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    try:
        r1 = await memory.submit_async_consolidation(
            bank_id=bank_id, request_context=request_context, observation_scopes=[["proj-a"]]
        )
        r2 = await memory.submit_async_consolidation(
            bank_id=bank_id, request_context=request_context, observation_scopes=[["proj-b"]]
        )
        assert r1["operation_id"] != r2["operation_id"]
        assert not r2.get("deduplicated")
        assert await _count_pending(pool, bank_id) == 2
    finally:
        await _cleanup(pool, bank_id)
