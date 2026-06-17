"""Tests for the async-operation progress snapshot surface (issue #1840).

Long-running consolidation and batch-retain write a coarse ``progress`` snapshot
(stage + processed/total + counters) into ``async_operations.result_metadata`` at
phase/batch boundaries. Operators read it back through the operation status/list API
to tell a healthy long-running job from a frozen one.

Split per project convention: the mechanics here are fully deterministic (DB writes,
JSON merge, API mapping, call wiring), so everything is a direct assert — no LLM judge.
"""

import json
import uuid

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute("INSERT INTO banks (bank_id) VALUES ($1) ON CONFLICT (bank_id) DO NOTHING", bank_id)


async def _insert_operation(pool, bank_id: str, *, status: str = "processing", result_metadata: str = "{}") -> str:
    op_id = uuid.uuid4()
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, result_metadata, updated_at)
        VALUES ($1, $2, 'consolidation', $3, $4::jsonb, now() - interval '1 hour')
        """,
        op_id,
        bank_id,
        status,
        result_metadata,
    )
    return str(op_id)


def _spy_progress(memory: MemoryEngine, captured: list[dict]):
    """Wrap _write_operation_progress so we record every call but keep the real DB write."""
    real = memory._write_operation_progress

    async def _wrapper(operation_id, **kwargs):
        captured.append({"operation_id": operation_id, **kwargs})
        await real(operation_id, **kwargs)

    return _wrapper


def _read_meta(row) -> dict:
    """Normalize result_metadata (asyncpg may hand back jsonb as str or dict)."""
    meta = row["result_metadata"]
    return json.loads(meta) if isinstance(meta, str) else (meta or {})


def _const(value):
    """Return an async function that ignores its args and returns `value` (for monkeypatching resolvers)."""

    async def _inner(*_args, **_kwargs):
        return value

    return _inner


@pytest.mark.asyncio
async def test_write_operation_progress_merges_without_clobbering(memory: MemoryEngine):
    """The helper replaces only the `progress` key and bumps updated_at, leaving siblings intact."""
    bank_id = f"op_progress_merge_{uuid.uuid4().hex[:8]}"
    pool = memory._pool
    await _ensure_bank(pool, bank_id)
    op_id = await _insert_operation(pool, bank_id, result_metadata='{"is_parent": true, "items_count": 5}')

    await memory._write_operation_progress(op_id, stage="processing_batch", processed=3, total=10, detail={"round": 1})

    row = await pool.fetchrow(
        "SELECT result_metadata, updated_at, created_at FROM async_operations WHERE operation_id = $1",
        uuid.UUID(op_id),
    )
    meta = _read_meta(row)

    # Sibling keys survive the merge.
    assert meta["is_parent"] is True
    assert meta["items_count"] == 5
    # Progress snapshot written with the expected shape.
    assert meta["progress"]["stage"] == "processing_batch"
    assert meta["progress"]["processed"] == 3
    assert meta["progress"]["total"] == 10
    assert meta["progress"]["detail"] == {"round": 1}
    assert "at" in meta["progress"]
    # updated_at advanced past the deliberately-stale insert value.
    assert row["updated_at"] > row["created_at"]

    # A second write replaces (not appends to) the progress key.
    await memory._write_operation_progress(op_id, stage="refreshing_mental_models", processed=10, total=10)
    row2 = await pool.fetchrow("SELECT result_metadata FROM async_operations WHERE operation_id = $1", uuid.UUID(op_id))
    meta2 = _read_meta(row2)
    assert meta2["progress"]["stage"] == "refreshing_mental_models"
    assert meta2["progress"].get("detail") is None
    assert meta2["is_parent"] is True  # still intact


@pytest.mark.asyncio
async def test_progress_surfaced_via_get_and_list(api_client, memory: MemoryEngine):
    """A written progress snapshot appears as a typed `progress` object on get + list."""
    bank_id = f"op_progress_api_{uuid.uuid4().hex[:8]}"
    pool = memory._pool
    await _ensure_bank(pool, bank_id)
    op_id = await _insert_operation(pool, bank_id)
    await memory._write_operation_progress(
        op_id, stage="processing_batch", processed=250, total=1200, detail={"observations_created": 12}
    )

    get_resp = await api_client.get(f"/v1/default/banks/{bank_id}/operations/{op_id}")
    assert get_resp.status_code == 200
    progress = get_resp.json()["progress"]
    assert progress is not None
    assert progress["stage"] == "processing_batch"
    assert progress["processed"] == 250
    assert progress["total"] == 1200
    assert progress["detail"]["observations_created"] == 12

    list_resp = await api_client.get(f"/v1/default/banks/{bank_id}/operations")
    assert list_resp.status_code == 200
    op = next(o for o in list_resp.json()["operations"] if o["id"] == op_id)
    assert op["progress"]["stage"] == "processing_batch"
    assert op["progress"]["processed"] == 250


@pytest.mark.asyncio
async def test_progress_absent_returns_null(api_client, memory: MemoryEngine):
    """Operations that never reached a checkpoint expose no progress value.

    The field is omitted from the JSON when null (responses drop null fields), so
    ``.get("progress")`` is None whether the key is absent or explicitly null.
    """
    bank_id = f"op_progress_none_{uuid.uuid4().hex[:8]}"
    pool = memory._pool
    await _ensure_bank(pool, bank_id)
    op_id = await _insert_operation(pool, bank_id)

    get_resp = await api_client.get(f"/v1/default/banks/{bank_id}/operations/{op_id}")
    assert get_resp.status_code == 200
    assert get_resp.json().get("progress") is None

    list_resp = await api_client.get(f"/v1/default/banks/{bank_id}/operations")
    op = next(o for o in list_resp.json()["operations"] if o["id"] == op_id)
    assert op.get("progress") is None


@pytest.mark.asyncio
async def test_consolidation_records_advancing_progress(memory: MemoryEngine, request_context, monkeypatch):
    """A real consolidation run emits scanning → processing_batch → refreshing_mental_models,
    with processed advancing and a durable snapshot left on the operation row."""
    config = _get_raw_config()
    monkeypatch.setattr(config, "enable_observations", True)

    bank_id = f"op_progress_consol_{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    fake_no_obs = type(config)(
        **{**{f: getattr(config, f) for f in config.__dataclass_fields__}, "enable_observations": False}
    )
    monkeypatch.setattr(memory._config_resolver, "resolve_full_config", _const(fake_no_obs))
    for i in range(4):
        await memory.retain_async(
            bank_id=bank_id,
            content=f"Fact {i}: the user enjoys hobby number {i}.",
            request_context=request_context,
        )

    # Restore a consolidation-enabled config for the run itself.
    monkeypatch.setattr(memory._config_resolver, "resolve_full_config", _const(config))

    pool = memory._pool
    op_id = await _insert_operation(pool, bank_id, status="processing")

    captured: list[dict] = []
    monkeypatch.setattr(memory, "_write_operation_progress", _spy_progress(memory, captured))

    result = await run_consolidation_job(
        memory_engine=memory,
        bank_id=bank_id,
        request_context=request_context,
        operation_id=op_id,
    )
    assert result["status"] == "completed"

    stages = [c["stage"] for c in captured]
    assert "consolidating" in stages
    assert "refreshing_mental_models" in stages

    # The initial "consolidating" snapshot reports the discovered total (some positive
    # number of facts extracted from the 4 retained statements) with nothing done yet.
    consolidating = [c for c in captured if c["stage"] == "consolidating"]
    initial = next(c for c in consolidating if c["processed"] == 0)
    assert initial["total"] > 0

    # Per-LLM-batch "consolidating" snapshots report real forward progress against the
    # same total, advancing to cover every memory by the final batch — this is the
    # heartbeat that previously sat frozen at the pre-batch count.
    assert all(c["total"] == initial["total"] for c in consolidating)
    assert max(c["processed"] for c in consolidating) == initial["total"]
    assert any(c["processed"] > 0 for c in consolidating)

    # The durable row carries the last snapshot for an operator polling the API.
    row = await pool.fetchrow("SELECT result_metadata FROM async_operations WHERE operation_id = $1", uuid.UUID(op_id))
    meta = _read_meta(row)
    assert meta["progress"]["stage"] == "refreshing_mental_models"

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_batch_retain_records_chunk_progress(memory: MemoryEngine, request_context, monkeypatch):
    """Batch retain reports chunk-level "storing N/total" progress from the streaming
    pipeline, reaching total/total on completion — so a finished retain's last snapshot
    reflects done, and a long document shows chunks committing rather than an opaque tick."""
    bank_id = f"op_progress_retain_{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

    pool = memory._pool
    op_id = await _insert_operation(pool, bank_id, status="processing")
    # Reuse the consolidation-typed row purely as a tracking target.
    await pool.execute(
        "UPDATE async_operations SET operation_type = 'batch_retain' WHERE operation_id = $1", uuid.UUID(op_id)
    )

    captured: list[dict] = []
    monkeypatch.setattr(memory, "_write_operation_progress", _spy_progress(memory, captured))

    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": "The user lives in Berlin."}, {"content": "The user works as an architect."}],
        operation_id=op_id,
        request_context=request_context,
    )

    storing_calls = [c for c in captured if c["stage"] == "storing"]
    assert storing_calls, "expected at least one 'storing' chunk-progress write"
    # The streaming pipeline writes after each consumer batch commits, so the final
    # snapshot reaches total/total and every write reports its committed-fact count.
    last = storing_calls[-1]
    assert last["total"] >= 1
    assert last["processed"] == last["total"]
    assert all("facts_committed" in c["detail"] for c in storing_calls)

    # The durable row reflects completion, so a consumer polling a finished retain
    # never sees a partial "still storing" snapshot.
    row = await pool.fetchrow("SELECT result_metadata FROM async_operations WHERE operation_id = $1", uuid.UUID(op_id))
    progress = _read_meta(row)["progress"]
    assert progress["stage"] == "storing"
    assert progress["processed"] == progress["total"]

    await memory.delete_bank(bank_id, request_context=request_context)
