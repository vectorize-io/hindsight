"""At most one *pending* refresh operation per (bank, mental model) — issue #3487.

A bank whose refresh queue drains slower than it fills used to accumulate one
`refresh_mental_model` operation per model per consolidation round: ~12k pending
operations covering 259 models (~45 identical copies each), every copy a full
recall + LLM refresh when it eventually ran. The floor is now enforced for every
submit path rather than opt-in per caller, so no enqueue site can reintroduce the
pile by forgetting to ask for dedupe.

Deterministic — no LLM: the worker is stalled so submitted operations stay queued.
"""

import asyncio
import json
import uuid

import pytest

from hindsight_api.engine.consolidation.consolidator import _trigger_mental_model_refreshes
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.memory_engine import MemoryEngine


async def _make_bank(memory: MemoryEngine, request_context) -> str:
    bank_id = f"mmdedupe-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_auto_refresh_mm(conn, bank_id: str) -> str:
    """A model that refreshes after consolidation, last refreshed a day ago."""
    mm_id = f"mm-{uuid.uuid4().hex}"
    await conn.execute(
        """
        INSERT INTO mental_models
          (id, bank_id, subtype, name, source_query, content, tags, trigger, last_refreshed_at)
        VALUES ($1, $2, 'pinned', 'auto model', 'what changed', 'body', $3, $4::jsonb,
                now() - INTERVAL '1 day')
        """,
        mm_id,
        bank_id,
        [],
        json.dumps({"refresh_after_consolidation": True}),
    )
    return mm_id


async def _insert_fact(conn, bank_id: str) -> None:
    """A memory newer than the model's last refresh, so the model reads as stale."""
    await conn.execute(
        "INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at) "
        "VALUES ($1, $2, 'a fresh fact', 'experience', $3, now(), now())",
        uuid.uuid4(),
        bank_id,
        [],
    )


def _stall_worker(memory: MemoryEngine, monkeypatch) -> None:
    """Queue operations without executing them — the slow-draining bank of #3487."""

    async def _never_runs(task_dict):
        return None

    monkeypatch.setattr(memory._task_backend, "submit_task", _never_runs)


async def _refresh_ops(memory: MemoryEngine, bank_id: str) -> list:
    async with memory._pool.acquire() as conn:
        return await conn.fetch(
            "SELECT operation_id, status FROM async_operations "
            "WHERE bank_id = $1 AND operation_type = 'refresh_mental_model'",
            bank_id,
        )


@pytest.mark.asyncio
async def test_repeated_flushes_never_queue_a_second_pending_refresh(
    memory: MemoryEngine, request_context, monkeypatch
):
    """Consolidation rounds and explicit refreshes converge on one queued operation.

    Regression for #3487: the after-consolidation flush fires once per round, and the
    round chain repeats for as long as the backlog lasts. With the operation queued but
    undrained, every one of those submits — and every explicit refresh landing next to
    them — must fold into the operation already waiting for this model.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_auto_refresh_mm(conn, bank)
        await _insert_fact(conn, bank)
    _stall_worker(memory, monkeypatch)

    for _ in range(5):
        await _trigger_mental_model_refreshes(memory, bank, request_context)
        await memory.submit_async_refresh_mental_model(
            bank_id=bank, mental_model_id=mm_id, request_context=request_context
        )

    ops = await _refresh_ops(memory, bank)
    assert len(ops) == 1
    assert ops[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_user_triggered_refresh_folds_into_a_queued_one(memory: MemoryEngine, request_context, monkeypatch):
    """An explicit refresh returns the queued operation instead of adding a copy.

    Nothing is lost by folding: a refresh operation carries no per-request options, so
    the queued one does exactly the work this call is asking for, and it has not started
    — it still reads whatever the caller just edited. The caller gets that operation's id
    back and polls it to completion.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_auto_refresh_mm(conn, bank)
    _stall_worker(memory, monkeypatch)

    first = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )
    second = await memory.submit_async_refresh_mental_model(
        bank_id=bank, mental_model_id=mm_id, request_context=request_context
    )

    assert second["operation_id"] == first["operation_id"]
    assert second["deduplicated"] is True
    assert not first.get("deduplicated")
    assert len(await _refresh_ops(memory, bank)) == 1


@pytest.mark.asyncio
async def test_concurrent_submits_queue_one_refresh(memory: MemoryEngine, request_context, monkeypatch):
    """Simultaneous submits for the same model still queue exactly one operation.

    The lookup and the INSERT are separate statements, so under READ COMMITTED they
    would race on their own: each caller's lookup takes its own snapshot, all of them
    see no queued refresh, and all of them insert. What makes the pair atomic is the
    bank row, locked FOR NO KEY UPDATE at the top of the same transaction and held to
    commit — concurrent submits for the bank serialise on it, so each lookup runs
    against a snapshot that already contains its predecessor's committed row.

    The window is forced open rather than hoped for: every in-flight lookup stalls
    before returning, so all eight submits would sit between their lookup and their
    INSERT at the same time. They cannot, because the lock is taken *before* the
    lookup — the seven losers are still blocked on the bank row while the winner
    commits. Without it this test inserts eight rows (verified by removing the lock).

    Eight at once, none passing skip_if_in_flight: the guarantee cannot depend on the
    caller opting in.
    """
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_auto_refresh_mm(conn, bank)
    _stall_worker(memory, monkeypatch)

    original_fetchval = PostgresConnection.fetchval

    async def _stall_in_flight_lookup(self, query, *args, **kwargs):
        result = await original_fetchval(self, query, *args, **kwargs)
        if "task_payload->>" in query:
            await asyncio.sleep(0.2)
        return result

    monkeypatch.setattr(PostgresConnection, "fetchval", _stall_in_flight_lookup)

    results = await asyncio.gather(
        *(
            memory.submit_async_refresh_mental_model(
                bank_id=bank, mental_model_id=mm_id, request_context=request_context
            )
            for _ in range(8)
        )
    )

    ops = await _refresh_ops(memory, bank)
    assert len(ops) == 1
    # Every loser reports the winner's operation, so all eight callers poll the one
    # refresh that actually runs.
    assert {r["operation_id"] for r in results} == {str(ops[0]["operation_id"])}
    assert sum(1 for r in results if r.get("deduplicated")) == 7


@pytest.mark.asyncio
async def test_other_models_are_unaffected(memory: MemoryEngine, request_context, monkeypatch):
    """The floor is per model, not per bank — a second model still gets its own refresh."""
    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        first_id = await _insert_auto_refresh_mm(conn, bank)
        second_id = await _insert_auto_refresh_mm(conn, bank)
    _stall_worker(memory, monkeypatch)

    for mm_id in (first_id, second_id, first_id, second_id):
        await memory.submit_async_refresh_mental_model(
            bank_id=bank, mental_model_id=mm_id, request_context=request_context
        )

    assert len(await _refresh_ops(memory, bank)) == 2


@pytest.mark.asyncio
async def test_dedupe_statements_survive_the_oracle_rewrite(memory: MemoryEngine, request_context, monkeypatch):
    """Every statement the deduping submit issues must be valid on Oracle too.

    The dedupe used to live in an ``INSERT ... SELECT ... WHERE NOT EXISTS`` with the
    JSON key as a bind parameter. Neither survives the Oracle rewriter — a SELECT with
    no FROM is a syntax error there, and ``->>:7`` is left untouched because only a
    literal key is mapped to JSON_VALUE — so on Oracle the submit raised instead of
    deduping, and the after-consolidation flush swallowed the error as a warning.
    """
    from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

    bank = await _make_bank(memory, request_context)
    async with memory._pool.acquire() as conn:
        mm_id = await _insert_auto_refresh_mm(conn, bank)
    _stall_worker(memory, monkeypatch)

    statements: list[str] = []
    for method in ("execute", "fetchval"):
        original = getattr(PostgresConnection, method)

        def _record(self, query, *args, _original=original, **kwargs):
            statements.append(query)
            return _original(self, query, *args, **kwargs)

        monkeypatch.setattr(PostgresConnection, method, _record)

    await memory.submit_async_refresh_mental_model(bank_id=bank, mental_model_id=mm_id, request_context=request_context)
    await memory.submit_async_refresh_mental_model(bank_id=bank, mental_model_id=mm_id, request_context=request_context)

    json_lookups = [s for s in statements if "task_payload->>" in s]
    assert json_lookups, "the in-flight lookup did not run"
    for statement in json_lookups:
        rewritten = _rewrite_pg_to_oracle(statement).query
        assert "JSON_VALUE(task_payload, '$.mental_model_id')" in rewritten
        assert "->>" not in rewritten

    inserts = [s for s in statements if "INSERT INTO" in s and "async_operations" in s]
    assert inserts, "the operation was never inserted"
    for statement in inserts:
        # A FROM-less SELECT is what Oracle rejects; the insert must stay a VALUES form.
        assert "VALUES" in statement
