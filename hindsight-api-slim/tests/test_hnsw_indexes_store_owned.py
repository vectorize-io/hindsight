"""Per-bank vector indexes must not be created for a bank whose memories live outside SQL.

A store-owned bank writes no ``memory_units`` rows, so its three partial indexes can
only ever be empty. Empty is not free: Postgres plans against every index on a relation,
so they are charged to every OTHER statement that names the shared table. A tenant with
27,315 store-owned banks carried 82,795 such indexes and paid ~975 ms of planning for a
query returning zero rows (#4615).

There are **two** builders, and the bug is only fixed if both stop: bank creation
(``create_bank_vector_indexes``) and the reconcile behind ``repair-bank`` /
``vector_index_maintenance`` (``plan_bank_vector_indexes``). Once creation stops, a
store-owned bank reads to the reconcile exactly like one whose CREATE INDEX lost a
deadlock — all three missing — so a single ``repair-bank --all`` would put all 82,795
back. That is the sibling this suite exists to keep honest.

Creation only, in both directions: nothing here drops an index a bank already carries.
Turning a memories extension on makes every existing bank store-owned at once, and
shedding tens of thousands of indexes is an operator's decision with its own timing.
So the suite also pins that a reconcile leaves an existing bank's indexes alone.

Runs via: uv run pytest tests/test_hnsw_indexes_store_owned.py -v
"""

import uuid

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api.admin.cli import _run_repair_bank
from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine import vector_index_health
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.retain import bank_utils
from hindsight_api.engine.retain.bank_utils import _vector_index_clause
from hindsight_api.engine.transfer import export_bank
from hindsight_api.engine.vector_index_health import plan_bank_vector_indexes, reconcile_bank_vector_indexes
from tests.test_memories_extension import InMemoryMemories

# The suite runs against the base schema, as test_repair_bank_vector_indexes.py does.
# Named rather than repeated so the reconcile calls and the hand-built index below
# cannot drift apart into asserting about two different schemas.
_TEST_SCHEMA = "public"


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    """Pin eager mode on for every test here, rather than inheriting the default.

    These tests are about WHICH banks get indexes, not about the size threshold, and
    the two policies are alternatives: with ``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS``
    set, creation builds nothing for *any* bank — which would fail the SQL-owned
    assertions and, worse, make the store-owned ones pass vacuously. Patched on every
    module that imported the helper by name; a missed one silently restores the
    ambient default (the same trap ``test_hnsw_indexes.py::threshold_set`` documents).
    """
    for module in (vector_index_health, bank_utils, memory_engine_module):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: True)


@pytest.fixture(autouse=True)
def per_bank_backend():
    """Skip the whole module on a backend that has no per-bank indexes to speak of.

    Applied to every test, not just the ones asserting three indexes: on AlloyDB ScaNN
    or Oracle the negative assertions (``== []``) hold no matter what the store guard
    does, so they would report a pass for a guard they never exercised.
    """
    if _vector_index_clause() is None:
        pytest.skip("configured vector backend does not use per-bank vector indexes")


@pytest.fixture
def store_owned():
    """Route the store to one that owns its memory rows.

    Through ``set_memories`` rather than by patching ``get_memories``: that is the
    repo's own override hook, it also resets the cached graph retriever (chosen from
    the store), and it reaches ``admin/cli.py`` — the one module that binds
    ``get_memories`` at import time, so a monkeypatch of the accessor would leave a
    CLI-driven test silently on the real Postgres store.
    """
    previous = memories_mod.get_memories()
    store = InMemoryMemories()
    memories_mod.set_memories(store)
    try:
        yield store
    finally:
        # Back to whatever the suite resolved, not to None: set_memories also clears
        # the cached graph retriever, which is chosen from the store.
        memories_mod.set_memories(previous)


class _Unanswerable:
    """The real store, except that the capability probe raises.

    Delegates everything else rather than stubbing it: bank creation calls several
    other store methods, and a fake thin enough to isolate the probe would fail on
    those instead — testing the fake, not the fallback.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def store_owned_for(self, bank_id: str) -> bool:
        raise RuntimeError("router is unreachable")


async def _bank_indexes(pool, bank_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = $1
              AND tablename = 'memory_units'
              AND indexname LIKE 'idx_mu_emb_%'
              -- strpos, not LIKE: every bank id here contains underscores, which LIKE
              -- reads as single-char wildcards. This is the same exact-match shape
              -- _index_health uses in production.
              AND strpos(indexdef, $2) > 0
            ORDER BY indexname
            """,
            _TEST_SCHEMA,
            f"bank_id = '{bank_id}'",
        )
    return [row["indexname"] for row in rows]


async def test_store_owned_bank_gets_no_vector_indexes(memory, request_context, store_owned):
    """The fix. Three indexes on a table this bank will never write a row to."""
    bank_id = f"test_so_none_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        indexes = await _bank_indexes(memory._pool, bank_id)
        assert indexes == [], f"a store-owned bank must get no per-bank vector indexes, got: {indexes}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_sql_owned_bank_still_gets_all_three(memory, request_context):
    """The silent half: inverting the condition strips ANN from every ordinary bank.

    Nothing fails when that happens — recall falls back to the exact ``(bank_id,
    fact_type)`` scan and returns the same rows, more slowly — so only an explicit
    assertion catches it. Overlaps ``test_repair_bank_vector_indexes.py``'s
    ``test_bank_creation_builds_all_three_indexes`` on purpose: it is the in-file
    control that gives the store-owned assertion above its meaning.
    """
    bank_id = f"test_so_sql_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_a_store_that_cannot_answer_still_gets_its_indexes(memory, request_context):
    """A router that raises must fall back to SQL-backed, not to a failed bank create.

    ``store_owned_for`` is an extension's method, so it can raise. This runs inside the
    bank-create transaction, so an exception escaping here fails an ordinary first
    retain to a new bank with a 500 — and the safe direction is the pre-#4615 one:
    build the indexes, because a bank that turns out to be SQL-owned without them
    silently loses ANN, while three unused indexes on one bank cost almost nothing.
    """

    real_store = memories_mod.get_memories()
    bank_id = f"test_so_raises_{uuid.uuid4().hex[:8]}"
    try:
        # Scoped to bank creation only. The probe is consulted on other paths too —
        # delete_bank among them — and leaving it raising would fail this test in its
        # own teardown, on a call that is not what is under test.
        memories_mod.set_memories(_Unanswerable(real_store))
        try:
            await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
        finally:
            memories_mod.set_memories(real_store)

        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_repair_does_not_rebuild_what_creation_declined_to_build(memory, request_context, store_owned):
    """The second builder, and the one that would silently undo the whole fix.

    With creation guarded, a store-owned bank has no indexes — which is indistinguishable,
    to the reconcile, from a bank whose CREATE INDEX lost a deadlock or that was restored
    around the gate. At the default threshold entitlement does not consult row counts, so
    without its own guard the eager branch puts all three fact types in ``to_build`` and
    one ``repair-bank --all`` rebuilds every index #4615 is about: 82,795 on that tenant,
    from a command that reads as a repair.

    Asserted on the PLAN, not just on the catalog afterwards, so the test names the
    decision rather than a side effect of it.
    """
    bank_id = f"test_so_norebuild_{uuid.uuid4().hex[:8]}"
    index_clause = _vector_index_clause()
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
        assert await _bank_indexes(memory._pool, bank_id) == [], "setup: creation should have built nothing"

        async with memory._pool.acquire() as conn:
            plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)
            assert plan.to_build == [], f"repair must not rebuild a store-owned bank's indexes, got {plan.to_build}"

            result = await reconcile_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id, index_clause)

        assert result.created == 0, f"reconcile built indexes for a store-owned bank: {result}"
        assert await _bank_indexes(memory._pool, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_indexes_an_existing_store_owned_bank_already_has_are_left_alone(memory, request_context, monkeypatch):
    """Adopting a memories store must not silently drop what is already built.

    The bank is created SQL-owned so it really gets its three indexes, then the store
    takes it over — which is what flipping the extension on does to every existing bank
    at once. Dropping tens of thousands of indexes is an operator's decision with its
    own timing: this change only stops NEW ones being created, so the reconcile must
    leave these exactly where they are.

    Asserted with a size threshold SET, deliberately, against the eager default the rest
    of this file pins. Eager mode never populates ``to_drop`` for any bank, so asserting
    it there is a tautology that would pass with the guard deleted. With a threshold the
    drop is live: a store-owned bank holds zero rows, which is below any keep bound, so
    the branch below would shed all three on the next write that queues a reconcile.
    That is the configuration where "nothing is dropped" is a real promise.

    The complement of the test above: that one pins ``to_build`` empty, this one pins
    ``to_drop`` empty, and together they are what "an empty plan" has to mean.
    """
    for module in (vector_index_health, bank_utils, memory_engine_module):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: False)
    monkeypatch.setattr(vector_index_health, "per_bank_index_build_bound", lambda: 4)
    monkeypatch.setattr(vector_index_health, "per_bank_index_keep_bound", lambda: 2)

    index_clause = _vector_index_clause()
    bank_id = f"test_so_keep_{uuid.uuid4().hex[:8]}"
    previous_store = memories_mod.get_memories()
    try:
        # Built by hand: with the threshold on, creation no longer makes them, and this
        # test is about a bank that already HAS them when the store takes over.
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
        backend = await memory._get_backend()
        async with memory._pool.acquire() as conn:
            internal_id = str(await conn.fetchval("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id))
            await backend.ops.create_bank_vector_indexes(
                conn,
                f"{_TEST_SCHEMA}.memory_units",
                bank_id,
                internal_id,
                index_clause,
                bank_utils._BANK_INDEX_FACT_TYPES,
            )
        before = await _bank_indexes(memory._pool, bank_id)
        assert len(before) == 3, f"setup: the bank needs indexes that could be dropped, got {before}"

        # Flipped only now, so the bank is built SQL-owned and adopted afterwards.
        # One instance, held: a fresh store per call happens to work while only the
        # stateless capability probe is read, and turns any later store write in this
        # test into a mystery product bug.
        adopted = InMemoryMemories()
        memories_mod.set_memories(adopted)

        async with memory._pool.acquire() as conn:
            plan = await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)
            assert plan.to_drop == [], f"a reconcile must not shed existing indexes, got {plan.to_drop}"

            result = await reconcile_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id, index_clause)

        assert result.dropped == 0, f"a store-owned bank's existing indexes must survive a reconcile, got {result}"
        assert await _bank_indexes(memory._pool, bank_id) == before
    finally:
        memories_mod.set_memories(previous_store)
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_importing_a_bank_into_a_store_owned_deployment_creates_no_indexes(memory, request_context, store_owned):
    """The import seam, driven through the real ``import_bank_async``.

    Import restores the ``banks`` row directly, so the fresh-INSERT gate never fires and
    it builds the indexes itself (#2645) — a second call site for the guard, and the one
    a creation-only fix would miss. Exercised end to end rather than by hand-calling
    ``create_bank_vector_indexes``, so it would also catch import bypassing that helper
    or passing it the wrong bank id.
    """
    bank_id = f"test_so_import_{uuid.uuid4().hex[:8]}"
    backend = await memory._get_backend()
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank_id)

        # Deleted then restored into the same id, which is the shape that reaches the
        # explicit build: the restored banks row exists before the bank is set up.
        await memory.delete_bank(bank_id, request_context=request_context)
        result = await memory.import_bank_async(archive, request_context)
        assert result.bank_id == bank_id

        indexes = await _bank_indexes(memory._pool, bank_id)
        assert indexes == [], f"import into a store-owned deployment must build no indexes, got: {indexes}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_repair_refuses_to_guess_when_the_store_cannot_answer(memory, request_context):
    """The reconcile's fallback is the opposite of bank creation's, and must stay so.

    Bank creation guesses SQL-backed when the probe raises, because its downside is
    three unused indexes on one bank. Here the downside is the whole bug: a transient
    router blip partway through ``repair-bank --all`` would classify every bank it
    failed on as SQL-backed and rebuild all three indexes for each — 82,795 on the
    tenant from #4615 — and the command would still exit 0, reading as a successful
    repair. The only trace would be a log line, and the admin CLI logs at INFO.

    So planning must RAISE rather than return a plan. The sweep then reports that
    schema skipped and exits non-zero, and the write path logs and does nothing —
    instead of quietly undoing the fix.
    """
    bank_id = f"test_so_blip_{uuid.uuid4().hex[:8]}"
    real_store = memories_mod.get_memories()
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        memories_mod.set_memories(_Unanswerable(real_store))
        try:
            with pytest.raises(RuntimeError, match="router is unreachable"):
                async with memory._pool.acquire() as conn:
                    await plan_bank_vector_indexes(conn, _TEST_SCHEMA, bank_id)
        finally:
            memories_mod.set_memories(real_store)

        # And nothing was built on the way out.
        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_the_sweep_skips_a_schema_it_cannot_classify_instead_of_dying(memory, request_context, pg0_db_url):
    """The other half of the contract above, at the command level.

    Planning raises so the sweep cannot silently rebuild — but the raise must be
    caught per SCHEMA, not escape the whole command. It used to: the guard in
    ``_run_repair_bank`` wrapped only ``list_bank_ids``, so one unclassifiable bank
    took down the entire run from inside a list comprehension. A deployment whose
    admin process cannot reach the memories store then repaired NOTHING — including
    the ordinary SQL-owned banks the operator ran the command for — and got a bare
    traceback naming neither the bank nor the schema.

    So: the schema is reported skipped, and `repair-bank` exits non-zero on it.
    """
    bank_id = f"test_so_sweep_{uuid.uuid4().hex[:8]}"
    real_store = memories_mod.get_memories()
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)

        memories_mod.set_memories(_Unanswerable(real_store))
        try:
            sweep = await _run_repair_bank(
                pg0_db_url, base_schema=_TEST_SCHEMA, schema=_TEST_SCHEMA, bank_id=bank_id, dry_run=False
            )
        finally:
            memories_mod.set_memories(real_store)

        assert sweep.skipped_schemas == [_TEST_SCHEMA], f"the schema should be reported skipped, got {sweep}"
        assert sweep.banks == [], "a schema that could not be classified must contribute no results"
        # And the bank kept exactly what it had — nothing rebuilt, nothing dropped.
        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
