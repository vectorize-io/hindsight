"""Per-document retain serialization, claim-time folding, and the append CAS.

Three layers guard concurrent appends to one document, and they are tested at
the level each one lives at:

* the **claim predicate** and the **fold**, at the queue — deterministic SQL
  behaviour, no LLM and no retain pipeline involved;
* the **fold planner**, as pure functions;
* the **append compare-and-swap**, end-to-end through the real retain pipeline,
  because the whole point of it is what happens across an extraction.

The end-to-end race test is the regression test for the original bug: three
concurrent appends to one document used to leave only one of them stored, with
all three reporting success.
"""

import asyncio
import hashlib
import json
import logging
import os
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api.engine.db.postgresql import PostgreSQLBackend
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.fold import (
    DEFAULT_MAX_FOLD_PEERS,
    FoldMember,
    FoldMemberRef,
    max_fold_peers_for_retry,
    merge_fold_contents,
    plan_retain_fold,
)
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.extensions.operation_validator import (
    OperationValidatorExtension,
    RetainResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)

_DIM = 384


def _ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _fake_count_tokens(text: str) -> int:
    """One 'token' per character — keeps budget assertions readable."""
    return len(text)


# ---------------------------------------------------------------------------
# Fold planner (pure)
# ---------------------------------------------------------------------------


def _member(
    op_id: str,
    content: str,
    *,
    tenant: str | None = "t1",
    api_key: str | None = "k1",
    update_mode: str | None = "append",
    document_tags: list[str] | None = None,
    strategy: str | None = None,
    has_file_metadata: bool = False,
) -> FoldMember:
    item: dict = {"content": content, "document_id": "doc"}
    if update_mode is not None:
        item["update_mode"] = update_mode
    return FoldMember(
        operation_id=op_id,
        contents=[item],
        tenant_id=tenant,
        api_key_id=api_key,
        document_tags=document_tags,
        strategy=strategy,
        has_file_metadata=has_file_metadata,
    )


def test_fold_takes_peers_in_submission_order():
    plan = plan_retain_fold(
        _member("op1", "one"),
        [_member("op2", "two"), _member("op3", "three")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == ["op2", "op3"]
    assert [c["content"] for c in merge_fold_contents(plan.members)] == ["one", "two", "three"]
    assert plan.deferred == []


def test_fold_stops_at_the_token_budget():
    """The fold must fit one orchestrator pass — see plan_retain_fold."""
    plan = plan_retain_fold(
        _member("op1", "a" * 10),
        [_member("op2", "b" * 10), _member("op3", "c" * 10)],
        max_peers=8,
        token_budget=25,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == ["op2"]
    assert plan.deferred == ["op3"]


def test_fold_never_skips_a_turn_to_reach_a_later_one():
    """A disqualified peer stops the fold — taking op3 would reorder the document."""
    plan = plan_retain_fold(
        _member("op1", "one"),
        [_member("op2", "two", api_key="other-key"), _member("op3", "three")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []
    assert plan.deferred == ["op2", "op3"]


def test_fold_respects_max_peers():
    peers = [_member(f"op{i}", "x") for i in range(2, 8)]
    plan = plan_retain_fold(
        _member("op1", "x"),
        peers,
        max_peers=2,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == ["op2", "op3"]
    assert plan.deferred == ["op4", "op5", "op6", "op7"]


def test_replace_mode_never_folds():
    """Replace is not cumulative: folding two of them would concatenate bodies
    where the second should supersede the first."""
    plan = plan_retain_fold(
        _member("op1", "first body", update_mode="replace"),
        [_member("op2", "second body", update_mode="replace")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == [], "a replace must run alone"
    assert plan.deferred == ["op2"]


def test_append_primary_does_not_swallow_a_replace_peer():
    """A replace queued behind an append must keep its wipe-and-store semantics."""
    plan = plan_retain_fold(
        _member("op1", "a turn"),
        [_member("op2", "a fresh body", update_mode="replace"), _member("op3", "another turn")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []
    assert plan.deferred == ["op2", "op3"]


def test_item_without_update_mode_is_not_foldable():
    """Absent update_mode means replace (the default) — not foldable."""
    plan = plan_retain_fold(
        _member("op1", "a turn"),
        [_member("op2", "plain item", update_mode=None)],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []


def test_fold_requires_matching_document_tags():
    """The execution applies the primary's tags to the document, so a
    differently-tagged peer would silently lose its own."""
    plan = plan_retain_fold(
        _member("op1", "one", document_tags=["a"]),
        [_member("op2", "two", document_tags=["a", "b"])],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []
    assert plan.deferred == ["op2"]


def test_fold_ignores_document_tag_ordering():
    plan = plan_retain_fold(
        _member("op1", "one", document_tags=["a", "b"]),
        [_member("op2", "two", document_tags=["b", "a"])],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == ["op2"]


def test_fold_requires_matching_strategy():
    plan = plan_retain_fold(
        _member("op1", "one", strategy="fast"),
        [_member("op2", "two", strategy="thorough")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []


def test_file_backed_retain_never_folds():
    """A converted upload attaches its storage key afterwards, and that step
    only runs for a single-item retain."""
    as_primary = plan_retain_fold(
        _member("op1", "converted", has_file_metadata=True),
        [_member("op2", "a turn")],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )
    assert as_primary.peer_ids == []

    as_peer = plan_retain_fold(
        _member("op1", "a turn"),
        [_member("op2", "converted", has_file_metadata=True)],
        max_peers=8,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )
    assert as_peer.peer_ids == []


def test_fold_width_narrows_with_retries_until_the_poison_runs_alone():
    assert max_fold_peers_for_retry(0) == DEFAULT_MAX_FOLD_PEERS
    assert max_fold_peers_for_retry(1) == DEFAULT_MAX_FOLD_PEERS // 2
    assert max_fold_peers_for_retry(2) == DEFAULT_MAX_FOLD_PEERS // 4
    # Deep enough into the retries, an operation stops taking anything with it.
    assert max_fold_peers_for_retry(99) == 0


def test_fold_with_zero_width_defers_every_peer():
    plan = plan_retain_fold(
        _member("op1", "one"),
        [_member("op2", "two")],
        max_peers=0,
        token_budget=1000,
        count_tokens=_fake_count_tokens,
    )

    assert plan.peer_ids == []
    assert plan.deferred == ["op2"]


def test_oversized_primary_still_runs_alone():
    """A single submission over budget must not be stuck forever unclaimable."""
    plan = plan_retain_fold(
        _member("op1", "a" * 500),
        [_member("op2", "b")],
        max_peers=8,
        token_budget=100,
        count_tokens=_fake_count_tokens,
    )

    assert plan.members[0].operation_id == "op1"
    assert plan.peer_ids == []


# ---------------------------------------------------------------------------
# Claim predicate + fold (against the database)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def isolated_ops_schema(pg0_db_url):
    """A private, migrated Postgres schema for this file's queue tests.

    ``ops.claim_tasks`` claims across the whole schema on the connection's
    search_path: whatever is claimable, it takes. Sharing ``public`` with the rest
    of the suite therefore makes these assertions unownable — another xdist worker
    running a real poller claims this file's rows before it can (its own claim then
    sees none of them, which reads as "the predicate excluded them"), and any test
    that deletes pending rows queue-wide removes them outright. Filtering the result
    to our own bank, which these tests already do, observes the shortfall but cannot
    prevent it.

    That is the same defect ``tests/test_worker.py`` and
    ``tests/test_claim_bank_serialization.py`` were given a private schema for
    (#3963); this file is the third one that claims queue-wide, and it flaked in CI
    with ``expected only the oldest same-document retain, got []``. Fix it the same
    way: one schema per worker, created + migrated once and dropped at session end,
    so "the whole schema" is only this file's rows.

    Only the queue tests take ``backend``; the end-to-end append tests build their
    own engine on ``memory_stub_emb`` and are unaffected.
    """
    from hindsight_api.engine.db import create_database_backend
    from hindsight_api.pg0 import resolve_database_url

    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    schema = f"retaindoc_iso_{worker}"

    async def _provision() -> str:
        url = await resolve_database_url(pg0_db_url)
        b = create_database_backend("postgresql")
        await b.initialize(url, min_size=1, max_size=2)
        try:
            async with b.get_pool().acquire() as conn:
                # Rebuild from scratch so a schema left by a crashed prior run
                # can't carry stale state into this session.
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await conn.execute(f'CREATE SCHEMA "{schema}"')
            # run_migrations is sync; call it with the loop running (as elsewhere
            # in the suite) — it builds banks/async_operations/etc. in the schema.
            b.run_migrations(url, schema=schema)
        finally:
            await b.shutdown()
        return url

    async def _drop(url: str) -> None:
        b = create_database_backend("postgresql")
        await b.initialize(url, min_size=1, max_size=2)
        try:
            async with b.get_pool().acquire() as conn:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await b.shutdown()

    loop = asyncio.new_event_loop()
    try:
        url = loop.run_until_complete(_provision())
    finally:
        loop.close()

    yield schema

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_drop(url))
    finally:
        loop.close()


@pytest_asyncio.fixture
async def backend(pg0_db_url, isolated_ops_schema):
    """A backend whose pool is pinned to this file's private schema."""
    from hindsight_api.pg0 import resolve_database_url

    async def _use_isolated_schema(conn):
        # init runs once per new connection and setup on every acquire (after
        # asyncpg's release-time RESET ALL), so this pins search_path for the
        # pool's whole lifetime: every unqualified table here resolves into the
        # private schema, and a claim never sees the shared public one.
        await conn.execute(f'SET search_path TO "{isolated_ops_schema}", public')

    pool = PostgreSQLBackend()
    await pool.initialize(
        await resolve_database_url(pg0_db_url), min_size=1, max_size=5, init_callback=_use_isolated_schema
    )
    yield pool
    await pool.shutdown()


async def _insert_retain_op(pool, bank_id: str, document_id: str | None, *, contents: list[dict]) -> str:
    """Queue a pending append-mode retain, as submit_async_retain would.

    Append mode because that is the shape folding applies to at all — a replace
    is not cumulative and runs alone (see plan_retain_fold).
    """
    contents = [
        {**item, "document_id": item.get("document_id", document_id), "update_mode": "append"} if document_id else item
        for item in contents
    ]
    operation_id = uuid.uuid4()
    payload = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": contents,
        "_tenant_id": "tenant-a",
        "_api_key_id": "key-a",
    }
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO async_operations
                (operation_id, bank_id, operation_type, result_metadata, status,
                 task_payload, serialization_key, created_at)
            VALUES ($1, $2, 'retain', '{}'::jsonb, 'pending', $3::jsonb, $4, now())
            """,
            operation_id,
            bank_id,
            json.dumps(payload),
            document_id,
        )
    # created_at ties would make claim order ambiguous; space the rows out.
    await asyncio.sleep(0.01)
    return str(operation_id)


def _own(rows, bank_id: str) -> list[str]:
    """Claimed operation ids belonging to this test's bank.

    The claim query is queue-wide by design and the test database is shared
    with every other module's pending work, so assertions have to be scoped.
    """
    return [str(r["operation_id"]) for r in rows if r["bank_id"] == bank_id]


class _Rollback(Exception):
    """Unwinds a test's claim transaction so the shared queue is left untouched."""


# A claim limit far above anything this file queues, so the ORDER BY created_at
# window can never be the reason a row is missing. What is under test is which
# rows are *claimable*, not how many a worker takes per cycle.
_UNBOUNDED_CLAIM = 1000


async def _claiming(backend, body):
    """Run ``body(conn)`` inside a claim transaction, then roll it back.

    Claiming is queue-wide by design: ``claim_tasks`` takes whatever is claimable
    in the schema. :func:`isolated_ops_schema` makes that only this file's rows, and
    rolling back on top of it leaves even those untouched — so each test starts from
    the queue it built itself while still exercising the real claim SQL, which is the
    point of these tests.
    """
    captured = {}
    try:
        async with backend.acquire() as conn:
            async with conn.transaction():
                captured["value"] = await body(conn)
                raise _Rollback
    except _Rollback:
        pass
    return captured["value"]


@pytest_asyncio.fixture
async def bank(backend):
    bank_id = f"test_serialization_{_ts()}"
    async with backend.acquire() as conn:
        await conn.execute("INSERT INTO banks (bank_id) VALUES ($1) ON CONFLICT DO NOTHING", bank_id)
    yield bank_id
    async with backend.acquire() as conn:
        await conn.execute("DELETE FROM async_operations WHERE bank_id = $1", bank_id)
        await conn.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)


async def _claim_rows(backend, conn, worker_id):
    """Claim, and hand back just the rows.

    ``claim_tasks`` also reports where the per-bank rotation got to (#3861);
    these tests are about per-document serialisation, which the rotation does
    not touch, so they only look at the rows.
    """
    claimed = await backend.ops.claim_tasks(conn, "async_operations", worker_id, {}, _UNBOUNDED_CLAIM)
    return claimed.rows


@pytest.mark.asyncio
async def test_claim_takes_only_one_retain_per_document(backend, bank):
    """Three queued appends to one document: only the oldest is claimable."""
    first = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "one"}])
    await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "two"}])
    await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "three"}])

    rows = await _claiming(backend, lambda conn: _claim_rows(backend, conn, "w1"))

    claimed = _own(rows, bank)
    assert claimed == [first], f"expected only the oldest same-document retain, got {claimed}"


@pytest.mark.asyncio
async def test_claim_does_not_serialize_across_documents(backend, bank):
    """Different documents are independent — a busy one must not block the fleet."""
    a = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "a1"}])
    b = await _insert_retain_op(backend, bank, "doc-b", contents=[{"content": "b1"}])
    c = await _insert_retain_op(backend, bank, None, contents=[{"content": "c1"}])

    rows = await _claiming(backend, lambda conn: _claim_rows(backend, conn, "w1"))

    assert set(_own(rows, bank)) == {a, b, c}


@pytest.mark.asyncio
async def test_claim_waits_for_a_processing_peer(backend, bank):
    """A document with a retain in flight yields nothing, even after a restart."""
    first = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "one"}])
    await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "two"}])
    async with backend.acquire() as conn:
        await conn.execute(
            "UPDATE async_operations SET status = 'processing' WHERE operation_id = $1",
            uuid.UUID(first),
        )

    rows = await _claiming(backend, lambda conn: _claim_rows(backend, conn, "w2"))

    assert _own(rows, bank) == [], "a document with a retain in flight must yield nothing"


@pytest.mark.asyncio
async def test_fetch_foldable_peers_returns_submission_order(backend, bank):
    first = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "one"}])
    second = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "two"}])
    third = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "three"}])
    await _insert_retain_op(backend, bank, "doc-b", contents=[{"content": "other"}])

    rows = await _claiming(
        backend,
        lambda conn: backend.ops.fetch_foldable_retain_peers(conn, "async_operations", bank, "doc-a", _UNBOUNDED_CLAIM),
    )

    ids = [str(r["operation_id"]) for r in rows]
    assert ids == [first, second, third], "peers must come back in (created_at, operation_id) order"


def _poller(backend, worker_id: str):
    from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
    from hindsight_api.worker.poller import WorkerPoller

    return WorkerPoller(
        backend=backend,
        worker_id=worker_id,
        executor=lambda _: None,
        tenant_extension=DefaultTenantExtension(config={}),
    )


@dataclass
class _ClaimedRetain:
    """One claimed retain and the peers folded into it."""

    operation_id: str
    folded_operation_ids: list[str]
    task_dict: dict


@dataclass
class _ClaimOutcome:
    """What a rolled-back claim observed for one bank."""

    tasks: list[_ClaimedRetain]
    statuses: list


async def _claim_own(backend, bank_id: str, worker_id: str) -> "_ClaimOutcome":
    """Claim, fold, and assert on this bank's rows without committing the claim.

    Drives the poller's real claim path (``claim_tasks`` then
    ``_fold_retain_peers``) rather than reimplementing it, so what is asserted
    is what a worker actually does — and rolls it back, so other modules'
    queued work is left alone.
    """
    poller = _poller(backend, worker_id)
    observed: dict = {}

    async def _body(conn):
        rows = await _claim_rows(backend, conn, worker_id)
        tasks = []
        for row in rows:
            if row["bank_id"] != bank_id:
                continue
            payload = row["task_payload"]
            task_dict = json.loads(payload) if isinstance(payload, str) else payload
            folded = await poller._fold_retain_peers(conn, "async_operations", row, task_dict)
            tasks.append(
                _ClaimedRetain(
                    operation_id=str(row["operation_id"]),
                    folded_operation_ids=folded,
                    task_dict=task_dict,
                )
            )
        # Read the claim's effect back inside the same transaction, before the
        # rollback: this is the state a committed claim would leave behind.
        observed["statuses"] = await conn.fetch(
            "SELECT operation_id, status, worker_id FROM async_operations WHERE bank_id = $1",
            bank_id,
        )
        return tasks

    tasks = await _claiming(backend, _body)
    return _ClaimOutcome(tasks=tasks, statuses=observed["statuses"])


@pytest.mark.asyncio
async def test_claim_folds_queued_peers_into_one_execution(backend, bank):
    """The whole backlog for a document is claimed and merged as one task."""
    first = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "turn one"}])
    second = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "turn two"}])
    third = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "turn three"}])

    claimed = await _claim_own(backend, bank, "w-fold")

    assert len(claimed.tasks) == 1, "the fold must produce a single execution"
    claim = claimed.tasks[0]
    assert claim.operation_id == first
    assert claim.folded_operation_ids == [second, third]
    assert [c["content"] for c in claim.task_dict["contents"]] == ["turn one", "turn two", "turn three"]
    assert [m["operation_id"] for m in claim.task_dict["_fold_members"]] == [first, second, third]

    # Invariant: every folded member is claimed by this worker, none left pending.
    assert {r["status"] for r in claimed.statuses} == {"processing"}
    assert {r["worker_id"] for r in claimed.statuses} == {"w-fold"}


@pytest.mark.asyncio
async def test_fold_stops_at_a_foreign_document(backend, bank):
    """Only peers for the same document join; another document is untouched."""
    first = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "a1"}])
    second = await _insert_retain_op(backend, bank, "doc-a", contents=[{"content": "a2"}])
    other = await _insert_retain_op(backend, bank, "doc-b", contents=[{"content": "b1"}])

    claimed = await _claim_own(backend, bank, "w-fold2")

    folds = {claim.operation_id: claim.folded_operation_ids for claim in claimed.tasks}
    assert folds[first] == [second]
    assert folds[other] == []


# ---------------------------------------------------------------------------
# Append compare-and-swap (end to end)
# ---------------------------------------------------------------------------


class _StubEmbeddings(Embeddings):
    """Deterministic, torch-free embeddings — same text, same vector.

    The local sentence-transformers model segfaults under heavy in-process
    async concurrency (a torch limitation, not a product issue: in production
    embeddings are served out-of-process), and these tests only care about what
    ends up stored.
    """

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return _DIM

    async def initialize(self) -> None:
        return None

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            vals: list[float] = []
            i = 0
            while len(vals) < _DIM:
                block = hashlib.sha256(seed + struct.pack("<I", i)).digest()
                for j in range(0, len(block), 4):
                    if len(vals) >= _DIM:
                        break
                    (u,) = struct.unpack("<I", block[j : j + 4])
                    vals.append((u % 100000) / 100000.0)
                i += 1
            out.append(vals)
        return out


@pytest_asyncio.fixture
async def memory_stub_emb(pg0_db_url, cross_encoder, query_analyzer):
    mem = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=cross_encoder,
        query_analyzer=query_analyzer,
        pool_min_size=1,
        pool_max_size=20,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await mem.initialize()
    # Auto-consolidation dominates wall-clock here and is irrelevant to what
    # these tests assert.
    mem._config_resolver._global_config.enable_auto_consolidation = False
    yield mem
    await asyncio.sleep(1)
    try:
        if mem._pool and not mem._pool._closing:
            await mem.close()
    except Exception:
        pass


async def _append(memory, request_context, bank_id: str, document_id: str, body: str) -> None:
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": body, "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_concurrent_appends_keep_every_turn(memory_stub_emb, request_context):
    """Regression: parallel appends used to drop all but one turn.

    Each append reads the stored document, concatenates onto it and reprocesses.
    Before the compare-and-swap, three appends that all read the same base each
    replaced the document with their own version — the last writer won, the
    other two turns were deleted, and all three calls returned success.
    """
    bank_id = f"test_append_race_{_ts()}"
    document_id = "conversation"

    await _append(memory_stub_emb, request_context, bank_id, document_id, "TURN_ONE Alice works at Google.")

    await asyncio.gather(
        _append(memory_stub_emb, request_context, bank_id, document_id, "TURN_TWO Bob works at Microsoft."),
        _append(memory_stub_emb, request_context, bank_id, document_id, "TURN_THREE Carol works at Amazon."),
        _append(memory_stub_emb, request_context, bank_id, document_id, "TURN_FOUR Dave works at Meta."),
    )

    doc = await memory_stub_emb.get_document(document_id, bank_id, request_context=request_context)
    text = doc["original_text"] or ""
    missing = [marker for marker in ("TURN_ONE", "TURN_TWO", "TURN_THREE", "TURN_FOUR") if marker not in text]
    assert not missing, f"appends lost {missing}; document is {text!r}"

    # The turns must also survive as memories, not just as document text: the
    # losing writers used to cascade-delete the winner's units.
    listing = await memory_stub_emb.list_memory_units(
        bank_id, document_id=document_id, limit=1000, request_context=request_context
    )
    stored = " ".join(item["text"] for item in listing["items"])
    assert "TURN_ONE" in stored and "TURN_FOUR" in stored


@pytest.mark.asyncio
async def test_sequential_appends_are_unaffected(memory_stub_emb, request_context):
    """The compare-and-swap must not cost the uncontended path anything."""
    bank_id = f"test_append_sequential_{_ts()}"
    document_id = "conversation"

    for marker in ("ONE", "TWO", "THREE"):
        await _append(memory_stub_emb, request_context, bank_id, document_id, f"TURN_{marker} content here.")

    doc = await memory_stub_emb.get_document(document_id, bank_id, request_context=request_context)
    text = doc["original_text"] or ""
    assert all(f"TURN_{m}" in text for m in ("ONE", "TWO", "THREE"))
    assert text.index("TURN_ONE") < text.index("TURN_TWO") < text.index("TURN_THREE")


# ---------------------------------------------------------------------------
# Post-retain hooks across a folded execution
# ---------------------------------------------------------------------------


class _RecordingValidator(OperationValidatorExtension):
    """Minimal stand-in for a metering extension — records what it is told.

    A real subclass rather than a duck-typed stub: the engine calls the
    pre-operation validators too, so a bare recorder would never be reached.
    """

    def __init__(self) -> None:
        super().__init__(config={})
        self.results: list[RetainResult] = []

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def on_retain_complete(self, result: RetainResult) -> None:
        self.results.append(result)


@pytest.mark.asyncio
async def test_folded_execution_reports_one_hook_per_operation(memory_stub_emb, request_context):
    """A fold must not make submissions invisible to metering, or double-count them.

    Each folded operation gets its own hook with its own content slice, and the
    LLM usage summed across the fold equals what the single execution spent —
    the same total the callers would have seen retaining one at a time.
    """
    validator = _RecordingValidator()
    memory_stub_emb._operation_validator = validator
    try:
        bank_id = f"test_fold_hooks_{_ts()}"
        document_id = "conversation"
        fold_members = [
            FoldMemberRef(operation_id="op-1", items_count=1),
            FoldMemberRef(operation_id="op-2", items_count=1),
            FoldMemberRef(operation_id="op-3", items_count=1),
        ]

        _, usage = await memory_stub_emb.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "TURN_ONE Alice works at Google.", "document_id": document_id},
                {"content": "TURN_TWO Bob works at Microsoft.", "document_id": document_id},
                {"content": "TURN_THREE Carol works at Amazon.", "document_id": document_id},
            ],
            request_context=request_context,
            return_usage=True,
            fold_members=fold_members,
        )
    finally:
        memory_stub_emb._operation_validator = None

    assert len(validator.results) == 3, "one post-retain hook per folded operation"

    # Totals are preserved: the fold reports exactly the execution's usage.
    assert sum(r.llm_total_tokens for r in validator.results) == usage.total_tokens
    assert sum(r.llm_input_tokens for r in validator.results) == usage.input_tokens
    assert sum(r.llm_output_tokens for r in validator.results) == usage.output_tokens

    # ...carried by the first member, with the rest at zero (see RetainResult.folded_with).
    assert validator.results[0].llm_total_tokens == usage.total_tokens
    assert all(r.llm_total_tokens == 0 for r in validator.results[1:])

    # Every member knows the operations it ran with, itself excluded.
    assert validator.results[0].folded_with == ["op-2", "op-3"]
    assert validator.results[1].folded_with == ["op-1", "op-3"]
    assert validator.results[2].folded_with == ["op-1", "op-2"]

    # Each member sees only the content it submitted.
    assert [r.contents[0]["content"][:8] for r in validator.results] == ["TURN_ONE", "TURN_TWO", "TURN_THR"]


@pytest.mark.asyncio
async def test_unfolded_retain_reports_one_hook_with_full_usage(memory_stub_emb, request_context):
    """The ordinary path is untouched: one hook, all the usage, no folded_with."""
    validator = _RecordingValidator()
    memory_stub_emb._operation_validator = validator
    try:
        bank_id = f"test_unfolded_hooks_{_ts()}"
        _, usage = await memory_stub_emb.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice works at Google.", "document_id": "doc"}],
            request_context=request_context,
            return_usage=True,
        )
    finally:
        memory_stub_emb._operation_validator = None

    assert len(validator.results) == 1
    assert validator.results[0].llm_total_tokens == usage.total_tokens
    assert validator.results[0].folded_with is None
