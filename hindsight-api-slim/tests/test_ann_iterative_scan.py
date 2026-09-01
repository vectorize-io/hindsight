"""ANN scan depth follows each query's LIMIT, instead of a fixed candidate list.

An ANN scan explores a bounded candidate list and returns what it found, so that list
— not the SQL LIMIT — decided how many rows a recall arm could come back with. On
pgvector it is ``hnsw.ef_search``, pinned at 200 for the connection's lifetime by the
pool's init callback, which silently capped every recall at ~200 dense candidates
however large the budget: the budget moved the SQL and nothing else.

``hnsw.iterative_scan`` (pgvector 0.8+) resolves that without a per-query statement.
With it on, a drained candidate list is refilled in ``ef_search``-sized rounds until
the query's LIMIT is met, so depth follows the budget on the connection settings the
pool already applies — which matters behind a transaction-mode pooler, where a
session GUC issued between statements can land on a different backend.

Covers:
- Both tuning profiles: recall resumes, retain-side link probing explicitly does not.
- That the arms fetch what both their consumers read (the semantic list, and the graph
  arm's seeds) and that ``search`` issues no session statement of its own.
"""

from __future__ import annotations

import dataclasses

import random
import uuid
from types import SimpleNamespace

import pytest

from hindsight_api._vector_index import ann_max_scan_tuples, ann_search_tuning_settings
from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.engine.search import bm25_term_selection as bm25_mod
from hindsight_api.engine.search import retrieval as retrieval_mod
from hindsight_api.engine.search.link_expansion_retrieval import GRAPH_SEED_LIMIT

BUDGET_MID = 300


# ---------------------------------------------------------------------------
# Tuning profiles
# ---------------------------------------------------------------------------


def test_recall_connections_resume_the_scan():
    """Without this the scan stops at ef_search rows and the budget cannot reach the index."""
    settings = dict(ann_search_tuning_settings("pgvector", kind="high_recall"))

    assert settings["hnsw.iterative_scan"] == "strict_order"
    # relaxed_order would return rows out of distance order, which the Python-side
    # trim in retrieve_semantic_bm25_combined_sql assumes it can rely on.
    assert settings["hnsw.ef_search"] == "200"
    # Bounded, so a heavily filtered query cannot resume its way into a huge scan.
    assert settings["hnsw.max_scan_tuples"] == str(ann_max_scan_tuples())
    assert ann_max_scan_tuples() < 20000  # pgvector's default


def test_retain_link_probing_does_not_resume():
    """Link probing is tuned for latency; resuming past its small list would defeat that."""
    settings = dict(ann_search_tuning_settings("pgvector", kind="low_latency"))

    assert settings["hnsw.iterative_scan"] == "off"
    assert settings["hnsw.ef_search"] == "60"


def test_backends_without_the_knobs_get_no_settings():
    for ext in ("vchord", "pgvectorscale", "pg_diskann", "scann"):
        assert ann_search_tuning_settings(ext, kind="high_recall") == ()
        assert ann_search_tuning_settings(ext, kind="low_latency") == ()


# ---------------------------------------------------------------------------
# What the arms ask for
# ---------------------------------------------------------------------------


class FakeDialect:
    """Captures what each semantic arm asks the index for."""

    def __init__(self):
        self.fetch_limits: list[int] = []

    def build_semantic_arm(self, *, fetch_limit, **kwargs):
        self.fetch_limits.append(fetch_limit)
        return "SELECT 'semantic' AS source"

    def build_bm25_arm(self, **kwargs):
        return "SELECT 'bm25' AS source"

    def prepare_bm25_text(self, tokens, query_text, **kwargs):
        return " | ".join(tokens)


class FakeConn:
    """Fails the test if recall issues a session setting or opens a transaction."""

    backend_type = "postgresql"

    def transaction(self):
        raise AssertionError("recall must not open a transaction to tune the scan")

    async def execute(self, sql, *params):
        raise AssertionError(f"recall must not issue session settings per query: {sql!r}")

    async def fetch(self, query, *params):
        return []


@pytest.fixture
def search_path(monkeypatch):
    dialect = FakeDialect()
    from hindsight_api.config import _get_raw_config

    config = dataclasses.replace(
        _get_raw_config(),
        semantic_min_similarity=0.0,
        bm25_min_score=0.0,
        text_search_extension="native",
        text_search_extension_native_language="english",
    )
    monkeypatch.setattr(retrieval_mod, "create_sql_dialect", lambda backend: dialect)
    monkeypatch.setattr(retrieval_mod, "get_config", lambda: config)
    monkeypatch.setattr(retrieval_mod, "fq_table", lambda name: name)
    monkeypatch.setattr(bm25_mod, "get_current_schema", lambda: None)
    return dialect


async def _search(conn, limit: int, **kwargs):
    await PostgresMemories({}).search(
        conn=conn,
        bank_id="bank-1",
        fact_types=["world", "experience"],
        query_embedding="[0.0]",
        query_text="alpha beta",
        limit=limit,
        **kwargs,
    )


async def test_arms_ask_for_exactly_the_rows_they_keep(search_path):
    """No row over-fetch: the arms are ordered by distance, so trimming N of 5N in
    Python returned precisely what LIMIT N would have — the surplus was fetched,
    decoded and dropped unread."""
    await _search(FakeConn(), BUDGET_MID)

    assert search_path.fetch_limits == [BUDGET_MID, BUDGET_MID]  # one arm per fact_type


async def test_small_budget_still_covers_the_graph_arms_seeds(search_path):
    """The graph arm reads its entry points from these same rows, so a budget below
    GRAPH_SEED_LIMIT must not starve it."""
    await _search(FakeConn(), 5, graph_seed_min_similarity=0.3)

    assert search_path.fetch_limits == [GRAPH_SEED_LIMIT, GRAPH_SEED_LIMIT]


async def test_no_seed_threshold_means_no_seed_floor(search_path):
    """With the graph arm off, nothing reads past the semantic list itself."""
    await _search(FakeConn(), 5)

    assert search_path.fetch_limits == [5, 5]


# ---------------------------------------------------------------------------
# The property the change delivers, against a real index
# ---------------------------------------------------------------------------

EMBED_DIM = 384
# Enough to exceed the 200-row candidate list at a budget of 400, and no more: this
# runs alongside timing-sensitive tests, and a bulk load large enough to saturate the
# database starves them.
_ROWS = 600


def _near_query_vector(seed: int) -> str:
    """A distinct vector from one tight cluster.

    Clustered rather than uniformly random on purpose: an HNSW graph over scattered
    vectors is sparsely connected, so a resumed scan exhausts the reachable set before
    it reaches the requested budget and the test measures graph connectivity instead of
    the setting under test.
    """
    rng = random.Random(seed)
    values = [1.0] + [rng.uniform(-0.05, 0.05) for _ in range(EMBED_DIM - 1)]
    norm = sum(v * v for v in values) ** 0.5
    return "[" + ",".join(f"{v / norm:.5f}" for v in values) + "]"


@pytest.mark.asyncio
async def test_the_kill_switch_flips_real_retrieval_depth(memory, request_context, ann_config):
    """End to end, through the pool: on, the budget reaches the index; off, it does not.

    Both halves matter. On is the fix — with iterative scans off the ground-layer search
    runs once and the scan ends when its ef_search-sized list drains, so the arm cannot
    return more than ~200 rows however large the recall budget. Off is the operational
    revert, and it has to land on exactly that pre-existing behaviour rather than some
    third state nobody runs.

    Driven by the environment variable through the pool's own session setup, not by
    setting the GUCs by hand, so it covers the path production actually takes. Rows and
    index are built directly: the property belongs to the index scan, and going through
    retain would drag in extraction and consolidation.
    """
    from hindsight_api._vector_index import uses_per_bank_vector_indexes
    from hindsight_api.config import get_config
    from hindsight_api.engine.search.retrieval import retrieve_semantic_bm25_combined_sql
    from hindsight_api.engine.retain.bank_utils import (
        _bank_index_name,
        create_bank_vector_indexes,
        get_or_create_bank_profile,
    )
    from hindsight_api.engine.task_backend import fq_table

    if not uses_per_bank_vector_indexes(get_config().vector_extension):
        pytest.skip("backend uses a global vector index, so there is no per-bank ANN index to name")

    bank_id = f"test_iter_scan_{uuid.uuid4().hex[:8]}"
    budget = 400  # deliberately above the standing ef_search of 200
    # Creating the bank also builds its per-(bank, fact_type) partial vector index —
    # the same one recall uses — so this exercises the production index, not a stand-in.
    await get_or_create_bank_profile(memory._backend, bank_id)
    pool = await memory._get_pool()
    probe = _near_query_vector(0)
    table = fq_table("memory_units")
    try:
        async with pool.acquire() as conn:
            internal_id = await conn.fetchval(
                f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", bank_id
            )
            assert internal_id is not None, f"bank {bank_id} was created without an internal_id"
            # The plan is asserted against this exact name below. "Index Scan" alone is
            # not enough: memory_units carries ~19 btrees, any of which can feed a scan
            # plus a Sort and satisfy the substring while returning every row regardless
            # of the candidate list — the very plan this test exists to exclude (#3619).
            index_name = _bank_index_name("world", str(internal_id))
            # Unqualified, like the table names above: resolved through the session's
            # search_path, so it names the same index the query below would plan against.
            if not await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", index_name):
                # Only reachable with HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS set, where a
                # fresh bank has not earned its indexes and vector_index_maintenance owns
                # them. Build them here so the assertion measures the plan rather than the
                # threshold. Deliberately not unconditional: at the default of 0 this
                # branch does not run, so the test still fails if bank creation ever stops
                # building the index recall depends on.
                ann_config("vector_index_min_rows", 0)
                await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=memory._backend.ops)

            await conn.executemany(
                f"INSERT INTO {table} (bank_id, text, fact_type, embedding) VALUES ($1, $2, 'world', $3::vector)",
                [(bank_id, f"filler fact {i}", _near_query_vector(i)) for i in range(_ROWS)],
            )
            await conn.execute(f"ANALYZE {table}")

        async def semantic_rows(iterative: bool) -> int:
            ann_config("ann_iterative_scan", iterative)
            # The pool re-applies its session settings on every acquire, so a fresh
            # connection resolves the flag again rather than inheriting the value the
            # process started with.
            async with pool.acquire() as conn:
                # The property under test belongs to the ANN scan, not to the planner's
                # choice: on a table this size a full scan plus a sort is genuinely
                # cheaper, and inflating the fixture until ANN wins would only make the
                # test slow. Discourage both alternatives so the ordered path is taken.
                await conn.execute("SET enable_seqscan = off")
                await conn.execute("SET enable_sort = off")
                plan = "\n".join(
                    r[0]
                    for r in await conn.fetch(
                        f"EXPLAIN SELECT id FROM {table} WHERE bank_id = $1 AND fact_type = 'world' "
                        f"AND embedding IS NOT NULL ORDER BY embedding <=> $2::vector LIMIT {budget}",
                        bank_id,
                        probe,
                    )
                )
                # An ANN scan emits rows already ordered, so a Sort node means the
                # planner fell back to a btree plus a top-N sort.
                assert index_name in plan, f"expected an ANN scan using {index_name}, got:\n{plan}"
                assert "Sort" not in plan, f"expected an ANN scan without a Sort, got:\n{plan}"
                result = await retrieve_semantic_bm25_combined_sql(
                    conn, probe, "", bank_id, ["world"], budget, min_semantic=0.0
                )
                return len(result["world"].semantic)

        with_resume = await semantic_rows(True)
        without_resume = await semantic_rows(False)

        # On: the budget reaches the index.
        assert with_resume == budget, f"expected the full budget, got {with_resume}"
        # Off: capped by the candidate list, exactly as before the fix existed.
        assert without_resume <= 250, f"expected the scan to stop at ~ef_search, got {without_resume}"
        assert without_resume < with_resume
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Operational controls
# ---------------------------------------------------------------------------


@pytest.fixture
def ann_config(monkeypatch):
    """Override an ANN config field for one test, without disturbing anything else.

    Set on the cached config instance rather than by setting the env var and clearing
    the cache: clearing it is process-wide, so every engine built earlier in the
    session would silently start resolving a config rebuilt from the current
    environment. monkeypatch restores the attribute at teardown.
    """
    from hindsight_api.config import _get_raw_config

    def _set(field: str, value) -> None:
        monkeypatch.setattr(_get_raw_config(), field, value)

    return _set


def test_the_kill_switch_removes_the_resume_settings(ann_config):
    """Turning it off must leave a connection exactly as it was before the feature.

    Dropping the GUCs rather than sending iterative_scan=off matters for two reasons:
    a pgvector older than 0.8 rejects them outright (it reserves the "hnsw." prefix),
    and an operator who pinned values server-side keeps them.
    """
    ann_config("ann_iterative_scan", False)
    settings = ann_search_tuning_settings("pgvector", kind="high_recall")

    assert settings == (("hnsw.ef_search", "200"),)


def test_the_scan_ceiling_is_tunable(ann_config):
    """The dial between the previous behaviour and full budget depth."""
    ann_config("ann_max_scan_tuples", 1500)
    settings = dict(ann_search_tuning_settings("pgvector", kind="high_recall"))

    assert settings["hnsw.max_scan_tuples"] == "1500"
    assert settings["hnsw.iterative_scan"] == "strict_order"


def test_an_unreadable_ceiling_is_rejected_at_config_load(monkeypatch):
    """Parsing and validation belong to HindsightConfig, not to this module."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_ANN_MAX_SCAN_TUPLES", "not-a-number")
    with pytest.raises(ValueError):
        HindsightConfig.from_env()


def test_retain_probing_is_unaffected_by_the_switch(ann_config):
    """Link probing never resumed; the switch has nothing to take from it."""
    ann_config("ann_iterative_scan", False)
    settings = dict(ann_search_tuning_settings("pgvector", kind="low_latency"))

    assert settings == {"hnsw.ef_search": "60"}
