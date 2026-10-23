"""Every observation write keeps ``search_vector`` in step with the text it writes (#5299).

VectorChord's ``search_vector`` is a plain ``bm25vector`` column nothing fills on its own, so a
rewrite or dedup fold that changes the text without re-tokenizing it leaves keyword search
matching the old wording. Each write must build the column the same way the insert does.
"""

import uuid
from types import SimpleNamespace

import pytest

from hindsight_api import config as hs_config
from hindsight_api.engine import schema
from hindsight_api.engine.consolidation.consolidator import _TemporalBounds
from hindsight_api.engine.memories.pg import admin, consolidation

_BOUNDS = _TemporalBounds(event_date=None, occurred_start=None, occurred_end=None, mentioned_at=None)

EXPECTED = {
    "vchord": "search_vector = tokenize(COALESCE($1, ''), 'llmlingua2')::bm25_catalog.bm25vector",
    "native": "search_vector = to_tsvector('english'::regconfig, COALESCE($1, ''))",
}


class _CapturingConn:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def execute_rows_affected(self, query, *params):
        self.queries.append(query)
        return 1

    async def fetchval(self, query, *params):
        self.queries.append(query)
        return uuid.uuid4()

    async def fetchrow(self, query, *params):
        self.queries.append(query)
        if query.lstrip().startswith("SELECT source_memory_ids"):
            return {"source_memory_ids": [uuid.uuid4()]}
        return {"id": uuid.uuid4()}

    async def fetch(self, query, *params):
        # lock_live_memory_ids: report every requested source as live.
        return [{"id": u} for u in params[0]]


@pytest.fixture
def backend(monkeypatch, request):
    ext, oracle = request.param
    cfg = SimpleNamespace(text_search_extension=ext, text_search_extension_native_language="english")
    monkeypatch.setattr(consolidation, "get_config", lambda: cfg)
    monkeypatch.setattr(hs_config, "get_config", lambda: cfg)
    monkeypatch.setattr(schema, "_is_oracle", lambda: oracle)
    return ext, oracle


async def _run_every_write(conn: _CapturingConn) -> None:
    fq = lambda t: t  # noqa: E731
    obs = str(uuid.uuid4())
    await consolidation.rewrite_observation(
        conn=conn,
        fq_table=fq,
        observation_id=obs,
        text="new",
        embedding="[1]",
        source_memory_ids=[uuid.uuid4()],
        tags=[],
        bounds=_BOUNDS,
    )
    await consolidation.fold_sources_into_observation(
        conn=conn,
        fq_table=fq,
        observation_id=obs,
        expected_text="old",
        merged_text="new",
        source_memory_ids=[uuid.uuid4()],
        bounds=_BOUNDS,
    )
    await consolidation.fold_observation_into_twin(
        conn=conn,
        fq_table=fq,
        bank_id="b",
        observation_id=obs,
        observation_text="old",
        twin_id=str(uuid.uuid4()),
        twin_text="twin",
        merged_text="new",
    )
    await consolidation.insert_observation(
        conn=conn,
        ops=SimpleNamespace(uses_observation_sources_table=False),
        fq_table=fq,
        bank_id="b",
        observation_id=uuid.uuid4(),
        text="new",
        embedding="[1]",
        source_memory_ids=[],
        tags=[],
        event_date=None,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
    )


def _text_writes(conn: _CapturingConn) -> list[str]:
    return [q for q in conn.queries if "UPDATE" in q or "INSERT" in q]


@pytest.mark.parametrize("backend", [("vchord", False), ("native", False)], indirect=True)
async def test_every_observation_write_rebuilds_search_vector(backend):
    ext, _ = backend
    conn = _CapturingConn()
    await _run_every_write(conn)
    writes = _text_writes(conn)
    assert len(writes) == 4
    # The rewrite and both folds bind the new text as $1; the insert binds it as $3.
    for q in writes[:3]:
        assert EXPECTED[ext] in " ".join(q.split()), q
    assert EXPECTED[ext].replace("$1", "$3").split(" = ", 1)[1] in " ".join(writes[3].split())


@pytest.mark.parametrize("backend", [("pgroonga", False), ("pg_search", False), ("native", True)], indirect=True)
async def test_backends_without_an_inline_column_write_no_search_vector(backend):
    conn = _CapturingConn()
    await _run_every_write(conn)
    for q in _text_writes(conn):
        assert "search_vector" not in q, q


class _RepairConn:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetchval(self, query, *params):
        self.queries.append(query)
        return 7

    async def execute(self, query, *params):
        self.queries.append(query)
        return "UPDATE 7"


async def _repair(dry_run: bool) -> tuple[int, _RepairConn]:
    conn = _RepairConn()
    n = await admin.rebuild_observation_search_vectors(
        conn=conn, fq_table_explicit=lambda t, s: f"{s}.{t}", schema="s", bank_id="b", dry_run=dry_run
    )
    return n, conn


@pytest.mark.parametrize("backend", [("vchord", False)], indirect=True)
async def test_repair_retokenizes_every_observation_under_vchord(backend):
    """repair-bank heals observations a pre-#5299 rewrite left stale, from their current text."""
    n, conn = await _repair(dry_run=False)
    assert n == 7
    (q,) = conn.queries
    assert "SET search_vector = tokenize(COALESCE(text, ''), 'llmlingua2')::bm25_catalog.bm25vector" in q
    assert "fact_type = 'observation'" in q

    n, conn = await _repair(dry_run=True)
    assert n == 7
    assert not any("UPDATE" in q for q in conn.queries), "a dry run only counts"


@pytest.mark.parametrize("backend", [("native", False), ("pgroonga", False)], indirect=True)
async def test_repair_is_a_no_op_off_vchord(backend):
    """Native always rebuilt the column inline and the others keep none, so nothing to heal."""
    n, conn = await _repair(dry_run=False)
    assert n == 0 and conn.queries == []
