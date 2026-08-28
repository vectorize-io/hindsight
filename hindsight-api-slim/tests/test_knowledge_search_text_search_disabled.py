"""Knowledge-page search under a bank with ``enable_text_search=false``.

``search_knowledge_pages`` is a hybrid: a vector arm and a BM25 arm, RRF-fused in
SQL. With the bank's text search off it must collapse to its vector half — the same
thing the flag does to recall — rather than emit a BM25 arm the bank opted out of.
Because the flag is per bank it is resolved through the config resolver here, not
read off the global config. Asserts on the generated SQL, like
``test_knowledge_bm25_dispatch``, so no live extension or database is needed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hindsight_api.engine import memory_engine as engine_mod
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_utils


class _CapturingConn:
    """Records the SQL the search emitted; returns no rows."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.params: list[tuple] = []

    async def fetch(self, query, *params):
        self.queries.append(query)
        self.params.append(params)
        return []


@pytest.fixture
def search(monkeypatch):
    """A MemoryEngine stubbed down to the collaborators this one method touches."""
    conn = _CapturingConn()

    engine = MemoryEngine.__new__(MemoryEngine)
    engine._operation_validator = None
    engine.embeddings = object()
    resolved: dict[str, object] = {}

    async def fake_authenticate(request_context):
        return None

    async def fake_get_backend():
        return object()

    async def fake_get_bank_config(bank_id, context=None):
        return resolved

    engine._authenticate_tenant = fake_authenticate
    engine._get_backend = fake_get_backend
    engine._config_resolver = SimpleNamespace(get_bank_config=fake_get_bank_config)

    @asynccontextmanager
    async def fake_acquire(pool, *args, **kwargs):
        yield conn

    monkeypatch.setattr(engine_mod, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(engine_mod, "fq_table", lambda name: name)

    async def run(*, enable_text_search: bool, embedding: list[float] | None):
        async def fake_embed(embeddings, texts, *, input_type=None):
            return [embedding]

        monkeypatch.setattr(embedding_utils, "generate_embeddings_batch", fake_embed)
        monkeypatch.setattr(engine_mod, "get_config", lambda: SimpleNamespace(text_search_extension="native"))
        resolved.clear()
        resolved["enable_text_search"] = enable_text_search
        return await engine.search_knowledge_pages("bank-1", "some query", request_context=object())

    return SimpleNamespace(run=run, conn=conn)


@pytest.mark.asyncio
async def test_enabled_fuses_both_arms(search):
    """Baseline: the default still emits the RRF-fused hybrid."""
    await search.run(enable_text_search=True, embedding=[0.1, 0.2])

    sql = search.conn.queries[0]
    assert "ts_rank_cd" in sql  # the native BM25 arm
    assert "FULL OUTER JOIN" in sql  # fused with the vector arm
    assert search.conn.params[0] == ("[0.1, 0.2]", "bank-1", "some query")


@pytest.mark.asyncio
async def test_disabled_emits_vector_only_sql(search):
    await search.run(enable_text_search=False, embedding=[0.1, 0.2])

    sql = search.conn.queries[0]
    assert "mm.embedding <=> $1::vector" in sql
    assert "ts_rank_cd" not in sql
    assert "FULL OUTER JOIN" not in sql
    # The query text is no longer bound — nothing left to match it against.
    assert search.conn.params[0] == ("[0.1, 0.2]", "bank-1")
    # Bank isolation survives the rewrite.
    assert "kp.bank_id = $2" in sql


@pytest.mark.asyncio
async def test_disabled_without_embedding_returns_nothing(search):
    """Both arms unavailable: no BM25-only fallback to degrade to.

    The fallback exists for a missing embedding, not for a deployment that turned
    text search off — running it would query the very index the flag opted out of.
    """
    result = await search.run(enable_text_search=False, embedding=None)

    assert result == []
    assert search.conn.queries == []


@pytest.mark.asyncio
async def test_enabled_without_embedding_still_falls_back_to_bm25(search):
    """Guards the test above: the pre-existing fallback is untouched."""
    await search.run(enable_text_search=True, embedding=None)

    assert "ts_rank_cd" in search.conn.queries[0]
    assert search.conn.params[0] == ("bank-1", "some query")
