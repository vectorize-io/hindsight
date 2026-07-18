from __future__ import annotations

import re
import uuid

import pytest

from hindsight_api.engine.retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name
from hindsight_api.engine.vector_index_reconcile import bank_vector_indexes_healthy, reconcile_vector_indexes


class _FakeConnection:
    def __init__(self, *, lock_acquired: bool = True) -> None:
        self.lock_acquired = lock_acquired
        self.bank_id = "restored-bank"
        self.internal_id = uuid.uuid4()
        self.indexes: dict[str, bool] = {}
        self.statements: list[str] = []
        self.fail_once: set[str] = set()
        self.drop_fail_once: set[str] = set()
        self.catalog_queries = 0

    async def fetch(self, query: str, *args):
        if "pg_index" in query:
            self.catalog_queries += 1
            return [
                {"index_name": name, "healthy": healthy} for name, healthy in self.indexes.items() if name in args[1]
            ]
        return [{"bank_id": self.bank_id, "internal_id": self.internal_id}]

    async def fetchval(self, query: str, *args):
        if "pg_try_advisory_lock" in query:
            return self.lock_acquired
        if "pg_advisory_unlock" in query:
            return True
        if "quote_literal" in query:
            value = args[0]
            return "'" + str(value).replace("'", "''") + "'"
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def execute(self, query: str):
        self.statements.append(query)
        match = re.search(
            r'(?:CREATE INDEX CONCURRENTLY IF NOT EXISTS|DROP INDEX CONCURRENTLY IF EXISTS)\s+(?:"[^"]+"\.)?"?([a-zA-Z0-9_]+)"?',
            query,
        )
        assert match, query
        index_name = match.group(1)
        if query.startswith("CREATE"):
            if index_name in self.fail_once:
                self.fail_once.remove(index_name)
                self.indexes[index_name] = False
                raise RuntimeError("simulated concurrent build failure")
            self.indexes[index_name] = True
        else:
            if index_name in self.drop_fail_once:
                self.drop_fail_once.remove(index_name)
                raise RuntimeError("simulated concurrent drop failure")
            self.indexes.pop(index_name, None)


def _expected_names(conn: _FakeConnection) -> list[str]:
    return [_bank_index_name(fact_type, str(conn.internal_id)) for fact_type in _BANK_INDEX_FACT_TYPES]


@pytest.mark.asyncio
async def test_bank_health_checks_all_expected_indexes_in_one_catalog_query() -> None:
    calls = []

    class HealthConnection:
        async def fetchval(self, query: str, *args):
            calls.append((query, args))
            return True

    healthy = await bank_vector_indexes_healthy(HealthConnection(), "tenant_a", "restored-bank")

    assert healthy is True
    assert len(calls) == 1
    query, args = calls[0]
    assert 'FROM "tenant_a".banks' in query
    assert "i.indisvalid AND i.indisready" in query
    assert "am.amname = ANY($4::text[])" in query
    assert args[0] == "restored-bank"
    assert args[1] == "tenant_a"
    assert args[2] == len(_BANK_INDEX_FACT_TYPES)


@pytest.mark.asyncio
async def test_dry_run_reports_missing_indexes_without_changing_catalog() -> None:
    conn = _FakeConnection()
    expected = _expected_names(conn)
    conn.indexes[expected[0]] = False

    results = await reconcile_vector_indexes(
        conn,
        ["public"],
        "USING hnsw (embedding vector_cosine_ops)",
        dry_run=True,
    )

    assert results[0].created == 0
    assert results[0].skipped == len(_BANK_INDEX_FACT_TYPES)
    assert conn.indexes == {expected[0]: False}
    assert conn.statements == []


@pytest.mark.asyncio
async def test_reconcile_creates_missing_and_replaces_invalid_indexes() -> None:
    conn = _FakeConnection()
    expected = _expected_names(conn)
    conn.indexes[expected[0]] = True
    conn.indexes[expected[1]] = False

    results = await reconcile_vector_indexes(conn, ["public"], "USING hnsw (embedding vector_cosine_ops)")

    assert len(results) == 1
    result = results[0]
    assert result.already_present == 1
    assert result.created == 2
    assert result.failed == 0
    assert all(conn.indexes[name] for name in expected)
    assert any(statement.startswith("DROP INDEX CONCURRENTLY") for statement in conn.statements)
    assert all("CONCURRENTLY" in statement for statement in conn.statements)
    assert conn.catalog_queries == 1


@pytest.mark.asyncio
async def test_reconcile_skips_when_another_instance_holds_lock() -> None:
    conn = _FakeConnection(lock_acquired=False)

    results = await reconcile_vector_indexes(conn, ["public"], "USING hnsw (embedding vector_cosine_ops)")

    assert len(results) == 1
    assert results[0].skipped_lock_busy is True
    assert conn.statements == []


@pytest.mark.asyncio
async def test_failed_build_is_cleaned_up_and_retried_on_next_reconcile() -> None:
    conn = _FakeConnection()
    failed_index = _expected_names(conn)[0]
    conn.fail_once.add(failed_index)

    first = await reconcile_vector_indexes(conn, ["public"], "USING hnsw (embedding vector_cosine_ops)")
    second = await reconcile_vector_indexes(conn, ["public"], "USING hnsw (embedding vector_cosine_ops)")

    assert first[0].failed == 1
    assert f"public.{failed_index}" in first[0].failed_indexes
    assert second[0].created == 1
    assert conn.indexes[failed_index] is True


@pytest.mark.asyncio
async def test_failed_invalid_index_drop_does_not_abort_other_indexes() -> None:
    conn = _FakeConnection()
    expected = _expected_names(conn)
    conn.indexes[expected[0]] = False
    conn.drop_fail_once.add(expected[0])

    results = await reconcile_vector_indexes(conn, ["public"], "USING hnsw (embedding vector_cosine_ops)")

    assert results[0].failed == 1
    assert f"public.{expected[0]}" in results[0].failed_indexes
    assert conn.indexes[expected[1]] is True
    assert conn.indexes[expected[2]] is True
