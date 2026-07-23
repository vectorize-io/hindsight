"""Regression coverage for deterministic chunk deletion ordering."""

import pytest

from hindsight_api.engine.retain import chunk_storage


class RecordingConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, sql: str, *args: object) -> None:
        self.calls.append((sql, args))


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_predeletes_links_before_chunks():
    conn = RecordingConn()
    chunk_ids = ["chunk-b", "chunk-a"]

    await chunk_storage.delete_chunks_by_ids(conn, chunk_ids)

    assert len(conn.calls) == 2
    link_sql, link_args = conn.calls[0]
    chunk_sql, chunk_args = conn.calls[1]

    assert link_args == (chunk_ids,)
    assert chunk_args == (chunk_ids,)

    assert "DELETE FROM" in link_sql
    assert "memory_links" in link_sql
    assert "target_units AS MATERIALIZED" in link_sql
    assert "ordered_links AS MATERIALIZED" in link_sql
    assert "ORDER BY" in link_sql
    assert "FOR UPDATE OF ml" in link_sql

    assert "DELETE FROM" in chunk_sql
    assert "chunks" in chunk_sql
    assert "ordered_chunks AS MATERIALIZED" in chunk_sql
    assert "ORDER BY chunk_id" in chunk_sql
    assert "FOR UPDATE" in chunk_sql


@pytest.mark.asyncio
async def test_delete_chunks_by_ids_noops_without_chunks():
    conn = RecordingConn()

    await chunk_storage.delete_chunks_by_ids(conn, [])

    assert conn.calls == []
