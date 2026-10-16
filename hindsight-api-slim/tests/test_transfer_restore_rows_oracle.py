"""Bank-archive row restore on the Oracle backend (no information_schema, upper-case identifiers)."""

import asyncio
from datetime import datetime


class _FakeConn:
    def __init__(self, columns):
        self.columns = columns
        self.fetches = []
        self.executes = []
        self.batches = []

    async def fetch(self, query, *args):
        self.fetches.append((query, args))
        return self.columns

    async def executemany(self, query, rows):
        self.batches.append((query, len(rows)))
        self.executes.extend((query, row) for row in rows)


def test_restore_rows_on_oracle(monkeypatch):
    monkeypatch.setattr("hindsight_api.engine.schema._is_oracle", lambda: True)
    from hindsight_api.engine.transfer.importer import _restore_rows

    conn = _FakeConn(
        [
            {"column_name": "id", "data_type": "raw"},
            {"column_name": "created_at", "data_type": "timestamp(6) with time zone"},
            {"column_name": "tags", "data_type": "clob"},
        ]
    )
    rows = [{"id": "3f1c2e4a-0000-4000-8000-000000000001", "created_at": "2026-09-23T01:00:00+00:00", "tags": ["a"]}]

    assert asyncio.run(_restore_rows(conn, "mental_models", rows)) == 1

    col_query, col_args = conn.fetches[0]
    assert "information_schema" not in col_query
    assert "all_tab_columns" in col_query
    assert col_args == ("mental_models",)

    insert, values = conn.executes[0]
    assert '("ID", "CREATED_AT", "TAGS")' in insert
    assert isinstance(values[1], datetime)


def test_restore_rows_streams_in_batches_per_column_set(monkeypatch):
    """Rows arrive as a stream and go out one executemany per batch of rows sharing a column set."""
    monkeypatch.setattr("hindsight_api.engine.schema._is_oracle", lambda: True)
    from hindsight_api.engine.transfer import importer

    monkeypatch.setattr(importer, "_RESTORE_BATCH_SIZE", 2)
    conn = _FakeConn([{"column_name": "a", "data_type": "number"}, {"column_name": "b", "data_type": "number"}])
    rows = iter([{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4, "b": 5}])

    assert asyncio.run(importer._restore_rows(conn, "audit_log", rows)) == 4
    assert [count for _query, count in conn.batches] == [2, 1, 1]
    assert conn.batches[0][0] == conn.batches[1][0] != conn.batches[2][0]
