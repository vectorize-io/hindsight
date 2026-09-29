"""Bank-archive row restore on the Oracle backend (no information_schema, upper-case identifiers)."""

import asyncio
from datetime import datetime


class _FakeConn:
    def __init__(self, columns):
        self.columns = columns
        self.fetches = []
        self.executes = []

    async def fetch(self, query, *args):
        self.fetches.append((query, args))
        return self.columns

    async def execute(self, query, *args):
        self.executes.append((query, args))
        return "INSERT 0 1"


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

    assert asyncio.run(_restore_rows(conn, "memory_units", rows)) == 1

    col_query, col_args = conn.fetches[0]
    assert "information_schema" not in col_query
    assert "all_tab_columns" in col_query
    assert col_args == ("memory_units",)

    insert, values = conn.executes[0]
    assert '("ID", "CREATED_AT", "TAGS")' in insert
    assert isinstance(values[1], datetime)
