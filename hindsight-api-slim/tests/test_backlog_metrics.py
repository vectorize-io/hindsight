"""
Tests for the async-operation queue and consolidation backlog gauges
(``_setup_backlog_metrics`` / ``_refresh_backlog`` in metrics.py).

These gauges expose, as scrapable time-series, the same counts the bank-stats
endpoint already returns per bank (``operations_by_status``,
``pending_consolidation``, ``failed_consolidation``):

- ``hindsight_async_operations{operation_type,status}`` — worker queue depth
  (pending=backlog, processing=in-flight, failed=stranded)
- ``hindsight_consolidation_backlog`` — source memories not yet consolidated
- ``hindsight_consolidation_failed`` — source memories permanently failed
"""

from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.metrics import MetricsCollector


class _FakeConn:
    """asyncpg-like connection whose fetch() is dispatched by SQL substring."""

    def __init__(self, fetch_fn):
        self._fetch_fn = fetch_fn

    async def fetch(self, sql, *args):
        return self._fetch_fn(sql, *args)


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, fetch_fn):
        self._conn = _FakeConn(fetch_fn)

    def acquire(self):
        return _FakeAcquire(self._conn)


def _collector(include_bank_id=False):
    mock_config = MagicMock()
    mock_config.metrics_include_bank_id = include_bank_id
    with (
        patch("hindsight_api.metrics.get_meter", return_value=MagicMock()),
        patch("hindsight_api.config.get_config", return_value=mock_config),
    ):
        return MetricsCollector()


def _rows_for(sql):
    if "information_schema.tables" in sql:
        return [{"table_schema": "public"}]
    if "async_operations" in sql:
        return [
            {"operation_type": "retain", "status": "pending", "count": 5},
            {"operation_type": "consolidation", "status": "pending", "count": 12},
            {"operation_type": "consolidation", "status": "processing", "count": 1},
            {"operation_type": "consolidation", "status": "failed", "count": 2},
        ]
    if "memory_units" in sql:
        return [{"pending": 42, "failed": 3}]
    return []


@pytest.mark.asyncio
async def test_refresh_backlog_aggregates_queue_and_consolidation():
    collector = _collector(include_bank_id=False)
    collector._db_pool = _FakePool(lambda sql, *a: _rows_for(sql))

    await collector._refresh_backlog()

    # Worker queue depth keyed by (schema, operation_type, status, bank=None)
    assert collector._async_ops_counts[("public", "retain", "pending", None)] == 5
    assert collector._async_ops_counts[("public", "consolidation", "pending", None)] == 12
    assert collector._async_ops_counts[("public", "consolidation", "processing", None)] == 1
    assert collector._async_ops_counts[("public", "consolidation", "failed", None)] == 2
    # Consolidation backlog (source memories), keyed by (schema, bank=None)
    assert collector._consolidation_backlog[("public", None)] == 42
    assert collector._consolidation_failed[("public", None)] == 3


@pytest.mark.asyncio
async def test_refresh_backlog_excludes_terminal_statuses_from_query():
    """The queue gauge must not count completed/cancelled work — assert the
    SQL filters to the non-terminal states only."""
    captured = []
    collector = _collector()

    def fetch(sql, *a):
        captured.append(sql)
        return _rows_for(sql)

    collector._db_pool = _FakePool(fetch)
    await collector._refresh_backlog()

    ops_sql = next(s for s in captured if "async_operations" in s and "GROUP BY" in s)
    assert "status IN ('pending', 'processing', 'failed')" in ops_sql
    assert "completed" not in ops_sql and "cancelled" not in ops_sql


def test_gauges_register_and_emit_cached_values_without_bank_id():
    collector = _collector(include_bank_id=False)
    # Sync call: no running loop, so gauges register but no background task spawns.
    collector.set_db_pool(MagicMock())

    gauges = {
        c.kwargs["name"]: c.kwargs["callbacks"][0]
        for c in collector.meter.create_observable_gauge.call_args_list
        if "callbacks" in c.kwargs
    }
    assert "hindsight.async_operations" in gauges
    assert "hindsight.consolidation.backlog" in gauges
    assert "hindsight.consolidation.failed" in gauges

    collector._async_ops_counts = {
        ("public", "retain", "pending", None): 7,
        ("public", "consolidation", "processing", None): 1,
    }
    collector._consolidation_backlog = {("public", None): 9}

    obs = list(gauges["hindsight.async_operations"](None))
    by_label = {(o.attributes["operation_type"], o.attributes["status"]): o.value for o in obs}
    assert by_label[("retain", "pending")] == 7
    assert by_label[("consolidation", "processing")] == 1
    assert all("bank_id" not in o.attributes for o in obs)  # cardinality guard

    backlog_obs = list(gauges["hindsight.consolidation.backlog"](None))
    assert backlog_obs[0].value == 9
    assert backlog_obs[0].attributes["tenant"] == "public"
