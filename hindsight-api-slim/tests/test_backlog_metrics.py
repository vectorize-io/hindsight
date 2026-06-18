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

from hindsight_api.metrics import MetricsCollector, _AsyncOpKey, _BacklogKey


class _FakeTxn:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    """asyncpg-like connection whose fetch() is dispatched by SQL substring."""

    def __init__(self, fetch_fn):
        self._fetch_fn = fetch_fn
        self.executed = []

    async def fetch(self, sql, *args):
        return self._fetch_fn(sql, *args)

    async def execute(self, sql, *args):
        self.executed.append(sql)

    def transaction(self):
        return _FakeTxn()


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


def _set_db_pool_with_backlog_enabled(collector, pool):
    """Call set_db_pool with the backlog flag forced on (it's off by default)."""
    mock_config = MagicMock()
    mock_config.metrics_backlog_enabled = True
    with patch("hindsight_api.config.get_config", return_value=mock_config):
        collector.set_db_pool(pool)


def _rows_for(sql):
    """Canned results, keyed off distinctive substrings of each query."""
    if "information_schema.tables" in sql:
        return [{"table_schema": "public"}]
    if "async_operations" in sql:
        return [
            {"operation_type": "retain", "status": "pending", "count": 5},
            {"operation_type": "consolidation", "status": "pending", "count": 12},
            {"operation_type": "consolidation", "status": "processing", "count": 1},
            {"operation_type": "consolidation", "status": "failed", "count": 2},
        ]
    if "memory_units" in sql and "consolidated_at IS NULL" in sql:
        return [{"count": 42}]
    if "memory_units" in sql and "consolidation_failed_at IS NOT NULL" in sql:
        return [{"count": 3}]
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
async def test_refresh_backlog_uses_index_matched_predicates_not_filter_scan():
    """Backlog/failed must be two separate COUNT(*) queries whose WHERE matches
    a partial-index predicate exactly (no FILTER over a full-table scan), and
    the queue query must exclude terminal statuses."""
    captured = []
    collector = _collector()
    collector._db_pool = _FakePool(lambda sql, *a: (captured.append(sql), _rows_for(sql))[1])
    await collector._refresh_backlog()

    mem_queries = [s for s in captured if "memory_units" in s and "COUNT(*)" in s]
    assert len(mem_queries) == 2  # split, not a single two-FILTER aggregate
    assert all("FILTER" not in s for s in mem_queries)
    assert any("consolidated_at IS NULL AND fact_type IN ('experience', 'world')" in s for s in mem_queries)
    assert any("consolidation_failed_at IS NOT NULL AND fact_type IN ('experience', 'world')" in s for s in mem_queries)

    ops_sql = next(s for s in captured if "async_operations" in s and "GROUP BY" in s)
    assert "status IN ('pending', 'processing', 'failed')" in ops_sql
    assert "completed" not in ops_sql and "cancelled" not in ops_sql


@pytest.mark.asyncio
async def test_backlog_count_runs_with_seqscan_disabled():
    """`consolidated_at IS NULL` is true for a large fraction of the table, so
    the planner misjudges selectivity and won't use the partial index without a
    nudge — the backlog count must issue SET LOCAL enable_seqscan=off."""
    collector = _collector()
    pool = _FakePool(lambda sql, *a: _rows_for(sql))
    collector._db_pool = pool
    await collector._refresh_backlog()

    assert any("enable_seqscan" in s.lower() and "off" in s.lower() for s in pool._conn.executed)
    # the result is still correct under the nudge
    assert collector._consolidation_backlog[("public", None)] == 42


@pytest.mark.asyncio
async def test_refresh_backlog_per_bank_labels_and_group_by_when_enabled():
    """With metrics_include_bank_id on, bank_id enters the cache key and the
    SQL switches to GROUP BY bank_id."""
    captured = []

    def fetch(sql, *a):
        captured.append(sql)
        if "information_schema.tables" in sql:
            return [{"table_schema": "public"}]
        if "async_operations" in sql:
            return [{"operation_type": "retain", "status": "pending", "bank_id": "bankA", "count": 4}]
        if "memory_units" in sql and "consolidated_at IS NULL" in sql:
            return [{"bank_id": "bankA", "count": 11}]
        if "memory_units" in sql and "consolidation_failed_at IS NOT NULL" in sql:
            return [{"bank_id": "bankA", "count": 2}]
        return []

    collector = _collector(include_bank_id=True)
    collector._db_pool = _FakePool(fetch)
    await collector._refresh_backlog()

    assert collector._async_ops_counts[("public", "retain", "pending", "bankA")] == 4
    assert collector._consolidation_backlog[("public", "bankA")] == 11
    assert collector._consolidation_failed[("public", "bankA")] == 2
    # bank_id must be grouped in every per-bank count query
    assert all("GROUP BY bank_id" in s for s in captured if "memory_units" in s and "COUNT(*)" in s)


def test_gauges_register_and_emit_cached_values_without_bank_id():
    collector = _collector(include_bank_id=False)
    # Sync call: no running loop, so gauges register but no background task spawns.
    _set_db_pool_with_backlog_enabled(collector, MagicMock())

    gauges = {
        c.kwargs["name"]: c.kwargs["callbacks"][0]
        for c in collector.meter.create_observable_gauge.call_args_list
        if "callbacks" in c.kwargs
    }
    assert "hindsight.async_operations" in gauges
    assert "hindsight.consolidation.backlog" in gauges
    assert "hindsight.consolidation.failed" in gauges

    collector._async_ops_counts = {
        _AsyncOpKey("public", "retain", "pending", None): 7,
        _AsyncOpKey("public", "consolidation", "processing", None): 1,
    }
    collector._consolidation_backlog = {_BacklogKey("public", None): 9}

    obs = list(gauges["hindsight.async_operations"](None))
    by_label = {(o.attributes["operation_type"], o.attributes["status"]): o.value for o in obs}
    assert by_label[("retain", "pending")] == 7
    assert by_label[("consolidation", "processing")] == 1
    assert all("bank_id" not in o.attributes for o in obs)  # cardinality guard

    backlog_obs = list(gauges["hindsight.consolidation.backlog"](None))
    assert backlog_obs[0].value == 9
    assert backlog_obs[0].attributes["tenant"] == "public"


def test_gauge_emits_bank_id_attribute_when_present():
    collector = _collector(include_bank_id=True)
    _set_db_pool_with_backlog_enabled(collector, MagicMock())
    gauges = {
        c.kwargs["name"]: c.kwargs["callbacks"][0]
        for c in collector.meter.create_observable_gauge.call_args_list
        if "callbacks" in c.kwargs
    }
    collector._consolidation_backlog = {_BacklogKey("public", "bankA"): 4}
    obs = list(gauges["hindsight.consolidation.backlog"](None))
    assert obs[0].value == 4
    assert obs[0].attributes["bank_id"] == "bankA"


def test_backlog_gauges_not_registered_when_flag_disabled():
    """Backlog metrics are off by default: set_db_pool must not register the
    gauges unless metrics_backlog_enabled is set."""
    collector = _collector()
    mock_config = MagicMock()
    mock_config.metrics_backlog_enabled = False
    with patch("hindsight_api.config.get_config", return_value=mock_config):
        collector.set_db_pool(MagicMock())

    names = [
        c.kwargs.get("name") for c in collector.meter.create_observable_gauge.call_args_list if "callbacks" in c.kwargs
    ]
    assert "hindsight.async_operations" not in names
    assert "hindsight.consolidation.backlog" not in names
    assert "hindsight.consolidation.failed" not in names
    assert collector._backlog_task is None
