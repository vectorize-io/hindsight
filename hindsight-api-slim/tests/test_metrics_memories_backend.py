"""The ``memories_backend`` metric label: which store served an operation.

Lets per-backend latency be compared on a deployment whose banks live in different stores, without
turning on the per-tenant label (one series set per schema, too high-cardinality to leave on).

Opt-in by construction: a store that names nothing adds no label, so the series a deployment
already has are left exactly as they are and keep their history.
"""

from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine.memories.postgres import PostgresMemories
from hindsight_api.metrics import MetricsCollector, memories_backend_for


class _RoutedStore:
    """Stands in for a router that names only its non-default store, leaving the default unlabelled."""

    def backend_name_for(self, bank_id: str) -> str:
        return "other-store" if bank_id.startswith("ext-") else ""


@pytest.fixture
def collector():
    config = MagicMock()
    config.metrics_include_bank_id = False
    config.metrics_include_tenant = False
    config.recall_diagnostic_phases = True
    config.recall_phase_sample_every = 1
    meter = MagicMock()
    # A fresh mock per instrument, so each histogram's calls are its own.
    meter.create_histogram.side_effect = lambda *a, **k: MagicMock()
    meter.create_counter.side_effect = lambda *a, **k: MagicMock()
    with (
        patch("hindsight_api.metrics.get_meter", return_value=meter),
        patch("hindsight_api.config.get_config", return_value=config),
    ):
        return MetricsCollector()


@pytest.fixture
def routed():
    with patch("hindsight_api.engine.memories.get_memories", return_value=_RoutedStore()):
        yield


def test_default_is_no_label_so_existing_series_are_untouched(collector):
    """A store that does not override backend_name_for names nothing: no label, so no new series
    and no break in the history of the ones a deployment already has."""
    store = PostgresMemories({})
    assert store.backend_name_for("any-bank") == ""
    with patch("hindsight_api.engine.memories.get_memories", return_value=store):
        with collector.record_operation("recall", bank_id="bank"):
            collector.record_recall_phase("engine_call", 0.01)

    assert "memories_backend" not in collector.operation_duration.record.call_args.args[1]
    assert "memories_backend" not in collector.operation_total.add.call_args.args[1]
    assert "memories_backend" not in collector.recall_phase_duration.record.call_args.args[1]


@pytest.mark.usefixtures("routed")
def test_only_a_named_backend_is_labelled(collector):
    with collector.record_operation("recall", bank_id="ext-bank"):
        pass
    collector.record_operation_result("recall", "pg-bank", success=True, duration=0.1)

    labels = [c.args[1].get("memories_backend") for c in collector.operation_duration.record.call_args_list]
    assert labels == ["other-store", None]
    totals = [c.args[1].get("memories_backend") for c in collector.operation_total.add.call_args_list]
    assert totals == ["other-store", None]


@pytest.mark.usefixtures("routed")
def test_recall_phases_inside_an_operation_inherit_its_backend(collector):
    """Phases carry no bank id, so they take the label from the operation they run inside."""
    collector.record_recall_phase("dep_auth", 0.01)  # before the operation: no backend known yet
    with collector.record_operation("recall", bank_id="ext-bank"):
        collector.record_recall_phase("engine_call", 0.01)
    collector.record_recall_phase("post_engine", 0.01)  # after it: the label must not leak

    by_phase = {
        c.args[1]["phase"]: c.args[1].get("memories_backend")
        for c in collector.recall_phase_duration.record.call_args_list
    }
    assert by_phase == {"dep_auth": None, "engine_call": "other-store", "post_engine": None}


@pytest.mark.usefixtures("routed")
def test_label_is_reset_when_the_operation_fails(collector):
    with pytest.raises(RuntimeError):
        with collector.record_operation("recall", bank_id="ext-bank"):
            raise RuntimeError("boom")
    collector.record_recall_phase("after", 0.01)

    assert collector.operation_duration.record.call_args.args[1]["memories_backend"] == "other-store"
    assert "memories_backend" not in collector.recall_phase_duration.record.call_args.args[1]


def test_an_unresolvable_store_omits_the_label_instead_of_failing(collector):
    """It is only a metric attribute: a store that cannot answer must not fail the operation."""
    broken = MagicMock()
    broken.backend_name_for.side_effect = RuntimeError("no tenant context")
    with patch("hindsight_api.engine.memories.get_memories", return_value=broken):
        assert memories_backend_for("bank") == ""
        with collector.record_operation("recall", bank_id="bank"):
            pass

    assert "memories_backend" not in collector.operation_duration.record.call_args.args[1]


@pytest.mark.usefixtures("routed")
@pytest.mark.parametrize(("bank", "expected"), [("ext-bank", "other-store"), ("pg-bank", "_RoutedStore")])
async def test_retain_store_label_names_a_named_backend_and_otherwise_keeps_its_value(bank, expected):
    """The retain `store` label was the extension's class, so behind a router every bank reported
    the router. A bank whose store is named now reports that name; one whose store names nothing
    keeps the class name, so its series is unchanged."""
    from hindsight_api.engine.retain import timing

    seen = {}

    class _Ctx:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc):
            return False

    @timing.timed_retain
    async def retain(self, bank_id, contents):
        return "ok"

    with patch.object(timing, "retain_timing", _Ctx):
        assert await retain(object(), bank, [{"content": "x"}]) == "ok"

    assert seen["store"] == expected
