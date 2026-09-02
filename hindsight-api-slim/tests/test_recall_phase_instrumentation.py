"""Every recall phase is timed, and none of them had to be named to get there.

`hindsight.operation.duration` for a recall is one number. Subtracting the store's own timings from
it leaves a residual — hydration, entity build, token filtering, serialization — that no metric
covers; on a measured window that residual was 37% of the request, and the only place the split
existed was a per-request log line.

The phases are already computed: `SearchTracer.add_phase_metric` is the single funnel every one of
them goes through, and the tracer is constructed for EVERY recall so the `[phases]` line has
somewhere to write. So these assert the STRUCTURAL property — whatever goes through the funnel is
recorded — rather than a list of today's phase names, because the failure this guards against is a
phase added later that quietly is not timed.
"""

import pytest

from hindsight_api.engine.search.tracer import SearchTracer


class _Collector:
    def __init__(self):
        self.recorded: list[tuple[str, float, bool]] = []

    def record_recall_phase(self, phase: str, seconds: float, *, diagnostic: bool = False) -> None:
        self.recorded.append((phase, seconds, diagnostic))


@pytest.fixture
def collector(monkeypatch):
    c = _Collector()
    import hindsight_api.metrics as m

    monkeypatch.setattr(m, "get_metrics_collector", lambda: c)
    return c


def _tracer() -> SearchTracer:
    return SearchTracer(query="anything", budget=10, max_tokens=1000)


def test_a_phase_is_recorded_with_its_duration(collector):
    _tracer().add_phase_metric("hydrate_results", 0.25)
    assert collector.recorded == [("hydrate_results", 0.25, False)]


def test_a_subset_phase_is_marked_so_a_sum_can_exclude_it(collector):
    """Some phases are children of another — a per-arm timing inside `parallel_retrieval`. Summing
    them with their parent double-counts, so the attribute is what lets a consumer add up the
    request without inventing a list of which names are subsets."""
    t = _tracer()
    t.add_phase_metric("parallel_retrieval", 0.40)
    t.add_phase_metric("retrieval_semantic", 0.30, {"diagnostic": True})

    assert ("parallel_retrieval", 0.40, False) in collector.recorded
    assert ("retrieval_semantic", 0.30, True) in collector.recorded
    siblings = sum(s for _, s, diag in collector.recorded if not diag)
    assert siblings == pytest.approx(0.40), "a subset must not inflate the request total"


def test_every_phase_reaches_the_collector_whatever_it_is_called(collector):
    """The structural property. A phase added later is timed because it goes through the funnel,
    not because someone remembered to add it to a list."""
    t = _tracer()
    for name in (
        "embed",
        "parallel_retrieval",
        "rrf_merge",
        "hydrate_results",
        "entity_build",
        "token_filtering",
        "serialize",
        "a_phase_invented_tomorrow",
    ):
        t.add_phase_metric(name, 0.01)

    assert [p for p, _, _ in collector.recorded] == [
        "embed",
        "parallel_retrieval",
        "rrf_merge",
        "hydrate_results",
        "entity_build",
        "token_filtering",
        "serialize",
        "a_phase_invented_tomorrow",
    ]


def test_the_trace_itself_is_unchanged_by_the_recording(collector):
    """The `[phases]` line and any returned trace read from `phase_metrics`; recording must add to
    that, not replace it."""
    t = _tracer()
    t.add_phase_metric("hydrate_results", 0.25, {"rows": 12})

    assert len(t.phase_metrics) == 1
    assert t.phase_metrics[0].phase_name == "hydrate_results"
    assert t.phase_metrics[0].duration_seconds == 0.25
    assert t.phase_metrics[0].details == {"rows": 12}


def test_a_broken_collector_never_fails_the_recall(monkeypatch):
    """Instrumentation must not be the thing that fails the request it measures."""
    import hindsight_api.metrics as m

    def _boom():
        raise RuntimeError("metrics backend is down")

    monkeypatch.setattr(m, "get_metrics_collector", _boom)
    t = _tracer()
    t.add_phase_metric("hydrate_results", 0.25)
    assert len(t.phase_metrics) == 1, "the trace still records even when the metric cannot"
