"""One /metrics covering every worker, each series labelled with its worker.

What matters: every live worker's series appear in a single scrape, they are kept apart (never
summed), the answering worker's own values are current, a gone worker drops out, and a bad
snapshot never breaks the scrape.
"""

import os
import time

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.parser import text_string_to_metric_families

from hindsight_api import metrics_multiworker as mw


def _registry(requests: float, latency: float) -> CollectorRegistry:
    r = CollectorRegistry()
    Counter("demo_requests", "requests", ["route"], registry=r).labels("/recall").inc(requests)
    Histogram("demo_seconds", "latency", registry=r, buckets=(0.1, 1.0)).observe(latency)
    return r


def _samples(exposition: bytes) -> dict[tuple[str, tuple], float]:
    out = {}
    for family in text_string_to_metric_families(exposition.decode()):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def _worker(tmp_path, slot: int, registry: CollectorRegistry) -> mw.WorkerMetrics:
    w = mw.WorkerMetrics(str(tmp_path), slot, registry=registry)
    w.write_snapshot()
    return w


def test_every_worker_appears_in_one_scrape_labelled_and_not_summed(tmp_path):
    w0 = _worker(tmp_path, 0, _registry(requests=3, latency=0.05))
    _worker(tmp_path, 1, _registry(requests=5, latency=0.5))

    got = _samples(w0.render())

    assert got[("demo_requests_total", (("api_worker", "0"), ("route", "/recall")))] == 3
    assert got[("demo_requests_total", (("api_worker", "1"), ("route", "/recall")))] == 5
    assert got[("demo_seconds_bucket", (("api_worker", "0"), ("le", "0.1")))] == 1
    assert got[("demo_seconds_bucket", (("api_worker", "1"), ("le", "0.1")))] == 0
    # Nothing without the label: no series from an unknown worker.
    assert all(dict(labels).get("api_worker") in ("0", "1") for _, labels in got)


def test_the_answering_workers_own_series_are_live_not_its_snapshot(tmp_path):
    registry = CollectorRegistry()
    counter = Counter("demo_requests", "requests", registry=registry)
    w0 = _worker(tmp_path, 0, registry)  # snapshot taken at 0
    counter.inc(7)

    got = _samples(w0.render())

    assert got[("demo_requests_total", (("api_worker", "0"),))] == 7


def test_a_gone_workers_stale_snapshot_is_skipped(tmp_path):
    w0 = _worker(tmp_path, 0, _registry(requests=1, latency=0.05))
    _worker(tmp_path, 1, _registry(requests=9, latency=0.05))
    old = time.time() - mw.STALE_AFTER_S - 5
    os.utime(tmp_path / "worker-1.prom", (old, old))

    got = _samples(w0.render())

    assert not any(dict(labels).get("api_worker") == "1" for _, labels in got)
    assert got[("demo_requests_total", (("api_worker", "0"), ("route", "/recall")))] == 1


def test_a_corrupt_snapshot_is_skipped_without_breaking_the_scrape(tmp_path):
    w0 = _worker(tmp_path, 0, _registry(requests=2, latency=0.05))
    (tmp_path / "worker-1.prom").write_text("this is {not an exposition\n")

    got = _samples(w0.render())

    assert got[("demo_requests_total", (("api_worker", "0"), ("route", "/recall")))] == 2


def test_the_merged_output_round_trips_through_the_parser(tmp_path):
    w0 = _worker(tmp_path, 0, _registry(requests=1, latency=0.2))
    _worker(tmp_path, 1, _registry(requests=2, latency=2.0))

    families = {f.name: f.type for f in text_string_to_metric_families(w0.render().decode())}

    assert families["demo_requests"] == "counter"
    assert families["demo_seconds"] == "histogram"


def test_slots_are_distinct_and_a_released_slot_is_reused(tmp_path):
    first = mw.claim_slot(str(tmp_path), 2)
    second = mw.claim_slot(str(tmp_path), 2)
    assert (first[0], second[0]) == (0, 1)
    assert mw.claim_slot(str(tmp_path), 2) is None  # both held
    first[1].close()  # the worker holding slot 0 exits
    again = mw.claim_slot(str(tmp_path), 2)
    assert again is not None and again[0] == 0
    again[1].close()
    second[1].close()


def test_start_publishes_immediately_and_never_raises(tmp_path):
    worker = mw.start_worker_metrics(2, directory=str(tmp_path))
    try:
        assert worker is not None
        assert (tmp_path / f"worker-{worker.slot}.prom").exists()
    finally:
        worker.stop()
    # No writable directory: disabled, not an exception.
    assert mw.start_worker_metrics(2, directory="/proc/definitely-not-writable/x") is None
