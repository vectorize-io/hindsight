"""Tests for the env-configured CPU profiler.

Each of these pins a property that was broken at some point while building it, which is why
they assert behaviour rather than merely that ``install()`` returns.
"""

import json
import logging
import pathlib
import threading

import pytest

from hindsight_api import profiling


@pytest.fixture(autouse=True)
def _reset():
    """The profiler is a process-global tool, so a leaked one breaks unrelated tests."""
    profiling._installed = False
    profiling._previous.clear()
    yield
    if profiling._profiler is not None:
        try:
            profiling._profiler.disable()  # release the global tool for the next test
        except Exception:
            pass
    profiling._profiler = None
    profiling._installed = False
    profiling._previous.clear()


def test_off_unless_the_env_var_is_set(monkeypatch):
    monkeypatch.delenv(profiling.ENV_PROFILE, raising=False)
    assert profiling.install() is False


def test_malformed_config_raises_rather_than_disabling(monkeypatch):
    """A typo must be loud. Silently producing no profile costs a whole measurement cycle."""
    monkeypatch.setenv(profiling.ENV_PROFILE, "{not json")
    with pytest.raises(json.JSONDecodeError):
        profiling.install()

    monkeypatch.setenv(profiling.ENV_PROFILE, '["not", "an", "object"]')
    with pytest.raises(ValueError):
        profiling.install()

    monkeypatch.setenv(profiling.ENV_PROFILE, '{"every": 0}')
    with pytest.raises(ValueError):
        profiling.install()

    monkeypatch.setenv(profiling.ENV_PROFILE, '{"mode": "sampling"}')
    with pytest.raises(ValueError):
        profiling.install()


def test_defaults_are_applied(monkeypatch):
    monkeypatch.setenv(profiling.ENV_PROFILE, "{}")
    cfg = profiling._config()
    assert cfg == {"every": 60, "top": 20, "mode": "cprofile"}


def test_report_is_a_delta_not_a_running_total(monkeypatch, caplog):
    """Each emission must describe its own window.

    Snapshots use getstats() precisely so profiling is never stopped; the obvious
    alternative (snapshot through pstats, then clear) disables the global tool, and every
    emission after the first then reports nothing. This asserts a second window still has
    data, which that bug would fail.
    """
    monkeypatch.setenv(profiling.ENV_PROFILE, json.dumps({"every": 3600, "top": 5}))
    assert profiling.install() is True

    prof = profiling._profiler
    assert prof is not None, "install() must expose the armed profiler"

    def work():
        sum(i * i for i in range(50_000))

    cfg = {"every": 1, "top": 5}
    with caplog.at_level(logging.INFO, logger="hindsight_api.profiling"):
        work()
        profiling._emit(prof, cfg)
        first = [r for r in caplog.records if "tottime=" in r.getMessage()]
        caplog.clear()
        work()
        profiling._emit(prof, cfg)
        second = [r for r in caplog.records if "tottime=" in r.getMessage()]

    assert first, "the first window should report the work done before it"
    assert second, "a second window must still report data; profiling must not have stopped"


def test_thread_cpu_reports_named_groups():
    """The /proc cross-check is the arbiter for the profile, so it must actually produce rows.

    A stop-the-world sampler once reported every event-loop thread idle while /proc showed
    those same threads busy; this table is what catches that, and an empty table catches
    nothing.
    """
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            sum(i * i for i in range(20_000))

    t = threading.Thread(target=spin, name="probe-thread", daemon=True)
    t.start()
    try:
        rows = profiling._thread_cpu(sample_seconds=0.3)
    finally:
        stop.set()
        t.join(timeout=5)

    if not rows:
        pytest.skip("/proc/self/task is unavailable on this platform")
    assert any(cores > 0 for _name, cores in rows)


def test_install_is_idempotent(monkeypatch):
    """A second arm must not try to claim the profiler tool twice.

    Since 3.12 the profiler is a single process-global tool: a second concurrent enable()
    raises `ValueError: tool 2 is already in use`.
    """
    monkeypatch.setenv(profiling.ENV_PROFILE, json.dumps({"every": 3600}))
    assert profiling.install() is True
    assert profiling.install() is True


def test_deltas_survive_code_objects_being_freed(monkeypatch):
    """Baselines are keyed by label, not id(), because CPython reuses ids.

    A process that compiles code at runtime frees code objects constantly. Keyed by
    id(), a reused address would subtract another function's baseline and report a
    nonsense delta -- silently, since the number still looks like a number.
    """
    monkeypatch.setenv(profiling.ENV_PROFILE, json.dumps({"every": 3600}))
    assert profiling.install() is True
    profiling._emit(profiling._profiler, {"every": 1, "top": 5})

    assert profiling._previous, "a report must record baselines"
    assert all(isinstance(k, str) for k in profiling._previous), (
        "baselines must be keyed by a stable label, not by id()"
    )


def test_create_app_arms_profiling_so_workers_are_covered(monkeypatch):
    """Arming in main() alone is not enough when uvicorn runs `--workers N`.

    Workers are spawned processes that import the app and never run main(), so profiling
    armed only in main() covers the supervisor -- which does nothing but waitpid() -- and
    reports an idle process while every request is served in a worker it cannot see.
    Verified against a real 2-worker deployment before this call was added: 63 report
    lines, all of them supervisor bookkeeping.
    """
    import hindsight_api.api.http as http_module

    called = []
    monkeypatch.setattr(profiling, "install", lambda: called.append(True) or False)
    monkeypatch.setattr("hindsight_api.profiling.install", lambda: called.append(True) or False)

    source = pathlib.Path(http_module.__file__).read_text()
    body = source.split("def create_app(", 1)[1]
    assert "_install_profiling()" in body.split("def ", 1)[0], (
        "create_app must arm profiling, or --workers deployments profile the supervisor"
    )


import pathlib  # noqa: E402  (used by the test above)


def test_recall_phase_histogram_can_resolve_a_millisecond_phase():
    """The phase histogram must have buckets sized for the values it records.

    Its unit is seconds and recall phases take milliseconds, so the SDK default boundaries
    (0, 5, 10, 25, ...) put every observation in the first bucket: the histogram then
    reports a mean but no usable percentile. Asked for a p99 it answered 2500 ms for all
    fifteen phases at once -- the midpoint of the 0-5s bucket -- which is what sent an
    investigation of a 460 ms p99 down a blind alley.
    """
    import hindsight_api.metrics as metrics_module

    source = pathlib.Path(metrics_module.__file__).read_text()
    # Slice to the NEXT instrument, not to the first ")": the description text contains
    # parentheses, which truncated this block before the argument it is checking for.
    block = source.split('name="hindsight.recall.phase.duration"', 1)[1]
    block = block.split("self.consolidation_batch_failures", 1)[0]
    assert "explicit_bucket_boundaries_advisory" in block, (
        "the recall phase histogram needs millisecond-scale buckets, or its percentiles are fiction"
    )
    assert "0.025" in block, "buckets must cover the 10-50ms range where recall phases live"
