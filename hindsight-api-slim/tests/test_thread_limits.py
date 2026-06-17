"""Tests for native ML thread-pool caps (hindsight_api/_thread_limits.py)."""

import os
import subprocess
import sys

import pytest

from hindsight_api._thread_limits import (
    _MAX_NATIVE_THREADS,
    _NATIVE_THREAD_VARS,
    _available_cpu_count,
    _parse_cgroup_v2_cpu_max,
    _quota_to_cpus,
    apply_default_thread_limits,
    default_native_thread_count,
)


def test_default_is_bounded_by_available_cpus() -> None:
    """The default is the thread ceiling, or the available CPU budget if smaller."""
    assert default_native_thread_count() == min(_MAX_NATIVE_THREADS, _available_cpu_count())


def test_available_cpu_count_within_host_bound() -> None:
    """Available CPUs is at least 1 and never exceeds the host's logical CPU count."""
    assert 1 <= _available_cpu_count() <= (os.cpu_count() or 1)


def test_quota_to_cpus() -> None:
    """CFS quota/period maps to whole CPUs, flooring and never below 1."""
    assert _quota_to_cpus(400000, 100000) == 4
    assert _quota_to_cpus(50000, 100000) == 1  # 0.5 CPU floors to 1, not 0
    assert _quota_to_cpus(-1, 100000) is None  # unlimited
    assert _quota_to_cpus(0, 100000) is None


def test_parse_cgroup_v2_cpu_max() -> None:
    """cgroup v2 cpu.max parsing: a CPU-limited container reports its quota."""
    assert _parse_cgroup_v2_cpu_max("400000 100000") == 4  # --cpus=4
    assert _parse_cgroup_v2_cpu_max("150000 100000") == 1  # 1.5 CPU floors to 1
    assert _parse_cgroup_v2_cpu_max("max 100000") is None  # unlimited
    assert _parse_cgroup_v2_cpu_max("") is None
    assert _parse_cgroup_v2_cpu_max("garbage") is None


def test_applies_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each cap is set to the default when the operator has not set it."""
    for var in _NATIVE_THREAD_VARS:
        monkeypatch.delenv(var, raising=False)

    apply_default_thread_limits()

    expected = str(default_native_thread_count())
    for var in _NATIVE_THREAD_VARS:
        assert os.environ[var] == expected


def test_respects_operator_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly-set value is preserved rather than overwritten."""
    for var in _NATIVE_THREAD_VARS:
        monkeypatch.setenv(var, "3")

    apply_default_thread_limits()

    for var in _NATIVE_THREAD_VARS:
        assert os.environ[var] == "3"


def test_covers_the_documented_native_libraries() -> None:
    """Lock in the set of vars so a future edit can't silently drop one."""
    assert set(_NATIVE_THREAD_VARS) == {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }


# A BLAS-heavy snippet that forces OpenBLAS to create its worker pool, then
# reports this process's OS thread count from /proc (which, unlike
# threading.active_count(), includes native threads). It imports hindsight_api
# first so the real __init__ ordering is exercised: the caps must be applied
# before numpy pulls in OpenBLAS, otherwise the pool is already sized.
_OPENBLAS_THREAD_PROBE = """
import hindsight_api  # noqa: F401 -- __init__ applies thread caps (no-op if vars preset)
import numpy as np
a = np.random.rand(1024, 1024)
for _ in range(3):
    a = a @ a
    a /= (a.max() or 1.0)
with open("/proc/self/status") as status:
    print(next(int(line.split()[1]) for line in status if line.startswith("Threads:")))
"""


def _probe_native_thread_count(var_overrides: dict[str, str]) -> int:
    """Run the BLAS probe in a fresh interpreter and return its OS thread count."""
    env = {k: v for k, v in os.environ.items() if k not in _NATIVE_THREAD_VARS}
    env.update(var_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _OPENBLAS_THREAD_PROBE],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr}"
    return int(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc (Linux only)")
@pytest.mark.skipif(
    (os.cpu_count() or 1) <= _MAX_NATIVE_THREADS,
    reason=f"cap only observable on hosts with > {_MAX_NATIVE_THREADS} cores",
)
def test_import_caps_native_thread_oversubscription() -> None:
    """Reproduce the native-thread buildup and prove the import cap bounds it.

    A single BLAS workload makes OpenBLAS fan its pool out to one thread per
    core. Importing hindsight_api applies the cap before numpy loads, holding
    the pool at the configured ceiling; an operator who sets the vars to the
    core count keeps the full per-core fan-out. Only observable when the host
    has more cores than the ceiling — otherwise the cap equals the core count.
    """
    ncpu = os.cpu_count() or 4
    capped = _probe_native_thread_count({})  # vars unset -> hindsight caps to the ceiling
    uncapped = _probe_native_thread_count({var: str(ncpu) for var in _NATIVE_THREAD_VARS})

    if uncapped <= capped:
        pytest.skip(f"numpy BLAS is not multi-threaded here (capped={capped}, uncapped={uncapped})")

    assert capped < uncapped, f"thread cap had no effect: capped={capped}, uncapped={uncapped}"
