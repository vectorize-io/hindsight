"""Tests for native ML thread-pool caps (hindsight_api/_thread_limits.py)."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hindsight_api._thread_limits import (
    _MAX_NATIVE_THREADS,
    _NATIVE_THREAD_VARS,
    _available_cpu_count,
    _cgroup_cpu_quota,
    _parse_cgroup_v2_cpu_max,
    _quota_to_cpus,
    apply_default_thread_limits,
    bound_onnx_session,
    native_thread_summary,
    onnx_intra_op_threads,
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


def _cgroup_tree(tmp_path, proc_line: str, files: dict[str, str]) -> tuple[str, str]:
    root = tmp_path / "cgroup"
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    root.mkdir(exist_ok=True)
    proc = tmp_path / "proc_cgroup"
    proc.write_text(proc_line)
    return str(proc), str(root)


def test_cgroup_quota_found_on_own_systemd_unit(tmp_path) -> None:
    """A systemd unit's CPUQuota sits on its own cgroup, not the root (#5725)."""
    proc, root = _cgroup_tree(
        tmp_path, "0::/system.slice/hindsight.service\n", {"system.slice/hindsight.service/cpu.max": "200000 100000"}
    )
    assert _cgroup_cpu_quota(proc, root) == 2


def test_cgroup_quota_takes_tightest_ancestor(tmp_path) -> None:
    proc, root = _cgroup_tree(
        tmp_path,
        "0::/user.slice/app.scope\n",
        {"user.slice/cpu.max": "100000 100000", "user.slice/app.scope/cpu.max": "max 100000"},
    )
    assert _cgroup_cpu_quota(proc, root) == 1


def test_cgroup_quota_container_root(tmp_path) -> None:
    """With a private cgroup namespace the own path is '/', i.e. the mount root."""
    proc, root = _cgroup_tree(tmp_path, "0::/\n", {"cpu.max": "300000 100000"})
    assert _cgroup_cpu_quota(proc, root) == 3


def test_cgroup_quota_v1_cpu_controller(tmp_path) -> None:
    proc, root = _cgroup_tree(
        tmp_path,
        "5:memory:/system.slice/x.service\n4:cpu,cpuacct:/system.slice/x.service\n",
        {
            "cpu/system.slice/x.service/cpu.cfs_quota_us": "400000",
            "cpu/system.slice/x.service/cpu.cfs_period_us": "100000",
        },
    )
    assert _cgroup_cpu_quota(proc, root) == 4


def test_cgroup_quota_none_when_unlimited(tmp_path) -> None:
    proc, root = _cgroup_tree(
        tmp_path, "0::/system.slice/x.service\n", {"system.slice/x.service/cpu.max": "max 100000"}
    )
    assert _cgroup_cpu_quota(proc, root) is None


def test_bound_onnx_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """ONNX Runtime ignores OMP_NUM_THREADS, so each session gets the count explicitly."""
    entries: dict[str, str] = {}
    options = SimpleNamespace(
        intra_op_num_threads=0, inter_op_num_threads=0, add_session_config_entry=entries.__setitem__
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    bound_onnx_session(options)
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (2, 1)
    assert entries == {"session.intra_op.allow_spinning": "0"}
    bound_onnx_session(options, threads=5)
    assert options.intra_op_num_threads == 5


def test_cgroup_quota_ignores_path_outside_namespace(tmp_path) -> None:
    """A '..' path has no directory under the mount; only the root is read."""
    proc, root = _cgroup_tree(tmp_path, "0::/../sibling\n", {"cpu.max": "200000 100000"})
    assert _cgroup_cpu_quota(proc, root) == 2


def test_onnx_intra_op_threads_falls_back_on_bad_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "lots")
    assert onnx_intra_op_threads() == default_native_thread_count()
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert onnx_intra_op_threads() == default_native_thread_count()


def test_native_thread_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert native_thread_summary().startswith("Native threads: 3 per pool (available CPUs=")
    assert "cgroup CPU quota=" in native_thread_summary()
