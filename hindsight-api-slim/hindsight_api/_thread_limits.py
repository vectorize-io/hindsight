"""Process-level caps for native ML thread pools.

OpenBLAS, OpenMP, and MKL each spawn a worker pool sized to the host CPU count
the first time they are loaded (numpy pulls in OpenBLAS eagerly; torch and
onnxruntime load their pools lazily on first inference). Hindsight already
parallelizes at the request level via thread-pool executors (embeddings on the
default executor, the reranker on its own pool), so these native intra-op pools
oversubscribe the CPU: on a many-core host the process accumulates 100+ native
threads, which inflates memory and, under contention, can degrade throughput.

We bound each pool to ``_MAX_NATIVE_THREADS`` (or the available CPU count, if
smaller). "Available" is the CPU budget actually granted to the process, not
``os.cpu_count()``: in a CPU-limited container ``os.cpu_count()`` still reports
the host's cores, so sizing pools by it oversubscribes the container's real
quota — the exact failure mode this guards against. We therefore take the
smallest of the CPU-affinity set, the cgroup CPU quota, and ``os.cpu_count()``.
The quota is read from the process's own cgroup and its parents, not only the
mount root: a systemd unit's ``CPUQuota=`` lives at ``/system.slice/<unit>``, and
reading only the root missed it and sized every pool to the host (#5725).

ONNX Runtime's CPU pool does not read ``OMP_NUM_THREADS`` — it sizes itself to the
host's physical cores — so every ``ort.SessionOptions`` must go through
:func:`bound_onnx_session` to get the same cap.

Every cap is applied with ``setdefault`` so an operator who has deliberately
tuned one of these variables keeps their value. This must run *before* numpy,
torch, or onnxruntime are imported — those libraries read the variables only at
load time — which is why it is invoked at the very top of
``hindsight_api/__init__.py``, ahead of the package's other imports.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnxruntime import SessionOptions

# Native threading env vars, each read by the respective library at load time.
_NATIVE_THREAD_VARS = (
    "OMP_NUM_THREADS",  # OpenMP — torch, some BLAS builds (not ONNX Runtime: see bound_onnx_session)
    "OPENBLAS_NUM_THREADS",  # OpenBLAS — numpy's default BLAS
    "MKL_NUM_THREADS",  # Intel MKL — numpy/torch when MKL-backed
    "NUMEXPR_NUM_THREADS",  # numexpr expression engine
)

# Upper bound on intra-op threads per native pool. Bounds runaway growth on
# many-core hosts without serialising single-request inference.
_MAX_NATIVE_THREADS = 16


def _quota_to_cpus(quota: int, period: int) -> int | None:
    """Whole CPUs from a CFS quota/period pair, or None if unlimited."""
    if quota > 0 and period > 0:
        # Floor (never round up) so we never exceed the granted budget.
        return max(1, quota // period)
    return None


def _parse_cgroup_v2_cpu_max(text: str) -> int | None:
    """Parse cgroup v2 ``cpu.max`` ("<quota> <period>", or "max <period>")."""
    parts = text.split()
    if len(parts) >= 2 and parts[0] != "max":
        try:
            return _quota_to_cpus(int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return None


def _cgroup_ancestors(mount: str, path: str) -> list[str]:
    """The process's own cgroup directory under ``mount``, then each parent up to the mount."""
    parts = [p for p in path.split("/") if p]
    return [os.path.join(mount, *parts[:i]) for i in range(len(parts), -1, -1)]


def _read_cpu_quota(directory: str) -> int | None:
    """CPUs granted by one cgroup directory (v2 ``cpu.max`` or v1 cfs files), or None."""
    try:
        with open(os.path.join(directory, "cpu.max")) as fh:
            return _parse_cgroup_v2_cpu_max(fh.read())
    except OSError:
        pass
    try:
        with open(os.path.join(directory, "cpu.cfs_quota_us")) as fh:
            quota = int(fh.read())
        with open(os.path.join(directory, "cpu.cfs_period_us")) as fh:
            period = int(fh.read())
        return _quota_to_cpus(quota, period)
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> int | None:
    """Effective CPUs from the tightest cgroup CPU quota over this process, or None.

    The quota lives on the process's own cgroup (or an ancestor), not on the mount's
    root: inside a container with its own cgroup namespace the two coincide, but a
    systemd unit (``CPUQuota=``) sits at ``/system.slice/<unit>.service`` and the
    root has no limit. So read ``/proc/self/cgroup`` and walk from the own cgroup up
    to the mount, keeping the smallest quota found.
    """
    candidates: list[str] = []
    try:
        with open(proc_cgroup) as fh:
            for line in fh:
                _, controllers, path = line.rstrip("\n").split(":", 2)
                if ".." in path.split("/"):
                    # Our cgroup is outside this cgroup namespace's root, so it has no
                    # directory under the mount; the root fallback below still applies.
                    continue
                if controllers == "":  # cgroup v2 unified hierarchy
                    candidates += _cgroup_ancestors(root, path)
                elif "cpu" in controllers.split(","):  # cgroup v1 cpu controller
                    candidates += _cgroup_ancestors(os.path.join(root, "cpu"), path)
    except (OSError, ValueError):
        pass
    # Fallback for when /proc/self/cgroup is unreadable: the mount roots, as before.
    candidates += [root, os.path.join(root, "cpu")]
    quotas = [q for q in map(_read_cpu_quota, candidates) if q is not None]
    return min(quotas) if quotas else None


def _available_cpu_count() -> int:
    """CPUs actually available to this process.

    The smallest of the CPU-affinity set (cpuset / ``--cpuset-cpus``), the
    cgroup CPU quota (``--cpus``), and ``os.cpu_count()`` — each captures a
    different way the budget can be constrained, and the last alone overcounts
    inside a limited container.
    """
    candidates = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    quota = _cgroup_cpu_quota()
    if quota is not None:
        candidates.append(quota)
    return max(1, min(candidates))


def available_cpu_count() -> int:
    """Public alias for :func:`_available_cpu_count`.

    Admission-control defaults size themselves from the real CPU budget, and that
    budget must come from here rather than ``os.cpu_count()``: under ``--cpus`` the
    latter reports the host's cores, which would size limits for a machine the
    process cannot use.
    """
    return _available_cpu_count()


def default_native_thread_count() -> int:
    """Per-pool cap: ``_MAX_NATIVE_THREADS``, or available CPUs if fewer."""
    return min(_MAX_NATIVE_THREADS, _available_cpu_count())


def apply_default_thread_limits() -> None:
    """Cap native ML thread pools unless the operator has set the var already."""
    value = str(default_native_thread_count())
    for var in _NATIVE_THREAD_VARS:
        os.environ.setdefault(var, value)


def onnx_intra_op_threads() -> int:
    """Intra-op threads for an ONNX Runtime session.

    ONNX Runtime's CPU pool ignores ``OMP_NUM_THREADS`` and sizes itself to the
    host's physical cores, so the cap must be passed on each session. Use the same
    value the other native pools got (an operator-set ``OMP_NUM_THREADS`` wins).
    """
    try:
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    except (KeyError, ValueError):
        return default_native_thread_count()


def bound_onnx_session(options: "SessionOptions", threads: int = 0) -> None:
    """Size an ``ort.SessionOptions`` to the CPU budget; ``threads`` > 0 overrides.

    Idle workers otherwise busy-wait between ops, which under a CFS quota burns the
    budget the real work needs (every scheduler period throttled).
    """
    options.intra_op_num_threads = threads or onnx_intra_op_threads()
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")


def native_thread_summary() -> str:
    """One line for the startup log, so a throttled service is diagnosable without ``cpu.stat``."""
    quota = _cgroup_cpu_quota()
    return (
        f"Native threads: {onnx_intra_op_threads()} per pool (available CPUs={_available_cpu_count()}, "
        f"cgroup CPU quota={quota if quota is not None else 'none found'})"
    )
