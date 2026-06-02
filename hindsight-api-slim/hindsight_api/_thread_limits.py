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

Every cap is applied with ``setdefault`` so an operator who has deliberately
tuned one of these variables keeps their value. This must run *before* numpy,
torch, or onnxruntime are imported — those libraries read the variables only at
load time — which is why it is invoked at the very top of
``hindsight_api/__init__.py``, ahead of the package's other imports.
"""

from __future__ import annotations

import os

# Native threading env vars, each read by the respective library at load time.
_NATIVE_THREAD_VARS = (
    "OMP_NUM_THREADS",  # OpenMP — torch, onnxruntime, some BLAS builds
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


def _cgroup_cpu_quota() -> int | None:
    """Effective CPUs from the cgroup CPU quota, or None if unlimited/unknown."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            return _parse_cgroup_v2_cpu_max(fh.read())
    except OSError:
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        return _quota_to_cpus(quota, period)
    except (OSError, ValueError):
        return None


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


def default_native_thread_count() -> int:
    """Per-pool cap: ``_MAX_NATIVE_THREADS``, or available CPUs if fewer."""
    return min(_MAX_NATIVE_THREADS, _available_cpu_count())


def apply_default_thread_limits() -> None:
    """Cap native ML thread pools unless the operator has set the var already."""
    value = str(default_native_thread_count())
    for var in _NATIVE_THREAD_VARS:
        os.environ.setdefault(var, value)
