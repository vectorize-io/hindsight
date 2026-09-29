"""Periodic CPU profile of the API process, emitted to the log stream.

Configured by ONE variable, ``HINDSIGHT_API_PROFILE``, holding JSON -- the shape this
codebase already uses for structured env config (``json.loads(os.getenv(...))`` in
config.py). Unset, nothing starts and there is no cost:

    HINDSIGHT_API_PROFILE='{"every": 60}'

    every    seconds between emissions (default 60)
    top      rows per emission (default 20)
    mode     "cprofile" (default); reserved so another mode can be added later

**Why the log stream, and not a file or an endpoint.** The case this has to serve is a
process dying without explanation -- typically a container nobody can exec into. A file inside the container dies with it unless somebody mounted
a volume in advance; an endpoint needs a live process to answer and a network path to reach
it, which is exactly what a crashing pod does not offer. Container runtimes keep the
*previous* container's stdout (``kubectl logs --previous``), so the last emission before the
crash is still readable afterwards -- the report survives the failure it exists to explain. Each emission is flushed as
it is written, because a fatal signal takes buffered output with it.

One log record per row, each tagged ``[profile]``, so ``kubectl logs <pod> | grep
'\\[profile\\]'`` reconstructs the table and the repo's JsonFormatter wraps each line as
structured JSON on the way out.

**The profiler is process-wide, and that is a property of the interpreter, not a choice.**
Since 3.12 cProfile is a global monitoring tool: ``enable()`` covers every thread whatever
thread calls it, and a second concurrent profiler raises ``ValueError: tool 2 is already in
use``. So one profiler is armed for the process and its report spans all threads. It also
means nothing else in the process may profile at the same time.

**Numbers are per-window, by subtraction.** Snapshots use ``getstats()``, which reads the
accumulated entries WITHOUT stopping the profiler, and each emission reports the delta since
the last one. The obvious alternative -- snapshot and clear -- silently stops profiling,
because taking the snapshot through ``pstats`` disables the global tool and every emission
after the first then reports nothing.

**Read the report with two things in mind.**

``tottime`` is a function's OWN CPU; ``cumtime`` includes everything it called, so a
coroutine high in ``cumtime`` may merely be awaiting. Attributing cost per task once put a
middleware at 77% of loop CPU when removing it saved 13%.

cProfile roughly halves throughput and over-weights functions called very often, so the
RANKING is the signal and the milliseconds are not. Every emission therefore also carries
per-thread CPU read from ``/proc``, which no profiler overhead can distort: if the two
disagree, ``/proc`` is right. That check is not decorative -- a stop-the-world sampler over
``sys._current_frames()`` reported every event-loop thread idle in ``selectors.select``
while ``/proc`` showed those same threads burning 50-65% of a core.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from .config import ENV_PROFILE, get_config  # noqa: E402

logger = logging.getLogger("hindsight_api.profiling")

_installed = False
_profiler = None  # the armed cProfile.Profile; module-level so a report can be forced in tests
_previous: dict[str, tuple[int, float, float]] = {}


def _config() -> dict | None:
    """Parse the single env var, or None when profiling is off.

    A malformed value raises rather than quietly disabling: whoever set this wants a
    profile, and a typo that produces no output costs a whole test cycle to notice.
    """
    raw = get_config().profile
    if not raw:
        return None
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise ValueError(f"{ENV_PROFILE} must be a JSON object, got {type(cfg).__name__}")
    every = int(cfg.get("every", 60))
    if every < 1:
        raise ValueError(f"{ENV_PROFILE}: 'every' must be >= 1")
    mode = str(cfg.get("mode", "cprofile"))
    if mode != "cprofile":
        raise ValueError(f"{ENV_PROFILE}: unknown mode {mode!r}")
    return {"every": every, "top": int(cfg.get("top", 20)), "mode": mode}


def _thread_cpu(sample_seconds: float = 1.0) -> list[tuple[str, float]]:
    """Per-thread-group CPU over a short sample, straight from /proc.

    The arbiter for everything else in the report, because it cannot be distorted by
    profiler overhead. Linux only; returns [] elsewhere, since a missing cross-check is
    better than a fabricated one.
    """

    def snap() -> dict[str, tuple[int, str]]:
        out: dict[str, tuple[int, str]] = {}
        try:
            tids = os.listdir("/proc/self/task")
        except OSError:
            return {}
        for tid in tids:
            try:
                with open(f"/proc/self/task/{tid}/comm") as fh:
                    comm = fh.read().strip().rstrip("0123456789")
                with open(f"/proc/self/task/{tid}/stat") as fh:
                    fields = fh.read().rsplit(") ", 1)[1].split()
                out[tid] = (int(fields[11]) + int(fields[12]), comm)
            except (OSError, IndexError, ValueError):
                continue
        return out

    first = snap()
    if not first:
        return []
    time.sleep(sample_seconds)
    second = snap()
    totals: dict[str, int] = {}
    for tid, (ticks, comm) in second.items():
        if tid in first:
            delta = ticks - first[tid][0]
            if delta > 0:
                totals[comm] = totals.get(comm, 0) + delta
    per_sec = os.sysconf("SC_CLK_TCK") * sample_seconds
    return sorted(((c, t / per_sec) for c, t in totals.items()), key=lambda kv: -kv[1])


def _label(code) -> str:
    """A row label for a profiler entry: builtins arrive as a string, Python as a code object."""
    if isinstance(code, str):
        return code
    return "%s:%d(%s)" % (
        code.co_filename.rsplit("/", 1)[-1],
        code.co_firstlineno,
        code.co_name,
    )


def _emit(prof, cfg: dict) -> None:
    for group, cores in _thread_cpu()[:8]:
        logger.info("[profile] thread %-24s %5.2f cores", group, cores)
    try:
        rows = []
        for stat in prof.getstats():
            # Keyed by the label, NOT id(code): CPython reuses ids once an object is
            # freed, and code objects here are not all long-lived -- a process compiling
            # code at runtime frees them constantly. A reused id would silently subtract
            # the wrong baseline and report a nonsense delta.
            key = _label(stat.code)
            prev = _previous.get(key)
            d_calls = stat.callcount - (prev[0] if prev else 0)
            d_inline = stat.inlinetime - (prev[1] if prev else 0.0)
            d_total = stat.totaltime - (prev[2] if prev else 0.0)
            _previous[key] = (stat.callcount, stat.inlinetime, stat.totaltime)
            if d_inline > 0 or d_calls > 0:
                rows.append((d_inline, d_total, d_calls, key))
        rows.sort(reverse=True)
        logger.info("[profile] %d functions active in the last %ds", len(rows), cfg["every"])
        for d_inline, d_total, d_calls, label in rows[: cfg["top"]]:
            logger.info(
                "[profile] tottime=%.3f cumtime=%.3f ncalls=%d %s",
                d_inline,
                d_total,
                d_calls,
                label,
            )
    except Exception as err:  # a diagnostic must never take the process down
        logger.warning("[profile] report failed: %r", err)
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()  # a fatal signal takes buffered output with it
        except Exception:
            pass


def install() -> bool:
    """Arm profiling if the env var asks for it. Safe to call more than once."""
    global _installed, _profiler
    if _installed:
        return True
    cfg = _config()
    if cfg is None:
        return False

    import cProfile

    prof = cProfile.Profile()
    try:
        prof.enable()
    except ValueError as err:
        # The profiler is a single process-global tool; something else already holds it.
        logger.warning("[profile] not armed: %r", err)
        return False

    def pump() -> None:
        while True:
            time.sleep(cfg["every"])
            try:
                _emit(prof, cfg)
            except Exception as err:
                logger.warning("[profile] emission failed: %r", err)

    threading.Thread(target=pump, daemon=True, name="hs-profile-pump").start()
    _profiler = prof
    _installed = True
    logger.info("[profile] armed: %s", json.dumps(cfg))
    return True
