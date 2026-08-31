"""Where a retain's wall time actually goes.

Retain crosses four subsystems -- chunking, the embedder, the memories store and Postgres --
and without this a slow retain could only be attributed by reasoning
about which of them was likely. That reasoning got it wrong more than once: the store's share
was read as Postgres, and Postgres work on a store-owned bank (queries against tables holding
no rows for such a bank) was invisible because nothing counted the round trips.

Two things are recorded per phase, and the SECOND is usually the more useful one:

  * seconds -- how long the phase took, and
  * calls   -- how many round trips it took to get there.

A phase that is slow because each call is slow and one that is slow because it makes ten times
as many calls need opposite fixes, and only the call count separates them.

**Phases overlap.** Sub-batches run concurrently (`retain_subbatch_concurrency`), so the shares
sum to more than the retain's duration -- a phase at 300% of wall is work that parallelised,
one at 95% is work that did not. Read a phase against the retain's own duration, never as a
fraction of the others.

Usage -- the accumulator is per-retain and lives in a contextvar, so the phases do not have to
be threaded through every call:

    async with retain_timing(bank_id="b", store="the-store") as t:
        async with t.phase("embed"):
            ...

Nothing is emitted for a retain that recorded no phases, so an uninstrumented path stays silent
rather than logging an empty breakdown that reads like "this retain did nothing".
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_CURRENT: contextvars.ContextVar["RetainTiming | None"] = contextvars.ContextVar(
    "hindsight_retain_timing", default=None
)


@dataclass
class RetainTiming:
    """One retain's phase totals. Not thread-safe by design: a retain is one task tree, and the
    += on a float is not a suspension point, so concurrent sub-batches of the SAME retain
    accumulate correctly without a lock."""

    bank_id: str = ""
    store: str = ""
    documents: int = 0
    content_bytes: int = 0
    started: float = field(default_factory=time.perf_counter)
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    @contextlib.asynccontextmanager
    async def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float, calls: int = 1) -> None:
        """Record a phase directly, for a caller that timed it itself."""
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + calls

    @property
    def wall(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> dict:
        wall = self.wall
        return {
            "bank_id": self.bank_id,
            "store": self.store,
            "documents": self.documents,
            "content_bytes": self.content_bytes,
            "wall_s": round(wall, 3),
            "kb_per_s": round(self.content_bytes / 1024 / wall, 1) if wall > 0 and self.content_bytes else 0.0,
            "phases": {
                name: {
                    "s": round(self.seconds[name], 3),
                    "calls": self.calls.get(name, 0),
                    # Per-document, because that is the number that should stay flat as a batch
                    # grows -- a phase whose per-document cost rises with batch size is the one
                    # that is not scaling.
                    "per_doc_ms": (round(1000 * self.seconds[name] / self.documents, 1) if self.documents else None),
                    "pct_wall": round(100 * self.seconds[name] / wall) if wall > 0 else 0,
                }
                for name in sorted(self.seconds, key=lambda n: -self.seconds[n])
            },
        }


def current() -> RetainTiming | None:
    """The retain currently being timed, or None outside one. Callers use `record()` instead of
    this; it exists for a caller that needs to know whether timing is on at all."""
    return _CURRENT.get()


def record(phase: str, seconds: float, calls: int = 1) -> None:
    """Attribute time to a phase of the retain in scope. A no-op outside one, so an instrumented
    helper can be called from a non-retain path without guarding every call site."""
    t = _CURRENT.get()
    if t is not None:
        t.add(phase, seconds, calls)


@contextlib.asynccontextmanager
async def timed(phase: str):
    """Time a block into the retain in scope. A no-op outside one."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(phase, time.perf_counter() - t0)


@contextlib.asynccontextmanager
async def retain_timing(*, bank_id: str = "", store: str = "", documents: int = 0, content_bytes: int = 0):
    """Install a fresh accumulator for one retain, and emit its breakdown on the way out.

    Emitted even when the retain raises: a failed retain's breakdown is the one most worth
    having, and losing it to the exception is how "it was slow, then it failed" stays
    unexplained.
    """
    t = RetainTiming(bank_id=bank_id, store=store, documents=documents, content_bytes=content_bytes)
    token = _CURRENT.set(t)
    try:
        yield t
    finally:
        _CURRENT.reset(token)
        if t.seconds:
            _emit(t)


def _emit(t: RetainTiming) -> None:
    summary = t.summary()
    try:
        from ...metrics import get_metrics_collector

        collector = get_metrics_collector()
        for name, s in t.seconds.items():
            collector.record_retain_phase(name, s, calls=t.calls.get(name, 0), store=t.store)
    except Exception:
        # Instrumentation must never be the reason a retain fails.
        logger.debug("retain phase metrics not recorded", exc_info=True)

    parts = " ".join(f"{name}={p['s']}s/{p['calls']}c/{p['pct_wall']}%" for name, p in summary["phases"].items())
    logger.info(
        "[retain-timing] bank=%s store=%s docs=%s wall=%ss %sKB/s | %s",
        summary["bank_id"] or "-",
        summary["store"] or "-",
        summary["documents"],
        summary["wall_s"],
        summary["kb_per_s"],
        parts,
    )


def timed_retain(fn):
    """Open a phase accumulator for the whole of one retain call.

    A decorator rather than an `async with` inside the method: the body it needs to cover is the
    entire method, and wrapping it in place would re-indent a few hundred lines for no behavioural
    reason -- a diff nobody can review against a change that does nothing.

    Labels come from the call's own arguments, so nothing has to be threaded down. The store label
    is resolved lazily and defensively: it is only a metric attribute, and a retain must not fail
    because the label could not be worked out.
    """
    import functools

    @functools.wraps(fn)
    async def wrapper(self, bank_id, contents, *args, **kwargs):
        try:
            from ..memories import get_memories

            store = type(get_memories()).__name__
        except Exception:
            store = ""
        try:
            content_bytes = sum(len(c.get("content") or "") for c in contents)
        except Exception:
            content_bytes = 0
        async with retain_timing(
            bank_id=bank_id,
            store=store,
            documents=len(contents) if contents else 0,
            content_bytes=content_bytes,
        ):
            return await fn(self, bank_id, contents, *args, **kwargs)

    return wrapper
