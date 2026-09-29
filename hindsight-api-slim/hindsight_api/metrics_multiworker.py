"""One ``/metrics`` that covers every worker, each series labelled with the worker it came from.

With ``--workers N`` every uvicorn worker is its own process with its own metrics registry, but
they share one listening socket, so a scrape of ``/metrics`` reaches ONE worker, picked by the
kernel. Two consequences:

* counters and histograms jump between processes from one scrape to the next, and a monotonic
  counter that goes backwards reads to Prometheus as a reset -- rates over them are wrong;
* per-process series such as ``process_cpu_seconds_total`` describe a random worker, so a worker
  whose event loop is saturated is invisible while the pod total still looks like headroom.

Labelling each worker's series is not enough on its own: a scrape would still return one worker's
series, the others would be missing from about half the scrapes, and Prometheus marks a missing
series stale. So when ``HINDSIGHT_API_METRICS_WORKER_LABEL`` is on, every worker:

* claims a slot (0..N-1) with an exclusive ``flock`` on a per-slot lock file -- the kernel drops it
  when the process exits, so a worker the supervisor respawns reuses its predecessor's slot and
  the label stays bounded to N values;
* writes its registry's exposition to a directory shared by the server's workers every
  ``SNAPSHOT_INTERVAL_S``;
* answers ``/metrics`` with every live worker's series, each carrying ``api_worker="<slot>"``: its
  own read live, the others from their latest snapshot. A snapshot older than ``STALE_AFTER_S`` is
  a worker that is gone, and is skipped.

One port, one scrape target, and every worker in every scrape.
"""

from __future__ import annotations

import fcntl
import glob
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterable
from typing import IO

from prometheus_client import REGISTRY, CollectorRegistry, generate_latest
from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

#: The label every series gets. Not ``worker``: the task poller already uses that key.
LABEL = "api_worker"

#: How often a worker publishes its snapshot. Well under a typical 15-30 s scrape interval, so the
#: other workers' series are at most this old.
SNAPSHOT_INTERVAL_S = 5.0

#: A snapshot older than this belongs to a worker that is gone (crashed, or replaced and not yet
#: republished under the same slot).
STALE_AFTER_S = 3 * SNAPSHOT_INTERVAL_S


def shared_dir() -> str:
    """The directory this server's workers share. Keyed by the supervisor (the workers' parent),
    so two servers on one host, or a restarted server, never read each other's snapshots."""
    return os.path.join(tempfile.gettempdir(), f"hindsight-metrics-{os.getppid()}")


def claim_slot(directory: str, slots: int) -> tuple[int, IO[str]] | None:
    """Claim the lowest free slot in ``0..slots-1``; return ``(slot, lock_file)`` or None.

    The caller must keep ``lock_file`` open for as long as it publishes under the slot.
    """
    os.makedirs(directory, exist_ok=True)
    for slot in range(slots):
        handle = open(os.path.join(directory, f"slot-{slot}.lock"), "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        handle.write(str(os.getpid()))
        handle.flush()
        return slot, handle
    return None


class _Families:
    """A collector that yields already-built metric families."""

    def __init__(self, families: Iterable[Metric]) -> None:
        self._families = list(families)

    def collect(self) -> Iterable[Metric]:
        return iter(self._families)


def merge_expositions(parts: Iterable[tuple[str, str]]) -> bytes:
    """Merge ``(slot, exposition text)`` pairs into one exposition, labelling every sample.

    Series are kept apart per worker, never summed: summing is a query-time decision, and a sum
    would hide exactly the per-worker skew this exists to show.
    """
    families: dict[str, Metric] = {}
    for slot, text in parts:
        for family in text_string_to_metric_families(text):
            merged = families.get(family.name)
            if merged is None:
                merged = Metric(family.name, family.documentation, family.type, family.unit)
                families[family.name] = merged
            for sample in family.samples:
                merged.add_sample(
                    sample.name, {**sample.labels, LABEL: slot}, sample.value, sample.timestamp, sample.exemplar
                )
    registry = CollectorRegistry(auto_describe=False)
    registry.register(_Families(families.values()))
    return generate_latest(registry)


class WorkerMetrics:
    """This worker's share of the multi-worker ``/metrics``: publishes its snapshot, renders all."""

    def __init__(
        self,
        directory: str,
        slot: int,
        lock_file: IO[str] | None = None,
        registry: CollectorRegistry = REGISTRY,
    ) -> None:
        self.directory = directory
        self.slot = slot
        self._lock_file = lock_file  # held open: closing it would release the slot
        self._registry = registry
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned: set[str] = set()

    def _snapshot_path(self, slot: int) -> str:
        return os.path.join(self.directory, f"worker-{slot}.prom")

    def write_snapshot(self) -> None:
        """Publish this worker's exposition, atomically (a reader never sees half a file)."""
        path = self._snapshot_path(self.slot)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(generate_latest(self._registry))
        os.replace(tmp, path)

    def start(self, interval_s: float = SNAPSHOT_INTERVAL_S) -> None:
        self.write_snapshot()

        def _loop() -> None:
            while not self._stop.wait(interval_s):
                try:
                    self.write_snapshot()
                except Exception as e:  # never let the publisher die silently or kill the worker
                    self._warn_once("snapshot", f"[metrics] could not publish worker snapshot: {e}")

        self._thread = threading.Thread(target=_loop, name="hindsight-metrics-snapshot", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def render(self, *, stale_after_s: float = STALE_AFTER_S) -> bytes:
        """Every live worker's series, each labelled with its slot. This worker's are read live."""
        parts: list[tuple[str, str]] = [(str(self.slot), generate_latest(self._registry).decode())]
        now = time.time()
        for path in sorted(glob.glob(os.path.join(self.directory, "worker-*.prom"))):
            slot = os.path.basename(path)[len("worker-") : -len(".prom")]
            if slot == str(self.slot):
                continue
            try:
                if now - os.stat(path).st_mtime > stale_after_s:
                    continue
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                list(text_string_to_metric_families(text))  # reject a corrupt snapshot up front
            except Exception as e:
                self._warn_once(f"read:{slot}", f"[metrics] skipping worker {slot}'s snapshot: {e}")
                continue
            parts.append((slot, text))
        return merge_expositions(parts)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


def start_worker_metrics(slots: int, *, directory: str | None = None) -> WorkerMetrics | None:
    """Claim a slot and start publishing. Returns None -- plain ``/metrics`` -- if no slot is free.

    Never raises: metrics must not stop the API from starting.
    """
    directory = directory or shared_dir()
    try:
        claimed = claim_slot(directory, max(1, slots))
        if claimed is None:
            logger.warning("[metrics] no free worker slot in %s; this worker serves only its own metrics", directory)
            return None
        slot, lock_file = claimed
        worker = WorkerMetrics(directory, slot, lock_file)
        worker.start()
        logger.info("[metrics] worker-labelled metrics: slot %d (pid %d)", slot, os.getpid())
        return worker
    except Exception as e:
        logger.warning("[metrics] worker-labelled metrics disabled: %s", e)
        return None
