"""A metrics endpoint per worker process.

With ``--workers N`` every uvicorn worker is its own process with its own metrics registry, but
they share one listening socket, so a scrape of ``/metrics`` on the API port reaches ONE worker,
picked by the kernel. Two consequences:

* counters and histograms jump between processes from one scrape to the next, and a monotonic
  counter that goes backwards reads to Prometheus as a reset -- ``rate()`` over any of them is
  wrong;
* per-process series such as ``process_cpu_seconds_total`` describe a random worker, so a worker
  whose event loop is saturated is invisible while the pod total still looks like headroom.

When ``HINDSIGHT_API_METRICS_WORKER_BASE_PORT`` is set, each worker additionally serves its own
registry on ``base_port + slot``, where ``slot`` (0..workers-1) is claimed with an exclusive
``flock`` on a per-slot lock file. The kernel drops the lock when the process exits, so a worker the
supervisor respawns takes over the slot its predecessor held. A scraper then targets every port and
sees each worker as its own target. ``/metrics`` on the API port is unchanged.
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from typing import IO

from prometheus_client import REGISTRY, CollectorRegistry, start_http_server

logger = logging.getLogger(__name__)

# The lock file of the slot this process holds. Kept referenced for the life of the process:
# closing it would release the flock and let another worker claim the same port.
_held: IO[str] | None = None
_port: int | None = None


def _lock_dir(base_port: int) -> str:
    # Per base port, so two API servers on one host with different base ports do not share slots.
    return os.path.join(tempfile.gettempdir(), f"hindsight-metrics-slots-{base_port}")


def claim_slot(base_port: int, slots: int, *, lock_dir: str | None = None) -> tuple[int, IO[str]] | None:
    """Claim the lowest free slot in ``0..slots-1``; return ``(port, lock_file)`` or None.

    The caller must keep ``lock_file`` open for as long as it serves the port.
    """
    directory = lock_dir or _lock_dir(base_port)
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
        return base_port + slot, handle
    return None


def start(
    base_port: int,
    slots: int,
    *,
    lock_dir: str | None = None,
    registry: CollectorRegistry = REGISTRY,
    addr: str = "0.0.0.0",
) -> int | None:
    """Serve ``registry`` on this worker's own port. Returns the port, or None if not started.

    Idempotent within a process: a second call returns the port already being served. Never raises:
    a missing per-worker endpoint must not stop the API from starting.
    """
    global _held, _port
    if _port is not None:
        return _port
    if base_port <= 0 or slots <= 0:
        return None
    claimed = claim_slot(base_port, slots, lock_dir=lock_dir)
    if claimed is None:
        logger.warning(
            "[metrics] no free per-worker metrics slot in %d-%d; this worker is only visible on /metrics",
            base_port,
            base_port + slots - 1,
        )
        return None
    port, handle = claimed
    try:
        start_http_server(port, addr=addr, registry=registry)
    except OSError as e:
        handle.close()
        logger.warning("[metrics] could not serve per-worker metrics on port %d: %s", port, e)
        return None
    _held, _port = handle, port
    logger.info("[metrics] per-worker metrics on port %d (pid %d)", port, os.getpid())
    return port
