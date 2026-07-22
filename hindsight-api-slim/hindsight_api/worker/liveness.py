"""Thread-based liveness server for the Hindsight worker.

The worker runs its poller and its async ``/health`` (readiness) server on a
single asyncio event loop. During Bedrock inference, litellm performs
*synchronous* botocore credential resolution and SigV4 signing on the loop
thread, which can block the loop for several seconds. A Kubernetes
livenessProbe pointed at the async ``/health`` endpoint then times out and the
pod is SIGKILLed (exitCode 137) even though the process is busy, not dead.

This module provides a liveness signal that survives a busy loop: a heartbeat
timestamp bumped ~once per second by a task on the main loop, served by a tiny
stdlib HTTP server on a *separate daemon thread*. While the loop is blocked in
sync signing the heartbeat goes momentarily stale, but the loop resumes and the
heartbeat catches up well within the threshold, so the probe keeps returning
200. A genuinely wedged loop never catches up and the probe fails, letting
Kubernetes restart the pod.

The liveness server deliberately does NOT touch the asyncpg pool: the pool is
bound to the main event loop and cannot be awaited from another thread. The
DB check stays on the async ``/health`` endpoint as the *readiness* signal.
"""

from __future__ import annotations

import http.server
import logging
import threading
import time

logger = logging.getLogger(__name__)


class Heartbeat:
    """A monotonic timestamp the main loop bumps to prove it can make progress."""

    def __init__(self) -> None:
        self._last = time.monotonic()

    def beat(self) -> None:
        """Record that the main loop just ran."""
        self._last = time.monotonic()

    def age(self) -> float:
        """Seconds elapsed since the last :meth:`beat`."""
        return time.monotonic() - self._last


def is_alive(age_seconds: float, threshold_seconds: float) -> bool:
    """Liveness predicate.

    Alive while the heartbeat is younger than ``threshold_seconds``. A busy loop
    (e.g. blocked in sync SigV4 signing) stays alive as long as it resumes
    within the threshold; a wedged loop that never resumes fails.
    """
    return age_seconds < threshold_seconds


def start_liveness_server(
    host: str,
    port: int,
    heartbeat: Heartbeat,
    threshold_seconds: float,
) -> threading.Thread:
    """Start a daemon-thread HTTP liveness server and return its thread.

    Any GET returns 200 while the heartbeat is fresh and 503 once it goes stale.
    The server runs on its own thread with its own socket loop, so a blocked
    main event loop cannot starve it. The thread is a daemon and needs no
    explicit shutdown; it dies with the process.
    """

    class _LivenessHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - name mandated by BaseHTTPRequestHandler
            alive = is_alive(heartbeat.age(), threshold_seconds)
            body = b"ok\n" if alive else b"stale\n"
            self.send_response(200 if alive else 503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args, **kwargs):  # silence per-request stderr logging
            pass

    server = http.server.ThreadingHTTPServer((host, port), _LivenessHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="worker-liveness",
        daemon=True,
    )
    thread.start()
    logger.info(f"Liveness server started on http://{host}:{port}/ (staleness threshold={threshold_seconds}s)")
    return thread
