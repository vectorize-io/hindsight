"""DB-free liveness payloads, shared by the API server and the worker.

Liveness answers exactly one question: *is this process wedged beyond
recovery, so that a restart is the only fix?* It must never touch the
database. A liveness probe that runs ``SELECT 1`` turns database
degradation into an outage: every pod fails the probe at once, is killed
mid-flight (claimed async operations are requeued with ``retry_count``
incremented, walking real work toward the permanent-failure cliff), and
then reconnects to re-warm its pool against a database that is already
saturated.

Dependency checks belong in *readiness* (``/health``, ``/health/ready``),
because failing readiness pulls the pod out of the Service instead of
killing it — degradation stays degradation.

Serving this handler at all still proves the one condition a restart does
fix: the event loop is scheduling coroutines. Hindsight runs request
handlers and task work on a single loop, so a loop blocked by a synchronous
call cannot answer even a trivial request within the probe timeout (see
``loop_watchdog.py``).
"""

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Import time is close enough to process start for a probe payload: this
# module is imported while the app is being constructed, long before the
# server binds its port.
_PROCESS_START = time.monotonic()


def uptime_seconds() -> float:
    """Seconds this process has been running, rounded for readability."""
    return round(time.monotonic() - _PROCESS_START, 1)


class LivenessResponse(BaseModel):
    """Payload for the API's DB-free liveness probe."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "alive",
                "version": "0.4.0",
                "uptime_seconds": 812.4,
            }
        }
    )

    status: Literal["alive"] = Field(description='Always "alive" — reaching this handler is the check')
    version: str = Field(description="Hindsight version this process is running")
    uptime_seconds: float = Field(description="Seconds since the process started")


class WorkerLivenessResponse(LivenessResponse):
    """Payload for the worker's DB-free liveness probe."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "alive",
                "version": "0.4.0",
                "uptime_seconds": 812.4,
                "worker_id": "hindsight-worker-0",
                "is_shutdown": False,
                "seconds_since_last_poll": 0.4,
            }
        }
    )

    worker_id: str = Field(description="Identifier of this worker process")
    is_shutdown: bool = Field(description="Whether graceful shutdown has been signalled")
    seconds_since_last_poll: float | None = Field(
        default=None,
        description=(
            "Age of the last completed poll cycle, or null before the first one. "
            "Reported for alerting only — the endpoint stays 200 however stale it "
            "gets, so a saturated database can never trigger a restart."
        ),
    )


def liveness_response() -> LivenessResponse:
    """Build the API liveness payload without touching any dependency."""
    # Imported lazily: ``hindsight_api/__init__`` pulls in MemoryEngine, so a
    # module-level import here would make this module unusable from anything the
    # engine itself imports.
    from hindsight_api import __version__

    return LivenessResponse(status="alive", version=__version__, uptime_seconds=uptime_seconds())


def worker_liveness_response(
    *,
    worker_id: str,
    is_shutdown: bool,
    seconds_since_last_poll: float | None,
) -> WorkerLivenessResponse:
    """Build the worker liveness payload without touching any dependency."""
    # Lazy for the same reason as liveness_response() above.
    from hindsight_api import __version__

    return WorkerLivenessResponse(
        status="alive",
        version=__version__,
        uptime_seconds=uptime_seconds(),
        worker_id=worker_id,
        is_shutdown=is_shutdown,
        seconds_since_last_poll=seconds_since_last_poll,
    )
