from datetime import datetime


def format_task_error(e: BaseException) -> str:
    """Render an exception for a task failure log line / stored error_message.

    Always prefixes the exception class. Plenty of exceptions carry an empty
    ``str()`` — ``TimeoutError()``, ``CancelledError()``, a bare ``raise
    SomeError()`` — and the bare interpolation those log lines used produced
    ``Task execution failed: graph_maintenance, error: ``, which says nothing at
    all about what went wrong (issue #3218).
    """
    message = str(e)
    return f"{type(e).__name__}: {message}" if message else type(e).__name__


class RetryTaskAt(Exception):
    """Raise from a task handler to schedule a retry at a specific time."""

    def __init__(self, retry_at: datetime, message: str = ""):
        self.retry_at = retry_at
        super().__init__(message)


class DeferOperation(Exception):
    """Raise from an extension hook (or task handler) to requeue the
    operation for execution at a later time, without counting as a retry.

    Unlike `RetryTaskAt`, this is not a failure: `retry_count` is not
    incremented and `error_message` is not written. Use this for
    backpressure / "not yet, try later" decisions made before or during
    task execution (e.g. quota windows, warming dependencies, upstream
    rate limits).

    Worker-only: raising this from a hook called in HTTP request context
    (e.g. `validate_recall` for a synchronous recall) will surface as an
    unhandled 500 — there is no queue to defer to.
    """

    def __init__(self, exec_date: datetime, reason: str = ""):
        self.exec_date = exec_date
        self.reason = reason
        super().__init__(reason)
