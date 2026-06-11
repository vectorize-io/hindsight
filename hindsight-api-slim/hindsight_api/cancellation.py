"""Cooperative cancellation for long-running engine operations.

Recall runs as a staged pipeline whose heavy stages — graph expansion and
cross-encoder reranking — execute in worker threads (``run_in_executor``) that
asyncio task cancellation cannot interrupt once they have started. Cancelling
the awaiting task only unblocks the ``await``; the thread keeps burning CPU to
completion. So rather than rely on task cancellation, callers thread a
``CancellationToken`` through ``RequestContext`` and the engine checks it at
stage boundaries (``raise_if_cancelled``), bailing out *before* dispatching the
next expensive stage.

This is cooperative by design: it cannot stop a computation already inside a
worker thread, but it does stop an abandoned recall from progressing into — or
past — that work, which is what starves the instance in issue #2122. The token
lives on ``RequestContext``, so any operation that receives one (recall today;
reflect/consolidation/MCP later) can adopt the same checkpoints, and any driver
(client disconnect today; a deadline tomorrow) can fire it.
"""

from __future__ import annotations

import asyncio


class OperationCancelledError(Exception):
    """Raised at a checkpoint when the operation has been cancelled.

    Carries the ``reason`` set by whoever cancelled (e.g. "client disconnected")
    so the HTTP layer can translate it into the appropriate status code instead
    of a generic 500.

    NOTE: this is a plain ``Exception`` on purpose, NOT ``BaseException``. The
    recall/reflect pipelines have broad ``except Exception`` handlers that would
    otherwise swallow it — those handlers re-raise ``OperationCancelledError``
    explicitly (see ``_search_with_retries``) so cancellation propagates to the
    HTTP layer. A ``BaseException`` would dodge those handlers but also slip past
    legitimate ``isinstance(result, Exception)`` checks (e.g. the reflect agent's
    ``asyncio.gather(..., return_exceptions=True)`` tool-result handling), which
    expect every non-tuple result to be an ``Exception``.
    """

    def __init__(self, reason: str = "operation cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class CancellationToken:
    """A one-shot, cooperative cancellation signal.

    Cheap to poll (``raise_if_cancelled``) at stage boundaries and awaitable
    (``wait``) so a driver task can block until cancellation. Safe to share
    across an engine call tree; polling is a no-op until something cancels, and
    cancellation is idempotent (the first reason wins).
    """

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = "operation cancelled"

    def cancel(self, reason: str = "operation cancelled") -> None:
        """Signal cancellation. Idempotent; the first reason recorded wins."""
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been signalled."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """The reason recorded by the first ``cancel`` call."""
        return self._reason

    def raise_if_cancelled(self) -> None:
        """Raise ``OperationCancelledError`` if cancellation has been signalled."""
        if self._event.is_set():
            raise OperationCancelledError(self._reason)

    async def wait(self) -> None:
        """Block until cancellation is signalled."""
        await self._event.wait()
