"""Recognising a write the worker should bring back later rather than fail.

Two shapes of that answer live here: a store shedding under its own indexing backlog
(`is_store_backpressure`) and an append that lost the race on the document it extends
(`is_append_conflict`). Neither says anything about the payload, and both clear on their own, so
both are deferred instead of counting against the operation's retries.

A store under sustained ingest can refuse a write because its own indexing has fallen behind —
not because the request is bad, not because anything is broken, and not in a way that says
anything about the payload. It is `UNAVAILABLE` in the gRPC sense: retryable, and self-clearing as
soon as the store catches up.

Treated as an ordinary failure it is none of those things. The worker gives a failing task a small
number of attempts over a few minutes, so a long ingest whose backlog takes longer than that to
drain exhausts them while the store is still legitimately shedding, and the operation is marked
permanently `failed`. Ingesting a large corpus then loses whichever documents happened to be in
flight when the backlog crossed the bound — the tail of the run, silently, with a `failed` row
that reads like the content was at fault.

So it is classified as backpressure and deferred instead (`DeferOperation`: "not yet, try later",
no `retry_count` bump), which is what the store is actually asking for. That holds for every task
type: the engine re-raises a shed as is rather than turning it into a counted retry (#6528). The
worker counts these deferrals and fails the task once there are too many of them, so a store that
never recovers surfaces as a failure instead of a task deferred forever (#6529).

Matched on the message rather than on a typed error because the store speaks gRPC and the
provider re-raises `AioRpcError`; there is no shared exception class to catch, and inventing one
would mean a coordinated change across the store's client. The strings below are the store's own
refusal text, and a false positive costs a deferral rather than a lost operation — a task that is
genuinely broken still fails on its next attempt, when the message no longer matches.
"""

# Substrings a store's own backpressure refusal carries. `fold behind` and the write-bound message
# are a particular store's wording; `UNAVAILABLE` covers the gRPC status a shedding store returns
# generally.
_BACKPRESSURE_MARKERS = (
    "fold behind",
    "write bound",
    "StatusCode.UNAVAILABLE",
    "writes are shed",
)


def is_store_backpressure(exc: BaseException) -> bool:
    """Is `exc` a store asking for this write to be retried later?

    Walks the `__cause__`/`__context__` chain: the provider wraps the store's error before the
    worker sees it, so the marker is usually not on the outermost exception.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # Matched by wording only, not by `StoreWriteUnavailable` type. When these deferrals had no cap,
        # a store that raises that type for good (a cutover that never ends, a test that raises it on
        # purpose) deferred for ever, and matching it by type took the api-slim CI shard that runs those
        # tests down every time, until the runner was killed. The deferrals are capped now (#6529), but
        # why the shard died was never pinned down, so adding the type is its own change.
        text = str(cur)
        if any(marker in text for marker in _BACKPRESSURE_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def is_append_conflict(exc: BaseException) -> bool:
    """Is `exc` an append that lost its race on the document it extends?

    A lost race is the opposite of a fatal error: the precondition rejects the write whole, so
    nothing was stored, and redoing it on a fresh read is always safe. Failing it terminally —
    which is what the worker does with any other exception — drops the turn it carried.

    Walks the cause/context chain like `is_store_backpressure`, since the conflict is usually
    wrapped by the time the worker sees it. Matched on the exception type here, not the message:
    unlike the store's gRPC refusal, both of these are our own classes.
    """
    # Imported here, not at module scope: the worker is imported by `worker.main` before the
    # engine is, and pulling the engine package in from this module would invert that order.
    from ..engine.memories.base import StoreWriteConflict
    from ..engine.retain.types import ConcurrentAppendConflict

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ConcurrentAppendConflict, StoreWriteConflict)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


#: Task types whose upstream 429s are deferred instead of failed. Knowledge ingest only: an LLM can answer
#: 429 for a quota that is used up, which never clears, so a retain's or a reflect's 429 keeps failing as
#: before. Shared by the engine (which passes such a 429 up) and the poller (which defers it); the engine
#: used to test ``task_type == "knowledge_ingest"`` on its own, which could drift from the poller's set.
RATE_LIMIT_DEFERRED_TASKS = frozenset({"knowledge_ingest"})


def is_upstream_rate_limit(exc: BaseException) -> bool:
    """Is `exc` an upstream (an embedder, through its gateway) answering 429, "too many requests"?

    Like a shedding store, it says nothing about the payload and clears as the upstream's load does —
    but only after the call's own retries ran out, so the load lasted longer than they cover. Walks the
    cause/context chain like the others; the status is read with `remote_retry.status_code_of`, which
    knows the attribute each HTTP client keeps it on.
    """
    from ..engine.remote_retry import status_code_of

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if status_code_of(cur) == 429:
            return True
        cur = cur.__cause__ or cur.__context__
    return False
