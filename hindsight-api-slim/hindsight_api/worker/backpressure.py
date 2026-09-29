"""Recognising a store that is asking for the write to come back later.

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
no `retry_count` bump), which is what the store is actually asking for.

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
        text = str(cur)
        if any(marker in text for marker in _BACKPRESSURE_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False
