"""A store shedding a write must not burn the operation's retries.

Ingesting a large corpus, two of ten documents ended `failed` with the store's own refusal:

    StatusCode.UNAVAILABLE
    "fold behind: un-folded tail ... holds 1,075,346,560 bytes, over the
     1,073,741,824-byte write bound"

That is the write guard working — it sheds rather than running out of memory — and it clears on
its own as the fold catches up. But the worker gave the task the same small retry budget it gives
a genuinely broken one, so the attempts ran out while the store was still legitimately shedding
and the documents were lost. The same corpus's shed documents succeeded immediately when
resubmitted against a drained bank.

These pin the classification, not the wording of any one message: what must hold is that a
backpressure refusal defers (no retry consumed) and everything else still fails.
"""

import uuid

import pytest

from hindsight_api.engine.embeddings import REEMBED_GUARD_SQLSTATE
from hindsight_api.models import RequestContext
from hindsight_api.worker.backpressure import is_store_backpressure
from hindsight_api.worker.exceptions import DeferOperation, RetryTaskAt


class _AioRpcErrorLike(Exception):
    """Shaped like what the provider re-raises: the status and detail live in `str()`."""


def _shed_error() -> Exception:
    return _AioRpcErrorLike(
        "AioRpcError: <AioRpcError of RPC that terminated with:\n"
        "\tstatus = StatusCode.UNAVAILABLE\n"
        '\tdetails = "fold behind: un-folded tail of tenant_x__bank holds 1075346560 bytes, '
        'over the 1073741824-byte write bound (half the max tail scan)"\n>"'
    )


# --- what counts as backpressure -------------------------------------------------------------


def test_the_stores_own_refusal_is_recognised():
    assert is_store_backpressure(_shed_error())


def test_it_is_found_through_the_wrapping_the_provider_adds():
    """The worker never sees the store's error bare.

    The provider raises its own error `from` the RPC one, so a check that only looked at the
    outermost exception would classify every real shed as an ordinary failure — which is the bug
    this fixes, not a hypothetical.
    """
    try:
        try:
            raise _shed_error()
        except Exception as inner:
            raise RuntimeError("retain failed for document beam-9") from inner
    except Exception as outer:
        assert is_store_backpressure(outer)


def test_an_implicit_context_chain_is_walked_too():
    """`raise X` inside an `except` sets `__context__`, not `__cause__`."""
    try:
        try:
            raise _shed_error()
        except Exception:
            raise RuntimeError("wrapped without from")
    except Exception as outer:
        assert is_store_backpressure(outer)


# --- what does NOT ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "ValueError: document_id must not be empty",
        "asyncpg.exceptions.UniqueViolationError: duplicate key",
        "TimeoutError: task exceeded the wall-clock limit",
        'AioRpcError: status = StatusCode.INVALID_ARGUMENT, details = "bad vector dimension"',
        "wal: namespace tenant_x__bank has no manifest",
    ],
)
def test_ordinary_failures_are_not_deferred(message):
    """A broken task must still fail.

    Deferring does not consume a retry, so misclassifying a permanent error would requeue it
    forever instead of surfacing it — the failure mode of being too generous here, and the reason
    the markers are the store's refusal text rather than "UNAVAILABLE appears anywhere".
    """
    assert not is_store_backpressure(Exception(message))


def test_a_cycle_in_the_chain_terminates():
    """Defensive: a self-referencing chain must not hang the worker's failure path."""
    a = Exception("nothing to see")
    b = Exception("also fine")
    a.__cause__ = b
    b.__cause__ = a
    assert is_store_backpressure(a) is False


# --- an upstream's 429 -----------------------------------------------------------------------


def test_an_upstream_429_is_recognised_through_wrapping_and_nothing_else_is():
    from hindsight_api.engine.aiohttp_session import UpstreamHTTPError
    from hindsight_api.worker.backpressure import is_upstream_rate_limit

    def raised(status: int) -> BaseException:
        try:
            try:
                raise UpstreamHTTPError(status, "busy", {}, "http://gateway/v1/embeddings")
            except Exception as inner:
                raise RuntimeError("embed round failed") from inner
        except Exception as outer:
            return outer

    assert is_upstream_rate_limit(raised(429))
    assert not is_upstream_rate_limit(raised(500))
    assert not is_upstream_rate_limit(raised(400))
    assert not is_upstream_rate_limit(ValueError("429 in a message is not a status"))


# --- the engine hands a shed to the poller untouched (#6528) ----------------------------------


@pytest.mark.asyncio
async def test_a_retain_the_store_sheds_reaches_the_poller_as_is(memory, monkeypatch):
    """Turned into a counted `RetryTaskAt`, a shed would spend one of the retain's few retries.

    The poller only defers what it can recognise, so the engine must re-raise the store's own
    error for every task type, not just the ones that know about backpressure.
    """

    async def shedding_retain(_task_dict: object) -> None:
        raise RuntimeError("retain failed") from _shed_error()

    monkeypatch.setattr(memory, "_handle_batch_retain", shedding_retain)
    bank_id = f"test-store-shed-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=RequestContext(internal=True))

    with pytest.raises(RuntimeError) as exc_info:
        await memory.execute_task({"type": "batch_retain", "bank_id": bank_id, "contents": [{"content": "x"}]})
    assert not isinstance(exc_info.value, RetryTaskAt)
    assert is_store_backpressure(exc_info.value)


@pytest.mark.asyncio
async def test_the_reembed_guard_keeps_its_own_deferral(memory, monkeypatch):
    """The re-embed write guard's refusal also says "writes are shed".

    It is not a store backlog: it clears once a server on the new embedding model runs the task, so
    it keeps its own deferral and never counts toward the store-shed cap.
    """

    class _Guard(Exception):
        sqlstate = REEMBED_GUARD_SQLSTATE

    async def guarded_retain(_task_dict: object) -> None:
        raise _Guard("hindsight re-embed: writes are shed until this server runs embedding model new-model")

    monkeypatch.setattr(memory, "_handle_batch_retain", guarded_retain)
    bank_id = f"test-store-shed-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=RequestContext(internal=True))

    with pytest.raises(DeferOperation):
        await memory.execute_task({"type": "batch_retain", "bank_id": bank_id, "contents": [{"content": "x"}]})
