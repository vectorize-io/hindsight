"""Completion events must observe committed store-owned memories."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memories.base import StoreWriteConflict
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain.types import RetainBatchResult, RetainContentDict


@pytest.mark.asyncio
@pytest.mark.parametrize("use_factory", [False, True])
@pytest.mark.parametrize("previous_count,committed_count", [(0, 4), (4, 4), (0, 0)])
@pytest.mark.parametrize("failure", [None, "commit", "retain", "outbox"])
async def test_completion_counts_committed_document(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    use_factory: bool,
    previous_count: int,
    committed_count: int,
    failure: str | None,
) -> None:
    engine = MemoryEngine.__new__(MemoryEngine)
    fire_event = AsyncMock(side_effect=RuntimeError("outbox failed") if failure == "outbox" else None)
    engine._webhook_manager = MagicMock(fire_event_with_conn=fire_event)
    engine._resolve_retain_config = AsyncMock()
    # Tokenization is unrelated to completion ordering; keep this unit test offline.
    monkeypatch.setattr("hindsight_api.engine.memory_engine.count_tokens", lambda text: 1)
    conn = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    steps: list[str] = []
    visible_count = previous_count

    async def commit() -> None:
        nonlocal visible_count
        steps.append("commit")
        if failure == "commit":
            raise RuntimeError("commit failed")
        visible_count = committed_count

    async def count(**kwargs) -> int:
        assert kwargs["bank_id"] == "test-bank"
        assert kwargs["document_id"] == "test-document"
        steps.append("count")
        return visible_count

    async def abort() -> None:
        steps.append("abort")

    session = MagicMock(commit=AsyncMock(side_effect=commit), abort=AsyncMock(side_effect=abort))
    store = MagicMock(
        store_owned_for=MagicMock(return_value=True),
        begin_retain=AsyncMock(return_value=session),
        count_document_memories=AsyncMock(side_effect=count),
    )
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)

    @asynccontextmanager
    async def acquire(_backend) -> AsyncIterator[MagicMock]:
        yield conn

    monkeypatch.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", acquire)
    engine._get_backend = AsyncMock()
    contents: list[RetainContentDict] = [{"content": "Alice works at Google", "document_id": "test-document"}]

    async def retain(**kwargs) -> RetainBatchResult:
        assert kwargs["retain_session"] is session
        callback = kwargs["outbox_callback_factory"](contents) if use_factory else kwargs["outbox_callback"]
        assert callback is not None
        await callback(conn)
        if failure == "retain":
            raise RuntimeError("retain failed")
        # Unchanged retains create no units, but must still report the existing total.
        return RetainBatchResult([[]], TokenUsage(), 0)

    engine._retain_batch_async_internal = AsyncMock(side_effect=retain)
    callback = engine._build_retain_outbox_callback("test-bank", contents, "test-operation", schema="test-schema")
    factory = engine._build_retain_outbox_callback_factory("test-bank", "test-operation", schema="test-schema")
    execution = engine._run_retain_execution(
        bank_id="test-bank",
        contents=contents,
        request_context=RequestContext(),
        document_id=None,
        fact_type_override=None,
        document_tags=None,
        operation_id="test-operation",
        strategy=None,
        outbox_callback=None if use_factory else callback,
        outbox_callback_factory=factory if use_factory else None,
        start_time=time.time(),
    )

    if failure == "outbox":
        # The store has already committed by the time the outbox runs, so a failed
        # delivery-row write cannot be undone by failing the retain — that would only
        # report a stored document as lost and invite a duplicate re-submit. The retain
        # succeeds and the dropped event is logged.
        with caplog.at_level(logging.ERROR, logger="hindsight_api.engine.memory_engine"):
            await execution
        assert steps == ["commit", "count"]
        assert "outbox write failed" in caplog.text
        assert "test-operation" in caplog.text
        return

    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await execution
        # A failed single-batch retain aborts instead of committing: committing would store the
        # document's chunks with none of their facts, and the retry would skip them as unchanged.
        # A failed commit aborts too — it releases what the session still buffers. Either way no
        # success event may escape.
        assert steps == (["abort"] if failure == "retain" else ["commit", "abort"])
        engine._webhook_manager.fire_event_with_conn.assert_not_awaited()
        return

    await execution
    assert steps == ["commit", "count"]
    engine._webhook_manager.fire_event_with_conn.assert_awaited_once()
    call = engine._webhook_manager.fire_event_with_conn.await_args
    assert call.args[0].data.memory_unit_count == committed_count
    assert call.args[0].bank_id == "test-bank"
    assert call.args[0].operation_id == "test-operation"
    assert call.kwargs["schema"] == "test-schema"
    assert call.kwargs["schema"] == "test-schema"


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicts", range(1, MemoryEngine._APPEND_CONFLICT_ATTEMPTS + 1))
async def test_append_commit_conflict_is_redone_in_process(
    monkeypatch: pytest.MonkeyPatch,
    conflicts: int,
) -> None:
    """An append's session commit is a compare-and-set on the document it read, so it can
    lose that race *after* the pipeline returned — past ``_retain_batch_with_append_retry``,
    which only sees the pipeline. The execution redoes the whole append on a fresh session
    instead of failing: a lost race used to drop the turn.

    Three attempts is the ceiling (``_APPEND_CONFLICT_ATTEMPTS``), so a third conflict
    raises rather than looping. Either way the losing attempts must deliver no completion
    event — their work was never committed.
    """
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._webhook_manager = MagicMock(fire_event_with_conn=AsyncMock())
    engine._resolve_retain_config = AsyncMock()
    monkeypatch.setattr("hindsight_api.engine.memory_engine.count_tokens", lambda text: 1)
    conn = MagicMock()
    conn.transaction.return_value.__aenter__ = AsyncMock()
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=False)
    steps: list[str] = []
    sessions: list[MagicMock] = []

    async def begin_retain(**_kwargs) -> MagicMock:
        attempt = len(sessions) + 1

        async def commit() -> None:
            steps.append(f"commit{attempt}")
            if attempt <= conflicts:
                raise StoreWriteConflict("document moved")

        async def abort() -> None:
            steps.append(f"abort{attempt}")

        session = MagicMock(commit=AsyncMock(side_effect=commit), abort=AsyncMock(side_effect=abort))
        sessions.append(session)
        return session

    store = MagicMock(
        store_owned_for=MagicMock(return_value=True),
        begin_retain=AsyncMock(side_effect=begin_retain),
        count_document_memories=AsyncMock(return_value=2),
    )
    monkeypatch.setattr("hindsight_api.engine.memories.get_memories", lambda: store)

    @asynccontextmanager
    async def acquire(_backend) -> AsyncIterator[MagicMock]:
        yield conn

    monkeypatch.setattr("hindsight_api.engine.memory_engine.acquire_with_retry", acquire)
    engine._get_backend = AsyncMock()
    contents: list[RetainContentDict] = [
        {"content": "Alice works at Google", "document_id": "test-document", "update_mode": "append"}
    ]

    async def retain(**kwargs) -> RetainBatchResult:
        steps.append("retain")
        # The pipeline consumes its input, so a redo must have been handed a pristine copy.
        assert kwargs["contents"][0]["content"] == "Alice works at Google"
        kwargs["contents"][0]["content"] = ""
        callback = kwargs["outbox_callback"]
        assert callback is not None
        await callback(conn)
        return RetainBatchResult([[]], TokenUsage(), 0)

    engine._retain_batch_async_internal = AsyncMock(side_effect=retain)
    callback = engine._build_retain_outbox_callback("test-bank", contents, "test-operation", schema="test-schema")
    execution = engine._run_retain_execution(
        bank_id="test-bank",
        contents=contents,
        request_context=RequestContext(),
        document_id=None,
        fact_type_override=None,
        document_tags=None,
        operation_id="test-operation",
        strategy=None,
        outbox_callback=callback,
        outbox_callback_factory=None,
        start_time=time.time(),
    )

    if conflicts == MemoryEngine._APPEND_CONFLICT_ATTEMPTS:
        with pytest.raises(StoreWriteConflict):
            await execution
        assert steps == ["retain", "commit1", "abort1", "retain", "commit2", "abort2", "retain", "commit3", "abort3"]
        # The last attempt raises instead of opening a session it would never commit.
        assert len(sessions) == conflicts
        # Nothing was committed, so nothing may be announced.
        engine._webhook_manager.fire_event_with_conn.assert_not_awaited()
        return

    await execution
    lost = [step for attempt in range(1, conflicts + 1) for step in ("retain", f"commit{attempt}", f"abort{attempt}")]
    assert steps == [*lost, "retain", f"commit{conflicts + 1}"]
    # A redo gets its OWN session: reusing the aborted one would commit onto a session the
    # store has already released.
    assert len(sessions) == conflicts + 1
    # One event, not one per attempt: the losing attempts' deferred outbox writes are dropped
    # with the session they belonged to.
    engine._webhook_manager.fire_event_with_conn.assert_awaited_once()
