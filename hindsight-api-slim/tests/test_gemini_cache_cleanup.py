"""Best-effort cache cleanup cannot hold a completed reflect open indefinitely."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager


@pytest.mark.asyncio
async def test_session_cleanup_bounds_hung_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.import_module(GeminiCacheManager.__module__), "_DEFAULT_DELETE_TIMEOUT_SECONDS", 0.01, raising=False
    )
    cancelled: set[str] = set()

    async def delete_cache(*, name: str) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(name)

    delete = AsyncMock(side_effect=delete_cache)
    client = SimpleNamespace(aio=SimpleNamespace(caches=SimpleNamespace(delete=delete)))
    manager = GeminiCacheManager(client)
    with patch.object(manager, "_create_cache", side_effect=["cachedContents/one", "cachedContents/two"]):
        for _ in range(2):
            await manager.create_incremental(
                session_id="reflect-session", model="model", system_instruction="prefix", contents=[]
            )

    # This outer bound is only a regression safety net: production must enforce
    # its own bound and cancel the stalled SDK operations before it expires.
    await asyncio.wait_for(manager.delete_session("reflect-session"), timeout=0.2)

    assert cancelled == {"cachedContents/one", "cachedContents/two"}
    assert delete.await_count == 2
    await manager.delete_session("reflect-session")
    assert delete.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_still_propagates_caller_cancellation() -> None:
    started = asyncio.Event()

    async def delete_cache(*, name: str) -> None:
        started.set()
        await asyncio.Event().wait()

    client = SimpleNamespace(aio=SimpleNamespace(caches=SimpleNamespace(delete=delete_cache)))
    manager = GeminiCacheManager(client)
    task = asyncio.create_task(manager.delete("cachedContents/one"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
