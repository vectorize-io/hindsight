"""A provider-owned cache manager must work across successive event loops."""

import asyncio
from unittest.mock import patch

from hindsight_api.engine.providers.gemini_cache import GeminiCacheManager


def test_cache_creation_coalesces_on_each_event_loop() -> None:
    manager = GeminiCacheManager(object())
    creates: list[str] = []

    async def create_cache(**kwargs: object) -> str:
        # Yield so the second caller actually contends for the manager's lock.
        await asyncio.sleep(0)
        creates.append(str(kwargs["system_instruction"]))
        return f"cachedContents/{len(creates)}"

    async def concurrent_loads(prefix: str) -> list[str | None]:
        return await asyncio.gather(
            *(manager.get_or_create(model="model", system_instruction=prefix) for _ in range(2))
        )

    with patch.object(manager, "_create_cache", side_effect=create_cache):
        first = asyncio.run(concurrent_loads("first prefix"))
        second = asyncio.run(concurrent_loads("second prefix"))

    assert first == ["cachedContents/1"] * 2
    assert second == ["cachedContents/2"] * 2
    assert creates == ["first prefix", "second prefix"]


def test_fresh_cache_entries_are_shared_across_event_loops() -> None:
    manager = GeminiCacheManager(object())

    async def load() -> str | None:
        return await manager.get_or_create(model="model", system_instruction="shared prefix")

    with patch.object(manager, "_create_cache", return_value="cachedContents/shared") as create:
        assert asyncio.run(load()) == "cachedContents/shared"
        assert asyncio.run(load()) == "cachedContents/shared"
        create.assert_awaited_once()
