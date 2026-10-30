"""The wrapper forwards reflect's abstain and search_queries options into ReflectRequest.

Mirrors the TypeScript wrapper's ``reflect_abstain_options.test.ts``. A wrapper that dropped
``abstain`` would hand a memory watcher a written answer for every moment it asks about.
"""

from unittest.mock import AsyncMock

from hindsight_client import Hindsight


def _captured_request(mock):
    return mock.call_args.args[1]


async def test_options_default_to_unset():
    client = Hindsight(base_url="http://localhost:8888")
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "does my change go against a decision?")
    request = _captured_request(client._memory_api.reflect)
    assert not request.abstain
    assert request.search_queries is None


async def test_options_reach_the_request():
    client = Hindsight(base_url="http://localhost:8888")
    client._memory_api.reflect = AsyncMock()

    await client.areflect(
        "bank",
        "does my change go against a decision?",
        abstain=True,
        search_queries=["parse_quantity units", "disk quotas"],
    )
    request = _captured_request(client._memory_api.reflect)
    assert request.abstain is True
    assert request.search_queries == ["parse_quantity units", "disk quotas"]
