"""The wrapper forwards the search_observations budget options into ReflectRequest.

These are per-call overrides of the bank's ``reflect_default_options`` (#4483). A
wrapper that silently dropped them would leave every Python consumer stuck with the
bank default, with nothing in the response to say the argument was ignored.
"""

from unittest.mock import AsyncMock

from hindsight_client import Hindsight


def _make_client():
    return Hindsight(base_url="http://localhost:8888")


def _captured_request(mock):
    return mock.call_args.args[1]


async def test_options_default_to_none_so_the_bank_decides():
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query")
    request = _captured_request(client._memory_api.reflect)
    assert request.reflect_search_observations_max_tokens is None
    assert request.reflect_search_observations_include_entities is None


async def test_options_reach_the_request():
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect(
        "bank",
        "query",
        reflect_search_observations_max_tokens=3000,
        reflect_search_observations_include_entities=False,
    )
    request = _captured_request(client._memory_api.reflect)
    assert request.reflect_search_observations_max_tokens == 3000
    assert request.reflect_search_observations_include_entities is False
