"""
Test that reflect()/areflect() forward include_tool_calls into the ReflectRequest.

The high-level wrapper previously only exposed include_facts, so there was no way
to request the reflect trace (tool_calls / llm_calls) without dropping down to the
generated API. These tests pin the wrapper -> ReflectRequest.include mapping so the
trace stays reachable from the convenience layer.
"""

from unittest.mock import AsyncMock

from hindsight_client import Hindsight


def _make_client():
    return Hindsight(base_url="http://localhost:8888")


def _captured_include(mock):
    """Pull the ReflectRequest.include built by the wrapper from the mock call."""
    request_obj = mock.call_args.args[1]
    return request_obj.include


async def test_areflect_no_includes_by_default():
    """Without any include flags, include should be None (smallest request)."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query")
    assert _captured_include(client._memory_api.reflect) is None


async def test_areflect_include_tool_calls_sets_trace_request():
    """include_tool_calls=True should populate include.tool_calls with output on by default."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query", include_tool_calls=True)
    include = _captured_include(client._memory_api.reflect)
    assert include is not None
    assert include.tool_calls is not None
    assert include.tool_calls.output is True
    # facts stays disabled when only tool calls are requested
    assert include.facts is None


async def test_areflect_include_tool_call_output_false():
    """include_tool_call_output=False should request inputs-only trace."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query", include_tool_calls=True, include_tool_call_output=False)
    include = _captured_include(client._memory_api.reflect)
    assert include.tool_calls is not None
    assert include.tool_calls.output is False


async def test_areflect_facts_and_tool_calls_combine():
    """include_facts and include_tool_calls should both be representable in one request."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query", include_facts=True, include_tool_calls=True)
    include = _captured_include(client._memory_api.reflect)
    assert include.facts is not None
    assert include.tool_calls is not None


async def test_areflect_tool_call_output_ignored_without_tool_calls():
    """include_tool_call_output is a no-op unless include_tool_calls is set."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    await client.areflect("bank", "query", include_tool_call_output=False)
    assert _captured_include(client._memory_api.reflect) is None


def test_reflect_sync_forwards_include_tool_calls():
    """The sync wrapper should forward include_tool_calls through to the request."""
    client = _make_client()
    client._memory_api.reflect = AsyncMock()

    client.reflect("bank", "query", include_tool_calls=True)
    include = _captured_include(client._memory_api.reflect)
    assert include is not None
    assert include.tool_calls is not None
    assert include.tool_calls.output is True
