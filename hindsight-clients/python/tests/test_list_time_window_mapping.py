"""The maintained wrapper threads the list time window through to the request.

Kept in step with the TypeScript wrapper's equivalent,
tests/list_time_window_mapping.test.ts: the wrappers enumerate query parameters
rather than passing a dict through, so a parameter added to the API but not to a
wrapper is silently dropped for every consumer of that language — with no type
error to catch it. The `client-coverage-check` CI tool only validates request
*bodies*, so a GET query parameter has nothing but this test behind it.
"""

from unittest.mock import MagicMock

import pytest

from hindsight_client import Hindsight


@pytest.fixture
def captured(monkeypatch):
    client = Hindsight(base_url="http://example.invalid")
    seen: dict[str, object] = {}

    async def fake_list(**kwargs):
        seen.update(kwargs)
        return MagicMock(items=[], total=0)

    monkeypatch.setattr(client._memory_api, "list_memories", fake_list)
    return client, seen


def test_list_memories_forwards_the_time_window(captured):
    client, seen = captured

    client.list_memories(
        "test-bank",
        time_field="mentioned_at",
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-02-01T00:00:00Z",
    )

    assert seen["time_field"] == "mentioned_at"
    assert seen["start_date"] == "2024-01-01T00:00:00Z"
    assert seen["end_date"] == "2024-02-01T00:00:00Z"


def test_list_memories_omits_the_window_by_default(captured):
    client, seen = captured

    client.list_memories("test-bank")

    # None rather than a default axis: the endpoint keeps its own ordering and
    # drops nothing unless a caller asks for a window.
    assert seen["time_field"] is None
    assert seen["start_date"] is None
    assert seen["end_date"] is None
