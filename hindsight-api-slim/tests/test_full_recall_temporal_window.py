"""``FullRecallRequest.temporal_window`` reaches the store as the type it declares.

The field is annotated ``tuple[datetime, datetime] | None``, and a store that claims the recall
unpacks it as one. The caller's own value is a :class:`TemporalWindow`, so the engine converts at
the boundary -- the same thing it does for ``recall_unified`` before handing the window to the
retrieval pipeline.

``FullRecallRequest`` is a plain dataclass with no validation at construction, so a model passed
straight through is accepted here and only fails inside the store, on the first subscript, as a
``TypeError`` the recall surfaces as a 500. That is why this test drives the real endpoint and
asserts on what the store *received*: a test that built a ``FullRecallRequest`` from the declared
type would pass while the engine still handed over the model.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.memories import set_memories
from hindsight_api.engine.memories.base import FullRecallRequest
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import RecallResult
from hindsight_api.models import RequestContext
from tests.test_memories_extension import InMemoryMemories

WINDOW_START = datetime(2023, 4, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2023, 6, 30, 23, 59, 59, tzinfo=timezone.utc)


class _RecordingStore(InMemoryMemories):
    """Claims the recall and keeps the window it was handed, exactly as it arrived."""

    def __init__(self, bank_id: str) -> None:
        super().__init__({})
        self.bank_id = bank_id
        #: Sentinel rather than ``None``: "never called" and "called with no window" differ here.
        self.seen: object = "not-called"

    async def full_recall(self, request: FullRecallRequest) -> RecallResult:
        self.seen = request.temporal_window
        return RecallResult(results=[])


@pytest.fixture
def restore_default_store() -> Iterator[None]:
    yield
    set_memories(None)


async def _claiming_store(memory: MemoryEngine, request_context: RequestContext) -> _RecordingStore:
    bank_id = f"so-temporal-{uuid.uuid4().hex[:8]}"
    store = _RecordingStore(bank_id)
    set_memories(store)
    # The store owns the facts, never the bank row, so create it the way a retain would.
    await memory.ensure_bank_profile(bank_id, request_context=request_context)
    return store


@pytest.mark.asyncio
async def test_a_claiming_store_receives_the_window_as_a_tuple(
    api_client, memory, request_context, restore_default_store
):
    store = await _claiming_store(memory, request_context)

    response = await api_client.post(
        f"/v1/default/banks/{store.bank_id}/memories/recall",
        json={
            "query": "what shipped that quarter",
            "types": ["world"],
            "limit": 10,
            "temporal_window": {"start": "2023-04-01T00:00:00Z", "end": "2023-06-30T23:59:59Z"},
        },
    )

    assert response.status_code == 200, response.text
    assert store.seen != "not-called", "the store never saw the recall"
    # The declared type, both bounds, in order -- not a TemporalWindow and not a list.
    assert isinstance(store.seen, tuple), f"store got {type(store.seen).__name__}, not a tuple"
    assert store.seen == (WINDOW_START, WINDOW_END)


@pytest.mark.asyncio
async def test_no_window_stays_none(api_client, memory, request_context, restore_default_store):
    """The conversion must not invent a window for a recall that asked for none."""
    store = await _claiming_store(memory, request_context)

    response = await api_client.post(
        f"/v1/default/banks/{store.bank_id}/memories/recall",
        json={"query": "what shipped that quarter", "types": ["world"], "limit": 10},
    )

    assert response.status_code == 200, response.text
    assert store.seen is None
