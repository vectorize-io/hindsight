"""JSON append keeps independent extraction inputs while merging the stored body."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine import memories
from hindsight_api.engine.retain import orchestrator
from hindsight_api.engine.response_models import TokenUsage


@pytest.mark.asyncio
async def test_json_append_windows_reset_between_incoming_items(monkeypatch) -> None:
    base = [{"role": "assistant", "content": "Stored option 2: Monday with Theo."}]
    first = [{"role": "assistant", "content": "PRIVATE sibling option 2: Friday with Nia."}]
    second = [{"role": "user", "content": "Independent decision: choose option 2."}]
    texts = [json.dumps(part) for part in (base, first, second)]
    conn = SimpleNamespace(fetchrow=AsyncMock(return_value={"original_text": texts[0], "content_hash": "base"}))

    @asynccontextmanager
    async def connection(pool: object) -> AsyncIterator[SimpleNamespace]:
        yield conn

    store = SimpleNamespace(assert_writable=AsyncMock(), store_owned_for=lambda bank_id: True)
    monkeypatch.setattr(memories, "get_memories", lambda: store)
    monkeypatch.setattr(orchestrator, "acquire_with_retry", connection)
    streaming = AsyncMock(return_value=orchestrator.RetainBatchResult([[]], TokenUsage(), 0))
    monkeypatch.setattr(orchestrator, "_streaming_retain_batch", streaming)
    await orchestrator.retain_batch(
        pool=None,
        embeddings_model=None,
        llm_config=None,
        entity_resolver=None,
        format_date_fn=None,
        bank_id="bank",
        document_id="doc",
        contents_dicts=[{"content": text, "update_mode": "append", "document_id": "doc"} for text in texts[1:]],
        config=replace(HindsightConfig.from_env(), retain_context_chars=2000, retain_chunk_size=500),
    )
    prepared = streaming.call_args.kwargs
    assert prepared["all_pre_chunks"] == texts
    assert prepared["chunk_to_content"] == [0, 1, 2]
    assert prepared["previous_sources"] == ["", texts[0], texts[0]]
    assert json.loads(prepared["full_document_body"]) == base + first + second
