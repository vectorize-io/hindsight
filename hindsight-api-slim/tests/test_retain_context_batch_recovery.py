"""Provider checkpoints through the streaming retain caller, across restart."""

import asyncio
import copy
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine import db_utils, memories
from hindsight_api.engine.chunk_ids import build_chunk_id
from hindsight_api.engine.retain import orchestrator
from hindsight_api.engine.retain.types import RetainContent

OPERATION = "2b1e277e-e45e-4a74-81b1-d44e2a78c626"
CHUNKS = ["chunk A source", "chunk B target"]
DATE = datetime(2026, 9, 24, tzinfo=timezone.utc)


@dataclass
class BatchRun:
    """State retained across worker attempts at the storage/provider boundaries."""

    metadata: dict[str, Any] = field(default_factory=dict)
    requests: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    facts: dict[tuple[str, str, int], list[str]] = field(default_factory=dict)
    first_committed: asyncio.Event = field(default_factory=asyncio.Event)
    crash: bool = True
    unavailable_accounts: set[str] = field(default_factory=set)

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any]:
        snapshot = copy.deepcopy(self.metadata)
        # Both chunk tasks can observe no checkpoint before either submits.
        await asyncio.sleep(0)
        return {"result_metadata": snapshot}

    async def execute(self, query: str, *args: Any) -> None:
        if "result_metadata || $1::jsonb" in query:
            self.metadata.update(json.loads(args[0]))
            if "facts_committed_document_ids" in query:
                self.metadata.setdefault("facts_committed_document_ids", []).extend(json.loads(args[1]))

    async def submit_batch(self, requests: list[dict[str, Any]]) -> dict[str, str]:
        batch_id = f"batch-{len(self.requests)}"
        self.requests[batch_id] = requests
        await asyncio.sleep(0)
        return {"batch_id": batch_id}

    def target(self, batch_id: str) -> str:
        return self.requests[batch_id][0]["body"]["messages"][-1]["content"].rsplit("\nContent:\n", 1)[1]

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        if self.crash and self.target(batch_id) == CHUNKS[1]:
            await asyncio.wait_for(self.first_committed.wait(), timeout=5)
            raise RuntimeError("worker interrupted after A committed")
        return {"status": "completed", "request_counts": {"completed": 1, "total": 1}}

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        return [
            {
                "custom_id": "chunk_0",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": json.dumps({"facts": [{"what": self.target(batch_id)}]})}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                },
            }
        ]

    async def write_facts(self, **kwargs: Any) -> list[list[str]]:
        bank, doc = kwargs["bank_id"], kwargs["effective_doc_id"]
        if kwargs["is_first_batch"] and not kwargs["doc_replace_done"][0]:
            self.facts = {key: text for key, text in self.facts.items() if key[:2] != (bank, doc)}
            kwargs["doc_replace_done"][0] = True
        kwargs["doc_tracking_done"][0] = True
        ids = []
        for chunk, extracted in zip(kwargs["batch_chunk_meta"], kwargs["batch_extracted"]):
            assert extracted.mentioned_at == DATE
            self.facts.setdefault((bank, doc, chunk.chunk_index), []).append(extracted.fact_text)
            ids.append(f"{doc}-{chunk.chunk_index}")
        self.first_committed.set()
        return [ids]


@pytest.fixture
def batch_run(monkeypatch) -> BatchRun:
    state = BatchRun()

    @asynccontextmanager
    async def connection(pool: object) -> AsyncIterator[BatchRun]:
        yield state

    monkeypatch.setattr(db_utils, "acquire_with_retry", connection)
    monkeypatch.setattr(orchestrator, "acquire_with_retry", connection)
    store = SimpleNamespace(
        store_owned_for=lambda bank_id: True, derives_semantic_links_internally_for=lambda bank_id: True
    )
    monkeypatch.setattr(memories, "get_memories", lambda: store)
    monkeypatch.setattr(orchestrator, "_store_document_bodies", AsyncMock())
    monkeypatch.setattr(orchestrator, "_pre_resolve_phase1", AsyncMock())
    monkeypatch.setattr(orchestrator, "_streaming_store_owned_retain", state.write_facts)
    monkeypatch.setattr(orchestrator, "_record_retain_document_outcome", AsyncMock())
    monkeypatch.setattr(
        orchestrator.embedding_processing, "generate_embeddings_batch", AsyncMock(return_value=[[0.0, 1.0]])
    )
    return state


async def stream(
    state: BatchRun,
    *,
    doc: str = "doc",
    offset: int = 0,
    changed: str | None = None,
    event_date: datetime = DATE,
) -> None:
    impl = SimpleNamespace(
        provider="openai",
        model="test",
        openai_service_tier=None,
        batch_account_key="replacement" if state.unavailable_accounts else "account",
        submit_batch=state.submit_batch,
        get_batch_status=state.get_batch_status,
        retrieve_batch_results=state.retrieve_batch_results,
    )
    llm = SimpleNamespace(
        batch_provider_impl=AsyncMock(
            side_effect=lambda account_key=None: None if account_key in state.unavailable_accounts else impl
        )
    )
    config = replace(
        HindsightConfig.from_env(),
        retain_context_chars=800,
        retain_batch_enabled=True,
        retain_extract_causal_links=False,
    )
    chunks = list(CHUNKS)
    if changed == "source":
        chunks[0] = "edited chunk A source"
    elif changed == "settings":
        config = replace(config, retain_mission="Changed extraction instructions")
    elif changed == "disable":
        config = replace(config, retain_context_chars=0)
    source = RetainContent(
        content="\n".join(chunks),
        event_date=event_date,
        event_date_is_default=True,
    )
    await orchestrator._streaming_retain_batch(
        pool=object(),
        embeddings_model=None,
        llm_config=llm,
        entity_resolver=SimpleNamespace(discard_pending_stats=Mock(), flush_pending_stats=AsyncMock()),
        format_date_fn=str,
        bank_id="bank",
        contents_dicts=[{"content": source.content}],
        contents=[source],
        config=config,
        document_id=doc,
        is_first_batch=offset == 0,
        fact_type_override=None,
        document_tags=None,
        log_buffer=[],
        start_time=time.time(),
        all_pre_chunks=chunks,
        previous_sources=["", chunks[0]],
        chunk_to_content=[0, 0],
        chunk_batch_size=1,
        operation_id=OPERATION,
        force_reextract=True,
        chunk_index_offset=offset,
    )


@pytest.mark.asyncio
async def test_streaming_context_batch_resumes_own_checkpoint_after_partial_commit(batch_run: BatchRun) -> None:
    with pytest.raises(RuntimeError, match="worker interrupted"):
        await stream(batch_run)
    assert list(batch_run.facts.values()) == [[CHUNKS[0]]]
    assert len(batch_run.requests) == 2
    batch_run.crash = False
    await stream(batch_run, event_date=DATE.replace(day=25))
    assert len(batch_run.requests) == 2  # Reuse both original provider batches.
    assert batch_run.facts == {("bank", "doc", 0): [CHUNKS[0]], ("bank", "doc", 1): [CHUNKS[1]]}
    # Same operation and text, different document or later internal slice.
    await stream(batch_run, doc="other")
    await stream(batch_run, offset=10)
    assert len(batch_run.requests) == 6
    assert batch_run.facts["bank", "other", 0] == [CHUNKS[0]]
    assert batch_run.facts["bank", "doc", 10] == [CHUNKS[0]]
    assert batch_run.facts["bank", "doc", 11] == [CHUNKS[1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "settings", "disable"])
async def test_streaming_context_batch_rejects_changed_inputs(batch_run: BatchRun, changed: str) -> None:
    with pytest.raises(RuntimeError, match="worker interrupted"):
        await stream(batch_run)
    batch_run.crash = False
    with pytest.raises(RuntimeError, match="batch prompts changed"):
        await stream(batch_run, changed=changed)
    assert len(batch_run.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("matches", [True, False])
async def test_streaming_context_batch_matches_legacy_checkpoint(batch_run: BatchRun, matches: bool) -> None:
    with pytest.raises(RuntimeError, match="worker interrupted"):
        await stream(batch_run)
    # The previous implementation persisted one top-level checkpoint, which
    # could belong to either chunk. Only its fingerprint can establish ownership.
    key_b = f"retain_batch:{build_chunk_id('bank', 'doc', 1)}"
    saved_b = batch_run.metadata.pop(key_b)
    if matches:
        batch_run.metadata.update(saved_b)
    else:
        key_a = f"retain_batch:{build_chunk_id('bank', 'doc', 0)}"
        batch_run.metadata.update(batch_run.metadata[key_a])
    batch_run.crash = False
    await stream(batch_run, event_date=DATE.replace(day=25) if matches else DATE)
    assert len(batch_run.requests) == (2 if matches else 3)
    assert batch_run.facts == {("bank", "doc", 0): [CHUNKS[0]], ("bank", "doc", 1): [CHUNKS[1]]}


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False])
async def test_batch_account_guard_requires_checkpoint_ownership(batch_run: BatchRun, legacy: bool) -> None:
    with pytest.raises(RuntimeError, match="worker interrupted"):
        await stream(batch_run)
    if legacy:
        key_b = f"retain_batch:{build_chunk_id('bank', 'doc', 1)}"
        batch_run.metadata = batch_run.metadata[key_b]
    batch_run.crash = False
    batch_run.unavailable_accounts.add("account")
    if legacy:
        # The shared checkpoint cannot be matched without its serving account.
        # It must not stop a chunk whose ownership was never established.
        await stream(batch_run)
        assert len(batch_run.requests) == 4
        assert batch_run.facts == {("bank", "doc", 0): [CHUNKS[0]], ("bank", "doc", 1): [CHUNKS[1]]}
    else:
        with pytest.raises(RuntimeError, match="submitted by the LLM member"):
            await stream(batch_run)
        assert len(batch_run.requests) == 2
