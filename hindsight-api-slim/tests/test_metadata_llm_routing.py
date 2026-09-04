"""End-to-end retain coverage for metadata-based LLM routing."""

import copy
import dataclasses
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from hindsight_api.config import LLMMetadataRoute, LLMStrategyConfig, _get_raw_config
from hindsight_api.engine import memory_engine as engine_module
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.multi_llm import MultiLLMProvider


@dataclasses.dataclass
class _CallCounts:
    primary: int = 0
    secondary: int = 0

    def reset(self) -> None:
        self.primary = 0
        self.secondary = 0


def _install_metadata_router(
    memory,
    monkeypatch,
    *,
    operation: str = "retain",
    key: str = "tags",
    value: str = "sensitive",
) -> _CallCounts:
    counts = _CallCounts()
    primary = LLMProvider(provider="mock", api_key="", base_url="", model="primary")
    secondary = LLMProvider(provider="mock", api_key="", base_url="", model="secondary")
    primary_call = primary.call
    secondary_call = secondary.call
    primary_call_with_tools = primary.call_with_tools
    secondary_call_with_tools = secondary.call_with_tools

    async def record_primary(*args, **kwargs):
        counts.primary += 1
        return await primary_call(*args, **kwargs)

    async def record_secondary(*args, **kwargs):
        counts.secondary += 1
        return await secondary_call(*args, **kwargs)

    async def record_primary_with_tools(*args, **kwargs):
        counts.primary += 1
        return await primary_call_with_tools(*args, **kwargs)

    async def record_secondary_with_tools(*args, **kwargs):
        counts.secondary += 1
        return await secondary_call_with_tools(*args, **kwargs)

    monkeypatch.setattr(primary, "call", record_primary)
    monkeypatch.setattr(secondary, "call", record_secondary)
    monkeypatch.setattr(primary, "call_with_tools", record_primary_with_tools)
    monkeypatch.setattr(secondary, "call_with_tools", record_secondary_with_tools)
    setattr(
        memory,
        f"_{operation}_llm_config",
        MultiLLMProvider(
            [primary, secondary],
            LLMStrategyConfig(
                mode="metadata",
                routes=[LLMMetadataRoute(key=key, value=value, member=1)],
            ),
        ),
    )
    return counts


def _install_ambiguous_metadata_router(memory, monkeypatch) -> None:
    members = [
        LLMProvider(provider="mock", api_key="", base_url="", model="primary"),
        LLMProvider(provider="mock", api_key="", base_url="", model="internal"),
        LLMProvider(provider="mock", api_key="", base_url="", model="sensitive"),
    ]

    async def unexpected_call(*args, **kwargs):
        pytest.fail("ambiguous retain reached an LLM provider")

    for member in members:
        monkeypatch.setattr(member, "call", unexpected_call)
        monkeypatch.setattr(member, "call_with_tools", unexpected_call)

    memory._retain_llm_config = MultiLLMProvider(
        members,
        LLMStrategyConfig(
            mode="metadata",
            routes=[
                LLMMetadataRoute(key="metadata.classification", value="internal", member=1),
                LLMMetadataRoute(key="metadata.clearance", value="restricted", member=2),
            ],
        ),
    )


async def test_http_preserves_explicit_empty_classification(api_client, memory, monkeypatch) -> None:
    captured: list[dict] = []

    async def capture_retain(*args, **kwargs):
        captured.extend(kwargs["contents"])
        return [[]], None

    monkeypatch.setattr(memory, "retain_batch_async", capture_retain)
    response = await api_client.post(
        "/v1/default/banks/metadata-routing-http/memories",
        json={
            "items": [
                {
                    "content": "Explicitly declassified append.",
                    "document_id": "document-1",
                    "update_mode": "append",
                    "tags": [],
                    "metadata": {},
                }
            ]
        },
    )

    assert response.status_code == 200
    assert captured[0]["tags"] == []
    assert captured[0]["metadata"] == {}


async def test_split_sync_retain_rejects_cross_member_classification_before_llm(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    _install_ambiguous_metadata_router(memory_no_llm_verify, monkeypatch)
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    bank_id = f"metadata-routing-mixed-sync-{uuid.uuid4().hex[:8]}"
    contents = [
        {"content": " ".join(["internal"] * 80), "metadata": {"classification": "internal"}},
        {"content": " ".join(["sensitive"] * 80), "metadata": {"clearance": "restricted"}},
    ]

    with pytest.raises(ValueError, match="select multiple members"):
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            contents,
            request_context=request_context,
        )


async def test_queued_retain_rejects_cross_member_classification_before_children(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    _install_ambiguous_metadata_router(memory_no_llm_verify, monkeypatch)
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    bank_id = f"metadata-routing-mixed-queued-{uuid.uuid4().hex[:8]}"
    contents = [
        {"content": " ".join(["internal"] * 80), "metadata": {"classification": "internal"}},
        {"content": " ".join(["sensitive"] * 80), "metadata": {"clearance": "restricted"}},
    ]

    with pytest.raises(ValueError, match="select multiple members"):
        await memory_no_llm_verify.submit_async_retain(
            bank_id,
            contents,
            request_context=request_context,
        )

    operations = await memory_no_llm_verify.list_operations(bank_id, request_context=request_context)
    assert operations["total"] == 0


async def test_tag_routed_reflect_stays_on_secondary_for_every_scope(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    bank_id = f"metadata-routing-reflect-{uuid.uuid4().hex[:8]}"
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "The restricted launch code is 2468.", "tags": ["sensitive"]}],
        request_context=request_context,
    )
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "The public office opens at nine.", "tags": ["public"]}],
        request_context=request_context,
    )

    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch, operation="reflect")
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "Summarize the available information.",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    # Expand can turn a public fact into its full mixed-classification document,
    # so even an exact public scope must stay on the protected lane.
    calls.reset()
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "When does the office open?",
        tags=["public"],
        tags_match="exact",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.reflect_async(
        bank_id,
        "What is the launch code?",
        tags=["sensitive"],
        tags_match="exact",
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0


async def test_sensitive_pending_facts_route_consolidation_to_secondary(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    from hindsight_api.engine.consolidation import consolidator

    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch, operation="consolidation")

    async def capture_run(memory_engine, bank_id, context, config, llm_config, *args):
        await llm_config.call(messages=[{"role": "user", "content": "Sensitive facts"}], scope="consolidation")
        return {"status": "complete"}

    monkeypatch.setattr(consolidator, "_run_consolidation_job", capture_run)
    result = await consolidator.run_consolidation_job(
        memory_no_llm_verify,
        "metadata-routing-consolidation",
        request_context,
    )

    assert result == {"status": "complete"}
    assert calls.secondary == 1
    assert calls.primary == 0


async def test_sensitive_retain_uses_secondary_without_touching_primary(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    calls = _install_metadata_router(memory_no_llm_verify, monkeypatch)

    bank_id = f"metadata-routing-{uuid.uuid4().hex[:8]}"
    document_id = "private-account"
    unit_ids = await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Alice's private account number is 1234.",
                "document_id": document_id,
                "tags": ["sensitive"],
            }
        ],
        request_context=request_context,
    )

    assert unit_ids[0]
    assert calls.secondary > 0
    assert calls.primary == 0

    memories = await memory_no_llm_verify.list_memory_units(
        bank_id,
        fact_type=["world", "experience"],
        tags=["sensitive"],
        tags_match="all_strict",
        request_context=request_context,
    )
    assert memories["total"] == len(unit_ids[0])

    calls.reset()
    narrowed = dataclasses.replace(_get_raw_config(), retain_batch_tokens=20)
    monkeypatch.setattr(engine_module, "get_config", lambda: narrowed)
    sub_batch_count = 0
    real_iter_sub_batches = engine_module.iter_sub_batches

    def count_sub_batches(*args, **kwargs):
        nonlocal sub_batch_count
        for sub_batch in real_iter_sub_batches(*args, **kwargs):
            sub_batch_count += 1
            yield sub_batch

    monkeypatch.setattr(engine_module, "iter_sub_batches", count_sub_batches)
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": " ".join(
                    f"The private account review entry number {index} was completed today." for index in range(120)
                ),
                "document_id": document_id,
                "update_mode": "append",
            }
        ],
        request_context=request_context,
    )

    assert sub_batch_count > 1
    assert calls.secondary > 0
    assert calls.primary == 0

    # Omitted tags inherit the stored classification, including for later
    # appends after the first append has replaced the document record.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "A second review was recorded.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None and document["tags"] == ["sensitive"]

    # Adding an unrelated tag must not implicitly remove the classifier.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Customer scope added.",
                "document_id": document_id,
                "update_mode": "append",
                "tags": ["customer"],
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None and set(document["tags"]) == {"customer", "sensitive"}

    # Explicitly clearing tags declassifies future appends, but this operation
    # still reprocesses the old sensitive body on the protected lane.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Classification cleared.", "document_id": document_id, "update_mode": "append", "tags": []}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "A public follow-up.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.primary > 0
    assert calls.secondary == 0


async def test_custom_metadata_is_inherited_for_append_routing(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    calls = _install_metadata_router(
        memory_no_llm_verify,
        monkeypatch,
        key="metadata.classification",
        value="restricted",
    )
    bank_id = f"metadata-routing-custom-{uuid.uuid4().hex[:8]}"
    document_id = "restricted-document"

    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Restricted source material.",
                "document_id": document_id,
                "metadata": {"classification": "restricted"},
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    for content in ("First append without metadata.", "Second append without metadata."):
        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [{"content": content, "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0

    # Adding an unrelated key retains the stored classifier.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [
            {
                "content": "Source metadata added.",
                "document_id": document_id,
                "update_mode": "append",
                "metadata": {"source": "crm"},
            }
        ],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None
    assert document["document_metadata"] == {"classification": "restricted", "source": "crm"}

    # An explicit empty map clears custom routing metadata for later appends.
    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Classification cleared.", "document_id": document_id, "update_mode": "append", "metadata": {}}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0

    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Public follow-up.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.primary > 0
    assert calls.secondary == 0


async def test_shared_document_persists_non_first_item_metadata_for_append_routing(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    calls = _install_metadata_router(
        memory_no_llm_verify,
        monkeypatch,
        key="metadata.classification",
        value="restricted",
    )
    bank_id = f"metadata-routing-shared-{uuid.uuid4().hex[:8]}"
    document_id = "shared-restricted-document"
    contents = [
        {"content": "Public preface.", "document_id": document_id},
        {
            "content": "Restricted details.",
            "document_id": document_id,
            "metadata": {"classification": "restricted"},
        },
    ]
    original_contents = copy.deepcopy(contents)

    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        contents,
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0
    assert contents == original_contents

    document = await memory_no_llm_verify.get_document(document_id, bank_id, request_context=request_context)
    assert document is not None
    assert document["document_metadata"] == {"classification": "restricted"}

    calls.reset()
    await memory_no_llm_verify.retain_batch_async(
        bank_id,
        [{"content": "Append without metadata.", "document_id": document_id, "update_mode": "append"}],
        request_context=request_context,
    )
    assert calls.secondary > 0
    assert calls.primary == 0


async def test_append_routing_bulk_reads_store_after_releasing_sql_connection(
    memory_no_llm_verify, monkeypatch
) -> None:
    from hindsight_api.engine.memories import set_memories
    from tests.test_memories_extension import InMemoryMemories

    connection_held = False
    sql_calls: list[tuple[str, list[str], str]] = []

    class FakeConnection:
        async def fetch(self, query, document_ids, bank_id):
            assert connection_held
            sql_calls.append((query, document_ids, bank_id))
            return []

    @asynccontextmanager
    async def fake_acquire(_backend):
        nonlocal connection_held
        connection_held = True
        try:
            yield FakeConnection()
        finally:
            connection_held = False

    class BulkDocumentStore(InMemoryMemories):
        def __init__(self):
            super().__init__({})
            self.bulk_calls = 0

        async def get_document_records(self, *, bank_id, document_ids):
            assert not connection_held
            self.bulk_calls += 1
            return {
                "doc-a": {"tags": ["sensitive"], "metadata": {}},
                "doc-b": {
                    "tags": [],
                    "metadata": {"retain_params": {"metadata": {"classification": "restricted"}}},
                },
            }

    _install_metadata_router(memory_no_llm_verify, monkeypatch)
    monkeypatch.setattr(engine_module, "acquire_with_retry", fake_acquire)
    monkeypatch.setattr(memory_no_llm_verify, "_get_backend", AsyncMock(return_value=object()))
    store = BulkDocumentStore()
    set_memories(store)
    try:
        states = await memory_no_llm_verify._stored_append_routing_states(
            "bank",
            [
                {"content": "a", "document_id": "doc-a", "update_mode": "append"},
                {"content": "b", "document_id": "doc-b", "update_mode": "append"},
            ],
        )
    finally:
        set_memories(None)

    assert len(sql_calls) == 1
    assert "id = ANY($1::text[])" in sql_calls[0][0]
    assert sql_calls[0][1:] == (["doc-a", "doc-b"], "bank")
    assert store.bulk_calls == 1
    assert states["doc-a"].tags == ["sensitive"]
    assert states["doc-b"].metadata == {"classification": "restricted"}


async def test_store_owned_sensitive_append_uses_authoritative_document_tags(
    memory_no_llm_verify, request_context, monkeypatch
) -> None:
    from hindsight_api.engine.memories import set_memories
    from tests.test_memories_extension import InMemoryMemories

    class MetadataAwareStore(InMemoryMemories):
        async def index_facts(self, bank_id, unit_ids, facts, document_id=None, unit_entity_ids=None):
            # The store-owned retain session supplies its public FactRecord
            # shape, while this shared test store predates that seam and still
            # expects ProcessedFact. Preserve the store's observable behavior
            # without making this routing regression depend on that mismatch.
            from hindsight_api.engine.memories.base import StoredMemory

            self.calls.append("index_facts")
            for unit_id, fact in zip(unit_ids, facts):
                self.rows[unit_id] = StoredMemory(
                    unit_id=unit_id,
                    text=fact.text,
                    fact_type=fact.fact_type,
                    context=fact.context,
                    document_id=document_id,
                    chunk_id=fact.chunk_id,
                    tags=list(fact.tags or []),
                    metadata=fact.metadata,
                    created_at=fact.created_at,
                )

        async def get_document_record(self, *, bank_id, document_id, include_text=False):
            record = await super().get_document_record(
                bank_id=bank_id,
                document_id=document_id,
                include_text=include_text,
            )
            if record is not None:
                record["metadata"] = dict(self.documents[document_id]["metadata"])
            return record

    store = MetadataAwareStore({})
    set_memories(store)
    try:
        calls = _install_metadata_router(memory_no_llm_verify, monkeypatch)
        bank_id = f"metadata-routing-store-{uuid.uuid4().hex[:8]}"
        document_id = "store-owned-sensitive"
        store.documents[document_id] = {
            "id": document_id,
            "content_hash": "sensitive-seed",
            "original_text": "Sensitive store-owned content.",
            "chunk_texts": ["Sensitive store-owned content."],
            "chunks": ["Sensitive store-owned content."],
            "tags": ["sensitive"],
            "metadata": {"retain_params": {}},
        }

        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [{"content": "An append without tags.", "document_id": document_id, "update_mode": "append"}],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0

        calls = _install_metadata_router(
            memory_no_llm_verify,
            monkeypatch,
            key="metadata.classification",
            value="restricted",
        )
        metadata_document_id = "store-owned-restricted"
        store.documents[metadata_document_id] = {
            "id": metadata_document_id,
            "content_hash": "restricted-seed",
            "original_text": "Restricted store-owned content.",
            "chunk_texts": ["Restricted store-owned content."],
            "chunks": ["Restricted store-owned content."],
            "tags": [],
            "metadata": {"retain_params": {"metadata": {"classification": "restricted"}}},
        }

        calls.reset()
        await memory_no_llm_verify.retain_batch_async(
            bank_id,
            [
                {
                    "content": "An append without metadata.",
                    "document_id": metadata_document_id,
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )
        assert calls.secondary > 0
        assert calls.primary == 0
    finally:
        set_memories(None)
