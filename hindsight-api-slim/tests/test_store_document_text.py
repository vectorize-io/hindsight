"""
Tests for the HINDSIGHT_API_STORE_DOCUMENT_TEXT flag.

When disabled, the retain pipeline still extracts facts/entities and embeds
them, but the raw source text is dropped: documents.original_text is stored as
NULL and chunks.chunk_text is stored as an empty string. Recall must be
unaffected because it reads from memory_units, not original_text.
"""

import os
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api import config as config_module
from hindsight_api.api import create_app
from hindsight_api.engine.memory_engine import Budget
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools


@pytest_asyncio.fixture
async def api_client(memory):
    """Async HTTP client against the FastAPI app (exercises response-model validation)."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


LONG_CONTENT = """
Alice Johnson is a senior software engineer at Acme Corp. She specializes in
distributed systems and leads the platform team. Bob Smith works in marketing
and reports to Carol. The team uses Kubernetes and deploys to AWS. Code reviews
are mandatory before merging.
"""


@pytest.fixture
def store_document_text_disabled(memory):
    """Disable raw document/chunk text storage (server-wide) for a test.

    ``store_document_text`` is hierarchical (env → tenant → bank) and the
    ConfigResolver snapshots the global layer at construction, so flipping the
    env var alone is not enough — the snapshot must be updated too (mirrors
    ``enable_audit_default``). Restores both afterwards so other tests in the
    same worker see the default behaviour again.
    """
    from dataclasses import replace

    resolver = memory._config_resolver
    original_env = os.environ.get("HINDSIGHT_API_STORE_DOCUMENT_TEXT")
    original_global = resolver._global_config
    os.environ["HINDSIGHT_API_STORE_DOCUMENT_TEXT"] = "false"
    config_module.clear_config_cache()
    resolver._global_config = replace(original_global, store_document_text=False)
    try:
        yield
    finally:
        resolver._global_config = original_global
        if original_env is None:
            os.environ.pop("HINDSIGHT_API_STORE_DOCUMENT_TEXT", None)
        else:
            os.environ["HINDSIGHT_API_STORE_DOCUMENT_TEXT"] = original_env
        config_module.clear_config_cache()


@pytest.mark.asyncio
async def test_text_storage_disabled_nulls_text_but_keeps_memories(
    memory, request_context, store_document_text_disabled
):
    """With the flag off, raw text is dropped but facts/recall still work."""
    bank_id = f"test_text_off_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-text-001"

    try:
        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=LONG_CONTENT,
            context="team overview",
            document_id=document_id,
            request_context=request_context,
        )

        # Pipeline still ran: facts were extracted and stored.
        assert len(unit_ids) > 0, "Facts should still be extracted when text storage is disabled"

        # documents.original_text is dropped (NULL).
        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["original_text"] is None, "Raw document text must not be stored"
        assert doc["memory_unit_count"] > 0, "Memory units should still be created"

        # chunks.chunk_text is blanked.
        chunks = await memory.list_document_chunks(
            bank_id=bank_id, document_id=document_id, request_context=request_context
        )
        assert chunks["total"] > 0, "Chunks should still be stored (for graph/structure)"
        for chunk in chunks["items"]:
            assert chunk["chunk_text"] == "", "Raw chunk text must not be stored"

        # Recall is unaffected — it reads from memory_units, not original_text.
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.LOW,
            max_tokens=500,
            request_context=request_context,
        )
        assert len(result.results) > 0, "Recall must still return facts when text storage is disabled"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_default_mode_stores_text(memory, request_context):
    """By default (flag on) raw document and chunk text are persisted."""
    bank_id = f"test_text_on_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-text-002"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=LONG_CONTENT,
            context="team overview",
            document_id=document_id,
            request_context=request_context,
        )

        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["original_text"] is not None
        assert "Alice Johnson" in doc["original_text"]

        chunks = await memory.list_document_chunks(
            bank_id=bank_id, document_id=document_id, request_context=request_context
        )
        assert chunks["total"] > 0
        assert any(chunk["chunk_text"] for chunk in chunks["items"]), "Chunk text should be stored by default"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_append_mode_rejected_when_text_disabled(memory, request_context, store_document_text_disabled):
    """update_mode='append' must be rejected when document text storage is disabled.

    Append rebuilds the document by reading back the stored original_text; with
    storage off there is nothing to read, so appending would silently drop the
    prior content. The pipeline rejects it instead of losing data.
    """
    bank_id = f"test_append_text_off_{datetime.now(timezone.utc).timestamp()}"

    with pytest.raises(ValueError, match="update_mode='append' is not supported"):
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Some content",
                    "document_id": "doc-append-text-off",
                    "update_mode": "append",
                }
            ],
            request_context=request_context,
        )


@pytest.mark.asyncio
async def test_get_document_endpoint_returns_null_text(
    api_client, memory, request_context, store_document_text_disabled
):
    """GET document must return 200 with null original_text (not fail response validation).

    The DocumentResponse model declares original_text as optional; a non-optional
    str would raise ResponseValidationError -> HTTP 500 when the text is NULL.
    """
    bank_id = f"test_get_doc_text_off_{datetime.now(timezone.utc).timestamp()}"
    document_id = "doc-http-text-off"

    try:
        retain = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={"items": [{"content": "Alice works at Acme Corp.", "document_id": document_id}]},
        )
        assert retain.status_code == 200, retain.text

        resp = await api_client.get(f"/v1/default/banks/{bank_id}/documents/{document_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["original_text"] is None, "Raw text must be null when text storage is disabled"
        assert body["memory_unit_count"] > 0

        # Append to the same document is rejected as a client error (400), not a 500.
        append = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={"items": [{"content": "More text.", "document_id": document_id, "update_mode": "append"}]},
        )
        assert append.status_code == 400, append.text
        assert "append" in append.json()["detail"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_version_endpoint_reports_store_document_text(api_client, store_document_text_disabled):
    """The /version feature flags expose store_document_text so the UI can warn."""
    resp = await api_client.get("/version")
    assert resp.status_code == 200, resp.text
    assert resp.json()["features"]["store_document_text"] is False


def test_reflect_excludes_expand_tool_when_text_disabled():
    """The reflect 'expand' tool (get chunk/document source text) is dropped when text storage is disabled."""
    with_text = {t["function"]["name"] for t in get_reflect_tools(include_expand=True)}
    without_text = {t["function"]["name"] for t in get_reflect_tools(include_expand=False)}

    assert "expand" in with_text, "expand should be available by default"
    assert "expand" not in without_text, "expand must be excluded when document text is not stored"
    # Other reflect tools are unaffected.
    assert "recall" in without_text
    assert "done" in without_text


def test_store_document_text_is_bank_configurable():
    """store_document_text must be overridable per bank (in the configurable set)."""
    from hindsight_api.config import HindsightConfig

    assert "store_document_text" in HindsightConfig.get_configurable_fields()


@pytest.mark.asyncio
async def test_store_document_text_per_bank_override(memory, request_context):
    """The flag is overridable per bank: one bank drops raw text while another,
    on the same server default, keeps it."""
    ts = datetime.now(timezone.utc).timestamp()
    off_bank = f"test_sdt_off_{ts}"
    on_bank = f"test_sdt_on_{ts}"
    document_id = "doc-perbank-001"

    try:
        # Both banks exist; disable raw-text storage on off_bank only.
        await memory.get_bank_profile(bank_id=off_bank, request_context=request_context)
        await memory.get_bank_profile(bank_id=on_bank, request_context=request_context)
        await memory._config_resolver.update_bank_config(off_bank, {"store_document_text": False})

        for bank_id in (off_bank, on_bank):
            await memory.retain_async(
                bank_id=bank_id,
                content=LONG_CONTENT,
                context="team overview",
                document_id=document_id,
                request_context=request_context,
            )

        # off_bank: raw source dropped, facts still present.
        off_doc = await memory.get_document(document_id, off_bank, request_context=request_context)
        assert off_doc is not None
        assert off_doc["original_text"] is None, "off_bank must not store raw document text"
        assert off_doc["memory_unit_count"] > 0
        off_chunks = await memory.list_document_chunks(
            bank_id=off_bank, document_id=document_id, request_context=request_context
        )
        assert off_chunks["total"] > 0
        assert all(c["chunk_text"] == "" for c in off_chunks["items"]), "off_bank chunk text must be blank"

        # on_bank (default): raw source kept.
        on_doc = await memory.get_document(document_id, on_bank, request_context=request_context)
        assert on_doc is not None
        assert on_doc["original_text"] is not None and "Alice Johnson" in on_doc["original_text"]
        on_chunks = await memory.list_document_chunks(
            bank_id=on_bank, document_id=document_id, request_context=request_context
        )
        assert any(c["chunk_text"] for c in on_chunks["items"]), "on_bank chunk text must be stored"
    finally:
        await memory.delete_bank(off_bank, request_context=request_context)
        await memory.delete_bank(on_bank, request_context=request_context)
