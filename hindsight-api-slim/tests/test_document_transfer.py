"""Tests for document export/import between banks (LLM-free transfer).

These exercise the full export → import round trip on a real (pg0) database with
the mock LLM fixture, and crucially assert that import does NOT invoke fact
extraction (the LLM) — it replays the deterministic pipeline and re-embeds.
"""

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.consolidation.consolidator import _create_observation_directly
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.schema import fq_table
from hindsight_api.engine.transfer import import_documents
from hindsight_api.engine.transfer.importer import parse_archive
from hindsight_api.engine.transfer.schema import SCHEMA_VERSION, TransferManifest
from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    RetainResult,
    ValidationResult,
)
from hindsight_api.webhooks.manager import WebhookManager


class _RetainResultCapture(OperationValidatorExtension):
    """Records each RetainResult the engine reports via on_retain_complete.

    The pre-operation validators are required by the abstract base; they always
    accept so they don't interfere with the operations under test.
    """

    def __init__(self) -> None:
        self.results: list[RetainResult] = []

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def on_retain_complete(self, result: RetainResult) -> None:
        self.results.append(result)


@pytest_asyncio.fixture
async def api_client(memory):
    """Async HTTP client over the FastAPI app backed by the mock-LLM engine."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _unique_bank(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).timestamp()}"


async def _retain(memory, bank_id, content, request_context, document_id):
    await memory.retain_async(
        bank_id=bank_id,
        content=content,
        context="Test context",
        document_id=document_id,
        request_context=request_context,
    )


async def _import(memory, bank_id, archive, request_context, on_conflict="skip"):
    """Submit an import and return its result_metadata counts.

    Import is async; the test fixture uses SyncTaskBackend so the operation runs
    inline and is already completed when submit returns.
    """
    submission = await memory.import_documents_async(bank_id, archive, request_context, on_conflict)
    status = await memory.get_operation_status(
        bank_id, submission["operation_id"], request_context=request_context
    )
    assert status["status"] == "completed", status
    return status["result_metadata"]


@pytest.mark.asyncio
async def test_export_import_roundtrip_without_llm(memory, request_context, monkeypatch):
    """Export from one bank and import into another without re-running the LLM."""
    src = _unique_bank("transfer_src")
    dst = _unique_bank("transfer_dst")
    try:
        await _retain(
            memory,
            src,
            "Alice works at Google. Bob works at Microsoft.",
            request_context,
            document_id="doc-1",
        )

        archive = await memory.export_documents_async(src, request_context)
        assert isinstance(archive, bytes) and len(archive) > 0

        parsed = parse_archive(archive)
        assert parsed.manifest.source_bank_id == src
        assert parsed.manifest.document_count == 1
        assert parsed.manifest.fact_count > 0
        # The archive must not carry embeddings or raw db ids (no "embedding" anywhere,
        # now that the manifest no longer includes embedding model/dimension metadata).
        assert "embedding" not in archive.decode("utf-8", errors="ignore")

        exported_texts = {fact.text for doc in parsed.documents for fact in doc.facts}
        assert exported_texts

        # Importing must never call the LLM fact extractor — make it explode if it does.
        def _boom(*args, **kwargs):
            raise AssertionError("import must not invoke LLM fact extraction")

        monkeypatch.setattr(
            "hindsight_api.engine.retain.fact_extraction.extract_facts_from_contents",
            _boom,
        )

        result = await _import(memory, dst, archive, request_context)
        assert result["documents_imported"] == 1
        assert result["documents_skipped"] == 0
        assert result["facts_imported"] == parsed.manifest.fact_count

        # Facts landed in the destination bank with matching text. Import triggers
        # consolidation, which may synthesize observation units in the destination,
        # so filter those out — the imported facts are world/experience only.
        units = await memory.list_memory_units(dst, request_context=request_context)
        imported_units = [item for item in units["items"] if item["fact_type"] != "observation"]
        assert len(imported_units) == result["facts_imported"]
        assert {item["text"] for item in imported_units} == exported_texts

        # Entities were re-resolved in the destination bank.
        entities = await memory.list_entities(dst, request_context=request_context)
        entity_names = {e["canonical_name"].lower() for e in entities["items"]}
        assert any("alice" in n for n in entity_names)
        assert any("bob" in n for n in entity_names)

        # Embeddings were regenerated locally (not null) in the destination.
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            null_embeddings = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND embedding IS NULL",
                dst,
            )
        assert null_embeddings == 0

        # And the imported memories are retrievable.
        recall = await memory.recall_async(bank_id=dst, query="Where does Alice work?", request_context=request_context)
        assert recall is not None
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


async def _bank_snapshot(memory, bank_id):
    """Count everything persisted for a bank, for round-trip integrity comparison."""
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        docs = await conn.fetch(
            f"SELECT id, COALESCE(length(original_text), 0) AS len FROM {fq_table('documents')} "
            f"WHERE bank_id = $1",
            bank_id,
        )
        chunks = await conn.fetch(
            f"SELECT document_id, chunk_index, length(chunk_text) AS len FROM {fq_table('chunks')} "
            f"WHERE bank_id = $1",
            bank_id,
        )
        ftypes = await conn.fetch(
            f"SELECT fact_type, count(*) AS c FROM {fq_table('memory_units')} WHERE bank_id = $1 GROUP BY fact_type",
            bank_id,
        )
        links = await conn.fetch(
            f"SELECT ml.link_type, count(*) AS c FROM {fq_table('memory_links')} ml "
            f"JOIN {fq_table('memory_units')} m ON m.id = ml.from_unit_id "
            f"WHERE m.bank_id = $1 GROUP BY ml.link_type",
            bank_id,
        )
        unit_entities = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('unit_entities')} ue "
            f"JOIN {fq_table('memory_units')} m ON m.id = ue.unit_id WHERE m.bank_id = $1",
            bank_id,
        )
        entities = await conn.fetchval(f"SELECT count(*) FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
        facts_with_chunk = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND chunk_id IS NOT NULL",
            bank_id,
        )
    by_type = {r["fact_type"]: r["c"] for r in ftypes}
    return {
        "doc_count": len(docs),
        "doc_lens": {r["id"]: r["len"] for r in docs},
        "chunk_count": len(chunks),
        # (document_id, chunk_index) -> chunk_text length: verifies attribution AND size.
        "chunk_map": {(r["document_id"], r["chunk_index"]): r["len"] for r in chunks},
        "world": by_type.get("world", 0),
        "experience": by_type.get("experience", 0),
        "observation": by_type.get("observation", 0),
        "unit_entities": unit_entities,
        "entities": entities,
        "facts_with_chunk": facts_with_chunk,
        "links_by_type": {r["link_type"]: r["c"] for r in links},
        "links_total": sum(r["c"] for r in links),
    }


@pytest.mark.asyncio
async def test_full_roundtrip_integrity(memory, request_context):
    """Full export → import must reproduce every persisted artifact (counts + sizes)."""
    src = _unique_bank("transfer_integ_src")
    dst = _unique_bank("transfer_integ_dst")
    try:
        # A multi-chunk document (content > chunk_size) plus a short one, so chunk
        # numbering and fact→chunk attribution across chunks are exercised.
        long_doc = " ".join(f"Person{i} works at Company{i} in City{i}." for i in range(220))
        await _retain(memory, src, long_doc, request_context, "doc-long")
        await _retain(memory, src, "Carol moved to Berlin in 2024 and joined Acme.", request_context, "doc-short")

        before = await _bank_snapshot(memory, src)
        # Sanity: the fixture actually produced multiple chunks + links + observations.
        assert before["chunk_count"] >= 2
        assert before["links_total"] > 0
        assert before["observation"] > 0

        archive = await memory.export_documents_async(src, request_context, include_observations=True)
        await _import(memory, dst, archive, request_context)
        after = await _bank_snapshot(memory, dst)

        # Documents: same count and same original_text sizes (by id).
        assert after["doc_count"] == before["doc_count"]
        assert after["doc_lens"] == before["doc_lens"]
        # Chunks: same count, and same (document, chunk_index) -> size map. This is
        # the chunk-attribution guarantee.
        assert after["chunk_count"] == before["chunk_count"]
        assert after["chunk_map"] == before["chunk_map"]
        # Facts: same world/experience/observation counts, same chunk linkage count.
        assert after["world"] == before["world"]
        assert after["experience"] == before["experience"]
        assert after["observation"] == before["observation"]
        assert after["facts_with_chunk"] == before["facts_with_chunk"]
        # Entities + entity links re-resolved to the same counts.
        assert after["entities"] == before["entities"]
        assert after["unit_entities"] == before["unit_entities"]
        # Links are regenerated against the target bank; for the same facts/embeddings
        # the deterministic temporal + causal links must match exactly.
        for link_type in ("temporal", "caused_by"):
            assert after["links_by_type"].get(link_type, 0) == before["links_by_type"].get(link_type, 0), (
                link_type,
                before["links_by_type"],
                after["links_by_type"],
            )
        # And links overall must be present (semantic counts can vary slightly with
        # ANN ordering, so we don't assert exact equality on the total).
        assert after["links_total"] > 0
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_export_import_observations(memory, request_context):
    """With include_observations, observations transfer and their sources re-link."""
    src = _unique_bank("transfer_obs_src")
    dst = _unique_bank("transfer_obs_dst")
    try:
        await _retain(memory, src, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
        # Sources must be world/experience facts (not auto-consolidation observations).
        units = await memory.list_memory_units(src, fact_type="world", request_context=request_context)
        source_ids = [uuid.UUID(str(i["id"])) for i in units["items"][:2]]
        assert len(source_ids) == 2

        # Create a real observation over those source facts.
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                await _create_observation_directly(
                    conn, memory, src, source_ids, "Alice and Bob are colleagues."
                )

        # Export WITHOUT observations -> none in the archive (the bank may also
        # contain auto-consolidation observations; the flag is what gates them).
        plain = parse_archive(await memory.export_documents_async(src, request_context))
        assert plain.manifest.observation_count == 0
        assert plain.observations == []

        # Export WITH observations. (The mock LLM's auto-consolidation may have
        # produced extra observations too, so assert on our specific one.)
        archive = await memory.export_documents_async(src, request_context, include_observations=True)
        parsed = parse_archive(archive)
        assert parsed.manifest.observation_count == len(parsed.observations) >= 1
        mine = next((o for o in parsed.observations if o.text == "Alice and Bob are colleagues."), None)
        assert mine is not None
        assert len(mine.sources) == 2  # both sources resolved within the export
        assert "embedding" not in archive.decode("utf-8", errors="ignore")

        # Import into a fresh bank. Every exported observation's sources are in
        # the single exported document, so all import and none are skipped.
        result = await _import(memory, dst, archive, request_context)
        assert result["observations_imported"] == parsed.manifest.observation_count
        assert result["observations_skipped"] == 0

        # Our observation landed with source_memory_ids pointing at dst's facts,
        # and those source facts are marked consolidated.
        async with acquire_with_retry(backend) as conn:
            obs_row = await conn.fetchrow(
                f"SELECT source_memory_ids FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND fact_type = 'observation' AND text = $2",
                dst,
                "Alice and Bob are colleagues.",
            )
            assert obs_row is not None
            dst_sources = list(obs_row["source_memory_ids"] or [])
            assert len(dst_sources) == 2
            consolidated = await conn.fetchval(
                f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND id = ANY($2) AND consolidated_at IS NOT NULL",
                dst,
                dst_sources,
            )
            assert consolidated == 2
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_import_triggers_consolidation(memory, request_context):
    """Importing (without observations) triggers consolidation in the target bank,
    so observations get generated there — same as a normal retain."""
    src = _unique_bank("transfer_consol_src")
    dst = _unique_bank("transfer_consol_dst")
    try:
        await _retain(memory, src, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
        # Export WITHOUT observations: the archive carries only world/experience facts.
        archive = await memory.export_documents_async(src, request_context)
        assert parse_archive(archive).observations == []

        # Import into a fresh bank. The post-import consolidation trigger runs
        # inline (SyncTaskBackend) and the mock LLM produces observations.
        await _import(memory, dst, archive, request_context)

        obs = await memory.list_memory_units(dst, fact_type="observation", request_context=request_context)
        assert obs["total"] > 0, "import should have triggered consolidation to generate observations"
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_import_fires_retain_complete_hook(memory, request_context):
    """Import fires the post-retain extension hook once per imported document,
    mirroring retain — with zero LLM tokens (import runs no extraction)."""
    src = _unique_bank("transfer_hook_src")
    dst = _unique_bank("transfer_hook_dst")
    await _retain(memory, src, "Alice works at Google.", request_context, "doc-1")
    await _retain(memory, src, "Bob works at Microsoft.", request_context, "doc-2")
    archive = await memory.export_documents_async(src, request_context)

    capture = _RetainResultCapture()
    original_validator = memory._operation_validator
    memory._operation_validator = capture
    try:
        result = await _import(memory, dst, archive, request_context)
        assert result["documents_imported"] == 2

        # One hook call per imported document.
        assert len(capture.results) == 2
        by_doc = {r.document_id: r for r in capture.results}
        assert set(by_doc) == {"doc-1", "doc-2"}
        for res in capture.results:
            assert res.bank_id == dst
            assert res.success is True
            # Import runs no LLM extraction: token counts are zero and
            # processed_content_tokens is 0 ("nothing went through extraction").
            assert res.llm_input_tokens == 0
            assert res.llm_output_tokens == 0
            assert res.llm_total_tokens == 0
            assert res.processed_content_tokens == 0
            # unit_ids are reported per content item, with the created facts.
            assert res.unit_ids and res.unit_ids[0]
    finally:
        memory._operation_validator = original_validator
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_import_queues_retain_webhook(memory, request_context):
    """Import queues a retain.completed webhook delivery per document, like retain."""
    src = _unique_bank("transfer_wh_src")
    dst = _unique_bank("transfer_wh_dst")
    webhook_id = uuid.uuid4()
    await _retain(memory, src, "Carol lives in Paris.", request_context, "doc-wh")
    archive = await memory.export_documents_async(src, request_context)

    # The destination bank is created lazily by import; create it now so the
    # webhook row's FK to banks is satisfied, then subscribe it to retain.completed.
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        await conn.execute(
            f"INSERT INTO {fq_table('banks')} (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            dst,
            dst,
        )
        await conn.execute(
            f"INSERT INTO {fq_table('webhooks')} "
            f"(id, bank_id, url, secret, event_types, enabled, created_at, updated_at) "
            f"VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())",
            webhook_id,
            dst,
            "https://example.com/retain-hook",
            ["retain.completed"],
        )

    original_manager = memory._webhook_manager
    memory._webhook_manager = WebhookManager(backend=memory._backend, global_webhooks=[])
    try:
        await _import(memory, dst, archive, request_context)

        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"SELECT task_payload FROM {fq_table('async_operations')} "
                f"WHERE operation_type = 'webhook_delivery' AND bank_id = $1 "
                f"AND task_payload->>'event_type' = 'retain.completed'",
                dst,
            )
        assert len(rows) == 1, "import should queue one retain.completed delivery for the imported document"
        payload = rows[0]["task_payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        inner = json.loads(payload["payload"])
        assert inner.get("data", {}).get("document_id") == "doc-wh"
    finally:
        memory._webhook_manager = original_manager
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_include_observations_requires_whole_bank_export(memory, request_context):
    """include_observations is only valid for a whole-bank export, not a subset."""
    src = _unique_bank("transfer_obs_subset")
    try:
        await _retain(memory, src, "Alice works at Google.", request_context, "doc-1")
        # Subset export (document_ids set) + observations must be rejected.
        with pytest.raises(ValueError, match="whole bank"):
            await memory.export_documents_async(
                src, request_context, ["doc-1"], include_observations=True
            )
        # Whole-bank export with observations is fine; subset without observations is fine.
        await memory.export_documents_async(src, request_context, include_observations=True)
        await memory.export_documents_async(src, request_context, ["doc-1"])
    finally:
        await memory.delete_bank(src, request_context=request_context)


@pytest.mark.asyncio
async def test_import_on_conflict_modes(memory, request_context):
    """skip leaves the document untouched; replace re-imports; new-id duplicates under a fresh id."""
    src = _unique_bank("transfer_conf")
    try:
        await _retain(memory, src, "Carol lives in Paris.", request_context, document_id="doc-x")
        archive = await memory.export_documents_async(src, request_context)

        # Re-importing into the SAME bank with skip is a no-op.
        skipped = await _import(memory, src, archive, request_context, on_conflict="skip")
        assert skipped["documents_imported"] == 0
        assert skipped["documents_skipped"] == 1
        assert skipped["skipped_document_ids"] == ["doc-x"]

        docs_after_skip = await memory.list_documents(src, request_context=request_context)
        assert docs_after_skip["total"] == 1

        # replace re-imports under the same id.
        replaced = await _import(memory, src, archive, request_context, on_conflict="replace")
        assert replaced["documents_imported"] == 1
        assert replaced["documents_skipped"] == 0
        docs_after_replace = await memory.list_documents(src, request_context=request_context)
        assert docs_after_replace["total"] == 1

        # new-id imports a copy under a freshly generated id.
        remapped = await _import(memory, src, archive, request_context, on_conflict="new-id")
        assert remapped["documents_imported"] == 1
        assert "doc-x" in remapped["remapped_document_ids"]
        docs_after_newid = await memory.list_documents(src, request_context=request_context)
        assert docs_after_newid["total"] == 2
    finally:
        await memory.delete_bank(src, request_context=request_context)


@pytest.mark.asyncio
async def test_http_export_import_endpoints(api_client, memory, request_context):
    """Round trip through the HTTP export (GET) and import (POST multipart) endpoints."""
    src = _unique_bank("transfer_http_src")
    dst = _unique_bank("transfer_http_dst")
    try:
        await _retain(memory, src, "Dana lives in Berlin.", request_context, document_id="doc-http")

        export = await api_client.get(f"/v1/default/banks/{src}/document-transfer")
        assert export.status_code == 200
        assert export.headers["content-type"] == "application/zip"
        archive = export.content
        assert len(archive) > 0

        # include_observations + a document_id subset is a 400.
        bad = await api_client.get(
            f"/v1/default/banks/{src}/document-transfer",
            params={"document_id": "meeting-notes", "include_observations": "true"},
        )
        assert bad.status_code == 400

        # Import is async: returns 202 + operation_id (runs inline under the
        # SyncTaskBackend test fixture, so it's completed by the time we poll).
        imported = await api_client.post(
            f"/v1/default/banks/{dst}/document-transfer",
            files={"file": ("transfer.zip", archive, "application/zip")},
            params={"on_conflict": "skip"},
        )
        assert imported.status_code == 202
        operation_id = imported.json()["operation_id"]

        status = await api_client.get(f"/v1/default/banks/{dst}/operations/{operation_id}")
        assert status.status_code == 200
        op = status.json()
        assert op["status"] == "completed"
        assert op["result_metadata"]["documents_imported"] == 1
        assert op["result_metadata"]["facts_imported"] >= 1

        # Exporting a bank that does not exist is a 404.
        missing = await api_client.get("/v1/default/banks/does-not-exist-bank/document-transfer")
        assert missing.status_code == 404
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_endpoints_disabled_by_config(api_client, monkeypatch):
    """When the feature flags are off, the endpoints return 404 and /version reports disabled."""
    from hindsight_api.config import clear_config_cache

    # The static config is a cached singleton; override via env + cache reset.
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_DOCUMENT_EXPORT_API", "false")
    monkeypatch.setenv("HINDSIGHT_API_ENABLE_DOCUMENT_IMPORT_API", "false")
    clear_config_cache()
    try:
        export = await api_client.get("/v1/default/banks/any-bank/document-transfer")
        assert export.status_code == 404
        assert "disabled" in export.json()["detail"].lower()

        imported = await api_client.post(
            "/v1/default/banks/any-bank/document-transfer",
            files={"file": ("x.zip", b"not-a-zip", "application/zip")},
        )
        assert imported.status_code == 404
        assert "disabled" in imported.json()["detail"].lower()

        version = await api_client.get("/version")
        features = version.json()["features"]
        assert features["document_export_api"] is False
        assert features["document_import_api"] is False
    finally:
        # Restore the cache so the reverted env is picked up by later tests.
        clear_config_cache()


@pytest.mark.asyncio
async def test_import_rejects_unsupported_schema_version(memory, request_context):
    """An archive with an unknown schema version is rejected before any writes."""
    manifest = TransferManifest(schema_version=SCHEMA_VERSION + 999, source_bank_id="whatever")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("manifest.json", manifest.model_dump_json())

    with pytest.raises(ValueError, match="schema version"):
        await memory.import_documents_async("any-bank", buffer.getvalue(), request_context)


@pytest.mark.asyncio
async def test_import_rejects_invalid_on_conflict(memory, request_context):
    """An unknown on_conflict mode is rejected with a ValueError."""
    manifest = TransferManifest(source_bank_id="whatever")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("manifest.json", manifest.model_dump_json())

    with pytest.raises(ValueError, match="on_conflict"):
        await import_documents(
            backend=await memory._get_backend(),
            embeddings_model=memory.embeddings,
            entity_resolver=memory.entity_resolver,
            config=None,
            format_date_fn=memory._format_readable_date,
            bank_id="any-bank",
            archive_bytes=buffer.getvalue(),
            on_conflict="bogus",
        )
