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
from hindsight_api.engine.transfer.schema import (
    SCHEMA_VERSION,
    TransferCausalRelation,
    TransferChunk,
    TransferDocument,
    TransferFact,
    TransferManifest,
    TransferObservation,
    TransferObservationSource,
)
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
    status = await memory.get_operation_status(bank_id, submission["operation_id"], request_context=request_context)
    assert status["status"] == "completed", status
    return status["result_metadata"]


@pytest.mark.asyncio
async def test_import_filters_degenerate_fact_without_shifting_archive_ordinals(memory, request_context):
    """A rejected archive fact must not shift chunks, causal links, or observation sources."""
    dst = _unique_bank("transfer_degenerate_alignment")
    document_id = "doc-alignment"
    initial_text = "Alignment test initial event"
    middle_text = "Alignment test surviving middle event"
    later_text = "Alignment test later consequence"
    observation_text = "Alignment test imported observation"
    document = TransferDocument(
        id=document_id,
        original_text="Four extracted facts, one of which is degenerate.",
        chunks=[TransferChunk(chunk_index=index, chunk_text=f"chunk-{index}") for index in range(4)],
        facts=[
            TransferFact(text=initial_text, fact_type="world", chunk_index=0),
            TransferFact(text="...", fact_type="world", chunk_index=1),
            TransferFact(text=middle_text, fact_type="world", chunk_index=2),
            TransferFact(
                text=later_text,
                fact_type="world",
                chunk_index=3,
                causal_relations=[
                    TransferCausalRelation(relation_type="caused_by", target_fact_index=2),
                    TransferCausalRelation(relation_type="causes", target_fact_index=2),
                    TransferCausalRelation(relation_type="prevents", target_fact_index=1),
                ],
            ),
        ],
    )
    observation = TransferObservation(
        text=observation_text,
        sources=[TransferObservationSource(document_id=document_id, fact_index=3)],
    )
    manifest = TransferManifest(
        source_bank_id="source",
        document_count=1,
        fact_count=4,
        observation_count=1,
    )
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("manifest.json", manifest.model_dump_json())
        archive.writestr("documents/000000.json", document.model_dump_json())
        archive.writestr("observations.json", json.dumps([observation.model_dump(mode="json")]))

    try:
        result = await _import(memory, dst, archive_buffer.getvalue(), request_context)
        assert result["facts_imported"] == 3
        assert result["observations_imported"] == 1

        chunks = await memory.list_document_chunks(dst, document_id, limit=10, request_context=request_context)
        assert sorted(chunk["chunk_index"] for chunk in chunks["items"]) == [0, 1, 2, 3]

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            units = await conn.fetch(
                f"SELECT id, text, chunk_id, fact_type, source_memory_ids "
                f"FROM {fq_table('memory_units')} WHERE bank_id = $1",
                dst,
            )
            causal_links = await conn.fetch(
                f"SELECT ml.link_type, source.text AS source_text, target.text AS target_text "
                f"FROM {fq_table('memory_links')} ml "
                f"JOIN {fq_table('memory_units')} source ON source.id = ml.from_unit_id "
                f"JOIN {fq_table('memory_units')} target ON target.id = ml.to_unit_id "
                f"WHERE ml.bank_id = $1 AND ml.link_type = ANY($2)",
                dst,
                ["caused_by", "causes", "prevents"],
            )

        units_by_text = {unit["text"]: unit for unit in units}
        assert "..." not in units_by_text
        assert units_by_text[initial_text]["chunk_id"] == f"{dst}_{document_id}_0"
        assert units_by_text[middle_text]["chunk_id"] == f"{dst}_{document_id}_2"
        assert units_by_text[later_text]["chunk_id"] == f"{dst}_{document_id}_3"
        assert {str(source_id) for source_id in units_by_text[observation_text]["source_memory_ids"]} == {
            str(units_by_text[later_text]["id"])
        }
        assert {(row["link_type"], row["source_text"], row["target_text"]) for row in causal_links} == {
            ("caused_by", later_text, middle_text),
            ("causes", later_text, middle_text),
        }
    finally:
        await memory.delete_bank(dst, request_context=request_context)


def test_export_bank_covers_schema():
    """Every bank-scoped table must be classified by export_bank — logical, carried,
    history, or explicitly skipped — so a future migration can't silently drop one."""
    from hindsight_api.admin.cli import BACKUP_TABLES
    from hindsight_api.engine.transfer.export import _BANK_ROW_TABLES, _REPLAYED_TABLES, _SKIP_TABLES
    from hindsight_api.engine.transfer.schema import CARRIED_HISTORY_TABLES, HISTORY_TABLES

    buckets = [
        set(_REPLAYED_TABLES),
        set(_BANK_ROW_TABLES),
        set(CARRIED_HISTORY_TABLES),
        set(HISTORY_TABLES),
        set(_SKIP_TABLES),
    ]
    classified = set().union(*buckets)
    assert classified == set(BACKUP_TABLES), (
        f"export-bank classification drifted from BACKUP_TABLES: "
        f"missing={set(BACKUP_TABLES) - classified}, extra={classified - set(BACKUP_TABLES)}"
    )
    # No table may appear in two buckets.
    assert sum(len(b) for b in buckets) == len(classified), "a table is classified in more than one bucket"


@pytest.mark.asyncio
async def test_export_bank_contents(memory, request_context):
    """export_bank produces a whole-bank archive: docs + bank config + webhooks,
    no embeddings, with history gated behind include_history."""
    from hindsight_api.engine.transfer import export_bank

    bank = _unique_bank("export_bank")
    webhook_id = uuid.uuid4()
    try:
        await _retain(memory, bank, "Carol lives in Paris.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('webhooks')} "
                f"(id, bank_id, url, secret, event_types, enabled, created_at, updated_at) "
                f"VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())",
                webhook_id,
                bank,
                "https://example.com/hook",
                ["retain.completed"],
            )

        # Without history.
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank, include_history=False)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
            bank_rows = json.loads(zf.read("banks.json"))
            webhooks = json.loads(zf.read("webhooks.json"))

        assert manifest.archive_type == "bank"
        assert manifest.document_count == 1
        assert manifest.webhook_count == 1
        assert "mental_models.json" in names and "directives.json" in names
        assert "mental_model_history.json" in names
        assert any(d.endswith(".json") and d.startswith("documents/") for d in names)
        # No history files unless requested.
        assert not any(n.startswith("history/") for n in names)
        # The bank row and webhook are carried.
        assert [r["bank_id"] for r in bank_rows] == [bank]
        assert webhooks[0]["bank_id"] == bank and webhooks[0]["url"] == "https://example.com/hook"
        # No embeddings anywhere — the target instance regenerates them.
        assert "embedding" not in archive.decode("utf-8", errors="ignore")

        # With history.
        async with acquire_with_retry(backend) as conn:
            archive_h = await export_bank(conn, bank, include_history=True)
        with zipfile.ZipFile(io.BytesIO(archive_h)) as zf:
            names_h = set(zf.namelist())
            manifest_h = TransferManifest.model_validate_json(zf.read("manifest.json"))
        assert manifest_h.includes_history is True
        assert "history/audit_log.json" in names_h and "history/llm_requests.json" in names_h
    finally:
        await memory.delete_bank(bank, request_context=request_context)


def _as_json(value):
    """Normalize a jsonb column value (str or already-decoded) to a Python object."""
    return json.loads(value) if isinstance(value, str) else value


async def _bank_content_snapshot(memory, bank_id):
    """Capture the meaningful (non-embedding, non-volatile) content of a bank for
    exact round-trip comparison across export → import."""
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        bank = await conn.fetchrow(
            f"SELECT name, disposition, mission, config FROM {fq_table('banks')} WHERE bank_id = $1", bank_id
        )
        docs = await conn.fetch(
            f"SELECT id, original_text, tags FROM {fq_table('documents')} WHERE bank_id = $1", bank_id
        )
        facts = await conn.fetch(
            f"SELECT text, fact_type, context FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type != 'observation'",
            bank_id,
        )
        obs = await conn.fetch(
            f"SELECT text, proof_count FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )
        ents = await conn.fetch(f"SELECT canonical_name FROM {fq_table('entities')} WHERE bank_id = $1", bank_id)
        links = await conn.fetch(
            f"SELECT link_type, count(*) AS c FROM {fq_table('memory_links')} WHERE bank_id = $1 GROUP BY link_type",
            bank_id,
        )
        hooks = await conn.fetch(
            f"SELECT url, event_types, enabled FROM {fq_table('webhooks')} WHERE bank_id = $1", bank_id
        )
        dirs = await conn.fetch(
            f"SELECT name, content, priority, is_active FROM {fq_table('directives')} WHERE bank_id = $1", bank_id
        )
        mms = await conn.fetch(
            f"SELECT subtype, name, description, tags FROM {fq_table('mental_models')} WHERE bank_id = $1", bank_id
        )
        null_emb = await conn.fetchval(
            f"SELECT count(*) FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type != 'observation' AND embedding IS NULL",
            bank_id,
        )
    return {
        "bank": (bank["name"], _as_json(bank["disposition"]), bank["mission"], _as_json(bank["config"])),
        "documents": sorted((d["id"], d["original_text"], tuple(sorted(d["tags"] or []))) for d in docs),
        "facts": sorted((f["text"], f["fact_type"], f["context"]) for f in facts),
        "observations": sorted((o["text"], o["proof_count"]) for o in obs),
        "entities": sorted(e["canonical_name"].lower() for e in ents),
        "links": {row["link_type"]: row["c"] for row in links},
        "webhooks": sorted((h["url"], tuple(h["event_types"] or []), h["enabled"]) for h in hooks),
        "directives": sorted((d["name"], d["content"], d["priority"], d["is_active"]) for d in dirs),
        "mental_models": sorted(
            (m["subtype"], m["name"], m["description"], tuple(sorted(m["tags"] or []))) for m in mms
        ),
        "null_embeddings": null_emb,
    }


@pytest.mark.asyncio
async def test_bank_export_import_exact_roundtrip(memory, request_context):
    """A whole-bank archive restores EXACT bank content (config, docs, facts,
    observations, entities, links, webhooks, directives, mental models) with facts
    re-embedded. Uses export → delete → import so ids round-trip without collisions
    (mirroring a fresh target instance)."""
    bank = _unique_bank("bank_exact")
    try:
        await _retain(memory, bank, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
        await _retain(memory, bank, "Carol lives in Paris.", request_context, "doc-2")

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('banks')} SET name = $2, disposition = $3::jsonb, "
                f"mission = $4, config = $5::jsonb WHERE bank_id = $1",
                bank,
                "My Bank",
                json.dumps({"skepticism": 5, "literalism": 2, "empathy": 4}),
                "Be terse and precise.",
                json.dumps({"reflect_mission": "be terse"}),
            )
            await conn.execute(
                f"INSERT INTO {fq_table('webhooks')} "
                f"(id, bank_id, url, secret, event_types, enabled, created_at, updated_at) "
                f"VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())",
                uuid.uuid4(),
                bank,
                "https://example.com/hook",
                ["retain.completed", "consolidation.completed"],
            )
            await conn.execute(
                f"INSERT INTO {fq_table('directives')} "
                f"(id, bank_id, name, content, priority, is_active, tags, created_at, updated_at) "
                f"VALUES ($1, $2, $3, $4, $5, true, $6, NOW(), NOW())",
                uuid.uuid4(),
                bank,
                "tone",
                "Always be concise.",
                7,
                ["style"],
            )
        await memory.create_mental_model(
            bank,
            name="Work model",
            source_query="where do people work",
            content="User tracks where people work.",
            mental_model_id="mm-1",
            tags=["people"],
            request_context=request_context,
        )

        before = await _bank_content_snapshot(memory, bank)
        # Sanity: the source genuinely has rich content in every section we carry.
        assert before["facts"] and before["entities"] and before["links"]
        assert before["webhooks"] and before["directives"] and before["mental_models"]
        assert before["bank"][0] == "My Bank"

        from hindsight_api.engine.transfer import export_bank

        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        # Delete then restore into the same id — exact round-trip, no PK collisions.
        await memory.delete_bank(bank, request_context=request_context)
        result = await memory.import_bank_async(archive, request_context)
        assert result.bank_id == bank
        assert result.webhooks_imported == 1
        assert result.directives_imported == 1
        assert result.mental_models_imported == 1

        after = await _bank_content_snapshot(memory, bank)
        # Semantic links are an ANN-approximate retrieval index regenerated from the
        # (re-embedded) facts; their count depends on whether ANN runs incrementally
        # per document (import) or as a final whole-bank pass (original retain), so
        # compare them loosely. Everything else — source data and deterministic
        # temporal links — must match exactly.
        after_semantic = after["links"].pop("semantic", 0)
        before["links"].pop("semantic", None)
        assert after == before
        assert after_semantic > 0, "semantic links should be regenerated on import"
        # Facts were re-embedded on import (no NULL vectors).
        assert after["null_embeddings"] == 0
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_roundtrip_carries_mental_model_history(memory, request_context):
    """Mental-model refresh history survives export/import. Mental models keep a
    stable (id, bank_id), so the dedicated mental_model_history rows are carried
    (the surrogate id is dropped on export; the target reassigns it)."""
    bank = _unique_bank("bank_mm_hist")
    try:
        await memory.get_bank_profile(bank, request_context=request_context)
        await memory.create_mental_model(
            bank,
            name="Work model",
            source_query="where do people work",
            content="v1",
            mental_model_id="mm-1",
            request_context=request_context,
        )
        await memory.update_mental_model(bank, mental_model_id="mm-1", content="v2", request_context=request_context)
        await memory.update_mental_model(bank, mental_model_id="mm-1", content="v3", request_context=request_context)
        # Two refreshes → two snapshots (previous content v1 then v2), newest-first.
        before = await memory.get_mental_model_history(bank, "mm-1", request_context=request_context)
        assert [h["previous_content"] for h in before] == ["v2", "v1"]

        from hindsight_api.engine.transfer import export_bank

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        await memory.delete_bank(bank, request_context=request_context)
        result = await memory.import_bank_async(archive, request_context)
        assert result.mental_model_history_imported == 2

        after = await memory.get_mental_model_history(bank, "mm-1", request_context=request_context)
        assert [h["previous_content"] for h in after] == ["v2", "v1"]
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_import_bank_rejects_documents_archive(memory, request_context):
    """A documents-only archive must be rejected by the bank importer."""
    bank = _unique_bank("bank_reject")
    try:
        await _retain(memory, bank, "Alice works at Google.", request_context, "doc-1")
        docs_archive = await memory.export_documents_async(bank, request_context)
        with pytest.raises(ValueError, match="whole-bank archive"):
            await memory.import_bank_async(docs_archive, request_context)
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_import_bank_refuses_existing_bank(memory, request_context):
    """import-bank restores a whole bank, not a merge — it must refuse an existing target."""
    from hindsight_api.engine.transfer import export_bank

    bank = _unique_bank("bank_exists")
    try:
        await _retain(memory, bank, "Alice works at Google.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        # The source bank still exists — importing the archive back must refuse
        # (restoring into the same id after delete is covered by the exact round-trip test).
        with pytest.raises(ValueError, match="already exists"):
            await memory.import_bank_async(archive, request_context)
    finally:
        await memory.delete_bank(bank, request_context=request_context)


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
            f"SELECT id, COALESCE(length(original_text), 0) AS len FROM {fq_table('documents')} WHERE bank_id = $1",
            bank_id,
        )
        chunks = await conn.fetch(
            f"SELECT document_id, chunk_index, length(chunk_text) AS len FROM {fq_table('chunks')} WHERE bank_id = $1",
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
async def test_transfer_preserves_legacy_causal_links(memory, request_context):
    """Legacy causal edges survive export/import without becoming retain inputs."""
    src = _unique_bank("transfer_legacy_causal_src")
    dst = _unique_bank("transfer_legacy_causal_dst")
    legacy_types = ("causes", "enables", "prevents")
    try:
        await _retain(
            memory,
            src,
            "Alice completed the design. Bob began implementation after the design.",
            request_context,
            "doc-legacy-causal",
        )
        units = await memory.list_memory_units(src, fact_type="world", request_context=request_context)
        assert len(units["items"]) >= 2
        from_unit_id = uuid.UUID(str(units["items"][0]["id"]))
        to_unit_id = uuid.UUID(str(units["items"][1]["id"]))
        from_text = units["items"][0]["text"]
        to_text = units["items"][1]["text"]

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.executemany(
                f"INSERT INTO {fq_table('memory_links')} "
                "(from_unit_id, to_unit_id, link_type, entity_id, bank_id, weight) "
                "VALUES ($1, $2, $3, NULL, $4, 1.0)",
                [(from_unit_id, to_unit_id, link_type, src) for link_type in legacy_types],
            )

        archive = await memory.export_documents_async(src, request_context)
        await _import(memory, dst, archive, request_context)

        async with acquire_with_retry(backend) as conn:
            imported_types = await conn.fetch(
                f"SELECT ml.link_type, source.text AS source_text, target.text AS target_text "
                f"FROM {fq_table('memory_links')} ml "
                f"JOIN {fq_table('memory_units')} source ON source.id = ml.from_unit_id "
                f"JOIN {fq_table('memory_units')} target ON target.id = ml.to_unit_id "
                "WHERE ml.bank_id = $1 AND ml.link_type = ANY($2)",
                dst,
                list(legacy_types),
            )
        assert {(row["link_type"], row["source_text"], row["target_text"]) for row in imported_types} == {
            (link_type, from_text, to_text) for link_type in legacy_types
        }
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
                await _create_observation_directly(conn, memory, src, source_ids, "Alice and Bob are colleagues.")

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
            await memory.export_documents_async(src, request_context, ["doc-1"], include_observations=True)
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
