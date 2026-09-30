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
from typing import Any
from urllib.parse import quote

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app
from hindsight_api.engine.chunk_ids import build_chunk_id
from hindsight_api.engine.consolidation.consolidator import (
    _apply_create_observation,
    _embed_observation_text,
)
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.schema import fq_table
from hindsight_api.engine.storage import bank_storage_prefix
from hindsight_api.engine.transfer import import_documents
from hindsight_api.engine.transfer.importer import _EMBED_BATCH_SIZE, _embed_in_batches, parse_archive
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


async def _seed_observation(*, pool, memory, bank_id, source_memory_ids, observation_text):
    """Insert one observation the way consolidation does: embed off-connection, write in a txn.

    The consolidator itself only exposes the two halves — it batches every write from one LLM
    response into a single transaction (#3876) — so this seeds a fixture observation with the
    same pair of calls.
    """
    embedding_str = await _embed_observation_text(memory, observation_text)
    async with acquire_with_retry(pool) as conn:
        async with conn.transaction():
            return await _apply_create_observation(
                conn=conn,
                memory_engine=memory,
                bank_id=bank_id,
                source_memory_ids=source_memory_ids,
                observation_text=observation_text,
                embedding_str=embedding_str,
            )


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
    """Retain one document and return the ids of the facts it created.

    Returned rather than left to a follow-up `list_memory_units` call: retain
    already knows exactly which facts it wrote, whereas the list query answers a
    different question ("what is in the bank now") that also reflects
    auto-consolidation and anything else touching the bank concurrently.
    """
    return await memory.retain_async(
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


async def _export_async(memory, bank_id, request_context, **kwargs):
    """Submit an async export and return (result_metadata, archive_bytes).

    Export is async; the SyncTaskBackend fixture runs it inline, so the operation
    is completed (with the archive stashed in file storage) when submit returns.
    """
    submission = await memory.submit_export_documents_async(bank_id, request_context, **kwargs)
    status = await memory.get_operation_status(bank_id, submission["operation_id"], request_context=request_context)
    assert status["status"] == "completed", status
    meta = status["result_metadata"]
    archive = await memory._file_storage.retrieve(meta["storage_key"])
    return meta, archive


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
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
        assert units_by_text[initial_text]["chunk_id"] == build_chunk_id(dst, document_id, 0)
        assert units_by_text[middle_text]["chunk_id"] == build_chunk_id(dst, document_id, 2)
        assert units_by_text[later_text]["chunk_id"] == build_chunk_id(dst, document_id, 3)
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
    from hindsight_api.engine.transfer.export import (
        _BANK_ROW_TABLES,
        _DATA_ROW_TABLES,
        _REPLAYED_TABLES,
        _SKIP_TABLES,
    )
    from hindsight_api.engine.transfer.schema import CARRIED_HISTORY_TABLES, HISTORY_TABLES, KNOWLEDGE_TABLES

    buckets = [
        set(_REPLAYED_TABLES),
        set(_BANK_ROW_TABLES),
        set(_DATA_ROW_TABLES),
        set(CARRIED_HISTORY_TABLES),
        set(KNOWLEDGE_TABLES),
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


def test_remap_mental_model_evidence_updates_current_and_history():
    """Transfer remapping changes only known memory-unit ids in both payloads.

    The two rows carry the two real shapes. A ``mental_models`` row has its own
    ``reflect_response`` column; a ``mental_model_history`` row stores the whole
    snapshot in a JSONB ``content`` blob, so its evidence is nested one level
    down and there is no top-level ``previous_reflect_response`` to find.
    """
    from hindsight_api.engine.transfer.importer import _remap_mental_model_evidence

    rows = [
        {
            "reflect_response": {"based_on": {"world": [{"id": "old", "text": "fact"}]}},
        },
        {
            "content": json.dumps(
                {
                    "previous_content": "an earlier draft",
                    "previous_reflect_response": {"based_on": {"observation": [{"id": "obs-old"}]}},
                }
            ),
        },
    ]

    _remap_mental_model_evidence(rows, {"old": "new", "obs-old": "obs-new"})

    assert rows[0]["reflect_response"]["based_on"]["world"][0]["id"] == "new"
    history_content = rows[1]["content"]
    assert history_content["previous_reflect_response"]["based_on"]["observation"][0]["id"] == "obs-new"
    # The rest of the snapshot is copied through untouched.
    assert history_content["previous_content"] == "an earlier draft"


def test_topological_page_order_is_parent_first():
    """Nodes always sort so a parent precedes its children (self-FK safe)."""
    from hindsight_api.engine.transfer.importer import _topological_page_order
    from hindsight_api.engine.transfer.schema import TransferKnowledgePage

    def _page(pid, parent):
        kind = "page" if pid.startswith("p") else "folder"
        return TransferKnowledgePage(id=pid, parent_id=parent, kind=kind, name=pid)

    # Deliberately shuffled: child before parent, grandchild before both.
    pages = [_page("pC", "fB"), _page("fB", "fA"), _page("fA", None), _page("pRoot", None)]
    ordered = [p.id for p in _topological_page_order(pages)]
    assert ordered.index("fA") < ordered.index("fB") < ordered.index("pC")
    assert ordered.index("fA") < ordered.index("pC")
    assert set(ordered) == {"pC", "fB", "fA", "pRoot"}


def test_topological_page_order_tolerates_cycles_and_dangling_parents():
    """A cycle or missing parent (only possible in a corrupt export) is emitted
    rather than dropped, so the DB FK — not a silent loss — surfaces it."""
    from hindsight_api.engine.transfer.importer import _topological_page_order
    from hindsight_api.engine.transfer.schema import TransferKnowledgePage

    cycle = [
        TransferKnowledgePage(id="a", parent_id="b", kind="folder", name="a"),
        TransferKnowledgePage(id="b", parent_id="a", kind="folder", name="b"),
    ]
    assert {p.id for p in _topological_page_order(cycle)} == {"a", "b"}
    dangling = [TransferKnowledgePage(id="x", parent_id="missing", kind="page", name="x")]
    assert [p.id for p in _topological_page_order(dangling)] == ["x"]


def test_export_jsonb_coercion_preserves_decoded_scalar_string():
    """Admin connections decode JSONB before the transfer exporter sees it."""
    from hindsight_api.engine.transfer.export import _as_jsonb

    assert _as_jsonb("combined") == "combined"
    assert _as_jsonb('"combined"') == "combined"
    assert _as_jsonb('{"scope": "combined"}') == {"scope": "combined"}


def test_legacy_bank_archive_defaults_to_decoded_json_rows():
    """Released v1 bank archives came from the codec-enabled admin CLI."""
    from hindsight_api.engine.transfer.importer import _resolve_bank_rows_json_encoding

    manifest = TransferManifest(source_bank_id="legacy", archive_type="bank")

    assert manifest.bank_rows_json_encoding is None
    assert _resolve_bank_rows_json_encoding(manifest) == "decoded"


@pytest.mark.asyncio
async def test_restore_rows_normalizes_jsonb_strings(memory):
    """JSONB restore follows archive provenance instead of guessing from strings."""
    from hindsight_api.engine.transfer.importer import _restore_rows

    decoded_request_id = uuid.uuid4()
    serialized_request_id = uuid.uuid4()
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        try:
            await _restore_rows(
                conn,
                "llm_requests",
                [
                    {
                        "id": str(decoded_request_id),
                        "status": "completed",
                        "input": "I am an already-decoded scalar",
                        "output": '{"answer":"JSON-looking decoded scalar"}',
                        "llm_info": {"shape": "decoded-object"},
                    }
                ],
                bank_rows_json_encoding="decoded",
            )
            await _restore_rows(
                conn,
                "llm_requests",
                [
                    {
                        "id": str(serialized_request_id),
                        "status": "completed",
                        "input": json.dumps("serialized scalar"),
                        "output": json.dumps({"answer": "serialized object"}),
                    }
                ],
                bank_rows_json_encoding="serialized",
            )
            decoded_row = await conn.fetchrow(
                f"SELECT input::text, output::text, llm_info::text FROM {fq_table('llm_requests')} WHERE id = $1",
                decoded_request_id,
            )
            serialized_row = await conn.fetchrow(
                f"SELECT input::text, output::text FROM {fq_table('llm_requests')} WHERE id = $1",
                serialized_request_id,
            )
            assert decoded_row is not None
            assert json.loads(decoded_row["input"]) == "I am an already-decoded scalar"
            assert json.loads(decoded_row["output"]) == '{"answer":"JSON-looking decoded scalar"}'
            assert json.loads(decoded_row["llm_info"]) == {"shape": "decoded-object"}
            assert serialized_row is not None
            assert json.loads(serialized_row["input"]) == "serialized scalar"
            assert json.loads(serialized_row["output"]) == {"answer": "serialized object"}
        finally:
            await conn.execute(
                f"DELETE FROM {fq_table('llm_requests')} WHERE id = ANY($1)",
                [decoded_request_id, serialized_request_id],
            )


@pytest.mark.asyncio
async def test_export_bank_contents(memory, request_context):
    """export_bank produces a whole-bank archive: docs + bank config + webhooks,
    no embeddings, with history gated behind scope.history."""
    from hindsight_api.engine.transfer import TransferScope, export_bank

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
            archive = await export_bank(conn, bank, scope=TransferScope(history=False))
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
            bank_rows = json.loads(zf.read("banks.json"))
            webhooks = json.loads(zf.read("webhooks.json"))

        assert manifest.archive_type == "bank"
        assert manifest.bank_rows_json_encoding == "serialized"
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
            archive_h = await export_bank(conn, bank, scope=TransferScope(history=True))
        with zipfile.ZipFile(io.BytesIO(archive_h)) as zf:
            names_h = set(zf.namelist())
            manifest_h = TransferManifest.model_validate_json(zf.read("manifest.json"))
        assert manifest_h.includes_history is True
        assert "history/audit_log.json" in names_h and "history/llm_requests.json" in names_h
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_export_tolerates_legacy_null_and_numeric_fact_metadata(memory, request_context):
    """A bank holding legacy metadata must still be exportable (issue #3209).

    Rows written before retain normalized its input can hold a JSON null or a
    raw integer in memory_units.metadata. TransferFact.metadata is dict[str, str],
    so exporting such a bank used to fail validation — locking an operator out of
    the one operation (backup / move) that gets them off the bad data. Export
    applies the same read contract as recall: nulls dropped, the rest stringified.
    """
    bank = _unique_bank("export_legacy_metadata")
    try:
        await _retain(memory, bank, "Carol lives in Paris.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            updated = await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET metadata = $2::jsonb WHERE bank_id = $1",
                bank,
                json.dumps({"ocr_engine": None, "original_id": 348}),
            )
        assert updated != "UPDATE 0"

        parsed = parse_archive(await memory.export_documents_async(bank, request_context))
        exported = [fact.metadata for doc in parsed.documents for fact in doc.facts]
        assert exported
        assert all(metadata == {"original_id": "348"} for metadata in exported)
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
            f"SELECT id, original_text, tags, created_at FROM {fq_table('documents')} WHERE bank_id = $1", bank_id
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
        "documents": sorted(
            (d["id"], d["original_text"], tuple(sorted(d["tags"] or [])), d["created_at"]) for d in docs
        ),
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


async def _fact_lifecycle(memory, bank_id):
    """Sorted (text, created_at, consolidated_at, consolidation_failed_at) for
    every world/experience fact — the exact per-fact consolidation lifecycle a
    whole-bank transfer must preserve."""
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        rows = await conn.fetch(
            f"SELECT text, created_at, consolidated_at, consolidation_failed_at "
            f"FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type IN ('world', 'experience')",
            bank_id,
        )
    return sorted((r["text"], r["created_at"], r["consolidated_at"], r["consolidation_failed_at"]) for r in rows)


async def _eligible_fact_count(memory, bank_id):
    """Facts the maintenance reconciler would treat as unconsolidated backlog —
    the exact predicate of ``banks_needing_consolidation()``."""
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type IN ('world', 'experience') "
            f"AND consolidated_at IS NULL AND consolidation_failed_at IS NULL",
            bank_id,
        )


async def _observation_count(memory, bank_id):
    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_bank_import_preserves_consolidation_lifecycle(memory, request_context):
    """Whole-bank import restores each fact's consolidation lifecycle verbatim, so
    previously-consolidated and previously-failed facts are never re-consolidated
    and the reconciler sees no phantom backlog. Regression for #2965.

    Crucially the source has consolidated facts that do NOT back any surviving
    observation, plus a ``consolidation_failed_at`` fact — state the old
    observation-lineage reconstruction could not recover, so those facts became
    re-eligible and the target re-derived observations."""
    bank = _unique_bank("bank_lifecycle")
    try:
        await _retain(
            memory,
            bank,
            "Alice works at Google. Bob works at Microsoft. Carol lives in Paris.",
            request_context,
            "doc-1",
        )
        backend = await memory._get_backend()

        # Deterministic baseline: drop any auto-consolidation observations so the
        # only observation is the one created explicitly below.
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"DELETE FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'observation'",
                bank,
            )

        async with acquire_with_retry(backend) as conn:
            wf_ids = [
                r["id"]
                for r in await conn.fetch(
                    f"SELECT id FROM {fq_table('memory_units')} "
                    f"WHERE bank_id = $1 AND fact_type IN ('world', 'experience') ORDER BY created_at, id",
                    bank,
                )
            ]
        assert len(wf_ids) >= 3, "need enough facts to exercise the lineage gap"

        # One surviving observation over the first two facts.
        obs_source_ids = [uuid.UUID(str(i)) for i in wf_ids[:2]]
        await _seed_observation(
            pool=backend,
            memory=memory,
            bank_id=bank,
            source_memory_ids=obs_source_ids,
            observation_text="Alice and Bob are colleagues.",
        )

        # Fully-processed source (zero eligible): every fact is consolidated except
        # the last — deliberately NOT an observation source — which records a
        # consolidation failure. Most consolidated facts do not back the
        # observation, exactly the lineage gap the fix must preserve.
        consolidated_ts = datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        failed_ts = datetime(2020, 6, 7, 8, 9, 10, tzinfo=timezone.utc)
        failed_fact_id = wf_ids[-1]
        assert uuid.UUID(str(failed_fact_id)) not in obs_source_ids
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} "
                f"SET consolidated_at = $2, consolidation_failed_at = NULL "
                f"WHERE bank_id = $1 AND fact_type IN ('world', 'experience') AND id != $3",
                bank,
                consolidated_ts,
                failed_fact_id,
            )
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} "
                f"SET consolidated_at = NULL, consolidation_failed_at = $2 "
                f"WHERE bank_id = $1 AND id = $3",
                bank,
                failed_ts,
                failed_fact_id,
            )

        source_lifecycle = await _fact_lifecycle(memory, bank)
        source_obs_count = await _observation_count(memory, bank)
        assert source_obs_count == 1
        assert await _eligible_fact_count(memory, bank) == 0

        from hindsight_api.engine.transfer import export_bank

        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        # Delete then restore into the same id — exact round-trip.
        await memory.delete_bank(bank, request_context=request_context)
        await memory.import_bank_async(archive, request_context)

        # Lifecycle preserved verbatim: consolidated stays consolidated (same
        # timestamp — not now()), the failed fact keeps consolidation_failed_at.
        assert await _fact_lifecycle(memory, bank) == source_lifecycle
        # No phantom backlog for the reconciler, observation not re-derived.
        assert await _eligible_fact_count(memory, bank) == 0
        assert await _observation_count(memory, bank) == source_obs_count
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
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
@pytest.mark.memory_backend_incompatible
async def test_bank_roundtrip_remaps_mental_model_based_on_ids(memory, request_context):
    """Whole-bank restore rewrites current and historical evidence ids.

    Mental-model rows keep their ids, but replayed facts receive new ids. The
    persisted reflect response must follow those new ids so provenance and
    retraction checks continue to work after migration.
    """
    bank = _unique_bank("bank_mm_based_on")
    try:
        await _retain(memory, bank, "Alice works at Google.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            fact_id = await conn.fetchval(
                f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'world' LIMIT 1", bank
            )
        await memory.create_mental_model(
            bank,
            name="Work model",
            source_query="where does Alice work",
            content="Alice works at Google.",
            mental_model_id="mm-based-on",
            request_context=request_context,
        )
        based_on = {"world": [{"id": str(fact_id), "text": "Alice works at Google."}]}
        # Force persisted evidence because the public API generates this field
        # during refresh and cannot create a deterministic source-id fixture.
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('mental_models')} SET reflect_response = $3::jsonb WHERE bank_id = $1 AND id = $2",
                bank,
                "mm-based-on",
                json.dumps({"text": "Alice works at Google.", "based_on": based_on}),
            )
        await memory.update_mental_model(
            bank,
            mental_model_id="mm-based-on",
            content="Alice works at Google, in California.",
            request_context=request_context,
        )

        from hindsight_api.engine.transfer import export_bank

        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        await memory.delete_bank(bank, request_context=request_context)
        await memory.import_bank_async(archive, request_context)

        restored = await memory.get_mental_model(bank, "mm-based-on", detail="full", request_context=request_context)
        restored_ids = {
            fact["id"] for fact in (restored["reflect_response"] or {}).get("based_on", {}).get("world", [])
        }
        async with acquire_with_retry(backend) as conn:
            live_ids = {
                str(row["id"])
                for row in await conn.fetch(
                    f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND fact_type = 'world'", bank
                )
            }
        assert restored_ids <= live_ids
        assert str(fact_id) not in restored_ids

        history = await memory.get_mental_model_history(bank, "mm-based-on", request_context=request_context)
        historical_ids = {
            fact["id"] for fact in (history[0]["previous_reflect_response"] or {}).get("based_on", {}).get("world", [])
        }
        assert historical_ids <= live_ids
        assert str(fact_id) not in historical_ids
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_import_into_new_id_on_same_instance(memory, request_context):
    """Export a bank and import it RIGHT BACK into a new id on the same instance
    (source bank left in place). The banks row carries a globally-unique
    ``internal_id``; if that were kept, the copy's banks INSERT would collide with
    the still-present source and ``ON CONFLICT DO NOTHING`` would skip the parent
    row, so the mental_models insert would trip fk_mental_models_bank_id. Import
    must mint a fresh internal_id so the copy lands cleanly. Regression for #3270."""
    source = _unique_bank("bank_src")
    target = _unique_bank("bank_copy")
    try:
        await _retain(memory, source, "Alice works at Google.", request_context, "doc-1")
        await memory.create_mental_model(
            source,
            name="Work model",
            source_query="where do people work",
            content="User tracks where people work.",
            mental_model_id="mm-1",
            request_context=request_context,
        )

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            source_internal_id = await conn.fetchval(
                f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", source
            )

        from hindsight_api.engine.transfer import export_bank

        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, source)
        # Source bank is left in place — this is the same-instance "make a copy" flow.
        result = await memory.import_bank_async(archive, request_context, target_bank_id=target)
        assert result.bank_id == target
        assert result.mental_models_imported == 1

        async with acquire_with_retry(backend) as conn:
            target_internal_id = await conn.fetchval(
                f"SELECT internal_id FROM {fq_table('banks')} WHERE bank_id = $1", target
            )
        # The copy exists (parent row landed) and got a fresh, non-colliding id.
        assert target_internal_id is not None
        assert target_internal_id != source_internal_id
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_roundtrip_carries_mental_model_history(memory, request_context):
    """Mental-model refresh history survives export/import. Mental models keep a
    stable (id, bank_id), so the dedicated mental_model_history rows are carried
    (the surrogate id is dropped on export; the target reassigns it)."""
    bank = _unique_bank("bank_mm_hist")
    try:
        await memory.ensure_bank_profile(bank, request_context=request_context)
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
        assert [h["previous_content"] for h in before] == ["v2\n", "v1\n"]

        from hindsight_api.engine.transfer import export_bank

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        await memory.delete_bank(bank, request_context=request_context)
        result = await memory.import_bank_async(archive, request_context)
        assert result.mental_model_history_imported == 2

        after = await memory.get_mental_model_history(bank, "mm-1", request_context=request_context)
        assert [h["previous_content"] for h in after] == ["v2\n", "v1\n"]
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_bank_roundtrip_carries_knowledge_pages(memory, request_context):
    """A whole-bank archive restores the Knowledge Pages tree — nested folders +
    pages, parent_id / mental_model_id / managed / sort_order preserved — and
    regenerates each backing mental model's embedding + lexical state on the
    target, so pages stay searchable after import (#3308, #3323)."""
    bank = _unique_bank("bank_kb")
    try:
        await memory.ensure_bank_profile(bank, request_context=request_context)
        root = await memory.create_knowledge_folder(bank, "Runbooks", managed=True, request_context=request_context)
        sub = await memory.create_knowledge_folder(
            bank, "Billing", parent_id=root["id"], request_context=request_context
        )
        page = await memory.create_knowledge_page(
            bank,
            name="Net-30 policy",
            source_query="what is our billing policy",
            content="Invoices are due Net-30. Late payments accrue interest.",
            parent_id=sub["id"],
            request_context=request_context,
        )
        # A root-level page (NULL parent) exercises the non-nested path too.
        await memory.create_knowledge_page(
            bank,
            name="Overview",
            source_query="overview",
            content="Company overview and mission statement.",
            request_context=request_context,
        )

        def _tree(nodes):
            return sorted(
                (n["id"], n["kind"], n["parent_id"], n["mental_model_id"], n["managed"], n["name"]) for n in nodes
            )

        before = _tree(await memory.list_knowledge_nodes(bank, request_context=request_context))
        before_search = await memory.search_knowledge_pages(
            bank, "net-30 billing", limit=5, request_context=request_context
        )
        assert any(r["id"] == page["id"] for r in before_search), "page should be searchable before export"

        from hindsight_api.engine.transfer import export_bank

        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        # Delete then restore into the same id — exact round-trip, no PK collisions.
        await memory.delete_bank(bank, request_context=request_context)
        result = await memory.import_bank_async(archive, request_context)
        assert result.knowledge_pages_imported == 4  # 2 folders + 2 pages

        # Tree restored exactly: ids, parents, backing mental models, managed flag.
        after = _tree(await memory.list_knowledge_nodes(bank, request_context=request_context))
        assert after == before

        # Backing mental models re-embedded on the target (no NULL vectors), so both
        # the vector and lexical arms of knowledge search work again.
        async with acquire_with_retry(backend) as conn:
            null_embeddings = await conn.fetchval(
                f"SELECT count(*) FROM {fq_table('mental_models')} WHERE bank_id = $1 AND embedding IS NULL",
                bank,
            )
        assert null_embeddings == 0, "restored mental models must be re-embedded"
        after_search = await memory.search_knowledge_pages(
            bank, "net-30 billing", limit=5, request_context=request_context
        )
        assert any(r["id"] == page["id"] for r in after_search), "page must be searchable after import"
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
@pytest.mark.memory_backend_incompatible
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
@pytest.mark.memory_backend_incompatible
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
@pytest.mark.memory_backend_incompatible
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
@pytest.mark.memory_backend_incompatible
async def test_export_import_observations(memory, request_context):
    """With include_observations, observations transfer and their sources re-link."""
    src = _unique_bank("transfer_obs_src")
    dst = _unique_bank("transfer_obs_dst")
    try:
        # Sources must be world/experience facts, never auto-consolidation
        # observations -- which is what retain returns, so take them from there.
        created = await _retain(memory, src, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
        assert len(created) >= 2, f"setup: retain created {len(created)} facts, need at least 2"
        source_ids = [uuid.UUID(str(i)) for i in created[:2]]

        # Create a real observation over those source facts. The helper self-acquires a
        # short-lived connection now (the embed runs off-connection), so pass the backend.
        backend = await memory._get_backend()
        archived_event_date = datetime(2001, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
        await _seed_observation(
            pool=backend,
            memory=memory,
            bank_id=src,
            source_memory_ids=source_ids,
            observation_text="Alice and Bob are colleagues.",
        )
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('memory_units')} SET event_date = $1 "
                f"WHERE bank_id = $2 AND fact_type = 'observation' AND text = $3",
                archived_event_date,
                src,
                "Alice and Bob are colleagues.",
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
        assert mine.event_date == archived_event_date
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
                f"SELECT source_memory_ids, event_date FROM {fq_table('memory_units')} "
                f"WHERE bank_id = $1 AND fact_type = 'observation' AND text = $2",
                dst,
                "Alice and Bob are colleagues.",
            )
            assert obs_row is not None
            assert obs_row["event_date"] == archived_event_date
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
@pytest.mark.memory_backend_incompatible
async def test_export_import_mental_models_and_knowledge_pages(memory, request_context):
    """The document-transfer flags carry optional bank knowledge into a target bank."""
    source = _unique_bank("transfer_knowledge_src")
    target = _unique_bank("transfer_knowledge_dst")
    try:
        await _retain(memory, source, "Alice works at Google.", request_context, "doc-1")
        page = await memory.create_knowledge_page(
            source,
            name="Work policy",
            source_query="what is the work policy",
            content="People's workplaces are useful context.",
            request_context=request_context,
        )

        archive = await memory.export_documents_async(
            source,
            request_context,
            include_observations=True,
            include_knowledge_base=True,
        )
        parsed = parse_archive(archive)
        assert parsed.manifest.mental_model_count == 1
        assert parsed.manifest.knowledge_page_count == 1
        assert parsed.observations and all(observation.created_at is not None for observation in parsed.observations)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            assert "mental_models.json" in zf.namelist()
            assert "knowledge_pages.json" in zf.namelist()

        result = await _import(memory, target, archive, request_context)
        assert result["mental_models_imported"] == 1
        assert result["knowledge_pages_imported"] == 1

        mental_models = await memory.list_mental_models(target, with_staleness=True, request_context=request_context)
        assert mental_models.total == 1
        assert mental_models.items[0]["name"] == "Work policy"
        nodes = await memory.list_knowledge_nodes(target, request_context=request_context)
        assert [(node["name"], node["mental_model_id"]) for node in nodes] == [("Work policy", page["mental_model_id"])]
        search_results = await memory.search_knowledge_pages(
            target, "work policy", limit=5, request_context=request_context
        )
        assert any(result["name"] == "Work policy" for result in search_results)
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
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
        # Bank-level knowledge cannot be combined with a document subset.
        with pytest.raises(ValueError, match="whole bank"):
            await memory.export_documents_async(src, request_context, ["doc-1"], include_observations=True)
        with pytest.raises(ValueError, match="whole bank"):
            await memory.export_documents_async(src, request_context, ["doc-1"], include_knowledge_base=True)
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
@pytest.mark.memory_backend_incompatible
async def test_http_export_import_endpoints(api_client, memory, request_context):
    """Round trip through the async HTTP export (POST + poll + download) and import endpoints."""
    src = _unique_bank("transfer_http_src")
    dst = _unique_bank("transfer_http_dst")
    try:
        await _retain(memory, src, "Dana lives in Berlin.", request_context, document_id="doc-http")

        # The old synchronous GET export is removed — it returns 410 pointing at
        # the async endpoint (it could take down the shared API on large banks).
        removed = await api_client.get(f"/v1/default/banks/{src}/document-transfer")
        assert removed.status_code == 410
        assert "document-transfer/export" in removed.json()["detail"]

        # Async export: POST returns 202 + operation_id, runs inline under the
        # SyncTaskBackend test fixture, so it's completed by the time we poll.
        submit = await api_client.post(f"/v1/default/banks/{src}/document-transfer/export")
        assert submit.status_code == 202
        export_op = submit.json()["operation_id"]

        export_status = await api_client.get(f"/v1/default/banks/{src}/operations/{export_op}")
        assert export_status.status_code == 200
        export_meta = export_status.json()["result_metadata"]
        assert export_meta["byte_size"] > 0
        download_url = export_meta["download_url"]
        assert download_url.startswith("/v1/default/files/download/tenants/")

        # Download the finished archive through the download route.
        download = await api_client.get(download_url)
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        archive = download.content
        assert len(archive) > 0
        # It is a real transfer archive for this bank.
        parsed = parse_archive(archive)
        assert parsed.manifest.source_bank_id == src

        # include_observations + a document_id subset is a 400 (validated up front).
        bad = await api_client.post(
            f"/v1/default/banks/{src}/document-transfer/export",
            params={"document_id": "meeting-notes", "include_observations": "true"},
        )
        assert bad.status_code == 400

        # Import is async: returns 202 + operation_id.
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
        missing = await api_client.post("/v1/default/banks/does-not-exist-bank/document-transfer/export")
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
        export = await api_client.post("/v1/default/banks/any-bank/document-transfer/export")
        assert export.status_code == 404
        assert "disabled" in export.json()["detail"].lower()

        # The download route serves export archives, so it is gated on the same flag.
        download = await api_client.get("/v1/default/files/download/banks/any-bank/exports/x/transfer.zip")
        assert download.status_code == 404
        assert "disabled" in download.json()["detail"].lower()

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


def test_parse_archive_rejects_a_file_that_is_not_a_zip():
    """Garbage bytes are a caller error (400), not a zipfile.BadZipFile crash (500)."""
    with pytest.raises(ValueError, match="not a readable .zip"):
        parse_archive(b"%PDF-1.7 this is not a zip at all")


def test_parse_archive_rejects_a_plain_zip_of_files():
    """A zip of ordinary documents is refused with a message that names the fix.

    Regression for #3327: users read "Import from zip" as a bulk upload of their
    own PDFs/text files, so the rejection has to say where that actually lives
    instead of only naming the missing manifest.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("notes.txt", "Dana lives in Berlin.")
        zf.writestr("report.pdf", "%PDF-1.7")

    with pytest.raises(ValueError, match="manifest.json is missing") as excinfo:
        parse_archive(buffer.getvalue())
    assert "retain" in str(excinfo.value)


@pytest.mark.asyncio
async def test_http_import_rejects_non_transfer_zip_with_400(api_client):
    """The wrong zip fails fast with a 400 whose detail explains what to upload."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("notes.txt", "Dana lives in Berlin.")

    response = await api_client.post(
        "/v1/default/banks/any-bank/document-transfer",
        files={"file": ("my-documents.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 400
    assert "manifest.json is missing" in response.json()["detail"]

    not_a_zip = await api_client.post(
        "/v1/default/banks/any-bank/document-transfer",
        files={"file": ("notes.pdf", b"%PDF-1.7", "application/pdf")},
    )
    assert not_a_zip.status_code == 400
    assert "not a readable .zip" in not_a_zip.json()["detail"]


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


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_bank_import_classifies_label_entities(memory, request_context):
    """An imported bank's label entities are stored with entity_kind='label'.

    Regression for #3236. `import_bank_async` resolved the target bank's config
    before restoring the archive's bank row — and import refuses to write into an
    existing bank, so `entity_labels` was necessarily empty for the whole import.
    Every label entity was then classified as regular, which exposes label values
    to fuzzy merging (#3187) and leaves them inside the trigram index the partial
    index (#3208) exists to keep them out of, so an imported bank silently lost
    that fix. Measured on a real 12k-entity export: 5,355 of its entities were
    label values and every one of them came back as 'regular'.
    """
    bank = _unique_bank("bank_label_kind")
    label_entity = "brief_bio:enjoys long walks on the beach"
    regular_entity = "Alice"
    try:
        await memory.ensure_bank_profile(bank_id=bank, request_context=request_context)
        await memory._config_resolver.update_bank_config(
            bank,
            {"entity_labels": [{"key": "brief_bio", "type": "text", "description": "one-line bio"}]},
        )
        await _retain(memory, bank, "Alice enjoys long walks.", request_context, "doc-1")

        backend = await memory._get_backend()
        # Link the entities to a fact directly: the mock LLM's extraction does not
        # emit a label-shaped entity, and what matters here is what the *import*
        # makes of the entities the archive carries (export derives a fact's
        # entities from unit_entities, so linking is what puts them in the archive).
        async with acquire_with_retry(backend) as conn:
            # Must be an exported fact type attached to a document, or export
            # never sees the link and the archive carries no entities at all.
            unit_id = await conn.fetchval(
                f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 "
                "AND document_id IS NOT NULL AND fact_type IN ('world', 'experience') LIMIT 1",
                bank,
            )
            assert unit_id is not None, "no facts to attach entities to"
            for name in (label_entity, regular_entity):
                # Retain may already have created the regular one.
                entity_id = await conn.fetchval(
                    f"SELECT id FROM {fq_table('entities')} WHERE bank_id = $1 AND LOWER(canonical_name) = LOWER($2)",
                    bank,
                    name,
                ) or await conn.fetchval(
                    f"INSERT INTO {fq_table('entities')} (bank_id, canonical_name) VALUES ($1, $2) RETURNING id",
                    bank,
                    name,
                )
                await conn.execute(
                    f"INSERT INTO {fq_table('unit_entities')} (unit_id, entity_id) VALUES ($1, $2) "
                    "ON CONFLICT DO NOTHING",
                    unit_id,
                    entity_id,
                )

        from hindsight_api.engine.transfer import export_bank

        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, bank)
        await memory.delete_bank(bank, request_context=request_context)
        await memory.import_bank_async(archive, request_context)

        async with acquire_with_retry(backend) as conn:
            kinds = {
                row["canonical_name"]: row["entity_kind"]
                for row in await conn.fetch(
                    f"SELECT canonical_name, entity_kind FROM {fq_table('entities')} WHERE bank_id = $1",
                    bank,
                )
            }
        assert kinds.get(label_entity) == "label", kinds
        assert kinds.get(regular_entity) == "regular", kinds
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_async_export_roundtrip(memory, request_context):
    """The async export operation stashes a real archive that re-imports cleanly.

    Mirrors the synchronous round trip, but through submit_export_documents_async:
    the worker (inline under SyncTaskBackend) builds the ZIP, stores it, and
    records the storage key / download URL / size in the operation's
    result_metadata.
    """
    src = _unique_bank("async_export_src")
    dst = _unique_bank("async_export_dst")
    try:
        await _retain(memory, src, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")

        meta, archive = await _export_async(memory, src, request_context)
        # The bank id carries a dot, which the key encodes, so derive the prefix.
        assert meta["storage_key"].startswith(f"{bank_storage_prefix(src)}exports/")
        # URL-quoted, so the key's own %-escapes survive the server's path decoding.
        assert meta["download_url"] == f"/v1/default/files/download/{quote(meta['storage_key'])}"
        assert meta["byte_size"] == len(archive)
        assert meta["filename"] == f"{src}-documents.zip"

        parsed = parse_archive(archive)
        assert parsed.manifest.source_bank_id == src
        exported_texts = {fact.text for doc in parsed.documents for fact in doc.facts}
        assert exported_texts

        result = await _import(memory, dst, archive, request_context)
        assert result["facts_imported"] == parsed.manifest.fact_count

        units = await memory.list_memory_units(dst, request_context=request_context)
        imported = {u["text"] for u in units["items"] if u["fact_type"] != "observation"}
        assert imported == exported_texts
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_async_export_include_observations_subset_rejected(memory, request_context):
    """include_observations with a document subset fails fast (before enqueue)."""
    with pytest.raises(ValueError, match="whole bank"):
        await memory.submit_export_documents_async(
            "any-bank", request_context, document_ids=["doc-1"], include_observations=True
        )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_export_attach_batching_preserves_entities_and_causal_links(memory, request_context, monkeypatch):
    """Batched attach queries carry every fact's entities and cross-batch causal edges.

    With _ATTACH_BATCH_SIZE forced to 1 each unit lands in its own batch, so a
    causal edge whose endpoints fall in different batches is exactly the case the
    old ``to_unit_id = ANY(<full set>)`` filter covered — the Python-side target
    check must still attach it.
    """
    from hindsight_api.engine.transfer import export as export_mod

    bank = _unique_bank("attach_batch")
    try:
        await _retain(memory, bank, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")

        # Insert a synthetic caused_by edge between two facts of the same document,
        # ordered the same way export assigns fact ordinals (created_at, id).
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND document_id = 'doc-1' "
                "AND fact_type IN ('world', 'experience') ORDER BY created_at, id",
                bank,
            )
            assert len(rows) >= 2, "need at least two facts to link"
            source_id, target_id = rows[0]["id"], rows[1]["id"]
            await conn.execute(
                f"INSERT INTO {fq_table('memory_links')} (bank_id, from_unit_id, to_unit_id, link_type) "
                "VALUES ($1, $2, $3, 'caused_by') ON CONFLICT DO NOTHING",
                bank,
                source_id,
                target_id,
            )

        monkeypatch.setattr(export_mod, "_ATTACH_BATCH_SIZE", 1)
        _, archive = await _export_async(memory, bank, request_context)
        parsed = parse_archive(archive)

        doc = next(d for d in parsed.documents if d.id == "doc-1")
        # Every fact kept its entities despite one-unit-per-batch fetching.
        all_entities = {name for fact in doc.facts for name in fact.entities}
        assert any("alice" in n.lower() for n in all_entities), all_entities
        assert any("bob" in n.lower() for n in all_entities), all_entities
        # The cross-batch causal edge survived: fact 0 points at fact 1.
        relations = doc.facts[0].causal_relations
        assert any(r.relation_type == "caused_by" and r.target_fact_index == 1 for r in relations), relations
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_operation_removes_export_archive(memory, request_context):
    """Deleting an export operation also deletes its stored archive (no orphan blob)."""
    bank = _unique_bank("export_delete")
    try:
        await _retain(memory, bank, "Alice works at Google.", request_context, "doc-1")
        submission = await memory.submit_export_documents_async(bank, request_context)
        op_id = submission["operation_id"]
        status = await memory.get_operation_status(bank, op_id, request_context=request_context)
        storage_key = status["result_metadata"]["storage_key"]

        # The archive exists while the operation does.
        assert await memory._file_storage.retrieve(storage_key)

        # Deleting the operation deletes the archive with it.
        await memory.delete_operation(bank, op_id, request_context=request_context)
        with pytest.raises(FileNotFoundError):
            await memory._file_storage.retrieve(storage_key)
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_purge_expired_export_archives(memory, request_context):
    """Retention's archive purge deletes the blobs of export ops past the cutoff."""
    from datetime import timedelta

    bank = _unique_bank("export_purge")
    try:
        await _retain(memory, bank, "Bob works at Microsoft.", request_context, "doc-1")
        submission = await memory.submit_export_documents_async(bank, request_context)
        op_id = submission["operation_id"]
        status = await memory.get_operation_status(bank, op_id, request_context=request_context)
        storage_key = status["result_metadata"]["storage_key"]
        assert await memory._file_storage.retrieve(storage_key)

        # The purge is schema-wide (it doesn't take a bank), and this DB is shared
        # across xdist workers — so backdate THIS op and use a past cutoff to target
        # it specifically. A future cutoff would purge other concurrent tests' fresh
        # export archives too (they'd be < cutoff), making both this count and those
        # tests flaky.
        backend = await memory._get_backend()
        old = datetime.now(timezone.utc) - timedelta(days=100)
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('async_operations')} SET updated_at = $1 WHERE operation_id = $2",
                old,
                uuid.UUID(op_id),
            )
            purged = await memory.purge_expired_export_archives(
                conn, fq_table("async_operations"), cutoff, batch_size=100
            )
        assert purged >= 1
        with pytest.raises(FileNotFoundError):
            await memory._file_storage.retrieve(storage_key)
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_purge_expired_export_archives_includes_export_bank(memory, request_context):
    """Retention's archive purge also deletes archives produced by whole-bank exports."""
    from datetime import timedelta

    bank = _unique_bank("bank_export_purge")
    try:
        await _retain(memory, bank, "Whole bank export retention test.", request_context, "doc-1")
        submission = await memory.submit_bank_export_async(bank, request_context)
        op_id = submission["operation_id"]
        status = await memory.get_operation_status(bank, op_id, request_context=request_context)
        storage_key = status["result_metadata"]["storage_key"]
        assert await memory._file_storage.retrieve(storage_key)

        backend = await memory._get_backend()
        old = datetime.now(timezone.utc) - timedelta(days=100)
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"UPDATE {fq_table('async_operations')} SET updated_at = $1 WHERE operation_id = $2",
                old,
                uuid.UUID(op_id),
            )
            purged = await memory.purge_expired_export_archives(
                conn, fq_table("async_operations"), cutoff, batch_size=100
            )
        assert purged >= 1
        with pytest.raises(FileNotFoundError):
            await memory._file_storage.retrieve(storage_key)
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_purge_expired_export_archives_honours_the_batch_bound(memory, request_context):
    """The purge deletes at most ``batch_size`` archives per call.

    Unbounded, it re-selected every expired export on every cleanup cycle and
    re-issued a blob delete for each — ``storage_key`` stays in the row until the
    row itself is pruned, so nothing marks an archive as already handled. The
    prune next to it is batched, so the purge shares that bound and the two walk
    the same ``ORDER BY updated_at, operation_id`` window together.
    """
    from datetime import timedelta

    bank = _unique_bank("export_purge_bound")
    try:
        await memory.ensure_bank_profile(bank_id=bank, request_context=request_context)
        backend = await memory._get_backend()
        # Fabricated rows rather than real exports: the purge counts rows carrying a
        # storage_key and swallows the blob delete, so no archive needs to exist for
        # the bound to be observable. Backdated far past any other test's rows so the
        # ORDER BY puts these first on the shared pg0 database.
        old = datetime.now(timezone.utc) - timedelta(days=500)
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        async with acquire_with_retry(backend) as conn:
            for i in range(2):
                await conn.execute(
                    f"""INSERT INTO {fq_table("async_operations")}
                        (operation_id, bank_id, operation_type, status, task_payload,
                         result_metadata, updated_at)
                        VALUES ($1, $2, 'export_documents', 'completed', '{{}}'::jsonb, $3::jsonb, $4)""",
                    uuid.uuid4(),
                    bank,
                    json.dumps({"storage_key": f"banks/{bank}/exports/absent-{i}.zip"}),
                    old,
                )
            # LIMIT 1 caps the result at one row regardless of which expired export
            # sorts first, so this holds even with other tests' rows in the schema.
            purged = await memory.purge_expired_export_archives(
                conn, fq_table("async_operations"), cutoff, batch_size=1
            )
        assert purged == 1
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_download_route_rejects_unauthorized_keys(api_client, memory, request_context):
    """The download route only serves bank-scoped keys for banks the caller can see."""
    bank = _unique_bank("download_guard")
    try:
        await memory.ensure_bank_profile(bank_id=bank, request_context=request_context)

        # Non-"banks/"-prefixed key: not a downloadable resource.
        r = await api_client.get("/v1/default/files/download/etc/passwd")
        assert r.status_code == 404
        # Path-traversal attempt is rejected structurally.
        r = await api_client.get("/v1/default/files/download/banks/../secrets/x.zip")
        assert r.status_code == 404
        # Well-formed key for a bank that does not exist (IDOR guard via bank read).
        r = await api_client.get("/v1/default/files/download/banks/no-such-bank/exports/x/transfer.zip")
        assert r.status_code == 404
        # Well-formed key for a visible bank but no such stored file: 404, not 500.
        r = await api_client.get(f"/v1/default/files/download/banks/{bank}/exports/missing/transfer.zip")
        assert r.status_code == 404
    finally:
        await memory.delete_bank(bank, request_context=request_context)


# --------------------------------------------------------------------------- store-owned refusal


class _StoreOwnedMemories:
    """A memories store that keeps memories outside SQL, like an external store extension."""

    def store_owned_for(self, bank_id: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_export_bank_asks_for_the_store_when_the_caller_did_not_pass_one(monkeypatch):
    """`memories=None` must not be read as "SQL-backed".

    The loaders read `documents` / `memory_units` directly, which a store-owned bank leaves empty,
    so treating an absent store as SQL produced a VALID, EMPTY archive with a success status. That
    already happened once and was fixed at the two call sites while the default that causes it
    stayed. Asserted on archive contents, because an empty archive is exactly what the broken
    version returned successfully.
    """
    import hindsight_api.engine.memories as memories_mod
    from hindsight_api.engine.transfer import export as export_mod

    # Patch the lookup, not `_resolve_memories` itself — the resolution is what is under test, and
    # `_resolve_memories` imports `get_memories` at call time.
    monkeypatch.setattr(memories_mod, "get_memories", lambda: _FakeStoreOwned())

    conn = _BankRowsOnlyConn()
    archive = await export_mod.export_bank(conn, "bank-x")  # note: no memories= argument

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        doc_names = [n for n in zf.namelist() if n.startswith("documents/")]
    assert manifest["document_count"] == 1, f"empty archive from the default path: {manifest}"
    assert doc_names, "the archive carried no document files"


@pytest.mark.asyncio
async def test_resolve_target_id_asks_the_store_that_holds_the_document(monkeypatch):
    """Every `on_conflict` mode depends on the existence check, and it was SQL-only.

    For a bank whose document store is external the SQL row is absent, so the check always said "no
    conflict": `skip` re-imported the document it was told to leave alone, `new-id` kept the original
    id, and `replace` degenerated to a plain insert — all reported as success.
    """
    from hindsight_api.engine.transfer import importer as importer_mod

    store = _RecordingStoreOwned()
    await store.put_document(bank_id="bank-x", document_id="doc-1", content_hash="h")
    monkeypatch.setattr(importer_mod, "get_memories", lambda: store, raising=False)
    import hindsight_api.engine.memories as memories_mod

    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)

    # Present in the store: skip declines, new-id remaps, replace keeps the id.
    assert await importer_mod._resolve_target_id(None, "bank-x", "doc-1", "skip") is None
    remapped = await importer_mod._resolve_target_id(None, "bank-x", "doc-1", "new-id")
    assert remapped not in (None, "doc-1")
    assert await importer_mod._resolve_target_id(None, "bank-x", "doc-1", "replace") == "doc-1"

    # Absent from the store: no conflict, the original id is used.
    assert await importer_mod._resolve_target_id(None, "bank-x", "doc-absent", "skip") == "doc-absent"


class _RecordingStoreOwned(_StoreOwnedMemories):
    """A store-owned bank that records the writes an import makes against it.

    The importer's document write is what the live suite exercises; this covers the same call
    without a server, so it runs on Postgres CI too — which is where a regression would otherwise
    only show up as a store-owned bank that restores with no documents.
    """

    def __init__(self):
        self.documents: dict[str, dict] = {}

    def store_owned_for(self, bank_id: str) -> bool:
        return True

    async def get_document_record(self, *, bank_id, document_id, include_text=False):
        return self.documents.get(document_id)

    async def put_document(self, *, bank_id, document_id, **kw):
        self.documents[document_id] = {"content_hash": kw.get("content_hash", ""), **kw}


class _BankRowsOnlyConn:
    """A connection that answers the bank-config queries and nothing else.

    Returns no rows for everything, which is a legitimate state (a bank with no mental models or
    webhooks) and lets the test assert that Postgres was consulted at all — the half of
    `export_bank` that must NOT move to the store.
    """

    def __init__(self):
        self.fetched: list[str] = []

    async def fetch(self, sql: str, *args):
        self.fetched.append(sql)
        return []

    async def fetchrow(self, sql: str, *args):
        self.fetched.append(sql)
        return None

    async def fetchval(self, sql: str, *args):
        self.fetched.append(sql)
        return None


class _FakeStoreOwned(_StoreOwnedMemories):
    """A minimal store-owned bank: one document, two chunks, two causally-linked facts.

    Hand-built rather than driven through a real store so the export's assembly is what is under
    test — ordering, ordinals, entity resolution — without a server in the loop.
    """

    _DOC_ID = "doc-1"

    def __init__(self):
        from datetime import datetime, timezone

        from hindsight_api.engine.memories.base import CausalEdgeRecord, ScanPage, StoredMemory

        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._cause = StoredMemory(
            unit_id="unit-cause",
            text="the cause",
            fact_type="world",
            document_id=self._DOC_ID,
            created_at=t0,
            entity_ids=["e-ada"],
        )
        self._effect = StoredMemory(
            unit_id="unit-effect",
            text="the effect",
            fact_type="world",
            document_id=self._DOC_ID,
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            causal_edges=[CausalEdgeRecord(target_unit_id="unit-cause", relation_type="caused_by")],
        )
        self._page = ScanPage
        self._t0 = t0

    async def list_documents(self, *, bank_id, limit=100, offset=0, **_kw):
        if offset:
            return {"items": [], "total": 1}
        return {
            "items": [{"id": self._DOC_ID, "tags": ["t"], "created_at": self._t0, "retain_params": None}],
            "total": 1,
        }

    async def get_document_record(self, *, bank_id, document_id, include_text=False):
        return {"id": document_id, "original_text": "the source text"}

    async def list_chunk_texts(self, *, bank_id, document_id):
        return ["chunk one", "chunk two"]

    async def scan_memories(self, *, bank_id, fact_types=None, page_token="", **_kw):
        if page_token:
            return self._page(memories=[], next_page_token="")
        if fact_types and "observation" in fact_types:
            return self._page(memories=[], next_page_token="")
        return self._page(memories=[self._effect, self._cause], next_page_token="")

    async def resolve_entity_names(self, *, conn, fq_table, bank_id, entity_ids):
        return {"e-ada": "Ada Lovelace"}


class _SqlMemories:
    def store_owned_for(self, bank_id: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_export_of_a_store_owned_bank_contains_its_memories():
    """The archive must carry the bank's facts, entities and causal edges — not be empty.

    This is the shape of the original defect: the loaders read `memory_units`, `unit_entities` and
    `memory_links`, which for a store-owned bank are empty, so the export produced a well-formed
    archive with nothing in it and returned 200. Asserting on archive CONTENTS rather than on a
    status code is the point — an empty archive is exactly what the broken version returned
    successfully.
    """
    from hindsight_api.engine.transfer.export import export_documents

    archive = await export_documents(None, "bank-x", None, memories=_FakeStoreOwned())

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        doc_names = [n for n in names if n.startswith("documents/")]
        assert doc_names, names
        doc = json.loads(zf.read(doc_names[0]))
        manifest = json.loads(zf.read("manifest.json"))

    assert manifest["fact_count"] == 2, manifest
    assert manifest["document_count"] == 1, manifest
    assert doc["original_text"] == "the source text"
    assert [c["chunk_text"] for c in doc["chunks"]] == ["chunk one", "chunk two"]

    texts = [f["text"] for f in doc["facts"]]
    assert texts == ["the cause", "the effect"], texts
    # The entity name has to come back through the store's registry, not a SQL join.
    assert doc["facts"][0]["entities"] == ["Ada Lovelace"], doc["facts"][0]
    # And the causal edge has to survive as an ordinal into this document's fact list.
    assert doc["facts"][1]["causal_relations"] == [{"relation_type": "caused_by", "target_fact_index": 0}]


@pytest.mark.asyncio
async def test_export_bank_of_a_store_owned_bank_carries_its_memories():
    """The whole-bank archive takes memories from the store and everything else from SQL.

    `export_bank` is a superset of the document export: only the memories move, while bank config,
    mental models, directives, webhooks, knowledge pages and the history tails stay in Postgres for
    every deployment. Both halves are asserted, because routing all of it to the store would lose
    the config and routing none of it would lose the memories.
    """
    from hindsight_api.engine.transfer.export import export_bank

    conn = _BankRowsOnlyConn()
    archive = await export_bank(conn, "bank-x", memories=_FakeStoreOwned())

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        doc_names = [n for n in names if n.startswith("documents/")]
        assert doc_names, names
        doc = json.loads(zf.read(doc_names[0]))

    # The memories came from the store...
    assert [f["text"] for f in doc["facts"]] == ["the cause", "the effect"]
    assert doc["facts"][1]["causal_relations"] == [{"relation_type": "caused_by", "target_fact_index": 0}]
    # ...and Postgres was still consulted for the bank's own rows.
    assert conn.fetched, "export_bank must still read bank config/history from Postgres"


@pytest.mark.asyncio
async def test_a_sql_backed_bank_is_not_read_through_the_store():
    """A Postgres bank must keep taking the connection path, not be routed to the store."""
    from hindsight_api.engine.transfer.export import export_documents

    with pytest.raises(Exception) as ei:  # noqa: PT011 - backend=None fails once SQL is reached
        await export_documents(None, "bank-x", None, memories=_SqlMemories())
    assert "list_documents" not in str(ei.value), "a SQL bank must not be read through the store"


class _RecordingEmbedder:
    """Minimal embeddings backend that records the size of every encode call."""

    dimension = 2

    def __init__(self):
        self.batch_sizes: list[int] = []

    async def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(len(text)), 0.0] for text in texts]


@pytest.mark.asyncio
async def test_embed_in_batches_bounds_call_size_and_keeps_order():
    """Import embeds bank-sized lists; nothing below it bounds an in-process provider.

    Without this slice, ``_import_observations`` hands the embedder every observation in
    the bank in a single call and peak memory scales with the bank (issue #3891).
    """
    embedder = _RecordingEmbedder()
    texts = [f"observation {index}" for index in range(_EMBED_BATCH_SIZE * 2 + 5)]

    vectors = await _embed_in_batches(embedder, texts)

    assert embedder.batch_sizes == [_EMBED_BATCH_SIZE, _EMBED_BATCH_SIZE, 5]
    assert vectors == [[float(len(text)), 0.0] for text in texts]


@pytest.mark.asyncio
async def test_embed_in_batches_handles_empty_input():
    embedder = _RecordingEmbedder()

    assert await _embed_in_batches(embedder, []) == []
    assert embedder.batch_sizes == []


@pytest.mark.asyncio
async def test_bank_copy_carries_directives_and_webhooks(memory, request_context):
    """Copying a bank on the same instance must actually write its directives and webhooks.

    Both tables have a globally-unique ``id`` primary key. The archive carried
    those ids verbatim and ``_restore_rows`` inserts ON CONFLICT DO NOTHING, so
    every row collided with the still-present source row, wrote nothing, and was
    still counted as imported — a copy that reported success and silently had no
    directives and no webhooks.
    """
    from hindsight_api.engine.transfer import export_bank

    source = _unique_bank("copy_src")
    target = _unique_bank("copy_dst")
    try:
        await _retain(memory, source, "Dana works at Vectorize.", request_context, "doc-1")
        await memory.create_directive(source, name="tone", content="Answer briefly.", request_context=request_context)
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('webhooks')} "
                f"(id, bank_id, url, secret, event_types, enabled, created_at, updated_at) "
                f"VALUES ($1, $2, $3, NULL, $4, true, NOW(), NOW())",
                uuid.uuid4(),
                source,
                "https://example.com/hook",
                ["retain.completed"],
            )
            archive = await export_bank(conn, source, file_storage=memory._file_storage)

        result = await memory.import_bank_async(archive, request_context, target_bank_id=target)
        assert result.directives_imported == 1
        assert result.webhooks_imported == 1

        # The counts above are what lied before the fix; these are the rows, read
        # back through the same API a user would.
        directives = await memory.list_directives(target, active_only=False, request_context=request_context)
        webhooks = await memory.list_webhooks(target, request_context=request_context)
        assert [d["name"] for d in directives.items] == ["tone"]
        assert [w["url"] for w in webhooks["items"]] == ["https://example.com/hook"]

        # Fresh ids: keeping the source's is what made the insert a no-op. Read
        # directly because the id is the mechanism rather than the observable
        # outcome — the assertions above are what a user sees, this is why they hold.
        async with acquire_with_retry(backend) as conn:
            copied_ids = {
                r["id"] for r in await conn.fetch(f"SELECT id FROM {fq_table('directives')} WHERE bank_id = $1", target)
            }
            source_ids = {
                r["id"] for r in await conn.fetch(f"SELECT id FROM {fq_table('directives')} WHERE bank_id = $1", source)
            }
        assert copied_ids and not (copied_ids & source_ids)
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_scope_exports_and_restores_only_what_was_asked_for(memory, request_context):
    """The three booleans select what travels, on both halves of the transfer."""
    from hindsight_api.engine.transfer import TransferScope, export_bank

    source = _unique_bank("scope_src")
    config_only = _unique_bank("scope_cfg")
    data_only = _unique_bank("scope_data")
    try:
        await _retain(memory, source, "Eve lives in Berlin.", request_context, "doc-1")
        await memory.create_mental_model(
            source,
            name="Places",
            source_query="where do people live",
            content="People and their cities.",
            mental_model_id="mm-1",
            request_context=request_context,
        )
        backend = await memory._get_backend()

        async with acquire_with_retry(backend) as conn:
            config_archive = await export_bank(
                conn, source, scope=TransferScope(data=False, bank_config=True), file_storage=memory._file_storage
            )
            data_archive = await export_bank(
                conn, source, scope=TransferScope(data=True, bank_config=False), file_storage=memory._file_storage
            )

        with zipfile.ZipFile(io.BytesIO(config_archive)) as zf:
            config_names = set(zf.namelist())
        with zipfile.ZipFile(io.BytesIO(data_archive)) as zf:
            data_names = set(zf.namelist())
        assert not any(n.startswith("documents/") for n in config_names)
        assert any(n.startswith("documents/") for n in data_names)
        # Mental models are a reading of the bank's facts, so they travel with the
        # data rather than with the settings — a config-only archive carrying them
        # would restore a synthesis whose evidence resolves to nothing.
        assert "mental_models.json" in data_names
        assert "mental_models.json" not in config_names
        assert "directives.json" in config_names
        assert "directives.json" not in data_names

        config_result = await memory.import_bank_async(config_archive, request_context, target_bank_id=config_only)
        assert config_result.documents_imported == 0
        assert config_result.mental_models_imported == 0

        data_result = await memory.import_bank_async(data_archive, request_context, target_bank_id=data_only)
        assert data_result.documents_imported == 1
        assert data_result.mental_models_imported == 1
        # The bank row came from this instance's defaults rather than the archive,
        # but it exists — the facts had to land somewhere.
        assert await memory.get_bank_profile(data_only, request_context=request_context) is not None
    finally:
        for bank in (source, config_only, data_only):
            await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_restored_operations_log_cannot_re_run_the_source_bank_work(memory, request_context):
    """An operation still in flight at export time restores as cancelled.

    ``async_operations`` is the task queue as well as the log: a restored
    ``pending`` row is work the target's worker would actually run, re-firing the
    source bank's webhooks against a bank that never asked for it.
    """
    from hindsight_api.engine.transfer import export_bank

    source = _unique_bank("ops_src")
    target = _unique_bank("ops_dst")
    try:
        await _retain(memory, source, "Frank plays the cello.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            await conn.execute(
                f"INSERT INTO {fq_table('async_operations')} "
                f"(operation_id, bank_id, operation_type, status, task_payload) "
                f"VALUES ($1, $2, 'retain', 'pending', '{{}}'::jsonb), "
                f"       ($3, $2, 'retain', 'completed', '{{}}'::jsonb)",
                uuid.uuid4(),
                source,
                uuid.uuid4(),
            )
            archive = await export_bank(conn, source, file_storage=memory._file_storage)

        await memory.import_bank_async(archive, request_context, target_bank_id=target)

        async with acquire_with_retry(backend) as conn:
            statuses = [
                r["status"]
                for r in await conn.fetch(
                    f"SELECT status FROM {fq_table('async_operations')} WHERE bank_id = $1 AND operation_type = 'retain'",
                    target,
                )
            ]
        # The finished work is the copy's history; the in-flight row belonged to
        # the source and is not carried at all. Restoring it as cancelled was the
        # first attempt, and it put a cancelled clone_bank row — the clone's own
        # operation — in every copy.
        assert statuses == ["completed"]
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
# Seeds the attachment link with a raw INSERT that needs the document's SQL row.
@pytest.mark.memory_backend_incompatible
async def test_attachment_bytes_travel_with_the_bank(memory, request_context):
    """An attachment's bytes ride in the archive and land under the target's own key.

    The storage key encodes tenant and bank, so carrying the source's key would
    point the copy at the source's blob — which deleting the source bank then
    sweeps out from under it.
    """
    from hindsight_api.engine.retain.attachment_content import RetainAttachment
    from hindsight_api.engine.retain.attachment_store import store_images
    from hindsight_api.engine.transfer import export_bank

    source = _unique_bank("att_src")
    target = _unique_bank("att_dst")
    payload = b"\x89PNG\r\n\x1a\n-not-really-a-png"
    try:
        await _retain(memory, source, "Grace shared a diagram.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            stored = await store_images(
                memory._file_storage,
                conn,
                source,
                "doc-1",
                [
                    RetainAttachment(
                        attachment_hash="a" * 64,
                        media_type="image/png",
                        data=payload,
                        block_index=0,
                        filename="diagram.png",
                    )
                ],
            )
            archive = await export_bank(conn, source, file_storage=memory._file_storage)

        result = await memory.import_bank_async(archive, request_context, target_bank_id=target)
        assert result.attachments_imported == 1

        async with acquire_with_retry(backend) as conn:
            row = await conn.fetchrow(
                f"SELECT storage_key, short_id, media_type, document_id, filename "
                f"FROM {fq_table('attachments')} WHERE bank_id = $1",
                target,
            )
        assert row is not None
        assert row["storage_key"] != stored[0].storage_key
        # The row carries its owning document and that document's name for it, so
        # there is no separate edge table to carry across.
        assert row["document_id"] == "doc-1"
        assert row["filename"] == "diagram.png"
        assert bytes(await memory._file_storage.retrieve(row["storage_key"])) == payload
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


async def _seed_attachment(memory, bank_id: str, payload: bytes, attachment_hash: str) -> str:
    from hindsight_api.engine.retain.attachment_content import RetainAttachment
    from hindsight_api.engine.retain.attachment_store import store_images

    backend = await memory._get_backend()
    async with acquire_with_retry(backend) as conn:
        stored = await store_images(
            memory._file_storage,
            conn,
            bank_id,
            "doc-1",
            [
                RetainAttachment(
                    attachment_hash=attachment_hash,
                    media_type="image/png",
                    data=payload,
                    block_index=0,
                    filename="diagram.png",
                )
            ],
        )
    return stored[0].storage_key


def _archive_entries(archive: bytes) -> dict[str, Any]:
    """Every entry of an archive, parsed where it is JSON; ``exported_at`` is the one field allowed to differ."""
    entries: dict[str, Any] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if not name.endswith(".json"):
                entries[name] = data
                continue
            parsed = json.loads(data)
            if name == "manifest.json":
                parsed.pop("exported_at")
            entries[name] = parsed
    return entries


@pytest.mark.asyncio
# Seeds the attachment link with a raw INSERT that needs the document's SQL row.
@pytest.mark.memory_backend_incompatible
async def test_streamed_bank_archive_matches_the_built_one(memory, request_context):
    """The streamed export and the clone's in-memory builder write the same archive.

    They are two implementations of one format — the export streams, the clone
    reads under one transaction and builds — so nothing else stops a section
    added to one from being forgotten in the other.
    """
    from hindsight_api.engine.transfer import TransferScope, build_bank_archive, load_bank_export, stream_export_bank

    bank = _unique_bank("parity")
    try:
        await _retain(memory, bank, "Grace shared a diagram of the Paris office.", request_context, "doc-1")
        await _retain(memory, bank, "Alan moved to Berlin in 2021.", request_context, "doc-2")
        await _seed_attachment(memory, bank, b"\x89PNG-parity", "b" * 64)
        backend = await memory._get_backend()
        scope = TransferScope(data=True, bank_config=True, history=True)
        streamed = b"".join(
            [
                chunk
                async for chunk in stream_export_bank(
                    backend, bank, scope=scope, file_storage=memory._file_storage, batch_size=1
                )
            ]
        )
        async with acquire_with_retry(backend) as conn:
            payload = await load_bank_export(conn, bank, scope=scope, file_storage=memory._file_storage)
        built = await build_bank_archive(payload)

        streamed_entries = _archive_entries(streamed)
        assert streamed_entries == _archive_entries(built)
        assert streamed_entries["manifest.json"]["document_count"] == 2
        assert streamed_entries["manifest.json"]["attachment_count"] == 1
        assert streamed_entries["blobs/000000.bin"] == b"\x89PNG-parity"
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
# Seeds the attachment link with a raw INSERT that needs the document's SQL row.
@pytest.mark.memory_backend_incompatible
async def test_an_attachment_whose_bytes_are_gone_is_left_out_of_both_archives(memory, request_context):
    """A row that outlived its blob is dropped with a warning; the rest of the bank still exports.

    Before, the missing blob raised out of storage and failed the whole export —
    one lost file made a bank impossible to move.
    """
    from hindsight_api.engine.transfer import TransferScope, build_bank_archive, load_bank_export, stream_export_bank

    bank = _unique_bank("att_gone")
    try:
        await _retain(memory, bank, "Grace shared a diagram.", request_context, "doc-1")
        kept = await _seed_attachment(memory, bank, b"kept-bytes", "c" * 64)
        gone = await _seed_attachment(memory, bank, b"gone-bytes", "d" * 64)
        await memory._file_storage.delete(gone)
        assert await memory._file_storage.exists(kept)

        backend = await memory._get_backend()
        streamed = b"".join(
            [chunk async for chunk in stream_export_bank(backend, bank, file_storage=memory._file_storage)]
        )
        async with acquire_with_retry(backend) as conn:
            built = await build_bank_archive(
                await load_bank_export(conn, bank, scope=TransferScope(), file_storage=memory._file_storage)
            )
        for archive in (streamed, built):
            entries = _archive_entries(archive)
            assert entries["manifest.json"]["attachment_count"] == 1
            assert [a["attachment_hash"] for a in entries["attachments.json"]] == ["c" * 64]
            assert entries[entries["attachments.json"][0]["entry"]] == b"kept-bytes"
            assert entries["manifest.json"]["document_count"] == 1
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_invalidated_facts_survive_a_bank_copy(memory, request_context):
    """The curation archive travels, so a copied bank can still revert what was retired."""
    from hindsight_api.engine.transfer import export_bank

    source = _unique_bank("inv_src")
    target = _unique_bank("inv_dst")
    try:
        unit_ids = await _retain(memory, source, "Helen owns a red bicycle.", request_context, "doc-1")
        await memory.update_memory_unit(
            source,
            str(unit_ids[0]),
            state="invalidated",
            reason="wrong colour",
            request_context=request_context,
        )
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, source, file_storage=memory._file_storage)

        result = await memory.import_bank_async(archive, request_context, target_bank_id=target)
        assert result.invalidated_memories_imported == 1

        async with acquire_with_retry(backend) as conn:
            rows = await conn.fetch(
                f"SELECT id, text, document_id, invalidation_reason FROM {fq_table('invalidated_memory_units')} "
                f"WHERE bank_id = $1",
                target,
            )
            source_ids = {
                r["id"]
                for r in await conn.fetch(
                    f"SELECT id FROM {fq_table('invalidated_memory_units')} WHERE bank_id = $1", source
                )
            }
        assert len(rows) == 1
        assert rows[0]["invalidation_reason"] == "wrong colour"
        # Through the store: one that owns its documents has no SQL row for the column's
        # foreign key to reference, so it keeps the document elsewhere and leaves the column NULL.
        from hindsight_api.engine.memories import get_memories

        async with acquire_with_retry(backend) as conn:
            archived = await get_memories().get_archived_memory(
                conn=conn, fq_table=fq_table, bank_id=target, unit_id=str(rows[0]["id"])
            )
        assert archived is not None and archived.document_id == "doc-1"
        # Fresh unit id: the source row is still there on a same-instance copy.
        assert rows[0]["id"] not in source_ids
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_transfer_endpoints_round_trip_a_bank(api_client, memory, request_context):
    """The unified endpoints export a bank and restore it under a new id."""
    source = _unique_bank("http_src")
    target = _unique_bank("http_dst")
    try:
        await _retain(memory, source, "Ivan speaks Portuguese.", request_context, "doc-1")

        response = await api_client.post(f"/v1/default/banks/{quote(source)}/transfer/export")
        assert response.status_code == 202, response.text
        operation_id = response.json()["operation_id"]

        operation = await api_client.get(f"/v1/default/banks/{quote(source)}/operations/{operation_id}")
        assert operation.status_code == 200, operation.text
        storage_key = operation.json()["result_metadata"]["storage_key"]
        archive = await memory._file_storage.retrieve(storage_key)

        response = await api_client.post(
            f"/v1/default/banks/{quote(source)}/transfer/import",
            params={"target_bank_id": target},
            files={"file": ("transfer.zip", bytes(archive), "application/zip")},
        )
        assert response.status_code == 202, response.text

        # The copy holds exactly the source's facts. Comparing the two banks rather
        # than naming the texts keeps the assertion total without pinning what the
        # mock LLM happens to extract from the prompt it echoes back.
        source_facts = await memory.list_memory_units(source, limit=100, request_context=request_context)
        target_facts = await memory.list_memory_units(target, limit=100, request_context=request_context)
        assert sorted(u["text"] for u in target_facts["items"]) == sorted(u["text"] for u in source_facts["items"])
        assert any(u["text"] == "Ivan speaks Portuguese." for u in target_facts["items"])
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_transfer_import_rejects_an_existing_target(api_client, memory, request_context):
    """Restoring into a bank that exists is a caller error, not a background failure."""
    from hindsight_api.engine.transfer import export_bank

    source = _unique_bank("exists_src")
    try:
        await _retain(memory, source, "Jo collects stamps.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            archive = await export_bank(conn, source, file_storage=memory._file_storage)

        response = await api_client.post(
            f"/v1/default/banks/{quote(source)}/transfer/import",
            params={"target_bank_id": source},
            files={"file": ("transfer.zip", archive, "application/zip")},
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    finally:
        await memory.delete_bank(source, request_context=request_context)


@pytest.mark.asyncio
async def test_transfer_endpoints_refuse_a_request_that_would_do_nothing(api_client, memory, request_context):
    """A scope that carries nothing, and a scope flag that a merge would ignore,
    are both caller errors — accepting either produces an archive or an import
    that silently is not what was asked for."""
    bank = _unique_bank("guards")
    try:
        await _retain(memory, bank, "Kim studies geology.", request_context, "doc-1")

        nothing = await api_client.post(
            f"/v1/default/banks/{quote(bank)}/transfer/export",
            params={"include_data": False, "include_bank_config": False, "include_history": False},
        )
        assert nothing.status_code == 400
        assert "Nothing to export" in nothing.json()["detail"]

        # A document subset is not a bank, so it cannot carry bank-level sections.
        subset = await api_client.post(
            f"/v1/default/banks/{quote(bank)}/transfer/export",
            params={"document_id": "doc-1", "include_bank_config": True},
        )
        assert subset.status_code == 400

        merge_with_scope = await api_client.post(
            f"/v1/default/banks/{quote(bank)}/transfer/import",
            params={"mode": "merge", "include_bank_config": True},
            files={"file": ("transfer.zip", b"not-a-zip", "application/zip")},
        )
        assert merge_with_scope.status_code == 400
        assert "mode=restore" in merge_with_scope.json()["detail"]
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_clone_copies_the_bank_and_leaves_the_two_independent(api_client, memory, request_context):
    """A clone holds the source's memories and configuration, and then goes its own way."""
    source = _unique_bank("clone_src")
    target = _unique_bank("clone_dst")
    try:
        await _retain(memory, source, "Lena restores violins.", request_context, "doc-1")
        await memory.create_directive(source, name="tone", content="Answer briefly.", request_context=request_context)

        response = await api_client.post(f"/v1/default/banks/{quote(source)}/clone", params={"target_bank_id": target})
        assert response.status_code == 202, response.text
        operation_id = response.json()["operation_id"]

        # The operation belongs to the source — the target did not exist when it was submitted.
        status = await memory.get_operation_status(source, operation_id, request_context=request_context)
        assert status["status"] == "completed", status
        assert status["result_metadata"]["target_bank_id"] == target

        source_facts = await memory.list_memory_units(source, limit=100, request_context=request_context)
        clone_facts = await memory.list_memory_units(target, limit=100, request_context=request_context)
        assert sorted(u["text"] for u in clone_facts["items"]) == sorted(u["text"] for u in source_facts["items"])
        directives = await memory.list_directives(target, active_only=False, request_context=request_context)
        assert [d["name"] for d in directives.items] == ["tone"]

        # Independent from here: a write to the clone must not reach the source.
        await _retain(memory, target, "Lena bought a workshop.", request_context, "doc-2")
        source_after = await memory.list_memory_units(source, limit=100, request_context=request_context)
        clone_after = await memory.list_memory_units(target, limit=100, request_context=request_context)
        assert any("workshop" in u["text"] for u in clone_after["items"])
        assert not any("workshop" in u["text"] for u in source_after["items"])
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_clone_refuses_a_target_that_would_be_overwritten(api_client, memory, request_context):
    """Cloning onto an existing bank, or onto itself, is refused up front — both
    would mix two banks' configuration into one with nothing to undo it."""
    source = _unique_bank("clone_guard_src")
    existing = _unique_bank("clone_guard_dst")
    try:
        await _retain(memory, source, "Milo tunes pianos.", request_context, "doc-1")
        # ensure_, not get_: since #4465 a profile read never creates the bank, and this
        # test needs the target to actually exist for the clone guard to have something
        # to refuse.
        await memory.ensure_bank_profile(existing, request_context=request_context)

        onto_existing = await api_client.post(
            f"/v1/default/banks/{quote(source)}/clone", params={"target_bank_id": existing}
        )
        assert onto_existing.status_code == 400
        assert "already exists" in onto_existing.json()["detail"]

        onto_itself = await api_client.post(
            f"/v1/default/banks/{quote(source)}/clone", params={"target_bank_id": source}
        )
        assert onto_itself.status_code == 400

        missing_source = await api_client.post(
            "/v1/default/banks/does-not-exist-bank/clone", params={"target_bank_id": _unique_bank("clone_never")}
        )
        assert missing_source.status_code == 404
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(existing, request_context=request_context)


@pytest.mark.asyncio
async def test_clone_can_leave_the_configuration_behind(api_client, memory, request_context):
    """Copying an agent's memory without copying what it is wired to do — including
    its webhooks, which point at the source's own consumer."""
    source = _unique_bank("clone_data_src")
    target = _unique_bank("clone_data_dst")
    try:
        await _retain(memory, source, "Nina keeps bees.", request_context, "doc-1")
        await memory.create_directive(source, name="tone", content="Answer briefly.", request_context=request_context)

        response = await api_client.post(
            f"/v1/default/banks/{quote(source)}/clone",
            params={"target_bank_id": target, "include_bank_config": False},
        )
        assert response.status_code == 202, response.text

        clone_facts = await memory.list_memory_units(target, limit=100, request_context=request_context)
        directives = await memory.list_directives(target, active_only=False, request_context=request_context)
        assert clone_facts["items"]
        assert directives.items == []
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_a_copy_does_not_keep_the_source_default_name(api_client, memory, request_context):
    """A bank's default name is its id, so a copy that kept it would read as the
    source in every list that shows names — two entries, same name, different ids."""
    source = _unique_bank("name_src")
    target = _unique_bank("name_dst")
    try:
        await _retain(memory, source, "Otto tunes harpsichords.", request_context, "doc-1")
        response = await api_client.post(f"/v1/default/banks/{quote(source)}/clone", params={"target_bank_id": target})
        assert response.status_code == 202, response.text

        profile = await memory.get_bank_profile(target, request_context=request_context)
        assert profile is not None
        assert profile["name"] == target
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


@pytest.mark.asyncio
async def test_a_copy_keeps_a_name_someone_chose(api_client, memory, request_context):
    """The rename only follows the default. A name a user set is theirs, and a
    clone that renamed it to a bank id would be losing information."""
    source = _unique_bank("named_src")
    target = _unique_bank("named_dst")
    try:
        await _retain(memory, source, "Pia restores clocks.", request_context, "doc-1")
        await memory.update_bank(source, name="Pia's workshop", request_context=request_context)

        response = await api_client.post(f"/v1/default/banks/{quote(source)}/clone", params={"target_bank_id": target})
        assert response.status_code == 202, response.text

        profile = await memory.get_bank_profile(target, request_context=request_context)
        assert profile is not None
        assert profile["name"] == "Pia's workshop"
    finally:
        await memory.delete_bank(source, request_context=request_context)
        await memory.delete_bank(target, request_context=request_context)


def test_archive_assembly_is_confined_to_threadable_builders():
    """Every non-streamed ZIP is written by a plain function whose name starts with `_build_`.

    Those are the ones the async wrappers hand to a worker thread. Both document
    and whole-bank external exports stream directly via ZipStreamer, while internal
    whole-bank archive assembly for clone_bank runs off the event loop via
    _build_bank_archive_bytes under a single-transaction read snapshot.
    """
    import ast
    from pathlib import Path

    from hindsight_api.engine.transfer import export as export_module

    source = Path(export_module.__file__).read_text()
    tree = ast.parse(source)

    offenders = []
    builders = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_build_"):
            builders.add(node.name)
            # A builder has to be callable from a thread, so it must not be async.
            assert isinstance(node, ast.FunctionDef), f"{node.name} must be a plain def to run in a thread"
            continue
        body = ast.get_source_segment(source, node) or ""
        if "zipfile.ZipFile(" in body:
            offenders.append(node.name)

    assert offenders == [], f"archive written outside a threadable builder: {offenders}"
    assert {"_build_bank_archive_bytes"} <= builders


def test_the_engine_never_compresses_inside_a_read_transaction():
    """No archive is built while a transaction is open.

    Splitting load from build only helps if the callers keep them apart: moving
    the build back inside the `async with conn.transaction()` block would pin a
    pooled connection for the whole compression again, and would look perfectly
    reasonable in review — which is why this is asserted structurally rather than
    left to the next reader to notice.
    """
    import ast
    from pathlib import Path

    from hindsight_api.engine import memory_engine

    source = Path(memory_engine.__file__).read_text()
    tree = ast.parse(source)

    def builds_an_archive(node: ast.AST) -> bool:
        return any(
            isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "build_bank_archive"
            for inner in ast.walk(node)
        )

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        header = (ast.get_source_segment(source, node) or "").splitlines()[:1]
        if header and ".transaction(" in header[0] and builds_an_archive(node):
            offenders.append(f"line {node.lineno}: {header[0].strip()}")

    assert offenders == [], f"archive built inside a transaction: {offenders}"


@pytest.mark.asyncio
async def test_bank_archive_builds_without_the_connection_that_read_it(memory, request_context):
    """The compression runs after the read transaction is closed.

    Loading and building are separate calls precisely so a pooled connection is
    not held for the length of a whole-bank DEFLATE. Building from a payload with
    no connection in scope is what proves the two halves are actually independent.
    """
    from hindsight_api.engine.transfer import build_bank_archive, load_bank_export

    bank = _unique_bank("export_seam")
    try:
        await _retain(memory, bank, "Rosa sails dinghies.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                payload = await load_bank_export(bank_id=bank, conn=conn, file_storage=memory._file_storage)

        # The connection is back in the pool here.
        archive = await build_bank_archive(payload)

        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
            manifest = TransferManifest.model_validate_json(zf.read("manifest.json"))
        assert any(n.startswith("documents/") for n in names)
        assert manifest.source_bank_id == bank
        assert manifest.document_count == 1
    finally:
        await memory.delete_bank(bank, request_context=request_context)
