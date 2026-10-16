"""Import reads a transfer archive as it replays it, never parsed whole (#5524).

A bank of a million facts used to be held in memory several times over: the upload,
every parsed document, every observation, each document's text until the end of the
import, and one remapped id per fact and observation. These pin the pieces that
replaced that.
"""

import io
import json
import tempfile
import uuid
import zipfile

import pytest

from hindsight_api.engine.transfer import importer
from tests.retain_result_capture import RetainResultCapture
from tests.test_document_transfer import _import, _retain, _seed_observation, _unique_bank


def _zip_with(name: str, payload: str) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(name, payload)
    return zipfile.ZipFile(buffer)


def test_json_array_items_survive_any_read_boundary(monkeypatch):
    """Items split across reads, numbers cut at a read's end, and multi-byte text all decode intact."""
    monkeypatch.setattr(importer, "_JSON_READ_CHARS", 2)
    items = [{"a": 1, "nested": [1, [2, {"b": None}]]}, 12345, "héllo ✓ 日本", -1.5e3, True, [], {}]
    for payload in (json.dumps(items), json.dumps(items, indent=2), json.dumps(items, separators=(",", ":"))):
        assert list(importer._iter_json_array(_zip_with("rows.json", payload), "rows.json")) == items


def test_json_array_empty_and_missing_entries_yield_nothing():
    assert list(importer._iter_json_array(_zip_with("rows.json", " [ ] "), "rows.json")) == []
    assert list(importer._iter_json_array(_zip_with("rows.json", "[]"), "absent.json")) == []


@pytest.mark.parametrize("payload", ['[{"a": 1}, {"b": ', '{"a": 1}', "", "[1, 2"])
def test_json_array_rejects_a_truncated_or_non_array_entry(payload):
    with pytest.raises(ValueError, match="Invalid transfer archive"):
        list(importer._iter_json_array(_zip_with("rows.json", payload), "rows.json"))


def test_referenced_ids_reach_into_json_serialized_as_text():
    """Mental-model evidence can arrive as a JSON string; the ids inside it still count."""
    direct, nested = str(uuid.uuid4()), str(uuid.uuid4())
    rows = [{"unit_id": direct, "reflect_response": json.dumps({"based_on": {"world": [{"id": nested}]}})}]
    assert importer._referenced_ids(rows, []) == {direct, nested}


@pytest.mark.asyncio
async def test_import_from_a_file_in_small_batches(memory, request_context, monkeypatch):
    """An archive handed over as a file imports whole across many batches, and the
    post-retain hook still gets every document's text, re-read from the archive."""
    src = _unique_bank("transfer_stream_src")
    dst = _unique_bank("transfer_stream_dst")
    capture = RetainResultCapture()
    original_validator = memory._operation_validator
    try:
        created = await _retain(memory, src, "Alice works at Google.", request_context, "doc-1")
        await _retain(memory, src, "Bob works at Microsoft.", request_context, "doc-2")
        backend = await memory._get_backend()
        await _seed_observation(
            pool=backend,
            memory=memory,
            bank_id=src,
            source_memory_ids=[uuid.UUID(str(created[0]))],
            observation_text="Alice is employed.",
        )
        archive = await memory.export_documents_async(src, request_context, include_observations=True)
        exported = importer.parse_archive(archive).manifest

        monkeypatch.setattr(importer, "_DOCUMENT_BATCH_SIZE", 1)
        monkeypatch.setattr(importer, "_OBSERVATION_BATCH_SIZE", 1)
        memory._operation_validator = capture
        with tempfile.TemporaryFile() as archive_file:
            archive_file.write(archive)
            archive_file.seek(0)
            result = await _import(memory, dst, archive_file, request_context)

        assert result["documents_imported"] == exported.document_count == 2
        assert result["facts_imported"] == exported.fact_count
        assert result["observations_imported"] == exported.observation_count >= 1
        assert {(r.document_id, r.contents[0]["content"]) for r in capture.results} == {
            ("doc-1", "Alice works at Google."),
            ("doc-2", "Bob works at Microsoft."),
        }
        assert all(r.unit_ids and r.unit_ids[0] for r in capture.results)
    finally:
        memory._operation_validator = original_validator
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)


@pytest.mark.asyncio
async def test_history_rows_stream_into_a_renamed_bank(memory, request_context):
    """History rows are read one at a time and land under the target bank, with fresh ids."""
    from hindsight_api.engine.db_utils import acquire_with_retry
    from hindsight_api.engine.schema import fq_table
    from hindsight_api.engine.transfer import TransferScope, export_bank

    src = _unique_bank("transfer_history_src")
    dst = _unique_bank("transfer_history_dst")
    try:
        await _retain(memory, src, "Alice works at Google.", request_context, "doc-1")
        backend = await memory._get_backend()
        async with acquire_with_retry(backend) as conn:
            # Seeded directly: nothing in the test engine writes audit rows on its own.
            await conn.execute(
                f"INSERT INTO {fq_table('audit_log')} (action, transport, bank_id) VALUES ($1, 'http', $2)",
                "streamed_history_probe",
                src,
            )
            archive = await export_bank(conn, src, scope=TransferScope(history=True))

        result = await memory.import_bank_async(
            archive, request_context, target_bank_id=dst, scope=TransferScope(history=True)
        )

        assert result.history_rows_imported >= 1
        restored = await memory.list_audit_logs(dst, request_context=request_context, action="streamed_history_probe")
        assert restored is not None and restored.total == 1
        source = await memory.list_audit_logs(src, request_context=request_context, action="streamed_history_probe")
        assert source is not None and source.total == 1
        assert restored.items[0].id != source.items[0].id
    finally:
        await memory.delete_bank(src, request_context=request_context)
        await memory.delete_bank(dst, request_context=request_context)
