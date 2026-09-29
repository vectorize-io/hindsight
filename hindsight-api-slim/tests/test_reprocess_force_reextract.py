"""#3899: an explicit reprocess must re-extract, never be classified as a no-op.

A reprocess replays the document's own stored text, so the content is byte-identical
by construction — and the retain pipeline has two skips for exactly that shape: the
delta path finds no changed chunk and updates document metadata only, and the
crash-recovery gate sees the matching ``content_hash`` plus surviving chunk hashes,
preserves every existing unit and makes zero LLM calls. Either one settles the
operation ``completed`` with ``unit_ids_count: 0``, indistinguishable from a real
re-extraction unless you read the facts.

Both skips stay in place for an ordinary re-push of unchanged content, which is what
they are for; only ``force_reextract`` suppresses them.
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio

from hindsight_api.api import create_app

_CONTENT = "Alice works at Google. Bob works at Meta."


@pytest.fixture(scope="session")
def db_url():
    """Isolated pg0 instance so this module never touches the dev database."""
    return "pg0://hs3899:55499"


@pytest_asyncio.fixture
async def api_client(memory):
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _child_metadata(memory, bank_id, parent_id, request_context):
    """The child operation's result_metadata — where ``unit_ids_count`` lives."""
    status = await memory.get_operation_status(bank_id=bank_id, operation_id=parent_id, request_context=request_context)
    assert status["status"] == "completed", status
    children = status.get("child_operations") or []
    assert len(children) == 1, children
    child = await memory.get_operation_status(
        bank_id=bank_id, operation_id=children[0]["operation_id"], request_context=request_context
    )
    return child["result_metadata"]


async def _unit_ids(memory, bank_id, document_id, request_context):
    result = await memory.list_memory_units(
        bank_id, document_id=document_id, limit=100, request_context=request_context
    )
    return {u["id"] for u in result["items"]}


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_reprocess_reextracts_unchanged_content(memory, request_context):
    """The reported symptom: reprocess settled ``completed`` with nothing extracted."""
    bank_id = f"test_reprocess_force_{datetime.now(timezone.utc).timestamp()}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _CONTENT, "document_id": "doc-force"}],
            request_context=request_context,
        )
        before = await _unit_ids(memory, bank_id, "doc-force", request_context)
        assert before

        result = await memory.reprocess_document(
            bank_id=bank_id, document_id="doc-force", request_context=request_context
        )
        assert result is not None
        await asyncio.sleep(0.5)

        meta = await _child_metadata(memory, bank_id, result["operation_id"], request_context)
        assert meta["unit_ids_count"] > 0, f"reprocess was a silent no-op: {meta}"

        # A replace, not an append: the old units are gone and the new ones are new rows.
        after = await _unit_ids(memory, bank_id, "doc-force", request_context)
        assert after
        assert not (after & before), "reprocess preserved the existing units instead of re-extracting"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_http_reprocess_reextracts(api_client, memory, request_context):
    """Same through the endpoint the issue reports, which is how operators reach it."""
    bank_id = f"test_reprocess_http_{datetime.now(timezone.utc).timestamp()}"
    try:
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={"items": [{"content": _CONTENT, "document_id": "doc-http"}]},
        )
        assert response.status_code == 200
        before = await _unit_ids(memory, bank_id, "doc-http", request_context)
        assert before

        response = await api_client.post(f"/v1/default/banks/{bank_id}/documents/doc-http/reprocess")
        assert response.status_code == 200, response.text
        await asyncio.sleep(0.5)

        meta = await _child_metadata(memory, bank_id, response.json()["operation_id"], request_context)
        assert meta["unit_ids_count"] > 0, f"reprocess was a silent no-op: {meta}"
        after = await _unit_ids(memory, bank_id, "doc-http", request_context)
        assert not (after & before)
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_plain_re_retain_of_unchanged_content_still_skips(memory, request_context):
    """The other half of the fix: forcing must not leak into ordinary retains.

    A sync layer that re-pushes unchanged content relies on this skip — re-extracting
    every unchanged push is the cost the delta/recovery paths exist to avoid.
    """
    bank_id = f"test_reprocess_noforce_{datetime.now(timezone.utc).timestamp()}"
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": _CONTENT, "document_id": "doc-nochange"}],
            request_context=request_context,
        )
        before = await _unit_ids(memory, bank_id, "doc-nochange", request_context)
        assert before

        result = await memory.submit_async_retain(
            bank_id,
            [{"content": _CONTENT, "document_id": "doc-nochange"}],
            request_context=request_context,
        )
        await asyncio.sleep(0.5)

        meta = await _child_metadata(memory, bank_id, result["operation_id"], request_context)
        assert meta["unit_ids_count"] == 0, f"unchanged re-push re-extracted: {meta}"
        assert await _unit_ids(memory, bank_id, "doc-nochange", request_context) == before
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
