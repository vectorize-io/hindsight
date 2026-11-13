"""A knowledge bank holds authoritative files: ingest them, search them, replace and delete them.

Unlike a memory bank nothing here calls an LLM. A file becomes passages, the passages are embedded
(here by the stub's lexical embeddings, the memory banks' embedder), and search ranks them by
meaning, best first. The story walks one bank through that whole life,
through the HTTP API alone: the parts that only meet here are the sync and async ingest paths, the
worker, search over what they wrote, and the replace and delete paths that must leave search
consistent.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from collections.abc import AsyncIterator

import pytest

from hindsight_system_tests import KnowledgeBanks, UploadFile

pytestmark = pytest.mark.asyncio

GUIDES = {
    "keys": "Rotate every api key each month from the security settings page.",
    "billing": "Invoices are emailed on the first day of each month with a PDF receipt.",
    "outage": "During an outage the status page shows code ZX-4417 until service returns.",
}


@pytest.fixture
async def kb(client) -> AsyncIterator[KnowledgeBanks]:
    yield KnowledgeBanks(client)


@pytest.fixture
async def knowledge_bank(kb: KnowledgeBanks) -> AsyncIterator[str]:
    bank_id = f"systest-kb-{uuid.uuid4().hex[:12]}"
    r = await kb.create(bank_id)
    assert r.status == 200, r.body
    yield bank_id
    await kb.delete(bank_id)


async def _hits(kb: KnowledgeBanks, bank_id: str, first: int | None = None, **body) -> list[str]:
    """The documents of the hits, best first; ``first`` keeps that many."""
    r = await kb.search(bank_id, **body)
    assert r.status == 200, r.body
    return [h["document_id"] for h in r.body["results"]][:first]


async def test_a_sync_ingest_returns_once_every_document_is_searchable(kb, knowledge_bank):
    items = [{"document_id": d, "text": t} for d, t in GUIDES.items()]
    r = await kb.ingest(knowledge_bank, items)

    assert r.status == 200, r.body
    assert r.body == {"document_ids": ["keys", "billing", "outage"], "operation_ids": [], "unchanged_document_ids": []}
    # Searchable on return: no waiting, no polling. The whole hit is deterministic with the stub embedder.
    r = await kb.search(knowledge_bank, query="how do I rotate an api key")
    hit = r.body["results"][0]
    # A one-passage document: its text is the passage itself, with nothing around it.
    assert {k: hit[k] for k in ("document_id", "index", "text", "media_type", "page")} == {
        "document_id": "keys",
        "index": 0,
        "text": GUIDES["keys"],
        "media_type": "text",
        "page": None,
    }
    assert hit["filename"] == "keys.md" and "score" not in hit and set(r.body) == {"results"}
    assert (await _hits(kb, knowledge_bank, query="when are invoices emailed", first=1)) == ["billing"]
    bank = (await kb.get(knowledge_bank)).body
    assert (bank["document_count"], bank["ready_count"], bank["failed_count"]) == (3, 3, 0)


async def test_an_async_ingest_is_one_operation_the_worker_finishes_even_split_into_children(
    client, kb, knowledge_bank
):
    # Over a child's byte budget (1 MiB by default): the notes fill one child and the manual, bigger than
    # the budget on its own, is a second, so the worker runs the upload as two children. The one operation
    # the caller holds completes only once both have. (A request carries at most 100 items, under the
    # 100 documents a child holds, so only bytes can split a JSON ingest.)
    items = [{"document_id": f"note-{n}", "text": f"note {n} about the red pipeline"} for n in range(30)]
    items.append({"document_id": "manual", "text": "Unrelated filler words. " * 46_000})
    r = await kb.ingest(knowledge_bank, items, wait=False)

    assert r.status == 202, r.body
    # One operation for the whole batch, named for every document it carries.
    assert len(r.body["operation_ids"]) == 31 and len(set(r.body["operation_ids"])) == 1
    [parent_id] = set(r.body["operation_ids"])
    await kb.wait_until_processed(knowledge_bank, 31, timeout=60.0)
    page = (await kb.documents(knowledge_bank, status="ready", limit=1)).body
    assert page["total"] == 31
    assert (await _hits(kb, knowledge_bank, query="note 7", first=1)) == ["note-7"]
    # The children are only visible as the bank's operations: the parent plus two of them.
    ops = (await client.operations.list_operations(knowledge_bank, limit=100)).operations
    ingests = [op for op in ops if op.task_type == "knowledge_ingest"]
    assert len(ingests) == 3 and parent_id in {op.id for op in ingests}, ingests
    deadline = asyncio.get_running_loop().time() + 30
    while (status := await client.operations.get_operation_status(knowledge_bank, parent_id)).status != "completed":
        assert status.status in ("pending", "processing") and asyncio.get_running_loop().time() < deadline, status
        await asyncio.sleep(0.2)


async def test_a_text_hit_comes_back_with_the_passages_around_it_as_the_budget_allows(kb, knowledge_bank):
    sections = [f"# Step {n}\n\n" + ("Unrelated filler words. " * 12) for n in range(6)]
    sections[3] = "# Step 3\n\nOpen the valve slowly until the gauge reads green."
    await kb.ingest(knowledge_bank, [{"document_id": "manual", "text": "\n\n".join(sections)}])

    r = await kb.search(knowledge_bank, query="open the valve until the gauge reads green")
    hit = r.body["results"][0]
    assert hit["document_id"] == "manual" and "Open the valve slowly" in hit["text"]
    # The whole manual fits in the default budget: the hit's text grows to take it all in, once.
    assert "Step 0" in hit["text"] and "Step 5" in hit["text"] and "context" not in hit
    assert [h["document_id"] for h in r.body["results"]].count("manual") == 1


async def test_reingesting_skips_what_is_unchanged_and_replaces_what_is_not(kb, knowledge_bank):
    await kb.ingest(knowledge_bank, [{"document_id": d, "text": t} for d, t in GUIDES.items()])

    again = await kb.ingest(knowledge_bank, [{"document_id": "keys", "text": GUIDES["keys"]}])
    assert again.body["unchanged_document_ids"] == ["keys"]

    replaced = await kb.ingest(
        knowledge_bank, [{"document_id": "keys", "text": "Api keys never expire; revoke a leaked one at once."}]
    )
    assert replaced.status == 200 and replaced.body["unchanged_document_ids"] == []
    detail = (await kb.document(knowledge_bank, "keys")).body
    assert [p["text"] for p in detail["passages"]] == ["Api keys never expire; revoke a leaked one at once."]
    # Search sees the new text only: the old passages are gone, not shadowed.
    assert "keys" not in await _hits(kb, knowledge_bank, query="rotate each month security settings page", first=1)
    assert (await _hits(kb, knowledge_bank, query="revoke a leaked api key", first=1)) == ["keys"]


async def test_a_banks_own_passage_size_splits_what_it_ingests_and_a_reset_puts_the_servers_back(kb, knowledge_bank):
    # The config, the ingest that reads it and the content hash that remembers it meet only here: a
    # smaller passage size must re-split the same text on its next upload, not skip it as unchanged.
    text = "\n\n".join(f"Section {n}. " + ("Unrelated filler words. " * 20) for n in range(8))
    await kb.ingest(knowledge_bank, [{"document_id": "manual", "text": text}])
    default = (await kb.document(knowledge_bank, "manual")).body["passage_count"]
    before = (await kb.config(knowledge_bank)).body
    assert before["overrides"] == {} and "provider" not in before["config"]

    r = await kb.update_config(knowledge_bank, {"text_passage_tokens": 32})
    assert r.status == 200, r.body
    assert r.body["config"]["text_passage_tokens"] == 32 and r.body["overrides"] == {"text_passage_tokens": 32}
    again = await kb.ingest(knowledge_bank, [{"document_id": "manual", "text": text}])
    assert again.body["unchanged_document_ids"] == []
    smaller = (await kb.document(knowledge_bank, "manual")).body["passage_count"]
    assert smaller > default, (smaller, default)

    reset = await kb.reset_config(knowledge_bank)
    assert reset.status == 200 and reset.body == before
    assert (await kb.update_config(knowledge_bank, {"embeddings_model": "x"})).status == 400


async def test_one_bad_document_fails_alone(kb, knowledge_bank):
    r = await kb.ingest(
        knowledge_bank,
        [
            {"document_id": "good", "text": "The red pipeline deploys at noon."},
            {"document_id": "empty", "text": " \n "},
        ],
    )

    assert r.status == 400 and "1 of 2 documents failed; 'empty'" in r.body["detail"]
    assert (await kb.document(knowledge_bank, "good")).body["status"] == "ready"
    failed = (await kb.document(knowledge_bank, "empty")).body
    assert failed["status"] == "failed" and "nothing to search" in failed["error"]


async def test_one_document_twice_in_a_request_is_refused_before_anything_is_stored(kb, knowledge_bank):
    twice = [{"document_id": "x", "text": "a"}, {"document_id": "x", "text": "b"}]
    r = await kb.ingest(knowledge_bank, twice)
    assert r.status == 400 and "more than once" in r.body["detail"]
    assert (await kb.document(knowledge_bank, "x")).status == 404


async def test_files_are_ingested_and_served_back(kb, knowledge_bank):
    pdf = _text_pdf("Emergency exits are on floor nine")
    r = await kb.ingest_files(
        knowledge_bank,
        [UploadFile("exits.pdf", pdf), UploadFile("faq.md", b"# FAQ\n\nThe cafeteria opens at eight.")],
        [{"document_id": "exits", "tags": ["safety"]}, {"document_id": "faq"}],
    )
    assert r.status == 202, r.body
    await kb.wait_until_processed(knowledge_bank, 2)

    exits = (await kb.document(knowledge_bank, "exits")).body
    assert (exits["status"], exits["media_type"], exits["tags"]) == ("ready", "document", ["safety"])
    assert exits["passages"][0]["page"] == 1
    # The text-only embedder found the page by its text layer.
    assert (await _hits(kb, knowledge_bank, query="where are the emergency exits", first=1)) == ["exits"]
    assert (await _hits(kb, knowledge_bank, query="emergency exits", tags=["safety"])) == ["exits"]
    assert (await kb.file(knowledge_bank, "exits")).raw == pdf


async def test_deleting_a_document_removes_it_from_search(kb, knowledge_bank):
    await kb.ingest(knowledge_bank, [{"document_id": d, "text": t} for d, t in GUIDES.items()])

    assert (await kb.delete_document(knowledge_bank, "billing")).status == 200
    assert "billing" not in await _hits(kb, knowledge_bank, query="when are invoices emailed")
    assert (await kb.document(knowledge_bank, "billing")).status == 404
    assert (await kb.get(knowledge_bank)).body["document_count"] == 2
    assert (await kb.delete_document(knowledge_bank, "billing")).status == 404


def _text_pdf(text: str) -> bytes:
    """A one-page PDF with a real text layer, built by hand: the suite has no PDF library."""
    stream = f"BT /F1 24 Tf 40 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>stream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    out.writelines(b"%010d 00000 n \n" % o for o in offsets)
    out.write(b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objects) + 1, xref))
    return out.getvalue()
