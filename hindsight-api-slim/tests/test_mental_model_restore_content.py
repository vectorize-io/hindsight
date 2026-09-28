"""Restore a mental model's content through the public API (#4861).

A bad refresh could leave a mental model holding garbage while ``GET
.../history`` still returned every prior version's ``previous_content`` —
the data to undo the damage was stored, but no API path could write it back:
``PATCH`` accepted only name/source_query/max_tokens/tags/trigger, and there
was no restore/revert route. The only way back was another refresh (the very
operation that failed) or direct database access.

The fix is the smallest one that closes the gap: ``PATCH`` gains a ``content``
field and threads it to the engine's existing ``update_mental_model(content=)``
path, which already derives the structured document from markdown (#3361's
invariant that the two columns never diverge), re-embeds the full document
(#3926), and snapshots the overwritten version into history — so a restore is
itself undoable through the same history.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def client(memory):
    import httpx

    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def bank(memory):
    bank_id = f"mm-restore-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id, request_context=RequestContext())
    return bank_id


async def _model(memory, bank_id: str, content: str) -> dict:
    return await memory.create_mental_model(
        bank_id=bank_id,
        name="Team Info",
        source_query="Tell me about the team",
        content=content,
        request_context=RequestContext(),
    )


GOOD = "# Team\n\nAlice leads the platform team. Bob owns data ingest.\n"
BAD = "OK"


@pytest.mark.asyncio
async def test_patch_accepts_content_and_stores_it(client, bank, memory):
    """The reported gap: PATCH with {"content": ...} is no longer silently ignored."""
    mm = await _model(memory, bank, GOOD)
    resp = await client.patch(f"/v1/default/banks/{bank}/mental-models/{mm['id']}", json={"content": BAD})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["content"].strip() == BAD, "the content field was dropped by the request model"

    stored = await memory.get_mental_model(bank, mm["id"], request_context=RequestContext())
    assert stored is not None and stored["content"].strip() == BAD


@pytest.mark.asyncio
async def test_patch_content_snapshots_the_overwritten_version_into_history(client, bank, memory):
    """Restoring is itself undoable: the bad version lands in history as previous_content."""
    mm = await _model(memory, bank, GOOD)

    resp = await client.patch(f"/v1/default/banks/{bank}/mental-models/{mm['id']}", json={"content": BAD})
    assert resp.status_code == 200, resp.text

    history = await memory.get_mental_model_history(bank, mm["id"], request_context=RequestContext())
    assert len(history) == 1
    assert history[0]["previous_content"].strip() == GOOD.strip()


@pytest.mark.asyncio
async def test_restore_workflow_end_to_end(client, bank, memory):
    """The issue's scenario, driven only through the public API: bad write, then
    read the good version back from history, then PATCH it in."""
    mm = await _model(memory, bank, GOOD)

    # A refresh (or any content write) leaves a bad document behind.
    bad = await client.patch(f"/v1/default/banks/{bank}/mental-models/{mm['id']}", json={"content": BAD})
    assert bad.status_code == 200, bad.text

    # The operator reads history through the API, exactly as the issue describes.
    hist = await client.get(f"/v1/default/banks/{bank}/mental-models/{mm['id']}/history")
    assert hist.status_code == 200, hist.text
    entries = hist.json()
    assert entries, "the overwrite left no history to restore from"
    good_version = entries[0]["previous_content"]
    assert good_version is not None and good_version.strip() == GOOD.strip()

    # And writes it back.
    restore = await client.patch(f"/v1/default/banks/{bank}/mental-models/{mm['id']}", json={"content": good_version})
    assert restore.status_code == 200, restore.text
    assert restore.json()["content"] == GOOD

    final = await memory.get_mental_model(bank, mm["id"], request_context=RequestContext())
    assert final["content"] == GOOD, "the restored document did not survive verbatim"


@pytest.mark.asyncio
async def test_patch_without_content_still_leaves_content_alone(client, bank, memory):
    """A metadata-only PATCH must not touch the document (regression guard)."""
    mm = await _model(memory, bank, GOOD)
    resp = await client.patch(f"/v1/default/banks/{bank}/mental-models/{mm['id']}", json={"name": "Renamed"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "Renamed"
    assert resp.json()["content"].strip() == GOOD.strip()


@pytest.mark.asyncio
async def test_patch_content_on_missing_model_is_404(client, bank):
    resp = await client.patch(f"/v1/default/banks/{bank}/mental-models/no-such-id", json={"content": "x"})
    assert resp.status_code == 404
