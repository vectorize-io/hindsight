"""A retain that fails after the API stored its attachments must not leave them behind (#6350).

The ingress stores a retain's images and files before the retain runs. When the retain then
fails — an LLM error, a dropped connection, a worker giving up — it creates no document, and
nothing else ever removes them. The daily maintenance sweep does: an attachment its document does
not reference once its retain has had time to finish — the document was never created, or the
update that brought the attachment failed — belongs to nothing.

Driven per bank through the engine, not through the maintenance loop: the loop sweeps every
bank in the database, and other tests running alongside hold attachments mid-retain.
"""

import base64
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hindsight_api.engine.memories import set_memories
from hindsight_api.engine.retain.attachment_content import compute_attachment_hash, short_attachment_id
from tests.test_attachments_store_owned import _CarryingStore

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_ID = short_attachment_id(compute_attachment_hash(PNG))


def _item(document_id: str) -> dict:
    image = {"type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}
    return {
        "content": [{"type": "text", "text": "The reset button:"}, {"type": "image", "source": image}],
        "document_id": document_id,
    }


async def _retain(api_client, bank_id: str, document_id: str):
    return await api_client.post(
        f"/v1/default/banks/{bank_id}/memories", json={"items": [_item(document_id)], "async": False}
    )


async def _failed_retain(api_client, memory, monkeypatch, bank_id: str, document_id: str) -> None:
    async def llm_timeout(*args, **kwargs):
        raise RuntimeError("LLM timed out")

    with monkeypatch.context() as m:
        m.setattr(memory, "_run_retain_execution", llm_timeout)
        response = await _retain(api_client, bank_id, document_id)
    assert response.status_code >= 500, response.text


async def _sweep(memory, bank_id: str, request_context, *, grace: timedelta = timedelta(seconds=-1)) -> int:
    """Sweep one bank as the daily job would, with ``grace`` in place of its day."""
    created_before = datetime.now(UTC) - grace
    return await memory.reclaim_unreferenced_attachments(
        bank_id, created_before - timedelta(days=7), created_before, request_context
    )


async def _attachment_status(api_client, bank_id: str) -> int:
    return (await api_client.get(f"/v1/default/banks/{bank_id}/attachments/{PNG_ID}")).status_code


@pytest.mark.asyncio
async def test_the_sweep_deletes_the_attachment_of_a_retain_that_failed(
    api_client, memory, request_context, monkeypatch
):
    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    await _failed_retain(api_client, memory, monkeypatch, bank_id, "d1")
    assert await _attachment_status(api_client, bank_id) == 200, "the failed retain left its image behind"

    # Inside the grace period the retain may still be running, or about to be retried.
    assert await _sweep(memory, bank_id, request_context, grace=timedelta(hours=1)) == 0
    assert await _attachment_status(api_client, bank_id) == 200

    assert await _sweep(memory, bank_id, request_context) == 1
    assert await _attachment_status(api_client, bank_id) == 404


@pytest.mark.asyncio
async def test_the_sweep_keeps_the_attachments_of_a_document_that_exists(api_client, memory, request_context):
    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    response = await _retain(api_client, bank_id, "d1")
    assert response.status_code == 200, response.text

    assert await _sweep(memory, bank_id, request_context) == 0
    assert await _attachment_status(api_client, bank_id) == 200


@pytest.mark.asyncio
async def test_the_sweep_deletes_the_attachment_of_a_failed_update(api_client, memory, request_context, monkeypatch):
    """The document exists, but with its old text: the image the failed update brought is in none of it."""
    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    first = await api_client.post(
        f"/v1/default/banks/{bank_id}/memories", json={"items": [{"content": "No image yet.", "document_id": "d1"}]}
    )
    assert first.status_code == 200, first.text
    await _failed_retain(api_client, memory, monkeypatch, bank_id, "d1")
    assert await _attachment_status(api_client, bank_id) == 200, "the failed update left its image behind"

    assert await _sweep(memory, bank_id, request_context) == 1
    assert await _attachment_status(api_client, bank_id) == 404
    document = await api_client.get(f"/v1/default/banks/{bank_id}/documents/d1")
    assert document.status_code == 200, "the document itself must be left alone"


@pytest.mark.asyncio
async def test_the_sweep_waits_for_a_retain_still_queued(api_client, memory, request_context, monkeypatch):
    """A queued retain's document does not exist yet: its attachments are not orphans."""
    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    await _failed_retain(api_client, memory, monkeypatch, bank_id, "d1")
    # Forged rather than submitted: the point is a retain that has not run yet, and a real
    # submission cannot be held in the queue on purpose.
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        await conn.execute(
            "INSERT INTO async_operations (operation_id, bank_id, operation_type, status) "
            "VALUES ($1, $2, 'retain', 'pending')",
            uuid.uuid4(),
            bank_id,
        )

    assert await _sweep(memory, bank_id, request_context) == 0
    assert await _attachment_status(api_client, bank_id) == 200


@pytest.mark.asyncio
async def test_a_store_owned_bank_is_answered_by_its_store(api_client, memory, request_context, monkeypatch):
    """Its documents are not in SQL, so only the store can say which ones exist."""
    set_memories(_CarryingStore(answers_full_recall=True))
    try:
        bank_id = f"swp-so-{uuid.uuid4().hex[:8]}"
        assert (await api_client.put(f"/v1/default/banks/{bank_id}", json={})).status_code == 200
        config = await api_client.patch(
            f"/v1/default/banks/{bank_id}/config", json={"updates": {"retain_extraction_mode": "chunks"}}
        )
        assert config.status_code == 200, config.text
        landed = await _retain(api_client, bank_id, "landed")
        assert landed.status_code == 200, landed.text
        await _failed_retain(api_client, memory, monkeypatch, bank_id, "failed")

        assert await _sweep(memory, bank_id, request_context) == 1

        backend = await memory._get_backend()
        async with backend.acquire() as conn:
            rows = await conn.fetch("SELECT document_id FROM attachments WHERE bank_id = $1", bank_id)
        assert [row["document_id"] for row in rows] == ["landed"]
    finally:
        set_memories(None)


@pytest.mark.asyncio
async def test_the_daily_job_finds_the_bank_through_the_routine(api_client, memory, monkeypatch):
    """One cross-tenant call finds the bank; the engine sweep then runs for it.

    The engine's sweep is spied rather than run: the job sees every bank in the test
    database, and other tests running alongside hold attachments mid-retain.
    """
    import hindsight_api.engine.maintenance as maintenance_mod
    from hindsight_api.engine.maintenance import MaintenanceLoop

    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    await _failed_retain(api_client, memory, monkeypatch, bank_id, "d1")

    visited: list[str] = []

    async def spy(bank, created_after, created_before, request_context):
        visited.append(bank)
        return 0

    monkeypatch.setattr(maintenance_mod, "ATTACHMENT_SWEEP_GRACE", timedelta(seconds=-1))
    monkeypatch.setattr(memory, "reclaim_unreferenced_attachments", spy)
    await MaintenanceLoop(memory)._run_attachment_sweep()

    assert bank_id in visited


@pytest.mark.asyncio
async def test_repair_bank_collects_what_leaked_before_the_daily_window(api_client, memory, monkeypatch, pg0_db_url):
    """``repair-bank --attachments`` applies the same rule with no lower bound.

    The daily sweep only looks back a week, so an image a retain leaked a month ago —
    before the sweep existed — is never reached by it. The admin repair is how an
    upgraded deployment collects those once.
    """
    from hindsight_api.admin.cli import _run_repair_bank

    bank_id = f"swp-{uuid.uuid4().hex[:8]}"
    landed = await _retain(api_client, bank_id, "kept")
    assert landed.status_code == 200, landed.text
    await _failed_retain(api_client, memory, monkeypatch, bank_id, "leaked")
    backend = await memory._get_backend()
    async with backend.acquire() as conn:
        # Aged by hand: a month-old leak is the case under test, and no API sets created_at.
        await conn.execute("UPDATE attachments SET created_at = now() - interval '30 days' WHERE bank_id = $1", bank_id)

    dry = await _run_repair_bank(
        pg0_db_url, base_schema="public", schema="public", bank_id=bank_id, dry_run=True, attachments=True
    )
    assert dry.unreferenced_attachments == 1
    assert await _attachment_status(api_client, bank_id) == 200, "a dry run must remove nothing"

    swept = await _run_repair_bank(
        pg0_db_url, base_schema="public", schema="public", bank_id=bank_id, dry_run=False, attachments=True
    )
    assert swept.unreferenced_attachments == 1
    async with backend.acquire() as conn:
        rows = await conn.fetch("SELECT document_id FROM attachments WHERE bank_id = $1", bank_id)
    assert [row["document_id"] for row in rows] == ["kept"]

    # Without the flag the repair leaves attachments alone.
    plain = await _run_repair_bank(pg0_db_url, base_schema="public", schema="public", bank_id=bank_id, dry_run=True)
    assert plain.unreferenced_attachments == 0
