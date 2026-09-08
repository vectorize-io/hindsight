"""An imported observation must never cite a memory unit that does not exist.

``memory_units.source_memory_ids`` is a bare ``uuid[]`` with no foreign key, so
nothing at the database level stops the importer from writing a reference to a
unit that is absent from the destination bank. That corruption is invisible on
write and only surfaces much later, as an observation whose sources resolve to
nothing. The importer therefore re-checks liveness inside its write transaction.
"""

import uuid

import pytest

from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.schema import fq_table
from hindsight_api.engine.transfer import importer as importer_mod
from hindsight_api.engine.transfer.schema import TransferObservation, TransferObservationSource

from .test_document_transfer import _import, _retain, _seed_observation, _unique_bank, parse_archive

pytestmark = pytest.mark.asyncio


async def _import_one(memory, bank_id, ref_map, source_refs):
    """Run the observation import phase directly against a hand-built ref_map."""
    backend = await memory._get_backend()
    observation = TransferObservation(
        text=OBSERVATION_TEXT,
        sources=[TransferObservationSource(document_id=d, fact_index=i) for d, i in source_refs],
        proof_count=len(source_refs),
    )
    return await importer_mod._import_observations(
        backend=backend,
        embeddings_model=memory.embeddings,
        bank_id=bank_id,
        observations=[observation],
        ref_map=ref_map,
        ops=backend.ops,
    )


# Retain auto-consolidates, so the bank already holds observations of its own by
# the time these tests run. Match on the text this test imports to look at only
# the observation under test. Direct SQL because source_memory_ids is internal
# consolidation state that no public read method exposes.
OBSERVATION_TEXT = "Alice and Bob are colleagues."


async def _imported_observation_sources(backend, bank_id):
    async with acquire_with_retry(backend) as conn:
        rows = await conn.fetch(
            f"SELECT source_memory_ids FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type = 'observation' AND text = $2",
            bank_id,
            OBSERVATION_TEXT,
        )
    return [{str(sid) for sid in (row["source_memory_ids"] or [])} for row in rows]


async def test_observation_with_a_missing_source_is_skipped(memory, request_context):
    """A ref that resolves to an id no longer in the bank must not be written."""
    bank = _unique_bank("obs_liveness_missing")
    created = await _retain(memory, bank, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
    live_id = str(created[0])
    ghost_id = str(uuid.uuid4())  # resolves, but no such row exists

    outcome = await _import_one(
        memory,
        bank,
        ref_map={("doc-1", 0): live_id, ("doc-1", 1): ghost_id},
        source_refs=[("doc-1", 0), ("doc-1", 1)],
    )

    assert outcome.imported == 0
    assert outcome.skipped == 1
    backend = await memory._get_backend()
    assert await _imported_observation_sources(backend, bank) == []


async def test_observation_with_live_sources_is_imported(memory, request_context):
    """The guard must not reject the ordinary case it is wrapped around."""
    bank = _unique_bank("obs_liveness_ok")
    created = await _retain(memory, bank, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
    assert len(created) >= 2
    live_ids = [str(i) for i in created[:2]]

    outcome = await _import_one(
        memory,
        bank,
        ref_map={("doc-1", 0): live_ids[0], ("doc-1", 1): live_ids[1]},
        source_refs=[("doc-1", 0), ("doc-1", 1)],
    )

    assert outcome.imported == 1
    assert outcome.skipped == 0
    backend = await memory._get_backend()
    assert await _imported_observation_sources(backend, bank) == [set(live_ids)]


@pytest.mark.asyncio
async def test_retain_replacing_the_document_mid_import_is_not_cited(memory, request_context, monkeypatch):
    """A concurrent replace of the document must not leave observations citing it.

    Import commits its documents one at a time and writes the observations later,
    in a separate transaction. A retain of the same document_id into the same bank
    takes the full-replace path and deletes that document's memory_units. Landing
    in that window used to produce observations citing deleted units; the liveness
    check turns it into a skip.

    The interleaving is forced rather than raced -- the deleter is a real retain
    through the public API, scheduled at the point the window is open.
    """
    backend = await memory._get_backend()
    src = _unique_bank(f"race_src_{uuid.uuid4().hex[:6]}")
    dst = _unique_bank(f"race_dst_{uuid.uuid4().hex[:6]}")

    created = await _retain(memory, src, "Alice works at Google. Bob works at Microsoft.", request_context, "doc-1")
    assert len(created) >= 2
    src_ids = [uuid.UUID(str(i)) for i in created[:2]]
    await _seed_observation(
        pool=backend,
        memory=memory,
        bank_id=src,
        source_memory_ids=src_ids,
        observation_text="Alice and Bob are colleagues.",
    )
    archive = await memory.export_documents_async(src, request_context, include_observations=True)
    assert len(parse_archive(archive).observations) >= 1

    # Open the window: the documents are committed, the observations are not yet
    # written. A concurrent retain of the same document replaces it, deleting the
    # units the pending observations are about to cite.
    original = importer_mod._import_observations
    fired = []

    async def _deleting_import_observations(**kwargs):
        if not fired:
            fired.append(True)
            await _retain(memory, dst, "Completely different content.", request_context, "doc-1")
        return await original(**kwargs)

    monkeypatch.setattr(importer_mod, "_import_observations", _deleting_import_observations)
    await _import(memory, dst, archive, request_context)
    assert fired, "the window never opened"

    async with acquire_with_retry(backend) as conn:
        rows = await conn.fetch(
            f"SELECT id, source_memory_ids FROM {fq_table('memory_units')} "
            f"WHERE bank_id = $1 AND fact_type = 'observation'",
            dst,
        )
        dangling = []
        for r in rows:
            for sid in r["source_memory_ids"] or []:
                if not await conn.fetchval(f"SELECT 1 FROM {fq_table('memory_units')} WHERE id = $1", sid):
                    dangling.append((str(r["id"])[:8], str(sid)[:8]))
    assert not dangling, f"observation cites deleted units: {dangling}"
