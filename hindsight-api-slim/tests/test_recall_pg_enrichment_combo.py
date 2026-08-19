"""Recall enrichment combinations on the DEFAULT (Postgres) store.

The seam tests in ``test_memories_extension.py`` prove a store that owns its rows
and bodies hydrates chunks / source facts / entities through the interface. This
file is the mirror on the default path: one recall with all three flags on
``PostgresMemories``, so the combination that had no coverage before is locked on
the store every deployment actually runs.

Everything is seeded directly through SQL (no LLM) so the assertions are exact:
- a raw ``world`` fact carrying a ``chunk_id`` (with real ``chunk_text`` in the
  SQL ``chunks`` row) and a ``unit_entities`` link to a named entity;
- an ``observation`` consolidated from that fact via ``source_memory_ids``.

With ``prefer_observations=True`` the raw fact is dropped from the results, so all
three enrichments have to reach it transitively: chunks via the observation's
source, source_facts from the source row, entities inherited from the source.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api.engine.retain import embedding_utils


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


@pytest_asyncio.fixture
async def seeded_combo(memory, request_context):
    bank_id = f"test-pg-enrich-combo-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id, request_context=request_context)

    fact_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    chunk_id = f"chunk-{uuid.uuid4().hex[:8]}"
    chunk_text = "Alice migrated the billing service to the new cluster over the weekend."
    fact_text = "Alice migrated the billing service to the new cluster"
    obs_text = "Alice handles infrastructure migrations"

    embeddings = await embedding_utils.generate_embeddings_batch(memory.embeddings, [fact_text, obs_text])

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash) VALUES ($1, $2, $3, 'h')",
            doc_id,
            bank_id,
            chunk_text,
        )
        await conn.execute(
            "INSERT INTO chunks (chunk_id, document_id, bank_id, chunk_index, chunk_text) VALUES ($1, $2, $3, 0, $4)",
            chunk_id,
            doc_id,
            bank_id,
            chunk_text,
        )
        ent_id = await conn.fetchval(
            "INSERT INTO entities (bank_id, canonical_name, mention_count) VALUES ($1, $2, 1) RETURNING id",
            bank_id,
            "billing service",
        )
        # Raw source fact: carries the chunk and the entity link.
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, document_id, chunk_id) "
            "VALUES ($1, $2, $3, 'world', $4::vector, now(), $5, $6)",
            fact_id,
            bank_id,
            fact_text,
            _to_str(embeddings[0]),
            doc_id,
            chunk_id,
        )
        await conn.execute(
            "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
            fact_id,
            ent_id,
        )
        # Observation consolidated from the raw fact.
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, source_memory_ids, "
            "proof_count) VALUES ($1, $2, $3, 'observation', $4::vector, now(), $5::uuid[], 1)",
            obs_id,
            bank_id,
            obs_text,
            _to_str(embeddings[1]),
            [fact_id],
        )

    yield {"bank_id": bank_id, "fact_id": fact_id, "obs_id": obs_id, "chunk_id": chunk_id, "chunk_text": chunk_text}

    await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_recall_all_enrichments_together_on_default_store(memory, request_context, seeded_combo):
    """All three enrichment flags at once on PostgresMemories, with prefer_observations."""
    bank_id = seeded_combo["bank_id"]
    fact_id = seeded_combo["fact_id"]
    obs_id = seeded_combo["obs_id"]
    chunk_id = seeded_combo["chunk_id"]

    result = await memory.recall_async(
        bank_id=bank_id,
        query="Alice billing service migration",
        fact_type=["world", "observation"],
        max_tokens=4096,
        prefer_observations=True,
        include_chunks=True,
        max_chunk_tokens=2000,
        include_source_facts=True,
        include_entities=True,
        max_entity_tokens=2000,
        request_context=request_context,
    )

    by_id = {str(r.id): r for r in result.results}
    assert obs_id in by_id, f"observation missing from results {list(by_id)}"
    assert fact_id not in by_id, "prefer_observations should have dropped the superseded raw fact"

    # chunks: resolved via the observation's source fact, text straight from the SQL chunks row
    assert result.chunks and chunk_id in result.chunks
    assert result.chunks[chunk_id].chunk_text == seeded_combo["chunk_text"]

    # source_facts: the raw fact the observation was built from
    assert result.source_facts and fact_id in result.source_facts
    assert result.source_facts[fact_id].text == "Alice migrated the billing service to the new cluster"
    assert by_id[obs_id].source_fact_ids == [fact_id]

    # entities: inherited by the observation from its source through source_memory_ids
    assert by_id[obs_id].entities and "billing service" in by_id[obs_id].entities
    assert result.entities and "billing service" in result.entities
