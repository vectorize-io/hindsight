"""Tests for memory curation: edit / invalidate / revert.

Invalidation MOVES a fact out of ``memory_units`` into the
``invalidated_memory_units`` archive, so the recall hot-path never sees it.
These tests cover the move semantics, lossless revert (incl. entity
associations), edit, the guards, listing, and recall exclusion.
"""

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_memory(
    conn,
    memory: MemoryEngine,
    bank_id: str,
    text: str,
    fact_type: str = "experience",
    metadata: dict | None = None,
) -> uuid.UUID:
    """Insert a live memory unit with a real embedding, bypassing the LLM pipeline."""
    mem_id = uuid.uuid4()
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    await conn.execute(
        """
        INSERT INTO memory_units (
            id, bank_id, text, fact_type, embedding, event_date, metadata, created_at, updated_at, consolidated_at
        )
        VALUES ($1, $2, $3, $4, $5::vector, NOW(), $6::jsonb, NOW(), NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
        str(emb[0]),
        json.dumps(metadata or {}),
    )
    return mem_id


async def _insert_observation(conn, bank_id: str, text: str, source_memory_ids: list[uuid.UUID]) -> uuid.UUID:
    obs_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO memory_units (
            id, bank_id, text, fact_type, event_date, source_memory_ids, proof_count, created_at, updated_at
        ) VALUES ($1, $2, $3, 'observation', NOW(), $4, $5, NOW(), NOW())
        """,
        obs_id,
        bank_id,
        text,
        source_memory_ids,
        len(source_memory_ids),
    )
    return obs_id


async def _insert_link(conn, bank_id: str, from_id: uuid.UUID, to_id: uuid.UUID) -> None:
    await conn.execute(
        """
        INSERT INTO memory_links (from_unit_id, to_unit_id, link_type, weight, bank_id)
        VALUES ($1, $2, 'temporal', 0.5, $3)
        """,
        from_id,
        to_id,
        bank_id,
    )


async def _insert_entity(conn, bank_id: str, name: str) -> uuid.UUID:
    eid = uuid.uuid4()
    await conn.execute(
        "INSERT INTO entities (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
        eid,
        bank_id,
        name,
    )
    return eid


async def _link_entity(conn, unit_id: uuid.UUID, entity_id: uuid.UUID) -> None:
    await conn.execute(
        "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1, $2)",
        unit_id,
        entity_id,
    )


async def _in_live(conn, mem_id: uuid.UUID) -> bool:
    return bool(await conn.fetchval("SELECT 1 FROM memory_units WHERE id = $1", mem_id))


async def _archive_row(conn, mem_id: uuid.UUID) -> dict | None:
    # No `embedding` column: the archive is cold storage and the schema drops it (#2209).
    row = await conn.fetchrow(
        "SELECT text, invalidation_reason, invalidated_at, entity_ids FROM invalidated_memory_units WHERE id = $1",
        mem_id,
    )
    return dict(row) if row else None


async def _archive_has_column(conn, column: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'invalidated_memory_units' AND column_name = $1",
            column,
        )
    )


async def _link_count(conn, mem_id: uuid.UUID) -> int:
    return await conn.fetchval(
        "SELECT COUNT(*) FROM memory_links WHERE from_unit_id = $1 OR to_unit_id = $1",
        mem_id,
    )


async def _entity_ids_for(conn, unit_id: uuid.UUID) -> list[uuid.UUID]:
    rows = await conn.fetch("SELECT entity_id FROM unit_entities WHERE unit_id = $1", unit_id)
    return [r["entity_id"] for r in rows]


async def _obs_ids(conn, bank_id: str) -> list[str]:
    rows = await conn.fetch(
        "SELECT id FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
        bank_id,
    )
    return [str(r["id"]) for r in rows]


async def _consolidated_at(conn, mem_id: uuid.UUID):
    return await conn.fetchval("SELECT consolidated_at FROM memory_units WHERE id = $1", mem_id)


async def _ensure_bank(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Invalidate / revert (table move)
# ---------------------------------------------------------------------------


class TestInvalidate:
    @pytest.mark.asyncio
    async def test_invalidate_moves_to_archive_and_prunes(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-inv-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "The deploy server srv-04 runs PostgreSQL 14.")
            m2 = await _insert_memory(conn, memory, bank_id, "srv-04 is in the eu-west datacenter.")
            obs_id = await _insert_observation(conn, bank_id, "srv-04 runs PG14 in eu-west.", [m1, m2])
            await _insert_link(conn, bank_id, m1, m2)
            await conn.execute(
                "UPDATE memory_units SET observation_scopes = $1::jsonb WHERE id = $2",
                json.dumps("shared"),
                m1,
            )

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            result = await memory.update_memory_unit(
                bank_id, str(m1), state="invalidated", reason="decommissioned", request_context=request_context
            )

        assert result is not None
        assert result["state"] == "invalidated"
        assert result["invalidation_reason"] == "decommissioned"
        assert result["invalidated_at"] is not None
        assert result["observation_scopes"] == "shared"

        async with pool.acquire() as conn:
            assert not await _in_live(conn, m1), "invalidated row must leave memory_units"
            arch = await _archive_row(conn, m1)
            assert arch is not None, "row must be in the archive"
            assert arch["invalidation_reason"] == "decommissioned"
            assert not await _archive_has_column(conn, "embedding"), (
                "archive is cold storage; the schema drops the embedding column (#2209)"
            )
            assert not await _archive_has_column(conn, "search_vector"), (
                "archive is cold storage with no index; the schema drops search_vector (#2503)"
            )
            assert await _link_count(conn, m1) == 0, "links cascade-pruned on move"
            assert str(obs_id) not in await _obs_ids(conn, bank_id), "derived observation removed"
            assert await _consolidated_at(conn, m2) is None, "surviving source reset for re-consolidation"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_revert_moves_back_and_restores_entities(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-rev-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Alice prefers tea over coffee.")
            e1 = await _insert_entity(conn, bank_id, "Alice")
            await _link_entity(conn, m1, e1)

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(m1), state="invalidated", request_context=request_context)
            async with pool.acquire() as conn:
                assert not await _in_live(conn, m1)
                arch = await _archive_row(conn, m1)
                assert arch is not None and e1 in (arch["entity_ids"] or []), "entity ids snapshotted on invalidate"
                assert await _entity_ids_for(conn, m1) == [], "unit_entities cascade-pruned on move"

            result = await memory.update_memory_unit(bank_id, str(m1), state="valid", request_context=request_context)

        assert result["state"] == "valid"
        assert result["invalidation_reason"] is None
        async with pool.acquire() as conn:
            assert await _in_live(conn, m1), "reverted row back in memory_units"
            assert await _archive_row(conn, m1) is None, "archive row removed on revert"
            assert await _consolidated_at(conn, m1) is None, "reverted memory re-consolidates"
            assert e1 in await _entity_ids_for(conn, m1), "entity associations restored on revert"
            reverted_emb = await conn.fetchval("SELECT embedding FROM memory_units WHERE id = $1", m1)
            assert reverted_emb is not None, "embedding recomputed on revert (archive keeps none)"
            # Native backend (test default) stores a real tsvector; it must be rebuilt on
            # revert so the reverted fact is keyword-searchable again (archive keeps none, #2503).
            reverted_sv = await conn.fetchval("SELECT search_vector FROM memory_units WHERE id = $1", m1)
            assert reverted_sv is not None, "search_vector recomputed on revert (archive keeps none)"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_invalidate_idempotent_updates_reason(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-idem-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Bob works at Google.")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(
                bank_id, str(m1), state="invalidated", reason="first", request_context=request_context
            )
            result = await memory.update_memory_unit(
                bank_id, str(m1), state="invalidated", reason="second", request_context=request_context
            )

        assert result["state"] == "invalidated"
        assert result["invalidation_reason"] == "second"
        async with pool.acquire() as conn:
            assert not await _in_live(conn, m1)
            assert (await _archive_row(conn, m1))["invalidation_reason"] == "second"
        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


class TestEdit:
    @pytest.mark.asyncio
    async def test_edit_changes_text_and_rederives(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-edit-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "The assistant visited Paris in 2023.")
            await conn.execute(
                "UPDATE memory_units SET search_vector = to_tsvector('english'::regconfig, text) WHERE id = $1",
                m1,
            )
            obs_id = await _insert_observation(conn, bank_id, "The assistant went to Paris.", [m1])

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            result = await memory.update_memory_unit(
                bank_id,
                str(m1),
                text="The user visited Paris in 2023.",
                reason="wrong subject",
                request_context=request_context,
            )

        assert result["text"] == "The user visited Paris in 2023."
        assert result["state"] == "valid"
        async with pool.acquire() as conn:
            assert await _in_live(conn, m1), "edited row stays live"
            row = dict(
                await conn.fetchrow(
                    "SELECT text, consolidated_at, search_vector::text AS search_vector "
                    "FROM memory_units WHERE id = $1",
                    m1,
                )
            )
            assert row["text"] == "The user visited Paris in 2023."
            assert row["consolidated_at"] is None, "edited memory re-consolidates"
            assert "'assist'" not in row["search_vector"], "old text must not stay in native FTS search_vector"
            assert "'user'" in row["search_vector"], "new text must refresh native FTS search_vector"
            assert str(obs_id) not in await _obs_ids(conn, bank_id), "stale observation re-derived"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_edit_fields_dates_facttype_context(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-editfields-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A world fact.", fact_type="world")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            result = await memory.update_memory_unit(
                bank_id,
                str(m1),
                context="from a chat",
                occurred_start="2023-06-01",
                new_fact_type="experience",
                request_context=request_context,
            )

        assert result["type"] == "experience"
        assert result["context"] == "from a chat"
        assert result["occurred_start"] is not None and result["occurred_start"].startswith("2023-06-01")
        assert result["edited_at"] is not None, "edit records edited_at (user-modified marker)"
        async with pool.acquire() as conn:
            row = dict(
                await conn.fetchrow(
                    "SELECT fact_type, context, occurred_start, event_date FROM memory_units WHERE id = $1", m1
                )
            )
            assert row["fact_type"] == "experience"
            assert row["context"] == "from a chat"
            assert row["occurred_start"].date().isoformat() == "2023-06-01"
            assert row["event_date"].date().isoformat() == "2023-06-01", "event_date tracks occurred_start"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_edit_replaces_entities(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-editent-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Alice met Bob in Paris.")
            # Pre-link a wrong entity the LLM extracted.
            wrong = await _insert_entity(conn, bank_id, "Carol")
            await _link_entity(conn, m1, wrong)

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            # Correct the entity set: drop Carol, attach Alice + Bob.
            result = await memory.update_memory_unit(
                bank_id,
                str(m1),
                entities=["Alice", "Bob"],
                request_context=request_context,
            )

        assert result is not None
        assert set(result["entities"]) == {"Alice", "Bob"}
        assert result["edited_at"] is not None, "entity edit records the user-modified marker"
        async with pool.acquire() as conn:
            names = await conn.fetch(
                "SELECT e.canonical_name FROM unit_entities ue "
                "JOIN entities e ON e.id = ue.entity_id WHERE ue.unit_id = $1",
                m1,
            )
            assert {r["canonical_name"] for r in names} == {"Alice", "Bob"}, "unit_entities rebuilt"
            assert wrong not in await _entity_ids_for(conn, m1), "wrong entity detached"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_edit_empty_entities_detaches_all(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-editent0-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A fact with a spurious entity.")
            e = await _insert_entity(conn, bank_id, "Spurious")
            await _link_entity(conn, m1, e)

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            result = await memory.update_memory_unit(bank_id, str(m1), entities=[], request_context=request_context)

        assert result["entities"] == []
        async with pool.acquire() as conn:
            assert await _entity_ids_for(conn, m1) == [], "empty list detaches all entities"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_cannot_edit_invalidated_memory(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-editinv-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Stale fact.")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(m1), state="invalidated", request_context=request_context)
            with pytest.raises(ValueError, match="revert"):
                await memory.update_memory_unit(bank_id, str(m1), text="corrected", request_context=request_context)

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Guards / listing / recall
# ---------------------------------------------------------------------------


class TestGuardsAndListing:
    @pytest.mark.asyncio
    async def test_cannot_curate_observation(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-obs-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "source fact")
            obs_id = await _insert_observation(conn, bank_id, "a synthesized observation", [m1])

        with pytest.raises(ValueError, match="observation"):
            await memory.update_memory_unit(bank_id, str(obs_id), state="invalidated", request_context=request_context)

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-404-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        result = await memory.update_memory_unit(
            bank_id, str(uuid.uuid4()), state="invalidated", request_context=request_context
        )
        assert result is None
        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_list_filters_by_state(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-list-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            keep = await _insert_memory(conn, memory, bank_id, "Valid fact one.")
            m2 = await _insert_memory(conn, memory, bank_id, "Fact to retire.")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(
                bank_id, str(m2), state="invalidated", reason="dup", request_context=request_context
            )

        # Default lists live facts only.
        live = (await memory.list_memory_units(bank_id, request_context=request_context))["items"]
        live_ids = {i["id"] for i in live}
        assert str(keep) in live_ids and str(m2) not in live_ids
        assert all(i["state"] == "valid" for i in live)

        # state=invalidated reads the archive.
        invalid = (await memory.list_memory_units(bank_id, state="invalidated", request_context=request_context))[
            "items"
        ]
        assert len(invalid) == 1
        assert invalid[0]["id"] == str(m2)
        assert invalid[0]["state"] == "invalidated"
        assert invalid[0]["invalidation_reason"] == "dup"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_list_and_get_memory_units_include_metadata(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-curation-metadata-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        metadata = {"source": "slack", "channel": "engineering", "thread_id": "T123"}

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            mem_id = await _insert_memory(conn, memory, bank_id, "Fact with metadata.", metadata=metadata)

        live = (await memory.list_memory_units(bank_id, request_context=request_context))["items"]
        live_item = next(item for item in live if item["id"] == str(mem_id))
        assert live_item["metadata"] == metadata

        detail = await memory.get_memory_unit(bank_id, str(mem_id), request_context=request_context)
        assert detail is not None
        assert detail["metadata"] == metadata

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(
                bank_id, str(mem_id), state="invalidated", reason="stale", request_context=request_context
            )

        invalid = (await memory.list_memory_units(bank_id, state="invalidated", request_context=request_context))[
            "items"
        ]
        assert invalid[0]["id"] == str(mem_id)
        assert invalid[0]["metadata"] == metadata

        invalid_detail = await memory.get_memory_unit(bank_id, str(mem_id), request_context=request_context)
        assert invalid_detail is not None
        assert invalid_detail["state"] == "invalidated"
        assert invalid_detail["metadata"] == metadata

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_list_filters_by_document(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-doc-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO documents (id, bank_id) VALUES ($1, $2)", doc_id, bank_id)
            m_doc = await _insert_memory(conn, memory, bank_id, "Fact from the document.")
            await _insert_memory(conn, memory, bank_id, "Fact from elsewhere.")
            await conn.execute("UPDATE memory_units SET document_id = $1 WHERE id = $2", doc_id, m_doc)

        # Live listing scoped to the document returns only its fact.
        live = (await memory.list_memory_units(bank_id, document_id=doc_id, request_context=request_context))["items"]
        assert {i["id"] for i in live} == {str(m_doc)}

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(m_doc), state="invalidated", request_context=request_context)

        # Invalidated archive is filterable by document too (carries document_id).
        scoped = (
            await memory.list_memory_units(
                bank_id, state="invalidated", document_id=doc_id, request_context=request_context
            )
        )["items"]
        assert {i["id"] for i in scoped} == {str(m_doc)}
        other = (
            await memory.list_memory_units(
                bank_id, state="invalidated", document_id="nope", request_context=request_context
            )
        )["items"]
        assert other == []

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recall_excludes_invalidated(self, memory: MemoryEngine, request_context: RequestContext):
        bank_id = f"test-curation-recall-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        unit_ids = await memory.retain_async(
            bank_id,
            "The Anaconda XR7 telescope has a 9000mm focal length.",
            request_context=request_context,
        )
        assert unit_ids, "retain should produce at least one memory unit"

        def _hit(res) -> bool:
            return any("anaconda" in f.text.lower() or "telescope" in f.text.lower() for f in res.results)

        before = await memory.recall_async(
            bank_id, "Anaconda XR7 telescope focal length", request_context=request_context
        )
        assert _hit(before), "fact should be recalled before invalidation"

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            for uid in unit_ids:
                await memory.update_memory_unit(bank_id, uid, state="invalidated", request_context=request_context)

        after = await memory.recall_async(
            bank_id, "Anaconda XR7 telescope focal length", request_context=request_context
        )
        assert not _hit(after), "invalidated fact must be excluded from recall"

        await memory.delete_bank(bank_id, request_context=request_context)
