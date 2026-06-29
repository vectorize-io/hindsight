"""Tests for memory curation: edit / invalidate / revert.

Invalidation MOVES a fact out of ``memory_units`` into the
``invalidated_memory_units`` archive, so the recall hot-path never sees it.
These tests cover the move semantics, lossless revert (incl. entity
associations), edit, the guards, listing, and recall exclusion.
"""

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.db_utils import acquire_with_retry
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import embedding_processing, link_utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_memory(
    conn, memory: MemoryEngine, bank_id: str, text: str, fact_type: str = "experience"
) -> uuid.UUID:
    """Insert a live memory unit with a real embedding, bypassing the LLM pipeline."""
    mem_id = uuid.uuid4()
    emb = await embedding_processing.generate_embeddings_batch(memory.embeddings, [text])
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, event_date, created_at, updated_at, consolidated_at)
        VALUES ($1, $2, $3, $4, $5::vector, NOW(), NOW(), NOW(), NOW())
        """,
        mem_id,
        bank_id,
        text,
        fact_type,
        str(emb[0]),
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


async def _archive_has_embedding_column(conn) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'invalidated_memory_units' AND column_name = 'embedding'"
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

        async with pool.acquire() as conn:
            assert not await _in_live(conn, m1), "invalidated row must leave memory_units"
            arch = await _archive_row(conn, m1)
            assert arch is not None, "row must be in the archive"
            assert arch["invalidation_reason"] == "decommissioned"
            assert not await _archive_has_embedding_column(conn), (
                "archive is cold storage; the schema drops the embedding column (#2209)"
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

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_revert_reembeds_from_survivors_when_archived_entity_pruned_midwindow(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-curation-rev-prune-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Alice met Bob in Paris.")
            e_alice = await _insert_entity(conn, bank_id, "Alice")
            e_bob = await _insert_entity(conn, bank_id, "Bob")
            await _link_entity(conn, m1, e_alice)
            await _link_entity(conn, m1, e_bob)

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(m1), state="invalidated", request_context=request_context)

            calls: list[list[str]] = []
            deleted = {"done": False}
            orig = memory._reembed_memory_text

            async def _spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
                calls.append(list(entities))
                if not deleted["done"]:
                    deleted["done"] = True
                    async with acquire_with_retry(backend) as c:
                        await c.execute(
                            "DELETE FROM entities WHERE bank_id = $1 AND canonical_name = $2", bank_id, "Bob"
                        )
                return await orig(
                    text=text,
                    occurred_start=occurred_start,
                    occurred_end=occurred_end,
                    mentioned_at=mentioned_at,
                    entities=entities,
                )

            with patch.object(memory, "_reembed_memory_text", new=_spy):
                result = await memory.update_memory_unit(
                    bank_id, str(m1), state="valid", request_context=request_context
                )

        assert result["state"] == "valid"
        assert len(calls) == 2, "a survivor mismatch re-embeds under the lock"
        assert sorted(calls[1]) == ["Alice"], "revert re-embeds from the restored (survivor) entity set"
        async with pool.acquire() as conn:
            names = await conn.fetch(
                "SELECT e.canonical_name FROM unit_entities ue "
                "JOIN entities e ON e.id = ue.entity_id WHERE ue.unit_id = $1",
                m1,
            )
            assert {r["canonical_name"] for r in names} == {"Alice"}, "only the surviving entity is restored"
            emb = await conn.fetchval("SELECT embedding FROM memory_units WHERE id = $1", m1)
            assert emb is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_revert_reembeds_when_archive_text_rewritten_midwindow(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # #3 regression: the archive row's TEXT is rewritten during the connection-free embed
        # window, so the Phase-1 embedding (old text) is stale. The revert must re-embed from the
        # LOCKED archive row, not store the precomputed (now-stale) vector against the new text.
        bank_id = f"test-curation-rev-txtrace-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Alice met Bob in Paris.")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(bank_id, str(m1), state="invalidated", request_context=request_context)

            calls: list[str] = []
            raced = {"done": False}
            orig = memory._reembed_memory_text

            async def _spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
                calls.append(text)
                if not raced["done"]:
                    raced["done"] = True
                    # Rewrite the archived row's text on a SEPARATE connection during the
                    # connection-free embed window (the between-phases race #3 guards against).
                    async with acquire_with_retry(backend) as c:
                        await c.execute(
                            "UPDATE invalidated_memory_units SET text = $2 WHERE id = $1 AND bank_id = $3",
                            m1,
                            "Rewritten archived text.",
                            bank_id,
                        )
                return await orig(
                    text=text,
                    occurred_start=occurred_start,
                    occurred_end=occurred_end,
                    mentioned_at=mentioned_at,
                    entities=entities,
                )

            with patch.object(memory, "_reembed_memory_text", new=_spy):
                result = await memory.update_memory_unit(
                    bank_id, str(m1), state="valid", request_context=request_context
                )

        assert result["state"] == "valid"
        assert len(calls) == 2, "a stale archive snapshot (text rewritten mid-window) re-embeds under the lock"
        assert calls[1] == "Rewritten archived text.", "the in-txn re-embed uses the locked (current) archive text"
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT text, embedding FROM memory_units WHERE id = $1", m1)
        assert row["text"] == "Rewritten archived text.", "revert restores the locked archive row"
        assert row["embedding"] is not None

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
            obs_id = await _insert_observation(conn, bank_id, "The assistant went to Paris.", [m1])
            orig_emb = await conn.fetchval("SELECT embedding FROM memory_units WHERE id = $1", m1)

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
                await conn.fetchrow("SELECT text, consolidated_at, embedding FROM memory_units WHERE id = $1", m1)
            )
            assert row["text"] == "The user visited Paris in 2023."
            assert row["consolidated_at"] is None, "edited memory re-consolidates"
            assert row["embedding"] is not None, "edit re-embeds (phase split must not drop the embedding)"
            assert row["embedding"] != orig_emb, "edit stores a freshly recomputed vector, not the stale one"
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
    async def test_entity_edit_reresolves_when_resolved_entity_pruned_midwindow(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-curation-edit-prune-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Alice met Bob in Paris.")

        calls: list[list[str]] = []
        deleted = {"done": False}
        orig_resolve = link_utils.resolve_entities_only
        orig_embed = memory._reembed_memory_text

        async def _resolve_spy(*args, **kwargs):
            # Resolve normally (find-or-create autocommits entities OFF the write txn), then prune
            # one resolved entity BEFORE update_memory_unit reads back canonical names. This is the
            # real resolve->name-fetch race: edit_plan.names is captured short, so a name-set match
            # check would commit a partial set. ID-coverage must detect the missing id and re-resolve.
            result = await orig_resolve(*args, **kwargs)
            if not deleted["done"]:
                deleted["done"] = True
                async with acquire_with_retry(backend) as c:
                    await c.execute("DELETE FROM entities WHERE bank_id = $1 AND canonical_name = $2", bank_id, "Bob")
            return result

        async def _embed_spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
            calls.append(list(entities))
            return await orig_embed(
                text=text,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                entities=entities,
            )

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
            patch.object(link_utils, "resolve_entities_only", new=_resolve_spy),
            patch.object(memory, "_reembed_memory_text", new=_embed_spy),
        ):
            result = await memory.update_memory_unit(
                bank_id, str(m1), entities=["Alice", "Bob"], request_context=request_context
            )

        assert result is not None
        assert set(result["entities"]) == {"Alice", "Bob"}, "edit must not commit a partial entity set"
        assert calls[0] == ["Alice"], (
            "resolved entity pruned before edit_plan.names was captured (the resolve->name-fetch race)"
        )
        assert len(calls) == 2, "a prune mismatch re-resolves and re-embeds under the lock"
        assert set(calls[1]) == {"Alice", "Bob"}, "the recovered re-embed uses the full re-resolved entity set"
        async with pool.acquire() as conn:
            names = await conn.fetch(
                "SELECT e.canonical_name FROM unit_entities ue "
                "JOIN entities e ON e.id = ue.entity_id WHERE ue.unit_id = $1",
                m1,
            )
            assert {r["canonical_name"] for r in names} == {"Alice", "Bob"}, "pruned entity re-created and linked"
            emb = await conn.fetchval("SELECT embedding FROM memory_units WHERE id = $1", m1)
            assert emb is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_edit_aborts_on_concurrent_field_change(self, memory: MemoryEngine, request_context: RequestContext):
        # A concurrent edit that commits during the off-connection embed must NOT be silently
        # clobbered by the precomputed (stale) edit. The Phase-2 re-lock detects the changed
        # column and aborts (rollback), so the concurrent writer's text survives.
        bank_id = f"test-curation-edit-race-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Original text.")

        orig_embed = memory._reembed_memory_text
        raced = {"done": False}

        async def _embed_spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
            # Inside the connection-free embed window, commit a concurrent text edit on a SEPARATE
            # backend connection (the real between-phases race the abort guards against).
            if not raced["done"]:
                raced["done"] = True
                async with acquire_with_retry(backend) as c:
                    await c.execute(
                        "UPDATE memory_units SET text = $2, updated_at = now() WHERE id = $1",
                        m1,
                        "Concurrently edited text.",
                    )
            return await orig_embed(
                text=text,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                entities=entities,
            )

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
            patch.object(memory, "_reembed_memory_text", new=_embed_spy),
        ):
            # A context-only edit still re-embeds (so the spy fires); the abort must fire before it
            # can overwrite the racing text edit with the Phase-1 snapshot.
            with pytest.raises(RuntimeError, match="modified concurrently"):
                await memory.update_memory_unit(
                    bank_id, str(m1), context="late annotation", request_context=request_context
                )

        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT text, context FROM memory_units WHERE id = $1", m1)
        assert row["text"] == "Concurrently edited text.", "concurrent edit preserved (no lost update)"
        assert row["context"] != "late annotation", "aborted edit did not apply"

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

    @pytest.mark.asyncio
    async def test_entity_edit_embed_failure_reclaims_orphan_entities(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-curation-edit-embedfail-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A standalone fact.")

        try:
            consolidation_mock = AsyncMock()
            # Leave submit_async_graph_maintenance REAL: under SyncTaskBackend the sweep runs inline,
            # so we can assert the leaked entities are actually pruned (not just that a mock was awaited).
            with (
                patch.object(memory, "submit_async_consolidation", new=consolidation_mock),
                patch.object(memory, "_reembed_memory_text", new=AsyncMock(side_effect=RuntimeError("embedder down"))),
            ):
                with pytest.raises(RuntimeError, match="embedder down"):
                    await memory.update_memory_unit(
                        bank_id, str(m1), entities=["Alice", "Bob"], request_context=request_context
                    )

            # resolve_entities_only autocommitted Alice/Bob in Phase 1, but the edit never linked them
            # (the re-embed raised first). The failure path must enqueue this unit so the bank-wide
            # orphan-entity prune runs and reclaims them; otherwise they leak.
            async with pool.acquire() as conn:
                orphan_count = await conn.fetchval("SELECT count(*) FROM entities WHERE bank_id = $1", bank_id)
            assert orphan_count == 0, "orphan entities from the failed edit were reclaimed by graph maintenance"
            consolidation_mock.assert_not_awaited()  # a failed edit must not trigger consolidation
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_entity_edit_resolve_partial_commit_reclaims_orphan_entities(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        bank_id = f"test-curation-edit-resolvefail-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A standalone fact.")

        # Simulate resolve_entities_only autocommitting an entity on the Phase-1 (autocommit) connection
        # and THEN raising. entities_maybe_committed is set BEFORE the resolve call, so the failure path
        # must still enqueue the unit and let the inline sweep reclaim the committed orphan. If the flag
        # were set after the call, this entity would leak.
        async def _partially_commit_then_raise(*args, **kwargs):
            phase1_conn = args[1]
            await phase1_conn.execute(
                "INSERT INTO entities (id, bank_id, canonical_name) VALUES ($1, $2, $3)",
                uuid.uuid4(),
                bank_id,
                "Ghost",
            )
            # Prove the insert autocommitted on the Phase-1 (autocommit) connection: a SEPARATE
            # backend connection must see it. If Phase 1 were wrapped in a transaction, this would
            # be 0 and the later orphan_count == 0 would prove nothing (a rollback, not the sweep).
            async with acquire_with_retry(backend) as other:
                seen = await other.fetchval(
                    "SELECT count(*) FROM entities WHERE bank_id = $1 AND canonical_name = $2",
                    bank_id,
                    "Ghost",
                )
            assert seen == 1, "Ghost must be autocommitted (visible cross-connection) before the resolver raises"
            raise RuntimeError("resolver down")

        try:
            consolidation_mock = AsyncMock()
            # submit_async_graph_maintenance stays REAL so the sweep runs inline under SyncTaskBackend.
            # resolve_entities_only is patched at its module path because update_memory_unit imports it
            # at call time (`from .retain.link_utils import resolve_entities_only`).
            with (
                patch.object(memory, "submit_async_consolidation", new=consolidation_mock),
                patch(
                    "hindsight_api.engine.retain.link_utils.resolve_entities_only",
                    new=_partially_commit_then_raise,
                ),
            ):
                with pytest.raises(RuntimeError, match="resolver down"):
                    await memory.update_memory_unit(
                        bank_id, str(m1), entities=["Alice", "Bob"], request_context=request_context
                    )

            async with pool.acquire() as conn:
                orphan_count = await conn.fetchval("SELECT count(*) FROM entities WHERE bank_id = $1", bank_id)
            assert orphan_count == 0, "an entity committed before the resolver raised was reclaimed by the sweep"
            consolidation_mock.assert_not_awaited()  # a failed edit must not trigger consolidation
        finally:
            await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_context_edit_reembeds_when_unit_entities_change_midwindow(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # A context-only edit (resolved_for_unit is None) re-embeds from the unit's CURRENT entity
        # names. If a concurrent writer changes unit_entities while the embedder runs off-connection,
        # the Phase-2 lock re-reads the names; because they differ from the Phase-1 snapshot, the
        # non-entity-edit branch re-embeds in-txn so the stored vector never names a stale entity set.
        # unit_entities is NOT an abort-guarded column, so this re-embeds rather than aborting.
        bank_id = f"test-curation-ctx-reembed-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A fact about gardening.")
            e1 = await _insert_entity(conn, bank_id, "Alpha")
            e2 = await _insert_entity(conn, bank_id, "Beta")
            await _link_entity(conn, m1, e1)

        calls: list[list[str]] = []
        orig_embed = memory._reembed_memory_text
        linked = {"done": False}

        async def _embed_spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
            calls.append(list(entities))
            # On the first (Phase-1) embed, link a second entity on a SEPARATE connection so the
            # Phase-2 re-lock observes a changed unit_entities set.
            if not linked["done"]:
                linked["done"] = True
                async with acquire_with_retry(backend) as c:
                    await _link_entity(c, m1, e2)
            return await orig_embed(
                text=text,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                entities=entities,
            )

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
            patch.object(memory, "_reembed_memory_text", new=_embed_spy),
        ):
            result = await memory.update_memory_unit(
                bank_id, str(m1), context="late note", request_context=request_context
            )

        assert result is not None
        assert len(calls) == 2, "a mid-window unit_entities change triggers an in-txn re-embed"
        assert calls[0] == ["Alpha"], "Phase-1 embed used the unit's entity set at read time"
        assert set(calls[1]) == {"Alpha", "Beta"}, "Phase-2 re-embed uses the concurrently-updated entity set"
        async with pool.acquire() as conn:
            names = await conn.fetch(
                "SELECT e.canonical_name FROM unit_entities ue "
                "JOIN entities e ON e.id = ue.entity_id WHERE ue.unit_id = $1",
                m1,
            )
            row = await conn.fetchrow("SELECT context, embedding FROM memory_units WHERE id = $1", m1)
        assert {r["canonical_name"] for r in names} == {"Alpha", "Beta"}, "both entities remain linked"
        assert row["context"] == "late note", "the context edit applied"
        assert row["embedding"] is not None

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_edit_then_invalidate_archives_edited_text(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # A single call that BOTH edits text and invalidates applies the edit first (Phase-2 UPDATE),
        # then moves the freshly-edited row to the archive -- so the archived text is the corrected one.
        bank_id = f"test-curation-edit-invalidate-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "Original.")

        with (
            patch.object(memory, "submit_async_consolidation", new=AsyncMock()),
            patch.object(memory, "submit_async_graph_maintenance", new=AsyncMock()),
        ):
            await memory.update_memory_unit(
                bank_id, str(m1), text="Corrected.", state="invalidated", request_context=request_context
            )

        async with pool.acquire() as conn:
            assert not await _in_live(conn, m1), "the row was moved out of memory_units"
            arch = await _archive_row(conn, m1)
        assert arch is not None, "the row landed in the archive"
        assert arch["text"] == "Corrected.", "the edit applied before the archive move"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_entity_edit_aborts_and_reclaims_orphans_on_concurrent_change(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # An entity-changing edit autocommits its resolved entities in Phase 1 (off-txn). If a
        # concurrent edit changes an abort-guarded column while the embedder runs, the Phase-2 re-lock
        # aborts (rollback) BEFORE the entities are linked, leaving them committed orphans. The
        # finally-block enqueue + forced graph maintenance must reclaim them.
        bank_id = f"test-curation-edit-abort-reclaim-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        pool = await memory._get_pool()
        backend = await memory._get_backend()
        async with pool.acquire() as conn:
            m1 = await _insert_memory(conn, memory, bank_id, "A standalone fact.")

        orig_embed = memory._reembed_memory_text
        raced = {"done": False}

        async def _embed_spy(*, text, occurred_start, occurred_end, mentioned_at, entities):
            # Commit a concurrent text edit during the off-connection embed so the Phase-2 re-lock
            # detects the changed column and aborts.
            if not raced["done"]:
                raced["done"] = True
                async with acquire_with_retry(backend) as c:
                    await c.execute(
                        "UPDATE memory_units SET text = $2, updated_at = now() WHERE id = $1",
                        m1,
                        "Concurrently edited text.",
                    )
            return await orig_embed(
                text=text,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                entities=entities,
            )

        consolidation_mock = AsyncMock()
        # Leave submit_async_graph_maintenance REAL so the inline SyncTaskBackend sweep reclaims orphans.
        with (
            patch.object(memory, "submit_async_consolidation", new=consolidation_mock),
            patch.object(memory, "_reembed_memory_text", new=_embed_spy),
        ):
            with pytest.raises(RuntimeError, match="modified concurrently"):
                await memory.update_memory_unit(
                    bank_id, str(m1), entities=["Alice", "Bob"], request_context=request_context
                )

        async with pool.acquire() as conn:
            orphan_count = await conn.fetchval("SELECT count(*) FROM entities WHERE bank_id = $1", bank_id)
            row = await conn.fetchrow("SELECT text FROM memory_units WHERE id = $1", m1)
        assert orphan_count == 0, "entities autocommitted in Phase 1 were reclaimed after the Phase-2 abort"
        assert row["text"] == "Concurrently edited text.", "the concurrent edit survived (no lost update)"
        consolidation_mock.assert_not_awaited()

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


# ---------------------------------------------------------------------------
# Update mental model (embed-before-acquire)
# ---------------------------------------------------------------------------


class TestUpdateMentalModel:
    @pytest.mark.asyncio
    async def test_update_mental_model_embeds_before_acquire(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # The new embedding is computed BEFORE a pooled connection is acquired, so a slow embedder
        # never pins a DB connection. (_authenticate_tenant does not touch the pool.)
        bank_id = f"test-mm-embed-order-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        order: list[str] = []
        real_acquire = acquire_with_retry

        async def _embed_spy(_embeddings, _texts):
            order.append("embed")
            # 384 dims to satisfy the mental_models.embedding vector(384) cast; the value is
            # irrelevant (the random id matches no row), only the embed-before-acquire order matters.
            return [[0.0] * 384]

        @asynccontextmanager
        async def _acquire_spy(*args, **kwargs):
            order.append("acquire")
            async with real_acquire(*args, **kwargs) as conn:
                yield conn

        with (
            patch("hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch", new=_embed_spy),
            patch("hindsight_api.engine.memory_engine.acquire_with_retry", new=_acquire_spy),
        ):
            await memory.update_mental_model(
                bank_id, str(uuid.uuid4()), content="new content", request_context=request_context
            )

        assert "embed" in order, "a content update must compute an embedding"
        assert "acquire" in order, "the write path must acquire a connection"
        assert order.index("embed") < order.index("acquire"), "embed must happen before acquiring a connection"

        await memory.delete_bank(bank_id, request_context=request_context)

    @pytest.mark.asyncio
    async def test_update_mental_model_skips_embed_when_content_none(
        self, memory: MemoryEngine, request_context: RequestContext
    ):
        # No content change -> no embedding (the embed is gated on `content is not None`).
        bank_id = f"test-mm-no-embed-{uuid.uuid4().hex[:8]}"
        await _ensure_bank(memory, bank_id, request_context)

        embed_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        with patch("hindsight_api.engine.retain.embedding_utils.generate_embeddings_batch", new=embed_mock):
            await memory.update_mental_model(bank_id, str(uuid.uuid4()), tags=["x"], request_context=request_context)

        embed_mock.assert_not_awaited()

        await memory.delete_bank(bank_id, request_context=request_context)
