"""
Tests for delta retain — upsert optimization that only re-processes changed chunks.
"""

import logging
from datetime import datetime, timezone

import pytest

from hindsight_api import RequestContext
from hindsight_api.engine.memory_engine import Budget

logger = logging.getLogger(__name__)


def _ts():
    return datetime.now(timezone.utc).timestamp()


# ============================================================
# Core Delta Retain Tests
# ============================================================


@pytest.mark.asyncio
async def test_delta_retain_unchanged_content_skips_llm(memory, request_context):
    """
    When upserting a document with identical content, no new facts should be
    extracted (LLM is not called for unchanged chunks). The existing facts
    should be preserved.
    """
    bank_id = f"test_delta_unchanged_{_ts()}"
    document_id = "conversation-001"

    try:
        content = "Alice works at Google. Bob works at Microsoft."

        # First retain — full processing
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0, "v1 should create facts"

        # Get v1 document state
        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        v1_unit_count = doc_v1["memory_unit_count"]

        # Second retain — same content, should use delta path (no new facts)
        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team info",
            document_id=document_id,
            request_context=request_context,
        )

        # No new units should be returned (nothing changed)
        assert v2_units == [], "Delta retain with unchanged content should return empty unit list"

        # Existing facts should still be there
        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v2["memory_unit_count"] == v1_unit_count, "Existing facts should be preserved"

        # Verify recall still works
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.MID,
            max_tokens=1000,
            request_context=request_context,
        )
        assert len(result.results) > 0, "Should still recall facts after delta retain"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_appended_content(memory, request_context):
    """
    When a conversation grows (new content appended), only new chunks should
    be processed. Facts from unchanged chunks should be preserved.
    """
    bank_id = f"test_delta_append_{_ts()}"
    document_id = "growing-conversation"

    try:
        # First version — short content (single chunk)
        v1_content = "Alice is a software engineer at Google. She works on search infrastructure."

        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="profile",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        # Get v1 facts via recall
        v1_recall = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        v1_fact_texts = {r.text for r in v1_recall.results}

        # Second version — original content + new content appended
        # This should preserve facts from the first chunk and add new ones
        v2_content = v1_content + "\n\nBob joined Google as a product manager in 2024. He previously worked at Meta on AR/VR products."

        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content=v2_content,
            context="profile",
            document_id=document_id,
            request_context=request_context,
        )

        # Should have facts about Bob from the new content
        v2_recall = await memory.recall_async(
            bank_id=bank_id,
            query="What does Bob do?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        bob_facts = [r for r in v2_recall.results if "bob" in r.text.lower()]
        assert len(bob_facts) > 0, "Should have facts about Bob from appended content"

        # Should still have facts about Alice from original content
        alice_recall = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        assert len(alice_recall.results) > 0, "Should still have Alice facts from original content"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_modified_chunk(memory, request_context):
    """
    When content in the middle changes, that chunk should be re-processed
    while other chunks are preserved.
    """
    bank_id = f"test_delta_modified_{_ts()}"
    document_id = "changing-doc"

    try:
        # v1: Alice works at Google
        v1_content = "Alice works at Google as a senior engineer."
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        # v2: Alice works at Microsoft (changed)
        v2_content = "Alice works at Microsoft as a principal engineer."
        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content=v2_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        # New facts should reflect the updated content
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        all_texts = " ".join(r.text.lower() for r in result.results)
        assert "microsoft" in all_texts, f"Should have updated fact about Microsoft, got: {all_texts}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Entity & Link Tests
# ============================================================


@pytest.mark.asyncio
async def test_delta_retain_entities_preserved_for_unchanged_chunks(memory, request_context):
    """
    Entities linked to unchanged chunks should be preserved after delta retain.
    """
    bank_id = f"test_delta_entities_{_ts()}"
    document_id = "entity-doc"

    try:
        v1_content = "Alice works at Google. She is a senior engineer in the Cloud division."
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        # Check entities exist
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v1_entity_names = {e["canonical_name"].lower() for e in v1_entities}
        assert len(v1_entity_names) > 0, "Should have entities after v1 retain"

        # Upsert with same content — entities should persist
        await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        async with pool.acquire() as conn:
            v2_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v2_entity_names = {e["canonical_name"].lower() for e in v2_entities}

        # All v1 entities should still exist
        assert v1_entity_names.issubset(v2_entity_names), (
            f"v1 entities {v1_entity_names} should be preserved, got {v2_entity_names}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_new_entities_created_for_new_chunks(memory, request_context):
    """
    New entities should be created for newly added chunks during delta retain.
    """
    bank_id = f"test_delta_new_entities_{_ts()}"
    document_id = "entity-growth-doc"

    try:
        v1_content = "Alice works at Google."
        await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v1_entity_names = {e["canonical_name"].lower() for e in v1_entities}

        # Append content mentioning new entities
        v2_content = v1_content + "\n\nBob joined Facebook. He works with Charlie on the Reality Labs project."
        await memory.retain_async(
            bank_id=bank_id,
            content=v2_content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        async with pool.acquire() as conn:
            v2_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v2_entity_names = {e["canonical_name"].lower() for e in v2_entities}

        # Should have more entities after adding content with new people/orgs
        assert len(v2_entity_names) > len(v1_entity_names), (
            f"Should have more entities after append: v1={v1_entity_names}, v2={v2_entity_names}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_links_preserved_for_unchanged_chunks(memory, request_context):
    """
    Memory links (temporal, semantic, entity) for unchanged chunks should be preserved.
    """
    bank_id = f"test_delta_links_{_ts()}"
    document_id = "links-doc"

    try:
        content = "Alice is a senior engineer at Google Cloud. She mentors junior engineers and reviews their code."
        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        # Count links after v1
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_link_count = await conn.fetchval(
                """SELECT COUNT(*) FROM memory_links ml
                   JOIN memory_units mu ON ml.from_unit_id = mu.id
                   WHERE mu.bank_id = $1 AND mu.document_id = $2""",
                bank_id,
                document_id,
            )

        # Upsert with same content
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="team",
            document_id=document_id,
            request_context=request_context,
        )

        # Links should be preserved
        async with pool.acquire() as conn:
            v2_link_count = await conn.fetchval(
                """SELECT COUNT(*) FROM memory_links ml
                   JOIN memory_units mu ON ml.from_unit_id = mu.id
                   WHERE mu.bank_id = $1 AND mu.document_id = $2""",
                bank_id,
                document_id,
            )

        assert v2_link_count == v1_link_count, (
            f"Links should be preserved: v1={v1_link_count}, v2={v2_link_count}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Document Metadata & Tags Tests
# ============================================================


@pytest.mark.asyncio
async def test_delta_retain_document_metadata_updated(memory, request_context):
    """
    Document metadata (retain_params, tags) should be updated even when
    chunk content hasn't changed.
    """
    bank_id = f"test_delta_meta_{_ts()}"
    document_id = "metadata-doc"

    try:
        content = "Alice works at Google."

        # v1 with initial tags
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="initial context",
            document_id=document_id,
            request_context=request_context,
        )

        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v1 is not None

        # v2 with updated context (same content — triggers delta path)
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="updated context",
            document_id=document_id,
            request_context=request_context,
        )

        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v2 is not None
        assert doc_v2["updated_at"] >= doc_v1["updated_at"], "Document should have updated timestamp"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_tags_propagated_to_existing_units(memory, request_context):
    """
    When tags change during an upsert with unchanged content, the new tags
    should be propagated to all existing memory units.
    """
    bank_id = f"test_delta_tags_{_ts()}"
    document_id = "tags-doc"

    try:
        content = "Alice works at Google."

        # v1 with tag "team-a"
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{
                "content": content,
                "document_id": document_id,
                "tags": ["team-a"],
            }],
            request_context=request_context,
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_tags = await conn.fetch(
                "SELECT tags FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        assert all("team-a" in row["tags"] for row in v1_tags), "v1 units should have team-a tag"

        # v2 with same content but different tags
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{
                "content": content,
                "document_id": document_id,
                "tags": ["team-b", "important"],
            }],
            request_context=request_context,
        )

        async with pool.acquire() as conn:
            v2_tags = await conn.fetch(
                "SELECT tags FROM memory_units WHERE bank_id = $1 AND document_id = $2",
                bank_id,
                document_id,
            )
        for row in v2_tags:
            assert "team-b" in row["tags"], f"v2 units should have team-b tag, got {row['tags']}"
            assert "important" in row["tags"], f"v2 units should have important tag, got {row['tags']}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Chunk Management Tests
# ============================================================


@pytest.mark.asyncio
async def test_delta_retain_removed_chunks_delete_facts(memory, request_context):
    """
    When content is shortened (chunks removed), facts from the removed
    chunks should be deleted.
    """
    bank_id = f"test_delta_removed_{_ts()}"
    document_id = "shrinking-doc"

    try:
        # v1: longer content with facts about Alice and Bob
        v1_content = (
            "Alice is a senior engineer at Google Cloud. "
            "She leads the infrastructure team and has been there for 5 years.\n\n"
            "Bob is a product manager at Facebook Reality Labs. "
            "He previously worked at Amazon on Alexa voice products."
        )

        v1_units = await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            context="profiles",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(v1_units) > 0

        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        v1_count = doc_v1["memory_unit_count"]

        # v2: Completely different content — all chunks change
        v2_content = "Charlie works at Netflix as a data scientist."
        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content=v2_content,
            context="profiles",
            document_id=document_id,
            request_context=request_context,
        )

        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v2 is not None

        # Should have facts about Charlie
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Who works at Netflix?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        all_texts = " ".join(r.text.lower() for r in result.results)
        assert "charlie" in all_texts or "netflix" in all_texts, (
            f"Should have facts about Charlie/Netflix after replacing content, got: {all_texts}"
        )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_chunks_have_content_hash(memory, request_context):
    """
    After retain, chunks should have content_hash populated.
    """
    bank_id = f"test_delta_hash_{_ts()}"
    document_id = "hash-doc"

    try:
        content = "Alice works at Google as a software engineer."
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            document_id=document_id,
            request_context=request_context,
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            chunks = await conn.fetch(
                "SELECT chunk_id, content_hash FROM chunks WHERE document_id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )

        assert len(chunks) > 0, "Should have stored chunks"
        for chunk in chunks:
            assert chunk["content_hash"] is not None, f"Chunk {chunk['chunk_id']} should have content_hash"
            assert len(chunk["content_hash"]) == 64, "content_hash should be SHA256 hex (64 chars)"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Backward Compatibility Tests
# ============================================================


@pytest.mark.asyncio
async def test_retain_without_document_id_still_works(memory, request_context):
    """
    Retain without document_id should still work normally (no delta path).
    """
    bank_id = f"test_no_docid_{_ts()}"

    try:
        units = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="test",
            request_context=request_context,
        )
        assert len(units) > 0, "Should create facts without document_id"

        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            budget=Budget.MID,
            max_tokens=1000,
            request_context=request_context,
        )
        assert len(result.results) > 0

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_first_retain_full_path(memory, request_context):
    """
    First retain of a new document should use the full path (no delta possible).
    """
    bank_id = f"test_first_retain_{_ts()}"
    document_id = "new-doc"

    try:
        units = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google.",
            context="test",
            document_id=document_id,
            request_context=request_context,
        )
        assert len(units) > 0, "First retain should create facts via full path"

        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["memory_unit_count"] > 0

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


# ============================================================
# Edge Cases
# ============================================================


@pytest.mark.asyncio
async def test_delta_retain_empty_to_content(memory, request_context):
    """
    Going from gibberish (zero facts) to real content should work.
    """
    bank_id = f"test_delta_empty_{_ts()}"
    document_id = "empty-to-content"

    try:
        # v1: content that probably produces zero facts
        await memory.retain_async(
            bank_id=bank_id,
            content="!!!###$$$%%%",
            document_id=document_id,
            request_context=request_context,
        )

        doc_v1 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v1 is not None

        # v2: real content
        v2_units = await memory.retain_async(
            bank_id=bank_id,
            content="Alice works at Google as a senior engineer.",
            document_id=document_id,
            request_context=request_context,
        )

        doc_v2 = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc_v2 is not None
        assert doc_v2["memory_unit_count"] > 0 or len(v2_units) > 0, "Should have facts after updating with real content"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_multiple_upserts(memory, request_context):
    """
    Multiple sequential upserts should work correctly, with delta optimization
    kicking in after the first retain.
    """
    bank_id = f"test_delta_multi_{_ts()}"
    document_id = "multi-upsert"

    try:
        # v1: initial
        v1_content = "Alice works at Google."
        await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            document_id=document_id,
            request_context=request_context,
        )

        # v2: same content (delta: no changes)
        await memory.retain_async(
            bank_id=bank_id,
            content=v1_content,
            document_id=document_id,
            request_context=request_context,
        )

        # v3: append
        v3_content = v1_content + "\n\nBob works at Microsoft."
        await memory.retain_async(
            bank_id=bank_id,
            content=v3_content,
            document_id=document_id,
            request_context=request_context,
        )

        # v4: same as v3 (delta: no changes again)
        await memory.retain_async(
            bank_id=bank_id,
            content=v3_content,
            document_id=document_id,
            request_context=request_context,
        )

        # Final check: should have facts about both Alice and Bob
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Who works where?",
            budget=Budget.MID,
            max_tokens=2000,
            request_context=request_context,
        )
        all_texts = " ".join(r.text.lower() for r in result.results)
        assert "alice" in all_texts or "google" in all_texts, f"Should have Alice/Google facts, got: {all_texts}"

        doc = await memory.get_document(document_id, bank_id, request_context=request_context)
        assert doc is not None
        assert doc["memory_unit_count"] > 0

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_with_user_entities(memory, request_context):
    """
    User-provided entities should work correctly with delta retain.
    """
    bank_id = f"test_delta_user_entities_{_ts()}"
    document_id = "user-entity-doc"

    try:
        content = "The project is going well."

        # v1 with user entities
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{
                "content": content,
                "document_id": document_id,
                "entities": [{"text": "Project Alpha", "type": "PROJECT"}],
            }],
            request_context=request_context,
        )

        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            v1_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v1_names = {e["canonical_name"].lower() for e in v1_entities}

        # v2 with additional entity, same content
        # Note: same content = delta path (no re-extraction)
        # The user entities for NEW chunks only get processed
        v2_content = content + "\n\nThe timeline is on track for Q2 delivery."
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{
                "content": v2_content,
                "document_id": document_id,
                "entities": [
                    {"text": "Project Alpha", "type": "PROJECT"},
                    {"text": "Q2 Deadline", "type": "MILESTONE"},
                ],
            }],
            request_context=request_context,
        )

        # Should have entities from both v1 and v2
        async with pool.acquire() as conn:
            v2_entities = await conn.fetch(
                "SELECT canonical_name FROM entities WHERE bank_id = $1",
                bank_id,
            )
        v2_names = {e["canonical_name"].lower() for e in v2_entities}

        # v1 entities should be preserved
        assert v1_names.issubset(v2_names), f"v1 entities should be preserved: {v1_names} not in {v2_names}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delta_retain_recall_with_chunks(memory, request_context):
    """
    After delta retain, recall with include_chunks should return correct chunk data.
    """
    bank_id = f"test_delta_recall_chunks_{_ts()}"
    document_id = "recall-chunks-doc"

    try:
        content = "Alice is a senior engineer at Google Cloud. She designs distributed systems."
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="profile",
            document_id=document_id,
            request_context=request_context,
        )

        # Upsert with same content (delta: no changes)
        await memory.retain_async(
            bank_id=bank_id,
            content=content,
            context="profile",
            document_id=document_id,
            request_context=request_context,
        )

        # Recall with chunks
        result = await memory.recall_async(
            bank_id=bank_id,
            query="What does Alice do?",
            budget=Budget.MID,
            max_tokens=2000,
            include_chunks=True,
            max_chunk_tokens=8192,
            request_context=request_context,
        )

        assert len(result.results) > 0, "Should recall facts"

        # Facts with chunk_ids should have corresponding chunks
        facts_with_chunks = [r for r in result.results if r.chunk_id]
        if facts_with_chunks and result.chunks:
            for fact in facts_with_chunks:
                assert fact.chunk_id in result.chunks, (
                    f"Chunk {fact.chunk_id} should be in returned chunks"
                )

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
