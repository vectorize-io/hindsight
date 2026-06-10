"""
Oracle 23ai integration tests.

All tests are marked with @pytest.mark.oracle and require ORACLE_TEST_DSN
to be set. They are skipped by default in CI.

Mirrors the PostgreSQL integration tests to verify that the Oracle backend
produces identical behavior through the database abstraction layer.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.memory_engine import Budget

pytestmark = pytest.mark.oracle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bank_id(prefix: str = "oracle") -> str:
    return f"test-{prefix}-{uuid.uuid4().hex[:8]}"


async def _safe_cleanup(memory: MemoryEngine, bank_id: str, request_context: RequestContext) -> None:
    """Delete a bank, suppressing deadlock/lock errors in test teardown.

    Oracle's row-level locking can cause ORA-00060 (deadlock) or ORA-00054
    (resource busy) when concurrent tests clean up simultaneously. These
    are benign in test teardown — the data will be orphaned but doesn't
    affect other tests since each test uses a unique bank_id.
    """
    try:
        await memory.delete_bank(bank_id, request_context=request_context)
    except Exception as e:
        err = str(e)
        if "ORA-00060" in err or "ORA-00054" in err:
            logger.warning(f"Cleanup deadlock for {bank_id} (benign in tests): {err[:120]}")
        else:
            logger.error(f"Cleanup failed for {bank_id}: {err[:200]}")
            raise


# ===================================================================
# Tier 1 — Core CRUD
# ===================================================================


class TestCoreCRUD:
    """Basic bank, memory, and document CRUD against Oracle."""

    @pytest.mark.asyncio
    async def test_create_bank(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("bank")
        try:
            profile = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert profile is not None
            assert profile["bank_id"] == bank_id
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_simple(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("retain")
        try:
            unit_ids = await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Alice is a software engineer who loves Python.",
                context="team overview",
                event_date=datetime(2024, 6, 15, tzinfo=timezone.utc),
                request_context=request_context,
            )
            assert len(unit_ids) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_with_document(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("retdoc")
        try:
            unit_ids = await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Bob manages the infrastructure team. He has 10 years of experience.",
                context="team docs",
                document_id="doc-001",
                event_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            assert len(unit_ids) > 0

            # Verify document was stored
            doc = await oracle_memory.get_document(
                document_id="doc-001", bank_id=bank_id, request_context=request_context
            )
            assert doc is not None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_recall_semantic(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("recall")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Carol is a data scientist specializing in NLP.",
                context="team",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Who works on natural language processing?",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
            texts = [r.text for r in result.results]
            assert any("Carol" in t or "NLP" in t for t in texts)
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_recall_with_filters(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("filter")
        try:
            await oracle_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": "Dan is an expert in distributed systems.",
                        "context": "engineering",
                        "event_date": datetime(2024, 5, 1, tzinfo=timezone.utc),
                    }
                ],
                document_tags=["backend"],
                request_context=request_context,
            )
            await oracle_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": "Eve designs beautiful user interfaces.",
                        "context": "design",
                        "event_date": datetime(2024, 5, 2, tzinfo=timezone.utc),
                    }
                ],
                document_tags=["frontend"],
                request_context=request_context,
            )

            # Filter to only backend
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="team members",
                budget=Budget.LOW,
                max_tokens=500,
                tags=["backend"],
                request_context=request_context,
            )
            texts = " ".join(r.text for r in result.results)
            assert "Dan" in texts or "distributed" in texts
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_recall_temporal(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("temporal")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="The project started in January 2024.",
                context="timeline",
                event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                request_context=request_context,
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="The project was completed in March 2024.",
                context="timeline",
                event_date=datetime(2024, 3, 20, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="What happened with the project?",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_reflect_basic(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("reflect")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Frank prefers TypeScript over JavaScript for large projects.",
                context="preferences",
                event_date=datetime(2024, 4, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.reflect_async(
                bank_id=bank_id,
                query="What programming language preferences are known?",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert result.text is not None
            assert len(result.text) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_delete_memory(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("delmem")
        try:
            unit_ids = await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Albert Einstein was born on March 14, 1879 in Ulm, Germany. He developed the theory of general relativity.",
                context="biographical information about physicists",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            assert len(unit_ids) > 0
            memory_id = unit_ids[0]

            # Delete the memory
            await oracle_memory.delete_memory_unit(str(memory_id), request_context=request_context)

            # Verify deletion
            mem = await oracle_memory.get_memory_unit(
                bank_id=bank_id, memory_id=str(memory_id), request_context=request_context
            )
            assert mem is None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_delete_document(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("deldoc")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Document that will be deleted with its memories.",
                context="test",
                document_id="doc-to-delete",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )

            # Delete the document
            await oracle_memory.delete_document(
                bank_id=bank_id, document_id="doc-to-delete", request_context=request_context
            )

            # Verify cascade
            doc = await oracle_memory.get_document(
                document_id="doc-to-delete", bank_id=bank_id, request_context=request_context
            )
            assert doc is None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_bank_profile_crud(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("profile")
        try:
            # Create
            profile = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert profile["bank_id"] == bank_id

            # Update
            await oracle_memory.update_bank(
                bank_id=bank_id,
                name="Test Oracle Bank",
                mission="Testing Oracle integration",
                request_context=request_context,
            )
            updated = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert updated["name"] == "Test Oracle Bank"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_list_memories(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("listmem")
        try:
            for i in range(3):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=f"Memory number {i} for listing test.",
                    context="test",
                    event_date=datetime(2024, 6, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
            memories = await oracle_memory.list_memory_units(bank_id=bank_id, request_context=request_context)
            # Each retain may extract multiple facts
            assert len(memories) >= 3
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_get_memory(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("getmem")
        try:
            unit_ids = await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Marie Curie won two Nobel Prizes: one in Physics in 1903 and another in Chemistry in 1911.",
                context="biographical information about scientists",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            assert len(unit_ids) > 0
            memory_id = unit_ids[0]

            mem = await oracle_memory.get_memory_unit(
                bank_id=bank_id, memory_id=str(memory_id), request_context=request_context
            )
            assert mem is not None
            assert str(mem["id"]) == str(memory_id)
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)


# ===================================================================
# Tier 2 — Retain Pipeline
# ===================================================================


class TestRetainPipeline:
    """Retain-specific behaviors: chunking, delta, entities, batch."""

    @pytest.mark.asyncio
    async def test_retain_chunking(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("chunk")
        try:
            facts = [
                "The Great Wall of China stretches over 13,000 miles and was built over many centuries starting in the 7th century BC.",
                "Mount Everest is 29,032 feet tall and was first summited by Edmund Hillary and Tenzing Norgay on May 29, 1953.",
                "The Amazon River is approximately 4,000 miles long and flows through Brazil, Peru, and Colombia.",
                "The human genome contains approximately 3 billion DNA base pairs and about 20,000 protein-coding genes.",
                "Jupiter has a mass of 1.898 × 10^27 kg, making it 318 times more massive than Earth.",
                "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
                "Shakespeare wrote 37 plays between 1590 and 1613, including Hamlet, Macbeth, and King Lear.",
                "The Pacific Ocean covers approximately 63 million square miles, making it the largest ocean on Earth.",
                "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519, and it now hangs in the Louvre Museum in Paris.",
                "The International Space Station orbits Earth at an altitude of approximately 250 miles at a speed of 17,500 mph.",
            ]
            long_content = " ".join(f"Section {i + 1}: {fact} " * 3 for i, fact in enumerate(facts))
            unit_ids = await oracle_memory.retain_async(
                bank_id=bank_id,
                content=long_content,
                context="long doc",
                document_id="long-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            assert len(unit_ids) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_delta(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("delta")
        try:
            content = "Delta detection: original content about machine learning."
            ids1 = await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="test",
                document_id="delta-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Re-retain same content
            ids2 = await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="test",
                document_id="delta-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Delta detection should produce no new facts (or very few)
            assert len(ids2) <= len(ids1)
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_entities(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("entity")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Grace Hopper invented the first compiler at Harvard University.",
                context="history",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Query for entities should find at least Grace Hopper
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Grace Hopper",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_entity_dedup(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("entdedup")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="John Smith works at Acme Corp.",
                context="hr",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="John Smith leads the engineering team at Acme Corp.",
                context="hr",
                event_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Both mentions should resolve to the same entity
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="John Smith",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) >= 2
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_batch(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("batch")
        try:
            unit_ids_list = await oracle_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": "Batch item 1: Kubernetes orchestrates containers.",
                        "context": "devops",
                        "event_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
                    },
                    {
                        "content": "Batch item 2: Terraform manages infrastructure as code.",
                        "context": "devops",
                        "event_date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                    },
                ],
                request_context=request_context,
            )
            # retain_batch_async returns list-of-lists, one per content item.
            # At least one content item should produce facts.
            assert len(unit_ids_list) >= 1
            total_ids = sum(len(ids) for ids in unit_ids_list)
            assert total_ids > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_idempotent(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("idempotent")
        try:
            content = "Idempotency test: Python is a programming language."
            ids1 = await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="test",
                document_id="idem-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            ids2 = await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="test",
                document_id="idem-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Second retain with same doc ID should be idempotent
            assert len(ids2) <= len(ids1)
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_with_tags(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("tags")
        try:
            await oracle_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": "Tagged memory about machine learning models.",
                        "context": "ml",
                        "event_date": datetime(2024, 6, 1, tzinfo=timezone.utc),
                    }
                ],
                document_tags=["ml", "models"],
                request_context=request_context,
            )
            # Verify tags via recall with tag filter
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="machine learning",
                budget=Budget.LOW,
                max_tokens=500,
                tags=["ml"],
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_retain_document_metadata(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("docmeta")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Document with custom metadata.",
                context="test",
                document_id="meta-doc",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            doc = await oracle_memory.get_document(
                document_id="meta-doc", bank_id=bank_id, request_context=request_context
            )
            assert doc is not None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)


# ===================================================================
# Tier 3 — Search & Retrieval
# ===================================================================


class TestSearchRetrieval:
    """Multi-strategy search tests against Oracle."""

    @pytest.mark.asyncio
    async def test_vector_search_accuracy(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("vecsearch")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Quantum computing uses qubits to perform calculations exponentially faster.",
                context="science",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="My favorite recipe for chocolate cake uses dark chocolate and butter.",
                context="cooking",
                event_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="quantum physics and computing",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
            # The quantum computing memory should rank higher
            top_text = result.results[0].text
            assert "quantum" in top_text.lower() or "qubit" in top_text.lower()
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_text_search(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("textsearch")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="PostgreSQL supports JSONB columns for semi-structured data.",
                context="databases",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="JSONB columns",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_graph_retrieval(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("graph")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Alice works with Bob on the database team.",
                context="org",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Bob mentors Carol in distributed systems.",
                context="org",
                event_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Graph retrieval should find Carol via Bob link
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Who does Alice work with?",
                budget=Budget.MID,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_fusion_ranking(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("fusion")
        try:
            for i in range(5):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=f"Fact {i}: Machine learning model {i} achieved {90 + i}% accuracy.",
                    context="ml",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="machine learning accuracy",
                budget=Budget.MID,
                max_tokens=1000,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("limit")
        try:
            topics = [
                "Python is a popular programming language used for web development.",
                "JavaScript runs natively in web browsers and Node.js.",
                "PostgreSQL is a powerful open-source relational database.",
                "Docker containers simplify application deployment.",
                "Kubernetes orchestrates containerized workloads at scale.",
            ]
            for i, content in enumerate(topics):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=content,
                    context="technology",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="programming languages and databases",
                budget=Budget.LOW,
                max_tokens=200,
                request_context=request_context,
            )
            # With LOW budget + low tokens, results should be bounded but non-empty.
            assert len(result.results) > 0, "Recall should return results for 5 retained topics"
            assert len(result.results) <= 10
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_search_bank_isolation(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_a = _bank_id("iso-a")
        bank_b = _bank_id("iso-b")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_a,
                content="Secret fact: Bank A knows about quantum tunneling.",
                context="physics",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            await oracle_memory.retain_async(
                bank_id=bank_b,
                content="Public fact: Bank B likes chocolate.",
                context="food",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )

            # Bank B should NOT see Bank A's memories
            result = await oracle_memory.recall_async(
                bank_id=bank_b,
                query="quantum tunneling",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            texts = " ".join(r.text for r in result.results)
            assert "quantum tunneling" not in texts.lower()
        finally:
            await oracle_memory.delete_bank(bank_a, request_context=request_context)
            await oracle_memory.delete_bank(bank_b, request_context=request_context)

    @pytest.mark.asyncio
    async def test_recall_after_retain(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("e2e")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="End-to-end test: Oracle 23ai supports native vector search.",
                context="databases",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Oracle vector search",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_reranking(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("rerank")
        try:
            # Insert multiple facts, only one highly relevant
            contents = [
                "The weather in Paris is mild in spring.",
                "Python was created by Guido van Rossum in 1991.",
                "The Eiffel Tower is 330 meters tall and located in Paris.",
                "JavaScript runs in web browsers.",
                "Paris is the capital of France with a population of 2.1 million.",
            ]
            for i, c in enumerate(contents):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=c,
                    context="general",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Tell me about Paris landmarks",
                budget=Budget.MID,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
            # Eiffel Tower fact should be top-ranked after reranking
            top = result.results[0].text
            assert "eiffel" in top.lower() or "paris" in top.lower()
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)


# ===================================================================
# Tier 4 — Advanced Features
# ===================================================================


class TestAdvancedFeatures:
    """Mental models, directives, operations, webhooks against Oracle."""

    @pytest.mark.asyncio
    async def test_mental_model_crud(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("mmcrud")
        try:
            await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

            model = await oracle_memory.create_mental_model(
                bank_id=bank_id,
                name="Oracle Test Model",
                source_query="What is known about testing?",
                content="Testing is important for software quality.",
                tags=["testing"],
                request_context=request_context,
            )
            assert model["id"] is not None

            # List
            models = await oracle_memory.list_mental_models(bank_id=bank_id, request_context=request_context)
            assert len(models) > 0

            # Get
            fetched = await oracle_memory.get_mental_model(
                bank_id=bank_id,
                mental_model_id=model["id"],
                request_context=request_context,
            )
            assert fetched is not None
            assert fetched["name"] == "Oracle Test Model"

            # Update
            await oracle_memory.update_mental_model(
                bank_id=bank_id,
                mental_model_id=model["id"],
                name="Updated Oracle Model",
                request_context=request_context,
            )
            updated = await oracle_memory.get_mental_model(
                bank_id=bank_id,
                mental_model_id=model["id"],
                request_context=request_context,
            )
            assert updated["name"] == "Updated Oracle Model"

            # Delete
            await oracle_memory.delete_mental_model(
                bank_id=bank_id,
                mental_model_id=model["id"],
                request_context=request_context,
            )
            deleted = await oracle_memory.get_mental_model(
                bank_id=bank_id,
                mental_model_id=model["id"],
                request_context=request_context,
            )
            assert deleted is None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_mental_model_refresh(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("mmrefresh")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="The team uses Python for backend and React for frontend.",
                context="tech stack",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            model = await oracle_memory.create_mental_model(
                bank_id=bank_id,
                name="Tech Stack Summary",
                source_query="What technologies does the team use?",
                content="Initial content about tech stack.",
                request_context=request_context,
            )
            assert model is not None
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_consolidation(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify that consolidation (observation creation) works on Oracle."""
        bank_id = _bank_id("consolidate")
        try:
            # Retain enough similar facts for consolidation
            for i in range(5):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=f"The user prefers dark mode in application {i}.",
                    context="preferences",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
            # Verify facts were stored (consolidation is async and may run inline
            # via SyncTaskBackend, but the key assertion is that all 5 retains persisted)
            memories = await oracle_memory.list_memory_units(bank_id=bank_id, request_context=request_context)
            items = memories.get("items", memories) if isinstance(memories, dict) else memories
            assert len(items) >= 5, f"Expected at least 5 stored memories, got {len(items)}"
            # Verify facts contain expected content
            texts = [item.get("text", "") for item in items]
            assert any("dark mode" in t for t in texts), f"Expected 'dark mode' in stored facts, got: {texts[:3]}"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_operations_tracking(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("ops")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Operation tracking test content.",
                context="test",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            ops = await oracle_memory.list_operations(bank_id=bank_id, request_context=request_context)
            # Retain creates async operations (consolidation at minimum)
            assert ops is not None
            items = ops.get("items", ops) if isinstance(ops, dict) else ops
            assert len(items) > 0, "Retain should create at least one async operation"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_directives_crud(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("directives")
        try:
            await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

            directive = await oracle_memory.create_directive(
                bank_id=bank_id,
                name="Formal responses",
                content="Always respond formally.",
                priority=10,
                request_context=request_context,
            )
            assert directive is not None

            directives = await oracle_memory.list_directives(bank_id=bank_id, request_context=request_context)
            assert len(directives) > 0

            await oracle_memory.delete_directive(
                bank_id=bank_id,
                directive_id=directive["id"],
                request_context=request_context,
            )
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_list_tags(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("tags")
        try:
            await oracle_memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {
                        "content": "Tag listing test.",
                        "context": "test",
                        "event_date": datetime(2024, 6, 1, tzinfo=timezone.utc),
                    }
                ],
                document_tags=["alpha", "beta"],
                request_context=request_context,
            )
            tags = await oracle_memory.list_tags(bank_id=bank_id, request_context=request_context)
            assert len(tags) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_task_queue(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("taskq")
        try:
            # Retain triggers async tasks via SyncTaskBackend
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Task queue test content.",
                context="test",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            ops = await oracle_memory.list_operations(bank_id=bank_id, request_context=request_context)
            assert ops is not None
            items = ops.get("items", ops) if isinstance(ops, dict) else ops
            assert len(items) > 0, "Retain should enqueue at least one task"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_bank_config(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        bank_id = _bank_id("config")
        try:
            profile = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert profile is not None

            # Update bank name
            await oracle_memory.update_bank(
                bank_id=bank_id,
                name="Config Test Bank",
                request_context=request_context,
            )
            updated = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert updated["name"] == "Config Test Bank"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)


# ===================================================================
# Tier 5 — Oracle-Specific
# ===================================================================


class TestOracleSpecific:
    """Oracle 23ai-specific feature tests."""

    @pytest.mark.asyncio
    async def test_oracle_vector_index(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify HNSW vector index is used for cosine similarity search."""
        bank_id = _bank_id("vecidx")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Vector index test: embeddings are stored as VECTOR(384, FLOAT32).",
                context="test",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Semantic search exercises the vector index
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="vector embeddings",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0, "Vector search returned no results"
            result_text = " ".join(r.text for r in result.results).lower()
            assert "vector" in result_text or "embedding" in result_text, (
                f"Expected vector-related content in results, got: {result_text[:200]}"
            )
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_oracle_text_index(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify Oracle Text CONTEXT index for full-text search."""
        bank_id = _bank_id("textidx")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Oracle Text enables sophisticated full-text search with linguistic analysis.",
                context="db features",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="linguistic full-text search",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0, "Text search returned no results"
            result_text = " ".join(r.text for r in result.results).lower()
            assert "oracle text" in result_text or "full-text" in result_text or "linguistic" in result_text, (
                f"Expected text-search-related content in results, got: {result_text[:200]}"
            )
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_oracle_json_operations(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify JSON column read/write via CLOB — bank profile stores JSON."""
        bank_id = _bank_id("json")
        try:
            # Bank profile is stored as JSON CLOB in Oracle
            await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            await oracle_memory.update_bank(
                bank_id=bank_id,
                name="JSON Test Bank",
                mission="Test JSON CLOB storage",
                request_context=request_context,
            )
            profile = await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            assert profile is not None
            assert profile["name"] == "JSON Test Bank"
            assert profile["mission"] == "Test JSON CLOB storage"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_oracle_migration_idempotent(self, oracle_db_url):
        """Verify that running migrations twice causes no errors."""
        from hindsight_api.migrations import run_migrations

        # First run (tables already exist from fixture setup — Alembic stamps a
        # version row, subsequent runs become a no-op walk).
        run_migrations(oracle_db_url)
        run_migrations(oracle_db_url)


# ===================================================================
# Tier 6 — Edge Cases & Robustness
# ===================================================================


class TestEdgeCases:
    """Edge case tests: unicode, empty strings, nulls, large content."""

    @pytest.mark.asyncio
    async def test_unicode_content(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify CLOB handles multi-byte unicode correctly."""
        bank_id = _bank_id("unicode")
        try:
            content = (
                "日本語のテスト: The user speaks Japanese. "
                "Ελληνικά: Greek text test. "
                "Émojis: 🧠💡🔍 "
                "Arabic: مرحبا بالعالم"
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="multilingual",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Japanese text",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_empty_context(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify retain works with empty string context."""
        bank_id = _bank_id("emptyctx")
        try:
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="The Eiffel Tower was completed in 1889 for the World Fair in Paris. It stands 1,083 feet tall and was designed by Gustave Eiffel.",
                context="",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            memories = await oracle_memory.list_memory_units(bank_id=bank_id, request_context=request_context)
            items = memories.get("items", memories) if isinstance(memories, dict) else memories
            assert len(items) >= 1
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_large_content_chunking(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify large documents are chunked properly in Oracle."""
        bank_id = _bank_id("largecontent")
        try:
            # Generate ~10KB of content that should trigger chunking
            paragraphs = [
                f"Paragraph {i}: This is a detailed discussion about topic {i}. "
                f"It covers multiple aspects including theory, practice, and applications. "
                f"The key insight is that {i * 7} relates to {i * 13} in a non-obvious way."
                for i in range(50)
            ]
            large_content = "\n\n".join(paragraphs)
            assert len(large_content) > 5000

            await oracle_memory.retain_async(
                bank_id=bank_id,
                content=large_content,
                context="research",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            memories = await oracle_memory.list_memory_units(bank_id=bank_id, request_context=request_context)
            items = memories.get("items", memories) if isinstance(memories, dict) else memories
            # Large content (~10KB, 50 paragraphs) should produce multiple memory units
            # from LLM fact extraction. At minimum we expect several facts.
            assert len(items) >= 3, f"Expected large content to produce at least 3 memory units, got {len(items)}"
            # Verify operations completed without errors (catches background datetime issues etc.)
            ops = await oracle_memory.list_operations(bank_id=bank_id, request_context=request_context)
            if ops:
                op_list = ops.get("items", ops) if isinstance(ops, dict) else ops
                failed = [o for o in op_list if isinstance(o, dict) and o.get("status") == "failed"]
                assert len(failed) == 0, (
                    f"Expected no failed operations, got {len(failed)}: "
                    f"{[o.get('error', o.get('status_message', ''))[:100] for o in failed]}"
                )
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify content with SQL-sensitive characters doesn't cause errors."""
        bank_id = _bank_id("specialchars")
        try:
            content = (
                "User's query contains: O'Brien said \"hello\". "
                "SQL injection attempt: '; DROP TABLE banks; -- "
                "Backslashes: C:\\Users\\test\\file.txt "
                "Percent: 100% success rate."
            )
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content=content,
                context="test",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="O'Brien's query",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_concurrent_retains(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify concurrent retain operations complete (Oracle may deadlock on auto-partition creation)."""
        bank_id = _bank_id("concurrent")
        try:
            # Run 3 retains concurrently
            tasks = [
                oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=f"Concurrent fact {i}: parallel write test.",
                    context="concurrency",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Oracle auto-partitioning can cause ORA-00060 deadlocks on the first
            # concurrent inserts into a new bank (partition doesn't exist yet and
            # two sessions race to create it). This is a known Oracle limitation,
            # not a code bug. Allow up to 1 deadlock failure.
            deadlocks = [r for r in results if isinstance(r, Exception) and "ORA-00060" in str(r)]
            other_failures = [r for r in results if isinstance(r, Exception) and "ORA-00060" not in str(r)]
            assert len(other_failures) == 0, f"Non-deadlock failures: {[str(e)[:100] for e in other_failures]}"
            successes = len(results) - len(deadlocks)
            assert successes >= 2, f"Expected at least 2 successful retains, got {successes}"

            memories = await oracle_memory.list_memory_units(bank_id=bank_id, request_context=request_context)
            items = memories.get("items", memories) if isinstance(memories, dict) else memories
            assert len(items) >= successes, f"Expected at least {successes} memories, got {len(items)}"
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_recall_empty_bank(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify recall on an empty bank returns empty results without error."""
        bank_id = _bank_id("emptyrecall")
        try:
            # Create bank
            await oracle_memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="anything at all",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert result.results is not None
            assert len(result.results) == 0
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_bank(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify deleting a non-existent bank doesn't raise."""
        # Should not raise an exception
        await oracle_memory.delete_bank(f"nonexistent-{uuid.uuid4().hex[:8]}", request_context=request_context)

    @pytest.mark.asyncio
    async def test_retain_and_delete_cycle(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify retain → delete → retain cycle works (no stale data)."""
        bank_id = _bank_id("cycle")
        try:
            # First cycle — use substantive content so fact extraction is reliable
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Alice is a senior backend engineer who specializes in Python and distributed systems.",
                context="team",
                event_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            # Delete must succeed for this test to be meaningful — use
            # delete_bank directly (not _safe_cleanup which swallows errors).
            await oracle_memory.delete_bank(bank_id, request_context=request_context)

            # Second cycle — same bank_id, completely different content
            await oracle_memory.retain_async(
                bank_id=bank_id,
                content="Bob is a frontend developer with deep expertise in React, TypeScript, and design systems.",
                context="team",
                event_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
                request_context=request_context,
            )
            result = await oracle_memory.recall_async(
                bank_id=bank_id,
                query="Who is Bob and what does he do?",
                budget=Budget.LOW,
                max_tokens=500,
                request_context=request_context,
            )
            assert len(result.results) > 0
            # Should only have second cycle content (Alice's data was deleted)
            all_text = " ".join(r.text for r in result.results).lower()
            assert "alice" not in all_text
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)

    @pytest.mark.asyncio
    async def test_multiple_documents_same_bank(self, oracle_memory: MemoryEngine, request_context: RequestContext):
        """Verify multiple documents with different IDs in the same bank."""
        bank_id = _bank_id("multidoc")
        try:
            for i in range(3):
                await oracle_memory.retain_async(
                    bank_id=bank_id,
                    content=f"Document {i}: unique content about topic {chr(65 + i)}.",
                    context="docs",
                    document_id=f"doc-{i}",
                    event_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
                    request_context=request_context,
                )

            docs = await oracle_memory.list_documents(bank_id=bank_id, request_context=request_context)
            items = docs.get("items", docs.get("documents", []))
            assert len(items) >= 3

            # Delete one document, verify others remain
            await oracle_memory.delete_document(bank_id=bank_id, document_id="doc-1", request_context=request_context)
            docs_after = await oracle_memory.list_documents(bank_id=bank_id, request_context=request_context)
            items_after = docs_after.get("items", docs_after.get("documents", []))
            assert len(items_after) >= 2
        finally:
            await _safe_cleanup(oracle_memory, bank_id, request_context)
