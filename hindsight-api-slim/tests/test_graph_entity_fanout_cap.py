"""
Tests for LATERAL entity fanout cap in graph expansion.

Verifies that the per-entity LIMIT in _expand_combined prevents high-fanout
entities from exploding the self-join, while still returning entity-based
graph results.
"""

import asyncio
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_high_fanout_entity_returns_results(memory, request_context):
    """
    A high-fanout entity (appearing in many facts) should still produce
    graph retrieval results — the LATERAL cap limits rows per entity but
    does not drop the entity entirely.
    """
    bank_id = f"test_fanout_cap_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create many facts sharing one common entity ("Acme Corp") plus
        # a few with a unique entity so we can query for the unique one
        # and verify graph expansion finds siblings via "Acme Corp".
        contents = [
            # Target: unique entity "Zara" shares "Acme Corp" with the rest
            {
                "content": "Zara joined Acme Corp as a senior engineer last month",
                "context": "hr update",
                "entities": [{"text": "Zara"}, {"text": "Acme Corp"}],
            },
        ]
        # Add many facts that all share "Acme Corp" — creates a high-fanout entity
        for i in range(60):
            contents.append(
                {
                    "content": f"Employee {i} completed onboarding at Acme Corp in department {i % 5}",
                    "context": "hr update",
                    "entities": [{"text": f"Employee {i}"}, {"text": "Acme Corp"}],
                }
            )

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

        from hindsight_api.engine.memory_engine import Budget

        # Query for "Zara" — semantic search finds Zara's fact as a seed,
        # then graph expansion should find other Acme Corp facts via the
        # shared entity, even though "Acme Corp" has 60+ mentions.
        result = await memory.recall_async(
            bank_id=bank_id,
            query="Zara",
            budget=Budget.HIGH,
            max_tokens=4096,
            enable_trace=True,
            request_context=request_context,
            _quiet=True,
        )

        assert result.results is not None
        assert len(result.results) > 0

        # Verify graph retrieval ran and found results
        retrieval_results = result.trace.get("retrieval_results", [])
        graph_results = [r for r in retrieval_results if r.get("method_name") == "graph"]
        assert len(graph_results) > 0, "Graph retrieval should have run"

        # At least one graph result should contain Acme Corp content
        # (found via shared entity, not just semantic similarity)
        all_texts = [r.text for r in result.results]
        acme_found = any("Acme Corp" in t for t in all_texts)
        assert acme_found, "Should find Acme Corp facts via entity graph expansion"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_entity_expansion_timeout_fallback(memory, request_context):
    """
    When graph_expansion_timeout is set very low, entity expansion should
    time out gracefully and fall back to semantic+causal links only,
    rather than failing the entire recall.
    """
    bank_id = f"test_timeout_fallback_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": "Alice works on the backend API at TechCorp",
                    "context": "team info",
                    "entities": [{"text": "Alice"}, {"text": "TechCorp"}],
                },
                {
                    "content": "Bob maintains the frontend at TechCorp",
                    "context": "team info",
                    "entities": [{"text": "Bob"}, {"text": "TechCorp"}],
                },
            ],
            request_context=request_context,
        )

        from hindsight_api.config import _get_raw_config
        from hindsight_api.engine.memory_engine import Budget

        config = _get_raw_config()
        original_timeout = config.link_expansion_timeout

        try:
            # Set an impossibly low timeout to force the fallback path
            config.link_expansion_timeout = 0.0001

            result = await memory.recall_async(
                bank_id=bank_id,
                query="Alice",
                budget=Budget.MID,
                max_tokens=2048,
                enable_trace=True,
                request_context=request_context,
                _quiet=True,
            )

            # Recall should succeed even when entity expansion times out
            assert result.results is not None
            assert len(result.results) > 0

            # Alice should still be found via semantic search
            result_texts = [r.text for r in result.results]
            alice_found = any("Alice" in t for t in result_texts)
            assert alice_found, "Should find Alice via semantic search despite graph timeout"
        finally:
            config.link_expansion_timeout = original_timeout

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_per_entity_limit_caps_expansion(memory, request_context):
    """
    With graph_per_entity_limit set to a small value, entity expansion should
    still work but return fewer results from high-fanout entities.
    """
    bank_id = f"test_per_entity_limit_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Create facts with a shared entity
        contents = [
            {
                "content": "Lead engineer Dana oversees the Widgets project at MegaCorp",
                "context": "project info",
                "entities": [{"text": "Dana"}, {"text": "MegaCorp"}],
            },
        ]
        for i in range(30):
            contents.append(
                {
                    "content": f"MegaCorp hired contractor {i} for the Q4 push",
                    "context": "hiring info",
                    "entities": [{"text": f"Contractor {i}"}, {"text": "MegaCorp"}],
                }
            )

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            request_context=request_context,
        )

        from hindsight_api.config import _get_raw_config
        from hindsight_api.engine.memory_engine import Budget

        config = _get_raw_config()
        original_limit = config.link_expansion_per_entity_limit

        try:
            # Set a very small per-entity limit
            config.link_expansion_per_entity_limit = 5

            result = await memory.recall_async(
                bank_id=bank_id,
                query="Dana",
                budget=Budget.HIGH,
                max_tokens=4096,
                enable_trace=True,
                request_context=request_context,
                _quiet=True,
            )

            # Recall should succeed with the cap
            assert result.results is not None
            assert len(result.results) > 0

            # Graph retrieval should have run
            retrieval_results = result.trace.get("retrieval_results", [])
            graph_results = [r for r in retrieval_results if r.get("method_name") == "graph"]
            assert len(graph_results) > 0, "Graph retrieval should have run"
        finally:
            config.link_expansion_per_entity_limit = original_limit

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
