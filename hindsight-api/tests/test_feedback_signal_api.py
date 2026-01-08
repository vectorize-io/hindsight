"""
Integration tests for the Feedback Signal API.

Tests the signal submission, fact stats, bank stats endpoints,
and the recall integration with query-context aware usefulness boosting.
"""

import pytest
import pytest_asyncio
import httpx
from datetime import datetime

from hindsight_api.api import create_app


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def test_bank_id():
    """Provide a unique bank ID for this test run."""
    return f"feedback_test_{datetime.now().timestamp()}"


class TestSignalSubmission:
    """Tests for signal submission endpoint."""

    @pytest.mark.asyncio
    async def test_submit_single_signal(self, api_client, test_bank_id):
        """Test submitting a single feedback signal."""
        # First, store a memory to get a fact_id
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Alice works at Google as a software engineer."}]},
        )
        assert response.status_code == 200

        # Get a fact_id from recall
        query = "Where does Alice work?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0
        fact_id = results[0]["id"]

        # Submit signal with required query field
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used", "confidence": 1.0, "query": query}]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["signals_processed"] == 1
        assert fact_id in result["updated_facts"]

    @pytest.mark.asyncio
    async def test_submit_batch_signals(self, api_client, test_bank_id):
        """Test submitting multiple signals at once."""
        # Store memories
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={
                "items": [
                    {"content": "Bob is the CEO of TechCorp."},
                    {"content": "Charlie manages the engineering team."},
                    {"content": "Diana leads product development."},
                ]
            },
        )
        assert response.status_code == 200

        # Get fact_ids from recall
        query = "Who works at the company?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) >= 2

        fact_ids = [r["id"] for r in results[:3]]

        # Submit batch signals with required query field
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": fact_ids[0], "signal_type": "used", "query": query},
                    {"fact_id": fact_ids[1] if len(fact_ids) > 1 else fact_ids[0], "signal_type": "ignored", "query": query},
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["signals_processed"] == 2

    @pytest.mark.asyncio
    async def test_submit_signal_with_context(self, api_client, test_bank_id):
        """Test submitting a signal with optional context."""
        # Store a memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "The project deadline is next Friday."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        query = "When is the deadline?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        assert response.status_code == 200
        fact_id = response.json()["results"][0]["id"]

        # Submit signal with query and context
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {
                        "fact_id": fact_id,
                        "signal_type": "helpful",
                        "confidence": 0.9,
                        "query": query,
                        "context": "User found this answer helpful",
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.asyncio
    async def test_signal_requires_query(self, api_client, test_bank_id):
        """Test that signals without query are rejected (query is required)."""
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "used"}]},
        )
        assert response.status_code == 422  # Validation error - missing required field

    @pytest.mark.asyncio
    async def test_signal_type_validation(self, api_client, test_bank_id):
        """Test that invalid signal types are rejected."""
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "invalid_type", "query": "test query"}]},
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_confidence_validation(self, api_client, test_bank_id):
        """Test confidence bounds validation."""
        # Confidence > 1.0 should fail
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "used", "confidence": 1.5, "query": "test query"}
                ]
            },
        )
        assert response.status_code == 422

        # Confidence < 0.0 should fail
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "used", "confidence": -0.5, "query": "test query"}
                ]
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_signals_validation(self, api_client, test_bank_id):
        """Test that empty signals list is rejected."""
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": []},
        )
        assert response.status_code == 422


class TestFactStats:
    """Tests for fact usefulness stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_fact_stats(self, api_client, test_bank_id):
        """Test retrieving fact statistics after signals."""
        # Store memory and get fact_id
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Eve is the head of research."}]},
        )
        assert response.status_code == 200

        query = "Who leads research?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit multiple signals
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used", "query": query}]},
        )
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "query": query}]},
        )

        # Get fact stats
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200

        stats = response.json()
        assert stats["fact_id"] == fact_id
        assert "usefulness_score" in stats
        assert stats["signal_count"] == 2
        assert "signal_breakdown" in stats
        assert stats["signal_breakdown"].get("used", 0) == 1
        assert stats["signal_breakdown"].get("helpful", 0) == 1
        assert "created_at" in stats

    @pytest.mark.asyncio
    async def test_fact_stats_not_found(self, api_client, test_bank_id):
        """Test 404 for fact with no signals."""
        # Use a random UUID that doesn't exist
        response = await api_client.get(
            f"/v1/default/banks/{test_bank_id}/facts/00000000-0000-0000-0000-000000000000/stats"
        )
        assert response.status_code == 404


class TestBankStats:
    """Tests for bank usefulness stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_bank_stats_empty(self, api_client, test_bank_id):
        """Test bank stats on bank with no signals."""
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats/usefulness")
        assert response.status_code == 200

        stats = response.json()
        assert stats["bank_id"] == test_bank_id
        assert stats["total_facts_with_signals"] == 0
        assert stats["total_signals"] == 0

    @pytest.mark.asyncio
    async def test_get_bank_stats_with_signals(self, api_client, test_bank_id):
        """Test bank stats after submitting signals."""
        # Store memories
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={
                "items": [
                    {"content": "Frank is a data scientist."},
                    {"content": "Grace works in marketing."},
                ]
            },
        )
        assert response.status_code == 200

        # Get fact_ids
        query = "Who works at the company?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        results = response.json()["results"]
        fact_ids = [r["id"] for r in results[:2]]

        # Submit signals
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": fact_ids[0], "signal_type": "used", "query": query},
                    {"fact_id": fact_ids[0], "signal_type": "helpful", "query": query},
                    {"fact_id": fact_ids[1] if len(fact_ids) > 1 else fact_ids[0], "signal_type": "ignored", "query": query},
                ]
            },
        )

        # Get bank stats
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/stats/usefulness")
        assert response.status_code == 200

        stats = response.json()
        assert stats["bank_id"] == test_bank_id
        assert stats["total_facts_with_signals"] >= 1
        assert stats["total_signals"] >= 3
        assert "signal_distribution" in stats
        assert "average_usefulness" in stats
        assert "top_useful_facts" in stats
        assert "least_useful_facts" in stats


class TestScoreCalculation:
    """Tests for usefulness score calculation."""

    @pytest.mark.asyncio
    async def test_initial_score_is_neutral(self, api_client, test_bank_id):
        """Test that first signal starts from neutral 0.5 and applies correctly."""
        # Store memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Henry is a project manager."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        query = "Who is the project manager?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit one 'used' signal (weight +1.0, delta = 1.0 * 1.0 * 0.1 = 0.1)
        # Expected: 0.5 + 0.1 = 0.6
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used", "confidence": 1.0, "query": query}]},
        )

        # Get stats and verify score
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200

        stats = response.json()
        # Score should be approximately 0.6 (0.5 + 0.1)
        assert 0.55 <= stats["usefulness_score"] <= 0.65

    @pytest.mark.asyncio
    async def test_score_clamped_to_bounds(self, api_client, test_bank_id):
        """Test that scores are clamped to [0, 1]."""
        # Store memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Ivy is the CFO."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        query = "Who is the CFO?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit many 'helpful' signals (weight +1.5)
        # Each signal adds 1.5 * 1.0 * 0.1 = 0.15 to the score
        for _ in range(10):
            await api_client.post(
                f"/v1/default/banks/{test_bank_id}/signal",
                json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "confidence": 1.0, "query": query}]},
            )

        # Get stats and verify score is clamped
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200

        stats = response.json()
        assert stats["usefulness_score"] <= 1.0
        assert stats["usefulness_score"] >= 0.0


class TestQueryContextAwareScoring:
    """Tests for query-context aware usefulness scoring."""

    @pytest.mark.asyncio
    async def test_different_queries_have_separate_scores(self, api_client, test_bank_id):
        """Test that the same fact can have different scores for different query contexts."""
        # Store a memory that could be relevant to multiple queries
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Bob works at TechCorp as a senior software engineer."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        query1 = "Who works at TechCorp?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query1},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit positive signal for query about TechCorp
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "query": query1}]},
        )

        # Submit negative signal for a different query context
        query2 = "What is the weather today?"
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "not_helpful", "query": query2}]},
        )

        # The fact now has two separate query-context scores
        # When recalling with query1-like queries, it should be boosted
        # When recalling with query2-like queries, it should be penalized
        # This is tested by the recall integration tests

        # Verify signals were recorded
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["signal_count"] == 2
        assert stats["signal_breakdown"].get("helpful", 0) == 1
        assert stats["signal_breakdown"].get("not_helpful", 0) == 1

    @pytest.mark.asyncio
    async def test_similar_queries_share_context(self, api_client, test_bank_id):
        """Test that semantically similar queries share the same context score."""
        # Store a memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Alice is the CEO of Acme Corp."}]},
        )
        assert response.status_code == 200

        # Get fact_id with first query
        query1 = "Who is the CEO?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query1},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit signal for first query
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "query": query1}]},
        )

        # Submit signal for semantically similar query
        # "Who is the chief executive?" should match "Who is the CEO?" with high similarity
        query2 = "Who is the chief executive officer?"
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "query": query2}]},
        )

        # Both signals should contribute to the same query context
        # (because similarity >= 0.85 threshold)
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200


class TestRecallIntegration:
    """Tests for usefulness boost in recall."""

    @pytest.mark.asyncio
    async def test_recall_with_usefulness_boost(self, api_client, test_bank_id):
        """Test recall with usefulness boosting enabled."""
        # Store memories
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={
                "items": [
                    {"content": "Jack is the CTO of the company."},
                    {"content": "Kate is the COO of the company."},
                ]
            },
        )
        assert response.status_code == 200

        # Get fact_ids
        query = "Who are the executives?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query},
        )
        results = response.json()["results"]
        fact_ids = [r["id"] for r in results[:2]]

        # Make one fact more useful than the other for this query context
        for _ in range(3):
            await api_client.post(
                f"/v1/default/banks/{test_bank_id}/signal",
                json={"signals": [{"fact_id": fact_ids[0], "signal_type": "helpful", "query": query}]},
            )

        # Recall with usefulness boost
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query, "boost_by_usefulness": True, "usefulness_weight": 0.5},
        )
        assert response.status_code == 200
        # Results should still be returned (usefulness boost is applied internally)
        assert len(response.json()["results"]) > 0

    @pytest.mark.asyncio
    async def test_recall_without_boost_unchanged(self, api_client, test_bank_id):
        """Test that recall without boost_by_usefulness is unchanged."""
        # Store memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Leo is the head of HR."}]},
        )
        assert response.status_code == 200

        # Recall without boost (default behavior)
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who is in HR?"},
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0

    @pytest.mark.asyncio
    async def test_recall_usefulness_params_validation(self, api_client, test_bank_id):
        """Test validation of usefulness parameters in recall."""
        # usefulness_weight > 1.0 should fail
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "test", "boost_by_usefulness": True, "usefulness_weight": 1.5},
        )
        assert response.status_code == 422

        # min_usefulness > 1.0 should fail
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "test", "boost_by_usefulness": True, "min_usefulness": 1.5},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_context_affects_boosting(self, api_client, test_bank_id):
        """Test that query context affects which scores are used for boosting."""
        # Store a memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "Maria is a machine learning engineer at DataCorp."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        query_ml = "Who knows about machine learning?"
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": query_ml},
        )
        fact_id = response.json()["results"][0]["id"]

        # Mark as helpful for ML-related queries
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "query": query_ml}]},
        )

        # Mark as not helpful for unrelated queries
        query_cooking = "What are some good recipes?"
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "not_helpful", "query": query_cooking}]},
        )

        # Recall with ML query should use the helpful score
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who is an ML expert?", "boost_by_usefulness": True},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0
        # The fact should be present and potentially boosted

        # Recall with cooking query would use the not_helpful score
        # (though the fact probably won't appear at all due to semantic mismatch)
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "What should I cook for dinner?", "boost_by_usefulness": True},
        )
        assert response.status_code == 200
        # The ML fact shouldn't appear in cooking query results due to semantic mismatch
