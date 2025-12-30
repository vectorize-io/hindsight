"""
Integration tests for the Feedback Signal API.

Tests the signal submission, fact stats, bank stats endpoints,
and the recall integration with usefulness boosting.
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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Where does Alice work?"},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) > 0
        fact_id = results[0]["id"]

        # Submit signal
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used", "confidence": 1.0}]},
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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who works at the company?"},
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) >= 2

        fact_ids = [r["id"] for r in results[:3]]

        # Submit batch signals
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": fact_ids[0], "signal_type": "used"},
                    {"fact_id": fact_ids[1] if len(fact_ids) > 1 else fact_ids[0], "signal_type": "ignored"},
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["signals_processed"] == 2

    @pytest.mark.asyncio
    async def test_submit_signal_with_query(self, api_client, test_bank_id):
        """Test submitting a signal with query for pattern tracking."""
        # Store a memory
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories",
            json={"items": [{"content": "The project deadline is next Friday."}]},
        )
        assert response.status_code == 200

        # Get fact_id
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "When is the deadline?"},
        )
        assert response.status_code == 200
        fact_id = response.json()["results"][0]["id"]

        # Submit signal with query
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {
                        "fact_id": fact_id,
                        "signal_type": "helpful",
                        "confidence": 0.9,
                        "query": "When is the deadline?",
                        "context": "User found this answer helpful",
                    }
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    @pytest.mark.asyncio
    async def test_signal_type_validation(self, api_client, test_bank_id):
        """Test that invalid signal types are rejected."""
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "invalid_type"}]},
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
                    {"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "used", "confidence": 1.5}
                ]
            },
        )
        assert response.status_code == 422

        # Confidence < 0.0 should fail
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": "00000000-0000-0000-0000-000000000000", "signal_type": "used", "confidence": -0.5}
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

        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who leads research?"},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit multiple signals
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used"}]},
        )
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "helpful"}]},
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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who works at the company?"},
        )
        results = response.json()["results"]
        fact_ids = [r["id"] for r in results[:2]]

        # Submit signals
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={
                "signals": [
                    {"fact_id": fact_ids[0], "signal_type": "used"},
                    {"fact_id": fact_ids[0], "signal_type": "helpful"},
                    {"fact_id": fact_ids[1] if len(fact_ids) > 1 else fact_ids[0], "signal_type": "ignored"},
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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who is the project manager?"},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit one 'used' signal (weight +1.0, delta = 1.0 * 1.0 * 0.1 = 0.1)
        # Expected: 0.5 + 0.1 = 0.6
        await api_client.post(
            f"/v1/default/banks/{test_bank_id}/signal",
            json={"signals": [{"fact_id": fact_id, "signal_type": "used", "confidence": 1.0}]},
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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who is the CFO?"},
        )
        fact_id = response.json()["results"][0]["id"]

        # Submit many 'helpful' signals (weight +1.5)
        # Each signal adds 1.5 * 1.0 * 0.1 = 0.15 to the score
        for _ in range(10):
            await api_client.post(
                f"/v1/default/banks/{test_bank_id}/signal",
                json={"signals": [{"fact_id": fact_id, "signal_type": "helpful", "confidence": 1.0}]},
            )

        # Get stats and verify score is clamped
        response = await api_client.get(f"/v1/default/banks/{test_bank_id}/facts/{fact_id}/stats")
        assert response.status_code == 200

        stats = response.json()
        assert stats["usefulness_score"] <= 1.0
        assert stats["usefulness_score"] >= 0.0


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
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who are the executives?"},
        )
        results = response.json()["results"]
        fact_ids = [r["id"] for r in results[:2]]

        # Make one fact more useful than the other
        for _ in range(3):
            await api_client.post(
                f"/v1/default/banks/{test_bank_id}/signal",
                json={"signals": [{"fact_id": fact_ids[0], "signal_type": "helpful"}]},
            )

        # Recall with usefulness boost
        response = await api_client.post(
            f"/v1/default/banks/{test_bank_id}/memories/recall",
            json={"query": "Who are the executives?", "boost_by_usefulness": True, "usefulness_weight": 0.5},
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
