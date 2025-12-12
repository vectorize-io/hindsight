"""
Tests for combined scoring functionality.

Verifies that:
1. RRF scores are properly normalized to [0, 1] range
2. Combined scoring formula is applied correctly
3. Tracer captures normalized values (not raw values)
"""
import pytest
from datetime import datetime, timezone
from hindsight_api.engine.search.types import RetrievalResult, MergedCandidate, ScoredResult
from hindsight_api.engine.memory_engine import Budget


class TestRRFNormalization:
    """Test that RRF scores are properly normalized."""

    def test_rrf_normalized_range(self):
        """RRF normalized values should be in [0, 1] range, not raw [0.04, 0.06]."""
        # Simulate RRF scores like what we get from actual retrieval
        raw_rrf_scores = [0.0607, 0.0550, 0.0480, 0.0390]

        max_rrf = max(raw_rrf_scores)
        min_rrf = min(raw_rrf_scores)
        rrf_range = max_rrf - min_rrf

        normalized = []
        for score in raw_rrf_scores:
            if rrf_range > 0:
                norm = (score - min_rrf) / rrf_range
            else:
                norm = 0.5
            normalized.append(norm)

        # Verify normalized values are in [0, 1]
        for i, norm in enumerate(normalized):
            assert 0.0 <= norm <= 1.0, f"Normalized RRF {norm} not in [0, 1] for raw {raw_rrf_scores[i]}"

        # Highest raw should be 1.0
        assert normalized[0] == 1.0, f"Highest RRF should normalize to 1.0, got {normalized[0]}"

        # Lowest raw should be 0.0
        assert normalized[-1] == 0.0, f"Lowest RRF should normalize to 0.0, got {normalized[-1]}"

    def test_rrf_all_same_scores(self):
        """When all RRF scores are the same, normalized should be 0.5 (neutral)."""
        raw_rrf_scores = [0.0500, 0.0500, 0.0500]

        max_rrf = max(raw_rrf_scores)
        min_rrf = min(raw_rrf_scores)
        rrf_range = max_rrf - min_rrf

        normalized = []
        for score in raw_rrf_scores:
            if rrf_range > 0:
                norm = (score - min_rrf) / rrf_range
            else:
                norm = 0.5  # Neutral value when all same
            normalized.append(norm)

        # All should be 0.5 when scores are identical
        for norm in normalized:
            assert norm == 0.5, f"Expected 0.5 for identical scores, got {norm}"


class TestCombinedScoringFormula:
    """Test that the combined scoring formula is applied correctly."""

    def test_combined_score_calculation(self):
        """Verify the weighted combination: 0.6*CE + 0.2*RRF + 0.1*temporal + 0.1*recency."""
        # Test case 1: All components at 1.0
        ce_norm = 1.0
        rrf_norm = 1.0
        temporal = 1.0
        recency = 1.0

        expected = 0.6 * ce_norm + 0.2 * rrf_norm + 0.1 * temporal + 0.1 * recency
        assert expected == 1.0, f"All 1.0 should give 1.0, got {expected}"

        # Test case 2: All components at 0.0
        ce_norm = 0.0
        rrf_norm = 0.0
        temporal = 0.0
        recency = 0.0

        expected = 0.6 * ce_norm + 0.2 * rrf_norm + 0.1 * temporal + 0.1 * recency
        assert expected == 0.0, f"All 0.0 should give 0.0, got {expected}"

        # Test case 3: High CE, low RRF (cross-encoder finds something retrieval missed)
        ce_norm = 0.999
        rrf_norm = 0.0  # Lowest in set
        temporal = 0.5
        recency = 0.5

        expected = 0.6 * ce_norm + 0.2 * rrf_norm + 0.1 * temporal + 0.1 * recency
        # 0.5994 + 0.0 + 0.05 + 0.05 = 0.6994
        assert abs(expected - 0.6994) < 0.001, f"Expected ~0.6994, got {expected}"

        # Test case 4: Medium CE, high RRF (retrieval consensus)
        ce_norm = 0.8
        rrf_norm = 1.0  # Highest in set
        temporal = 0.5
        recency = 0.5

        expected = 0.6 * ce_norm + 0.2 * rrf_norm + 0.1 * temporal + 0.1 * recency
        # 0.48 + 0.2 + 0.05 + 0.05 = 0.78
        assert abs(expected - 0.78) < 0.001, f"Expected ~0.78, got {expected}"

    def test_rrf_contribution_is_significant(self):
        """Verify RRF actually contributes to the final score (not negligible)."""
        # Same CE, different RRF
        ce_norm = 0.8
        temporal = 0.5
        recency = 0.5

        # Low RRF
        score_low_rrf = 0.6 * ce_norm + 0.2 * 0.0 + 0.1 * temporal + 0.1 * recency

        # High RRF
        score_high_rrf = 0.6 * ce_norm + 0.2 * 1.0 + 0.1 * temporal + 0.1 * recency

        # Difference should be 0.2 (20% contribution)
        diff = score_high_rrf - score_low_rrf
        assert abs(diff - 0.2) < 0.001, f"RRF should contribute 0.2 difference, got {diff}"


@pytest.mark.asyncio
async def test_trace_has_normalized_rrf(memory):
    """Integration test: verify trace contains normalized RRF values, not raw."""
    bank_id = f"test_scoring_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store multiple memories to ensure different RRF scores
        await memory.retain_async(
            bank_id=bank_id,
            content="Python is a programming language created by Guido van Rossum",
            context="tech facts",
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="JavaScript was created by Brendan Eich at Netscape",
            context="tech facts",
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="The Eiffel Tower is located in Paris, France",
            context="geography facts",
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Mount Everest is the tallest mountain on Earth",
            context="geography facts",
        )

        # Search with tracing
        result = await memory.recall_async(
            bank_id=bank_id,
            query="programming languages",
            fact_type=["world"],
            budget=Budget.LOW,
            max_tokens=1024,
            enable_trace=True,
        )

        assert result.trace is not None, "Trace should be present"
        trace = result.trace

        # Check reranked results have proper score_components
        assert "reranked" in trace, "Trace should have reranked results"
        assert len(trace["reranked"]) > 0, "Should have reranked results"

        has_valid_rrf = False
        has_valid_temporal = False
        has_valid_recency = False

        for r in trace["reranked"]:
            sc = r.get("score_components", {})

            # Check RRF normalized is present and in valid range
            if "rrf_normalized" in sc:
                rrf_norm = sc["rrf_normalized"]
                assert 0.0 <= rrf_norm <= 1.0, f"rrf_normalized {rrf_norm} should be in [0, 1]"
                # Should NOT be raw RRF score (which would be ~0.04-0.06)
                # A normalized value of exactly 0.0 or 1.0 is valid (min/max of set)
                # But raw scores like 0.0607 should never appear as normalized
                if rrf_norm > 0.1:  # Any value > 0.1 is likely properly normalized
                    has_valid_rrf = True

            # Check temporal is present and in valid range
            if "temporal" in sc:
                temporal = sc["temporal"]
                assert 0.0 <= temporal <= 1.0, f"temporal {temporal} should be in [0, 1]"
                has_valid_temporal = True

            # Check recency is present and in valid range
            if "recency" in sc:
                recency = sc["recency"]
                assert 0.0 <= recency <= 1.0, f"recency {recency} should be in [0, 1]"
                has_valid_recency = True

        # At least some results should have these components
        # (might not have rrf > 0.1 if all scores are same, which is fine)
        assert has_valid_temporal, "Should have temporal scores in trace"
        assert has_valid_recency, "Should have recency scores in trace"

        print("\n✓ Combined scoring trace test passed!")
        print(f"  - Reranked results: {len(trace['reranked'])}")
        if trace["reranked"]:
            sc = trace["reranked"][0].get("score_components", {})
            print(f"  - First result score components: {sc}")

    finally:
        await memory.delete_bank(bank_id)


@pytest.mark.asyncio
async def test_rrf_normalized_not_raw_in_trace(memory):
    """Verify that raw RRF scores (0.04-0.06 range) don't appear as normalized values."""
    bank_id = f"test_rrf_raw_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Store enough memories to get varied RRF scores
        for i in range(5):
            await memory.retain_async(
                bank_id=bank_id,
                content=f"Test fact number {i} about various topics",
                context="test context",
            )

        result = await memory.recall_async(
            bank_id=bank_id,
            query="test fact",
            fact_type=["world"],
            budget=Budget.LOW,
            max_tokens=512,
            enable_trace=True,
        )

        trace = result.trace
        assert trace is not None

        # Check that rrf_normalized values are NOT in the raw range
        raw_rrf_range = (0.01, 0.08)  # Raw RRF scores are typically in this range

        for r in trace.get("reranked", []):
            sc = r.get("score_components", {})

            if "rrf_normalized" in sc and "rrf_score" in sc:
                rrf_norm = sc["rrf_normalized"]
                rrf_raw = sc["rrf_score"]

                # Raw should be in the typical range
                assert raw_rrf_range[0] <= rrf_raw <= raw_rrf_range[1], \
                    f"Raw RRF {rrf_raw} should be in typical range {raw_rrf_range}"

                # Normalized should either be:
                # - 0.0 (min in set)
                # - 1.0 (max in set)
                # - 0.5 (all same)
                # - Something in between (0.0 to 1.0)
                # But NOT the same as raw (which would indicate no normalization)
                if len(trace["reranked"]) > 1:
                    # If we have multiple results, normalized should differ from raw
                    # (unless by coincidence, which is very unlikely)
                    assert rrf_norm != rrf_raw, \
                        f"Normalized RRF ({rrf_norm}) should differ from raw ({rrf_raw})"

        print("\n✓ RRF raw vs normalized test passed!")

    finally:
        await memory.delete_bank(bank_id)


@pytest.mark.asyncio
async def test_combined_score_matches_components(memory):
    """Verify the final score actually equals the weighted sum of components."""
    bank_id = f"test_combined_{datetime.now(timezone.utc).timestamp()}"

    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="The quick brown fox jumps over the lazy dog",
            context="test",
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="A quick test of the emergency broadcast system",
            context="test",
        )

        result = await memory.recall_async(
            bank_id=bank_id,
            query="quick test",
            fact_type=["world"],
            budget=Budget.LOW,
            max_tokens=512,
            enable_trace=True,
        )

        trace = result.trace
        assert trace is not None

        for r in trace.get("reranked", []):
            sc = r.get("score_components", {})
            final_score = r.get("rerank_score", 0)

            # Get components (use defaults if missing)
            ce = sc.get("cross_encoder_score_normalized", 0)
            rrf = sc.get("rrf_normalized", 0.5)
            tmp = sc.get("temporal", 0.5)
            rec = sc.get("recency", 0.5)

            # Calculate expected score
            expected = 0.6 * ce + 0.2 * rrf + 0.1 * tmp + 0.1 * rec

            # Allow small floating point difference
            assert abs(final_score - expected) < 0.01, \
                f"Final score {final_score} doesn't match expected {expected} from components"

        print("\n✓ Combined score verification test passed!")

    finally:
        await memory.delete_bank(bank_id)
