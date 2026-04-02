"""
Unit tests for proof_count boost in reranking.
"""

from datetime import datetime, timezone
import pytest
from uuid import uuid4

from hindsight_api.engine.search.types import RetrievalResult, MergedCandidate, ScoredResult
from hindsight_api.engine.search.reranking import apply_combined_scoring

UTC = timezone.utc

def create_mock_scored_result(proof_count: int | None = None, ce_score: float = 0.8) -> ScoredResult:
    """Helper to create a minimal ScoredResult suitable for scoring tests."""
    retrieval = RetrievalResult(
        id=uuid4(),
        text="Test mock fact",
        fact_type="observation" if proof_count is not None else "world",
        document_id=uuid4(),
        chunk_id=uuid4(),
        embedding=[0.1]*384,
        similarity=0.9,
        proof_count=proof_count,
        # Default neutral dates for testing so only proof_count changes score
        occurred_start=datetime.now(UTC),
        occurred_end=datetime.now(UTC)
    )
    candidate = MergedCandidate(
        id=retrieval.id,
        retrieval=retrieval,
        semantic_rank=1,
        bm25_rank=1,
        rrf_score=0.1
    )
    return ScoredResult(
        candidate=candidate,
        cross_encoder_score=ce_score,
        cross_encoder_score_normalized=ce_score,
        weight=ce_score,
    )

def test_proof_count_neutral_when_none():
    """Test that when proof_count is None (e.g. non-observation), it gets neutral 0.5 norm."""
    sr = create_mock_scored_result(proof_count=None, ce_score=0.8)
    now = datetime.now(UTC)
    
    apply_combined_scoring([sr], now, proof_count_alpha=0.1)
    
    # Neutral multiplier means score shouldn't be boosted by proof_count
    # Since recency is neutral (just created) and temporal is neutral, score should remain unchanged
    assert sr.combined_score == pytest.approx(0.8, rel=1e-3)

def test_proof_count_neutral_at_one():
    """Test that proof_count=1 gives neutral multiplier."""
    sr = create_mock_scored_result(proof_count=1, ce_score=0.8)
    now = datetime.now(UTC)
    
    apply_combined_scoring([sr], now, proof_count_alpha=0.1)
    
    # proof_count=1 -> math.log(1) = 0 -> 0.5 + 0/10 = 0.5 (neutral) -> multiplier 1.0
    assert sr.combined_score == pytest.approx(0.8, rel=1e-3)

def test_proof_count_increases_with_higher_counts():
    """Test that higher proof counts yield strictly higher scores."""
    now = datetime.now(UTC)
    
    # Create results with increasing proof counts
    sr_5 = create_mock_scored_result(proof_count=5, ce_score=0.8)
    sr_50 = create_mock_scored_result(proof_count=50, ce_score=0.8)
    sr_100 = create_mock_scored_result(proof_count=100, ce_score=0.8)
    
    # Process them
    apply_combined_scoring([sr_5, sr_50, sr_100], now, proof_count_alpha=0.1)
    
    # Assure scores strictly increase
    assert sr_5.combined_score > 0.8
    assert sr_50.combined_score > sr_5.combined_score
    assert sr_100.combined_score > sr_50.combined_score

def test_proof_count_no_hardcoded_cap_at_100():
    """Test that proof_count continues to scale within the clamped [0, 1] range."""
    now = datetime.now(UTC)
    
    # Use values that stay below the clamp ceiling (proof_norm < 1.0)
    # log(5)/10=0.16, log(20)/10=0.30, log(100)/10=0.46 → all below 0.5 headroom
    sr_5 = create_mock_scored_result(proof_count=5, ce_score=0.8)
    sr_20 = create_mock_scored_result(proof_count=20, ce_score=0.8)
    sr_100 = create_mock_scored_result(proof_count=100, ce_score=0.8)
    
    apply_combined_scoring([sr_5, sr_20, sr_100], now, proof_count_alpha=0.1)
    
    # Must strictly increase within the valid range
    assert sr_20.combined_score > sr_5.combined_score
    assert sr_100.combined_score > sr_20.combined_score

