import pytest
from pydantic import ValidationError

from hindsight_client_api.models.semantic_candidates_response import SemanticCandidatesResponse


def test_semantic_candidates_success_response_deserializes_boolean_constants() -> None:
    response = SemanticCandidatesResponse.from_dict(
        {
            "candidates": [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "type": "world",
                    "score": 0.82,
                }
            ],
            "limit": 100,
            "returned": 1,
            "limit_reached": False,
            "exhaustive": False,
            "total_relation": "unknown",
            "min_similarity": 0.4,
            "score": {"name": "cosine_similarity", "approximate": True},
            "corpus_scope": "full_bank",
            "scope": "valid_memory_units",
        }
    )

    assert response is not None
    assert response.exhaustive is False
    assert response.score is not None
    assert response.score.approximate is True


def test_semantic_candidates_response_requires_completeness_provenance() -> None:
    with pytest.raises(ValidationError):
        SemanticCandidatesResponse.from_dict(
            {
                "candidates": [],
                "limit": 10,
                "returned": 0,
                "limit_reached": False,
                "min_similarity": 0.4,
            }
        )
