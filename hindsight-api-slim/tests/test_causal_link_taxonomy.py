"""Regression tests for the causal link taxonomy used by retain."""

import pytest
from pydantic import ValidationError

from hindsight_api.engine.causal_links import (
    CANONICAL_CAUSAL_LINK_TYPE,
    CANONICAL_CAUSAL_LINK_TYPES,
    CAUSAL_LINK_TYPES,
    LEGACY_CAUSAL_LINK_TYPES,
)
from hindsight_api.engine.retain.fact_extraction import CausalRelation, FactCausalRelation


@pytest.mark.parametrize("relation_type", ["causes", "enables", "prevents"])
def test_retain_causal_models_reject_legacy_relation_types(relation_type: str) -> None:
    """New retain output is canonical even though storage reads legacy links."""
    with pytest.raises(ValidationError):
        CausalRelation(target_fact_index=0, relation_type=relation_type)

    with pytest.raises(ValidationError):
        FactCausalRelation(target_index=0, relation_type=relation_type)


def test_retain_causal_models_accept_caused_by() -> None:
    """The canonical causal relationship remains valid in both extraction schemas."""
    assert CausalRelation(target_fact_index=0, relation_type="caused_by").relation_type == "caused_by"
    assert FactCausalRelation(target_index=0, relation_type="caused_by").relation_type == "caused_by"


def test_causal_link_taxonomy_keeps_canonical_and_legacy_types_separate() -> None:
    assert CANONICAL_CAUSAL_LINK_TYPES == {CANONICAL_CAUSAL_LINK_TYPE}
    assert LEGACY_CAUSAL_LINK_TYPES == {"causes", "enables", "prevents"}
    assert CAUSAL_LINK_TYPES == (CANONICAL_CAUSAL_LINK_TYPE, "causes", "enables", "prevents")
