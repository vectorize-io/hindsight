"""Regression tests for the causal link taxonomy used by retain."""

import pytest
from pydantic import ValidationError

from hindsight_api.engine.causal_links import (
    CANONICAL_CAUSAL_LINK_TYPE,
    CANONICAL_CAUSAL_LINK_TYPES,
    CAUSAL_LINK_TYPES,
    LEGACY_CAUSAL_LINK_TYPES,
    CausalLinkDescriptor,
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


def test_causal_link_descriptor_round_trips_through_json() -> None:
    """The archive stores descriptors as plain JSON; the round-trip must be lossless."""
    descriptor = CausalLinkDescriptor(
        from_unit_id="11111111-1111-1111-1111-111111111111",
        to_unit_id="22222222-2222-2222-2222-222222222222",
        link_type="caused_by",
        weight=0.75,
    )
    assert CausalLinkDescriptor.from_json_dict(descriptor.as_json_dict()) == descriptor


def test_causal_link_descriptor_defaults_missing_weight() -> None:
    parsed = CausalLinkDescriptor.from_json_dict(
        {
            "from_unit_id": "11111111-1111-1111-1111-111111111111",
            "to_unit_id": "22222222-2222-2222-2222-222222222222",
            "link_type": "enables",
        }
    )
    assert parsed is not None
    assert parsed.weight == 1.0


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-dict",
        {"from_unit_id": None, "to_unit_id": "2", "link_type": "caused_by"},
        {"from_unit_id": "1", "to_unit_id": None, "link_type": "caused_by"},
        # 'temporal' is derived data, not a causal edge — and memory_links has a
        # link_type CHECK constraint, so an unusable entry must not abort a revert.
        {"from_unit_id": "1", "to_unit_id": "2", "link_type": "temporal"},
        {"from_unit_id": "1", "to_unit_id": "2"},
    ],
)
def test_causal_link_descriptor_skips_unusable_entries(raw) -> None:
    assert CausalLinkDescriptor.from_json_dict(raw) is None
