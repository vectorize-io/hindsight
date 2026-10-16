"""Ids a small model miscopied are matched back to the observation it was shown."""

from hindsight_api.engine.consolidation.consolidator import _resolve_pool_id

POOL = {"d0a3effa-1795-4c48-9a1b-0123456789ab", "77aa11bb-2222-4c48-9a1b-0123456789ab"}


def test_an_exact_id_is_kept() -> None:
    assert _resolve_pool_id("d0a3effa-1795-4c48-9a1b-0123456789ab", POOL) == "d0a3effa-1795-4c48-9a1b-0123456789ab"


def test_a_miscopied_tail_resolves_to_the_one_id_with_that_head() -> None:
    assert _resolve_pool_id("d0a3effa-1795-4ef0-0000-000000000000", POOL) == "d0a3effa-1795-4c48-9a1b-0123456789ab"


def test_an_unknown_or_ambiguous_head_is_returned_unchanged() -> None:
    assert _resolve_pool_id("ffffffff-1795-4ef0-0000-000000000000", POOL) == "ffffffff-1795-4ef0-0000-000000000000"
    twins = {"d0a3effa-1795-aaaa-0000-000000000000", "d0a3effa-1795-bbbb-0000-000000000000"}
    assert _resolve_pool_id("d0a3effa-1795-cccc-0000-000000000000", twins) == "d0a3effa-1795-cccc-0000-000000000000"
    assert _resolve_pool_id("the new observation", POOL) == "the new observation"
