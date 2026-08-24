"""Existing entities must not be merged onto by history alone (#3751).

The trigram probe admits candidates at a deliberately loose recall threshold (0.15) and the
composite score's non-name signals total 0.5 of the 0.6 needed, so before the floor a name
the probe merely *considered* similar could be reused purely because the bank had seen it
recently alongside the same entities — putting a new person's facts on an unrelated entity.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.entity_resolver import (
    EntityResolver,
    _build_cooccurrence_index,
    _cooccurrence_weight,
    _tokens_are_compatible,
)

NOW = datetime.now(UTC)


def _resolver(new_ids: dict[str, str], **kwargs) -> EntityResolver:
    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value=new_ids),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    return EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="trigram", **kwargs)


async def _resolve_one(
    resolver: EntityResolver,
    name: str,
    candidate: tuple,
    *,
    nearby: list[str],
    cooccurs_with: set[str],
    degrees: dict[str, int] | None = None,
    event_date: datetime | None = NOW,
):
    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[
            {
                "text": name,
                "nearby_entities": [{"text": n} for n in [name, *nearby]],
                "event_date": event_date,
            }
        ],
        unit_event_date=event_date,
        all_candidates={name: [candidate]},
        cooccurrence_map={candidate[0]: cooccurs_with},
        cooccurrence_degrees=degrees,
    )
    return resolved[0].canonical_name


@pytest.mark.asyncio
async def test_coincidental_short_name_is_not_absorbed_by_a_recent_well_connected_entity():
    """The reported shape: a new person named Tigran, a country named Iran already in the bank.

    similarity('tigran','iran') is 0.20 — above the 0.15 that admits it as a candidate, below
    anything that should merge it. Every other signal is maxed out: Iran co-occurs with both
    of the other entities in the fact and was mentioned today. It must still not win.
    """
    resolver = _resolver({"tigran": "new-tigran-id"})
    name = await _resolve_one(
        resolver,
        "Tigran",
        ("iran-id", "Iran", {}, NOW, 400),
        nearby=["user", "topic:finance"],
        cooccurs_with={"user", "topic:finance"},
    )
    assert name == "Tigran", "a 0.20-similar name must not be merged onto by co-occurrence and recency"


@pytest.mark.asyncio
async def test_surface_variant_of_the_same_name_still_merges():
    """The floor is a gate on coincidence, not a general tightening: the merges the resolver
    exists for run well above it ("alice"/"alice chen" is 0.55 by trigram)."""
    resolver = _resolver({"alice": "unused"})
    name = await _resolve_one(
        resolver,
        "Alice",
        ("alice-chen-id", "Alice Chen", {}, NOW, 12),
        nearby=["Google"],
        cooccurs_with={"google"},
    )
    assert name == "Alice Chen"


@pytest.mark.asyncio
async def test_typo_variant_with_no_cooccurrence_context_still_merges():
    """ "Dr Waler" -> "Dr Wall" has nothing but the name and same-day recency going for it
    (#3479). It clears the floor at 0.55, so the sequence-ratio score still carries it."""
    resolver = _resolver({"dr waler": "unused"})
    name = await _resolve_one(
        resolver,
        "Dr Waler",
        ("dr-wall-id", "Dr Wall", {}, NOW, 40),
        nearby=[],
        cooccurs_with=set(),
    )
    assert name == "Dr Wall"


@pytest.mark.asyncio
async def test_floor_is_configurable_for_short_name_corpora():
    """ "Jon"/"John" is a real variant that trigram scores at 0.29, under the default floor.
    Deployments whose names are mostly that short can lower it."""
    candidate = ("john-id", "John", {}, NOW, 5)
    strict = await _resolve_one(_resolver({"jon": "new-jon-id"}), "Jon", candidate, nearby=[], cooccurs_with=set())
    assert strict == "Jon"

    lenient = await _resolve_one(
        _resolver({"jon": "new-jon-id"}, merge_min_similarity=0.25),
        "Jon",
        candidate,
        nearby=[],
        cooccurs_with=set(),
    )
    assert lenient == "John"


@pytest.mark.asyncio
async def test_a_hub_partner_does_not_carry_a_weak_name_over_the_threshold():
    """Iran/Iraq clears the floor (0.43) but is two different countries. The only thing they
    share is `user`, which on a real bank co-occurs with everything — so it must not be worth
    the same as a selective partner."""
    candidate = ("iran-id", "Iraq", {}, NOW - timedelta(days=30), 200)
    hub_only = await _resolve_one(
        _resolver({"iran": "new-iran-id"}),
        "Iran",
        candidate,
        nearby=["user"],
        cooccurs_with={"user"},
        degrees={"user": 1400},
    )
    assert hub_only == "Iran", "an indiscriminate partner must not be enough on its own"

    selective = await _resolve_one(
        _resolver({"iran": "new-iran-id"}),
        "Iran",
        candidate,
        nearby=["Basra"],
        cooccurs_with={"basra"},
        degrees={"basra": 1},
    )
    assert selective == "Iraq", "a selective partner is real evidence and still merges"


def test_cooccurrence_weight_decays_with_how_indiscriminate_a_partner_is():
    assert _cooccurrence_weight(1) == 1.0
    assert _cooccurrence_weight(0) == 1.0, "degree is never below one shared edge"
    assert _cooccurrence_weight(100) == pytest.approx(0.1)
    assert _cooccurrence_weight(1400) < 0.03


def test_cooccurrence_index_counts_degree_over_all_partners_not_just_named_ones():
    """Degree measures how indiscriminate an entity is, so it counts partners the batch
    never looked up — otherwise a hub looks selective simply because the fact was short."""
    rows = [
        {"entity_id_1": "user-id", "entity_id_2": "iran-id"},
        {"entity_id_1": "user-id", "entity_id_2": "unnamed-1"},
        {"entity_id_1": "unnamed-2", "entity_id_2": "user-id"},
    ]
    index = _build_cooccurrence_index(rows, {"user-id": "user", "iran-id": "iran"})

    assert index.by_entity["iran-id"] == {"user"}
    assert index.degree_by_name["user"] == 3
    assert index.degree_by_name["iran"] == 1


@pytest.mark.asyncio
async def test_scores_landing_exactly_on_the_threshold_agree_with_each_other():
    """0.6 used to mean "merge" or "don't" depending only on which signals produced it, because
    0.4 + 0.2 is 0.6000000000000001 in binary floating point while 0.3 + 0.3 is 0.6."""
    # 0.30 name (sequence ratio 0.60) + 0.30 co-occurrence, both names well above the floor.
    resolver = _resolver({"jonathan reed": "new-id"})
    name = await _resolve_one(
        resolver,
        "Jonathan Reed",
        ("other-id", "Jonathan Reeding Rd", {}, None, 3),
        nearby=["Acme"],
        cooccurs_with={"acme"},
        degrees={"acme": 1},
        event_date=None,
    )
    assert name == "Jonathan Reeding Rd"


@pytest.mark.asyncio
async def test_a_different_given_name_is_not_absorbed_by_a_shared_surname():
    """Whole-name similarity lets one long shared word drown out a different short one: "John
    Smith"/"Jane Smith" is 0.47 by trigram and 0.80 by sequence ratio, so two people who share a
    surname and a workplace scored as one entity even with the floor in place."""
    resolver = _resolver({"john smith": "new-john-id"})
    name = await _resolve_one(
        resolver,
        "John Smith",
        ("jane-id", "Jane Smith", {}, NOW, 20),
        nearby=["Bletchley Park"],
        cooccurs_with={"bletchley park"},
        degrees={"bletchley park": 1},
    )
    assert name == "John Smith"


@pytest.mark.asyncio
async def test_a_decorated_form_of_a_stored_name_is_the_same_entity():
    """Identical trigram sets mean the same words, differing only in separators. The in-batch pass
    already unifies these on the name alone at a lower bar, so requiring history here made two forms
    one entity or two depending only on whether they arrived in the same retain (#3107)."""
    resolver = _resolver({"wren 🎵": "new-wren-id"})
    name = await _resolve_one(
        resolver,
        "Wren 🎵",
        # Stale, and sharing nothing: the name has to carry it alone.
        ("wren-id", "Wren", {}, NOW - timedelta(days=400), 3),
        nearby=[],
        cooccurs_with=set(),
    )
    assert name == "Wren"


def test_word_level_agreement_admits_real_variants_and_rejects_substitutions():
    """The cutoff is calibrated between john/jane (0.50) and the legitimate word differences."""
    assert _tokens_are_compatible("são paulo", "sao paulo")
    assert _tokens_are_compatible("ann arbor", "ann arbour")
    assert _tokens_are_compatible("dr waler", "dr wall")
    assert _tokens_are_compatible("microsoft corp", "microsoft corporation"), "abbreviation, via prefix"
    assert _tokens_are_compatible("john a smith", "john smith"), "an added initial is not a word conflict"
    assert _tokens_are_compatible("alice", "alice chen"), "an added word is not a word conflict"
    assert not _tokens_are_compatible("john smith", "jane smith")
    assert not _tokens_are_compatible("new york", "new jersey")
    assert not _tokens_are_compatible("united states", "united kingdom")


def test_single_word_names_are_exempt_from_word_level_agreement():
    """With one token the whole-name floor *is* the token floor, and applying this on top would
    reject real variants with no long shared word to hide behind."""
    assert _tokens_are_compatible("nick", "nicolas"), "0.55 by sequence ratio — under the word cutoff"
    assert _tokens_are_compatible("iran", "iraq"), "not compatible in truth, but not this rule's job"
