"""Candidate entity-name hygiene at resolution intake (issue #3275).

``_prepare_entities_for_resolution`` is the single choke point both entity
resolution entry paths funnel through (retain via
``entity_processing.resolve_entities``, and the memory-edit path in
``MemoryEngine``), so it is where a name is made safe to store:

1. whitespace runs — including the ``\\n`` extraction sometimes leaves behind —
   collapse to a single space, and the ends are stripped;
2. a name that is empty afterwards is dropped rather than stored as an entity
   with a blank ``canonical_name``;
3. a name longer than ``_MAX_ENTITY_NAME_CHARS`` is dropped rather than failing
   the whole retain on the btree index over ``canonical_name``;
4. candidates that normalization made identical are deduplicated per fact.

All of it runs before the flat list / ``entity_to_unit`` mapping is derived, so
the resolver's positional invariant is untouched.
"""

import random

import asyncpg
import pytest

from hindsight_api.engine.retain.link_utils import (
    _MAX_ENTITY_NAME_CHARS,
    _normalize_entity_name,
    _prepare_entities_for_resolution,
)


class _FakeEntity:
    """Object-style candidate: exposes ``.text``, like the extraction models."""

    def __init__(self, text: str):
        self.text = text


def _texts(all_entities_flat: list[dict]) -> list[str]:
    return [e["text"] for e in all_entities_flat]


def _prepare(entities: list, unit_ids: list[str] | None = None):
    """Run one fact's candidate list through intake."""
    return _prepare_entities_for_resolution(
        unit_ids=unit_ids or ["u1"],
        sentences=["fact text"],
        fact_dates=[None],
        llm_entities=[entities],
    )


# --- _normalize_entity_name ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Acme\nCorp", "Acme Corp"),
        ("a\r\n b\tc", "a b c"),
        ("  leading and trailing  ", "leading and trailing"),
        ("multiple   spaces    inside", "multiple spaces inside"),
        ("Normal Name", "Normal Name"),
        ("", ""),
        ("   \n\t  ", ""),
    ],
)
def test_normalize_entity_name(raw, expected):
    assert _normalize_entity_name(raw) == expected


def test_normalize_entity_name_preserves_case():
    # The registry matches on LOWER(canonical_name); lowercasing here would only
    # destroy the display form.
    assert _normalize_entity_name("MiXeD\nCaSe") == "MiXeD CaSe"


def test_normalize_entity_name_leaves_ordinary_punctuation_alone():
    assert _normalize_entity_name("Dr. Foo-Bar (ACME), Inc.") == "Dr. Foo-Bar (ACME), Inc."


# --- intake: normalization reaches the text handed to the resolver ---


def test_intake_normalizes_dict_style_candidates():
    all_entities_flat, _all, _map = _prepare([{"text": "Acme\nCorp", "type": "ORG"}])
    assert _texts(all_entities_flat) == ["Acme Corp"]
    assert all_entities_flat[0]["type"] == "ORG"


def test_intake_normalizes_object_style_candidates():
    all_entities_flat, _all, _map = _prepare([_FakeEntity("a\r\n b\tc")])
    assert _texts(all_entities_flat) == ["a b c"]


def test_intake_normalizes_nearby_entities_too():
    # nearby_entities is the co-occurrence signal the resolver scores against;
    # it must carry the normalized names, not the raw ones.
    all_entities_flat, all_entities, _map = _prepare(
        [{"text": "Acme\nCorp", "type": "CONCEPT"}, {"text": "Alice", "type": "CONCEPT"}]
    )
    assert [e["text"] for e in all_entities[0]] == ["Acme Corp", "Alice"]
    assert [e["text"] for e in all_entities_flat[1]["nearby_entities"]] == ["Acme Corp", "Alice"]


# --- intake: empty names are dropped, not stored blank ---


@pytest.mark.parametrize("raw", ["", "   ", "\n", " \t\r\n "])
def test_intake_drops_empty_and_whitespace_only_candidates(raw):
    all_entities_flat, all_entities, entity_to_unit = _prepare([{"text": raw, "type": "CONCEPT"}])
    assert all_entities_flat == []
    assert all_entities == [[]]
    assert entity_to_unit == []


def test_intake_drops_candidate_dict_without_text_key():
    all_entities_flat, _all, _map = _prepare([{"type": "CONCEPT"}, {"text": "Alice", "type": "CONCEPT"}])
    assert _texts(all_entities_flat) == ["Alice"]


def test_intake_keeps_real_entities_alongside_dropped_empties():
    all_entities_flat, _all, entity_to_unit = _prepare(
        [{"text": "  ", "type": "CONCEPT"}, {"text": " Alice ", "type": "CONCEPT"}]
    )
    assert _texts(all_entities_flat) == ["Alice"]
    # entity_to_unit stays index-aligned with the flat list the resolver receives.
    assert entity_to_unit == [("u1", 0, None)]


# --- intake: oversized names are dropped (btree index row limit) ---


def test_intake_keeps_a_name_exactly_at_the_cap():
    name = "a" * _MAX_ENTITY_NAME_CHARS
    all_entities_flat, _all, _map = _prepare([{"text": name, "type": "CONCEPT"}])
    assert _texts(all_entities_flat) == [name]


def test_intake_drops_a_name_one_past_the_cap():
    all_entities_flat, all_entities, entity_to_unit = _prepare(
        [{"text": "a" * (_MAX_ENTITY_NAME_CHARS + 1), "type": "CONCEPT"}]
    )
    assert all_entities_flat == []
    assert all_entities == [[]]
    assert entity_to_unit == []


def test_intake_measures_the_cap_after_whitespace_normalization():
    # Collapsing whitespace can bring an over-long raw string under the cap; it is
    # the stored (normalized) form that has to fit the index, so that is what counts.
    raw = "a" * (_MAX_ENTITY_NAME_CHARS - 2) + "\n\n\n\n" + "b"
    all_entities_flat, _all, _map = _prepare([{"text": raw, "type": "CONCEPT"}])
    assert _texts(all_entities_flat) == ["a" * (_MAX_ENTITY_NAME_CHARS - 2) + " b"]


def test_intake_keeps_real_entities_alongside_a_dropped_artifact():
    # Shape seen on prod: an SVG path handed back as an "entity" next to real ones.
    svg = "M156.4,82.8" + "L163.2,82.8" * 200
    all_entities_flat, all_entities, entity_to_unit = _prepare(
        [{"text": "Alice", "type": "PERSON"}, {"text": svg, "type": "CONCEPT"}, {"text": "Acme", "type": "ORG"}]
    )
    assert _texts(all_entities_flat) == ["Alice", "Acme"]
    # The dropped artifact must not leak into the co-occurrence signal either.
    assert [e["text"] for e in all_entities_flat[0]["nearby_entities"]] == ["Alice", "Acme"]
    assert entity_to_unit == [("u1", 0, None), ("u1", 1, None)]


def _incompressible(n_chars: int) -> str:
    """Random 4-byte UTF-8 characters: the worst case for an index tuple.

    PostgreSQL compresses index tuples, so a repetitive string would slip under the
    btree limit and prove nothing about the cap.
    """
    rng = random.Random(3032)
    return "".join(chr(rng.randint(0x1F300, 0x1FAFF)) for _ in range(n_chars))


@pytest.mark.asyncio
async def test_cap_fits_the_real_btree_limit_and_the_old_behaviour_did_not(pg0_db_url):
    """The cap is only meaningful if it clears the index at its worst case.

    Same index shapes as idx_entities_bank_name and idx_entities_bank_lower_name,
    on a temp table so nothing persists.
    """
    conn = await asyncpg.connect(pg0_db_url)
    try:
        await conn.execute("CREATE TEMP TABLE ent_cap (bank_id TEXT NOT NULL, canonical_name TEXT NOT NULL)")
        await conn.execute("CREATE INDEX ent_cap_bank_name ON ent_cap (bank_id, canonical_name)")
        # Both btrees the real schema carries over the name: the plain one prod
        # reported failing on, and the unique LOWER one the entity INSERT uses as
        # its ON CONFLICT target. Same ~2704-byte tuple ceiling on either, so the
        # cap has to clear both.
        await conn.execute("CREATE UNIQUE INDEX ent_cap_bank_lower_name ON ent_cap (bank_id, LOWER(canonical_name))")
        long_bank = "slack-" + "X" * 120

        # At the cap, 4 bytes per character, incompressible, long bank id: fits.
        await conn.execute("INSERT INTO ent_cap VALUES ($1, $2)", long_bank, _incompressible(_MAX_ENTITY_NAME_CHARS))

        # Well past it — what extraction actually produced on prod — does not.
        with pytest.raises(asyncpg.exceptions.ProgramLimitExceededError):
            await conn.execute("INSERT INTO ent_cap VALUES ($1, $2)", long_bank, _incompressible(1000))
    finally:
        await conn.close()


# --- intake: dedup of candidates that normalization made identical ---


def test_intake_dedupes_candidates_normalization_made_identical():
    # The upstream dedup in entity_processing runs on raw text, so these two
    # arrive here distinct and collide only after normalization.
    all_entities_flat, all_entities, entity_to_unit = _prepare(
        [{"text": "Acme\nCorp", "type": "CONCEPT"}, {"text": "Acme Corp", "type": "CONCEPT"}]
    )
    assert _texts(all_entities_flat) == ["Acme Corp"]
    assert [e["text"] for e in all_entities[0]] == ["Acme Corp"]
    assert len(entity_to_unit) == 1


def test_intake_dedupe_is_case_insensitive_and_keeps_first_spelling():
    all_entities_flat, _all, _map = _prepare(
        [{"text": "Acme Corp", "type": "CONCEPT"}, {"text": "acme\ncorp", "type": "CONCEPT"}]
    )
    assert _texts(all_entities_flat) == ["Acme Corp"]


def test_intake_dedupe_is_scoped_per_fact():
    # The same entity mentioned by two different facts must still be resolved
    # for each of them — the dedup is within a fact, not across the batch.
    all_entities_flat, _all, entity_to_unit = _prepare_entities_for_resolution(
        unit_ids=["u1", "u2"],
        sentences=["first", "second"],
        fact_dates=[None, None],
        llm_entities=[
            [{"text": "Acme\nCorp", "type": "CONCEPT"}],
            [{"text": "Acme Corp", "type": "CONCEPT"}],
        ],
    )
    assert _texts(all_entities_flat) == ["Acme Corp", "Acme Corp"]
    assert [unit_id for unit_id, _idx, _date in entity_to_unit] == ["u1", "u2"]


def test_intake_attaches_fact_dates_after_dropping():
    from datetime import UTC, datetime

    when = datetime(2026, 8, 8, tzinfo=UTC)
    all_entities_flat, _all, _map = _prepare_entities_for_resolution(
        unit_ids=["u1"],
        sentences=["fact text"],
        fact_dates=[when],
        llm_entities=[[{"text": " ", "type": "CONCEPT"}, {"text": "Alice", "type": "CONCEPT"}]],
    )
    # The event_date attach loop walks entity_to_unit positionally; a dropped
    # candidate must not shift it.
    assert _texts(all_entities_flat) == ["Alice"]
    assert all_entities_flat[0]["event_date"] == when


def test_intake_keeps_the_stricter_resolve_flag_when_normalization_collapses_names():
    """A caller's literal name must not become resolvable via a normalization collision (#3479).

    entity_processing dedups caller-supplied against extracted names on the RAW text, so
    "Acme Corp" and "Acme\nCorp" both arrive here and only collapse after normalization.
    Keeping the first entry verbatim would drop the caller's resolve=False with it.
    """
    all_entities_flat, _all, _map = _prepare(
        [
            {"text": "Acme\nCorp", "type": "ORG"},  # extracted: resolvable
            {"text": "Acme Corp", "type": "ORG", "resolve": False},  # caller: literal
        ]
    )

    assert _texts(all_entities_flat) == ["Acme Corp"], "still one mention after normalization"
    assert all_entities_flat[0]["resolve"] is False, "the caller's literal intent survives the merge"


def test_intake_defaults_resolve_to_true():
    all_entities_flat, _all, _map = _prepare([{"text": "Alice", "type": "PERSON"}, _FakeEntity("Bob")])
    assert [e["resolve"] for e in all_entities_flat] == [True, True]
