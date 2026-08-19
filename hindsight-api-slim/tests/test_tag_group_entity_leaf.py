"""Tests for the entity leaf in the tag-group grammar.

Why this exists: on a bank whose tag vocabulary is a handful of broad topics, a
tag scope cannot isolate a subject (measured 2026-08-08: best tag scope 25%
precision vs 15.5% base rate, entity association 94.2%/86.2%). The entity leaf
makes that association expressible everywhere a TagGroup already flows: the
mental-model refresh scope, the staleness gate, and retrieval filtering.

Layers covered, cheapest first:
  1. schema: parse/validate, incl. the not-placement bar
  2. SQL: the clause the builder emits, param accounting, `all` arity
  3. strip: the permissive reading for surfaces without entity postings
  4. python matcher: post-retrieval refinement + the permissive fallback
  5. database: `any_memory_updated_since` (the staleness gate) with direct and
     source-inherited entity association -- the load-bearing observation case
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter

from hindsight_api.engine.memories.pg.reads import any_memory_updated_since
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.engine.search.tags import (
    TagGroup,
    TagGroupEntityLeaf,
    build_tag_groups_where_clause,
    filter_results_by_tag_groups,
    strip_entity_leaves,
    validate_entity_leaf_placement,
)

_ADAPTER = TypeAdapter(TagGroup)


# ---------------------------------------------------------------- 1. schema


def test_entity_leaf_parses_from_dict():
    leaf = _ADAPTER.validate_python({"entities": ["atlas", "Northstar"]})
    assert isinstance(leaf, TagGroupEntityLeaf)
    assert leaf.match == "any"


def test_entity_leaf_parses_nested_in_compounds():
    group = _ADAPTER.validate_python(
        {
            "or": [
                {"entities": ["atlas"]},
                {"and": [{"tags": ["topic:infra"]}, {"entities": ["Northstar"], "match": "all"}]},
            ]
        }
    )
    validate_entity_leaf_placement([group])  # no NOT anywhere -> fine


def test_empty_entities_rejected():
    with pytest.raises(Exception):
        _ADAPTER.validate_python({"entities": []})


def test_entity_leaf_under_not_rejected():
    group = _ADAPTER.validate_python({"not": {"entities": ["atlas"]}})
    with pytest.raises(ValueError, match="entity filter may not appear under 'not'"):
        validate_entity_leaf_placement([group])


def test_entity_leaf_under_nested_not_rejected():
    group = _ADAPTER.validate_python({"not": {"or": [{"tags": ["a"]}, {"entities": ["atlas"]}]}})
    with pytest.raises(ValueError):
        validate_entity_leaf_placement([group])


def test_tag_only_groups_pass_placement_validation():
    group = _ADAPTER.validate_python({"not": {"tags": ["topic:infra"]}})
    validate_entity_leaf_placement([group])


# ------------------------------------------------------------------- 2. SQL


def test_sql_any_emits_correlated_exists_with_lowered_names():
    leaf = _ADAPTER.validate_python({"entities": ["Atlas", "NORTHSTAR", "  atlas  "]})
    clause, params, next_offset = build_tag_groups_where_clause([leaf], param_offset=3)
    assert "EXISTS" in clause
    assert "LOWER(e.canonical_name) = ANY($3)" in clause
    # correlation against the outer memory_units row, direct OR through sources
    assert ".id" in clause and "source_memory_ids" in clause
    assert params == [["atlas", "northstar"]]  # lowered, trimmed, deduped, sorted
    assert next_offset == 4


def test_sql_all_emits_distinct_count_arity():
    leaf = _ADAPTER.validate_python({"entities": ["atlas", "Northstar"], "match": "all"})
    clause, params, _ = build_tag_groups_where_clause([leaf], param_offset=1)
    assert "COUNT(DISTINCT LOWER(e.canonical_name))" in clause
    assert clause.rstrip(")").endswith("= 2")
    assert params == [["atlas", "northstar"]]


def test_sql_mixed_tags_and_entities_param_accounting():
    group = _ADAPTER.validate_python(
        {"and": [{"tags": ["topic:infra"], "match": "any_strict"}, {"entities": ["atlas"]}]}
    )
    clause, params, next_offset = build_tag_groups_where_clause([group], param_offset=5)
    assert "$5" in clause and "$6" in clause
    assert params == [["topic:infra"], ["atlas"]]
    assert next_offset == 7


# ------------------------------------------------------------------ 3. strip


def _strip_dicts(*groups: dict) -> list | None:
    return strip_entity_leaves([_ADAPTER.validate_python(g) for g in groups])


def test_strip_lone_entity_leaf_yields_none():
    assert _strip_dicts({"entities": ["atlas"]}) is None


def test_strip_keeps_tag_siblings_in_and():
    kept = _strip_dicts({"and": [{"tags": ["topic:infra"]}, {"entities": ["atlas"]}]})
    assert kept is not None and len(kept) == 1
    clause, params, _ = build_tag_groups_where_clause(kept, param_offset=1)
    assert "EXISTS" not in clause and params == [["topic:infra"]]


def test_strip_collapses_or_containing_entity_leaf():
    # True OR x is True: the whole OR becomes permissive and disappears.
    assert _strip_dicts({"or": [{"tags": ["topic:infra"]}, {"entities": ["atlas"]}]}) is None


def test_strip_leaves_tag_only_groups_untouched():
    kept = _strip_dicts({"tags": ["topic:infra"], "match": "all_strict"})
    assert kept is not None and len(kept) == 1


# --------------------------------------------------------- 4. python matcher


class _Result:
    def __init__(self, tags=None, entities=None):
        self.tags = tags
        self.entities = entities


def test_matcher_filters_by_entity_names_when_present():
    groups = [_ADAPTER.validate_python({"entities": ["atlas"]})]
    hit = _Result(entities=["atlas", "Northstar"])
    miss = _Result(entities=[{"canonical_name": "Docker"}])
    assert filter_results_by_tag_groups([hit, miss], groups) == [hit]


def test_matcher_reads_dict_shaped_entities():
    groups = [_ADAPTER.validate_python({"entities": ["northstar"]})]
    hit = _Result(entities=[{"entity_id": "x", "canonical_name": "Northstar"}])
    assert filter_results_by_tag_groups([hit], groups) == [hit]


def test_matcher_passes_results_without_entity_annotations():
    groups = [_ADAPTER.validate_python({"entities": ["atlas"]})]
    unannotated = _Result(entities=None)
    assert filter_results_by_tag_groups([unannotated], groups) == [unannotated]


def test_matcher_all_requires_every_name():
    groups = [_ADAPTER.validate_python({"entities": ["atlas", "northstar"], "match": "all"})]
    both = _Result(entities=["atlas", "Northstar", "extra"])
    one = _Result(entities=["atlas"])
    assert filter_results_by_tag_groups([both, one], groups) == [both]


# -------------------------------------------------------------- 5. database


async def _mk_unit(conn, bank_id: str, text: str, fact_type: str = "world", sources: list[str] | None = None) -> str:
    uid = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, source_memory_ids, created_at, updated_at)
        VALUES ($1::uuid, $2, $3, $4, $5::uuid[], now(), now())
        """,
        uid,
        bank_id,
        text,
        fact_type,
        sources,
    )
    return uid


async def _mk_entity(conn, bank_id: str, name: str) -> str:
    eid = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO entities (id, bank_id, canonical_name, entity_kind, first_seen, last_seen, mention_count)
        VALUES ($1::uuid, $2, $3, 'regular', now(), now(), 1)
        """,
        eid,
        bank_id,
        name,
    )
    return eid


async def _link(conn, unit_id: str, entity_id: str) -> None:
    await conn.execute(
        "INSERT INTO unit_entities (unit_id, entity_id) VALUES ($1::uuid, $2::uuid) ON CONFLICT DO NOTHING",
        unit_id,
        entity_id,
    )


def _groups(*dicts: dict) -> list:
    return [_ADAPTER.validate_python(d) for d in dicts]


@pytest.mark.asyncio
async def test_staleness_sees_direct_entity_match(memory: MemoryEngine):
    """any_memory_updated_since is the staleness gate; an entity-scoped model
    must go stale exactly when a fact ABOUT its entities arrives."""
    bank_id = f"test-ent-stale-{uuid.uuid4().hex[:8]}"
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with memory._pool.acquire() as conn:
        e_atlas = await _mk_entity(conn, bank_id, "atlas")
        unit = await _mk_unit(conn, bank_id, "the cli opened an unexpected window.")
        await _link(conn, unit, e_atlas)

        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                tag_groups=_groups({"entities": ["Atlas"]}),  # case-insensitive
            )
            is True
        )
        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                tag_groups=_groups({"entities": ["docker"]}),
            )
            is False
        )


@pytest.mark.asyncio
async def test_staleness_sees_source_inherited_match_for_observations(memory: MemoryEngine):
    """The load-bearing case: observations carry NO direct postings by design;
    their entity association is transitive through source_memory_ids. An
    entity-scoped model reading observations must still go stale when an
    observation built from entity-linked sources arrives."""
    bank_id = f"test-ent-stale-{uuid.uuid4().hex[:8]}"
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with memory._pool.acquire() as conn:
        e_gem = await _mk_entity(conn, bank_id, "Northstar")
        src = await _mk_unit(conn, bank_id, "Northstar shim fact.")
        await _link(conn, src, e_gem)
        # the observation itself gets NO direct unit_entities row
        await _mk_unit(conn, bank_id, "Synthesised: the Northstar lane.", "observation", sources=[src])

        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                fact_types=["observation"],
                tag_groups=_groups({"entities": ["northstar"]}),
            )
            is True
        )
        # entity exists in the bank, but no observation reaches it
        e_other = await _mk_entity(conn, bank_id, "Docker")
        _ = e_other
        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                fact_types=["observation"],
                tag_groups=_groups({"entities": ["docker"]}),
            )
            is False
        )


@pytest.mark.asyncio
async def test_staleness_entity_and_tag_conjunction(memory: MemoryEngine):
    """Mixed group: entity leaf AND tag leaf must both hold on the same row."""
    bank_id = f"test-ent-stale-{uuid.uuid4().hex[:8]}"
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with memory._pool.acquire() as conn:
        e_atlas = await _mk_entity(conn, bank_id, "atlas")
        tagged = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at, updated_at)
            VALUES ($1::uuid, $2, 'atlas under infra tag', 'world', ARRAY['topic:infra'], now(), now())
            """,
            tagged,
            bank_id,
        )
        await _link(conn, tagged, e_atlas)

        both = _groups({"and": [{"tags": ["topic:infra"], "match": "any_strict"}, {"entities": ["atlas"]}]})
        wrong_tag = _groups({"and": [{"tags": ["topic:vendors"], "match": "any_strict"}, {"entities": ["atlas"]}]})
        assert (
            await any_memory_updated_since(conn=conn, fq_table=fq_table, bank_id=bank_id, since=since, tag_groups=both)
            is True
        )
        assert (
            await any_memory_updated_since(
                conn=conn, fq_table=fq_table, bank_id=bank_id, since=since, tag_groups=wrong_tag
            )
            is False
        )


@pytest.mark.asyncio
async def test_staleness_all_match_requires_every_entity(memory: MemoryEngine):
    bank_id = f"test-ent-stale-{uuid.uuid4().hex[:8]}"
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with memory._pool.acquire() as conn:
        e_atlas = await _mk_entity(conn, bank_id, "atlas")
        e_gem = await _mk_entity(conn, bank_id, "Northstar")
        unit = await _mk_unit(conn, bank_id, "atlas warms Northstar.")
        await _link(conn, unit, e_atlas)
        await _link(conn, unit, e_gem)
        only_atlas = await _mk_unit(conn, bank_id, "atlas alone.")
        await _link(conn, only_atlas, e_atlas)

        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                tag_groups=_groups({"entities": ["atlas", "northstar"], "match": "all"}),
            )
            is True
        )
        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=since,
                tag_groups=_groups({"entities": ["atlas", "docker"], "match": "all"}),
            )
            is False
        )


@pytest.mark.asyncio
async def test_cross_bank_entity_names_do_not_leak(memory: MemoryEngine):
    """A same-named entity in another bank has a different id and no link to this
    bank's units -- the join must not cross banks."""
    bank_a = f"test-ent-a-{uuid.uuid4().hex[:8]}"
    bank_b = f"test-ent-b-{uuid.uuid4().hex[:8]}"
    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with memory._pool.acquire() as conn:
        e_b = await _mk_entity(conn, bank_b, "atlas")
        unit_b = await _mk_unit(conn, bank_b, "bank-b atlas fact.")
        await _link(conn, unit_b, e_b)
        # bank A has a recent unit but no atlas entity/link
        await _mk_unit(conn, bank_a, "unrelated fact.")

        assert (
            await any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_a,
                since=since,
                tag_groups=_groups({"entities": ["atlas"]}),
            )
            is False
        )
