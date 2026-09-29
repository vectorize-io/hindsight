"""Every text-search backend must push `min_scores.keyword` into its BM25 arm.

`bm25_min_score` started life in #1947 as a VectorChord-only gate (vchord's `<&>`
ranks every document, so it needed the analogue of native tsvector's boolean `@@`
match gate). #2422 then built the public `min_scores.keyword` floor on top of that
same parameter without revisiting the backends, so four of the six branches —
including `native`, the default — accepted the floor and silently ignored it: a
caller asking for `keyword >= 0.30` got rows scoring 0.2 back.

These are SQL-shape assertions rather than end-to-end queries so that a new
backend branch cannot repeat the omission without a red test, on machines with no
VectorChord/pgroonga/pg_search/Oracle available.
"""

import pytest

from hindsight_api.config import VALID_TEXT_SEARCH_EXTENSIONS
from hindsight_api.engine.sql.oracle import OracleDialect
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

# Enumerate the family from config rather than restating it, so a sixth backend
# is covered by these assertions the moment it becomes selectable.
PG_EXTENSIONS = VALID_TEXT_SEARCH_EXTENSIONS

# The backends that gate on the score even with no caller floor, because their
# operator ranks every document instead of pre-filtering to query-term matches.
RANKS_EVERY_DOC = ("vchord",)

ARM_KWARGS = dict(
    table="memory_units",
    cols="id, text",
    fact_type="world",
    bank_id_param="$2",
    limit_param="$3",
    text_param="$4",
)


def _pg_arm(extension: str, bm25_min_score: float) -> str:
    return PostgreSQLDialect().build_bm25_arm(
        **ARM_KWARGS,
        text_search_extension=extension,
        bm25_min_score=bm25_min_score,
    )


@pytest.mark.parametrize("extension", PG_EXTENSIONS)
def test_pg_backend_applies_the_caller_floor(extension):
    """A positive floor reaches the SQL on every backend, not just vchord."""
    assert ">= 0.3" in _pg_arm(extension, 0.3)


def test_oracle_applies_the_caller_floor():
    arm = OracleDialect().build_bm25_arm(**ARM_KWARGS, bm25_min_score=0.3)
    assert ">= 0.3" in arm


@pytest.mark.parametrize("extension", PG_EXTENSIONS)
def test_pg_floor_is_inclusive_not_exclusive(extension):
    """`min_scores` floors are documented as inclusive, and the semantic arm uses
    `>= min_similarity`. The keyword arm must agree: a row scoring exactly the
    floor is kept."""
    arm = _pg_arm(extension, 0.3)
    assert "> 0.3" not in arm.replace(">= 0.3", "")


@pytest.mark.parametrize("extension", PG_EXTENSIONS)
def test_pg_default_floor_keeps_the_structural_match_gate(extension):
    """At the 0.0 default there is no caller floor, so behaviour is unchanged:
    backends whose operator ranks every row keep their `> 0` gate, and backends
    with a boolean match gate get no score predicate at all."""
    arm = _pg_arm(extension, 0.0)
    if extension in RANKS_EVERY_DOC:
        assert "> 0" in arm
    else:
        assert "bm25_score >" not in arm
        assert "bm25_score >=" not in arm


def test_oracle_default_floor_keeps_the_structural_match_gate():
    arm = OracleDialect().build_bm25_arm(**ARM_KWARGS, bm25_min_score=0.0)
    assert "> 0" in arm


@pytest.mark.parametrize("extension", PG_EXTENSIONS)
def test_pg_arm_is_a_single_union_all_ready_subquery(extension):
    """The arms are joined with UNION ALL, so wrapping one to apply the floor from
    the outside must keep it a single parenthesised subquery."""
    arm = _pg_arm(extension, 0.3)
    assert arm.startswith("(") and arm.endswith(")")
    assert "bm25_score" in arm
    assert "'bm25' AS source" in arm
