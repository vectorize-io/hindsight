"""The link-expansion arms must all project the same ``memory_units`` columns.

Every arm of link expansion (entity / semantic / causal, and their observation
variants) is combined with ``UNION ALL`` and read back positionally, so the arms
have to agree on both the set *and* the order of the leading columns. Before
``memory_unit_columns`` the list was spelled out ~20 times across the two
backends; these tests are what makes that single source of truth enforceable,
and they fail loudly if a future arm hand-rolls the projection again.
"""

import re

import pytest

from hindsight_api.engine.db.ops import MEMORY_UNIT_COLUMNS, UpdatedWindow, memory_unit_columns
from hindsight_api.engine.db.ops_oracle import OracleOps
from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps

WINDOWS = [
    UpdatedWindow(after=None, before=None, first_param_index=4),
    UpdatedWindow(after=None, before=None, first_param_index=1),
]


def test_memory_unit_columns_qualifies_every_column():
    rendered = memory_unit_columns("mu")
    assert [c.strip() for c in rendered.replace("\n", " ").split(",")] == [
        f"mu.{column}" for column in MEMORY_UNIT_COLUMNS
    ]


def test_memory_unit_columns_unqualified_by_default():
    rendered = memory_unit_columns()
    assert [c.strip() for c in rendered.replace("\n", " ").split(",")] == list(MEMORY_UNIT_COLUMNS)


def test_memory_unit_columns_indents_continuation_lines_only():
    rendered = memory_unit_columns("mu", indent=4)
    lines = rendered.split("\n")
    assert len(lines) > 1, "expected the projection to wrap"
    assert not lines[0].startswith(" "), "the first line sits after SELECT and must not be indented"
    assert all(line.startswith("    ") for line in lines[1:])


def _arm_projections(sql: str) -> dict[str, list[str]]:
    """Leading column list of each ``<name> AS ( SELECT ... )`` arm in a CTE body."""
    arms: dict[str, list[str]] = {}
    for match in re.finditer(r"(\w+) AS \(\s*SELECT(?: DISTINCT ON \([^)]*\))?\s*([\s\S]*?)\n\s*FROM ", sql):
        name, projection = match.group(1), match.group(2)
        columns = [c.strip().split(".")[-1] for c in projection.split(",")]
        arms[name] = columns[: len(MEMORY_UNIT_COLUMNS)]
    return arms


@pytest.mark.parametrize("ops", [PostgreSQLOps(), OracleOps()], ids=["pg", "oracle"])
@pytest.mark.parametrize("window", WINDOWS, ids=["window_at_4", "window_at_1"])
def test_expansion_arms_share_the_memory_unit_projection(ops, window):
    sql = ops.build_entity_expansion_cte("mu_t", "ue_t", 5, window) + ",\n"
    sql += ops.build_semantic_causal_cte("ml_t", "mu_t", window)

    arms = _arm_projections(sql)
    expanded = {name: cols for name, cols in arms.items() if name.endswith("_expanded")}
    assert expanded, f"no expansion arms parsed out of:\n{sql}"

    for name, columns in expanded.items():
        assert columns == list(MEMORY_UNIT_COLUMNS), f"arm {name!r} projects {columns}"


def test_postgres_semantic_arm_groups_by_every_projected_column():
    """A projected column missing from GROUP BY is a runtime error, not a test failure.

    PostgreSQL only: Oracle cannot GROUP BY its CLOB columns, so its semantic arm
    deliberately groups by ``id`` alone in a scores subquery and joins back for the
    full projection.
    """
    sql = PostgreSQLOps().build_semantic_causal_cte("ml_t", "mu_t", WINDOWS[0])
    group_by = re.search(r"GROUP BY ([\s\S]*?)\n\s*ORDER BY", sql)
    assert group_by is not None, f"expected a grouped semantic arm in:\n{sql}"
    grouped = {c.strip().split(".")[-1] for c in group_by.group(1).split(",")}
    assert grouped == set(MEMORY_UNIT_COLUMNS)
