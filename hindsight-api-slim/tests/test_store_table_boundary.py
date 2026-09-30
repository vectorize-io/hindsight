"""Only the Postgres memories store may run SQL on the tables the store owns (#4969).

When another memories store is configured, it owns a bank's memories, links,
entities, documents and chunks, and Postgres holds none of them. SQL on those
tables from anywhere else reads empty tables for that bank: wrong results, a
crash, or rows nothing will ever read. ``fq_table`` refuses the names at runtime;
this test catches what the runtime guard cannot see:

* SQL text naming a store table directly (``FROM memory_units``), without a
  resolver call;
* a call to the guard-free resolver (``fq_store_table``) outside the store;
* code that is never run by a test.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from hindsight_api.engine.schema import STORE_TABLES, StoreTableAccessError, fq_table, fq_table_explicit

PKG = Path(__file__).resolve().parent.parent / "hindsight_api"

# Where the store's own SQL lives, plus schema management.
ALLOWED = (
    "engine/memories/pg/",
    "engine/memories/postgres.py",
    "alembic/",
    "migrations.py",
    "models.py",
    # The SQL dialect layer the store's writes go through.
    "engine/db/",
    "engine/schema.py",
)

_TABLES = "|".join(sorted(STORE_TABLES))
# A table named right after the SQL keyword that reads or writes it, optionally schema-qualified.
_SQL_TABLE = re.compile(
    rf"\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+(?:\"?[\w{{}}]+\"?\.)?\"?({_TABLES})\b(?!\s*\()",
)
_PRIVILEGED = {"fq_store_table", "fq_store_table_explicit"}


def _strings(node: ast.AST) -> list[str]:
    """The literal SQL text of a string or f-string node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return ["".join(v.value if isinstance(v, ast.Constant) else "{x}" for v in node.values)]
    return []


def _violations(path: Path) -> list[str]:
    rel = path.relative_to(PKG).as_posix()
    tree = ast.parse(path.read_text(), filename=rel)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name in _PRIVILEGED for a in node.names):
            out.append(f"{rel}:{node.lineno}: imports the store's private table resolver")
        elif isinstance(node, ast.Name) and node.id in _PRIVILEGED:
            out.append(f"{rel}:{node.lineno}: uses the store's private table resolver")
        elif (
            isinstance(node, ast.Call)
            and "fq_table" in ast.unparse(node.func)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in STORE_TABLES
        ):
            out.append(f"{rel}:{node.lineno}: names store table {node.args[0].value!r}")
        for text in _strings(node):
            m = _SQL_TABLE.search(text)
            if m:
                out.append(f"{rel}:{node.lineno}: SQL on store table {m.group(1)!r}")
    return out


def test_no_store_table_sql_outside_the_store() -> None:
    found: list[str] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith(ALLOWED):
            continue
        found.extend(_violations(path))
    assert not found, "Direct access to store-owned tables outside the memories store:\n" + "\n".join(
        sorted(set(found))
    )


def test_resolvers_refuse_store_tables() -> None:
    for table in STORE_TABLES:
        for resolve in (fq_table, fq_table_explicit):
            try:
                resolve(table)
            except StoreTableAccessError:
                continue
            raise AssertionError(f"{resolve.__name__}({table!r}) resolved a store table")
    assert fq_table_explicit("banks") == "banks"
