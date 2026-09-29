"""Lint test: ``flush_pending_stats()`` must never run while a pooled connection is held.

``flush_pending_stats()`` acquires its own connection. The write that precedes it
is only committed when the enclosing ``acquire_with_retry(...)`` block exits — on
Oracle explicitly (oracledb does not autocommit; the backend commits on clean exit
of ``acquire()``), on PostgreSQL implicitly.

Calling it *inside* that block therefore deadlocks permanently on Oracle:
connection #2 waits on the row locks the still-open connection #1 holds on
``entities``, while connection #1 cannot commit until the call returns. Oracle
never raises ORA-00060 for this, because session #1 is blocked in Python rather
than on the database — so retain simply hangs forever.

This regressed in three separate call sites at once (both retain paths and the
transfer importer) and only showed up as a CI timeout, so it is guarded
structurally rather than behaviourally: the deadlock cannot be reproduced against
PostgreSQL, which is what the test suite runs on.
"""

import ast
from pathlib import Path

import pytest

_ENGINE = Path(__file__).resolve().parent.parent / "hindsight_api" / "engine"

# Files that call flush_pending_stats() after a write.
_FILES = [
    _ENGINE / "retain" / "orchestrator.py",
    _ENGINE / "transfer" / "importer.py",
]


def _acquires_connection(node: ast.AsyncWith) -> bool:
    """True if this ``async with`` checks a connection out of the pool."""
    for item in node.items:
        call = item.context_expr
        if isinstance(call, ast.Call):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in {"acquire_with_retry", "acquire"}:
                return True
    return False


def _flush_calls(node: ast.AST) -> list[int]:
    """Line numbers of every flush_pending_stats() call under ``node``."""
    return [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "flush_pending_stats"
    ]


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_flush_pending_stats_is_not_called_while_holding_a_connection(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncWith) and _acquires_connection(node):
            for lineno in _flush_calls(node):
                offenders.append(f"{path.name}:{lineno}")

    assert not offenders, (
        "flush_pending_stats() is called inside a connection-holding block at "
        f"{', '.join(offenders)}. It acquires its own connection, so on Oracle this "
        "deadlocks forever (the outer connection cannot commit and release its row "
        "locks until the call returns). Move the call after the acquire block exits."
    )


def test_guard_detects_the_bug_it_is_meant_to_catch():
    """The guard must actually fire on the offending shape (not vacuously pass)."""
    bad = ast.parse(
        "async def f():\n"
        "    async with acquire_with_retry(pool) as conn:\n"
        "        await conn.execute('...')\n"
        "        await entity_resolver.flush_pending_stats()\n"
    )
    found = [
        lineno
        for node in ast.walk(bad)
        if isinstance(node, ast.AsyncWith) and _acquires_connection(node)
        for lineno in _flush_calls(node)
    ]
    assert found == [4]
