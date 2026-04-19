"""Graph-level sanity checks for the Alembic migration DAG.

These tests do not touch a database; they only parse the revision files on
disk, so they are cheap to run in CI and catch DAG accidents (divergent
heads, unreachable revisions) at merge time instead of at deploy time.
"""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def _script_directory() -> ScriptDirectory:
    cfg = Config()
    script_location = Path(__file__).parent.parent / "hindsight_api" / "alembic"
    cfg.set_main_option("script_location", str(script_location))
    return ScriptDirectory.from_config(cfg)


def test_single_head() -> None:
    """The DAG must have exactly one head.

    A second head means a branch was added without a merge revision, which
    makes ``alembic upgrade head`` (singular) ambiguous and forces the next
    migration author to orphan whichever head they don't pick as parent.
    v0.5.3 shipped in exactly that state; this test would have caught it.

    Fix for a new head: ``alembic merge heads -m "<reason>"``.
    """
    script = _script_directory()
    heads = script.get_heads()
    assert len(heads) == 1, (
        f"Alembic has {len(heads)} heads ({heads}); expected exactly 1. "
        "Unify them with ``alembic merge heads -m '<reason>'``."
    )


def test_single_base() -> None:
    """The DAG must have exactly one base (the initial schema).

    Multiple bases mean disconnected migration trees, which can only happen
    through manual file edits.
    """
    script = _script_directory()
    bases = script.get_bases()
    assert len(bases) == 1, f"Alembic has {len(bases)} bases ({bases}); expected exactly 1."
