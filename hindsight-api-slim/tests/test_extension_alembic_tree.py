"""An extension can ship its own Alembic tree, applied with core's.

Driven against a real database rather than by inspecting config, because every
interesting failure here is a runtime one: a `version_locations` that silently
drops core's revisions, a second head that the runner refuses, or a tree that is
collected but never applied. None of those are visible from the Config object.

Runs via: uv run pytest tests/test_extension_alembic_tree.py -v
"""

from __future__ import annotations

import re
import textwrap
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from hindsight_api.extensions.loader import EXTENSION_KINDS, collect_alembic_version_locations
from hindsight_api.extensions.tenant import TenantExtension

CORE_ALEMBIC = Path(__file__).resolve().parent.parent / "hindsight_api" / "alembic"

#: A standalone tree: its own base (`down_revision = None`), its own head, its own
#: branch label. It creates a table so the test can prove it actually RAN, rather
#: than only that it was collected.
_REVISION = textwrap.dedent(
    '''
    """extension tree probe"""

    from alembic import context, op

    from hindsight_api.alembic._dialect import run_for_dialect

    revision = "ext0probe0001"
    down_revision = None
    branch_labels = ("ext_probe",)
    depends_on = None


    def _schema_prefix() -> str:
        schema = context.config.get_main_option("target_schema")
        return f'"{schema}".' if schema else ""


    def _pg_upgrade() -> None:
        op.execute(f"CREATE TABLE IF NOT EXISTS {_schema_prefix()}ext_probe (id int)")


    def _pg_downgrade() -> None:
        op.execute(f"DROP TABLE IF EXISTS {_schema_prefix()}ext_probe")


    _oracle_upgrade = _pg_upgrade
    _oracle_downgrade = _pg_downgrade


    def upgrade() -> None:
        run_for_dialect(pg=_pg_upgrade, oracle=_oracle_upgrade)


    def downgrade() -> None:
        run_for_dialect(pg=_pg_downgrade, oracle=_oracle_downgrade)
    '''
).strip()


@pytest.fixture
def extension_tree(tmp_path: Path) -> Path:
    versions = tmp_path / "ext_alembic" / "versions"
    versions.mkdir(parents=True)
    (versions / "ext0probe0001_probe.py").write_text(_REVISION)
    return versions


class _ProbeExtension(TenantExtension):
    """Declares the tree above.

    A TenantExtension because the loader checks the base class per kind; the hook
    itself lives on `Extension`, so any kind would serve. The two abstract methods
    are satisfied and never called — this extension exists only to be asked for its
    migrations.
    """

    location: str = ""

    def alembic_version_locations(self) -> list[str]:
        return [type(self).location]

    async def authenticate(self, context):  # pragma: no cover - never called
        raise NotImplementedError

    async def list_tenants(self):  # pragma: no cover - never called
        return []


def test_a_missing_location_is_skipped_not_fatal(monkeypatch, tmp_path):
    """An extension naming a directory that is not there must not break migrations.

    Alembic treats a missing version location as a hard error, so collecting one
    blindly would turn an extension's packaging mistake into an unmigratable
    database — a far worse failure than that extension's state being stale.
    """
    _ProbeExtension.location = str(tmp_path / "does-not-exist")
    monkeypatch.setenv(
        "HINDSIGHT_API_TENANT_EXTENSION",
        f"{__name__}:_ProbeExtension",
    )
    assert collect_alembic_version_locations() == []


def test_a_broken_extension_does_not_block_migrations(monkeypatch):
    """An extension that cannot even be imported is skipped with a warning.

    Same reasoning: a misconfigured extension must not be able to stop the
    database being migrated.
    """
    monkeypatch.setenv("HINDSIGHT_API_TENANT_EXTENSION", "no.such.module:Nope")
    assert collect_alembic_version_locations() == []


def test_extension_kinds_covers_every_base():
    """Every extension base class must be listed in EXTENSION_KINDS.

    Adding a seventh kind and forgetting this registry fails silently: extensions of
    that kind are simply never asked for their revisions, and nothing errors. So the
    family is enumerated from the source tree rather than from the registry itself —
    a registry cannot prove its own completeness.
    """
    declared = {
        m.group(1)
        for path in CORE_ALEMBIC.parent.rglob("*.py")
        for m in re.finditer(r"^class (\w+)\(Extension(?:, ABC)?\):", path.read_text(), re.MULTILINE)
    }
    listed = {kind.class_name for kind in EXTENSION_KINDS}
    assert declared == listed, (
        f"EXTENSION_KINDS is out of date: {declared - listed} declared but unlisted "
        f"(their extensions would never contribute migrations), {listed - declared} listed but gone."
    )


def test_the_extension_tree_is_collected(monkeypatch, extension_tree):
    _ProbeExtension.location = str(extension_tree)
    monkeypatch.setenv(
        "HINDSIGHT_API_TENANT_EXTENSION",
        f"{__name__}:_ProbeExtension",
    )
    found = collect_alembic_version_locations()
    assert found == [str(extension_tree.resolve())]


def test_core_and_extension_revisions_both_apply(monkeypatch, extension_tree, pg0_db_url):
    """The load-bearing one: BOTH trees run, in one migration, into one schema.

    Asserting core applied as well as the extension is the point. Setting
    `version_locations` replaces Alembic's default rather than adding to it, so a
    naive implementation runs only the extension's revisions and reports success —
    a migration that appears to work while applying none of core's.
    """
    from hindsight_api.migrations import run_migrations

    _ProbeExtension.location = str(extension_tree)
    monkeypatch.setenv(
        "HINDSIGHT_API_TENANT_EXTENSION",
        f"{__name__}:_ProbeExtension",
    )

    schema = f"tenant_exttree_{uuid.uuid4().hex[:8]}"
    run_migrations(pg0_db_url, schema=schema)

    engine = create_engine(pg0_db_url)
    with engine.connect() as conn:
        # The extension's own table.
        assert _table_exists(conn, schema, "ext_probe"), "the extension's revision did not run"
        # ...and a core table, so we know core was not displaced.
        assert _table_exists(conn, schema, "memory_units"), (
            "core revisions did not run — version_locations replaced them instead of adding to them"
        )
        # Two independent trees means two heads, each with its own row. That is
        # native Alembic, and it is why the runner upgrades to "heads", plural.
        heads = _version_rows(conn, schema)
        assert "ext0probe0001" in heads, f"the extension head is not recorded: {heads}"
        assert len(heads) >= 2, f"expected core and extension heads, got {heads}"


def _table_exists(conn, schema: str, table: str) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM information_schema.tables WHERE table_schema = :schema AND table_name = :table"),
        {"schema": schema, "table": table},
    ).fetchone()
    return row is not None


def _version_rows(conn, schema: str) -> set[str]:
    rows = conn.execute(text(f'SELECT version_num FROM "{schema}".alembic_version')).fetchall()
    return {r[0] for r in rows}
