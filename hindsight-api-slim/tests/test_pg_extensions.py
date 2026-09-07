"""Every PostgreSQL extension Hindsight installs must land in the public schema.

Regression cover for #4118: in schema mode the migration search_path starts with
the tenant schema, so a bare ``CREATE EXTENSION pg_trgm`` installed pg_trgm
there. The runtime connects with the default search_path and fully-qualifies its
tables, so it could not resolve ``%`` and every retain failed forever, silently.
"""

from pathlib import Path

import pytest

from hindsight_api._pg_extensions import (
    MANAGED_EXTENSIONS,
    create_extension,
    ensure_extensions_in_public,
    extension_schema,
    relocate_extension_to_public,
)
from hindsight_api._vector_index import bootstrap_extension
from hindsight_api.migrations import _bootstrap_vector_extension_for_migrations
from tests.pg_extension_fakes import FakePgConnection

TENANT_SEARCH_PATH = '"hindsight", public'


def test_create_extension_pins_install_schema_to_public():
    conn = FakePgConnection(search_path=TENANT_SEARCH_PATH)

    create_extension(conn, "pg_trgm")

    assert conn.extensions["pg_trgm"][0] == "public"
    assert conn.statements[0].startswith("SELECT current_setting('search_path')")
    assert conn.params[1] == {"schema": "public"}
    assert conn.statements[2] == "CREATE EXTENSION IF NOT EXISTS pg_trgm"


def test_create_extension_restores_the_callers_search_path():
    conn = FakePgConnection(search_path=TENANT_SEARCH_PATH)

    create_extension(conn, "pg_trgm")

    assert conn.search_path == TENANT_SEARCH_PATH
    assert conn.params[-1] == {"previous": TENANT_SEARCH_PATH}


def test_create_extension_appends_cascade_only_when_asked():
    conn = FakePgConnection()

    create_extension(conn, "pgroonga", cascade=True)

    assert "CREATE EXTENSION IF NOT EXISTS pgroonga CASCADE" in conn.statements


def test_create_extension_propagates_failures_without_masking_them():
    conn = FakePgConnection(search_path=TENANT_SEARCH_PATH, fail_on="CREATE EXTENSION")

    with pytest.raises(RuntimeError, match="simulated failure"):
        create_extension(conn, "pg_trgm")


def test_create_extension_rejects_names_that_are_not_identifiers():
    conn = FakePgConnection()

    with pytest.raises(ValueError, match="Invalid PostgreSQL extension name"):
        create_extension(conn, 'pg_trgm"; DROP TABLE banks; --')

    assert conn.statements == []


def test_extension_schema_reports_where_an_extension_lives():
    conn = FakePgConnection(extensions={"vector": ("public", True)})

    assert extension_schema(conn, "vector") == "public"
    assert extension_schema(conn, "pg_trgm") is None


def test_relocate_moves_a_misplaced_relocatable_extension_into_public():
    conn = FakePgConnection(extensions={"pg_trgm": ("hindsight", True)})

    assert relocate_extension_to_public(conn, "pg_trgm") is True
    assert conn.extensions["pg_trgm"][0] == "public"
    assert 'ALTER EXTENSION pg_trgm SET SCHEMA "public"' in conn.statements


def test_relocate_is_a_noop_when_already_public_or_absent():
    conn = FakePgConnection(extensions={"vector": ("public", True)})

    assert relocate_extension_to_public(conn, "vector") is False
    assert relocate_extension_to_public(conn, "pg_trgm") is False
    assert not any("ALTER EXTENSION" in s for s in conn.statements)


def test_relocate_leaves_non_relocatable_extensions_alone():
    # vchord_bm25 pins bm25_catalog in its control file; PostgreSQL rejects the
    # move, and the runtime puts that schema on its search_path instead.
    conn = FakePgConnection(extensions={"vchord_bm25": ("bm25_catalog", False)})

    assert relocate_extension_to_public(conn, "vchord_bm25") is False
    assert not any("ALTER EXTENSION" in s for s in conn.statements)


def test_relocate_survives_a_permission_denied_alter():
    conn = FakePgConnection(extensions={"pg_trgm": ("hindsight", True)}, fail_on="ALTER EXTENSION")

    assert relocate_extension_to_public(conn, "pg_trgm") is False
    assert conn.rollbacks == 1


def test_ensure_extensions_in_public_repairs_every_managed_extension():
    conn = FakePgConnection(
        extensions={
            "vector": ("hindsight", True),
            "pg_trgm": ("hindsight", True),
            "pgroonga": ("public", True),
        }
    )

    ensure_extensions_in_public(conn)

    assert conn.extensions["vector"][0] == "public"
    assert conn.extensions["pg_trgm"][0] == "public"
    assert conn.extensions["pgroonga"][0] == "public"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("pgvector", ["vector"]),
        ("pgvectorscale", ["vector", "vectorscale"]),
        ("vchord", ["vchord"]),
        ("scann", ["vector", "alloydb_scann"]),
    ],
)
def test_vector_backends_install_into_public_in_schema_mode(backend, expected):
    conn = FakePgConnection(search_path=TENANT_SEARCH_PATH)

    bootstrap_extension(conn, backend)

    assert conn.created_extensions() == expected
    assert all(conn.extensions[name][0] == "public" for name in expected)


def test_migration_bootstrap_relocates_a_tenant_schema_pg_trgm():
    # The #4118 shape: an existing deployment whose pg_trgm was created inside
    # the tenant schema by an older version. Startup must repair it.
    conn = FakePgConnection(
        search_path=TENANT_SEARCH_PATH,
        extensions={"vector": ("public", True), "pg_trgm": ("hindsight", True)},
    )

    _bootstrap_vector_extension_for_migrations(conn, "pgvector")

    assert conn.extensions["pg_trgm"][0] == "public"


def test_managed_extensions_covers_every_extension_the_code_installs():
    source = _api_sources()
    for name in _extension_names_in_source(source):
        assert name in MANAGED_EXTENSIONS, f"{name} is created but not listed in MANAGED_EXTENSIONS"


def test_no_raw_create_extension_outside_the_helper():
    """All extension creation goes through create_extension(), so the rule holds everywhere."""
    offenders = [
        path
        for path, source in _api_sources().items()
        if path.name != "_pg_extensions.py" and "CREATE EXTENSION IF NOT EXISTS" in source
    ]

    assert offenders == [], (
        "These modules build CREATE EXTENSION SQL directly; use "
        "hindsight_api._pg_extensions.create_extension so the extension lands in public: "
        f"{[str(p) for p in offenders]}"
    )


def _api_sources() -> dict[Path, str]:
    root = Path(__file__).resolve().parent.parent / "hindsight_api"
    return {path: path.read_text() for path in root.rglob("*.py")}


def _extension_names_in_source(sources: dict[Path, str]) -> set[str]:
    import re

    names: set[str] = set()
    for path, source in sources.items():
        if path.name == "_pg_extensions.py":
            continue
        names.update(re.findall(r'create_extension\(\s*[^,]+,\s*"(\w+)"', source))
    return names
