"""Tests for the deterministic migration enumeration in the changelog generator."""

import re
import subprocess
from pathlib import Path

import pytest

from hindsight_dev import generate_changelog
from hindsight_dev.generate_changelog import (
    MIGRATIONS_DIR,
    Migration,
    _parse_migration_file,
    _pr_number_from_subject,
    build_changelog_markdown,
    extract_tables,
    get_new_migrations,
    render_migrations_section,
)

MIGRATION_TEMPLATE = '''"""{description}

Revision ID: {revision}
Revises: {down}
Create Date: 2026-01-01
"""

revision: str = "{revision}"
down_revision: str | None = "{down}"
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _commit_migration(repo: Path, filename: str, revision: str, description: str, down: str, subject: str) -> None:
    path = repo / MIGRATIONS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MIGRATION_TEMPLATE.format(description=description, revision=revision, down=down))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", subject)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo with a tagged baseline, used as the generator's REPO_PATH."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "chore: base")
    _git(repo, "tag", "v1.0.0")
    monkeypatch.setattr(generate_changelog, "REPO_PATH", repo)
    return repo


def test_pr_number_takes_the_merge_pr_not_the_issue_refs() -> None:
    assert _pr_number_from_subject("fix(x): thing (#3361, #3273) (#3622)") == 3622
    assert _pr_number_from_subject("feat(y): thing (#3552)") == 3552
    assert _pr_number_from_subject("chore: no pr reference") is None


def test_parse_migration_file_uses_revision_constant_and_docstring() -> None:
    source = MIGRATION_TEMPLATE.format(description="Add a column", revision="abc123def456", down="000000000000")
    doc = _parse_migration_file(source, f"{MIGRATIONS_DIR}/abc123def456_add_a_column.py")
    assert (doc.revision, doc.description) == ("abc123def456", "Add a column")


def test_parse_migration_file_falls_back_to_the_filename() -> None:
    doc = _parse_migration_file("# no docstring\n", f"{MIGRATIONS_DIR}/abc123_add_foo_index.py")
    assert doc.revision == "abc123"
    assert doc.description == "add foo index"


def test_enumerates_new_migrations_oldest_first_with_pr_numbers(repo: Path) -> None:
    _commit_migration(repo, "aaa111_first.py", "aaa111", "First migration", "000000", "feat(db): first (#101)")
    _commit_migration(repo, "bbb222_second.py", "bbb222", "Second migration", "aaa111", "fix(db): second (#7) (#202)")
    _git(repo, "tag", "v1.1.0")

    migrations = get_new_migrations("v1.0.0", "v1.1.0")

    assert [(m.revision, m.description, m.pr) for m in migrations] == [
        ("aaa111", "First migration", 101),
        ("bbb222", "Second migration", 202),
    ]


def test_migrations_are_scoped_to_the_release_range(repo: Path) -> None:
    _commit_migration(repo, "aaa111_first.py", "aaa111", "First migration", "000000", "feat(db): first (#101)")
    _git(repo, "tag", "v1.1.0")
    _commit_migration(repo, "bbb222_second.py", "bbb222", "Second migration", "aaa111", "feat(db): second (#202)")
    _git(repo, "tag", "v1.2.0")

    assert [m.revision for m in get_new_migrations("v1.1.0", "v1.2.0")] == ["bbb222"]


def test_migration_removed_before_the_tag_is_not_listed(repo: Path) -> None:
    _commit_migration(repo, "aaa111_first.py", "aaa111", "First migration", "000000", "feat(db): first (#101)")
    (repo / MIGRATIONS_DIR / "aaa111_first.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "revert: drop the migration (#102)")
    _git(repo, "tag", "v1.1.0")

    assert get_new_migrations("v1.0.0", "v1.1.0") == []


def test_release_with_no_migrations_renders_no_section(repo: Path) -> None:
    (repo / "README.md").write_text("changed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "docs: tweak (#303)")
    _git(repo, "tag", "v1.1.0")

    assert get_new_migrations("v1.0.0", "v1.1.0") == []
    assert render_migrations_section([]) == []


def test_section_links_to_the_pr_and_falls_back_to_the_commit() -> None:
    section = "\n".join(
        render_migrations_section(
            [
                Migration("aaa111", "First migration", f"{MIGRATIONS_DIR}/aaa111_first.py", "deadbee", 101),
                Migration("bbb222", "Second migration", f"{MIGRATIONS_DIR}/bbb222_second.py", "cafed00", None),
            ]
        )
    )

    assert "**Database Migrations**" in section
    assert "`aaa111` — First migration" in section
    assert "https://github.com/vectorize-io/hindsight/pull/101" in section
    assert ">#101</a>" in section
    # No PR in the subject (e.g. a direct push): link the commit instead.
    assert "https://github.com/vectorize-io/hindsight/commit/cafed00" in section


def test_migrations_only_release_is_not_reported_as_maintenance_only() -> None:
    markdown = build_changelog_markdown(
        "1.1.0",
        "v1.1.0",
        entries=[],
        migrations=[Migration("aaa111", "First migration", f"{MIGRATIONS_DIR}/aaa111_first.py", "deadbee", 101)],
    )

    assert "**Database Migrations**" in markdown
    assert "internal maintenance" not in markdown


def test_extracts_tables_from_alembic_ops_and_raw_sql() -> None:
    source = '''"""Add a column

Revision ID: abc123
"""

revision: str = "abc123"


def _pg_upgrade() -> None:
    op.add_column("mental_models", sa.Column("last_memory_seen_at", sa.DateTime()))
    op.execute(f"CREATE INDEX idx_x ON {schema}memory_units(bank_id, updated_at)")
    op.execute(f"UPDATE {schema}banks SET config = '{{}}'")
'''
    assert extract_tables(source) == ("memory_units", "mental_models", "banks")


def test_prose_and_read_only_references_are_not_reported_as_touched() -> None:
    source = '''"""Repair the index.

The reconcile path used to create this on memory_units at runtime.
"""

revision: str = "abc123"


def _pg_upgrade() -> None:
    # entities is only read here, to seed the queue
    op.execute("INSERT INTO entity_maintenance_queue (id) SELECT id FROM entities")
'''
    assert extract_tables(source) == ("entity_maintenance_queue",)


def test_dropped_index_attributes_the_table_it_locks() -> None:
    source = 'revision: str = "abc123"\nop.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_embedding")\n'
    assert extract_tables(source) == ("memory_units",)


def test_tables_are_ordered_by_volume() -> None:
    source = (
        'revision: str = "abc123"\n'
        'op.add_column("banks", c)\n'
        'op.add_column("documents", c)\n'
        'op.add_column("memory_units", c)\n'
    )
    assert extract_tables(source) == ("memory_units", "documents", "banks")


def test_high_volume_tables_get_a_release_level_warning() -> None:
    high = Migration("aaa111", "Reindex", f"{MIGRATIONS_DIR}/aaa111_x.py", "deadbee", 101, ("memory_units",))
    low = Migration("bbb222", "Tweak", f"{MIGRATIONS_DIR}/bbb222_y.py", "cafed00", 102, ("banks",))

    with_high = "\n".join(render_migrations_section([high, low]))
    assert "high-volume tables (`memory_units`)" in with_high
    assert "<code>memory_units</code>" in with_high
    assert "high volume" in with_high
    assert "<code>banks</code>" in with_high

    assert "high-volume tables" not in "\n".join(render_migrations_section([low]))


def test_every_table_a_migration_creates_is_classified() -> None:
    """A new table must be given a volume, or it silently drops out of the section."""
    versions = Path(__file__).resolve().parents[2] / MIGRATIONS_DIR
    created: set[str] = set()
    for migration_file in versions.glob("*.py"):
        body = generate_changelog._strip_prose(migration_file.read_text())
        created |= set(re.findall(r'op\.create_table\(\s*["\']([a-z_][a-z0-9_]*)["\']', body))
        created |= set(
            re.findall(
                r"(?i)\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\{schema\}|\"[^\"]*\"\.)?\"?([a-z_][a-z0-9_]*)",
                body,
            )
        )

    assert created, "no CREATE TABLE found — the extraction patterns regressed"
    assert created <= generate_changelog.TABLE_VOLUME.keys(), (
        f"unclassified tables in TABLE_VOLUME: {sorted(created - generate_changelog.TABLE_VOLUME.keys())}"
    )
