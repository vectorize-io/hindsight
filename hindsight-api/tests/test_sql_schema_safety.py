"""
Safety tests to ensure all SQL queries use fully-qualified table names.

This prevents cross-tenant data access by ensuring every table reference
includes the schema prefix (e.g., public.memory_units instead of just memory_units).
"""

import re
from pathlib import Path

import pytest

# All tables that MUST be schema-qualified in SQL queries
TABLES = [
    "memory_units",
    "memory_links",
    "unit_entities",
    "entities",
    "entity_cooccurrences",
    "banks",
    "documents",
    "chunks",
    "async_operations",
    "directives",
    "mental_models",
]

# Files to scan for SQL queries
SCAN_PATHS = [
    "hindsight_api/engine",
    "hindsight_api/api",
]

# Files to exclude (e.g., migrations, tests)
EXCLUDE_PATTERNS = [
    "alembic",
    "__pycache__",
    "test_",
]


def get_python_files() -> list[Path]:
    """Get all Python files to scan."""
    root = Path(__file__).parent.parent
    files = []
    for scan_path in SCAN_PATHS:
        path = root / scan_path
        if path.exists():
            for py_file in path.rglob("*.py"):
                # Check exclusions
                if any(excl in str(py_file) for excl in EXCLUDE_PATTERNS):
                    continue
                files.append(py_file)
    return files


def find_unqualified_table_refs(content: str, filename: str) -> list[tuple[int, str, str]]:
    """
    Find SQL statements with unqualified table references.

    Returns list of (line_number, table_name, line_content).
    """
    violations = []

    # Patterns that indicate SQL context
    sql_keywords = r"(?:FROM|JOIN|INTO|UPDATE|DELETE\s+FROM)\s+"

    # Additional SQL indicators to confirm this is actually SQL, not prose
    sql_indicators = re.compile(
        r"(SELECT|INSERT|DELETE|UPDATE|CREATE|ALTER|DROP|WHERE|SET|VALUES|"
        r'f"""|f\'\'\'|""".*SELECT|\'\'\'.*SELECT)',
        re.IGNORECASE,
    )

    lines = content.split("\n")
    for line_num, line in enumerate(lines, 1):
        # Skip comments and strings that are clearly not SQL
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for table in TABLES:
            # Pattern: SQL keyword followed by unqualified table name
            # Should match: FROM memory_units, JOIN memory_units, INTO memory_units
            # Should NOT match: FROM public.memory_units, FROM {schema}.memory_units
            # Should NOT match: fq_table("memory_units")

            # Check for unqualified table after SQL keyword
            pattern = rf"{sql_keywords}{table}(?:\s|$|,|\))"

            if re.search(pattern, line, re.IGNORECASE):
                # Check if it's actually qualified (has schema prefix)
                qualified_pattern = rf"\.\s*{table}(?:\s|$|,|\))"
                fq_table_pattern = rf'fq_table\s*\(\s*["\']?{table}'

                if not re.search(qualified_pattern, line) and not re.search(
                    fq_table_pattern, line
                ):
                    # Additional check: line must have SQL indicators
                    # This avoids false positives in docstrings like "split into chunks"
                    if sql_indicators.search(line):
                        violations.append((line_num, table, stripped))

    return violations


class TestSQLSchemaSafety:
    """Ensure all SQL uses schema-qualified table names."""

    def test_no_unqualified_table_references(self):
        """All SQL queries must use fq_table() or schema.table format."""
        all_violations = []

        for py_file in get_python_files():
            content = py_file.read_text()
            violations = find_unqualified_table_refs(content, py_file.name)

            for line_num, table, line in violations:
                all_violations.append(
                    f"{py_file.relative_to(py_file.parent.parent)}:{line_num} - "
                    f"unqualified '{table}': {line[:80]}..."
                )

        if all_violations:
            msg = (
                f"Found {len(all_violations)} unqualified table references!\n"
                "These could cause cross-tenant data access.\n"
                "Use fq_table('table_name') for all table references.\n\n"
                + "\n".join(all_violations[:20])  # Show first 20
            )
            if len(all_violations) > 20:
                msg += f"\n... and {len(all_violations) - 20} more"
            pytest.fail(msg)

    def test_tables_list_is_complete(self):
        """Verify we're checking for all tables (sanity check)."""
        # This is a sanity check - if you add a new table, add it to TABLES
        assert len(TABLES) >= 9, "Update TABLES list if you added new tables"
