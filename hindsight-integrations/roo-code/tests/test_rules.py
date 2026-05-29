"""Tests for the hindsight-memory.md rules file content."""

from pathlib import Path

RULES_FILE = Path(__file__).parent.parent / "rules" / "hindsight-memory.md"


def test_rules_file_exists() -> None:
    assert RULES_FILE.exists(), f"Rules file not found at {RULES_FILE}"


def test_rules_file_not_empty() -> None:
    content = RULES_FILE.read_text()
    assert len(content.strip()) > 0


def test_rules_references_recall_tool() -> None:
    content = RULES_FILE.read_text()
    assert "recall" in content


def test_rules_references_retain_tool() -> None:
    content = RULES_FILE.read_text()
    assert "retain" in content


def test_rules_instructs_recall_at_task_start() -> None:
    content = RULES_FILE.read_text().lower()
    # Must mention recalling at start of task
    assert "start" in content and "recall" in content


def test_rules_instructs_retain_at_task_end() -> None:
    content = RULES_FILE.read_text().lower()
    # Must mention retaining at end of task
    assert "end" in content and "retain" in content
