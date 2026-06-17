"""Tests for the hindsight-memory.md rules file content."""

from hindsight_roo_code import rules_text


def test_rules_text_not_empty() -> None:
    assert len(rules_text().strip()) > 0


def test_rules_references_recall_tool() -> None:
    assert "recall" in rules_text()


def test_rules_references_retain_tool() -> None:
    assert "retain" in rules_text()


def test_rules_instructs_recall_at_task_start() -> None:
    content = rules_text().lower()
    # Must mention recalling at start of task
    assert "start" in content and "recall" in content


def test_rules_instructs_retain_at_task_end() -> None:
    content = rules_text().lower()
    # Must mention retaining at end of task
    assert "end" in content and "retain" in content
