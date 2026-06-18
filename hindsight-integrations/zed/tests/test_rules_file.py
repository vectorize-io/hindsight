"""Tests for the global-rules writer."""

from hindsight_zed.rules_file import (
    BEGIN_MARKER,
    END_MARKER,
    RULE_TEXT,
    clear_rule,
    is_installed,
    write_rule,
)


def test_write_creates_file_with_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    text = path.read_text()
    assert BEGIN_MARKER in text and END_MARKER in text
    assert "recall" in text and "retain" in text
    assert is_installed(path)


def test_write_preserves_user_content_and_leads_with_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("# My project rules\n\nAlways use tabs.\n")
    write_rule(path)
    text = path.read_text()
    assert "Always use tabs." in text  # preserved
    assert text.index(BEGIN_MARKER) < text.index("Always use tabs.")  # block leads


def test_write_replaces_existing_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    write_rule(path)
    text = path.read_text()
    assert text.count(BEGIN_MARKER) == 1  # not duplicated


def test_clear_removes_block_but_keeps_user_content(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("Keep me.\n")
    write_rule(path)
    clear_rule(path)
    text = path.read_text()
    assert "Keep me." in text
    assert BEGIN_MARKER not in text


def test_clear_deletes_file_if_only_our_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    clear_rule(path)
    assert not path.exists()


def test_clear_noop_when_absent(tmp_path):
    path = tmp_path / "AGENTS.md"
    clear_rule(path)  # should not raise
    assert not path.exists()


def test_rule_text_mentions_all_three_tools():
    for tool in ("recall", "retain", "reflect"):
        assert tool in RULE_TEXT
