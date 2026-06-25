"""Tests for the copilot-instructions.md rule writer."""

from hindsight_copilot.instructions import BEGIN_MARKER, RULE_TEXT, clear_rule, is_installed, write_rule


def test_write_creates_with_block(tmp_path):
    path = tmp_path / "copilot-instructions.md"
    write_rule(path)
    text = path.read_text()
    assert BEGIN_MARKER in text and "recall" in text and "retain" in text
    assert is_installed(path)


def test_write_preserves_user_content_block_leads(tmp_path):
    path = tmp_path / "copilot-instructions.md"
    path.write_text("# Project\n\nUse TypeScript.\n")
    write_rule(path)
    text = path.read_text()
    assert "Use TypeScript." in text
    assert text.index(BEGIN_MARKER) < text.index("Use TypeScript.")


def test_write_replaces_existing_block(tmp_path):
    path = tmp_path / "copilot-instructions.md"
    write_rule(path)
    write_rule(path)
    assert path.read_text().count(BEGIN_MARKER) == 1


def test_clear_keeps_user_content(tmp_path):
    path = tmp_path / "copilot-instructions.md"
    path.write_text("Keep me.\n")
    write_rule(path)
    clear_rule(path)
    assert "Keep me." in path.read_text() and BEGIN_MARKER not in path.read_text()


def test_clear_deletes_if_only_block(tmp_path):
    path = tmp_path / "copilot-instructions.md"
    write_rule(path)
    clear_rule(path)
    assert not path.exists()


def test_rule_mentions_all_tools():
    for tool in ("recall", "retain", "reflect"):
        assert tool in RULE_TEXT
