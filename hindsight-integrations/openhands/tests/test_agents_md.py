"""Tests for the AGENTS.md rule writer."""

from hindsight_openhands.agents_md import BEGIN_MARKER, RULE_TEXT, clear_rule, is_installed, write_rule


def test_write_creates_with_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    text = path.read_text()
    assert BEGIN_MARKER in text
    assert "recall" in text and "retain" in text
    assert is_installed(path)


def test_write_preserves_user_content_block_leads(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("# Repo Purpose\n\nThis is a TODO app.\n")
    write_rule(path)
    text = path.read_text()
    assert "This is a TODO app." in text
    assert text.index(BEGIN_MARKER) < text.index("This is a TODO app.")


def test_write_replaces_existing_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    write_rule(path)
    assert path.read_text().count(BEGIN_MARKER) == 1


def test_clear_keeps_user_content(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("Keep me.\n")
    write_rule(path)
    clear_rule(path)
    text = path.read_text()
    assert "Keep me." in text and BEGIN_MARKER not in text


def test_clear_deletes_if_only_our_block(tmp_path):
    path = tmp_path / "AGENTS.md"
    write_rule(path)
    clear_rule(path)
    assert not path.exists()


def test_rule_mentions_all_tools():
    for tool in ("recall", "retain", "reflect"):
        assert tool in RULE_TEXT
