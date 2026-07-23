"""Tests for the global_rules.md managed-block writer."""

from hindsight_devin_desktop.global_rules import (
    BEGIN_MARKER,
    END_MARKER,
    clear_global_rule,
    default_global_rules_path,
    is_installed,
    write_global_rule,
)

GLOBAL = "devin-desktop"


def test_default_path():
    p = default_global_rules_path()
    assert p.name == "global_rules.md"
    assert p.parent.name == "memories"
    assert p.parent.parent.name == "windsurf"


def test_creates_with_managed_block(tmp_path):
    path = tmp_path / "global_rules.md"
    result = write_global_rule(path, GLOBAL)
    assert result.action == "created"
    text = path.read_text()
    assert BEGIN_MARKER in text and END_MARKER in text
    assert GLOBAL in text
    assert is_installed(path)


def test_preserves_user_content(tmp_path):
    path = tmp_path / "global_rules.md"
    path.write_text("# My global rules\n\nAlways use tabs.\n")
    write_global_rule(path, GLOBAL)
    text = path.read_text()
    assert text.startswith(BEGIN_MARKER)  # our block leads
    assert "My global rules" in text and "Always use tabs." in text


def test_idempotent_single_block(tmp_path):
    path = tmp_path / "global_rules.md"
    path.write_text("# Mine\n")
    write_global_rule(path, GLOBAL)
    write_global_rule(path, GLOBAL)
    text = path.read_text()
    assert text.count(BEGIN_MARKER) == 1
    assert "# Mine" in text


def test_second_write_is_unchanged(tmp_path):
    path = tmp_path / "global_rules.md"
    write_global_rule(path, GLOBAL)
    assert write_global_rule(path, GLOBAL).action == "unchanged"


def test_over_cap_reported(tmp_path):
    path = tmp_path / "global_rules.md"
    path.write_text("x" * 6000)
    result = write_global_rule(path, GLOBAL)
    assert result.over_cap is not None and result.over_cap > 6000


def test_clear_removes_block_keeps_user_content(tmp_path):
    path = tmp_path / "global_rules.md"
    path.write_text("# Mine\n\nkeep me\n")
    write_global_rule(path, GLOBAL)
    clear_global_rule(path)
    text = path.read_text()
    assert BEGIN_MARKER not in text
    assert "keep me" in text
    assert not is_installed(path)


def test_clear_deletes_file_if_only_our_block(tmp_path):
    path = tmp_path / "global_rules.md"
    write_global_rule(path, GLOBAL)
    clear_global_rule(path)
    assert not path.exists()
