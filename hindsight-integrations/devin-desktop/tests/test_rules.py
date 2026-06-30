"""Tests for the .devin/rules/hindsight.md rule writer."""

from hindsight_devin_desktop.rules import (
    RULE_TEXT,
    SENTINEL,
    clear_rule,
    default_rules_path,
    is_installed,
    render_rule,
    write_rule,
)


def test_default_path_is_devin_rules():
    p = default_rules_path()
    assert p.parent.name == "rules"
    assert p.parent.parent.name == ".devin"
    assert p.name == "hindsight.md"


def test_write_creates_dedicated_file(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path)
    text = path.read_text()
    assert SENTINEL in text and "recall" in text and "retain" in text
    assert is_installed(path)


def test_always_on_frontmatter(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path)
    text = path.read_text()
    # Frontmatter must lead the file and declare always-on activation.
    assert text.startswith("---\n")
    assert "trigger: always_on" in text.split("---", 2)[1]


def test_write_is_idempotent(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path)
    first = path.read_text()
    write_rule(path)
    assert path.read_text() == first
    assert path.read_text().count(SENTINEL) == 1


def test_clear_deletes_our_file(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path)
    clear_rule(path)
    assert not path.exists()


def test_clear_leaves_foreign_file(tmp_path):
    path = tmp_path / "hindsight.md"
    path.write_text("---\ntrigger: always_on\n---\n\nSomeone else's rule.\n")
    clear_rule(path)
    assert path.exists()  # no sentinel -> not ours -> untouched


def test_render_rule_mentions_all_tools():
    rendered = render_rule()
    for tool in ("recall", "retain", "reflect"):
        assert tool in RULE_TEXT and tool in rendered
