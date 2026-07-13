"""Tests for the .devin/rules/hindsight.md rule writer."""

from hindsight_devin_desktop.rules import (
    SENTINEL,
    clear_rule,
    default_rules_path,
    is_installed,
    render_rule,
    render_rule_text,
    write_rule,
)

PROJECT = "devin-desktop-acme-web"
GLOBAL = "devin-desktop"


def test_default_path_is_devin_rules():
    p = default_rules_path()
    assert p.parent.name == "rules"
    assert p.parent.parent.name == ".devin"
    assert p.name == "hindsight.md"


def test_write_creates_dedicated_file(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    text = path.read_text()
    assert SENTINEL in text and "recall" in text and "retain" in text
    assert is_installed(path)


def test_rule_names_both_banks(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    text = path.read_text()
    assert PROJECT in text and GLOBAL in text
    # explicit bank_id routing guidance
    assert f'bank_id: "{PROJECT}"' in text or f'bank_id "{PROJECT}"' in text


def test_rule_reverses_dont_mention_to_mention(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    text = path.read_text().lower()
    assert "do not mention" not in text
    assert "tell the user when you use memory" in text


def test_rule_mentions_new_tools():
    text = render_rule_text(PROJECT, GLOBAL)
    for tool in ("recall", "retain", "sync_retain", "reflect"):
        assert tool in text


def test_always_on_frontmatter(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    text = path.read_text()
    assert text.startswith("---\n")
    assert "trigger: always_on" in text.split("---", 2)[1]


def test_write_is_idempotent(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    first = path.read_text()
    write_rule(path, PROJECT, GLOBAL)
    assert path.read_text() == first
    assert path.read_text().count(SENTINEL) == 1


def test_clear_deletes_our_file(tmp_path):
    path = tmp_path / "hindsight.md"
    write_rule(path, PROJECT, GLOBAL)
    clear_rule(path)
    assert not path.exists()


def test_clear_leaves_foreign_file(tmp_path):
    path = tmp_path / "hindsight.md"
    path.write_text("---\ntrigger: always_on\n---\n\nSomeone else's rule.\n")
    clear_rule(path)
    assert path.exists()  # no sentinel -> not ours -> untouched


def test_render_rule_wraps_body_in_frontmatter():
    rendered = render_rule(PROJECT, GLOBAL)
    assert rendered.startswith("---\n")
    assert SENTINEL in rendered
    assert PROJECT in rendered
