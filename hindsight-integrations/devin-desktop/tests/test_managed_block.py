"""Tests for the shared managed-block writer."""

from hindsight_devin_desktop.managed_block import BEGIN_MARKER, END_MARKER, clear, has, upsert


def test_creates_with_block(tmp_path):
    p = tmp_path / "AGENTS.md"
    action, size = upsert(p, "hello body")
    assert action == "created"
    text = p.read_text()
    assert BEGIN_MARKER in text and END_MARKER in text and "hello body" in text
    assert size == len(text)
    assert has(p)


def test_preserves_user_content_on_top(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("# My rules\n\nBe terse.\n")
    upsert(p, "hindsight body")
    text = p.read_text()
    assert text.startswith(BEGIN_MARKER)
    assert "# My rules" in text and "Be terse." in text


def test_idempotent(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("# Mine\n")
    upsert(p, "body")
    action, _ = upsert(p, "body")
    assert action == "unchanged"
    assert p.read_text().count(BEGIN_MARKER) == 1
    assert "# Mine" in p.read_text()


def test_update_changes_body(tmp_path):
    p = tmp_path / "AGENTS.md"
    upsert(p, "old")
    action, _ = upsert(p, "new body")
    assert action == "updated"
    assert "new body" in p.read_text() and "old" not in p.read_text()


def test_clear_keeps_user_content(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("# Mine\n\nkeep\n")
    upsert(p, "body")
    clear(p)
    assert BEGIN_MARKER not in p.read_text()
    assert "keep" in p.read_text()
    assert not has(p)


def test_clear_deletes_file_if_only_block(tmp_path):
    p = tmp_path / "AGENTS.md"
    upsert(p, "body")
    clear(p)
    assert not p.exists()
