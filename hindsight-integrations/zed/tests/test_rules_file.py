"""Tests for the instruction-file memory-block writer."""

from pathlib import Path

from hindsight_zed.rules_file import (
    BEGIN_MARKER,
    END_MARKER,
    clear_memory_block,
    resolve_instruction_file,
    write_memory_block,
)


def test_resolve_prefers_existing_high_priority_file(tmp_path):
    # A project with AGENTS.md but no .rules: Zed reads AGENTS.md, so must we.
    (tmp_path / "AGENTS.md").write_text("# my agent rules\n")
    assert resolve_instruction_file(tmp_path).name == "AGENTS.md"


def test_resolve_dot_rules_wins_when_present(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x")
    (tmp_path / ".rules").write_text("y")
    assert resolve_instruction_file(tmp_path).name == ".rules"


def test_resolve_falls_back_to_dot_rules(tmp_path):
    # Empty project → we create .rules.
    assert resolve_instruction_file(tmp_path).name == ".rules"


def test_write_creates_dot_rules_when_no_instruction_file(tmp_path):
    target = write_memory_block(tmp_path, "- user likes pytest")
    assert target.name == ".rules"
    text = target.read_text()
    assert BEGIN_MARKER in text and END_MARKER in text
    assert "- user likes pytest" in text


def test_write_does_not_hijack_existing_agents_md(tmp_path):
    # The footgun the real Zed test surfaced: don't create .rules and suppress
    # the user's AGENTS.md — write into AGENTS.md instead, preserving it.
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Project rules\n\nAlways use tabs.\n")
    target = write_memory_block(tmp_path, "- user likes pytest")
    assert target.name == "AGENTS.md"
    assert not (tmp_path / ".rules").exists()
    text = agents.read_text()
    assert "Always use tabs." in text  # user content preserved
    assert "- user likes pytest" in text  # our block added
    assert text.index(BEGIN_MARKER) < text.index("Always use tabs.")  # memory leads


def test_rewrite_replaces_block_not_duplicate(tmp_path):
    write_memory_block(tmp_path, "- old memory")
    write_memory_block(tmp_path, "- new memory")
    text = (tmp_path / ".rules").read_text()
    assert text.count(BEGIN_MARKER) == 1  # exactly one block
    assert "- new memory" in text
    assert "- old memory" not in text


def test_rewrite_preserves_user_content_around_block(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Rules\n\nUse spaces.\n")
    write_memory_block(tmp_path, "- mem v1")
    write_memory_block(tmp_path, "- mem v2")
    text = agents.read_text()
    assert "Use spaces." in text
    assert "- mem v2" in text and "- mem v1" not in text
    assert text.count(BEGIN_MARKER) == 1


def test_clear_removes_block_keeps_user_content(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Rules\n\nUse spaces.\n")
    write_memory_block(tmp_path, "- mem")
    clear_memory_block(tmp_path)
    text = agents.read_text()
    assert BEGIN_MARKER not in text
    assert "Use spaces." in text


def test_clear_deletes_self_created_rules_file(tmp_path):
    # If we created .rules and it held only our block, removing leaves nothing →
    # delete the file rather than leave an empty one.
    write_memory_block(tmp_path, "- mem")
    assert (tmp_path / ".rules").exists()
    clear_memory_block(tmp_path)
    assert not (tmp_path / ".rules").exists()


def test_empty_memory_text_clears(tmp_path):
    write_memory_block(tmp_path, "- mem")
    write_memory_block(tmp_path, "   ")  # empty → clears
    assert not (tmp_path / ".rules").exists()


def test_preamble_included(tmp_path):
    write_memory_block(tmp_path, "- fact", preamble="Relevant memories:")
    text = (tmp_path / ".rules").read_text()
    assert "Relevant memories:" in text
    assert text.index("Relevant memories:") < text.index("- fact")


def test_malformed_block_recovered(tmp_path):
    # A begin marker with no end (file got truncated) must not corrupt rewrite.
    (tmp_path / ".rules").write_text(f"{BEGIN_MARKER}\n- half written")
    write_memory_block(tmp_path, "- clean")
    text = (tmp_path / ".rules").read_text()
    assert text.count(BEGIN_MARKER) == 1
    assert "- clean" in text and "- half written" not in text
