"""Tests for the Cursor sessionStart rules-file workaround.

The workaround module writes a workspace .cursor/rules/hindsight-session.mdc
file because Cursor's native sessionStart additionalContext path is broken.
These tests pin the on-disk shape (frontmatter, rotation behaviour, gitignore
handling) so the bug fix is mechanical when Cursor restores the native path.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from scripts.lib.rules_file import (
    GITIGNORE_RELPATH,
    RULES_FILE_RELPATH,
    ensure_gitignored,
    format_rule_content,
    rotate_session_rules,
    write_session_rules,
)


class TestFormatRuleContent:
    def test_includes_frontmatter_with_alwaysApply(self):
        out = format_rule_content("body", "preamble", "2026-06-02 12:00 UTC")
        # alwaysApply: true is what makes Cursor's rules engine inject this
        # file at every turn — pin it exactly.
        assert out.startswith("---\n")
        assert "alwaysApply: true\n" in out
        # frontmatter terminates before the body
        assert "---\n\n<!--" in out

    def test_includes_bug_link_so_future_readers_understand_purpose(self):
        out = format_rule_content("body", "preamble", "now")
        assert "forum.cursor.com" in out
        assert "158452" in out  # the staff-acknowledged thread

    def test_wraps_memories_in_hindsight_memories_block(self):
        out = format_rule_content("MEMORY_TEXT", "PREAMBLE_TEXT", "T")
        assert "<hindsight_memories>" in out
        assert "</hindsight_memories>" in out
        assert "PREAMBLE_TEXT" in out
        assert "MEMORY_TEXT" in out
        # The pre/post-amble appear inside, not duplicated outside
        before, _, after = out.partition("<hindsight_memories>")
        assert "MEMORY_TEXT" not in before
        assert "MEMORY_TEXT" in after


class TestRotateSessionRules:
    def test_removes_existing_file(self, tmp_path):
        target = tmp_path / RULES_FILE_RELPATH
        target.parent.mkdir(parents=True)
        target.write_text("stale")
        rotate_session_rules(str(tmp_path))
        assert not target.exists()

    def test_no_op_when_file_missing(self, tmp_path):
        # Must not raise — stale-file delete is best-effort
        rotate_session_rules(str(tmp_path))
        # nothing to assert; passing without exception is success

    def test_no_op_when_workspace_root_is_falsy(self):
        rotate_session_rules("")
        rotate_session_rules(None)

    def test_logs_through_debug_fn(self, tmp_path):
        target = tmp_path / RULES_FILE_RELPATH
        target.parent.mkdir(parents=True)
        target.write_text("stale")
        seen: list[str] = []
        rotate_session_rules(str(tmp_path), debug_fn=seen.append)
        assert any("Removed stale" in m for m in seen)


class TestWriteSessionRules:
    def test_creates_parent_dirs_and_writes_content(self, tmp_path):
        wrote = write_session_rules(str(tmp_path), "RULE_BODY")
        assert wrote is True
        path = tmp_path / RULES_FILE_RELPATH
        assert path.exists()
        assert path.read_text() == "RULE_BODY"

    def test_overwrites_existing_file(self, tmp_path):
        path = tmp_path / RULES_FILE_RELPATH
        path.parent.mkdir(parents=True)
        path.write_text("OLD")
        write_session_rules(str(tmp_path), "NEW")
        assert path.read_text() == "NEW"

    def test_returns_false_when_workspace_root_is_falsy(self):
        assert write_session_rules("", "x") is False
        assert write_session_rules(None, "x") is False

    def test_returns_false_and_logs_on_write_error(self, tmp_path):
        seen: list[str] = []
        # Force makedirs to raise — simulates a read-only workspace
        with mock.patch("os.makedirs", side_effect=OSError("read-only")):
            wrote = write_session_rules(str(tmp_path), "BODY", debug_fn=seen.append)
        assert wrote is False
        assert any("Could not write" in m for m in seen)


class TestEnsureGitignored:
    def _init_git_dir(self, root: Path) -> None:
        """Minimal '.git' marker — ensure_gitignored only checks existence."""
        (root / ".git").mkdir()

    def test_skips_when_not_a_git_repo(self, tmp_path):
        # No .git → no need to gitignore against anything
        appended = ensure_gitignored(str(tmp_path))
        assert appended is False
        assert not (tmp_path / GITIGNORE_RELPATH).exists()

    def test_appends_when_file_does_not_exist(self, tmp_path):
        self._init_git_dir(tmp_path)
        appended = ensure_gitignored(str(tmp_path))
        assert appended is True
        text = (tmp_path / GITIGNORE_RELPATH).read_text()
        assert "/" + RULES_FILE_RELPATH in text
        # Includes a human-readable explanation
        assert "hindsight-cursor" in text

    def test_appends_when_existing_gitignore_lacks_entry(self, tmp_path):
        self._init_git_dir(tmp_path)
        gitignore = tmp_path / GITIGNORE_RELPATH
        gitignore.write_text("node_modules/\n.env\n")
        appended = ensure_gitignored(str(tmp_path))
        assert appended is True
        text = gitignore.read_text()
        # Existing content is preserved
        assert "node_modules/" in text
        assert ".env" in text
        # Plus our pattern
        assert "/" + RULES_FILE_RELPATH in text

    def test_idempotent_when_entry_already_present(self, tmp_path):
        self._init_git_dir(tmp_path)
        gitignore = tmp_path / GITIGNORE_RELPATH
        gitignore.write_text("/" + RULES_FILE_RELPATH + "\n")
        before = gitignore.read_text()
        appended = ensure_gitignored(str(tmp_path))
        assert appended is False
        assert gitignore.read_text() == before

    def test_idempotent_when_bare_relative_form_is_present(self, tmp_path):
        # Some users write `.cursor/rules/hindsight-session.mdc` without the
        # leading slash; that's still a valid gitignore match — don't dupe.
        self._init_git_dir(tmp_path)
        gitignore = tmp_path / GITIGNORE_RELPATH
        gitignore.write_text(RULES_FILE_RELPATH + "\n")
        appended = ensure_gitignored(str(tmp_path))
        assert appended is False

    def test_handles_existing_file_without_trailing_newline(self, tmp_path):
        self._init_git_dir(tmp_path)
        gitignore = tmp_path / GITIGNORE_RELPATH
        gitignore.write_text(".env")  # no trailing newline
        ensure_gitignored(str(tmp_path))
        text = gitignore.read_text()
        # Pattern lives on its own line, not glued to the prior entry
        assert ".env/" not in text  # didn't get smashed together
        assert "/" + RULES_FILE_RELPATH in text

    def test_returns_false_when_workspace_root_is_falsy(self):
        assert ensure_gitignored("") is False
        assert ensure_gitignored(None) is False
