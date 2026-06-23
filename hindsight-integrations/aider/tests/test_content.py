"""Tests for query/memory/transcript formatting."""

from types import SimpleNamespace

from hindsight_aider.content import compose_recall_query, format_memory, format_transcript


class TestComposeQuery:
    def test_uses_message_flag(self):
        assert compose_recall_query(["-m", "fix auth", "f.py"], "default") == "fix auth"

    def test_uses_long_message_flag(self):
        assert compose_recall_query(["--message", "add tests"], "default") == "add tests"

    def test_uses_message_equals(self):
        assert compose_recall_query(["--message=refactor db"], "default") == "refactor db"

    def test_falls_back_to_default(self):
        assert compose_recall_query(["src/app.py"], "default query") == "default query"

    def test_truncates(self):
        q = compose_recall_query(["-m", "x" * 5000], "d", max_chars=100)
        assert len(q) == 100


class TestFormatMemory:
    def test_renders_results(self):
        results = [SimpleNamespace(text="a fact"), SimpleNamespace(text="another")]
        out = format_memory(results, "Preamble:")
        assert "Preamble:" in out and "- a fact" in out and "- another" in out

    def test_empty_returns_empty(self):
        assert format_memory([], "P") == ""
        assert format_memory([SimpleNamespace(text="")], "P") == ""


class TestFormatTranscript:
    def test_trims_to_last_chars(self):
        out = format_transcript("x" * 100 + "TAIL", max_chars=4)
        assert out == "TAIL"

    def test_strips_whitespace(self):
        assert format_transcript("  hi  \n") == "hi"
