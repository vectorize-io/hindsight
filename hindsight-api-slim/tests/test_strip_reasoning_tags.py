"""Tests for _strip_reasoning_tags helper in OpenAI-compatible LLM provider."""

from hindsight_api.engine.providers.openai_compatible_llm import _strip_reasoning_tags


class TestStripReasoningTags:
    """Test reasoning/thinking tag stripping from LLM responses."""

    def test_plain_text_unchanged(self):
        """Text without reasoning tags passes through (modulo edge whitespace)."""
        content = "User prefers functional programming patterns."
        assert _strip_reasoning_tags(content) == content

    def test_empty_string(self):
        """Empty string passes through."""
        assert _strip_reasoning_tags("") == ""

    def test_closed_think_stripped(self):
        """A closed <think>...</think> block is removed."""
        content = "<think>let me reason</think>The answer is 42."
        assert _strip_reasoning_tags(content) == "The answer is 42."

    def test_closed_thinking_stripped(self):
        assert _strip_reasoning_tags("<thinking>reasoning</thinking>Result") == "Result"

    def test_closed_thought_stripped(self):
        assert _strip_reasoning_tags("<thought>hmm</thought>Result") == "Result"

    def test_closed_reasoning_stripped(self):
        assert _strip_reasoning_tags("<reasoning>step by step</reasoning>Result") == "Result"

    def test_startthink_endthink_stripped(self):
        """The |startthink|...|endthink| marker style is removed."""
        content = "|startthink|internal monologue|endthink|Final output"
        assert _strip_reasoning_tags(content) == "Final output"

    def test_multiline_think_stripped(self):
        """DOTALL: a multi-line thinking block is fully removed."""
        content = "<think>\nline one\nline two\n</think>\nThe real content."
        assert _strip_reasoning_tags(content) == "The real content."

    def test_unclosed_think_stripped_to_end(self):
        """An unclosed <think> (truncated output) is removed to end-of-string."""
        content = "Partial answer.\n<think>I started thinking but got cut off"
        assert _strip_reasoning_tags(content) == "Partial answer."

    def test_unclosed_thinking_stripped_to_end(self):
        content = "result text\n<thinking>dangling reasoning with no close"
        assert _strip_reasoning_tags(content) == "result text"

    def test_only_unclosed_think_becomes_empty(self):
        """Content that is entirely an unclosed thinking block collapses to empty."""
        content = "<think>everything is reasoning and it never closed"
        assert _strip_reasoning_tags(content) == ""

    def test_multiple_blocks_stripped(self):
        """Multiple closed blocks are all removed."""
        content = "<think>a</think>Hello <think>b</think>World"
        assert _strip_reasoning_tags(content) == "Hello World"

    def test_mental_model_markdown_contamination(self):
        """Real-world MiniMax-M3 free-form leak: <think> wrapping a markdown mental model."""
        content = (
            "<think>\n"
            "The user keeps asking about FP. I should consolidate this.\n"
            "</think>\n"
            "# Mental Model: Coding Preferences\n\n"
            "The user prefers functional programming patterns and immutable data."
        )
        result = _strip_reasoning_tags(content)
        assert "<think>" not in result
        assert "</think>" not in result
        assert result.startswith("# Mental Model: Coding Preferences")

    def test_unclosed_think_after_json_payload(self):
        """Truncated <think> trailing valid JSON is stripped (closing tag absent)."""
        content = '{"facts": [{"what": "test"}]}\n<think>oops truncated'
        assert _strip_reasoning_tags(content) == '{"facts": [{"what": "test"}]}'

    def test_inline_think_literal_in_json_kept(self):
        """An inline <think> literal inside a JSON value is real content — must be kept.

        Regression for #2195: the old greedy ``.*`` to end-of-string deleted the
        inline tag plus all following JSON, surfacing as ``Unterminated string``
        when the retained conversation itself quoted ``<think>``.
        """
        content = '{"facts": [{"what": "analysis of <think> tag behavior"}]}'
        assert _strip_reasoning_tags(content) == content

    def test_inline_think_literal_mid_sentence_kept(self):
        """Inline unclosed <think> mid-sentence is a literal, not a block."""
        content = '{"facts": [{"what": "a <think discussion continues"}]}'
        assert _strip_reasoning_tags(content) == content

    def test_indented_think_line_stripped(self):
        """A line-start (indented) unclosed <think> block is stripped to end-of-string.

        The tag is unclosed because the output was truncated, so everything from
        the tag onward is reasoning to discard — not an answer to keep.
        """
        content = "  <think>dangling reasoning no close\nmore dangling reasoning"
        assert _strip_reasoning_tags(content) == ""

    def test_inline_multiple_think_literals_kept(self):
        """Multiple inline think-style literals in a sentence are all kept."""
        content = "analysis of <think> and <thinking> tag differences"
        assert _strip_reasoning_tags(content) == content

    def test_unclosed_multiline_think_stripped_to_end(self):
        """A line-start unclosed <think> that spans multiple lines is stripped whole.

        Truncated reasoning is usually multi-line (the close tag never arrived
        because output hit max_tokens). Stripping only the first line would leak
        the remaining reasoning into stored memory — the exact free-form
        contamination this helper exists to prevent.
        """
        content = (
            "# Mental Model: Coding Preferences\n"
            "The user prefers functional programming.\n"
            "<think>\n"
            "leaked reasoning line one\n"
            "leaked reasoning line two"
        )
        result = _strip_reasoning_tags(content)
        assert result == ("# Mental Model: Coding Preferences\nThe user prefers functional programming.")
        assert "leaked reasoning" not in result
