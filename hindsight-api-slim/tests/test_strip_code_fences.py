"""Tests for _strip_code_fences helper in OpenAI-compatible LLM provider."""

import json

from hindsight_api.engine.providers.openai_compatible_llm import _strip_code_fences


class TestStripCodeFences:
    """Test markdown code fence stripping from LLM responses."""

    def test_bare_json_unchanged(self):
        """Bare JSON passes through unchanged."""
        content = '{"facts": [{"what": "test"}]}'
        assert _strip_code_fences(content) == content

    def test_json_fence_stripped(self):
        """```json ... ``` fences are stripped."""
        content = '```json\n{"facts": [{"what": "test"}]}\n```'
        assert _strip_code_fences(content) == '{"facts": [{"what": "test"}]}'

    def test_plain_fence_stripped(self):
        """``` ... ``` fences without language tag are stripped."""
        content = '```\n{"facts": [{"what": "test"}]}\n```'
        assert _strip_code_fences(content) == '{"facts": [{"what": "test"}]}'

    def test_fence_with_trailing_whitespace(self):
        """Fences with extra whitespace are handled."""
        content = '```json\n{"facts": []}\n```\n'
        result = _strip_code_fences(content)
        assert result == '{"facts": []}'

    def test_fence_with_leading_whitespace(self):
        """Content with leading whitespace before fence."""
        content = '  ```json\n{"facts": []}\n```'
        # The function checks for ``` in content, not startswith
        result = _strip_code_fences(content)
        assert '{"facts": []}' in result

    def test_inner_backticks_preserved(self):
        """Inner triple-backticks inside a JSON string value must not truncate the JSON.

        Regression for the fact-extraction case where an extracted fact describes
        code-fence behavior, so the JSON payload itself contains a literal
        ```` ```json ```` — the old split-based stripper matched that inner
        occurrence and cut the JSON mid-string.
        """
        import json

        content = '```json\n{"facts": [{"what": "the model wraps output in ```json fences"}]}\n```'
        result = _strip_code_fences(content)
        assert result == '{"facts": [{"what": "the model wraps output in ```json fences"}]}'
        parsed = json.loads(result)
        assert parsed["facts"][0]["what"] == "the model wraps output in ```json fences"

    def test_no_fences_no_change(self):
        """Content without any backticks passes through."""
        content = "Just some text without fences"
        assert _strip_code_fences(content) == content

    def test_empty_string(self):
        """Empty string passes through."""
        assert _strip_code_fences("") == ""

    def test_multiline_json(self):
        """Multi-line JSON inside fences is preserved."""
        content = '```json\n{\n  "facts": [\n    {"what": "line1"},\n    {"what": "line2"}\n  ]\n}\n```'
        result = _strip_code_fences(content)
        assert '"line1"' in result
        assert '"line2"' in result
        assert "```" not in result

    def test_missing_closing_fence_recovers_json(self):
        """A fence with no closing ``` still recovers the JSON via the outer-span fallback."""
        content = '```json\n{"facts": []}'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"facts": []}

    def test_prose_wrapped_json_recovered(self):
        """JSON surrounded by prose (no usable fence) is recovered by the fallback."""
        content = 'Sure! Here is the result:\n{"facts": [{"what": "x"}]}\nLet me know if that helps.'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"facts": [{"what": "x"}]}

    def test_non_json_fence_left_for_retry(self):
        """A fenced block that is not JSON yields no valid candidate; content is returned unchanged."""
        content = "```\nnot json at all\n```"
        result = _strip_code_fences(content)
        # No parseable JSON anywhere -> caller sees the stripped body (still a str), never crashes.
        assert isinstance(result, str)
        assert "not json at all" in result

    def test_minimax_style_response(self):
        """Real-world MiniMax response format."""
        content = (
            "```json\n"
            "{\n"
            '  "facts": [\n'
            "    {\n"
            '      "what": "Sebastian switched the Hindsight extraction LLM",\n'
            '      "when": "2026-03-21",\n'
            '      "where": "N/A",\n'
            '      "who": "Sebastian",\n'
            '      "why": "MiniMax wraps JSON in code fences",\n'
            '      "fact_kind": "event",\n'
            '      "fact_type": "world",\n'
            '      "entities": [{"text": "Sebastian"}, {"text": "Hindsight"}],\n'
            '      "labels": {"source_type": "stated", "domain": ["infrastructure"]}\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```"
        )
        result = _strip_code_fences(content)
        assert not result.startswith("```")
        assert not result.endswith("```")
        # Should be valid JSON
        import json

        parsed = json.loads(result)
        assert len(parsed["facts"]) == 1
        assert parsed["facts"][0]["who"] == "Sebastian"

    def test_json_tagged_block_preferred_over_earlier_untagged_block(self):
        """Multiple fences: a ```json-tagged block wins over an earlier untagged one.

        Regression for #4817: a model/gateway that ignores a forced tool_choice
        and replies with free text may echo an unrelated example before its real
        JSON answer (e.g. a ```python snippet, then the ```json answer). The
        first-fence-by-position behavior this replaces would silently return the
        wrong block whenever it happens to also be valid JSON/parseable text.
        """
        content = (
            "Here's an example of the format:\n"
            "```python\n"
            'foo = {"a": 1}\n'
            "```\n"
            "And the real answer:\n"
            "```json\n"
            '{"real": "answer"}\n'
            "```"
        )
        result = _strip_code_fences(content)
        assert json.loads(result) == {"real": "answer"}

    def test_json_tagged_block_preferred_even_when_it_comes_second(self):
        """Two fences, neither obviously an 'example' — json tag alone decides."""
        content = '```text\nnot the answer\n```\n```json\n{"ok": true}\n```'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"ok": True}

    def test_first_json_tagged_block_wins_when_multiple_are_tagged(self):
        """Two ```json blocks: the first one (in document order) is used."""
        content = '```json\n{"first": 1}\n```\n```json\n{"second": 2}\n```'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"first": 1}

    def test_no_json_tag_falls_back_to_first_fence(self):
        """No block is tagged json: behavior matches the pre-multi-block code (first fence)."""
        content = '```text\n{"first": 1}\n```\n```text\n{"second": 2}\n```'
        result = _strip_code_fences(content)
        assert json.loads(result) == {"first": 1}
