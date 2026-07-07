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
