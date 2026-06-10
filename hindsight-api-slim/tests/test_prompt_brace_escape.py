"""Tests for brace escaping in prompt builders across all modules.

User-supplied text (missions, custom instructions) may contain literal braces
(e.g. JSON examples). These must survive ``str.format()`` without crashing.
"""

import pytest

from hindsight_api.engine.prompt_utils import escape_for_prompt

# ---------------------------------------------------------------------------
# Unit tests for the shared escape helper
# ---------------------------------------------------------------------------


class TestEscapeForPrompt:
    def test_lone_braces_doubled(self):
        assert escape_for_prompt("{x}") == "{{x}}"

    def test_already_escaped_left_alone(self):
        assert escape_for_prompt("{{x}}") == "{{x}}"

    def test_idempotent(self):
        once = escape_for_prompt('{"dedup": true}')
        twice = escape_for_prompt(once)
        assert once == twice

    def test_plain_text_unchanged(self):
        assert escape_for_prompt("no braces here") == "no braces here"

    def test_mixed_lone_and_escaped(self):
        assert escape_for_prompt("{x} and {{y}}") == "{{x}} and {{y}}"

    @pytest.mark.parametrize(
        "text",
        ["{single}", "}}weird{{", "trailing {", "leading }", ""],
    )
    def test_edge_cases_do_not_crash(self, text):
        result = escape_for_prompt(text)
        # Double-escaped text must survive .format() with no placeholders
        result.format()


# ---------------------------------------------------------------------------
# Consolidation prompt
# ---------------------------------------------------------------------------


class TestConsolidationBraceSafety:
    def test_mission_with_json_renders(self):
        from hindsight_api.engine.consolidation.prompts import (
            build_batch_consolidation_prompt,
        )

        mission = '{"dedup": true, "merge": true}'
        prompt = build_batch_consolidation_prompt(observations_mission=mission)
        rendered = prompt.format(facts_text="<facts>", observations_text="<obs>")
        assert mission in rendered

    def test_capacity_note_with_braces_renders(self):
        from hindsight_api.engine.consolidation.prompts import (
            build_batch_consolidation_prompt,
        )

        note = "Use shape {limit, used}"
        prompt = build_batch_consolidation_prompt(observations_mission="m", observation_capacity_note=note)
        rendered = prompt.format(facts_text="<facts>", observations_text="<obs>")
        assert "{limit, used}" in rendered


# ---------------------------------------------------------------------------
# Reflect prompt
# ---------------------------------------------------------------------------


class TestReflectBraceSafety:
    def test_mission_with_json_renders(self):
        from hindsight_api.engine.reflect.prompts import build_final_system_prompt

        mission = '{"role": "admin", "scope": "all"}'
        prompt = build_final_system_prompt(mission=mission)
        # The prompt has no remaining format placeholders, so it should be
        # a plain string that contains the original mission text.
        assert mission in prompt

    def test_mission_with_lone_braces_renders(self):
        from hindsight_api.engine.reflect.prompts import build_final_system_prompt

        mission = "Track {entity} changes"
        prompt = build_final_system_prompt(mission=mission)
        assert "{entity}" in prompt


# ---------------------------------------------------------------------------
# Retain / fact extraction prompt
# ---------------------------------------------------------------------------


class TestRetainBraceSafety:
    def _make_config(self, **overrides):
        """Minimal config-like object for _build_extraction_prompt_and_schema."""
        from types import SimpleNamespace

        defaults = {
            "retain_extraction_mode": "concise",
            "retain_extract_causal_links": False,
            "retain_mission": None,
            "retain_custom_instructions": None,
            "retain_taxonomy": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_retain_mission_with_json(self):
        from hindsight_api.engine.retain.fact_extraction import (
            _build_extraction_prompt_and_schema,
            _retain_mission_preamble,
        )

        config = self._make_config(retain_mission='{"focus": "compliance"}')
        prompt, _ = _build_extraction_prompt_and_schema(config)
        # The mission no longer lives in the (cached, bank-agnostic) system prompt;
        # it rides in the per-request user-message preamble, verbatim and unescaped
        # (the preamble is not passed through str.format(), so braces are safe).
        assert '{"focus": "compliance"}' not in prompt
        assert '{"focus": "compliance"}' in _retain_mission_preamble(config)

    def test_custom_instructions_with_braces(self):
        from hindsight_api.engine.retain.fact_extraction import (
            _build_extraction_prompt_and_schema,
        )

        config = self._make_config(
            retain_extraction_mode="custom",
            retain_custom_instructions="Output as {key: value} pairs",
        )
        prompt, _ = _build_extraction_prompt_and_schema(config)
        assert "{key: value}" in prompt

    def test_both_mission_and_custom_with_braces(self):
        from hindsight_api.engine.retain.fact_extraction import (
            _build_extraction_prompt_and_schema,
            _retain_mission_preamble,
        )

        config = self._make_config(
            retain_extraction_mode="custom",
            retain_mission='{"scope": "all"}',
            retain_custom_instructions="Format: {k: v}",
        )
        prompt, _ = _build_extraction_prompt_and_schema(config)
        # Mission → user-message preamble; custom instructions stay in the system prompt.
        assert '{"scope": "all"}' in _retain_mission_preamble(config)
        assert "{k: v}" in prompt
