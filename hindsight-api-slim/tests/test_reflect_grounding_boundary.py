"""Reflect must not manufacture a value for something its data does not cover.

Reflect is told throughout to infer rather than repeat literally, which is what
makes it useful. But "if the exact answer isn't stated, use what IS stated" had
no floor. Asked for a headcount in a year the bank did not cover, the model
extrapolated backwards from the following year's growth and reported a specific
number as "reliably deduced" — a fabricated data point wearing the language of
certainty, which is worse than "not recorded" because a reader cannot tell the
difference.

Split per the testing convention: the prompt wiring is deterministic and asserted
directly here; whether a model actually FOLLOWS the rule is not, so it gets one
judge test against a real LLM.
"""

import pytest

from hindsight_api.engine.reflect.prompts import (
    _FINAL_INSTRUCTIONS,
    _GROUNDING_BOUNDARY,
    build_final_system_prompt,
    build_system_prompt_for_tools,
)


class TestGroundingBoundaryWiring:
    """Deterministic: every path that writes an answer carries the rule."""

    def test_tool_loop_system_prompt_carries_it(self):
        prompt = build_system_prompt_for_tools({"name": "Bank", "mission": "testing"})
        assert _GROUNDING_BOUNDARY in prompt

    def test_forced_synthesis_system_prompt_carries_it(self):
        assert _GROUNDING_BOUNDARY in build_final_system_prompt("testing", None, None)

    def test_final_instructions_carry_it(self):
        # Reached on the split/reduce synthesis path, which uses the instructions
        # rather than the system prompt.
        assert _GROUNDING_BOUNDARY in _FINAL_INSTRUCTIONS

    @pytest.mark.parametrize(
        "phrasing",
        ["Extrapolating", "does not record it", "never describe one as"],
    )
    def test_rule_states_the_actual_constraint(self, phrasing: str):
        """Guards the substance, not just that some text is present.

        A rule that survives as a heading while its teeth are edited out would
        otherwise keep every wiring test above green.
        """
        assert phrasing in _GROUNDING_BOUNDARY

    def test_qualitative_inference_is_explicitly_preserved(self):
        """The rule must not read as "never infer" — that would break synthesis.

        Reflect's value is connecting memories. The boundary is about
        manufacturing values, so it says so out loud; without this carve-out a
        model reasonably reads the section as a blanket prohibition.
        """
        assert "Qualitative inference is unaffected" in _GROUNDING_BOUNDARY
