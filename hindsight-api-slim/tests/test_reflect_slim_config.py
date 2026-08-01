"""
Tests for the reflect_slim_tool_results configuration flag.

The slim LLM-facing rendering of reflect tool results (and the serialized-cost
entry budgeting that depends on it) is opt-in: default-off keeps the exact
pre-existing model-input behavior, and a bank or tenant can enable it via the
hierarchical config API.
"""

from hindsight_api.config import DEFAULT_REFLECT_SLIM_TOOL_RESULTS, HindsightConfig


class TestReflectSlimToolResultsConfig:
    def test_default_is_disabled(self):
        """The slim rendering must be opt-in: default-off preserves the exact
        model-input behavior existing deployments have today."""
        assert DEFAULT_REFLECT_SLIM_TOOL_RESULTS is False

    def test_config_is_configurable_per_bank(self):
        """The flag is hierarchical so a single bank can opt in (or out) without
        a server-wide change."""
        assert "reflect_slim_tool_results" in HindsightConfig.get_configurable_fields()
