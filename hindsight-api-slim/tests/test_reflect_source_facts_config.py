"""
Tests for reflect search_observations source_facts_max_tokens configuration.

Verifies that the source_facts_max_tokens parameter correctly controls
whether source facts are included in search_observations recall calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.reflect.tools import tool_search_observations
from hindsight_api.engine.response_models import RecallResult
from hindsight_api.models import RequestContext


def _make_mock_engine(recall_result=None):
    """Create a mock memory engine with a recall_async method."""
    if recall_result is None:
        recall_result = RecallResult(results=[], source_facts={})
    engine = MagicMock()
    engine.recall_async = AsyncMock(return_value=recall_result)
    return engine


@pytest.fixture
def mock_request_context():
    # Use a real dataclass instance — tool_search_observations calls
    # dataclasses.replace(request_context, internal=True), which fails on
    # MagicMock. The fields don't matter for these tests; we only inspect
    # the kwargs passed to the mocked recall_async.
    return RequestContext()


class TestSearchObservationsSourceFacts:
    """Test source_facts_max_tokens parameter in tool_search_observations."""

    @pytest.mark.asyncio
    async def test_default_disables_source_facts(self, mock_request_context):
        """Default source_facts_max_tokens=-1 should disable source facts."""
        engine = _make_mock_engine()

        await tool_search_observations(
            engine, "bank-1", "test query", mock_request_context
        )

        engine.recall_async.assert_called_once()
        call_kwargs = engine.recall_async.call_args.kwargs
        assert call_kwargs["include_source_facts"] is False
        assert "max_source_facts_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_zero_enables_source_facts_unlimited(self, mock_request_context):
        """source_facts_max_tokens=0 should enable source facts with no token limit."""
        engine = _make_mock_engine()

        await tool_search_observations(
            engine, "bank-1", "test query", mock_request_context,
            source_facts_max_tokens=0,
        )

        engine.recall_async.assert_called_once()
        call_kwargs = engine.recall_async.call_args.kwargs
        assert call_kwargs["include_source_facts"] is True
        assert "max_source_facts_tokens" not in call_kwargs

    @pytest.mark.asyncio
    async def test_positive_enables_source_facts_with_limit(self, mock_request_context):
        """source_facts_max_tokens>0 should enable source facts with a token budget."""
        engine = _make_mock_engine()

        await tool_search_observations(
            engine, "bank-1", "test query", mock_request_context,
            source_facts_max_tokens=5000,
        )

        engine.recall_async.assert_called_once()
        call_kwargs = engine.recall_async.call_args.kwargs
        assert call_kwargs["include_source_facts"] is True
        assert call_kwargs["max_source_facts_tokens"] == 5000

    @pytest.mark.asyncio
    async def test_negative_one_disables_source_facts(self, mock_request_context):
        """Explicit -1 should disable source facts (same as default)."""
        engine = _make_mock_engine()

        await tool_search_observations(
            engine, "bank-1", "test query", mock_request_context,
            source_facts_max_tokens=-1,
        )

        engine.recall_async.assert_called_once()
        call_kwargs = engine.recall_async.call_args.kwargs
        assert call_kwargs["include_source_facts"] is False
        assert "max_source_facts_tokens" not in call_kwargs


class TestReflectSourceFactsConfig:
    """Test that reflect_source_facts_max_tokens is properly wired in HindsightConfig."""

    def test_config_field_exists(self):
        """reflect_source_facts_max_tokens should be a valid config field."""
        from hindsight_api.config import HindsightConfig

        import dataclasses

        field_names = {f.name for f in dataclasses.fields(HindsightConfig)}
        assert "reflect_source_facts_max_tokens" in field_names

    def test_config_is_configurable(self):
        """reflect_source_facts_max_tokens should be a configurable (per-bank) field."""
        from hindsight_api.config import HindsightConfig

        assert "reflect_source_facts_max_tokens" in HindsightConfig.get_configurable_fields()

    def test_default_value_is_disabled(self):
        """Default should be -1 (disabled)."""
        from hindsight_api.config import DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS

        assert DEFAULT_REFLECT_SOURCE_FACTS_MAX_TOKENS == -1

    def test_env_var_constant_exists(self):
        """Env var constant should be defined."""
        from hindsight_api.config import ENV_REFLECT_SOURCE_FACTS_MAX_TOKENS

        assert ENV_REFLECT_SOURCE_FACTS_MAX_TOKENS == "HINDSIGHT_API_REFLECT_SOURCE_FACTS_MAX_TOKENS"

    @patch.dict("os.environ", {"HINDSIGHT_API_REFLECT_SOURCE_FACTS_MAX_TOKENS": "8000"})
    def test_from_env_reads_value(self):
        """from_env should parse the env var."""
        from hindsight_api.config import HindsightConfig

        config = HindsightConfig.from_env()
        assert config.reflect_source_facts_max_tokens == 8000
