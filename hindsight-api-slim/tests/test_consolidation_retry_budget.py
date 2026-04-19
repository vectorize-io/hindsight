"""Tests for consolidation retry budget configurability (issue #1042)."""

import pytest

from unittest.mock import AsyncMock, MagicMock

from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm


@pytest.fixture
def mock_llm_config():
    llm = AsyncMock()
    response = MagicMock()
    response.creates = []
    response.updates = []
    response.deletes = []
    llm.call.return_value = response
    return llm


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.observations_mission = None
    config.consolidation_max_attempts = 3
    config.consolidation_llm_max_retries = None
    return config


class TestConsolidationRetryBudget:
    @pytest.mark.asyncio
    async def test_config_is_required(self, mock_llm_config):
        """Passing config=None raises — it's a programmer error, not a runtime fallback."""
        with pytest.raises(ValueError, match="config is required"):
            await _consolidate_batch_with_llm(
                llm_config=mock_llm_config,
                memories=[{"id": "m1", "text": "test"}],
                union_observations=[],
                union_source_facts={},
                config=None,
            )

    @pytest.mark.asyncio
    async def test_configurable_max_attempts(self, mock_llm_config, mock_config):
        """consolidation_max_attempts controls the outer retry loop."""
        mock_config.consolidation_max_attempts = 5
        mock_llm_config.call.side_effect = RuntimeError("fail")
        result = await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert result.failed
        assert mock_llm_config.call.call_count == 5

    @pytest.mark.asyncio
    async def test_max_retries_threaded_to_call(self, mock_llm_config, mock_config):
        """consolidation_llm_max_retries is passed to llm_config.call()."""
        mock_config.consolidation_llm_max_retries = 3
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("max_retries") == 3

    @pytest.mark.asyncio
    async def test_max_retries_not_passed_when_none(self, mock_llm_config, mock_config):
        """When consolidation_llm_max_retries is None, max_retries is not passed."""
        mock_config.consolidation_llm_max_retries = None
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert "max_retries" not in mock_llm_config.call.call_args.kwargs

    @pytest.mark.asyncio
    async def test_reduced_budget_limits_total_calls(self, mock_llm_config, mock_config):
        """Setting both to low values caps total failure attempts."""
        mock_config.consolidation_max_attempts = 2
        mock_config.consolidation_llm_max_retries = 2
        mock_llm_config.call.side_effect = RuntimeError("upstream 503")
        result = await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert result.failed
        assert mock_llm_config.call.call_count == 2
        for call_args in mock_llm_config.call.call_args_list:
            assert call_args.kwargs.get("max_retries") == 2
