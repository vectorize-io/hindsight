"""Tests for consolidation retry budget configurability (issue #1042) and
failure classification (issue #3684)."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from hindsight_api.engine.consolidation.consolidator import _consolidate_batch_with_llm
from hindsight_api.engine.llm_interface import OutputTooLongError, ProviderRateLimitResetError
from hindsight_api.engine.providers.openai_compatible_llm import ProviderResponseError


@pytest.fixture
def mock_llm_config():
    llm = AsyncMock()
    response = MagicMock()
    response.creates = []
    response.updates = []
    response.deletes = []
    llm.call.return_value = response
    return llm


@pytest.fixture(autouse=True)
def no_real_sleep():
    """Neutralise the outer retry backoff (#3684) so these unit tests stay fast."""
    with patch(
        "hindsight_api.engine.consolidation.consolidator.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep:
        yield sleep


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.observations_mission = None
    config.consolidation_max_attempts = 3
    config.consolidation_llm_max_retries = None
    config.consolidation_max_completion_tokens = None
    config.llm_strict_schema_consolidation = False
    config.llm_temperature_consolidation = 0.0
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
    async def test_strict_schema_threaded_to_call(self, mock_llm_config, mock_config):
        """llm_strict_schema_consolidation is passed to llm_config.call()."""
        mock_config.llm_strict_schema_consolidation = True
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("strict_schema") is True

    @pytest.mark.asyncio
    async def test_temperature_threaded_to_call(self, mock_llm_config, mock_config):
        """llm_temperature_consolidation is passed to llm_config.call()."""
        mock_config.llm_temperature_consolidation = 0.65
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("temperature") == 0.65

    @pytest.mark.asyncio
    async def test_strict_schema_passed_as_explicit_false(self, mock_llm_config, mock_config):
        """A disabled per-operation flag is passed explicitly, not omitted.

        Omitting it would let the global HINDSIGHT_API_LLM_STRICT_SCHEMA flag win,
        which is exactly what the per-operation opt-out exists to prevent.
        """
        mock_config.llm_strict_schema_consolidation = False
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("strict_schema") is False

    @pytest.mark.asyncio
    async def test_max_completion_tokens_threaded_to_call(self, mock_llm_config, mock_config):
        """consolidation_max_completion_tokens is passed to llm_config.call()."""
        mock_config.consolidation_max_completion_tokens = 8192
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert mock_llm_config.call.call_args.kwargs.get("max_completion_tokens") == 8192

    @pytest.mark.asyncio
    async def test_max_completion_tokens_not_passed_when_none(self, mock_llm_config, mock_config):
        """When consolidation_max_completion_tokens is None, max_completion_tokens is omitted (no regression)."""
        mock_config.consolidation_max_completion_tokens = None
        await _consolidate_batch_with_llm(
            llm_config=mock_llm_config,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=mock_config,
        )
        assert "max_completion_tokens" not in mock_llm_config.call.call_args.kwargs

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


class _AuthError(Exception):
    """Stand-in for a provider SDK's 401/403, which the classifier duck-types."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _validation_error() -> ValidationError:
    """A real pydantic ValidationError, as ``response_format.model_validate`` raises."""

    class _Shape(BaseModel):
        n: int

    try:
        _Shape.model_validate({"n": "not-an-int"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")


class TestConsolidationFailureClassification:
    """Issue #3684: the outer batch ladder must not retry what a re-send cannot fix.

    The loop rebuilds a byte-identical payload every attempt, on top of the
    provider's own ladder, so retrying an input-shaped failure multiplies cost for
    nothing — and retrying a *control signal* loses it entirely.
    """

    async def _run(self, llm, config):
        return await _consolidate_batch_with_llm(
            llm_config=llm,
            memories=[{"id": "m1", "text": "test"}],
            union_observations=[],
            union_source_facts={},
            config=config,
        )

    @pytest.mark.asyncio
    async def test_quota_defer_propagates_instead_of_failing_the_batch(self, mock_llm_config, mock_config):
        """ProviderRateLimitResetError is a defer signal, not a batch failure.

        Swallowing it meant the batch was reported failed, the caller bisected it
        into sub-batches that each re-hit the same quota wall, and every memory was
        stamped consolidation_failed_at — permanently excluding facts from
        consolidation because quota was briefly exhausted.
        """
        retry_at = datetime(2030, 1, 1, tzinfo=timezone.utc)
        mock_llm_config.call.side_effect = ProviderRateLimitResetError(retry_at=retry_at, message="quota")

        with pytest.raises(ProviderRateLimitResetError) as excinfo:
            await self._run(mock_llm_config, mock_config)

        assert excinfo.value.retry_at == retry_at
        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_auth_error_propagates(self, mock_llm_config, mock_config, status_code):
        """A bad key is a server misconfiguration, not a property of these memories."""
        mock_llm_config.call.side_effect = _AuthError(status_code)

        with pytest.raises(_AuthError):
            await self._run(mock_llm_config, mock_config)

        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    async def test_json_decode_error_is_not_re_sent(self, mock_llm_config, mock_config):
        """The provider already burned its own ladder on this; re-sending adds nothing.

        Still reported failed, so the caller's adaptive bisection — which *does*
        vary the input — gets its chance.
        """
        mock_llm_config.call.side_effect = json.JSONDecodeError("Expecting ',' delimiter", "{}", 1)

        result = await self._run(mock_llm_config, mock_config)

        assert result.failed
        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    async def test_validation_error_is_not_re_sent(self, mock_llm_config, mock_config):
        """Schema-invalid output is input-shaped: the same prompt yields the same shape."""
        mock_llm_config.call.side_effect = _validation_error()

        result = await self._run(mock_llm_config, mock_config)

        assert result.failed
        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    async def test_output_too_long_is_not_re_sent(self, mock_llm_config, mock_config):
        """Bisection shortens the output; an identical re-send cannot."""
        mock_llm_config.call.side_effect = OutputTooLongError("too long")

        result = await self._run(mock_llm_config, mock_config)

        assert result.failed
        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    async def test_provider_flagged_non_retryable_is_not_re_sent(self, mock_llm_config, mock_config):
        """A provider that already decided the failure is permanent is believed."""
        mock_llm_config.call.side_effect = ProviderResponseError("empty body", retryable=False)

        result = await self._run(mock_llm_config, mock_config)

        assert result.failed
        assert mock_llm_config.call.call_count == 1

    @pytest.mark.asyncio
    async def test_transport_error_still_retries(self, mock_llm_config, mock_config):
        """Transport-shaped failures keep the full outer budget — this is the case it exists for."""
        mock_config.consolidation_max_attempts = 3
        mock_llm_config.call.side_effect = ConnectionResetError("connection reset")

        result = await self._run(mock_llm_config, mock_config)

        assert result.failed
        assert mock_llm_config.call.call_count == 3

    @pytest.mark.asyncio
    async def test_outer_loop_backs_off_between_retries(self, mock_llm_config, mock_config, no_real_sleep):
        """Without backoff a rate limit was re-sent immediately, three times."""
        mock_config.consolidation_max_attempts = 3
        mock_llm_config.call.side_effect = ConnectionResetError("connection reset")

        await self._run(mock_llm_config, mock_config)

        # One sleep between attempts, none after the last, and growing.
        delays = [call.args[0] for call in no_real_sleep.await_args_list]
        assert len(delays) == 2
        assert delays[0] < delays[1]

    @pytest.mark.asyncio
    async def test_no_backoff_when_failing_fast(self, mock_llm_config, mock_config, no_real_sleep):
        """A fail-fast class must not pay the backoff it is skipping the retry for."""
        mock_llm_config.call.side_effect = json.JSONDecodeError("bad", "{}", 1)

        await self._run(mock_llm_config, mock_config)

        assert no_real_sleep.await_count == 0
