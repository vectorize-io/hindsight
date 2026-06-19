"""Tests for the LLM failover provider feature."""

import pytest


@pytest.fixture
def clean_config(monkeypatch):
    """Clear cached config before/after each test so env changes take effect."""
    from hindsight_api.config import clear_config_cache

    # Ensure no pre-existing failover env vars leak in
    for key in (
        "HINDSIGHT_API_LLM_FAILOVER_PROVIDER",
        "HINDSIGHT_API_LLM_FAILOVER_API_KEY",
        "HINDSIGHT_API_LLM_FAILOVER_MODEL",
        "HINDSIGHT_API_LLM_FAILOVER_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    clear_config_cache()
    yield
    clear_config_cache()


def test_failover_fields_default_to_none(clean_config):
    """When no failover env vars are set, all four failover fields are None."""
    from hindsight_api.config import get_config

    config = get_config()
    assert config.llm_failover_provider is None
    assert config.llm_failover_api_key is None
    assert config.llm_failover_model is None
    assert config.llm_failover_base_url is None


def test_failover_fields_load_from_env(clean_config, monkeypatch):
    """Setting failover env vars populates the four fields."""
    from hindsight_api.config import clear_config_cache, get_config

    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_PROVIDER", "anthropic")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_API_KEY", "sk-test-failover")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("HINDSIGHT_API_LLM_FAILOVER_BASE_URL", "https://api.anthropic.com")
    clear_config_cache()

    config = get_config()
    assert config.llm_failover_provider == "anthropic"
    assert config.llm_failover_api_key == "sk-test-failover"
    assert config.llm_failover_model == "claude-3-5-sonnet-latest"
    assert config.llm_failover_base_url == "https://api.anthropic.com"


def test_failover_credentials_marked_as_credential_fields():
    """The failover api_key and base_url must be in _CREDENTIAL_FIELDS so the API never echoes them."""
    from hindsight_api.config import HindsightConfig

    credential_fields = HindsightConfig.get_credential_fields()
    assert "llm_failover_api_key" in credential_fields
    assert "llm_failover_base_url" in credential_fields


def test_failover_fields_are_static_not_configurable():
    """Per spec, provider/model/credentials are server-level only — never per-bank configurable."""
    from hindsight_api.config import HindsightConfig

    configurable = HindsightConfig.get_configurable_fields()
    for field in (
        "llm_failover_provider",
        "llm_failover_api_key",
        "llm_failover_model",
        "llm_failover_base_url",
    ):
        assert field not in configurable, f"{field} must not be per-bank configurable"


# ---------------------------------------------------------------------------
# Task 2: FailoverLLMProvider composite tests
# ---------------------------------------------------------------------------


from hindsight_api.engine.llm_interface import OutputTooLongError, ProviderRateLimitResetError


def _make_mock(model: str):
    from hindsight_api.engine.llm_wrapper import LLMProvider

    return LLMProvider(provider="mock", api_key="", base_url="", model=model)


@pytest.mark.asyncio
async def test_call_succeeds_on_primary_no_failover_call(clean_config):
    """When primary succeeds, failover is never invoked."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    primary.set_mock_response("primary response")
    failover.set_mock_response("failover response")

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    result = await composite.call(messages=[{"role": "user", "content": "hi"}])

    assert result == "primary response"
    assert len(primary.get_mock_calls()) == 1
    assert len(failover.get_mock_calls()) == 0


@pytest.mark.asyncio
async def test_call_falls_over_when_primary_raises(clean_config):
    """When primary raises a transient error, failover is invoked with same messages."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    # Reach into the underlying MockLLM to make the primary raise
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = RuntimeError("upstream 503")
    failover.set_mock_response("failover saved us")

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    result = await composite.call(messages=[{"role": "user", "content": "hi"}])

    assert result == "failover saved us"
    assert len(primary.get_mock_calls()) == 1
    assert len(failover.get_mock_calls()) == 1
    # Same payload was forwarded
    assert failover.get_mock_calls()[0]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_call_raises_failover_error_when_both_fail(clean_config):
    """When both primary and failover fail, the failover's error propagates."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    assert isinstance(primary._provider_impl, MockLLM)
    assert isinstance(failover._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = RuntimeError("primary down")
    failover._provider_impl._mock_exception = RuntimeError("failover also down")

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    with pytest.raises(RuntimeError, match="failover also down"):
        await composite.call(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_call_re_raises_when_no_failover_configured(clean_config):
    """When failover is None, primary errors propagate unchanged (backwards-compat path)."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = RuntimeError("primary down")

    composite = FailoverLLMProvider(primary=primary, failover=None)
    with pytest.raises(RuntimeError, match="primary down"):
        await composite.call(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_output_too_long_does_not_failover(clean_config):
    """OutputTooLongError is deterministic — failover cannot help, so it propagates."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = OutputTooLongError("output > max_tokens")

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    with pytest.raises(OutputTooLongError):
        await composite.call(messages=[{"role": "user", "content": "hi"}])
    assert len(failover.get_mock_calls()) == 0


@pytest.mark.asyncio
async def test_cancelled_error_does_not_failover(clean_config):
    """asyncio.CancelledError must propagate so cancellation isn't swallowed."""
    import asyncio

    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = asyncio.CancelledError()

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    with pytest.raises(asyncio.CancelledError):
        await composite.call(messages=[{"role": "user", "content": "hi"}])
    assert len(failover.get_mock_calls()) == 0


@pytest.mark.asyncio
async def test_rate_limit_reset_does_failover(clean_config):
    """ProviderRateLimitResetError IS a failover trigger — primary says 'come back later'."""
    from datetime import datetime, timedelta, timezone

    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    failover.set_mock_response("failover handled it")
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = ProviderRateLimitResetError(
        retry_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        message="rate limited until 12:05",
    )

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    result = await composite.call(messages=[{"role": "user", "content": "hi"}])

    assert result == "failover handled it"
    assert len(failover.get_mock_calls()) == 1


@pytest.mark.asyncio
async def test_attribute_passthrough_returns_primary(clean_config):
    """.provider / .model read from the primary for backward compat."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    composite = FailoverLLMProvider(primary=primary, failover=failover)

    assert composite.provider == "mock"
    assert composite.model == "primary-model"


@pytest.mark.asyncio
async def test_call_with_tools_falls_over(clean_config):
    """call_with_tools has identical failover semantics to call()."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider
    from hindsight_api.engine.providers.mock_llm import MockLLM
    from hindsight_api.engine.response_models import LLMToolCallResult

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    assert isinstance(primary._provider_impl, MockLLM)
    primary._provider_impl._mock_exception = RuntimeError("upstream 500")
    failover_result = LLMToolCallResult(content="from failover", tool_calls=[])
    failover.set_mock_response(failover_result)

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    result = await composite.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    )

    assert result.content == "from failover"


@pytest.mark.asyncio
async def test_cleanup_cleans_both():
    """cleanup() must clean both primary and failover."""
    from hindsight_api.engine.failover_llm import FailoverLLMProvider

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")
    composite = FailoverLLMProvider(primary=primary, failover=failover)

    # Should not raise — both MockLLM cleanups are no-ops
    await composite.cleanup()


@pytest.mark.asyncio
async def test_verify_connection_warns_on_failover_failure(caplog):
    """A failing failover.verify_connection() must NOT raise — only log a warning."""
    import logging

    from hindsight_api.engine.failover_llm import FailoverLLMProvider

    primary = _make_mock("primary-model")
    failover = _make_mock("failover-model")

    # Patch failover to raise on verify
    async def boom():
        raise RuntimeError("failover unreachable")

    failover.verify_connection = boom  # type: ignore[method-assign]

    composite = FailoverLLMProvider(primary=primary, failover=failover)
    caplog.set_level(logging.WARNING, logger="hindsight_api.engine.failover_llm")
    await composite.verify_connection()  # must not raise

    assert any(
        "failover" in record.message.lower() and "unreachable" in record.message.lower() for record in caplog.records
    )
