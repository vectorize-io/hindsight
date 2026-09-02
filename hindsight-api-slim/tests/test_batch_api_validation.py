"""Startup validation for the batch-retain knob.

``HINDSIGHT_API_RETAIN_BATCH_ENABLED=true`` must fail fast when the retain LLM
cannot actually serve a batch — otherwise every retain silently falls back to
sync mode and the knob does nothing. For a multi-LLM chain the capability is
evaluated across ALL members (#3645): batch capacity may live on a secondary,
and gating on the primary alone rejected configurations that would have worked.

These tests drive the real ``validate_retain_batch_support`` with real providers
(``mock`` has no batch API, ``openai`` does). An earlier version re-implemented
the check inline and asserted against its own copy, so it could not have caught
a regression in the engine.
"""

import pytest

from hindsight_api.config import LLM_STRATEGY_FAILOVER, HindsightConfig, LLMStrategyConfig
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.memory_engine import validate_retain_batch_support
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.retain.fact_extraction import RetainContent, extract_facts_from_contents_batch_api


def _provider(name: str) -> LLMProvider:
    """A real provider: ``mock`` has no batch API, ``openai``/``groq`` do."""
    return LLMProvider(
        provider=name,
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model=f"{name}-model",
    )


def _chain(*names: str) -> MultiLLMProvider:
    return MultiLLMProvider([_provider(n) for n in names], LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))


def _config(*, batch_enabled: bool) -> HindsightConfig:
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = batch_enabled
    return config


async def test_startup_rejects_batch_enabled_with_non_batch_provider() -> None:
    with pytest.raises(RuntimeError, match="does not support the batch API"):
        await validate_retain_batch_support(_provider("mock"), _config(batch_enabled=True))


async def test_startup_allows_batch_enabled_with_batch_provider() -> None:
    await validate_retain_batch_support(_provider("openai"), _config(batch_enabled=True))


async def test_startup_allows_batch_disabled_with_non_batch_provider() -> None:
    await validate_retain_batch_support(_provider("mock"), _config(batch_enabled=False))


async def test_startup_allows_batch_enabled_when_only_a_secondary_supports_it() -> None:
    """#3645: the knob used to be evaluated against the primary alone."""
    await validate_retain_batch_support(_chain("mock", "openai"), _config(batch_enabled=True))


async def test_startup_rejects_batch_enabled_when_no_chain_member_supports_it() -> None:
    with pytest.raises(RuntimeError, match="no member of the retain LLM chain") as excinfo:
        await validate_retain_batch_support(_chain("mock", "ollama"), _config(batch_enabled=True))
    # The operator has to know which members were considered, not just the primary.
    assert "'mock'" in str(excinfo.value)
    assert "'ollama'" in str(excinfo.value)


async def test_runtime_raises_if_batch_unsupported() -> None:
    """Belt-and-braces: the extraction path itself refuses a non-batch provider."""
    with pytest.raises(RuntimeError, match="does not support the batch API"):
        await extract_facts_from_contents_batch_api(
            contents=[RetainContent(content="Alice moved to Paris in 2023.")],
            llm_config=_provider("mock"),
            config=_config(batch_enabled=True),
            pool=None,
            operation_id=None,
            schema=None,
        )
