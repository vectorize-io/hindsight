"""Unit tests for multi-LLM batch routing.

``MultiLLMProvider`` must expose batch capability across the whole chain — not
just the primary — so a secondary member can supply batch capacity (see the
``supports_batch_api`` / ``batch_provider_impl`` methods). These tests use
lightweight fakes with a configurable ``_provider_impl.supports_batch_api``;
no real providers or network.
"""

from hindsight_api.config import LLMStrategyConfig
from hindsight_api.engine.multi_llm import MultiLLMProvider


class _FakeBatchImpl:
    """Fake provider implementation with a configurable batch capability."""

    def __init__(self, name: str, supports_batch: bool):
        self.provider = name
        self.model = f"{name}-model"
        self._supports_batch = supports_batch

    async def supports_batch_api(self) -> bool:
        return self._supports_batch


class _BatchMember:
    """Fake LLMProvider member whose ``_provider_impl`` reports batch capability."""

    def __init__(self, name: str, supports_batch: bool):
        self.provider = name
        self.model = f"{name}-model"
        self._provider_impl = _FakeBatchImpl(name, supports_batch)


def _chain(*members) -> MultiLLMProvider:
    return MultiLLMProvider(list(members), LLMStrategyConfig(mode="failover"))


async def test_supports_batch_api_true_when_only_secondary_supports() -> None:
    multi = _chain(_BatchMember("deepseek", False), _BatchMember("openai", True))
    assert await multi.supports_batch_api() is True


async def test_supports_batch_api_false_when_no_member_supports() -> None:
    multi = _chain(_BatchMember("deepseek", False), _BatchMember("gemini", False))
    assert await multi.supports_batch_api() is False


async def test_batch_provider_impl_selects_first_capable_member() -> None:
    primary = _BatchMember("deepseek", False)
    secondary = _BatchMember("openai", True)
    multi = _chain(primary, secondary)
    assert await multi.batch_provider_impl() is secondary._provider_impl


async def test_batch_provider_impl_prefers_primary_when_capable() -> None:
    primary = _BatchMember("openai", True)
    secondary = _BatchMember("groq", True)
    multi = _chain(primary, secondary)
    assert await multi.batch_provider_impl() is primary._provider_impl


async def test_batch_provider_impl_falls_back_to_primary_when_none_capable() -> None:
    primary = _BatchMember("deepseek", False)
    secondary = _BatchMember("gemini", False)
    multi = _chain(primary, secondary)
    # Preserve single-provider behaviour: the caller then raises the existing
    # "does not support the batch API" error, unchanged.
    assert await multi.batch_provider_impl() is primary._provider_impl
