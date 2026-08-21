"""Unit tests for multi-LLM batch routing.

``MultiLLMProvider`` must expose batch capability across the whole chain — not
just the primary — so a secondary member can supply batch capacity (#3645), and
the batch lifecycle must then target THAT member from submit through retrieval.
These tests use lightweight fakes with a configurable
``_provider_impl.supports_batch_api``; no real providers or network.
"""

import json
import uuid
from typing import Any

import pytest

from hindsight_api.config import LLM_STRATEGY_FAILOVER, HindsightConfig, LLMStrategyConfig, _generate_member_id
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.retain.fact_extraction import RetainContent, extract_facts_from_contents_batch_api


class _FakeBatchImpl:
    """Fake provider implementation recording the batch calls it serves."""

    def __init__(self, name: str, supports_batch: bool, service_tier: str | None = None, member_id: str | None = None):
        self.provider = name
        self.name = name
        self.member_id = member_id or name
        self.model = f"{name}-model"
        self.openai_service_tier = service_tier
        self._supports_batch = supports_batch
        self.calls: list[str] = []
        self.submitted_body: dict[str, Any] | None = None

    async def supports_batch_api(self) -> bool:
        return self._supports_batch

    async def submit_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append("submit")
        self.submitted_body = requests[0]["body"]
        return {"batch_id": "batch_123"}

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        self.calls.append("status")
        return {"status": "completed", "request_counts": {"total": 1, "completed": 1, "failed": 0}}

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        self.calls.append("retrieve")
        return [
            {
                "custom_id": "chunk_0",
                "response": {"body": {"choices": [{"message": {"content": json.dumps({"facts": []})}}], "usage": {}}},
            }
        ]


class _BatchMember:
    """Fake LLMProvider member exposing the batch surface the chain delegates to."""

    def __init__(self, name: str, supports_batch: bool, service_tier: str | None = None):
        self.provider = name
        self.model = f"{name}-model"
        self._provider_impl = _FakeBatchImpl(name, supports_batch, service_tier)

    async def supports_batch_api(self) -> bool:
        return await self._provider_impl.supports_batch_api()

    async def batch_provider_impl(self) -> _FakeBatchImpl | None:
        return self._provider_impl if await self.supports_batch_api() else None


class _FakeConn:
    """Serves the one ``result_metadata`` read the resume path makes."""

    def __init__(self, metadata: dict[str, Any]):
        self._metadata = metadata

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        return {"result_metadata": json.dumps(self._metadata)}


class _FakePool:
    def __init__(self, metadata: dict[str, Any]):
        self._conn = _FakeConn(metadata)

    async def acquire(self) -> _FakeConn:
        return self._conn

    async def release(self, conn: _FakeConn) -> None:
        return None

    def get_size(self) -> int:
        return 1

    def get_idle_size(self) -> int:
        return 1


def _chain(*members: _BatchMember) -> MultiLLMProvider:
    return MultiLLMProvider(list(members), LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))


def _batch_config() -> HindsightConfig:
    config = HindsightConfig.from_env()
    config.retain_batch_enabled = True
    return config


# ── capability + member selection ───────────────────────────────────────────────


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


async def test_batch_provider_impl_is_none_when_no_member_capable() -> None:
    """``None`` is the single 'cannot serve a batch' answer; the caller raises."""
    multi = _chain(_BatchMember("deepseek", False), _BatchMember("gemini", False))
    assert await multi.batch_provider_impl() is None


# ── the batch lifecycle targets the selected member ─────────────────────────────


async def test_batch_lifecycle_runs_entirely_on_the_batch_capable_member() -> None:
    """The point of #3645: submit, poll and retrieve all go to the secondary."""
    primary = _BatchMember("deepseek", False)
    secondary = _BatchMember("openai", True, service_tier="flex")

    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(primary, secondary),
        agent_name="test_agent",
        config=_batch_config(),
        pool=None,
        operation_id=None,
        schema=None,
    )

    assert secondary._provider_impl.calls == ["submit", "status", "retrieve"]
    assert primary._provider_impl.calls == []


async def test_batch_request_body_carries_the_serving_members_settings() -> None:
    """model/service_tier must match the account the batch is submitted to."""
    secondary = _BatchMember("openai", True, service_tier="flex")

    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(_BatchMember("deepseek", False), secondary),
        agent_name="test_agent",
        config=_batch_config(),
        pool=None,
        operation_id=None,
        schema=None,
    )

    body = secondary._provider_impl.submitted_body
    assert body is not None
    assert body["model"] == "openai-model"  # not the chain primary's model
    assert body["service_tier"] == "flex"


# ── crash-recovery resume ───────────────────────────────────────────────────────


async def test_resume_polls_the_member_that_submitted_the_batch() -> None:
    secondary = _BatchMember("openai", True)
    pool = _FakePool({"batch_id": "batch_123", "batch_provider": "openai", "chunk_count": 1})

    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(_BatchMember("deepseek", False), secondary),
        agent_name="test_agent",
        config=_batch_config(),
        pool=pool,
        operation_id=str(uuid.uuid4()),
        schema=None,
    )

    # Resumed, so no second submit — it picks up at polling.
    assert secondary._provider_impl.calls == ["status", "retrieve"]


async def test_resume_fails_loudly_when_the_chain_no_longer_serves_that_provider() -> None:
    """A batch_id only exists on the account that created it.

    If the chain is edited between submit and resume, silently polling a
    different account would hang until the wall clock ran out and then report a
    provider error nobody can act on.
    """
    groq = _BatchMember("groq", True)
    pool = _FakePool({"batch_id": "batch_123", "batch_provider": "openai", "chunk_count": 1})

    with pytest.raises(RuntimeError, match="Cannot resume batch batch_123"):
        await extract_facts_from_contents_batch_api(
            contents=[RetainContent(content="Alice moved to Paris in 2023.")],
            llm_config=_chain(_BatchMember("deepseek", False), groq),
            agent_name="test_agent",
            config=_batch_config(),
            pool=pool,
            operation_id=str(uuid.uuid4()),
            schema=None,
        )

    assert groq._provider_impl.calls == []


async def test_batch_resume_same_provider_member_reordered() -> None:
    """Same-provider reorder must not route recovery to the wrong account.

    Two members share the same provider string ("openai") but have distinct
    member_id values ("account_a", "account_b"). After submitting via member_b
    and restarting with member_a first in the chain, the persisted
    ``batch_member`` selector must still resolve to member_b — not member_a.
    """
    # Configure two wrapper members of the same provider ("openai")
    member_a = _BatchMember("openai", supports_batch=True)
    member_a._provider_impl.member_id = "account_a"

    member_b = _BatchMember("openai", supports_batch=True)
    member_b._provider_impl.member_id = "account_b"

    # Initial order where member_b is primary or chosen
    pool_submit = MultiLLMProvider([member_b, member_a], strategy=LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))
    batch_impl = await pool_submit.batch_provider_impl()
    assert getattr(batch_impl, "member_id", None) == "account_b"

    # Simulate saved state from submission
    persisted_metadata = {
        "batch_id": "batch_123",
        "batch_provider": "openai",
        "batch_member": "account_b",
    }

    # Simulate restart with pool order swapped: [member_a, member_b]
    pool_resumed = MultiLLMProvider([member_a, member_b], strategy=LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))

    # Resolving with selector should return member_b despite member_a being first
    resumed_impl = await pool_resumed.batch_provider_impl(selector=persisted_metadata["batch_member"])
    assert resumed_impl is not None
    assert getattr(resumed_impl, "member_id", None) == "account_b"

    # Call status & retrieve on the resumed implementation
    await resumed_impl.get_batch_status(persisted_metadata["batch_id"])
    await resumed_impl.retrieve_batch_results(persisted_metadata["batch_id"])

    # Assert account_b handled recovery and account_a received 0 calls
    assert member_b._provider_impl.calls == ["status", "retrieve"]
    assert member_a._provider_impl.calls == []


async def test_batch_resume_same_provider_real_fingerprint_reordered() -> None:
    """Same as test_batch_resume_same_provider_member_reordered, but member_id
    comes from the real _generate_member_id (differing only by api_key) rather
    than being hand-assigned — proves the fingerprint function itself is what
    makes recovery correct, not just the selector plumbing."""
    member_id_a = _generate_member_id("openai", "gpt-4o", "sk-account-a")
    member_id_b = _generate_member_id("openai", "gpt-4o", "sk-account-b")

    member_a = _BatchMember("openai", supports_batch=True)
    member_a._provider_impl.member_id = member_id_a

    member_b = _BatchMember("openai", supports_batch=True)
    member_b._provider_impl.member_id = member_id_b

    # Sanity check: the two accounts must actually fingerprint differently,
    # or this whole test would pass for the wrong reason.
    assert member_id_a != member_id_b

    # Initial order where member_b is primary or chosen
    pool_submit = MultiLLMProvider([member_b, member_a], strategy=LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))
    batch_impl = await pool_submit.batch_provider_impl()
    assert getattr(batch_impl, "member_id", None) == member_id_b

    # Simulate saved state from submission
    persisted_metadata = {
        "batch_id": "batch_123",
        "batch_provider": "openai",
        "batch_member": member_id_b,
    }

    # Simulate restart with pool order swapped: [member_a, member_b]
    pool_resumed = MultiLLMProvider([member_a, member_b], strategy=LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))

    # Resolving with selector should return member_b despite member_a being first
    resumed_impl = await pool_resumed.batch_provider_impl(selector=persisted_metadata["batch_member"])
    assert resumed_impl is not None
    assert getattr(resumed_impl, "member_id", None) == member_id_b

    # Call status & retrieve on the resumed implementation
    await resumed_impl.get_batch_status(persisted_metadata["batch_id"])
    await resumed_impl.retrieve_batch_results(persisted_metadata["batch_id"])

    # Assert account_b handled recovery and account_a received 0 calls
    assert member_b._provider_impl.calls == ["status", "retrieve"]
    assert member_a._provider_impl.calls == []
