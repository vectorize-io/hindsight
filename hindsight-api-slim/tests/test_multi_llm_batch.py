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

from hindsight_api.config import LLM_STRATEGY_FAILOVER, HindsightConfig, LLMStrategyConfig
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.retain.fact_extraction import RetainContent, extract_facts_from_contents_batch_api


class _FakeBatchImpl:
    """Fake provider implementation recording the batch calls it serves."""

    def __init__(self, name: str, supports_batch: bool, service_tier: str | None = None):
        self.provider = name
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
    """Fake LLMProvider member exposing the batch surface the chain delegates to.

    ``identity`` folds a per-account marker (mirroring the real credential
    digest) so two members with the same provider kind are still distinct
    accounts — the collision case ``batch_provider`` (the provider name) used to
    hit. This lets the account-identity resume path be exercised without real
    credentials.
    """

    def __init__(self, name: str, supports_batch: bool, service_tier: str | None = None, identity: str | None = None):
        self.provider = name
        self.model = f"{name}-model"
        self._identity = identity or f"{name}::{name}-model"
        self._provider_impl = _FakeBatchImpl(name, supports_batch, service_tier)

    async def supports_batch_api(self) -> bool:
        return await self._provider_impl.supports_batch_api()

    async def batch_provider_impl(self) -> _FakeBatchImpl | None:
        return self._provider_impl if await self.supports_batch_api() else None

    async def account_identity(self) -> str:
        return self._identity

    async def resolve_batch_account(self, identity: str) -> _FakeBatchImpl | None:
        return self._provider_impl if identity == self._identity else None


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


# ── account-identity resume (post-merge review follow-up) ──────────────────────


async def test_resume_resolves_to_submitting_account_after_reorder() -> None:
    """Regression for the post-merge review: recovery must resume on the ACCOUNT
    that submitted the batch, not the newly-selected first-capable member after a
    chain reorder.

    Scenario: the batch was submitted through openai-A (secondary). Between
    submit and resume the operator reorders the chain so a different openai
    account (openai-B) becomes first-capable, while openai-A moves down.
    ``resolve_batch_account`` must still find openai-A by its persisted identity.
    """
    openai_a = _BatchMember("openai", True, identity="openai::acct-A")
    submit_chain = _chain(_BatchMember("deepseek", False), openai_a)
    submitting_impl = await submit_chain.batch_provider_impl()
    persisted_identity = await submit_chain.account_identity()
    assert submitting_impl is openai_a._provider_impl
    assert persisted_identity == "openai::acct-A"

    # After restart the chain is reordered: openai-B is now primary+first-capable
    # and the original openai-A is a secondary.
    openai_b = _BatchMember("openai", True, identity="openai::acct-B")
    resume_chain = _chain(openai_b, openai_a)
    # Naive re-selection would poll through the new primary (openai-B)...
    assert await resume_chain.batch_provider_impl() is openai_b._provider_impl
    assert await resume_chain.batch_provider_impl() is not submitting_impl
    # ...but identity-driven recovery finds the submitting account (openai-A):
    resolved = await resume_chain.resolve_batch_account(persisted_identity)
    assert resolved is openai_a._provider_impl
    assert resolved is submitting_impl


async def test_resume_resolves_to_submitting_account_via_fact_extraction() -> None:
    """End-to-end: the persisted ``batch_account`` is written at submit and used
    on resume, so the batch polls through the SAME secondary account even after
    the chain is reordered."""
    openai_a = _BatchMember("openai", True, identity="openai::acct-A")
    openai_b = _BatchMember("openai", True, identity="openai::acct-B")
    submit_chain = _chain(_BatchMember("deepseek", False), openai_a)
    persisted_identity = await submit_chain.account_identity()

    # Resume with the stored identity; the new chain would naively pick openai-B.
    pool = _FakePool(
        {
            "batch_id": "batch_123",
            "batch_provider": "openai",
            "batch_account": persisted_identity,
            "chunk_count": 1,
        }
    )
    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(openai_b, openai_a),
        agent_name="test_agent",
        config=_batch_config(),
        pool=pool,
        operation_id=str(uuid.uuid4()),
        schema=None,
    )

    # Resumed on openai-A (the submitting account) — no re-submit, polls on A.
    assert openai_a._provider_impl.calls == ["status", "retrieve"]
    assert openai_b._provider_impl.calls == []


async def test_resume_fails_when_submitting_account_no_longer_configured() -> None:
    """If the account that submitted the batch is removed from the chain, recovery
    must surface a clear failure rather than silently polling a different account."""
    submit_chain = _chain(_BatchMember("deepseek", False), _BatchMember("openai", True))
    persisted_identity = await submit_chain.account_identity()

    resume_chain = _chain(_BatchMember("openai", True, identity="openai::acct-B"))
    with pytest.raises(RuntimeError, match="Cannot resume batch batch_123"):
        await extract_facts_from_contents_batch_api(
            contents=[RetainContent(content="Alice moved to Paris in 2023.")],
            llm_config=resume_chain,
            agent_name="test_agent",
            config=_batch_config(),
            pool=_FakePool({"batch_id": "batch_123", "batch_account": persisted_identity, "chunk_count": 1}),
            operation_id=str(uuid.uuid4()),
            schema=None,
        )


async def test_resume_legacy_metadata_still_guards_by_provider() -> None:
    """Metadata written before ``batch_account`` existed falls back to the
    coarse provider-name guard (upstream behaviour) so a cross-provider mismatch
    still fails loudly."""
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
