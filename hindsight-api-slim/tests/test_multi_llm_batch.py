"""Unit tests for multi-LLM batch routing.

``MultiLLMProvider`` must expose batch capability across the whole chain — not
just the primary — so a secondary member can supply batch capacity (#3645), and
the batch lifecycle must then target THAT member from submit through retrieval.
Crash recovery must resume on the exact *account* that submitted the batch, even
when the chain has been reordered and two members share a provider name (#3671).
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

    def __init__(
        self,
        name: str,
        supports_batch: bool,
        service_tier: str | None = None,
        account: str = "default",
    ):
        self.provider = name
        self.model = f"{name}-model"
        self.openai_service_tier = service_tier
        self._supports_batch = supports_batch
        self.calls: list[str] = []
        self.submitted_body: dict[str, Any] | None = None
        # Stands in for LLMInterface.batch_account_key: a stable, non-secret
        # selector that tells two accounts of the SAME provider apart.
        self.batch_account_key = f"{name}|{account}"

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

    def __init__(
        self,
        name: str,
        supports_batch: bool,
        service_tier: str | None = None,
        account: str = "default",
    ):
        self.provider = name
        self.model = f"{name}-model"
        self._provider_impl = _FakeBatchImpl(name, supports_batch, service_tier, account)

    async def supports_batch_api(self) -> bool:
        return await self._provider_impl.supports_batch_api()

    async def batch_provider_impl(self, account_key: str | None = None) -> _FakeBatchImpl | None:
        """Mirrors LLMProvider.batch_provider_impl, including the account pin."""
        if not await self.supports_batch_api():
            return None
        if account_key is not None and self._provider_impl.batch_account_key != account_key:
            return None
        return self._provider_impl


class _FakeConn:
    """Serves the ``result_metadata`` read the resume path makes, and records the
    batch-state write the submit path makes."""

    def __init__(self, metadata: dict[str, Any]):
        self._metadata = metadata
        self.written_state: dict[str, Any] | None = None

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        if not self._metadata:
            return None
        return {"result_metadata": json.dumps(self._metadata)}

    async def execute(self, query: str, *args: Any) -> None:
        self.written_state = json.loads(args[0])


class _FakePool:
    def __init__(self, metadata: dict[str, Any]):
        self._conn = _FakeConn(metadata)

    @property
    def written_state(self) -> dict[str, Any] | None:
        return self._conn.written_state

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
            config=_batch_config(),
            pool=pool,
            operation_id=str(uuid.uuid4()),
            schema=None,
        )

    assert groq._provider_impl.calls == []


# ── same-provider accounts (#3671) ──────────────────────────────────────────────


async def test_batch_provider_impl_resolves_the_stored_account_not_the_first_member() -> None:
    """Provider name is not an identity: pin the lookup to the stored account."""
    account_a = _BatchMember("openai", True, account="acct-a")
    account_b = _BatchMember("openai", True, account="acct-b")
    multi = _chain(account_a, account_b)

    assert await multi.batch_provider_impl() is account_a._provider_impl
    key_b = account_b._provider_impl.batch_account_key
    assert await multi.batch_provider_impl(account_key=key_b) is account_b._provider_impl


async def test_batch_provider_impl_is_none_when_the_stored_account_is_absent() -> None:
    multi = _chain(_BatchMember("openai", True, account="acct-a"))
    assert await multi.batch_provider_impl(account_key="openai|acct-gone") is None


async def test_submit_records_the_account_that_owns_the_batch() -> None:
    """Without the selector persisted, recovery has nothing to resolve."""
    member = _BatchMember("openai", True, account="acct-b")
    pool = _FakePool({})

    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(member),
        config=_batch_config(),
        pool=pool,
        operation_id=str(uuid.uuid4()),
        schema=None,
    )

    assert pool.written_state is not None
    assert pool.written_state["batch_account"] == member._provider_impl.batch_account_key
    assert "acct-b" in pool.written_state["batch_account"]


async def test_resume_polls_the_submitting_account_after_a_same_provider_reorder() -> None:
    """The #3671 repro: account B submitted, account A is now first in the chain.

    Both members are ``openai``, so the provider string cannot separate them —
    only the persisted account selector can. Polling A would send B's batch id
    with A's credentials.
    """
    account_a = _BatchMember("openai", True, account="acct-a")
    account_b = _BatchMember("openai", True, account="acct-b")
    pool = _FakePool(
        {
            "batch_id": "batch_123",
            "batch_provider": "openai",
            "batch_account": account_b._provider_impl.batch_account_key,
            "chunk_count": 1,
        }
    )

    await extract_facts_from_contents_batch_api(
        contents=[RetainContent(content="Alice moved to Paris in 2023.")],
        llm_config=_chain(account_a, account_b),
        config=_batch_config(),
        pool=pool,
        operation_id=str(uuid.uuid4()),
        schema=None,
    )

    assert account_b._provider_impl.calls == ["status", "retrieve"]
    assert account_a._provider_impl.calls == []


async def test_resume_fails_before_polling_when_the_submitting_account_is_gone() -> None:
    """A member that merely shares the provider name is not a substitute."""
    account_a = _BatchMember("openai", True, account="acct-a")
    pool = _FakePool(
        {
            "batch_id": "batch_123",
            "batch_provider": "openai",
            "batch_account": "openai|acct-b",
            "chunk_count": 1,
        }
    )

    with pytest.raises(RuntimeError, match="Cannot resume batch batch_123"):
        await extract_facts_from_contents_batch_api(
            contents=[RetainContent(content="Alice moved to Paris in 2023.")],
            llm_config=_chain(account_a),
            config=_batch_config(),
            pool=pool,
            operation_id=str(uuid.uuid4()),
            schema=None,
        )

    assert account_a._provider_impl.calls == []
