"""Routing behavior at call/call_with_tools; time and provider adapters are controlled."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import pytest

from hindsight_api.config import LLMStrategyConfig
from hindsight_api.engine.llm_interface import (
    LLMCooldownFailure,
    LLMFailureClassification,
    LLMTerminalFailure,
    OutputTooLongError,
    ProviderRateLimitResetError,
    ProviderReauthenticationRequiredError,
)
from hindsight_api.engine.multi_llm import MultiLLMProvider


class QuotaError(RuntimeError):
    pass


class NativeAuthError(RuntimeError):
    pass


class Member:
    """Provider adapter exercising the public member interface."""

    def __init__(self, name: str, result: str | BaseException) -> None:
        self.provider = "test"
        self.model = name
        self.member_label = name
        self.result = result
        self.delay: float | None = 10.0
        self.pending: Callable[[], Awaitable[Any]] | None = None

    def classify_failure(self, exc: BaseException) -> LLMFailureClassification | None:
        if isinstance(exc, NativeAuthError):
            return LLMTerminalFailure()
        return LLMCooldownFailure(retry_after_seconds=self.delay) if isinstance(exc, QuotaError) else None

    async def call(self, **kwargs: Any) -> Any:
        if self.pending is not None:
            return await self.pending()
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def call_with_tools(self, **kwargs: Any) -> Any:
        return await self.call(**kwargs)

    async def batch_provider_impl(self, account_key: str | None = None) -> "Member | None":
        return self if account_key is None or account_key == self.model else None


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_quota_skips_member_until_expiry_then_lazily_fails_back(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    kwargs = {"messages": [], **({"tools": []} if method == "call_with_tools" else {})}
    assert await getattr(router, method)(**kwargs) == "fallback"
    primary.result = "primary"
    assert await getattr(router, method)(**kwargs) == "fallback"
    now = 110.0
    assert await getattr(router, method)(**kwargs) == "primary"


async def test_only_one_half_open_probe_while_other_calls_use_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    assert await router.call(messages=[]) == "fallback"
    now = 110.0
    entered, release = asyncio.Event(), asyncio.Event()

    async def probe() -> str:
        if entered.is_set():
            return "duplicate-probe"
        entered.set()
        await release.wait()
        return "recovered"

    primary.pending = probe
    task = asyncio.create_task(router.call(messages=[]))
    try:
        await entered.wait()
        assert await router.call_with_tools(messages=[], tools=[]) == "fallback"
    finally:
        release.set()
        assert await task == "recovered"
    primary.pending = None
    primary.result = "primary"
    assert await router.call(messages=[]) == "primary"


@pytest.mark.parametrize("failure", [RuntimeError("temporary"), QuotaError("quota")])
async def test_failed_probe_recools(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    assert await router.call(messages=[]) == "fallback"
    now = 110.0
    primary.result = failure
    assert await router.call(messages=[]) == "fallback"
    primary.result = "primary"
    now = 119.0
    assert await router.call(messages=[]) == "fallback"
    now = 170.0 if not isinstance(failure, QuotaError) else 120.0
    assert await router.call(messages=[]) == "primary"


async def test_stale_inflight_success_cannot_clear_newer_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: 100.0)
    primary, fallback = Member("primary", "primary"), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed() -> str:
        entered.set()
        await release.wait()
        return "old-success"

    primary.pending = delayed
    task = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    primary.pending = None
    primary.result = QuotaError()
    try:
        assert await router.call(messages=[]) == "fallback"
    finally:
        release.set()
        assert await task == "old-success"
    primary.result = "primary"
    assert await router.call(messages=[]) == "fallback"


@pytest.mark.parametrize("delay", [None, float("nan"), float("inf"), -1.0, 20.0, 1e300])
async def test_all_cooling_returns_finite_retry_time(monkeypatch: pytest.MonkeyPatch, delay: float | None) -> None:
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: 100.0)
    primary = Member("primary", QuotaError())
    primary.delay = delay
    router = MultiLLMProvider([primary], LLMStrategyConfig(mode="failover"))
    for _ in range(2):
        before = datetime.now(timezone.utc)
        with pytest.raises(ProviderRateLimitResetError) as caught:
            await router.call(messages=[])
        assert caught.value.retry_at > before
        if delay != 1e300:
            assert 19 < (caught.value.retry_at - before).total_seconds() < 61


async def test_probe_only_chain_defers_and_cancelled_probe_releases_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary = Member("primary", QuotaError())
    router = MultiLLMProvider([primary], LLMStrategyConfig(mode="failover"))
    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[])
    now = 110.0
    entered = asyncio.Event()

    async def pending() -> None:
        entered.set()
        await asyncio.Event().wait()

    primary.pending = pending
    task = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    try:
        with pytest.raises(ProviderRateLimitResetError) as caught:
            await router.call(messages=[])
        assert 0 < (caught.value.retry_at - datetime.now(timezone.utc)).total_seconds() <= 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    primary.pending = None
    primary.result = "recovered"
    assert await router.call(messages=[]) == "recovered"


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_terminal_adapter_classification_stops_fallback(method: str) -> None:
    primary, fallback = Member("work", NativeAuthError("private-native-error")), Member("fallback", "wrong")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="round-robin"))
    kwargs = {"messages": [], **({"tools": []} if method == "call_with_tools" else {})}
    with pytest.raises(ProviderReauthenticationRequiredError, match="work") as caught:
        await getattr(router, method)(**kwargs)
    assert "private-native-error" not in str(caught.value)


async def test_cooldown_logs_only_safe_classification_fields(caplog: pytest.LogCaptureFixture) -> None:
    primary, fallback = Member("work", QuotaError("synthetic-sensitive-body")), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    assert await router.call(messages=[]) == "fallback"
    assert "synthetic-sensitive-body" not in caplog.text
    assert "work" in caplog.text and "rate_limit" in caplog.text and "cooldown" in caplog.text


@pytest.mark.parametrize(
    ("delay", "cooldown_source"),
    [(None, "default"), (60.0, "provider_retry_after")],
)
async def test_cooldown_log_distinguishes_default_from_provider_retry_after(
    caplog: pytest.LogCaptureFixture, delay: float | None, cooldown_source: str
) -> None:
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    primary.delay = delay
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))

    assert await router.call(messages=[]) == "fallback"

    assert f"cooldown_source={cooldown_source}" in caplog.text
    assert "retry_after=60.000s" in caplog.text


async def test_cooldown_skip_log_includes_remaining_monotonic_seconds(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    caplog.set_level(logging.DEBUG, logger="hindsight_api.engine.multi_llm")
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    assert await router.call(messages=[]) == "fallback"

    caplog.clear()
    primary.result = "primary"
    now = 103.0
    assert await router.call(messages=[]) == "fallback"

    assert "state=cooldown remaining=7.000s" in caplog.text


async def test_non_replayable_probe_failure_recools_but_does_not_fail_over(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    assert await router.call(messages=[]) == "fallback"
    now = 110.0
    primary.result = OutputTooLongError("too long")
    with pytest.raises(OutputTooLongError):
        await router.call(messages=[])
    primary.result = "primary"
    assert await router.call(messages=[]) == "fallback"


async def test_cooldown_is_local_to_router_instance_and_batch_affinity_is_unchanged() -> None:
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    separate = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    selected = await router.batch_provider_impl()
    assert await router.call(messages=[]) == "fallback"
    primary.result = "primary"
    assert await router.call(messages=[]) == "fallback"
    assert await separate.call(messages=[]) == "primary"
    assert await router.batch_provider_impl() is selected
    assert await router.batch_provider_impl("primary") is selected


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_unclassified_failure_does_not_start_sticky_cooldown(method: str) -> None:
    primary, fallback = Member("primary", RuntimeError("temporary")), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    kwargs = {"messages": [], **({"tools": []} if method == "call_with_tools" else {})}
    assert await getattr(router, method)(**kwargs) == "fallback"
    primary.result = "primary"
    assert await getattr(router, method)(**kwargs) == "primary"


async def test_probe_success_cannot_clear_a_newer_inflight_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100.0
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", lambda: now)
    primary, fallback = Member("primary", QuotaError()), Member("fallback", "fallback")
    router = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode="failover"))
    old_entered, old_release = asyncio.Event(), asyncio.Event()
    probe_entered, probe_release = asyncio.Event(), asyncio.Event()

    async def older_request() -> str:
        old_entered.set()
        await old_release.wait()
        raise QuotaError("new quota observed by older request")

    async def probe() -> str:
        probe_entered.set()
        await probe_release.wait()
        return "probe-success"

    primary.pending = older_request
    older = asyncio.create_task(router.call(messages=[]))
    await old_entered.wait()
    primary.pending = None
    assert await router.call(messages=[]) == "fallback"
    now = 110.0
    primary.pending = probe
    probing = asyncio.create_task(router.call(messages=[]))
    await probe_entered.wait()
    try:
        primary.delay = 20
        old_release.set()
        assert await older == "fallback"
        assert await router.call(messages=[]) == "fallback"
    finally:
        old_release.set()
        probe_release.set()
        await asyncio.gather(older, probing)
    primary.pending = None
    primary.result = "primary"
    assert await router.call(messages=[]) == "fallback"
    now = 130.0
    assert await router.call(messages=[]) == "primary"


async def test_unlabelled_secondary_terminal_error_names_its_router_position() -> None:
    primary = Member("primary", RuntimeError("temporary"))
    secondary = Member("secondary", ProviderReauthenticationRequiredError("standalone primary label"))
    secondary.member_label = None
    router = MultiLLMProvider([primary, secondary], LLMStrategyConfig(mode="failover"))
    with pytest.raises(ProviderReauthenticationRequiredError, match="member-1"):
        await router.call(messages=[])
