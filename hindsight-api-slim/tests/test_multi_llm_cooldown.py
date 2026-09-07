"""Deterministic cooldown routing tests; provider adapters and time are controlled."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from hindsight_api.config import LLMMetadataRoute, LLMStrategyConfig
from hindsight_api.engine.llm_interface import (
    LLMCooldownFailure,
    LLMFailureClassification,
    LLMTerminalFailure,
    OutputTooLongError,
    ProviderRateLimitResetError,
)
from hindsight_api.engine.multi_llm import MultiLLMProvider


class QuotaError(RuntimeError):
    pass


class NativeAuthError(RuntimeError):
    pass


class Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


class Member:
    """Provider adapter exercising the router's public member seam."""

    def __init__(
        self,
        name: str,
        *results: str | BaseException,
        delay: float | None = 10.0,
        max_backoff: float | None = None,
    ) -> None:
        self.provider = "test"
        self.model = name
        self.member_label = name
        self.max_backoff = max_backoff
        self.results = list(results)
        self.delay = delay
        self.pending: Callable[[], Awaitable[Any]] | None = None
        self.calls = 0
        self.call: Callable[..., Awaitable[Any]] = self._call

    def classify_failure(self, exc: BaseException) -> LLMFailureClassification | None:
        if isinstance(exc, NativeAuthError):
            return LLMTerminalFailure()
        if isinstance(exc, QuotaError):
            return LLMCooldownFailure(retry_after_seconds=self.delay)
        return None

    async def _call(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.pending is not None:
            return await self.pending()
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, BaseException):
            raise result
        return result

    async def call_with_tools(self, **kwargs: Any) -> Any:
        return await self.call(**kwargs)

    async def batch_provider_impl(self, account_key: str | None = None) -> "Member | None":
        return self if account_key is None or account_key == self.model else None


def _router(*members: Member, mode: str = "failover") -> MultiLLMProvider:
    return MultiLLMProvider(cast(Any, list(members)), LLMStrategyConfig(mode=mode))


def _metadata_router(*members: Member) -> MultiLLMProvider:
    strategy = LLMStrategyConfig(
        mode="metadata",
        routes=[LLMMetadataRoute(key="classification", value="sensitive", member=1)],
    )
    return MultiLLMProvider(cast(Any, list(members)), strategy)


async def test_metadata_primary_short_cooldown_waits_inline_without_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    primary = Member("primary", QuotaError("primary quota"), "recovered", delay=1.0)
    secondary = Member("metadata-secondary", "forbidden")
    router = _metadata_router(primary, secondary)

    assert await router.call(messages=[], max_backoff=5.0) == "recovered"
    assert clock.sleeps == [1.0]
    assert primary.calls == 2
    assert secondary.calls == 0


async def test_metadata_primary_long_cooldown_defers_without_secondary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError("primary quota"), delay=10.0)
    secondary = Member("metadata-secondary", "forbidden")
    router = _metadata_router(primary, secondary)

    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=5.0)

    assert primary.calls == 1
    assert secondary.calls == 0


async def test_metadata_primary_pending_probe_has_one_owner_and_bounded_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    initial = QuotaError("primary quota")
    primary = Member("primary", initial, delay=1.0)
    secondary = Member("metadata-secondary", "forbidden")
    router = _metadata_router(primary, secondary)

    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=0.5)
    clock.now = 101.0

    entered = asyncio.Event()
    release = asyncio.Event()

    async def pending_probe() -> str:
        entered.set()
        await release.wait()
        return "recovered"

    primary.pending = pending_probe
    owner = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    try:
        with pytest.raises(QuotaError) as caught:
            await router.call(messages=[], max_backoff=1.0)
        assert caught.value is initial
        assert clock.sleeps == [0.05]
        assert primary.calls == 2
        assert secondary.calls == 0
    finally:
        release.set()
        assert await owner == "recovered"


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_cooldown_skips_until_one_half_open_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError(), "recovered")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    kwargs = {"messages": [], **({"tools": []} if method == "call_with_tools" else {})}

    assert await getattr(router, method)(**kwargs) == "fallback"
    assert await getattr(router, method)(**kwargs) == "fallback"
    assert primary.calls == 1
    clock.now = 110.0
    assert await getattr(router, method)(**kwargs) == "recovered"
    assert primary.calls == 2
    assert router._states[0].cooldown_exception is None


async def test_only_one_concurrent_half_open_probe_uses_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError())
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    assert await router.call(messages=[]) == "fallback"
    clock.now = 110.0

    entered = asyncio.Event()
    release = asyncio.Event()

    async def probe() -> str:
        entered.set()
        await release.wait()
        return "recovered"

    primary.pending = probe
    probing = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    try:
        assert await router.call_with_tools(messages=[], tools=[]) == "fallback"
        assert primary.calls == 2
    finally:
        release.set()
        assert await probing == "recovered"


@pytest.mark.parametrize("failure", [RuntimeError("temporary"), QuotaError("quota")])
async def test_failed_probe_recools(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError(), failure, "recovered")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    assert await router.call(messages=[]) == "fallback"
    clock.now = 110.0
    assert await router.call(messages=[]) == "fallback"
    clock.now = 119.0
    assert await router.call(messages=[]) == "fallback"
    clock.now = 170.0 if not isinstance(failure, QuotaError) else 120.0
    assert await router.call(messages=[]) == "recovered"


async def test_stale_inflight_success_cannot_clear_newer_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", "unused")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_success() -> str:
        entered.set()
        await release.wait()
        return "old-success"

    primary.pending = delayed_success
    older = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    primary.pending = None
    primary.results = [QuotaError()]
    try:
        assert await router.call(messages=[]) == "fallback"
    finally:
        release.set()
        assert await older == "old-success"
    primary.results = ["should-stay-cooled"]
    assert await router.call(messages=[]) == "fallback"


async def test_cancelled_probe_releases_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError())
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    assert await router.call(messages=[]) == "fallback"
    clock.now = 110.0
    entered = asyncio.Event()

    async def pending() -> str:
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    primary.pending = pending
    task = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not router._states[0].probing
    primary.pending = None
    primary.results = ["recovered"]
    assert await router.call(messages=[]) == "recovered"


@pytest.mark.parametrize("method", ["call", "call_with_tools"])
async def test_terminal_classification_reraises_exact_object_without_fallback_or_cooldown(method: str) -> None:
    terminal = NativeAuthError("provider remediation")
    primary = Member("primary", terminal, "recovered")
    fallback = Member("fallback", "wrong")
    router = _router(primary, fallback)
    kwargs = {"messages": [], **({"tools": []} if method == "call_with_tools" else {})}

    with pytest.raises(NativeAuthError) as caught:
        await getattr(router, method)(**kwargs)
    assert caught.value is terminal
    assert fallback.calls == 0
    assert await getattr(router, method)(**kwargs) == "recovered"


async def test_unclassified_error_keeps_generic_failover_without_sticky_cooldown() -> None:
    primary = Member("primary", RuntimeError("temporary"), "primary")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    assert await router.call(messages=[]) == "fallback"
    assert await router.call(messages=[]) == "primary"


async def test_long_reset_defers_only_when_strictly_greater_than_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError(), delay=10.0)
    router = _router(primary)
    before = datetime.now(UTC)
    with pytest.raises(ProviderRateLimitResetError) as caught:
        await router.call(messages=[], max_backoff=5.0)
    assert caught.value.retry_at > before
    assert primary.calls == 1


async def test_reset_equal_to_max_backoff_waits_inline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    primary = Member("primary", QuotaError(), "recovered", delay=5.0)
    router = _router(primary)

    assert await router.call(messages=[], max_backoff=5.0) == "recovered"
    assert clock.sleeps == [5.0]
    assert primary.calls == 2


async def test_short_reset_does_not_reissue_without_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    first = QuotaError("first")
    second = QuotaError("second")
    primary = Member("primary", first, second, delay=1.0)
    router = _router(primary)

    with pytest.raises(QuotaError) as caught:
        await router.call(messages=[], max_backoff=5.0)
    assert caught.value is second
    assert primary.calls == 2
    assert clock.sleeps == [1.0]


async def test_followers_behind_an_indefinitely_pending_probe_raise_saved_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    initial = QuotaError("initial")
    primary = Member("primary", initial, delay=1.0)
    router = _router(primary)

    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=0.5)
    clock.now = 101.0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pending() -> str:
        entered.set()
        await release.wait()
        return "recovered"

    primary.pending = pending
    owner = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    try:
        followers = await asyncio.gather(
            *(router.call(messages=[], max_backoff=1.0) for _ in range(3)),
            return_exceptions=True,
        )
        assert followers == [initial, initial, initial]
        assert primary.calls == 2
    finally:
        release.set()
        assert await owner == "recovered"


@pytest.mark.parametrize("inline_budget_used", [False, True], ids=["unused", "consumed"])
def test_owner_success_between_follower_skip_and_snapshot_keeps_routing_bounded(
    monkeypatch: pytest.MonkeyPatch,
    inline_budget_used: bool,
) -> None:
    clock = Clock()
    real_sleep = asyncio.sleep
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    initial = QuotaError("initial")
    primary = Member("primary", initial, delay=1.0)
    router = _router(primary)

    with pytest.raises(ProviderRateLimitResetError):
        asyncio.run(router.call(messages=[], max_backoff=0.5))

    probe_entered = threading.Event()
    release_probe = threading.Event()
    owner_done = threading.Event()
    owner_results: list[str] = []
    owner_errors: list[BaseException] = []

    async def probe() -> str:
        probe_entered.set()
        while not release_probe.is_set():
            await real_sleep(0.001)
        return "owner-recovered"

    primary.pending = probe

    def run_owner() -> None:
        try:
            owner_results.append(asyncio.run(router.call(messages=[])))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            owner_errors.append(exc)
        finally:
            owner_done.set()

    owner = threading.Thread(target=run_owner)
    owner_started = False

    def start_owner() -> None:
        nonlocal owner_started
        if not owner_started:
            owner_started = True
            owner.start()

    class SnapshotGate:
        """Release the owner immediately before the follower's snapshot lock."""

        def __init__(self, snapshot_acquisition: int) -> None:
            self._lock = threading.Lock()
            self._snapshot_acquisition = snapshot_acquisition
            self._follower = threading.get_ident()
            self._follower_acquisitions = 0

        def __enter__(self) -> "SnapshotGate":
            if threading.get_ident() == self._follower:
                self._follower_acquisitions += 1
                if self._follower_acquisitions == self._snapshot_acquisition:
                    release_probe.set()
                    assert owner_done.wait(timeout=2.0), "probe owner did not recover before the snapshot"
            self._lock.acquire()
            return self

        def __exit__(self, *args: Any) -> None:
            self._lock.release()

    # Unused budget: follower skip then snapshot are acquisitions 1 and 2.
    # Consumed budget: its first cooldown skip/snapshot consume 1 and 2, then
    # the post-sleep probe skip/snapshot are acquisitions 3 and 4.
    router._state_lock = SnapshotGate(4 if inline_budget_used else 2)  # type: ignore[assignment]
    if inline_budget_used:
        clock.now = 100.0

        async def sleep_then_start_owner(delay: float) -> None:
            await clock.sleep(delay)
            start_owner()
            while not probe_entered.is_set():
                await real_sleep(0)

        monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", sleep_then_start_owner)
    else:
        clock.now = 101.0
        start_owner()
        assert probe_entered.wait(timeout=2.0), "probe owner never entered"

    try:
        if inline_budget_used:
            with pytest.raises(QuotaError) as caught:
                asyncio.run(router.call(messages=[], max_backoff=1.0))
            assert caught.value is initial
            assert clock.sleeps == [1.0]
        else:
            assert asyncio.run(router.call(messages=[], max_backoff=1.0)) == "owner-recovered"
            assert clock.sleeps == []
    finally:
        release_probe.set()
        if owner_started:
            owner.join(timeout=2.0)
            assert not owner.is_alive(), "probe owner was stranded"

    assert owner_errors == []
    assert owner_results == ["owner-recovered"]


async def test_follower_uses_saved_failure_from_its_original_round_robin_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    primary_failure = QuotaError("primary")
    fallback_failure = QuotaError("fallback")
    primary = Member("primary", primary_failure, delay=1.0)
    fallback = Member("fallback", fallback_failure, delay=1.0)
    router = _router(primary, fallback, mode="round-robin")

    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=0.5)
    clock.now = 101.0

    primary_entered = asyncio.Event()
    fallback_entered = asyncio.Event()
    release = asyncio.Event()

    async def pending_primary() -> str:
        primary_entered.set()
        await release.wait()
        return "primary-recovered"

    async def pending_fallback() -> str:
        fallback_entered.set()
        await release.wait()
        return "fallback-recovered"

    primary.pending = pending_primary
    fallback.pending = pending_fallback
    # The next two round-robin requests claim fallback and primary respectively.
    fallback_owner = asyncio.create_task(router.call(messages=[]))
    await fallback_entered.wait()
    primary_owner = asyncio.create_task(router.call(messages=[]))
    await primary_entered.wait()
    try:
        # This request starts at fallback. Both members remain behind another
        # request's probe, so its saved cause must come from fallback even
        # though primary has the lower member index.
        with pytest.raises(QuotaError) as caught:
            await router.call(messages=[], max_backoff=1.0)
        assert caught.value is fallback_failure
    finally:
        release.set()
        assert await fallback_owner == "fallback-recovered"
        assert await primary_owner == "primary-recovered"


async def test_follower_raises_latest_failure_after_short_probe_recools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    real_sleep = asyncio.sleep
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    initial = QuotaError("initial")
    recool = QuotaError("recool")
    primary = Member("primary", initial, delay=1.0)
    router = _router(primary)

    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=0.5)
    clock.now = 101.0
    entered = asyncio.Event()
    release = asyncio.Event()

    async def probe() -> str:
        entered.set()
        await release.wait()
        raise recool

    async def release_probe_and_wait_for_recool(delay: float) -> None:
        clock.sleeps.append(delay)
        clock.now += delay
        release.set()
        while router._states[0].probing:
            await real_sleep(0)

    primary.pending = probe
    owner = asyncio.create_task(router.call(messages=[]))
    await entered.wait()
    monkeypatch.setattr(
        "hindsight_api.engine.multi_llm.asyncio.sleep",
        release_probe_and_wait_for_recool,
    )
    with pytest.raises(QuotaError) as caught:
        await router.call(messages=[], max_backoff=5.0)
    assert caught.value is recool
    with pytest.raises(QuotaError) as owner_failure:
        await owner
    assert owner_failure.value is recool
    assert primary.calls == 2


async def test_cooldown_extension_during_inline_wait_rechecks_the_defer_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    original = QuotaError("original")
    extended = QuotaError("extended")
    primary = Member("primary", original, delay=1.0)
    router = _router(primary)

    async def extend_cooldown(delay: float) -> None:
        await clock.sleep(delay)
        with router._state_lock:
            state = router._states[0]
            state.cooldown_until = clock.now + 10.0
            state.cooldown_exception = extended
            state.generation += 1

    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", extend_cooldown)
    with pytest.raises(ProviderRateLimitResetError):
        await router.call(messages=[], max_backoff=5.0)
    assert primary.calls == 1
    assert clock.sleeps == [1.0]


async def test_round_robin_starts_each_direct_request_with_the_next_member() -> None:
    primary = Member("primary", "primary")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback, mode="round-robin")

    assert await router.call(messages=[]) == "primary"
    assert await router.call(messages=[]) == "fallback"
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_inline_wait_preserves_original_member_order(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    monkeypatch.setattr("hindsight_api.engine.multi_llm.asyncio.sleep", clock.sleep)
    events: list[str] = []
    primary = Member("primary", QuotaError(), "primary", delay=2.0)
    fallback = Member("fallback", QuotaError(), "fallback", delay=4.0)
    primary_call = primary.call
    fallback_call = fallback.call

    async def call_primary(**kwargs: Any) -> Any:
        events.append("primary")
        return await primary_call(**kwargs)

    async def call_fallback(**kwargs: Any) -> Any:
        events.append("fallback")
        return await fallback_call(**kwargs)

    primary.call = call_primary  # type: ignore[method-assign]
    fallback.call = call_fallback  # type: ignore[method-assign]
    assert await _router(primary, fallback).call(messages=[], max_backoff=5.0) == "primary"
    assert events == ["primary", "fallback", "primary"]


async def test_probe_failure_recools_but_non_failover_error_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("hindsight_api.engine.multi_llm.monotonic", clock.monotonic)
    primary = Member("primary", QuotaError(), OutputTooLongError("too long"))
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    assert await router.call(messages=[]) == "fallback"
    clock.now = 110.0
    with pytest.raises(OutputTooLongError):
        await router.call(messages=[])
    assert fallback.calls == 1
    primary.results = ["still cooling"]
    assert await router.call(messages=[]) == "fallback"


async def test_cooldown_is_router_local_and_batch_affinity_is_unchanged() -> None:
    primary = Member("primary", QuotaError(), "primary")
    fallback = Member("fallback", "fallback")
    router = _router(primary, fallback)
    separate = _router(primary, fallback)
    selected = await router.batch_provider_impl()

    assert await router.call(messages=[]) == "fallback"
    assert await separate.call(messages=[]) == "primary"
    assert await router.batch_provider_impl() is selected
    assert await router.batch_provider_impl("primary") is selected


async def test_classified_log_redacts_provider_exception_and_uses_label(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = Member("primary", QuotaError("synthetic-sensitive-body"))
    fallback = Member("fallback", "fallback")
    assert await _router(primary, fallback).call(messages=[]) == "fallback"
    assert "synthetic-sensitive-body" not in caplog.text
    assert "label=primary" in caplog.text
