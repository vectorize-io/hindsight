"""Unit tests for MultiLLMProvider routing strategies.

Deterministic: members are lightweight fakes that record calls and either return
a sentinel or raise a chosen exception. No real providers / network.
"""

import asyncio

import pytest

from hindsight_api.config import LLMMetadataRoute, LLMStrategyConfig
from hindsight_api.engine.llm_wrapper import OutputTooLongError
from hindsight_api.engine.multi_llm import (
    MultiLLMProvider,
    _should_failover,
    _WeightedRoundRobin,
)


class FakeMember:
    """Stands in for an LLMProvider member.

    ``behavior`` is a value to return, or an exception instance/class to raise.
    Each ``call`` / ``call_with_tools`` is recorded so tests can assert which
    member served (and how many times).
    """

    def __init__(self, name: str, behavior):
        self.provider = name
        self.model = f"{name}-model"
        self._provider_impl = object()
        self.behavior = behavior
        self.calls = 0
        self.verified = 0

    async def call(self, **kwargs):
        self.calls += 1
        return self._resolve()

    async def call_with_tools(self, **kwargs):
        self.calls += 1
        return self._resolve()

    async def verify_connection(self):
        self.verified += 1
        if isinstance(self.behavior, BaseException) or (
            isinstance(self.behavior, type) and issubclass(self.behavior, BaseException)
        ):
            raise self.behavior if isinstance(self.behavior, BaseException) else self.behavior()

    async def cleanup(self):
        pass

    def with_config(self, config, **kwargs):
        return self

    def _resolve(self):
        b = self.behavior
        if isinstance(b, BaseException):
            raise b
        if isinstance(b, type) and issubclass(b, BaseException):
            raise b()
        return b


def _failover(*members):
    return MultiLLMProvider(list(members), LLMStrategyConfig(mode="failover"))


def _round_robin(*members, weights=None):
    return MultiLLMProvider(list(members), LLMStrategyConfig(mode="round-robin", weights=weights))


def _metadata(*members):
    return MultiLLMProvider(
        list(members),
        LLMStrategyConfig(
            mode="metadata",
            routes=[LLMMetadataRoute(key="tags", value="sensitive", member=1)],
        ),
    )


# ── failover ─────────────────────────────────────────────────────────────────


async def test_failover_uses_primary_on_success():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    result = await _failover(a, b).call(messages=[])
    assert result == "RA"
    assert (a.calls, b.calls) == (1, 0)  # secondary untouched


async def test_failover_advances_only_after_member_raises():
    a = FakeMember("a", RuntimeError("primary down"))
    b = FakeMember("b", "RB")
    result = await _failover(a, b).call(messages=[])
    assert result == "RB"
    assert (a.calls, b.calls) == (1, 1)


async def test_failover_reraises_last_exception_when_all_fail():
    a = FakeMember("a", RuntimeError("first"))
    b = FakeMember("b", ValueError("last"))
    with pytest.raises(ValueError, match="last"):
        await _failover(a, b).call(messages=[])
    assert (a.calls, b.calls) == (1, 1)


async def test_call_with_tools_fails_over_too():
    a = FakeMember("a", RuntimeError("down"))
    b = FakeMember("b", "tools-ok")
    result = await _failover(a, b).call_with_tools(messages=[], tools=[])
    assert result == "tools-ok"
    assert (a.calls, b.calls) == (1, 1)


# ── error classification ──────────────────────────────────────────────────────


async def test_output_too_long_is_not_failed_over():
    a = FakeMember("a", OutputTooLongError("too long"))
    b = FakeMember("b", "RB")
    with pytest.raises(OutputTooLongError):
        await _failover(a, b).call(messages=[])
    assert b.calls == 0  # never tried the secondary


async def test_cancellation_propagates_without_failover():
    a = FakeMember("a", asyncio.CancelledError())
    b = FakeMember("b", "RB")
    with pytest.raises(asyncio.CancelledError):
        await _failover(a, b).call(messages=[])
    assert b.calls == 0


def test_should_failover_classification():
    assert _should_failover(RuntimeError()) is True
    assert _should_failover(ValueError()) is True
    assert _should_failover(OutputTooLongError("x")) is False
    assert _should_failover(asyncio.CancelledError()) is False
    assert _should_failover(KeyboardInterrupt()) is False
    assert _should_failover(SystemExit()) is False


# ── round-robin ───────────────────────────────────────────────────────────────


async def test_round_robin_rotates_start_member():
    a, b, c = FakeMember("a", "RA"), FakeMember("b", "RB"), FakeMember("c", "RC")
    rr = _round_robin(a, b, c)
    results = [await rr.call(messages=[]) for _ in range(3)]
    # Smooth WRR with uniform weights visits every member once per cycle.
    assert sorted(results) == ["RA", "RB", "RC"]
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


async def test_round_robin_falls_over_when_selected_member_fails():
    a = FakeMember("a", RuntimeError("a down"))
    b = FakeMember("b", "RB")
    # Two members, uniform weights: whichever is selected first, a failure must
    # still produce a successful result via the other member.
    rr = _round_robin(a, b)
    result = await rr.call(messages=[])
    assert result == "RB"


async def test_weighted_round_robin_honors_ratio():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    rr = _round_robin(a, b, weights=[3, 1])
    for _ in range(8):
        await rr.call(messages=[])
    # 3:1 over 8 requests → a served 6, b served 2.
    assert (a.calls, b.calls) == (6, 2)


# ── metadata routing ──────────────────────────────────────────────────────────


async def test_metadata_route_pins_matching_tag_to_secondary():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    routed = _metadata(a, b).with_config(object(), routing_metadata={"tags": ["sensitive"]})
    assert await routed.call(messages=[]) == "RB"
    assert (a.calls, b.calls) == (0, 1)


async def test_metadata_route_defaults_to_primary():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    routed = _metadata(a, b).with_config(object(), routing_metadata={"tags": ["public"]})
    assert await routed.call(messages=[]) == "RA"
    assert (a.calls, b.calls) == (1, 0)


async def test_metadata_route_does_not_fall_back_across_privacy_lanes():
    a = FakeMember("a", "RA")
    b = FakeMember("b", RuntimeError("sensitive lane down"))
    routed = _metadata(a, b).with_config(object(), routing_metadata={"tags": ["sensitive"]})
    with pytest.raises(RuntimeError, match="sensitive lane down"):
        await routed.call(messages=[])
    assert (a.calls, b.calls) == (0, 1)


def test_metadata_route_rejects_ambiguous_cross_member_match():
    a, b, c = FakeMember("a", "RA"), FakeMember("b", "RB"), FakeMember("c", "RC")
    router = MultiLLMProvider(
        [a, b, c],
        LLMStrategyConfig(
            mode="metadata",
            routes=[
                LLMMetadataRoute(key="metadata.classification", value="internal", member=1),
                LLMMetadataRoute(key="metadata.clearance", value="restricted", member=2),
            ],
        ),
    )

    with pytest.raises(ValueError, match="select multiple members"):
        router.with_config(
            object(),
            routing_metadata={
                "metadata.classification": ["internal"],
                "metadata.clearance": ["restricted"],
            },
        )
    assert (a.calls, b.calls, c.calls) == (0, 0, 0)


def test_metadata_routes_reject_conflicting_tag_members_at_construction():
    with pytest.raises(ValueError, match="routes with key 'tags' must all select the same member"):
        MultiLLMProvider(
            [FakeMember("a", "RA"), FakeMember("b", "RB"), FakeMember("c", "RC")],
            LLMStrategyConfig(
                mode="metadata",
                routes=[
                    LLMMetadataRoute(key="tags", value="internal", member=1),
                    LLMMetadataRoute(key="tags", value="sensitive", member=2),
                ],
            ),
        )


async def test_metadata_route_allows_multiple_matches_to_same_member():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    router = MultiLLMProvider(
        [a, b],
        LLMStrategyConfig(
            mode="metadata",
            routes=[
                LLMMetadataRoute(key="tags", value="internal", member=1),
                LLMMetadataRoute(key="tags", value="sensitive", member=1),
            ],
        ),
    )

    routed = router.with_config(object(), routing_metadata={"tags": ["internal", "sensitive"]})
    assert await routed.call(messages=[]) == "RB"
    assert (a.calls, b.calls) == (0, 1)


def test_weighted_scheduler_distribution():
    sched = _WeightedRoundRobin([5, 1])
    picks = [sched.next() for _ in range(6)]
    assert picks.count(0) == 5
    assert picks.count(1) == 1


# ── construction / delegation ─────────────────────────────────────────────────


def test_weights_length_must_match_members():
    with pytest.raises(ValueError, match="weights"):
        MultiLLMProvider(
            [FakeMember("a", "x"), FakeMember("b", "y")],
            LLMStrategyConfig(mode="round-robin", weights=[1, 1, 1]),
        )


def test_attribute_passthrough_to_primary():
    a, b = FakeMember("a", "RA"), FakeMember("b", "RB")
    multi = _failover(a, b)
    assert multi.provider == "a"  # primary
    assert multi.model == "a-model"
    assert multi._provider_impl is a._provider_impl


async def test_verify_connection_strict_primary_soft_secondary():
    # Primary down → raises (strict).
    down_primary = _failover(FakeMember("a", RuntimeError("down")), FakeMember("b", "RB"))
    with pytest.raises(RuntimeError):
        await down_primary.verify_connection()

    # Secondary down → swallowed (soft), primary still verified.
    a = FakeMember("a", "RA")
    b = FakeMember("b", RuntimeError("down"))
    await _failover(a, b).verify_connection()
    assert a.verified == 1 and b.verified == 1
