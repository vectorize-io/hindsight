"""Tests for per-operation admission control.

The property under test is the one the engine's existing semaphores do NOT have:
a request that cannot get a permit within its deadline is *refused*, rather than
queueing until the caller has given up.
"""

import asyncio

import pytest

from hindsight_api.api.admission import (
    AdmissionController,
    AdmissionRejected,
    LaneConfig,
)


def _controller(**lanes: LaneConfig) -> AdmissionController:
    return AdmissionController(dict(lanes))


async def test_admits_up_to_the_limit():
    controller = _controller(recall=LaneConfig(max_in_flight=2, max_wait_seconds=1.0))
    async with controller.admit("recall"):
        async with controller.admit("recall"):
            assert controller.stats()["recall"].in_flight == 2
    assert controller.stats()["recall"].in_flight == 0
    assert controller.stats()["recall"].admitted == 2


async def test_rejects_when_full_rather_than_queueing_forever():
    """The whole point: a bounded wait, then 503 -- not an unbounded queue."""
    controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=0.05))

    async with controller.admit("recall"):
        started = asyncio.get_running_loop().time()
        with pytest.raises(AdmissionRejected) as excinfo:
            async with controller.admit("recall"):
                pytest.fail("should not have been admitted")
        waited = asyncio.get_running_loop().time() - started

    # Refused at roughly the deadline, not after the holder finished.
    assert 0.04 <= waited < 1.0
    assert excinfo.value.lane == "recall"
    assert excinfo.value.limit == 1
    assert excinfo.value.retry_after_seconds >= 1
    assert controller.stats()["recall"].rejected == 1


async def test_permit_is_released_and_a_waiter_then_proceeds():
    controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=2.0))
    order: list[str] = []

    async def holder():
        async with controller.admit("recall"):
            order.append("holder-in")
            await asyncio.sleep(0.05)
        order.append("holder-out")

    async def waiter():
        await asyncio.sleep(0.01)
        async with controller.admit("recall"):
            order.append("waiter-in")

    await asyncio.gather(holder(), waiter())
    assert order == ["holder-in", "holder-out", "waiter-in"]


async def test_permit_released_when_the_body_raises():
    """A failing request must not leak its permit -- otherwise the lane shrinks."""
    controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=0.05))

    with pytest.raises(ValueError):
        async with controller.admit("recall"):
            raise ValueError("boom")

    assert controller.stats()["recall"].in_flight == 0
    # The permit came back, so the next request is admitted.
    async with controller.admit("recall"):
        pass
    assert controller.stats()["recall"].admitted == 2


async def test_lanes_are_independent():
    """A saturated lane must not refuse a different operation."""
    controller = _controller(
        recall=LaneConfig(max_in_flight=1, max_wait_seconds=0.05),
        reflect=LaneConfig(max_in_flight=1, max_wait_seconds=0.05),
    )
    async with controller.admit("recall"):
        # reflect has its own permit and is unaffected by recall being full.
        async with controller.admit("reflect"):
            pass
        with pytest.raises(AdmissionRejected):
            async with controller.admit("recall"):
                pass


async def test_unknown_operation_falls_through_ungated():
    controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=0.05))
    async with controller.admit("not_a_lane"):
        pass  # no exception, no accounting


async def test_lane_disabled_by_zero_in_flight():
    """0 is the documented kill switch and must not gate anything."""
    controller = _controller(recall=LaneConfig(max_in_flight=0, max_wait_seconds=1.0))
    async with controller.admit("recall"):
        async with controller.admit("recall"):
            pass
    assert controller.stats()["recall"].admitted == 0


async def test_zero_wait_refuses_immediately_when_full():
    controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=0.0))
    async with controller.admit("recall"):
        started = asyncio.get_running_loop().time()
        with pytest.raises(AdmissionRejected):
            async with controller.admit("recall"):
                pass
        assert asyncio.get_running_loop().time() - started < 0.05


class TestDerivedDefaults:
    """The in-flight default is derived from the CPU budget, not hardcoded."""

    def test_explicit_value_always_wins(self):
        from hindsight_api.config import admission_in_flight_for

        assert admission_in_flight_for(7, per_core=16, workers=4) == 7

    def test_derives_from_cores_divided_by_workers(self, monkeypatch):
        from hindsight_api import config as config_mod

        monkeypatch.setattr("hindsight_api._thread_limits.available_cpu_count", lambda: 8)
        # 8 cores / 2 workers = 4 cores each -> 4 x 16
        assert config_mod.admission_in_flight_for(0, per_core=16, workers=2) == 64

    def test_floored_at_per_core_on_a_fractional_core_box(self, monkeypatch):
        """A worker with less than a core still needs enough depth to stay busy
        while requests wait on embeddings or an LLM."""
        from hindsight_api import config as config_mod

        monkeypatch.setattr("hindsight_api._thread_limits.available_cpu_count", lambda: 1)
        assert config_mod.admission_in_flight_for(0, per_core=16, workers=4) == 16

    def test_reference_shape_matches_the_measured_default(self, monkeypatch):
        """2 vCPU / 2 workers is the shape the 16 was calibrated on."""
        from hindsight_api import config as config_mod

        monkeypatch.setattr("hindsight_api._thread_limits.available_cpu_count", lambda: 2)
        assert config_mod.admission_in_flight_for(0, per_core=16, workers=2) == 16


class TestAbandonedWhileQueued:
    """A queued request whose client disconnected must free its place at once.

    This is what makes a patient deadline affordable: without it a 30s wait means
    30s of held connections for callers that may already be gone.
    """

    async def test_disconnect_releases_the_waiter_immediately(self):
        from hindsight_api.api.admission import AdmissionAbandoned
        from hindsight_api.cancellation import CancellationToken

        controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=30.0))
        token = CancellationToken()

        async with controller.admit("recall"):
            started = asyncio.get_running_loop().time()

            async def disconnect_soon():
                await asyncio.sleep(0.05)
                token.cancel("client disconnected")

            async def queued():
                async with controller.admit("recall", abandoned=token):
                    pytest.fail("should not have been admitted")

            disconnector = asyncio.create_task(disconnect_soon())
            with pytest.raises(AdmissionAbandoned):
                await queued()
            waited = asyncio.get_running_loop().time() - started
            await disconnector

        # Gave up on disconnect, not after the 30s deadline.
        assert waited < 1.0

    async def test_abandoned_waiter_does_not_consume_a_permit(self):
        """The permit must still be there for the next real caller."""
        from hindsight_api.api.admission import AdmissionAbandoned
        from hindsight_api.cancellation import CancellationToken

        controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=5.0))
        token = CancellationToken()
        token.cancel("gone before it even queued")

        with pytest.raises(AdmissionAbandoned):
            async with controller.admit("recall", abandoned=token):
                pass

        # Lane is intact: a live caller is admitted straight away.
        async with controller.admit("recall"):
            assert controller.stats()["recall"].in_flight == 1
        assert controller.stats()["recall"].in_flight == 0

    async def test_live_client_still_admitted_normally(self):
        """An un-cancelled token must not change the happy path."""
        from hindsight_api.cancellation import CancellationToken

        controller = _controller(recall=LaneConfig(max_in_flight=1, max_wait_seconds=1.0))
        async with controller.admit("recall", abandoned=CancellationToken()):
            assert controller.stats()["recall"].in_flight == 1
        assert controller.stats()["recall"].admitted == 1

    def test_negative_disables_the_lane(self, monkeypatch):
        """0 means "derive", so the kill switch has to be a negative value."""
        from hindsight_api import config as config_mod

        monkeypatch.setattr("hindsight_api._thread_limits.available_cpu_count", lambda: 8)
        assert config_mod.admission_in_flight_for(-1, per_core=16, workers=1) == 0
