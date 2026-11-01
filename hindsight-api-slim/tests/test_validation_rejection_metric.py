"""A validator refusing a synchronous retain is not a failed retain.

The HTTP retain route runs the engine call inside ``metrics.record_operation``; a
validator rejection propagates through it as OperationValidationError and the route
returns the validator's status code. The operation counter must record that as
``success="rejected"``, or one client that keeps retaining after it has been refused
drives the retain failure rate to its share of the traffic. A real failure on the same
route must still be recorded as ``success="false"``.
"""

from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    ValidationResult,
)
from hindsight_api.metrics import MetricsCollector


class _RefusesRetain(OperationValidatorExtension):
    def __init__(self):
        super().__init__({})

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.reject("quota exceeded", status_code=402)

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


class _BreaksRetain(_RefusesRetain):
    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        raise RuntimeError("validator backend unreachable")


@pytest.fixture
def collector():
    """A real MetricsCollector over mock instruments, served to the HTTP layer."""
    meter = MagicMock()
    meter.create_histogram.side_effect = lambda *a, **k: MagicMock()
    meter.create_counter.side_effect = lambda *a, **k: MagicMock()
    with patch("hindsight_api.metrics.get_meter", return_value=meter):
        collector = MetricsCollector()
    with patch("hindsight_api.api.http.get_metrics_collector", return_value=collector):
        yield collector


def _retain_outcomes(collector) -> list[str]:
    return [
        call.args[1]["success"]
        for call in collector.operation_total.add.call_args_list
        if call.args[1]["operation"] == "retain" and call.args[1]["source"] == "api"
    ]


@pytest.mark.asyncio
async def test_sync_retain_refused_by_validator_is_recorded_as_rejected(api_client, memory, collector):
    memory._operation_validator = _RefusesRetain()

    resp = await api_client.post(
        "/v1/default/banks/rejected-metric-bank/memories",
        json={"items": [{"content": "Alice likes tea."}], "async": False},
    )

    assert resp.status_code == 402
    assert _retain_outcomes(collector) == ["rejected"]


@pytest.mark.asyncio
async def test_sync_retain_that_fails_is_still_recorded_as_a_failure(api_client, memory, collector):
    memory._operation_validator = _BreaksRetain()

    resp = await api_client.post(
        "/v1/default/banks/failed-metric-bank/memories",
        json={"items": [{"content": "Alice likes tea."}], "async": False},
    )

    assert resp.status_code >= 500
    assert _retain_outcomes(collector) == ["false"]
