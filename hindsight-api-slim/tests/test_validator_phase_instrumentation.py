"""Every operation-validator hook is timed, and none of them had to be named to get there.

The validator runs OUTSIDE the operation's own timer — `validate_*` before the work starts,
`on_*_complete` after it ends — while a recall's ``[phases]`` line measures only the inner search.
So a validator that reaches a database is latency nothing accounts for, and the ``[phases]`` line
still reads as complete, which is what makes the gap easy to miss.

The interface has nineteen hooks. These assert the STRUCTURAL property — every one is wrapped —
rather than checking the handful that exist today, because the failure this guards against is a
twentieth hook added later with no timing and nobody noticing.
"""

import inspect

import pytest

from hindsight_api.extensions.operation_validator import OperationValidatorExtension
from hindsight_api.extensions.validator_instrumentation import instrument_operation_validator


class _Collector:
    def __init__(self):
        self.recorded: list[tuple[str, str, float]] = []

    def record_validator_phase(self, operation: str, hook: str, seconds: float) -> None:
        self.recorded.append((operation, hook, seconds))


@pytest.fixture
def collector(monkeypatch):
    c = _Collector()
    import hindsight_api.metrics as m

    monkeypatch.setattr(m, "get_metrics_collector", lambda: c)
    return c


class _Validator(OperationValidatorExtension):
    """Implements only what the interface makes abstract and inherits the rest — the shape of a
    real extension, which overrides a couple of hooks and leaves the other seventeen defaulted."""

    name = "test-validator"

    async def validate_retain(self, ctx):
        return None

    async def validate_recall(self, ctx):
        return None

    async def validate_reflect(self, ctx):
        return None


def test_every_hook_on_the_interface_is_instrumented():
    """The property, not a list. A hook added to the interface is covered the day it exists.

    "Hook" is defined HERE, independently: every public coroutine method the interface declares.
    Asking the instrumentation's own predicate what counts would make this circular -- narrow the
    predicate and the expectation narrows with it, so the test keeps passing while coverage
    shrinks. It did exactly that until this was rewritten.
    """
    v = instrument_operation_validator(_Validator({}))

    hooks = {
        n
        for n in dir(OperationValidatorExtension)
        if not n.startswith("_") and inspect.iscoroutinefunction(getattr(OperationValidatorExtension, n, None))
    }
    assert len(hooks) > 10, f"only found {len(hooks)} hooks — has the interface moved?"

    missed = [n for n in sorted(hooks) if not hasattr(getattr(v, n), "__wrapped__")]
    assert missed == [], f"these hooks are not timed: {missed}"


@pytest.mark.asyncio
async def test_a_hook_records_its_operation_and_side(collector):
    v = instrument_operation_validator(_Validator({}))

    await v.validate_recall(None)
    await v.on_recall_complete(None)

    assert [(op, hook) for op, hook, _ in collector.recorded] == [
        ("recall", "pre"),
        ("recall", "post"),
    ], "pre and post must not share a series: they fail differently and are fixed differently"


@pytest.mark.asyncio
async def test_a_rejecting_hook_is_still_timed(collector):
    """The 402 path. A hook that refuses is the whole cost the caller pays, so leaving it
    unmeasured hides exactly the request someone complains about."""

    class _Rejects(_Validator):
        async def validate_retain(self, ctx):
            raise RuntimeError("insufficient credits")

    v = instrument_operation_validator(_Rejects({}))
    with pytest.raises(RuntimeError):
        await v.validate_retain(None)

    assert [(op, hook) for op, hook, _ in collector.recorded] == [("retain", "pre")]


@pytest.mark.asyncio
async def test_an_override_is_wrapped_not_bypassed(collector):
    """A real extension overrides a couple of hooks. Wrapping must reach the OVERRIDE, not the
    interface default it replaced, or the one hook that does work is the one not measured."""
    calls = []

    class _Overrides(_Validator):
        async def validate_recall(self, ctx):
            calls.append("mine")
            return None

    v = instrument_operation_validator(_Overrides({}))
    await v.validate_recall(None)

    assert calls == ["mine"], "the wrapper replaced the override instead of wrapping it"
    assert [(op, hook) for op, hook, _ in collector.recorded] == [("recall", "pre")]


def test_wrapping_is_idempotent():
    """Two engines over one extension must not double-count."""
    v = _Validator({})
    instrument_operation_validator(v)
    once = v.validate_recall
    instrument_operation_validator(v)
    assert v.validate_recall is once, "a second wrap would double-count every hook"


def test_none_is_left_alone():
    assert instrument_operation_validator(None) is None


@pytest.mark.asyncio
async def test_a_broken_collector_cannot_break_the_request(monkeypatch):
    """Instrumentation must never be the thing that fails the operation it measures."""
    import hindsight_api.metrics as m

    monkeypatch.setattr(m, "get_metrics_collector", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    v = instrument_operation_validator(_Validator({}))
    await v.validate_recall(None)  # must not raise


def test_non_hooks_are_untouched():
    """Only coroutine hooks are wrapped; ordinary attributes and sync helpers keep their identity."""
    v = _Validator({})
    before = v.name
    instrument_operation_validator(v)
    assert v.name == before
    assert not inspect.iscoroutinefunction(getattr(v, "name", None))


@pytest.mark.asyncio
async def test_an_unwrappable_validator_does_not_break_construction(collector):
    """The extension is loaded from an env var and is someone else's class. Instrumentation must
    degrade to "not timed" rather than fail the engine that takes it."""

    class _Unhashable(_Validator):
        __hash__ = None  # type: ignore[assignment]

    v = _Unhashable({})
    assert instrument_operation_validator(v) is v

    await v.validate_recall(None)  # still callable, simply untimed
    assert collector.recorded == []
