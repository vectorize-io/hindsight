"""Time every operation-validator hook, without naming any of them.

The validator runs OUTSIDE the operation's own timer: `validate_*` before the work starts,
`on_*_complete` after it ends. A recall's ``[phases]`` line measures the inner search, so whatever
the validator does is not in it — and the line reads as complete (``accounted=427ms of 433ms``),
which is what makes the gap easy to miss rather than obvious.

That is not a small omission. A validator may reach a database on every hook: the credits
validator reads an org row and a pricing table on the control pool, uncached, before AND after,
then writes the charge. All of it inline, none of it visible.

**Wrapped once, here, rather than timed at each call site.** The interface has nineteen hooks and
the engine calls them from many places; hand-instrumenting them means the next hook is added
without timing and nobody notices, which is exactly how this gap appeared. Wrapping the instance
means a new hook is covered the day it exists, and a hook the engine forgets to call is visible
by its absence rather than by its silence.

The wrapper is applied to the INSTANCE, not by subclassing or proxying, so `isinstance` checks,
attribute access and any non-hook methods behave exactly as before.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import weakref
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Which validators have been wrapped. A WeakSet rather than a flag ON the instance: the extension
#: is someone else's object, and instrumentation should not leave an attribute on it that its own
#: code (or a serializer, or a test asserting on its shape) can trip over. Weak, so wrapping does
#: not keep a discarded extension alive.
_INSTRUMENTED: "weakref.WeakSet[Any]" = weakref.WeakSet()

#: Hooks are recognised by shape, not by a list: an async method whose name says it validates or
#: reports completion. A list would be one more thing to update, which is the failure this module
#: exists to prevent.
_HOOK_PREFIXES = ("validate_", "on_", "precheck", "filter_")


def _is_hook(name: str, attr: Any) -> bool:
    if name.startswith("_"):
        return False
    if hasattr(attr, "__wrapped__"):  # already timed (a validator wrapped without being tracked)
        return False
    if not any(name == p or name.startswith(p) for p in _HOOK_PREFIXES):
        return False
    return inspect.iscoroutinefunction(attr)


def instrument_operation_validator(validator: T) -> T:
    """Wrap every hook on ``validator`` so its duration is recorded. Returns the same object.

    Safe to call on None (returns None) and idempotent: a validator already wrapped is left
    alone, so constructing two engines over one extension does not double-count.
    """
    if validator is None:
        return validator
    # Instrumenting must never be the thing that breaks the engine it measures: the extension is
    # loaded from an env var and is someone else's class, so an unhashable, weakref-less or
    # slotted one degrades to "not timed" rather than failing construction.
    try:
        if validator in _INSTRUMENTED:
            return validator
    except TypeError:  # unhashable extension -- cannot be tracked, so cannot be wrapped safely
        logger.debug("operation validator is not hashable; hooks left untimed")
        return validator

    wrapped = 0
    for name in dir(validator):
        try:
            attr = getattr(validator, name)
        except Exception:  # a property that raises is not a hook
            continue
        if not _is_hook(name, attr):
            continue
        try:
            setattr(validator, name, _timed(name, attr))
        except (AttributeError, TypeError):  # __slots__ or a read-only attribute
            logger.debug("could not instrument hook %s", name)
            continue
        wrapped += 1

    try:
        _INSTRUMENTED.add(validator)
    except TypeError:  # not weak-referenceable: wrapping stands, idempotence is by __wrapped__
        pass
    logger.debug("instrumented %d operation-validator hooks on %s", wrapped, type(validator).__name__)
    return validator


def _timed(hook_name: str, fn):
    """Wrap one hook. Timed in a `finally`, so a hook that REJECTS is measured too — that is the
    402 path, and it is the whole cost the caller pays."""
    operation, hook = _split(hook_name)

    @functools.wraps(fn)
    async def _wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return await fn(*args, **kwargs)
        finally:
            _record(operation, hook, time.perf_counter() - started)

    # `functools.wraps` sets `__wrapped__`, which is what makes a wrapped hook identifiable --
    # both to `inspect.signature` and to the test that asserts every hook is covered.
    return _wrapper


def _split(hook_name: str) -> tuple[str, str]:
    """``validate_recall`` -> ("recall", "pre"); ``on_recall_complete`` -> ("recall", "post").

    The pair matters more than the name: a pre-check and a post-charge fail differently and are
    fixed differently, so they must not share a series.
    """
    if hook_name.startswith("validate_"):
        return hook_name[len("validate_") :], "pre"
    if hook_name.startswith("on_") and hook_name.endswith("_complete"):
        return hook_name[len("on_") : -len("_complete")], "post"
    return hook_name, "other"


def _record(operation: str, hook: str, seconds: float) -> None:
    # Instrumentation must never be the thing that fails the request it measures.
    try:
        from ..metrics import get_metrics_collector

        get_metrics_collector().record_validator_phase(operation, hook, seconds)
    except Exception:
        logger.debug("validator phase metrics not recorded", exc_info=True)
