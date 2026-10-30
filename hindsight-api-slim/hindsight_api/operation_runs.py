"""Operation runs: one record per retain, recall, reflect, consolidation or mental-model refresh.

Each run has an id that every model call made inside it carries, so a run joins to its calls. A
run records its wall time, and where the time went: the model calls it made (their count, summed
time and tokens), the calls into the memories store (count and summed time, measured at the
store's interface, so it covers any store), and the phases the engine already times for metrics.
Run inside a worker task, it also records the task's id and when the task was queued.

A run opened inside another (the reflect pass of a mental-model refresh) is its own run with the
outer one as its parent; what it spends is counted in the inner run only, so a parent's figures
plus its children's add up without counting anything twice.

Runs are handed to the sink the engine installs (the model call extension's ``record_operation``)
when they end. Everything here is in-memory bookkeeping on the call path; nothing does I/O.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Run:
    id: str
    parent_id: str | None
    operation: str
    bank_id: str | None
    task_id: str | None
    queued_at: datetime | None
    started_at: datetime
    t0: float
    model_calls: int = 0
    model_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    store_calls: int = 0
    store_ms: float = 0.0
    phases_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Task:
    id: str
    queued_at: datetime | None


_current: ContextVar[_Run | None] = ContextVar("hindsight_operation_run", default=None)
_task: ContextVar[_Task | None] = ContextVar("hindsight_operation_task", default=None)
# Set while a memories-store call is running, so calls the store makes to itself count once.
_in_store: ContextVar[bool] = ContextVar("hindsight_operation_in_store", default=False)

_sink: Callable[[Any], None] | None = None


def set_sink(sink: Callable[[Any], None] | None) -> None:
    """Where finished runs go (an ``OperationRun`` each). Installed by the engine at startup."""
    global _sink
    _sink = sink


def current_operation_id() -> str | None:
    """The id of the operation run in progress, or None outside one.

    For a memories store to pass along with its own requests, so its server-side records join
    the run that caused them.
    """
    run = _current.get()
    return run.id if run else None


@contextmanager
def worker_task(task_id: str, queued_at: datetime | None) -> Iterator[None]:
    """Mark the runs opened inside as part of a queued task: they get its id and queue time."""
    token = _task.set(_Task(task_id, queued_at))
    try:
        yield
    finally:
        _task.reset(token)


@contextmanager
def timed_operation(operation: str, bank_id: str | None) -> Iterator[None]:
    """Time one operation run and hand it to the sink when it ends, successful or not."""
    parent = _current.get()
    task = _task.get()
    run = _Run(
        id=str(uuid.uuid4()),
        parent_id=parent.id if parent else None,
        operation=operation,
        bank_id=bank_id,
        task_id=task.id if task else None,
        queued_at=task.queued_at if task and parent is None else None,
        started_at=datetime.now(timezone.utc),
        t0=time.perf_counter(),
    )
    token = _current.set(run)
    error: BaseException | None = None
    try:
        yield
    except BaseException as e:
        error = e
        raise
    finally:
        _current.reset(token)
        _emit(run, error)


def _emit(run: _Run, error: BaseException | None) -> None:
    if _sink is None:
        return
    try:
        # Lazy: the extension module imports the engine.
        from hindsight_api.extensions.model_call import OperationRun
        from hindsight_api.schema_context import get_current_schema

        duration = time.perf_counter() - run.t0
        _sink(
            OperationRun(
                id=run.id,
                parent_id=run.parent_id,
                operation=run.operation,
                tenant=get_current_schema(),
                bank_id=run.bank_id,
                task_id=run.task_id,
                queued_at=run.queued_at,
                started_at=run.started_at,
                ended_at=run.started_at + timedelta(seconds=duration),
                duration_ms=int(duration * 1000),
                success=error is None,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
                model_calls=run.model_calls,
                model_ms=int(run.model_ms),
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
                store_calls=run.store_calls,
                store_ms=int(run.store_ms),
                phases_ms={k: round(v, 1) for k, v in run.phases_ms.items()},
            )
        )
    except Exception:
        # Bookkeeping must never fail the operation it describes.
        logger.debug("operation run not recorded", exc_info=True)


def add_model_call(duration_s: float, input_tokens: int, output_tokens: int) -> None:
    run = _current.get()
    if run is not None:
        run.model_calls += 1
        run.model_ms += duration_s * 1000
        run.input_tokens += input_tokens
        run.output_tokens += output_tokens


def add_phase(name: str, seconds: float) -> None:
    run = _current.get()
    if run is not None:
        run.phases_ms[name] = run.phases_ms.get(name, 0.0) + seconds * 1000


def time_store_calls(store: Any) -> Any:
    """Time every async call into ``store`` against the run in progress. Returns ``store`` itself.

    Wraps the methods on the instance rather than putting a proxy in front, so the store keeps
    its identity and type: callers compare it with the Postgres store and check isinstance.
    """
    for name, member in inspect.getmembers(type(store), inspect.iscoroutinefunction):
        if name.startswith("_"):
            continue
        setattr(store, name, _timed(getattr(store, name)))
    return store


def _timed(method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    async def call(*args: Any, **kwargs: Any) -> Any:
        if _in_store.get() or _current.get() is None:
            return await method(*args, **kwargs)
        token = _in_store.set(True)
        t0 = time.perf_counter()
        try:
            return await method(*args, **kwargs)
        finally:
            _in_store.reset(token)
            run = _current.get()
            if run is not None:
                run.store_calls += 1
                run.store_ms += (time.perf_counter() - t0) * 1000

    return call


def timed(operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate an engine method whose first argument is ``bank_id`` as one operation run."""

    def wrap(method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
            bank_id = kwargs["bank_id"] if "bank_id" in kwargs else (args[0] if args else None)
            with timed_operation(operation, bank_id):
                return await method(self, *args, **kwargs)

        return call

    return wrap
