"""Unit tests for the worker stage breadcrumb module."""

import asyncio

import pytest

from hindsight_api.worker.stage import StageHolder, bind_holder, get_stage, set_stage


def test_set_stage_is_noop_without_holder():
    # No holder bound in this context: must not raise, and get_stage returns None.
    set_stage("anything")
    assert get_stage() is None


@pytest.mark.asyncio
async def test_holder_bound_inside_task_is_visible_to_called_code():
    holder = StageHolder()

    async def inner():
        # The poller binds the holder from inside the task coroutine so it
        # lives in that task's contextvar scope; mirror that here.
        bind_holder(holder)
        set_stage("phase1")
        # Engine code further down the call stack reads via set_stage.
        set_stage("phase2")
        assert get_stage() == "phase2"

    await asyncio.create_task(inner())

    # Holder is mutable: the spawning context sees the latest stage written
    # by the child task without needing access to the contextvar.
    assert holder.stage == "phase2"


@pytest.mark.asyncio
async def test_holder_does_not_leak_across_tasks():
    # Each asyncio.create_task copies the parent's context. Binding inside
    # one task must not affect a sibling task's view.
    holder_a = StageHolder()
    holder_b = StageHolder()

    async def task_a():
        bind_holder(holder_a)
        set_stage("a")

    async def task_b():
        bind_holder(holder_b)
        set_stage("b")

    await asyncio.gather(asyncio.create_task(task_a()), asyncio.create_task(task_b()))

    assert holder_a.stage == "a"
    assert holder_b.stage == "b"
    # Outside both tasks, no holder is bound.
    assert get_stage() is None


@pytest.mark.asyncio
async def test_set_stage_updates_timestamp():
    holder = StageHolder()

    async def inner():
        bind_holder(holder)
        first = holder.updated_at
        # asyncio.sleep guarantees monotonic clock advances on next set.
        await asyncio.sleep(0.01)
        set_stage("next")
        assert holder.updated_at > first

    await asyncio.create_task(inner())
