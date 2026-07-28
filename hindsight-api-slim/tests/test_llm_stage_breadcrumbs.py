"""The LLM stage breadcrumb must distinguish waiting from calling (#3002).

`[WORKER_TASK] stage=llm.bedrock.retain_extract_facts+structured` was stamped
before the concurrency permits were acquired, so a task queued behind a
saturated semaphore looked identical to one the provider was actively running.
An operator debugging a wedged worker spent hours on Bedrock for tasks that had
never reached Bedrock.
"""

import asyncio

import pytest

from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.worker.stage import StageHolder, bind_holder


def _mock_llm() -> LLMConfig:
    return LLMConfig(provider="mock", api_key="", base_url="", model="m")


@pytest.mark.asyncio
async def test_stage_says_queued_while_waiting_for_a_permit(monkeypatch):
    llm = _mock_llm()
    holder = StageHolder()
    gate = asyncio.Semaphore(0)  # never free: stands in for a saturated cap

    monkeypatch.setattr("hindsight_api.engine.llm_wrapper._semaphores_for_scope", lambda scope: [gate])

    async def run_call():
        bind_holder(holder)
        await llm.call(messages=[{"role": "user", "content": "hi"}], scope="retain_extract_facts")

    task = asyncio.create_task(run_call())
    for _ in range(5):  # let it reach the acquire
        await asyncio.sleep(0)

    assert holder.stage == "llm.mock.retain_extract_facts.queued"

    gate.release()
    await task
    assert not holder.stage.endswith(".queued"), "stage stayed 'queued' after the permit was granted"


@pytest.mark.asyncio
async def test_stage_drops_queued_once_the_call_is_in_flight(monkeypatch):
    """With permits free the call proceeds, and the stage names the in-flight
    call — the state the label always claimed to describe."""
    llm = _mock_llm()
    holder = StageHolder()
    seen: list[str] = []

    async def fake_call(**_kwargs):
        seen.append(holder.stage)
        return "ok"

    monkeypatch.setattr(llm._provider_impl, "call", fake_call)

    async def run_call():
        bind_holder(holder)
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            scope="retain_extract_facts",
        )

    await asyncio.create_task(run_call())

    assert seen == ["llm.mock.retain_extract_facts"]
