"""A disconnected SDK request must not start more finalization attempts.

The server uses the real LiteLLM transport and retry loop against the HTTP stub.
A held finalization response times out after the SDK connection is cancelled;
that failure must retire the request instead of retrying or starting synthesis.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.server import HindsightServer, start_hindsight_server

pytestmark = pytest.mark.asyncio

REQUEST_TIMEOUT = 1.0
ANSWER = "No stored facts answer this question."


@pytest.fixture(scope="module")
def cancellation_server(stub_server, stubs, tmp_path_factory) -> Iterator[HindsightServer]:
    log_path = tmp_path_factory.mktemp("reflect-cancellation") / "server.log"
    # LiteLLM probes with a bare "test" prompt. This temporary rule is removed
    # after startup, before it could match the bank name or a reflect query.
    stubs.llm.on_chat(contains="test").returns_text("ok")
    server = start_hindsight_server(
        stub_url=stub_server.url,
        log_path=log_path,
        extra_env={
            "HINDSIGHT_API_REFLECT_LLM_PROVIDER": "litellm",
            "HINDSIGHT_API_REFLECT_LLM_MODEL": "openai/stub-model",
            "HINDSIGHT_API_REFLECT_LLM_BASE_URL": f"{stub_server.url}/v1",
            "HINDSIGHT_API_REFLECT_LLM_API_KEY": "stub-key",
            "HINDSIGHT_API_REFLECT_LLM_TIMEOUT": str(REQUEST_TIMEOUT),
            "HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES": "3",
            "HINDSIGHT_API_REFLECT_LLM_INITIAL_BACKOFF": "0",
            "HINDSIGHT_API_REFLECT_LLM_MAX_BACKOFF": "0",
            # mid gets two iterations (retrieval then closing); low gets one
            # and uses the standalone answer path. Both are public budgets.
            "HINDSIGHT_API_REFLECT_MAX_ITERATIONS": "2",
        },
    )
    assert not stubs.llm.unmatched, "startup sent an undeclared provider request"
    stubs.llm.reset()
    yield server
    server.stop()


@pytest.fixture
async def cancellation_client(cancellation_server) -> AsyncIterator[Hindsight]:
    client = Hindsight(base_url=cancellation_server.url)
    yield client
    await client.aclose()


@pytest.mark.parametrize("phase, budget", [("reflect_closing", "mid"), ("reflect_answer", "low")])
async def test_disconnect_retires_finalization_before_another_provider_attempt(
    cancellation_server, cancellation_client, llm, phase, budget
):
    bank = f"systest-cancel-{uuid.uuid4().hex[:12]}"
    await cancellation_client.acreate_bank(bank_id=bank, name="Cancellation test")
    reflect_loop(llm, answer=ANSWER)
    task = None
    try:
        async with llm.hold(phase) as held:
            task = asyncio.create_task(cancellation_client.areflect(bank_id=bank, query="What is known?", budget=budget))
            await held.reached()
            assert len(llm.prompts_for(phase)) == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # Observe completion in the public server log rather than guessing
            # whether a short sleep gave the retry loop time to run.
            cancelled = f"[REFLECT CANCELLED] bank={bank} reason=client disconnected"
            deadline = time.monotonic() + 10
            while cancelled not in cancellation_server.logs():
                assert time.monotonic() < deadline, "the disconnected reflect request never retired"
                await asyncio.sleep(0.05)

            assert len(llm.prompts_for(phase)) == 1, "provider retried after the SDK disconnected"
            if phase == "reflect_closing":
                assert not llm.prompts_for("reflect_answer"), "closing fell back after disconnect"

        # A new SDK request on the same bank still finishes normally.
        response = await cancellation_client.areflect(bank_id=bank, query="What is known?", budget=budget)
        assert response.text == ANSWER
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await cancellation_client.banks.delete_bank(bank)
