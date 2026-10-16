"""A reflect stops when its caller hangs up, even in the middle of a model call.

Reflect used to notice a gone caller only between turns of its search loop. Its
closing stretch has no such gap: the turn that asks the model for its `done` call,
the provider's retries of that turn, then the standalone answer prompt and *its*
retries all run back to back. A caller that gave up during that stretch left the
server paying for every one of those calls, ~270s of them, until the wall timeout
ended it (#5128).

The caller walks away via a short client timeout while the closing turn is held
open. The closing turn then answers without a `done` call — the reply that sends
reflect on to the standalone prompt. If the abandoned reflect were still running,
that prompt would reach the stub; it must not.
"""

from __future__ import annotations

import asyncio

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests.payloads import consolidation, extracted, fact

pytestmark = pytest.mark.asyncio

QUERY = "Where does Alice live?"

#: Shorter than the hold, long enough that the request is in the closing turn when it fires.
_CLIENT_TIMEOUT_SECONDS = 2.0

#: How long a still-running reflect would need to reach the standalone prompt once
#: the closing turn answers. It is one local round-trip, so this is generous.
_FALLBACK_GRACE_SECONDS = 2.0


@pytest.fixture
async def impatient(hindsight_server) -> Hindsight:
    """A client that gives up mid-reflect, the way a timing-out hook does."""
    client = Hindsight(base_url=hindsight_server.url, timeout=_CLIENT_TIMEOUT_SECONDS)
    yield client
    await client.aclose()


@pytest.fixture
async def bank_with_facts(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts").returns(
        extracted(fact("Alice moved to Berlin", who="Alice", entities=["Alice", "Berlin"]))
    )
    llm.on_step("consolidate").returns(consolidation())
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin.")
    await settled(bank_id)
    return bank_id


def _standalone_calls(llm) -> int:
    return sum(1 for call in llm.calls if "synthesizes answers from retrieved memories" in call.all_text)


async def test_an_abandoned_reflect_makes_no_more_model_calls(llm, impatient, bank_with_facts):
    # Registration order is match order: the closing turn also offers `done` and
    # carries the reflect preamble, so it is claimed first.
    llm.on_step("reflect_closing").returns_text("Alice lives in Berlin.")
    # The free turn stops with prose instead of `done`, which is what sends the
    # loop into its closing turn.
    llm.on_step("reflect", tool="done").returns_text("Alice lives in Berlin.")
    llm.on_step("reflect").calls_the_offered_tool(query=QUERY)
    # Declared so a still-running reflect is answered and counted, not reported as
    # an unscripted call.
    llm.on_step("reflect_answer").returns_text("Alice lives in Berlin.")

    async with llm.hold("reflect_closing") as held:
        with pytest.raises(BaseException) as abandoned:  # noqa: B017, PT011
            await impatient.areflect(bank_id=bank_with_facts, query=QUERY)
        # Broad on purpose, as in the retain story: what matters is that the client
        # gave up, not how its transport words it — and not a server error.
        assert "timeout" in repr(abandoned.value).lower(), f"gave up for the wrong reason: {abandoned.value!r}"
        await held.reached()

    # The hold is released: an abandoned reflect still running would now get its
    # closing reply and go on to the standalone prompt.
    for _ in range(int(_FALLBACK_GRACE_SECONDS / 0.1)):
        if _standalone_calls(llm):
            break
        await asyncio.sleep(0.1)
    assert _standalone_calls(llm) == 0, "the abandoned reflect went on to the standalone answer prompt"
