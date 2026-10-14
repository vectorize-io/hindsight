"""Each bank's LLM calls go out with its own key, or to its own LLM — on every operation.

A deployment fronting its LLM spend with a gateway gives each customer a key, and a
deployment choosing models per bank gives each bank an LLM; either loads an LLM
request extension to decide per call (here the `bank-llm-routing` community
extension, routing on bank-id and operation patterns). The promise is
all-or-nothing: one step that slips through on the server's LLM bills the wrong
account or runs the wrong model, and nothing visible breaks. So the story walks
every operation that calls the model —
sync retain and reflect in the HTTP process, async retain, consolidation and a
mental-model refresh in a standalone worker — and checks the key each call
carried at the provider.

The names are unusual on purpose: the stub is shared with every other server in
the session, and a straggling call from another story mentioning "Alice" must
not be mistaken for one of these.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from hindsight_client import Hindsight

from hindsight_system_tests import ChatRequest, reflect_loop, wait_until_settled
from hindsight_system_tests.payloads import extracted, fact, observes
from hindsight_system_tests.steps import anchor_for

from .conftest import BANK_PREFIX, CARL_MODEL, CARL_RETAIN_MODEL

pytestmark = pytest.mark.asyncio

OTTOLINE_ANSWER = "Ottoline lives in Tromsø."
BARNABY_ANSWER = "Barnaby lives in Valparaíso."


@pytest.fixture
async def keyed_bank(keyed_client: Hindsight) -> AsyncIterator[Callable[[str], str]]:
    """Name a bank for one of the keyed customers; delete them all afterwards."""
    made: list[str] = []

    def _make(customer: str) -> str:
        name = f"{BANK_PREFIX}{customer}-{uuid.uuid4().hex[:12]}"
        made.append(name)
        return name

    yield _make
    for name in made:
        with contextlib.suppress(Exception):
            await keyed_client.banks.delete_bank(name)


def _calls_about(llm, marker: str) -> list[ChatRequest]:
    return [call for call in llm.calls if marker in call.all_text]


def _step_of(call: ChatRequest) -> str:
    for step in ("extract_facts", "consolidate", "reflect"):
        if anchor_for(step) in call.all_text:
            return step
    return "other"


async def _exercise_every_operation(client: Hindsight, llm, bank: str, who: str, city: str, answer: str) -> None:
    llm.on_step("extract_facts", contains=who).returns(extracted(fact(f"{who} lives in {city}", who=who)))
    llm.on_step("consolidate", contains=who).answers_with(observes(f"{who} is settled in {city}"))
    reflect_loop(llm, answer=answer, query=f"Where does {who} live?")

    # HTTP process: synchronous retain extracts in the request.
    await client.aretain(bank_id=bank, content=f"{who} lives in {city}.")
    # Worker: async retain extracts there; consolidation always runs there.
    await client.aretain(bank_id=bank, content=f"{who} still lives in {city}.", retain_async=True)
    await wait_until_settled(client, bank)
    # HTTP process: reflect.
    reflected = await client.areflect(bank_id=bank, query=f"Where does {who} live?")
    assert reflected.text.strip() == answer
    # Worker: a mental model's first refresh.
    model = await client.mental_models.create_mental_model(
        bank, {"name": f"{who} housing", "source_query": f"Where does {who} live?"}
    )
    await wait_until_settled(client, bank)
    written = await client.mental_models.get_mental_model(bank, model.mental_model_id, detail="full")
    assert written.content.strip() == answer


async def test_every_operation_bills_the_bank_key(keyed_client, keyed_bank, llm):
    bank = keyed_bank("alice")
    await _exercise_every_operation(keyed_client, llm, bank, "Ottoline", "Tromsø", OTTOLINE_ANSWER)

    calls = _calls_about(llm, "Ottoline")
    assert {call.api_key for call in calls} == {"alice-key"}, "a call went out on another account"
    assert {call.headers.get("x-gateway-customer") for call in calls} == {"alice"}

    steps = [_step_of(call) for call in calls]
    assert steps.count("extract_facts") == 2, "sync (HTTP) and async (worker) retain each extract"
    assert "consolidate" in steps
    # Reflect from the HTTP request and the refresh in the worker both walk the loop.
    assert steps.count("reflect") >= 2


async def test_two_banks_are_billed_to_two_keys(keyed_client, keyed_bank, llm):
    await _exercise_every_operation(keyed_client, llm, keyed_bank("alice"), "Ottoline", "Tromsø", OTTOLINE_ANSWER)
    alice_keys = {call.api_key for call in _calls_about(llm, "Ottoline")}
    # The reflect rules answer any reflect turn, so Bob's need a rulebook of their own.
    llm.reset()
    await _exercise_every_operation(keyed_client, llm, keyed_bank("bob"), "Barnaby", "Valparaíso", BARNABY_ANSWER)

    assert alice_keys == {"alice-key"}
    assert {call.api_key for call in _calls_about(llm, "Barnaby")} == {"bob-key"}


async def test_a_bank_can_run_on_an_llm_of_its_own(keyed_client, keyed_bank, llm):
    """The route swaps the whole LLM — model, endpoint, key — and picks a different one
    again for retain, so each operation's calls are told apart by the model they reached."""
    bank = keyed_bank("carl")
    await _exercise_every_operation(keyed_client, llm, bank, "Hieronymus", "Lerwick", "Hieronymus lives in Lerwick.")

    calls = _calls_about(llm, "Hieronymus")
    assert {call.api_key for call in calls} == {"carl-key"}
    by_step = {(_step_of(call), call.model) for call in calls}
    assert by_step == {
        ("extract_facts", CARL_RETAIN_MODEL),
        ("consolidate", CARL_MODEL),
        ("reflect", CARL_MODEL),
    }


async def test_a_bank_with_no_key_is_refused_not_billed_to_the_server(keyed_client, keyed_bank, llm):
    """The extension's default for an unmatched bank is to refuse. The failure mode it
    exists to prevent is the silent one — the call going out on the server's key — so
    the assertion is that nothing about this bank reached the provider at all."""
    bank = keyed_bank("carol")
    llm.on_step("extract_facts", contains="Wilhelmina").returns(
        extracted(fact("Wilhelmina lives in Ushuaia", who="Wilhelmina"))
    )

    with pytest.raises(Exception, match="No LLM route matches"):
        await keyed_client.aretain(bank_id=bank, content="Wilhelmina lives in Ushuaia.")

    assert _calls_about(llm, "Wilhelmina") == []
