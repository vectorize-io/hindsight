"""What consolidation does when the model's answer is incomplete or aimed wrong.

A model's consolidation answer is advice, and a small model's advice has holes in
it. Every hole below was observed on gemini-3.1-flash-lite by the consolidation
eval (``hindsight-system-evals/evals/test_09_consolidation.py``) and used to cost
knowledge silently: the fact was stamped consolidated, backed nothing, and was
never shown to the model again. Each story makes the stub give one such answer
and checks the bank still ends up holding the fact.

The stub decides nothing a real model would not; these stories pin what the
SERVER does with the answer — which facts get re-asked, where evidence moves —
not whether a model would phrase it that way.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests.payloads import (
    Consolidation,
    Observation,
    ObservationDelete,
    ObservationUpdate,
    batch_fact_id,
    batch_fact_ids,
    extracted,
    fact,
    shown_observations,
)

pytestmark = pytest.mark.asyncio


async def _observations(client, bank: str) -> list:
    memories = await client.memory.list_memories(bank, limit=100)
    return [m for m in memories.items if m.fact_type == "observation"]


async def _fact_id(client, bank: str, prefix: str) -> str:
    memories = await client.memory.list_memories(bank, limit=100)
    return next(m.id for m in memories.items if m.fact_type != "observation" and m.text.startswith(prefix))


async def test_a_fact_the_answer_ignores_is_re_asked_on_its_own(client, llm, bank_id, settled):
    """Seven of eight facts handled, the eighth forgotten: in a batch that is an
    oversight as often as a judgement, so the forgotten fact gets one call alone."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", entities=["Alice", "Berlin"]),
            fact("Bob joined Acme", entities=["Bob", "Acme"]),
        )
    )

    def answer(request) -> Consolidation:
        alice = batch_fact_id(request, "Alice moved")
        if alice and len(batch_fact_ids(request)) > 1:
            return Consolidation(creates=[Observation(text="Alice lives in Berlin", source_fact_ids=[alice])])
        return Consolidation(creates=[Observation(text="Bob works at Acme", source_fact_ids=batch_fact_ids(request))])

    llm.on_step("consolidate").answers_with(answer)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Bob joined Acme.")
    await settled(bank_id)

    assert sorted(o.text for o in await _observations(client, bank_id)) == [
        "Alice lives in Berlin",
        "Bob works at Acme",
    ]
    assert len(llm.prompts_for("consolidate")) == 2, "the batch, then the ignored fact alone"


async def test_a_fact_ignored_twice_gets_a_call_of_its_own(client, llm, bank_id, settled):
    """The re-ask batch can forget a fact too (it handled two of three and dropped the
    third); the facts it leaves get one last call each."""
    llm.on_step("extract_facts").returns(
        extracted(
            fact("Alice moved to Berlin", entities=["Alice", "Berlin"]),
            fact("Bob joined Acme", entities=["Bob", "Acme"]),
            fact("Carol plays the cello", entities=["Carol"]),
        )
    )

    def answer(request) -> Consolidation:
        ids = batch_fact_ids(request)
        alice = batch_fact_id(request, "Alice moved")
        if len(ids) > 1:
            # In any batch, only Alice is ever handled.
            return Consolidation(
                creates=[Observation(text="Alice lives in Berlin", source_fact_ids=[alice])] if alice else []
            )
        text = "Bob works at Acme" if batch_fact_id(request, "Bob") else "Carol plays the cello"
        return Consolidation(creates=[Observation(text=text, source_fact_ids=ids)])

    llm.on_step("consolidate").answers_with(answer)
    await client.aretain(bank_id=bank_id, content="Alice moved to Berlin. Bob joined Acme. Carol plays the cello.")
    await settled(bank_id)

    assert sorted(o.text for o in await _observations(client, bank_id)) == [
        "Alice lives in Berlin",
        "Bob works at Acme",
        "Carol plays the cello",
    ]
    assert len(llm.prompts_for("consolidate")) == 4, "the batch, the re-ask of two, then one call each"


async def test_a_lone_fact_the_answer_ignores_is_not_re_asked(client, llm, bank_id, settled):
    """Alone, a skip is the model's answer (chatter, ephemera) — re-asking it would
    loop forever on every fact a model rightly leaves out."""
    llm.on_step("extract_facts").returns(extracted(fact("The user said thanks", entities=["user"])))
    llm.on_step("consolidate").answers_with(lambda request: Consolidation())

    await client.aretain(bank_id=bank_id, content="thanks!")
    await settled(bank_id)

    assert await _observations(client, bank_id) == []
    assert len(llm.prompts_for("consolidate")) == 1


@pytest.fixture
async def two_observations(client, llm, bank_id, settled) -> str:
    """'Alice lives in Berlin' and its restatement 'Alice is based in Berlin', one fact each."""
    llm.on_step("extract_facts", contains="lives").returns(
        extracted(
            fact("Alice lives in Berlin", entities=["Alice", "Berlin"]),
            fact("Alice is based in Berlin", entities=["Alice", "Berlin"]),
        )
    )

    def create_each(request) -> Consolidation:
        return Consolidation(
            creates=[
                Observation(text="Alice lives in Berlin", source_fact_ids=[batch_fact_id(request, "lives") or ""]),
                Observation(text="Alice is based in Berlin", source_fact_ids=[batch_fact_id(request, "based") or ""]),
            ]
        )

    llm.on_step("consolidate").answers_with(create_each)
    await client.aretain(bank_id=bank_id, content="Alice lives in Berlin. Alice is based in Berlin.")
    await settled(bank_id)
    llm.reset()
    llm.on_step("extract_facts", contains="lease").returns(
        extracted(fact("Alice renewed her Berlin lease", entities=["Alice", "Berlin"]))
    )
    return bank_id


def _fold(*, merged_into: str | None):
    """Update the survivor with the new fact; delete its restatement. ``merged_into`` is
    "survivor" (the id), any other text (a merge into something the model could not name
    by id), or None (a plain delete)."""

    def answer(request) -> Consolidation:
        shown = shown_observations(request)
        lease = batch_fact_id(request, "lease")
        if lease is None or "Alice lives in Berlin" not in shown:
            # The re-queued restatement fact, back on its own: evidence for the survivor.
            survivor = next(iter(v for k, v in shown.items() if "Berlin" in k), None)
            if survivor is None:
                return Consolidation()
            return Consolidation(
                updates=[
                    ObservationUpdate(
                        text="Alice lives in Berlin and renewed her lease",
                        observation_id=survivor,
                        source_fact_ids=batch_fact_ids(request),
                    )
                ]
            )
        survivor = shown["Alice lives in Berlin"]
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="Alice lives in Berlin and renewed her lease",
                    observation_id=survivor,
                    source_fact_ids=[lease],
                )
            ],
            deletes=[
                ObservationDelete(
                    observation_id=shown["Alice is based in Berlin"],
                    merged_into=survivor if merged_into == "survivor" else merged_into,
                )
            ],
        )

    return answer


async def test_a_folding_delete_moves_its_evidence_to_the_survivor(client, llm, two_observations, settled):
    """The survivor can only cite this batch's facts, so without the fold it would
    carry the restatement's content with none of its evidence."""
    llm.on_step("consolidate").answers_with(_fold(merged_into="survivor"))
    await client.aretain(bank_id=two_observations, content="Alice renewed her Berlin lease.")
    await settled(two_observations)

    [survivor] = await _observations(client, two_observations)
    based = await _fact_id(client, two_observations, "Alice is based in Berlin")
    assert based in survivor.source_memory_ids
    assert survivor.proof_count == 3
    assert len(llm.prompts_for("consolidate")) == 1, "a named fold needs no second call"


async def test_a_merge_into_an_unnamed_observation_re_queues_its_evidence(client, llm, two_observations, settled):
    """A model merging into an observation it creates in the same response has no id
    to name, and writes the observation's text instead. The evidence goes back in the
    queue, and the next pass attaches it to whatever covers it now."""
    llm.on_step("consolidate").answers_with(_fold(merged_into="the Berlin observation"))
    await client.aretain(bank_id=two_observations, content="Alice renewed her Berlin lease.")
    await settled(two_observations)

    based = await _fact_id(client, two_observations, "Alice is based in Berlin")
    cited = {sid for o in await _observations(client, two_observations) for sid in o.source_memory_ids}
    assert based in cited
    assert len(llm.prompts_for("consolidate")) == 2, "the batch, then the re-queued evidence"


async def test_a_plain_delete_retires_its_evidence(client, llm, two_observations, settled):
    """A delete that is not a merge retires a belief. Re-queueing its evidence was
    tried and brought retired beliefs back (eval ``correction-retires-false-belief``)."""
    llm.on_step("consolidate").answers_with(_fold(merged_into=None))
    await client.aretain(bank_id=two_observations, content="Alice renewed her Berlin lease.")
    await settled(two_observations)

    based = await _fact_id(client, two_observations, "Alice is based in Berlin")
    cited = {sid for o in await _observations(client, two_observations) for sid in o.source_memory_ids}
    assert based not in cited
    assert len(llm.prompts_for("consolidate")) == 1


@pytest.fixture
async def one_observation(client, llm, bank_id, settled) -> str:
    llm.on_step("extract_facts", contains="8443").returns(
        extracted(fact("ledger-sync listens on port 8443", entities=["ledger-sync"]))
    )
    llm.on_step("consolidate").answers_with(
        lambda request: Consolidation(
            creates=[Observation(text="ledger-sync listens on port 8443", source_fact_ids=batch_fact_ids(request))]
        )
    )
    await client.aretain(bank_id=bank_id, content="ledger-sync listens on port 8443.")
    await settled(bank_id)
    llm.reset()
    return bank_id


async def test_a_miscopied_observation_id_still_reaches_its_observation(client, llm, one_observation, settled):
    """`d0a3effa-1795-4c48-…` came back as `d0a3effa-1795-4ef0-…`: the head right,
    the tail invented. The one shown observation with that head is the target."""
    llm.on_step("extract_facts", contains="TLS").returns(
        extracted(fact("ledger-sync terminates TLS itself", entities=["ledger-sync"]))
    )

    def miscopied(request) -> Consolidation:
        [target] = shown_observations(request).values()
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="ledger-sync listens on port 8443 and terminates TLS itself",
                    observation_id=target[:13] + "-0000-0000-000000000000"[: len(target) - 13],
                    source_fact_ids=batch_fact_ids(request),
                )
            ]
        )

    llm.on_step("consolidate").answers_with(miscopied)
    await client.aretain(bank_id=one_observation, content="ledger-sync terminates TLS itself.")
    await settled(one_observation)

    [observation] = await _observations(client, one_observation)
    assert observation.text == "ledger-sync listens on port 8443 and terminates TLS itself"
    assert observation.proof_count == 2


async def test_an_update_aimed_at_the_new_fact_is_recorded_as_a_create(client, llm, one_observation, settled):
    """A model recording a new fact sometimes names that fact as the UPDATE target.
    There is no observation to update, and dropping the update lost the fact."""
    llm.on_step("extract_facts", contains="Kotlin").returns(
        extracted(fact("billing-api is written in Kotlin", entities=["billing-api"]))
    )

    def aimed_at_fact(request) -> Consolidation:
        [fact_id] = batch_fact_ids(request)
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="billing-api is written in Kotlin", observation_id=fact_id, source_fact_ids=[fact_id]
                )
            ]
        )

    llm.on_step("consolidate").answers_with(aimed_at_fact)
    await client.aretain(bank_id=one_observation, content="billing-api is written in Kotlin.")
    await settled(one_observation)

    texts = sorted(o.text for o in await _observations(client, one_observation))
    assert texts == ["billing-api is written in Kotlin", "ledger-sync listens on port 8443"]


async def test_a_malformed_fact_id_does_not_fail_the_batch(client, llm, one_observation, settled):
    """A truncated id among the cited facts names nothing and is ignored; it must not
    take down the batch — and with it the operation — on the way."""
    llm.on_step("extract_facts", contains="TLS").returns(
        extracted(fact("ledger-sync terminates TLS itself", entities=["ledger-sync"]))
    )

    def truncated(request) -> Consolidation:
        [target] = shown_observations(request).values()
        return Consolidation(
            updates=[
                ObservationUpdate(
                    text="ledger-sync listens on port 8443 and terminates TLS itself",
                    observation_id=target,
                    source_fact_ids=[*batch_fact_ids(request), "d0a3effa-1795"],
                )
            ]
        )

    llm.on_step("consolidate").answers_with(truncated)
    await client.aretain(bank_id=one_observation, content="ledger-sync terminates TLS itself.")
    await settled(one_observation)

    [observation] = await _observations(client, one_observation)
    assert observation.proof_count == 2
