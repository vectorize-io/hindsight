"""A fact shared through a tag does not share the note it came from (#5030).

Two people share a bank. Kate writes a note tagged `user:kate`; an entity label
with `tag: true` marks its team rules, so those facts also carry `kind:rule`. Dan
reads with `user:dan` + `kind:rule`, strict. He should see his own facts and the
rules — and nothing else of Kate's.

The facts were always scoped right. The seam is the step *after* the fact: recall
with chunks and reflect's `expand` hand back the fact's source text, and that text
is Kate's whole note, private paragraph included. Each path is a composition no
single-mechanism test spans: label tagging at retain, a tag filter at read, and a
source lookup that never looked at the document's own tags.

Every leak assertion is paired with Kate reading the same thing, so a passing test
means the filter held — not that the text was never there to leak.
"""

from __future__ import annotations

import pytest

from hindsight_system_tests import reflect_loop
from hindsight_system_tests.payloads import consolidation, extracted, fact
from hindsight_system_tests.rulebook import ChatRequest, LLMStub, ToolCall

pytestmark = pytest.mark.asyncio

KATE_NOTE = (
    "Team rules: every pull request needs two approvals before merge. Nobody deploys to "
    "production on Fridays. Customer data must never be copied to personal laptops.\n"
    "Personal: Kate is interviewing at another company next week and hasn't told anyone."
)
DAN_NOTE = "Dan is refactoring the billing service this sprint."
PRIVATE = "interviewing at another company"
FRIDAY_RULE = "Nobody deploys to production on Fridays"
RULES = [
    "Every pull request needs two approvals before merge",
    FRIDAY_RULE,
    "Customer data must never be copied to personal laptops",
]
QUERY = "What are the team rules about deploys, pull requests and customer data?"
DAN = {"tags": ["user:dan", "kind:rule"], "tags_match": "any_strict"}
KATE = {"tags": ["user:kate"], "tags_match": "any_strict"}
ANSWER = "It is a team rule."


@pytest.fixture
async def shared_bank(client, llm, bank_id, settled) -> str:
    await client.banks.update_bank_config(
        bank_id,
        {
            "updates": {
                "entity_labels": [
                    {
                        "key": "kind",
                        "type": "multi-values",
                        "tag": True,
                        "optional": True,
                        "values": [{"value": "rule", "description": "a team rule everyone follows"}],
                    }
                ]
            }
        },
    )
    llm.on_step("extract_facts", contains="Team rules").returns(
        extracted(
            *(fact(rule, entities=["kind:rule"]) for rule in RULES),
            fact("Kate is interviewing at another company next week", who="Kate", entities=["Kate"]),
        )
    )
    llm.on_step("extract_facts", contains="billing").returns(
        extracted(fact(DAN_NOTE.rstrip("."), who="Dan", entities=["Dan"]))
    )
    llm.on_step("consolidate").returns(consolidation())

    await client.aretain(bank_id=bank_id, content=KATE_NOTE, document_id="kate-sync", tags=["user:kate"])
    await client.aretain(bank_id=bank_id, content=DAN_NOTE, document_id="dan-notes", tags=["user:dan"])
    await settled(bank_id)
    return bank_id


async def _recall(client, bank: str, scope: dict):
    return await client.arecall(bank_id=bank, query=QUERY, include_chunks=True, budget="high", **scope)


def _chunks(response) -> str:
    return "\n".join(c.text for c in (response.chunks or {}).values())


async def _friday_rule_id(client, bank: str) -> str:
    response = await _recall(client, bank, DAN)
    ids = [r.id for r in response.results if FRIDAY_RULE in r.text]
    assert ids, f"setup: Dan should see the Friday rule. Got {[r.text for r in response.results]}"
    return ids[0]


async def test_the_facts_are_scoped_as_the_story_needs(client, shared_bank):
    """The premise, so nothing below passes vacuously: Dan sees the rules and his
    own fact, not Kate's private one, and not her document."""
    response = await _recall(client, shared_bank, DAN)

    texts = [r.text for r in response.results]
    for rule in RULES:
        assert any(rule in t for t in texts), f"missing rule {rule!r} in {texts}"
    assert not any(PRIVATE in t for t in texts)

    documents = await client.documents.list_documents(shared_bank, **DAN)
    assert [d.id for d in documents.items] == ["dan-notes"]


async def test_recall_chunks_do_not_carry_kates_note_to_dan(client, shared_bank):
    response = await _recall(client, shared_bank, DAN)

    assert any(FRIDAY_RULE in r.text for r in response.results), "the rules must still come back"
    assert PRIVATE not in _chunks(response)


async def test_recall_chunks_carry_the_note_to_kate(client, shared_bank):
    response = await _recall(client, shared_bank, KATE)

    assert PRIVATE in _chunks(response)


def _expand_once_then_finish(llm: LLMStub, memory_id: str) -> None:
    """On the free turn, expand the rule to its whole document; once that result is
    in the conversation, finish. Registered before `reflect_loop`, so it wins the
    turns that offer `expand`; the ladder and the prose answer stay with the loop."""

    def build(request: ChatRequest) -> ToolCall:
        if request.has_result_of("expand"):
            return ToolCall("done", {"answer": ANSWER})
        return ToolCall("expand", {"memory_ids": [memory_id], "depth": "document"})

    llm.on_step("reflect", tool="expand").answers_with_tool_call(build)
    reflect_loop(llm, answer=ANSWER)


def _reflect_prompts(llm: LLMStub) -> str:
    return "\n".join(llm.prompts_for("reflect"))


def _expand_ran(llm: LLMStub) -> bool:
    return any(call.has_result_of("expand") for call in llm.calls)


async def test_reflect_cannot_expand_the_rule_into_kates_note(client, llm, shared_bank):
    """The reported reproduction: asked where the rule comes from, reflect expands
    it to the full original note. Dan gets the rule, not the note."""
    _expand_once_then_finish(llm, await _friday_rule_id(client, shared_bank))

    response = await client.areflect(
        bank_id=shared_bank, query="Where does the no-Friday-deploy rule come from? Expand to the full note.", **DAN
    )

    assert response.text == ANSWER
    assert _expand_ran(llm), "setup: reflect never ran expand, so nothing was tested"
    assert PRIVATE not in _reflect_prompts(llm)


async def test_reflect_expands_the_rule_into_the_note_for_kate(client, llm, shared_bank):
    _expand_once_then_finish(llm, await _friday_rule_id(client, shared_bank))

    await client.areflect(
        bank_id=shared_bank, query="Where does the no-Friday-deploy rule come from? Expand to the full note.", **KATE
    )

    assert PRIVATE in _reflect_prompts(llm)


async def test_a_rules_mental_model_cannot_publish_kates_note(client, llm, shared_bank, settled):
    """A page scoped to `kind:rule` refreshes through the same reflect loop, and
    every rule reader sees what it writes. Expanding into Kate's note there would
    publish her private paragraph to the whole team."""
    _expand_once_then_finish(llm, await _friday_rule_id(client, shared_bank))

    await client.mental_models.create_mental_model(
        shared_bank,
        {
            "name": "Team rules",
            "source_query": "What are the team rules and where do they come from?",
            "tags": ["kind:rule"],
            "trigger": {"tags_match": "any_strict"},
        },
    )
    await settled(shared_bank)

    assert _expand_ran(llm), "setup: the refresh never ran expand, so nothing was tested"
    assert PRIVATE not in _reflect_prompts(llm)
