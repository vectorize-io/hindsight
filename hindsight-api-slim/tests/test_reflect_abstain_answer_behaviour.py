"""An abstain reflect's answer: written from the picked memories, NONE when they do not answer.

The decision model picks the memories; the LLM then reads only those and either answers or says
that nothing applies (``applies: false``, see ``_answer_picked``). Whether a model follows that
cannot be simulated with MockLLM, so this drives the real prompt pair. The note is judged; the
nothing-applies case is checked through the same parse the agent applies, since it is a field the
code reads, not a wording.
"""

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.reflect.agent import _abstain_answer_from
from hindsight_api.engine.reflect.models import AbstainAnswer
from hindsight_api.engine.reflect.prompts import build_abstain_answer_prompt, build_abstain_answer_system_prompt
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core

_CONFLICT = (
    "Does what I am doing right now (my plan, command or code change) contradict a project decision recorded in "
    "memory, miss one, or repeat an option recorded as rejected? If so, tell me in at most 50 words the recorded "
    "value or rule (exact literals verbatim) and what in my work disagrees. If nothing conflicts, reply with nothing."
)
_HEADSUP = (
    "Before I start on this request: which project decision recorded in memory (an exact value, a rule, an "
    "exception) about exactly the code this request will change must my fix respect, that the request itself does "
    "not state? Tell me in at most 70 words, copying exact literals verbatim. If memory records none, reply with "
    "nothing."
)
_MEMORY = [
    "- [2026-10-16] In infraconf/units.py, parse_quantity(value, kind) reads memory suffixes as binary powers "
    "(K=1024, M=1024**2, G=1024**3) and disk suffixes as decimal powers (M=1000**2, G=1000**3); 'K' is rejected for "
    "disk with a ValueError.",
    "  - source: user: Exactly: memory K/M/G/T as 1024 powers, disk M/G/T as 1000 powers, and no K for disk.",
]


async def _answer(question: str, context: str, memory: list[str] = _MEMORY) -> str:
    result = await LLMConfig.from_env().call(
        messages=[
            {"role": "system", "content": build_abstain_answer_system_prompt(None, None, None)},
            {"role": "user", "content": build_abstain_answer_prompt(question, context, memory, None)},
        ],
        scope="reflect",
        temperature=0.0,
        response_format=AbstainAnswer,
    )
    reply = _abstain_answer_from(result.content)
    return reply.answer.strip() if reply.applies else ""


def _is_none(answer: str) -> bool:
    return not answer


@pytest.mark.asyncio
async def test_a_change_against_the_memory_gets_the_recorded_value():
    answer = await _answer(
        _CONFLICT,
        "MY REQUEST:\nFix parse_quantity so pods get the memory they ask for.\n\n"
        'WHAT I JUST DID (edit):\n+_UNITS = {"K": 1000, "M": 1000 ** 2, "G": 1000 ** 3}  # one table for both kinds',
    )
    assert not _is_none(answer), answer
    await assert_meets_criteria(
        response=answer,
        criteria=(
            "The response tells the reader their change conflicts with the recorded decision: memory suffixes must be "
            "binary powers of 1024 (it states 1024 explicitly), unlike the 1000 they used. It does not hedge or say "
            "nothing conflicts."
        ),
        context="The memory records binary memory units (K=1024) and decimal disk units; the edit used 1000 for all.",
    )


@pytest.mark.asyncio
async def test_a_change_that_follows_the_memory_does_not_apply():
    answer = await _answer(
        _CONFLICT,
        "MY REQUEST:\nFix parse_quantity so pods get the memory they ask for.\n\n"
        "WHAT I JUST DID (edit):\n"
        '+_MEMORY_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}\n'
        '+_DISK_UNITS = {"M": 1000 ** 2, "G": 1000 ** 3, "T": 1000 ** 4}  # no K: rejected for disk',
    )
    assert _is_none(answer), f"expected NONE for a change that follows the memory, got: {answer!r}"


@pytest.mark.asyncio
async def test_a_heads_up_states_the_rule_with_its_literals():
    answer = await _answer(_HEADSUP, "MY REQUEST:\nFix parse_quantity so provisioned resources match the configs.")
    assert not _is_none(answer), answer
    await assert_meets_criteria(
        response=answer,
        criteria=(
            "The response states, before any work, that memory suffixes are binary (1024 powers) and disk suffixes "
            "are decimal (1000 powers), and that 'K' is rejected for disk. It gives the literal values."
        ),
        context="A heads-up for a request about parse_quantity, from one recorded decision.",
    )


@pytest.mark.asyncio
async def test_a_heads_up_follows_the_source_where_the_summary_is_stale():
    # Consolidation can merge an old and a new decision into one summary that keeps the old one; the source
    # text the memory carries says what was finally decided, so the note must follow it.
    memory = [
        "- [2026-10-16] orderflow's advance() enforces a strictly linear lifecycle: pending -> paid -> shipped -> "
        "returned -> refunded; branching transitions were rejected.",
        "  - source: user: First idea: keep it linear, pending, paid, shipped, returned, refunded.\n"
        "assistant: Done, linear.\n"
        "user: No, scrap linear, it blocks holds. Recap of the pinned lifecycle: pending -> paid/on_hold/cancelled; "
        "paid -> shipped/on_hold only; on_hold -> pending/cancelled; shipped -> returned; returned -> refunded; "
        "cancelled and refunded are terminal.",
    ]
    answer = await _answer(_HEADSUP, "MY REQUEST:\nFix advance() so held orders can be released.", memory)
    assert not _is_none(answer), answer
    await assert_meets_criteria(
        response=answer,
        criteria=(
            "The response gives the branching lifecycle from the user's final recap (for example pending can go to "
            "on_hold or cancelled, and on_hold back to pending). It does not say the lifecycle is strictly linear or "
            "that branching was rejected."
        ),
        context="The memory's summary kept a superseded linear rule; its source ends with the final branching recap.",
    )
