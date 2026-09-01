"""Judging answers, and judging what actually reached memory.

Two separate measurements, because they fail differently:

- **Answer correctness** — did reflect answer the question right? This is the
  outcome a user feels.
- **Fact coverage** — did the specific things visible only in the image reach the
  bank as facts at all? This is the mechanism, and it localizes a bad answer: an
  article whose image facts were never extracted is a retain problem, one whose
  facts are present but unanswered is a recall problem.

Answers are also scored for *confident wrongness*: the pre-feature failure was
not "I don't know", it was a fluent answer built from prose that pointed at a
picture. An arm that abstains is behaving far better than one that invents, so
the two are never collapsed into a single "incorrect" bucket.
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import BaseModel, Field

_JUDGE_PROVIDER = os.getenv("HINDSIGHT_TEST_JUDGE_PROVIDER", "gemini")
_JUDGE_MODEL = os.getenv("HINDSIGHT_TEST_JUDGE_MODEL", "gemini-2.5-flash")
_JUDGE_API_KEY = os.getenv(
    "HINDSIGHT_TEST_JUDGE_API_KEY",
    os.getenv("GEMINI_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")),
)

_provider_instance = None


def _provider():
    """One judge provider for the whole run — it is stateless and reused."""
    global _provider_instance
    if _provider_instance is None:
        from hindsight_api.engine.llm_wrapper import create_llm_provider

        _provider_instance = create_llm_provider(
            provider=_JUDGE_PROVIDER,
            api_key=_JUDGE_API_KEY,
            base_url=os.getenv("HINDSIGHT_TEST_JUDGE_BASE_URL", ""),
            model=_JUDGE_MODEL,
            # No reasoning_effort: the default judge is Gemini, which has no such
            # control and warns on every call when one is set.
            reasoning_effort="",
        )
    return _provider_instance


# Flakiness hardening, mirroring tests/llm_judge.py. A single temperature-0 judge
# call still flips on borderline phrasing, and it flips in one direction: towards
# "no". The first run of this benchmark scored the claim "the export control is
# labelled 'Download CSV'" as absent from a fact that read "...an export control
# that includes a 'Download CSV' button", which understated the feature by a
# third. So a negative is re-asked a couple of times at a higher temperature and
# only upheld if the majority agrees; a positive is taken at once, which keeps
# the common path at one call.
_CONFIRMATIONS = int(os.getenv("HINDSIGHT_BENCH_JUDGE_CONFIRMATIONS", "2"))
_CONFIRM_TEMPERATURE = float(os.getenv("HINDSIGHT_BENCH_JUDGE_CONFIRM_TEMPERATURE", "0.5"))


async def _ask_once(system: str, prompt: str, temperature: float) -> dict:
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    raw = await _provider().call(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
        scope="benchmark_judge",
    )
    text = raw if isinstance(raw, str) else str(raw)
    try:
        parsed = parse_llm_json(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _ask(system: str, prompt: str, *, confirm_key: str | None = None) -> dict:
    """Ask the judge, confirming a negative on ``confirm_key`` before believing it."""
    verdict = await _ask_once(system, prompt, temperature=0)
    if confirm_key is None or verdict.get(confirm_key) or _CONFIRMATIONS <= 0:
        return verdict

    retries = await asyncio.gather(
        *(_ask_once(system, prompt, temperature=_CONFIRM_TEMPERATURE) for _ in range(_CONFIRMATIONS)),
        return_exceptions=True,
    )
    votes = [verdict] + [r for r in retries if isinstance(r, dict)]
    if sum(1 for vote in votes if vote.get(confirm_key)) * 2 > len(votes):
        # The majority disagreed with the primary "no"; take the first vote that
        # said yes, so its reasoning explains the verdict being returned.
        return next(vote for vote in votes if vote.get(confirm_key))
    return verdict


class AnswerVerdict(BaseModel):
    """How one answer did — and, when it failed, in which direction."""

    question: str
    correct: bool
    abstained: bool = Field(
        default=False,
        description="The answer declined to state the detail rather than inventing one. Not a success, but not a harm.",
    )
    reasoning: str = ""


class FactVerdict(BaseModel):
    claim: str
    present: bool
    reasoning: str = ""


_ANSWER_SYSTEM = (
    "You grade an answer against a reference. Reply ONLY with JSON: "
    '{"correct": true|false, "abstained": true|false, "reasoning": "one sentence"}.\n'
    "correct: the answer conveys the reference's specific detail, in any wording. Paraphrase is fine; "
    "a near-miss on a specific label, name or number is NOT correct.\n"
    "abstained: the answer says it does not know, or that the information is not available, WITHOUT "
    "asserting a specific alternative. An answer that confidently states something wrong is not abstaining."
)

_FACT_SYSTEM = (
    "You check whether a list of remembered facts contains a claim. Reply ONLY with JSON: "
    '{"present": true|false, "reasoning": "one sentence"}. '
    "Present means some fact states the claim or entails it, in any wording. Silence or a vague "
    "gesture at the same topic is not present."
)


async def judge_answer(question: str, expected: str, answer: str) -> AnswerVerdict:
    result = await _ask(
        _ANSWER_SYSTEM,
        f"Question:\n{question}\n\nReference answer:\n{expected}\n\nAnswer to grade:\n{answer or '(empty)'}",
        confirm_key="correct",
    )
    return AnswerVerdict(
        question=question,
        correct=bool(result.get("correct")),
        abstained=bool(result.get("abstained")),
        reasoning=str(result.get("reasoning", "")),
    )


async def judge_fact_present(claim: str, facts: list[str]) -> FactVerdict:
    rendered = "\n".join(f"- {fact}" for fact in facts) or "(no facts)"
    result = await _ask(_FACT_SYSTEM, f"Claim:\n{claim}\n\nRemembered facts:\n{rendered}", confirm_key="present")
    return FactVerdict(
        claim=claim,
        present=bool(result.get("present")),
        reasoning=str(result.get("reasoning", "")),
    )


async def judge_all(coroutines: list, concurrency: int = 4) -> list:
    """Run judge calls with a bound, so a big corpus cannot rate-limit itself."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_guarded(coro) for coro in coroutines))
