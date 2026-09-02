"""LLM judging for the half of document quality that structure cannot measure.

Two questions, deliberately separate:

- **Coverage** — after all the rounds, does the document actually say what the
  facts said, and has it stopped saying what they superseded? Judged one claim
  at a time against the final document, so a miss points at a specific fact.
- **Preference** — put two builds' final documents side by side and ask which
  one better serves the topic. Absolute quality scores from an LLM do not
  calibrate; a blind pairwise choice does. Every pair is judged twice with the
  documents swapped, and a pair that flips its answer is recorded as a tie
  rather than silently counted for whichever side happened to go first.
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import BaseModel, Field

_JUDGE_PROVIDER = os.getenv("HINDSIGHT_TEST_JUDGE_PROVIDER", "gemini")
_JUDGE_MODEL = os.getenv("HINDSIGHT_TEST_JUDGE_MODEL", "gemini-3.1-flash-lite")
_JUDGE_API_KEY = os.getenv(
    "HINDSIGHT_TEST_JUDGE_API_KEY",
    os.getenv("GEMINI_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")),
)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reasoning: str = ""


class CoverageResult(BaseModel):
    """How much of the fact stream survived into the final document."""

    stated: list[ClaimVerdict] = Field(default_factory=list, description="Claims the document should support.")
    superseded: list[ClaimVerdict] = Field(
        default_factory=list, description="Claims the document should no longer make."
    )

    @property
    def recall(self) -> float:
        return _ratio(sum(v.supported for v in self.stated), len(self.stated))

    @property
    def staleness(self) -> float:
        """Fraction of superseded claims the document still makes. Lower is better."""
        return _ratio(sum(v.supported for v in self.superseded), len(self.superseded))


class PreferenceResult(BaseModel):
    winner: str = Field(description="A build label, or 'tie'.")
    reasoning: str = ""
    flipped: bool = Field(default=False, description="The two orderings disagreed, so this is a tie.")


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 1.0


_provider_instance = None


def _provider():
    """One judge provider for the whole comparison — it is stateless and reused."""
    global _provider_instance
    if _provider_instance is None:
        from hindsight_api.engine.llm_wrapper import create_llm_provider

        _provider_instance = create_llm_provider(
            provider=_JUDGE_PROVIDER,
            api_key=_JUDGE_API_KEY,
            base_url=os.getenv("HINDSIGHT_TEST_JUDGE_BASE_URL", ""),
            model=_JUDGE_MODEL,
            # No reasoning_effort: the default judge is Gemini, which has no such
            # control and logs a warning for every call when one is set.
            reasoning_effort="",
        )
    return _provider_instance


async def _ask(prompt: str, system: str) -> dict:
    provider = _provider()
    raw = await provider.call(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0,
        scope="benchmark_judge",
    )
    from hindsight_api.engine.llm_wrapper import parse_llm_json

    text = raw if isinstance(raw, str) else str(raw)
    try:
        return parse_llm_json(text)
    except json.JSONDecodeError:
        return {}


_CLAIM_SYSTEM = (
    "You check whether a document supports a claim. Answer ONLY with JSON: "
    '{"supported": true|false, "reasoning": "one sentence"}. '
    "A claim is supported when the document states it or states something that entails it, in any wording. "
    "It is not supported when the document is silent, vague, or says something incompatible."
)

_PREFERENCE_SYSTEM = (
    "You compare two versions of the same reference document and pick the better one. Answer ONLY with JSON: "
    '{"winner": "A"|"B"|"tie", "reasoning": "one or two sentences"}. '
    "Judge, in order of importance: (1) does it answer the topic completely and without contradicting itself, "
    "(2) is specific detail retained rather than flattened into generalities, "
    "(3) is it well formed — tables that are tables, lists that are lists, code that is fenced, "
    "(4) is it readable. Ignore length for its own sake. Answer 'tie' only when neither is better."
)


# A single temperature-0 judge call still flips on a borderline claim — a
# document saying an operation "synthesizes memories" was once scored as not
# supporting "answers questions over memories", which is a difference in wording
# rather than in content. Asking three times and taking the majority costs a few
# calls and stops one pedantic reading from moving a build's score.
_CLAIM_VOTES = 3


async def judge_coverage(document: str, stated: list[str], superseded: list[str]) -> CoverageResult:
    """Check every claim against the final document, by majority of three."""

    async def verdict(claim: str) -> ClaimVerdict:
        prompt = f"DOCUMENT:\n---\n{document}\n---\n\nCLAIM: {claim}"
        payloads = await asyncio.gather(*(_ask(prompt, _CLAIM_SYSTEM) for _ in range(_CLAIM_VOTES)))
        votes = [bool(p.get("supported")) for p in payloads]
        supported = sum(votes) * 2 > len(votes)
        # Keep a reasoning from the losing side when the vote was not unanimous:
        # a split verdict is the interesting one to read.
        dissent = next((p for p, v in zip(payloads, votes, strict=False) if v is not supported), None)
        agreeing = next((p for p, v in zip(payloads, votes, strict=False) if v is supported), {})
        reasoning = str((dissent or agreeing).get("reasoning") or "")
        if dissent is not None:
            reasoning = f"[{sum(votes)}/{len(votes)} supported] {reasoning}"
        return ClaimVerdict(claim=claim, supported=supported, reasoning=reasoning)

    stated_verdicts, superseded_verdicts = await asyncio.gather(
        asyncio.gather(*(verdict(c) for c in stated)),
        asyncio.gather(*(verdict(c) for c in superseded)),
    )
    return CoverageResult(stated=list(stated_verdicts), superseded=list(superseded_verdicts))


async def judge_preference(topic: str, label_a: str, doc_a: str, label_b: str, doc_b: str) -> PreferenceResult:
    """Blind pairwise comparison, run in both orderings to cancel position bias."""

    async def once(first: str, second: str) -> str:
        payload = await _ask(
            f"TOPIC: {topic}\n\nDOCUMENT A:\n---\n{first}\n---\n\nDOCUMENT B:\n---\n{second}\n---",
            _PREFERENCE_SYSTEM,
        )
        return str(payload.get("winner") or "tie").strip().lower()

    forward, backward = await asyncio.gather(once(doc_a, doc_b), once(doc_b, doc_a))
    # In the backward run the labels are swapped, so "a" there means label_b won.
    forward_winner = {"a": label_a, "b": label_b}.get(forward, "tie")
    backward_winner = {"a": label_b, "b": label_a}.get(backward, "tie")

    if forward_winner == backward_winner:
        return PreferenceResult(winner=forward_winner)
    if "tie" in (forward_winner, backward_winner):
        # One ordering was decisive and the other was not: keep the decisive one
        # rather than throwing away a real signal.
        decisive = forward_winner if forward_winner != "tie" else backward_winner
        return PreferenceResult(winner=decisive, reasoning="one ordering was undecided")
    return PreferenceResult(winner="tie", reasoning="the two orderings disagreed", flipped=True)
