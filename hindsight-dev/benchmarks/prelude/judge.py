"""LLM judge for the answer-level eval.

Mirrors ``hindsight-api-slim/tests/llm_judge.py`` — same defaults, same
majority-confirmation trick — but lives here because that module sits under
``tests/`` and is not importable from this package.

Two rules carried over from the test judge, both load-bearing:

* **The judge is independent of the model under test.** Judging an answer with
  the same call that produced it measures nothing. Default is Gemini, overridable
  via ``HINDSIGHT_TEST_JUDGE_*``.
* **A single temperature-0 verdict flips on borderline phrasing.** A "not met"
  is re-asked at a higher temperature and upheld only on majority agreement.
  Verdicts that pass first time are returned immediately, so the common path
  costs one call.
"""

from __future__ import annotations

import asyncio
import os

from hindsight_api.engine.llm_wrapper import create_llm_provider
from pydantic import BaseModel

_PROVIDER = os.getenv("HINDSIGHT_TEST_JUDGE_PROVIDER", "gemini")
_RAW_MODEL = os.getenv("HINDSIGHT_TEST_JUDGE_MODEL", "gemini-2.5-flash-lite")
_MODEL = _RAW_MODEL.removeprefix("google/") if _PROVIDER == "gemini" else _RAW_MODEL
_API_KEY = os.getenv(
    "HINDSIGHT_TEST_JUDGE_API_KEY",
    os.getenv("GEMINI_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")),
)
_CONFIRMATIONS = int(os.getenv("HINDSIGHT_TEST_JUDGE_CONFIRMATIONS", "2"))
_CONFIRM_TEMPERATURE = float(os.getenv("HINDSIGHT_TEST_JUDGE_CONFIRM_TEMPERATURE", "0.5"))

_judge = None


class Verdict(BaseModel):
    meets_criteria: bool
    reasoning: str


def _get_judge():
    global _judge
    if _judge is None:
        if not _API_KEY:
            raise RuntimeError(
                "The answer eval needs a judge model. Set GEMINI_API_KEY (or "
                "HINDSIGHT_TEST_JUDGE_API_KEY / _PROVIDER / _MODEL)."
            )
        _judge = create_llm_provider(
            provider=_PROVIDER, api_key=_API_KEY, base_url="", model=_MODEL, reasoning_effort=None
        )
    return _judge


async def _judge_once(response: str, criteria: str, context: str | None, temperature: float) -> Verdict:
    result = await _get_judge().call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You grade whether an answer satisfies a stated criterion. Judge ONLY the "
                    "criterion given — not style, length, or anything else. Answer strictly."
                ),
            },
            {
                "role": "user",
                "content": (
                    (f"## Context\n{context}\n\n" if context else "")
                    + f"## Criterion\n{criteria}\n\n## Answer to grade\n{response}"
                ),
            },
        ],
        response_format=Verdict,
        scope="prelude_eval_judge",
        temperature=temperature,
        max_retries=3,
    )
    content = result.content
    return content if isinstance(content, Verdict) else Verdict(**content)


async def evaluate(response: str, criteria: str, context: str | None = None) -> Verdict:
    """Grade ``response`` against ``criteria``, smoothing single-call judge noise."""
    primary = await _judge_once(response, criteria, context, temperature=0.0)
    if primary.meets_criteria or _CONFIRMATIONS <= 0:
        return primary

    confirmations = await asyncio.gather(
        *(_judge_once(response, criteria, context, _CONFIRM_TEMPERATURE) for _ in range(_CONFIRMATIONS)),
        return_exceptions=True,
    )
    verdicts = [primary] + [c for c in confirmations if isinstance(c, Verdict)]
    met = sum(1 for v in verdicts if v.meets_criteria)
    if met > len(verdicts) - met:
        return Verdict(
            meets_criteria=True,
            reasoning=f"majority of {len(verdicts)} judges met criteria (primary overruled as noise)",
        )
    return Verdict(
        meets_criteria=False,
        reasoning=f"{len(verdicts) - met}/{len(verdicts)} judges agree: {primary.reasoning}",
    )
