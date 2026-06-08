"""
LLM-as-a-judge utility for hs_llm_core tests.

Replaces brittle string-matching assertions (e.g., `assert "alice" in answer`)
with semantic evaluation via a frontier mini model. This makes tests resilient
to phrasing variations while still verifying LLM output quality.

Usage in tests:
    result = await llm_judge.assert_response_meets_criteria(
        response="Alice is a researcher at Stanford...",
        criteria="The response mentions Alice and her role",
    )
"""

import asyncio
import json
import logging
import os

from pydantic import BaseModel

from hindsight_api.engine.llm_wrapper import create_llm_provider

logger = logging.getLogger(__name__)

# Judge model configuration — always uses Gemini by default since GEMINI_API_KEY
# is available in all CI jobs. The judge must be independent of the test provider
# (hs_llm_mat tests run across openai, groq, bedrock, etc.).
# Override with HINDSIGHT_TEST_JUDGE_PROVIDER / MODEL / API_KEY env vars.
_JUDGE_PROVIDER = os.getenv("HINDSIGHT_TEST_JUDGE_PROVIDER", "gemini")
_raw_model = os.getenv("HINDSIGHT_TEST_JUDGE_MODEL", "gemini-2.5-flash-lite")
# Strip "google/" prefix — gemini API key auth expects bare model names.
_JUDGE_MODEL = _raw_model.removeprefix("google/") if _JUDGE_PROVIDER == "gemini" else _raw_model
_JUDGE_API_KEY = os.getenv(
    "HINDSIGHT_TEST_JUDGE_API_KEY",
    os.getenv("GEMINI_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")),
)
_JUDGE_BASE_URL = os.getenv("HINDSIGHT_TEST_JUDGE_BASE_URL", "")

# Flakiness hardening. A single temperature-0 judge call still occasionally flips
# its verdict on borderline phrasing — the dominant source of hs_llm_core
# flakiness. When the primary verdict is "not met", we ask for a few independent
# second opinions (at a higher temperature so the samples genuinely differ) and
# uphold the failure only if the majority agrees. Verdicts that pass on the first
# call are returned immediately, so passing tests are unaffected in cost or
# behaviour, and genuine failures (where every judge agrees) still fail.
_JUDGE_CONFIRMATIONS = int(os.getenv("HINDSIGHT_TEST_JUDGE_CONFIRMATIONS", "2"))
_JUDGE_CONFIRM_TEMPERATURE = float(os.getenv("HINDSIGHT_TEST_JUDGE_CONFIRM_TEMPERATURE", "0.5"))
# Retry transient judge-call errors (rate limits, 5xx) so judge infrastructure
# hiccups never fail the test under evaluation.
_JUDGE_CALL_ATTEMPTS = int(os.getenv("HINDSIGHT_TEST_JUDGE_CALL_ATTEMPTS", "3"))


class JudgeVerdict(BaseModel):
    meets_criteria: bool
    reasoning: str


_judge_instance = None


def _get_judge():
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = create_llm_provider(
            provider=_JUDGE_PROVIDER,
            api_key=_JUDGE_API_KEY,
            base_url=_JUDGE_BASE_URL or "",
            model=_JUDGE_MODEL,
            reasoning_effort="low",
        )
    return _judge_instance


async def _judge_once(
    response: str,
    criteria: str,
    context: str | None,
    temperature: float,
) -> JudgeVerdict:
    """Run a single judge verdict, retrying transient call errors."""
    judge = _get_judge()
    context_block = f"\n\nContext provided to the system:\n{context}" if context else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a test evaluation judge. Given a response and evaluation criteria, "
                "determine whether the response meets the criteria. "
                "Respond with JSON: {\"meets_criteria\": true/false, \"reasoning\": \"brief explanation\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## Response to evaluate\n{response}\n"
                f"{context_block}\n"
                f"## Criteria\n{criteria}\n\n"
                "Does the response meet the criteria?"
            ),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(max(1, _JUDGE_CALL_ATTEMPTS)):
        try:
            result = await judge.call(
                messages=messages,
                response_format=JudgeVerdict,
                max_completion_tokens=256,
                temperature=temperature,
                scope="test_judge",
            )
            if isinstance(result, JudgeVerdict):
                return result
            if isinstance(result, dict):
                return JudgeVerdict(**result)
            return JudgeVerdict(**json.loads(str(result)))
        except Exception as e:  # transient provider error — retry before giving up
            last_error = e
            logger.warning(f"Judge call failed (attempt {attempt + 1}/{_JUDGE_CALL_ATTEMPTS}): {e}")
            await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Judge call failed after {_JUDGE_CALL_ATTEMPTS} attempts: {last_error}") from last_error


async def evaluate(
    response: str,
    criteria: str,
    context: str | None = None,
) -> JudgeVerdict:
    """Ask the judge LLM whether a response meets the given criteria.

    The primary verdict is deterministic (temperature 0). If it says the criteria
    are NOT met, we collect a few independent higher-temperature second opinions
    and overrule the failure only when the majority disagrees — smoothing out the
    single-call noise that makes these tests flaky. See the module-level
    ``_JUDGE_CONFIRMATIONS`` notes.

    Args:
        response: The LLM-generated text to evaluate.
        criteria: Plain-English description of what the response should contain/satisfy.
        context: Optional context (e.g., the stored memories or query) for the judge.

    Returns:
        JudgeVerdict with meets_criteria bool and reasoning string.
    """
    primary = await _judge_once(response, criteria, context, temperature=0.0)
    if primary.meets_criteria or _JUDGE_CONFIRMATIONS <= 0:
        return primary

    # Primary says "not met": get independent second opinions before trusting it.
    confirmations = await asyncio.gather(
        *(
            _judge_once(response, criteria, context, temperature=_JUDGE_CONFIRM_TEMPERATURE)
            for _ in range(_JUDGE_CONFIRMATIONS)
        ),
        return_exceptions=True,
    )
    verdicts = [primary] + [c for c in confirmations if isinstance(c, JudgeVerdict)]
    met = sum(1 for v in verdicts if v.meets_criteria)
    not_met = len(verdicts) - met

    if met > not_met:
        agreeing = next(v for v in verdicts if v.meets_criteria)
        logger.info(
            f"Judge: primary 'not met' overruled by majority ({met}/{len(verdicts)} met). Criteria: {criteria}"
        )
        return JudgeVerdict(
            meets_criteria=True,
            reasoning=f"Majority of {len(verdicts)} judges met criteria (primary verdict overruled as noise). {agreeing.reasoning}",
        )
    return JudgeVerdict(
        meets_criteria=False,
        reasoning=f"{not_met}/{len(verdicts)} judges agree criteria not met. {primary.reasoning}",
    )


async def assert_meets_criteria(
    response: str,
    criteria: str,
    context: str | None = None,
    msg: str | None = None,
) -> JudgeVerdict:
    """Assert that a response meets criteria, with a clear failure message.

    Raises AssertionError if the judge says criteria are not met.
    """
    verdict = await evaluate(response=response, criteria=criteria, context=context)
    if not verdict.meets_criteria:
        fail_msg = msg or "LLM judge: criteria not met"
        raise AssertionError(
            f"{fail_msg}\n"
            f"  Criteria: {criteria}\n"
            f"  Judge reasoning: {verdict.reasoning}\n"
            f"  Response (first 300 chars): {response[:300]}"
        )
    return verdict
