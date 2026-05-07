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


async def evaluate(
    response: str,
    criteria: str,
    context: str | None = None,
) -> JudgeVerdict:
    """Ask the judge LLM whether a response meets the given criteria.

    Args:
        response: The LLM-generated text to evaluate.
        criteria: Plain-English description of what the response should contain/satisfy.
        context: Optional context (e.g., the stored memories or query) for the judge.

    Returns:
        JudgeVerdict with meets_criteria bool and reasoning string.
    """
    judge = _get_judge()

    context_block = f"\n\nContext provided to the system:\n{context}" if context else ""

    result = await judge.call(
        messages=[
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
        ],
        response_format=JudgeVerdict,
        max_completion_tokens=256,
        temperature=0.0,
        scope="test_judge",
    )

    if isinstance(result, JudgeVerdict):
        return result

    # Fallback: parse raw dict/string
    if isinstance(result, dict):
        return JudgeVerdict(**result)
    return JudgeVerdict(**json.loads(str(result)))


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
        fail_msg = msg or f"LLM judge: criteria not met"
        raise AssertionError(
            f"{fail_msg}\n"
            f"  Criteria: {criteria}\n"
            f"  Judge reasoning: {verdict.reasoning}\n"
            f"  Response (first 300 chars): {response[:300]}"
        )
    return verdict
