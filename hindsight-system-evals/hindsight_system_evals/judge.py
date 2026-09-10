"""An independent LLM judge.

Two rules carried over from ``hindsight-api-slim/tests/llm_judge.py``, both
load-bearing:

* **The judge must not be the model under test.** Judging an answer with the same
  model that wrote it measures agreement with itself. It is configured separately
  from the server's provider; the fixtures warn when the two resolve to the same
  model.
* **A single temperature-0 verdict flips on borderline phrasing.** A "not met" is
  re-asked at a higher temperature and upheld only on majority agreement, so one
  noisy sample cannot fail a run. Verdicts that pass first time cost one call.

Two backends, because the environments differ: a developer usually has a Gemini
API key, while the perf workflow authenticates every quality benchmark with a
VertexAI service account. Both go through ``google-genai`` — never through
``hindsight_api``, so nothing here can depend on engine internals.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from functools import cache

from google import genai
from google.genai import types

_CONFIRMATIONS = int(os.getenv("HINDSIGHT_EVAL_JUDGE_CONFIRMATIONS", "2"))
_CONFIRM_TEMPERATURE = float(os.getenv("HINDSIGHT_EVAL_JUDGE_CONFIRM_TEMPERATURE", "0.5"))

_SYSTEM = (
    "You grade whether an answer satisfies a stated criterion. Judge ONLY the criterion "
    "given — not style, length, or anything else. Reply with a single JSON object: "
    '{"meets_criteria": true|false, "reasoning": "<one sentence>"}. No prose outside it.'
)


@dataclass(frozen=True)
class Verdict:
    meets_criteria: bool
    reasoning: str


def judge_model() -> str:
    # The perf workflow names Vertex models with a "google/" prefix; the SDK wants
    # the bare name. Accepting both keeps one env convention across benchmarks.
    return os.getenv("HINDSIGHT_EVAL_JUDGE_MODEL", "gemini-2.5-flash").removeprefix("google/")


def _judge_api_key() -> str:
    return os.getenv("HINDSIGHT_EVAL_JUDGE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


@cache
def _client() -> genai.Client:
    """An API-key client when a key is set, otherwise VertexAI from a service account."""
    provider = os.getenv("HINDSIGHT_EVAL_JUDGE_PROVIDER", "gemini" if _judge_api_key() else "vertexai")
    if provider == "gemini":
        key = _judge_api_key()
        if not key:
            raise RuntimeError(
                "The judge needs HINDSIGHT_EVAL_JUDGE_API_KEY (or GEMINI_API_KEY), or "
                "HINDSIGHT_EVAL_JUDGE_PROVIDER=vertexai with a service account. It is configured "
                "separately from the server on purpose — a model grading its own output agrees with itself."
            )
        return genai.Client(api_key=key)

    from google.oauth2 import service_account

    key_file = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_SERVICE_ACCOUNT_KEY") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY", ""
    )
    project = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_PROJECT_ID") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID", ""
    )
    if not (key_file and project):
        raise RuntimeError(
            "A VertexAI judge needs a service-account key file and project id "
            "(HINDSIGHT_EVAL_JUDGE_VERTEXAI_* or the HINDSIGHT_API_LLM_VERTEXAI_* equivalents)."
        )
    credentials = service_account.Credentials.from_service_account_file(
        key_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    location = os.getenv("HINDSIGHT_EVAL_JUDGE_VERTEXAI_REGION") or os.getenv(
        "HINDSIGHT_API_LLM_VERTEXAI_REGION", "us-central1"
    )
    return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)


async def _judge_once(response: str, criteria: str, context: str | None, temperature: float) -> Verdict:
    prompt = (
        f"## Context\n{context}\n\n" if context else ""
    ) + f"## Criterion\n{criteria}\n\n## Answer to grade\n{response}"
    reply = await _client().aio.models.generate_content(
        model=judge_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    text = reply.text or ""
    try:
        parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except ValueError:
        # A judge that returned something unreadable must not silently pass the
        # thing it was asked to check.
        return Verdict(False, f"judge reply was not JSON: {text[:200]}")
    return Verdict(bool(parsed.get("meets_criteria")), str(parsed.get("reasoning", "")))


async def evaluate(response: str, criteria: str, context: str | None = None) -> Verdict:
    """Grade ``response`` against ``criteria``, smoothing single-call judge noise."""
    primary = await _judge_once(response, criteria, context, temperature=0.0)
    if primary.meets_criteria or _CONFIRMATIONS <= 0:
        return primary

    confirmations = await asyncio.gather(
        *(_judge_once(response, criteria, context, _CONFIRM_TEMPERATURE) for _ in range(_CONFIRMATIONS)),
        return_exceptions=True,
    )
    verdicts = [primary] + [v for v in confirmations if isinstance(v, Verdict)]
    met = sum(1 for v in verdicts if v.meets_criteria)
    if met > len(verdicts) - met:
        return Verdict(True, f"majority of {len(verdicts)} judges met the criteria (primary overruled as noise)")
    return Verdict(False, f"{len(verdicts) - met}/{len(verdicts)} judges agree: {primary.reasoning}")
