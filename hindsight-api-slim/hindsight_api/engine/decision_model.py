"""Client for a decision model: a model that answers typed questions instead of writing text.

A decision model answers typed questions (pick one of these, how likely is this, where on
this scale) about a state, in well under a second. Operations hand it the in-between
judgments an LLM call would otherwise make, and keep the LLM for writing. Every operation
reaches it through this one client, built from ``HindsightConfig.decision_model_config``.

Callers always ask in the System One shape (``noul`` / ``choice`` / ``score``, questions
keyed by id) and always get a :class:`DecisionResponse` back; the client speaks each
provider's wire format:

- ``typesafe`` (Jev): System One at ``<base_url>/v1/systemone``.
- ``cloudflare`` (Clef on Workers AI): the same System One body, posted to
  ``<base_url>/@cf/cloudflare/<model>``, where ``base_url`` is the account's
  ``.../client/v4/accounts/<account_id>/ai/run`` (or an AI Gateway ``.../workers-ai`` URL).
  The REST API wraps the answer in Cloudflare's ``{"result": ...}`` envelope.
- ``openai``: the Decisions API at ``<base_url>/v1/decisions``. Same three question kinds
  under other names (``predicate`` for noul, ``choices`` / ``levels`` for criteria), a list
  of named questions and answers instead of maps, and ``input`` instead of ``state``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from .._cross_loop import CrossLoopSemaphore
from ..config import DecisionModelConfig
from .aiohttp_session import LoopLocalSession, per_phase_timeout, raise_for_status
from .remote_retry import RetryPolicy, acall_with_retry
from .token_encoding import count_tokens

logger = logging.getLogger(__name__)


class DecisionQuestion(BaseModel):
    """One question, in the System One shape.

    ``criteria``: for a choice, option id -> description (None when the id says enough);
    for a score, the levels from lowest to highest.

    ``instructions`` may be named parts instead of one string — a task and the item it is
    about, say — which System One reads as such; providers that take a string get it as JSON.
    """

    type: Literal["noul", "choice", "score"]
    instructions: str | dict[str, str]
    criteria: dict[str, str | None] | list[str] | None = None

    def system_one(self) -> dict[str, Any]:
        # Built by hand, not model_dump(exclude_none=True): a choice option with no
        # description is sent as null, and must not be dropped from the criteria.
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body


class DecisionAnswer(BaseModel):
    """One answer. Which field is set depends on the question type: a choice fills
    ``choice`` and ``probabilities``, a noul fills ``noul``, a score fills ``score``."""

    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    noul: float | None = None
    score: float | None = None
    confidence: float | None = None


class DecisionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class DecisionResponse(BaseModel):
    #: Keyed by the question ids the caller sent.
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage = Field(default_factory=DecisionUsage)


class DecisionRefusedError(Exception):
    """The provider refused to answer a question (OpenAI's ``refusal`` answer)."""


class _CloudflareEnvelope(BaseModel):
    result: DecisionResponse


class _OpenAIProbability(BaseModel):
    value: str | int
    probability: float


class _OpenAIAnswer(BaseModel):
    type: Literal["predicate", "choice", "score", "refusal"]
    name: str
    probability: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: list[_OpenAIProbability] = Field(default_factory=list)
    confidence: float | None = None


class _OpenAIResponse(BaseModel):
    answers: list[_OpenAIAnswer]
    usage: DecisionUsage = Field(default_factory=DecisionUsage)


class _OpenAIOption(BaseModel):
    value: str
    description: str


class _OpenAILevel(BaseModel):
    label: str
    description: str


class _OpenAIQuestion(BaseModel):
    type: Literal["predicate", "choice", "score"]
    name: str
    instructions: str
    choices: list[_OpenAIOption] | None = None
    levels: list[_OpenAILevel] | None = None


def _openai_question(name: str, question: DecisionQuestion) -> _OpenAIQuestion:
    instructions = question.instructions
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    if question.type == "noul":
        return _OpenAIQuestion(type="predicate", name=name, instructions=instructions)
    criteria = question.criteria or []
    # The Decisions API wants a description for every option and level; a bare label doubles as one.
    labels = list(criteria.items()) if isinstance(criteria, dict) else [(label, None) for label in criteria]
    if question.type == "choice":
        choices = [_OpenAIOption(value=k, description=d or k) for k, d in labels]
        return _OpenAIQuestion(type="choice", name=name, instructions=instructions, choices=choices)
    levels = [_OpenAILevel(label=k, description=d or k) for k, d in labels]
    return _OpenAIQuestion(type="score", name=name, instructions=instructions, levels=levels)


def _from_openai(response: _OpenAIResponse) -> DecisionResponse:
    answers: dict[str, DecisionAnswer] = {}
    for answer in response.answers:
        if answer.type == "refusal":
            raise DecisionRefusedError(f"decision model refused to answer {answer.name!r}")
        if answer.type == "predicate":
            answers[answer.name] = DecisionAnswer(noul=answer.probability)
        elif answer.type == "choice":
            answers[answer.name] = DecisionAnswer(
                choice=answer.choice,
                probabilities={str(p.value): p.probability for p in answer.probabilities},
                confidence=answer.confidence,
            )
        else:
            answers[answer.name] = DecisionAnswer(score=answer.score, confidence=answer.confidence)
    return DecisionResponse(answers=answers, usage=response.usage)


class DecisionModel:
    # Most questions in one request. Measured on System One: roughly 750 single-candidate
    # questions go through and 775 come back max_tokens_exceeded, so this is kept clear of
    # the edge. The token ceiling is ``max_request_tokens``: it belongs to the model the
    # endpoint serves, not to this client, so it is a setting.
    MAX_QUESTIONS = 700

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_concurrent: int,
        max_request_tokens: int,
        retry_policy: RetryPolicy | None = None,
    ):
        self.provider = provider
        self.max_request_tokens = max_request_tokens
        self.retry_policy = retry_policy or RetryPolicy()
        self.model = model
        self.base_url = base_url.rstrip("/")
        # CrossLoopSemaphore, not asyncio.Semaphore: one client is built at startup and
        # reached from every loop in the process (worker threads run their own via
        # asyncio.run), and an asyncio.Semaphore binds to the first loop that waits on it.
        self._semaphore = CrossLoopSemaphore(max_concurrent)
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> DecisionModel:
        return cls(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            max_concurrent=config.max_concurrent,
            max_request_tokens=config.max_request_tokens,
        )

    async def ask(self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]) -> DecisionResponse:
        """Ask ``questions`` (keyed by caller-chosen ids) about ``state`` in one request.

        Retried on its own when it fails transiently — a 429 above all, which a single
        endpoint replica returns as soon as a few recalls overlap.
        """
        return await acall_with_retry(
            lambda: self._ask_once(state, questions),
            policy=self.retry_policy,
            budget=self.retry_policy.new_budget(),
            provider=f"decision_model:{self.provider}",
        )

    async def ask_all(
        self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]
    ) -> dict[str, DecisionAnswer]:
        """Ask ``questions`` in as many requests as it takes for each to fit the window.

        A request that still fails after its retries is logged and its answers are left
        out, so the caller decides what an unanswered question means rather than losing
        every answer to one failed request.
        """

        async def ask(part: dict[str, DecisionQuestion]) -> dict[str, DecisionAnswer]:
            try:
                return (await self.ask(state, part)).answers
            except Exception as e:
                logger.warning(f"Decision model: a request of {len(part)} questions failed: {e}")
                return {}

        answers: dict[str, DecisionAnswer] = {}
        for part in await asyncio.gather(*(ask(part) for part in self._split(state, questions))):
            answers.update(part)
        return answers

    def _split(
        self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]
    ) -> list[dict[str, DecisionQuestion]]:
        """``questions`` in groups whose request bodies each fit ``max_request_tokens``.

        Counted on the System One body — state, model and every question with its framing —
        since that is what the endpoint counts against its window. The OpenAI shape carries
        the same text under other keys. A single question over the ceiling still goes out
        alone, and fails there.
        """
        envelope = count_tokens(json.dumps({"model": self.model, "state": state, "questions": {}}))
        parts: list[dict[str, DecisionQuestion]] = []
        current: dict[str, DecisionQuestion] = {}
        used = envelope
        for key, question in questions.items():
            tokens = count_tokens(json.dumps({key: question.system_one()}))
            if current and (len(current) >= self.MAX_QUESTIONS or used + tokens > self.max_request_tokens):
                parts.append(current)
                current, used = {}, envelope
            current[key] = question
            used += tokens
        if current:
            parts.append(current)
        return parts

    async def _ask_once(self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]) -> DecisionResponse:
        if self.provider == "openai":
            # The Decisions API takes text, not a structured state.
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            body = {
                "model": self.model,
                "input": text,
                "questions": [_openai_question(name, q).model_dump(exclude_none=True) for name, q in questions.items()],
            }
            return _from_openai(_OpenAIResponse.model_validate(await self._post("/v1/decisions", body)))

        body = {
            "model": self.model,
            "state": state,
            "questions": {name: q.system_one() for name, q in questions.items()},
        }
        if self.provider == "cloudflare":
            raw = await self._post(f"/@cf/cloudflare/{self.model}", body)
            return _CloudflareEnvelope.model_validate(raw).result
        return DecisionResponse.model_validate(await self._post("/v1/systemone", body))

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        # The slot is held per attempt, so a request waiting out its backoff frees it.
        async with self._semaphore:
            async with self._session.get().post(f"{self.base_url}{path}", json=body) as response:
                await raise_for_status(response)
                return await response.json(content_type=None)
