"""Client for a decision model: a model that answers typed questions instead of writing text.

A decision model answers typed questions (pick one of these, how likely is this, where on
this scale) about a state, in well under a second. Operations hand it the in-between
judgments an LLM call would otherwise make, and keep the LLM for writing. Every operation
reaches it through this one client: :func:`decision_engine` builds it from
``HindsightConfig.decision_model_config``, once per operation and setup, and the operation
takes it as a dependency.

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
import base64
import dataclasses
import io
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from .. import config as hindsight_config
from .._cross_loop import CrossLoopSemaphore
from ..config import DecisionModelConfig, DecisionModelOperation
from ..tracing import get_span_recorder
from .aiohttp_session import LoopLocalSession, UpstreamHTTPError, per_phase_timeout, raise_for_status
from .bank_attribution import reranker_bank_attribution_headers
from .decision_rerank_interface import MAX_QUERY_TOKENS, MAX_REQUEST_TOKENS, DecisionRerank, EvidenceImage
from .remote_retry import RetryPolicy, acall_with_retry
from .token_encoding import count_tokens, truncate_to_tokens

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


class DecisionAsset(BaseModel):
    """An image a question looks at, in System One's ``assets`` shape. Its key (the placeholder) stands
    where the image belongs in the state or a question, and is replaced everywhere it appears."""

    type: Literal["image"] = "image"
    format: Literal["png", "jpg"]
    data_base64: str


#: What System One takes as an image asset (docs.typesafe.ai/images): PNG or baseline JPEG, each side
#: 280-1568 px, never resized for you. One image it refuses fails the whole request, so a caller leaves
#: out what it would refuse (``jev_accepts``).
JEV_IMAGE_FORMATS: dict[str, Literal["png", "jpg"]] = {"image/png": "png", "image/jpeg": "jpg"}
JEV_IMAGE_SIDE = (280, 1568)
JEV_IMAGE_MAX_BYTES = 16 * 1024 * 1024
#: Required on every System One request that carries an image: images are a preview there.
JEV_IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}


def jev_accepts(media_type: str, data: bytes) -> bool:
    """Whether System One takes this image as an asset."""
    if media_type not in JEV_IMAGE_FORMATS or len(data) > JEV_IMAGE_MAX_BYTES:
        return False
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as opened:  # reads the header only
            width, height = opened.size
            progressive = bool(opened.info.get("progressive"))
    except Exception:
        return False
    low, high = JEV_IMAGE_SIDE
    return low <= width <= high and low <= height <= high and not progressive


def as_assets(pictures: Mapping[str, EvidenceImage]) -> dict[str, DecisionAsset]:
    """Pictures by reference (as :meth:`DecisionModel.shown` returns them) as System One assets."""
    return {
        ref: DecisionAsset(
            format=JEV_IMAGE_FORMATS[picture.media_type],
            data_base64=base64.b64encode(picture.data).decode(),
        )
        for ref, picture in pictures.items()
    }


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


def overflowed_window(exc: BaseException) -> bool:
    """Whether System One refused a request as longer than its context window.

    The window is counted in the model's own tokens, which a client can only estimate:
    on mixed-script text Jev's tokenizer ran twice o200k_base (#5715), which is why requests
    are now sized with ``tokenizer`` ('jev' for typesafe). An estimate still misses, so a
    request sized to fit can still come back ``400 {"detail": {"error_type": "max_tokens_exceeded"}}``, or,
    from other System One servers, ``422 {"detail": "... exceeds the 32768-token context limit ..."}`` (#6522).
    Not retryable as sent; the caller has to send less.
    """
    return (
        isinstance(exc, UpstreamHTTPError)
        and exc.status_code in (400, 422)
        and ("max_tokens_exceeded" in exc.body or "context limit" in exc.body)
    )


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


def record_decision_call(
    *,
    provider: str,
    model: str,
    scope: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    input_tokens: int,
    started: float,
    error: BaseException | None,
) -> None:
    """Trace one decision-model request next to the LLM calls, under a ``decision_*`` scope."""
    get_span_recorder().record_llm_call(
        provider=provider,
        model=model,
        scope=scope,
        # No chat messages, as for embeddings: the request is not a chat, and turning it into
        # one string here would cost a json.dumps on every call even with tracing off. The row
        # stores it as a structure, serialized when the batch is written.
        messages=[],
        structured_input=request,
        response_content=response,
        input_tokens=input_tokens,
        output_tokens=0,
        duration=time.monotonic() - started,
        error=error,
    )


# Hindsight Inference refuses a body over 2 MB; this leaves room for the per-question counting
# being approximate. aiohttp sends json.dumps' ASCII output, so characters are bytes.
MAX_REQUEST_BYTES = 1_800_000

# The listwise questions: reflect's evidence pruning (rank, then cut), its "is this enough?"
# verdict and the page-impact gate of a delta page refresh. Each is one question in its own
# request, sized by _within_window.
_RANK_INSTRUCTIONS_PREFIX = "Which candidate answers the question: "
_CUT_INSTRUCTIONS = (
    "How far down this ranked list does genuine relevance to the question extend? "
    "Count a candidate as relevant only if it helps answer the question."
)
_SUFFICIENCY_INSTRUCTIONS = (
    "Does the evidence above answer the question? Judge only what the evidence states, "
    "not what could be guessed from it."
)

# The page-impact gate answers with the single most likely option, and anything but
# "no" keeps the memory. With only yes/no, a memory that bears on the topic by
# implication lost to "no": "the payment SDK vendor ends support on 1 September"
# against a page of risks to a 15 September launch got p(yes) 0.10-0.22, so a new
# launch risk never reached the page. Several ways to say "it may matter" split the
# doubt across options that all keep the memory, so "no" wins only when it is the
# model's clear answer: on replayed eval pages, 54/54 memories that should change a
# page kept (48/54 before), unrelated memories let through 13/80 (12/80 before).
_PAGE_IMPACT_INSTRUCTIONS = (
    "Is the new memory about the page's topic, or about anything the topic depends on (a "
    "dependency, vendor, date, person or system the topic relies on)? A memory that could "
    "threaten, block, delay or change what the topic is about counts, even when it never names "
    "the topic. Answer no only when it concerns a different subject altogether."
)
_PAGE_IMPACT_OPTIONS: dict[str, str | None] = {
    "yes": "Yes, it concerns the page's topic or something it depends on",
    "maybe": "Possibly, it might affect the topic or something it relies on",
    "unsure": "Unclear whether it bears on the topic",
    "depends": "It concerns something the topic relies on (a vendor, dependency, date, person or system)",
    "no": "No, a different subject altogether: nothing this page covers or relies on",
}

_OPTION_KEY_OVERHEAD = 7  # JSON envelope overhead per choice option: '"c249": "' + comma/newline
_LISTING_ITEM_OVERHEAD = 10  # Formatting overhead per item in cut listing: "[1] ...\n\n"
_T = TypeVar("_T")


class DecisionModel:
    # A request holds at most decision_rerank_interface.MAX_QUESTIONS questions. The token
    # ceiling is ``max_request_tokens``: it belongs to the model the endpoint serves, not to
    # this client, so it is a setting. None when the endpoint fits each request to its model's
    # window itself (Hindsight Inference): then only the body size is bounded.
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        max_concurrent: int,
        max_request_tokens: int | None,
        retry_policy: RetryPolicy | None = None,
        tokenizer: str | None = None,
        vision: bool = False,
    ):
        self.provider = provider
        # Whether it is shown pictures (HINDSIGHT_API_DECISION_MODEL_VISION); see shown().
        self.vision = vision
        self.max_request_tokens = max_request_tokens
        # The vocabulary max_request_tokens is counted in: the model's own when known,
        # since it is the model that refuses a request past its window.
        self.tokenizer = tokenizer
        self.retry_policy = retry_policy or RetryPolicy()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # CrossLoopSemaphore, not asyncio.Semaphore: one client is built at startup and
        # reached from every loop in the process (worker threads run their own via
        # asyncio.run), and an asyncio.Semaphore binds to the first loop that waits on it.
        self._semaphore = CrossLoopSemaphore(max_concurrent)
        self._session = LoopLocalSession(
            timeout=per_phase_timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    @property
    def takes_images(self) -> bool:
        """Whether its questions can look at images: System One's ``assets``; OpenAI's Decisions API has none."""
        return self.provider != "openai"

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
            tokenizer=config.tokenizer,
            vision=config.vision,
        )

    def shown(self, pictures: Sequence[EvidenceImage]) -> dict[str, EvidenceImage]:
        """The ``pictures`` this model is shown, keyed by the reference (``[[picture-1]]``, ...)
        the state names each by; :func:`as_assets` turns them into :meth:`ask`'s ``assets``.

        None unless vision is on and the provider takes images (:attr:`takes_images`), and
        never one System One would refuse (``jev_accepts``), since that fails the whole request.
        """
        if not self.vision or not self.takes_images:
            return {}
        accepted = [picture for picture in pictures if jev_accepts(picture.media_type, picture.data)]
        return {f"[[picture-{i + 1}]]": picture for i, picture in enumerate(accepted)}

    async def ask(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        scope: str,
        assets: dict[str, DecisionAsset] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> DecisionResponse:
        """Ask ``questions`` (keyed by caller-chosen ids) about ``state`` in one request.

        Retried on its own when it fails transiently — a 429 above all, which a single
        endpoint replica returns as soon as a few recalls overlap. A request the endpoint
        finds longer than its window is asked again in two halves, until one question
        alone is too long, which raises. Every attempt is traced under ``scope``, a
        ``decision_*`` name for what the caller is deciding. ``headers`` go on this request
        on top of the client's own (reflect's bank attribution).
        """
        try:
            return await acall_with_retry(
                lambda: self._ask_traced(state, questions, scope, assets, headers),
                policy=self.retry_policy,
                budget=self.retry_policy.new_budget(),
                provider=f"decision_model:{self.provider}",
            )
        except Exception as e:
            if not overflowed_window(e) or len(questions) < 2:
                raise
        items = list(questions.items())
        half = len(items) // 2
        logger.warning(f"Decision model: {len(items)} questions overflowed the window, asking them in two halves")
        first, second = await asyncio.gather(
            self.ask(state, dict(items[:half]), scope=scope, assets=assets, headers=headers),
            self.ask(state, dict(items[half:]), scope=scope, assets=assets, headers=headers),
        )
        return DecisionResponse(
            answers={**first.answers, **second.answers},
            usage=DecisionUsage(
                input_tokens=first.usage.input_tokens + second.usage.input_tokens,
                output_tokens=first.usage.output_tokens + second.usage.output_tokens,
            ),
        )

    async def ask_all(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        *,
        scope: str,
        assets: dict[str, DecisionAsset] | None = None,
    ) -> dict[str, DecisionAnswer]:
        """Ask ``questions`` in as many requests as it takes for each to fit the window.

        A request that still fails after its retries raises: an unanswered question is
        not a "no", and leaving it out would pass one off as the other. It used to be
        logged and left out, and recall's decision reranker then quietly ranked every
        candidate of a request Jev found too long behind the rest (#5715).
        """
        # A question looking at an image goes in a request of its own, with only its image: images are
        # not counted in the token split, and one per request is the form Jev was measured in (ViDoRe,
        # video, ATM-Bench). The text-only questions are split as before.
        seen: dict[str, dict[str, DecisionAsset]] = {}
        for key, question in questions.items() if assets else ():
            text = json.dumps(question.system_one(), ensure_ascii=False)
            if used := {name: asset for name, asset in assets.items() if name in text}:
                seen[key] = used
        text_only = {key: question for key, question in questions.items() if key not in seen}
        answers: dict[str, DecisionAnswer] = {}
        for part in await asyncio.gather(
            *(self.ask(state, part, scope=scope) for part in (self._split(state, text_only) if text_only else [])),
            *(self.ask(state, {key: questions[key]}, scope=scope, assets=used) for key, used in seen.items()),
        ):
            answers.update(part.answers)
        return answers

    def _split(
        self, state: str | dict[str, Any], questions: dict[str, DecisionQuestion]
    ) -> list[dict[str, DecisionQuestion]]:
        """``questions`` in groups whose request bodies each fit ``max_request_tokens``.

        Counted on the System One body — state, model and every question with its framing —
        since that is what the endpoint counts against its window, and in the model's own
        vocabulary when the config names it. The OpenAI shape carries the same text under
        other keys. A single question over the ceiling still goes out alone, and fails there.
        With no token ceiling, the groups only keep each body under ``MAX_REQUEST_BYTES``.
        """
        splitter = self.rerank_interface()
        if self.max_request_tokens is None:
            splitter = dataclasses.replace(splitter, count=len, max_request_tokens=MAX_REQUEST_BYTES)
        groups = splitter.split(self.model, state, {key: question.system_one() for key, question in questions.items()})
        return [{key: questions[key] for key in group} for group in groups]

    def rerank_interface(self) -> DecisionRerank:
        """Request building and splitting in this model's vocabulary and request ceiling."""
        return DecisionRerank(
            count=lambda text: count_tokens(text, self.tokenizer),
            truncate=lambda text, max_tokens: truncate_to_tokens(text, max_tokens, self.tokenizer).text,
            # With no token ceiling (the endpoint fits requests), the ceiling here only sizes the caps
            # on the query, shared context and each memory, which keep the default's.
            max_request_tokens=self.max_request_tokens or MAX_REQUEST_TOKENS,
        )

    # --- The listwise questions (see _RANK_INSTRUCTIONS_PREFIX), counted in ``tokenizer``.

    # A Choice accepts at most 255 options; stay clear of the edge. A pool larger than this or
    # over the question budget is ranked in chunks whose winners are then ranked against each
    # other, because probabilities are normalised within a call and so cannot be compared across two.
    MAX_OPTIONS = 250

    # How many of the ranked candidates the cut question is shown, and how many of each chunk's
    # best reach the finals. The cut only ever keeps a handful, so a longer list costs tokens to
    # no purpose; reflect raises it (see rank()).
    SHORTLIST = 12

    # The question budget when the setup names no window (max_request_tokens None: Hindsight
    # Inference, which fits requests itself), sized for Jev: a defensive margin (26k vs its 32k)
    # absorbs JSON framing and most cross-tokenizer divergence, but not all of it: on mixed-script
    # text Jev counts twice what we do (#5715). A request the endpoint still finds too long is
    # rebuilt at half the budget (see _within_window), down to MIN_QUESTION_TOKENS. A configured
    # window is taken as it is, since a smaller one refused every first request (#6659).
    MAX_QUESTION_TOKENS = 26_000
    # Three halvings: 26k -> 3.25k covers an 8x divergence, well past the 2x measured.
    # A request still refused there will never fit, so it fails rather than retrying on.
    MIN_QUESTION_TOKENS = 3_000

    # Ordered depths for the cut. The model picks the level; these are the granularity offered,
    # a design choice and not a tuned threshold. There is deliberately no "nothing is relevant"
    # level, so at least one candidate always survives: one weak memory the caller can dismiss
    # beats silence; adding that level cost 7% of queries returning nothing and dropped gold
    # retention from 0.81 to 0.65.
    CUT_LEVELS = [
        "Only the first candidate is relevant",
        "The first two are relevant",
        "The first three are relevant",
        "The first five are relevant",
        "The first ten are relevant",
        "All of the listed candidates are relevant",
    ]
    CUT_DEPTHS = [1, 2, 3, 5, 10, None]  # None keeps the whole shortlist

    # Ordered verdicts for reflect's sufficiency question; only the last one ends retrieval.
    # "Partly" is its own level so a half-answer keeps searching.
    SUFFICIENCY_LEVELS = [
        "The evidence does not answer the question",
        "The evidence answers part of the question; something it asks is missing",
        "The evidence fully answers the question",
    ]

    @property
    def question_tokens(self) -> int:
        """The budget a listwise question starts at: the configured window, else MAX_QUESTION_TOKENS."""
        return self.max_request_tokens or self.MAX_QUESTION_TOKENS

    def _count(self, text: str) -> int:
        return count_tokens(text, self.tokenizer)

    def _truncate(self, text: str, max_tokens: int) -> str:
        return truncate_to_tokens(text, max_tokens, self.tokenizer).text

    async def rank(
        self,
        query: str,
        docs: Sequence[str],
        *,
        prune: bool = False,
        shortlist: int | None = None,
        min_keep: int = 0,
    ) -> list[float]:
        """Scores for ``docs`` against ``query`` from one listwise Choice, and with ``prune`` a cut.

        **Rank: one Choice whose options are the candidates.** Judged together the model only has
        to say which candidate beats which; on a 200-question LoCoMo set this scored recall@1 0.94
        against 0.87 for one question per candidate (recall's DecisionModelReranker), at a
        thirtieth of the calls, though it ranks the tail worse. Pools over MAX_OPTIONS or the
        question budget are ranked in rounds (see _rank).

        **Cut: one Score over the ranked shortlist**, asking how far down relevance extends. A
        Score's levels are ordered, which a cut point needs; a "none of these" option on the
        Choice instead wins outright on hard queries (35 of 200 came back empty).

        The scores are positions, not confidences: descending from 1.0 in rank order, and 0.0
        past the cut. ``shortlist`` (default SHORTLIST) sizes the cut's list and each round's
        finalists; ``min_keep`` is the fewest candidates a cut keeps, by rank.
        """
        if not docs:
            return []
        shortlist = shortlist or self.SHORTLIST
        # The query appears in both state and instructions across rank and cut. Bounded once so
        # both share the identical query, preventing overflow in the cut and drift between the two.
        query_cap = min(MAX_QUERY_TOKENS, max(50, (self.question_tokens - 200) // 2))
        if self._count(query) > query_cap:
            query = self._truncate(query, query_cap)
        texts = list(docs)
        indices = list(range(len(texts)))
        order = await self._within_window(lambda budget: self._rank(query, texts, indices, budget, shortlist))
        keep = (
            await self._within_window(lambda budget: self._cut(query, texts, order, budget, shortlist))
            if prune
            else len(order)
        )
        keep = max(keep, min(min_keep, len(order)))
        scores = [0.0] * len(texts)
        for position, index in enumerate(order):
            scores[index] = (len(order) - position) / len(order) if position < keep else 0.0
        return scores

    async def evidence_is_sufficient(
        self, query: str, evidence: str, *, pictures: Sequence[EvidenceImage] = ()
    ) -> bool:
        """Whether ``evidence`` already answers ``query``: one Score question.

        Used by a decision-orchestrated reflect to skip the LLM turns that would otherwise decide
        to keep searching. A Score, not a yes/no Choice, for the reason the cut uses one: the
        levels are ordered, and "partly" must not be rounded up to "yes".

        ``pictures`` (the knowledge-bank pictures the evidence names by label) are shown too
        where :meth:`shown` allows, as image assets each referenced after its label.
        """
        references = self.shown(pictures)
        assets = as_assets(references)
        pictures_text = "".join(f"\n{picture.label}: {ref}" for ref, picture in references.items())
        overhead = (
            self._count(_SUFFICIENCY_INSTRUCTIONS) + sum(self._count(level) for level in self.SUFFICIENCY_LEVELS) + 50
        )
        prefix = f"Question: {query}\n\nEvidence:\n"
        question = DecisionQuestion(
            type="score", instructions=_SUFFICIENCY_INSTRUCTIONS, criteria=self.SUFFICIENCY_LEVELS
        )

        async def ask(question_budget: int) -> DecisionAnswer:
            # The pictures' own tokens are the endpoint's to count: a request it finds too long is rebuilt smaller.
            budget = max(100, question_budget - overhead - self._count(prefix) - self._count(pictures_text))
            # Keeps the head of the evidence; the agent's own results come first (pages, then
            # observations, then facts), so the tail is raw facts.
            fitted = self._truncate(evidence, budget) if self._count(evidence) > budget else evidence
            state = f"{prefix}{fitted}" + (f"\n\nPictures:{pictures_text}" if assets else "")
            return await self._ask_one(state, "sufficient", question, "decision_evidence_sufficiency", assets)

        answer = await self._within_window(ask)
        # The most likely level, not the rounded expected score: an expected 1.43 is "partly" when
        # rounded, though "fully" was the likeliest verdict at 0.54 (sde-bench boltons-budget), and
        # every such rounding cost two LLM turns. The score is the fallback for a reply without
        # per-level probabilities; a reply with neither fails, like any unanswered question.
        probabilities = answer.probabilities
        if probabilities:
            level = int(max(probabilities, key=lambda key: probabilities[key]))
        elif answer.score is not None:
            level = round(answer.score)
        else:
            raise ValueError("the decision model answered the sufficiency question with no score")
        return level >= len(self.SUFFICIENCY_LEVELS) - 1

    async def could_change_page(self, topic: str, page: str, memory: str) -> bool:
        """Whether ``memory`` could change a page about ``topic`` that reads ``page``: one Choice.

        Gates a delta page refresh: a memory answered "no" is never shown to the LLM. Worded so a
        doubt is a "yes", because a wrong "no" is silent and permanent (the refresh moves past the
        memory) while a wrong "yes" costs one LLM call, which is what every refresh pays without
        the gate.
        """
        prefix = f"Page topic: {topic}\n\nNew memory: {memory}\n\nThe page as it reads now:\n"
        overhead = (
            self._count(_PAGE_IMPACT_INSTRUCTIONS)
            + sum(self._count(option or "") for option in _PAGE_IMPACT_OPTIONS.values())
            + self._count(prefix)
            + 50
        )
        question = DecisionQuestion(
            type="choice", instructions=_PAGE_IMPACT_INSTRUCTIONS, criteria=_PAGE_IMPACT_OPTIONS
        )

        async def ask(question_budget: int) -> DecisionAnswer:
            budget = max(100, question_budget - overhead)
            # Keeps the head of the page (its first sections); a page past ~25k tokens is already
            # far over its own budget.
            fitted = self._truncate(page, budget) if self._count(page) > budget else page
            return await self._ask_one(f"{prefix}{fitted}", "impact", question, "decision_page_impact")

        answer = await self._within_window(ask)
        probabilities = answer.probabilities
        choice = answer.choice or max(probabilities, key=lambda key: probabilities[key], default="yes")
        return choice != "no"

    async def _ask_one(
        self,
        state: str,
        key: str,
        question: DecisionQuestion,
        scope: str,
        assets: dict[str, DecisionAsset] | None = None,
    ) -> DecisionAnswer:
        """One listwise question in its own request, carrying the bank attribution header."""
        response = await self.ask(
            state, {key: question}, scope=scope, assets=assets or None, headers=reranker_bank_attribution_headers()
        )
        return response.answers[key]

    async def _within_window(self, build: Callable[[int], Awaitable[_T]]) -> _T:
        """``build(budget)`` at the full question budget, halved each time the endpoint finds the
        request longer than its window, since only the endpoint can count its own tokens."""
        budget = self.question_tokens
        while True:
            try:
                return await build(budget)
            except Exception as e:
                if not overflowed_window(e) or budget // 2 < self.MIN_QUESTION_TOKENS:
                    raise
            budget //= 2
            logger.warning(f"Decision model: request overflowed the window, retrying at {budget} tokens")

    async def _rank_once(self, query: str, docs: list[str], indices: list[int]) -> list[int]:
        """Rank one group of candidates, returning their indices best first."""
        # A Choice needs two options ("criteria must map 2 or more options"), and one candidate is
        # already ranked. Guarded here, where every round asks, so a bank holding one memory does
        # not fail with a 400.
        if len(indices) < 2:
            return list(indices)
        question = DecisionQuestion(
            type="choice",
            instructions=f"{_RANK_INSTRUCTIONS_PREFIX}{query}",
            criteria={f"c{position}": docs[index] for position, index in enumerate(indices)},
        )
        probabilities = (await self._ask_one(f"Question: {query}", "rank", question, "decision_rank")).probabilities
        # Sort the option positions, not the indices themselves: the option key encodes the
        # position, and two candidates can carry the same text. An option missing from the
        # answer fails the round, as an unanswered question does.
        by_probability = sorted(range(len(indices)), key=lambda position: -probabilities[f"c{position}"])
        return [indices[position] for position in by_probability]

    async def _rank(self, query: str, docs: list[str], indices: list[int], budget: int, shortlist: int) -> list[int]:
        """Rank a whole pool best first, in rounds when it exceeds option or token limits.

        Each round's probabilities are normalised within its own call, so the winners are ranked
        against each other in a finals round. Candidates that do not make the finals keep their
        caller input order (initial RRF rank) behind the finalists, discarding intra-group model
        ranks since probabilities across separate rounds are not on a shared scale (#4599).
        """
        if not indices:
            return []

        state_tokens = self._count(f"Question: {query}")
        # +30: cushion for the question's JSON framing (type, keys) around the instructions.
        instr_tokens = self._count(f"{_RANK_INSTRUCTIONS_PREFIX}{query}") + 30
        net_budget = max(50, budget - state_tokens - instr_tokens)

        # Pre-truncate outlier documents that individually exceed the question budget, on a copy.
        effective_docs = list(docs)
        doc_tokens: dict[int, int] = {}
        for index in indices:
            tokens = self._count(effective_docs[index])
            if tokens + _OPTION_KEY_OVERHEAD > net_budget:
                cap = max(10, net_budget - _OPTION_KEY_OVERHEAD)
                effective_docs[index] = self._truncate(effective_docs[index], cap)
                tokens = self._count(effective_docs[index])
            doc_tokens[index] = tokens

        # Pack candidates into groups bounded by MAX_OPTIONS and net_budget.
        groups: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for index in indices:
            item_tokens = doc_tokens[index] + _OPTION_KEY_OVERHEAD
            if current and (len(current) >= self.MAX_OPTIONS or current_tokens + item_tokens > net_budget):
                groups.append(current)
                current, current_tokens = [index], item_tokens
            else:
                current.append(index)
                current_tokens += item_tokens
        if current:
            groups.append(current)

        if len(groups) == 1:
            return await self._rank_once(query, effective_docs, groups[0])

        # Single-candidate groups skip the preliminary round: a 1-option Choice is rejected and
        # the winner is trivial.
        multi = [i for i, group in enumerate(groups) if len(group) >= 2]
        ranked_groups: list[list[int]] = [list(group) for group in groups]
        if multi:
            ranked = await asyncio.gather(*(self._rank_once(query, effective_docs, groups[i]) for i in multi))
            for i, order in zip(multi, ranked):
                ranked_groups[i] = order

        # The top of each group advances to the finals; the rest fall back to RRF order (#4599).
        # The finals is one Choice, so the finalists must fit MAX_OPTIONS too: long documents can
        # split a pool into more than MAX_OPTIONS // shortlist groups, so the per-group quota
        # shrinks, and past MAX_OPTIONS groups the lowest-input-ranked winners overflow into the rest.
        quota = max(1, min(shortlist, self.MAX_OPTIONS // len(ranked_groups)))
        finalists = [index for group in ranked_groups for index in group[:quota]]
        rest = [index for group in ranked_groups for index in group[quota:]] + finalists[self.MAX_OPTIONS :]
        finalists = finalists[: self.MAX_OPTIONS]
        input_position = {index: position for position, index in enumerate(indices)}
        rest.sort(key=lambda index: input_position[index])

        # Finals round: if the finalists exceed the budget, cap each doc evenly.
        if sum(doc_tokens[index] + _OPTION_KEY_OVERHEAD for index in finalists) > net_budget:
            cap = max(10, (net_budget - len(finalists) * _OPTION_KEY_OVERHEAD) // len(finalists))
            for index in finalists:
                if doc_tokens[index] > cap:
                    effective_docs[index] = self._truncate(effective_docs[index], cap)

        return await self._rank_once(query, effective_docs, finalists) + rest

    async def _cut(self, query: str, docs: list[str], order: list[int], budget: int, shortlist: int) -> int:
        """How many of the ranked candidates are relevant, as the model sees it."""
        listed = order[:shortlist]
        if not listed:
            return 0
        # Instructions and levels are fixed, plus a cushion for the request's JSON framing.
        overhead = self._count(_CUT_INSTRUCTIONS) + sum(self._count(level) for level in self.CUT_LEVELS) + 50
        prefix = f"Question: {query}\n\nCandidates, already ranked best first:\n"
        available = max(100, budget - overhead - self._count(prefix))

        texts = [docs[index] for index in listed]
        tokens = [self._count(text) for text in texts]
        if sum(tokens) + len(listed) * _LISTING_ITEM_OVERHEAD > available:
            cap = max(10, (available - len(listed) * _LISTING_ITEM_OVERHEAD) // len(listed))
            texts = [self._truncate(text, cap) if t > cap else text for text, t in zip(texts, tokens)]

        listing = "\n\n".join(f"[{position + 1}] {text}" for position, text in enumerate(texts))
        question = DecisionQuestion(type="score", instructions=_CUT_INSTRUCTIONS, criteria=self.CUT_LEVELS)
        answer = await self._ask_one(f"{prefix}{listing}", "depth", question, "decision_cut")
        if answer.score is None:
            raise ValueError("the decision model answered the cut question with no score")
        level = round(answer.score)
        depth = self.CUT_DEPTHS[min(len(self.CUT_DEPTHS) - 1, max(0, level))]
        return len(listed) if depth is None else min(depth, len(listed))

    async def _ask_traced(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        scope: str,
        assets: dict[str, DecisionAsset] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> DecisionResponse:
        started = time.monotonic()
        answer: DecisionResponse | None = None
        error: BaseException | None = None
        try:
            answer = await self._ask_once(state, questions, assets, headers)
            return answer
        except BaseException as e:
            error = e
            raise
        finally:
            record_decision_call(
                provider=self.provider,
                model=self.model,
                scope=scope,
                # The images' names only: their bytes would fill the trace.
                request={
                    "state": state,
                    "questions": {k: q.system_one() for k, q in questions.items()},
                    **({"assets": sorted(assets)} if assets else {}),
                },
                # Defaults left out: an answer only fills the fields of its question's type. A
                # structure, not a JSON string: the row stores it as an object, like rerank scores.
                response=answer.model_dump(mode="json", exclude_defaults=True) if answer is not None else None,
                input_tokens=answer.usage.input_tokens if answer is not None else 0,
                started=started,
                error=error,
            )

    async def _ask_once(
        self,
        state: str | dict[str, Any],
        questions: dict[str, DecisionQuestion],
        assets: dict[str, DecisionAsset] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> DecisionResponse:
        if self.provider == "openai":
            if assets:
                raise ValueError("the openai decision model takes no images; use a System One provider")
            # The Decisions API takes text, not a structured state.
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            body = {
                "model": self.model,
                "input": text,
                "questions": [_openai_question(name, q).model_dump(exclude_none=True) for name, q in questions.items()],
            }
            raw = await self._post("/v1/decisions", body, dict(extra_headers) if extra_headers else None)
            return _from_openai(_OpenAIResponse.model_validate(raw))

        body: dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": {name: q.system_one() for name, q in questions.items()},
        }
        headers: dict[str, str] = dict(extra_headers or {})
        if assets:
            body["assets"] = {name: asset.model_dump() for name, asset in assets.items()}
            headers.update(JEV_IMAGES_HEADER)
        if self.provider == "cloudflare":
            raw = await self._post(f"/@cf/cloudflare/{self.model}", body, headers)
            return _CloudflareEnvelope.model_validate(raw).result
        return DecisionResponse.model_validate(await self._post("/v1/systemone", body, headers))

    async def _post(self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        # The slot is held per attempt, so a request waiting out its backoff frees it.
        async with self._semaphore:
            async with self._session.get().post(f"{self.base_url}{path}", json=body, headers=headers) as response:
                await raise_for_status(response)
                return await response.json(content_type=None)


#: Which decision model an operation asks, or None for none: :func:`decision_engine`, or one injected
#: through ``MemoryEngine(decision_engine_for=)``.
DecisionEngineFor = Callable[[DecisionModelOperation], DecisionModel | None]


@lru_cache(maxsize=8)
def _engine(operation: DecisionModelOperation, setup: DecisionModelConfig) -> DecisionModel:
    # One client per operation and setup for the process. Per operation, not shared between
    # them: HINDSIGHT_API_DECISION_MODEL_MAX_CONCURRENT caps each operation's requests in flight,
    # so consolidation triage cannot take the slots recall waits on.
    return DecisionModel.from_config(setup)


def decision_engine(operation: DecisionModelOperation) -> DecisionModel | None:
    """The decision model ``operation`` asks, or None when it has none set up.

    The one place an operation's decision model is built. Recall's reranker, consolidation triage, reflect
    and page refresh each take a callable of this shape as a dependency
    (``MemoryEngine(decision_engine_for=)``); this function is that callable when nothing is
    injected. Cached per operation and resolved setup (``HINDSIGHT_API_DECISION_MODEL_*`` with
    the operation's overrides), so an operation gets the same client every time while its setup
    stays the same.
    """
    setup = hindsight_config.get_config().decision_model_config(operation)
    return _engine(operation, setup) if setup is not None else None
