"""Recall's reranker when a decision model is set up for recall.

It asks the decision model of each candidate separately whether showing it would help
someone answer the question, and orders the pool by that, in one batch of requests. The
order is final: recall applies none of the recency, temporal or proof-count boosts it
applies to a cross-encoder's scores.

It replaces the deprecated ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER``):
configure the shared ``HINDSIGHT_API_DECISION_MODEL_*`` settings instead. This one is read by
``HindsightConfig.decision_model_config("recall")`` and talks to the endpoint through the
shared :class:`~hindsight_api.engine.decision_model.DecisionModel` client.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from collections.abc import Callable

from ...config import DecisionModelConfig
from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion
from ..decision_rerank_interface import MAX_MEMORY_TOKENS, DecisionRerank

PROVIDER_NAME = "decision_model"

# What every candidate in this rerank shares — the current date and the bank's own
# framing, which recall would otherwise paste onto each one. The decision model gets it in
# the question state once instead: repeated per candidate it cannot say which candidate
# beats which, and it spends the window on the same text once per candidate. Set by
# CrossEncoderReranker.rerank().
rerank_shared_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_shared_context", default=None
)

# The questions themselves, their state, caps and scores live in decision_rerank_interface,
# which services outside hindsight-api load to rank exactly as this does. Pruning below is
# opt-in and recall's own, so it stays here.

# Pruning (``prune=True``). A candidate is dropped as no help when its answer to the keep
# question falls below _PRUNE_BELOW, or below _PRUNE_BELOW_BEST of the best candidate's.
# The relative cut is needed because the answers move with the pool: asked alongside four
# memories answering at 0.9, an unrelated "staging cluster upgraded" came back 0.26, against
# 0.07 among weaker ones. On a labelled bench of pools mixing needed facts, related context
# and unrelated facts, over 6 shuffles, these two cuts lost no needed fact and dropped 58/132
# unrelated ones (0.2 alone: 42; half the best: lost two). Reusing the ranking answer costs
# no extra question. Asking separately whether a memory "has nothing to do with the
# question" was worse than useless: the model answers on topic overlap and ignores the
# negation, so it dropped the needed facts and kept the junk.
_PRUNE_BELOW = 0.2
_PRUNE_BELOW_BEST = 0.4
# A candidate that repeats one kept above it is dropped. Asked of the pair with both
# memories in the state: with both in the question's instructions the model answered
# ~0.9 for any pair, related or not, and asked of one candidate against the whole list
# above it, it dropped distinct customers in a "which customers" pool as repeats. Pairwise
# it dropped 72/72 repeats and no needed fact.
_REPEAT_INSTRUCTIONS = "Memory B is a repeat of Memory A."
_REPEAT_ABOVE = 0.5
# Pairs sharing fewer words than this (over the shorter memory) are not asked: a reworded
# repeat still shares its names and values. On the bench it asked a quarter of the pairs
# and still caught every repeat; 0.4 missed some.
_REPEAT_MIN_OVERLAP = 0.3
# Only the best candidates are checked for repeats, bounding the pairs one recall asks.
# ponytail: a repeat ranked below this survives; raise it if long pools show repeats.
_REPEAT_CANDIDATES = 40
_DATE_PREFIX = re.compile(r"^\[Date: [^\]]*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.]*")


def _date(doc: str) -> str | None:
    match = _DATE_PREFIX.match(doc)
    return match.group(0) if match else None


def _words(doc: str) -> set[str]:
    # The sentence only: the date prefix and the trailing "| When: … | Involving: …" fields
    # are shared by unrelated memories about the same day or person.
    body = _DATE_PREFIX.sub("", doc).split(" | ")[0]
    return {word for word in _WORD.findall(body.lower()) if len(word) > 2}


def _may_repeat(a: str, b: str) -> bool:
    # Two memories dated differently are two occasions — "went climbing" on the 3rd and on
    # the 10th — however alike they read; the model calls those repeats.
    if _date(a) and _date(b) and _date(a) != _date(b):
        return False
    words_a, words_b = _words(a), _words(b)
    return len(words_a & words_b) >= _REPEAT_MIN_OVERLAP * max(1, min(len(words_a), len(words_b)))


class DecisionModelReranker(CrossEncoderModel):
    """Ranks a recall pool with a decision model: one Noul per candidate, asking whether
    showing it would help rather than distract.

    Every candidate lands on one absolute scale, so the order holds all the way down to
    where the caller's budget cuts. Asking one comparative Choice over the whole pool
    instead ranks the head better (recall@1 0.94 against 0.87) but not the tail, and where
    the caller keeps dozens of memories for a generator it measured far worse end to end:
    0.59 against 0.79 on a 20-question answer-quality set.

    The scores handed back are positions, not confidences: descending from 1.0 in rank
    order. With ``prune`` on, a candidate it would leave out scores exactly 0.0, which is
    how :attr:`prunes_candidates` tells recall to drop it: one judged no help, or one that
    repeats a better-ranked candidate. The best candidate always survives.
    """

    # Its decision-model requests are traced one by one; a rerank row would count them twice.
    traces_calls = False

    def __init__(self, client: DecisionModel, prune: bool = False):
        self._client = client
        self.prunes_candidates = prune

    @classmethod
    def from_config(cls, config: DecisionModelConfig, prune: bool = False) -> DecisionModelReranker:
        return cls(DecisionModel.from_config(config), prune=prune)

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    async def initialize(self) -> None:
        pass

    async def _rerank(self, query: str, docs: list[str], indices: list[int], scores: list[float]) -> None:
        """Scores ``indices`` by position, best first."""
        rerank = self._client.rerank_interface()
        asked = rerank.questions([docs[index] for index in indices])
        questions = {question.key: DecisionQuestion.model_validate(question.system_one()) for question in asked}
        state = rerank.state(query, rerank_shared_context.get())
        # Every memory is capped (see DecisionRerank.questions), so no single question can
        # outgrow the window: a failed request now raises instead of ranking its candidates last.
        answers = await self._client.ask_all(state, questions, scope="decision_recall_rerank")
        # Stable on the caller's input order, so equal answers keep their RRF order.
        nouls = [answers[question.key].noul if question.key in answers else None for question in asked]
        ranked = rerank.scores(nouls)
        dropped: set[int] = set()
        if self.prunes_candidates:
            # The same order rerank.scores gives: by answer, ties kept in input order.
            order = sorted(range(len(nouls)), key=lambda slot: -(nouls[slot] or 0.0))
            dropped = await self._pruned(rerank, docs, indices, order, lambda slot: nouls[slot] or 0.0)
        for slot, score in enumerate(ranked):
            scores[indices[slot]] = 0.0 if slot in dropped else score

    async def _pruned(
        self,
        rerank: DecisionRerank,
        docs: list[str],
        indices: list[int],
        order: list[int],
        answer: Callable[[int], float],
    ) -> set[int]:
        """The slots to leave out: judged no help, or a repeat of one kept above them."""
        floor = max(_PRUNE_BELOW, _PRUNE_BELOW_BEST * answer(order[0]))
        relevant = [slot for position, slot in enumerate(order) if position == 0 or answer(slot) >= floor]
        dropped = set(order) - set(relevant)
        checked = relevant[:_REPEAT_CANDIDATES]
        pairs = [
            (above, slot)
            for position, slot in enumerate(checked)
            for above in checked[:position]
            if _may_repeat(docs[indices[above]], docs[indices[slot]])
        ]
        if not pairs:
            return dropped

        async def ask(above: int, slot: int) -> float:
            a, b = (rerank.capped(docs[indices[s]], MAX_MEMORY_TOKENS) for s in (above, slot))
            state = f"Memory A: {a}\nMemory B: {b}"
            question = DecisionQuestion(type="noul", instructions=_REPEAT_INSTRUCTIONS)
            found = await self._client.ask_all(state, {"repeat": question}, scope="decision_recall_prune")
            return found["repeat"].noul or 0.0

        answers = await asyncio.gather(*(ask(above, slot) for above, slot in pairs))
        repeats = {pair for pair, repeat in zip(pairs, answers) if repeat >= _REPEAT_ABOVE}
        # Greedy in rank order: a candidate goes only when it repeats one that stays, so of
        # two repeats the better-ranked one is kept.
        kept: list[int] = []
        for slot in checked:
            if any((above, slot) in repeats for above in kept):
                dropped.add(slot)
            else:
                kept.append(slot)
        return dropped

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores
