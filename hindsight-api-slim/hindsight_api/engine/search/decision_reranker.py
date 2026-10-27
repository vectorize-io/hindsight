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

from ...config import DecisionModelConfig
from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion

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
# which services outside hindsight-api load to rank exactly as this does.


class DecisionModelReranker(CrossEncoderModel):
    """Ranks a recall pool with a decision model: one Noul per candidate, asking whether
    showing it would help rather than distract.

    Every candidate lands on one absolute scale, so the order holds all the way down to
    where the caller's budget cuts. Asking one comparative Choice over the whole pool
    instead ranks the head better (recall@1 0.94 against 0.87) but not the tail, and where
    the caller keeps dozens of memories for a generator it measured far worse end to end:
    0.59 against 0.79 on a 20-question answer-quality set.

    The scores handed back are positions, not confidences: descending from 1.0 in rank
    order.
    """

    # Its decision-model requests are traced one by one; a rerank row would count them twice.
    traces_calls = False

    def __init__(self, client: DecisionModel):
        self._client = client

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> DecisionModelReranker:
        return cls(DecisionModel.from_config(config))

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
        for slot, score in enumerate(rerank.scores(nouls)):
            scores[indices[slot]] = score

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores
