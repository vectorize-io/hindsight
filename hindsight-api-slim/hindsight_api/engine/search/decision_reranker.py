"""Recall's reranker when a decision model is set up for recall.

It asks the decision model of each candidate separately whether showing it would help
someone answer the question, and orders the pool by that, in one batch of requests. The
order is final: recall applies none of the recency, temporal or proof-count boosts it
applies to a cross-encoder's scores.

This is not the ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER``),
which is configured on its own and ranks differently. This one is configured by
``HindsightConfig.decision_model_config("recall")`` and talks to the endpoint through the
shared :class:`~hindsight_api.engine.decision_model.DecisionModel` client.
"""

from __future__ import annotations

import asyncio
import contextvars

from ...config import DecisionModelConfig
from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion
from ..token_encoding import count_tokens, truncate_to_tokens

PROVIDER_NAME = "decision_model"

# What every candidate in this rerank shares — the current date and the bank's own
# framing, which recall would otherwise paste onto each one. The decision model gets it in
# the question state once instead: repeated per candidate it cannot say which candidate
# beats which, and it spends the window on the same text once per candidate. Set by
# CrossEncoderReranker.rerank().
rerank_shared_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_shared_context", default=None
)

# Defensive ceilings on the query and the shared context, which every request carries:
# past the request ceiling, every request would fail. Each also stays within a quarter of
# that ceiling, so a low one still leaves half of each request for the questions.
_MAX_QUERY_TOKENS = 2_000
_MAX_SHARED_CONTEXT_TOKENS = 4_000
# The same for each candidate's text, which one question carries alone.
_MAX_MEMORY_TOKENS = 4_000

# The question asked of each candidate. Four wordings were tried on the same pool; this
# one is the best calibrated, lifting the wanted candidate to 0.74 against 0.59 for
# "contains information needed to answer" while holding the pool median at 0.08. Framing it
# as help-versus-distract asks the question a keep/drop decision actually turns on.
_KEEP_INSTRUCTIONS = "Showing this memory to someone answering the question would help them, rather than distract them."


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

    def _capped(self, text: str, ceiling: int) -> str:
        ceiling = min(ceiling, self._client.max_request_tokens // 4)
        return truncate_to_tokens(text, ceiling).text if count_tokens(text) > ceiling else text

    def _state(self, query: str) -> str:
        """The question's state: what is being asked, plus what every candidate shares."""
        query = self._capped(query, _MAX_QUERY_TOKENS)
        shared = self._capped(rerank_shared_context.get() or "", _MAX_SHARED_CONTEXT_TOKENS)
        return f"{shared}\n\nQuestion: {query}" if shared else f"Question: {query}"

    async def _rerank(self, query: str, docs: list[str], indices: list[int], scores: list[float]) -> None:
        """Scores ``indices`` by position, best first."""
        # Each memory capped like the query, so no single question can outgrow the window:
        # a failed request now raises instead of ranking its candidates last.
        questions = {
            f"m{slot}": DecisionQuestion(
                type="noul",
                instructions={"task": _KEEP_INSTRUCTIONS, "memory": self._capped(docs[index], _MAX_MEMORY_TOKENS)},
            )
            for slot, index in enumerate(indices)
        }
        answers = await self._client.ask_all(self._state(query), questions, scope="decision_recall_rerank")

        def answer(slot: int) -> float:
            found = answers.get(f"m{slot}")
            return found.noul if found is not None and found.noul is not None else 0.0

        # Stable on the caller's input order, so equal answers keep their RRF order.
        order = sorted(range(len(indices)), key=lambda slot: -answer(slot))
        for position, slot in enumerate(order):
            scores[indices[slot]] = (len(order) - position) / len(order)

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores
