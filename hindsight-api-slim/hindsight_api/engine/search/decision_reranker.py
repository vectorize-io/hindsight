"""Recall's reranker when a decision model is set up for recall.

It asks the decision model of each candidate separately whether showing it would help
someone answer the question, and orders the pool by that, in one batch of requests. The
order is final: recall applies none of the recency, temporal or proof-count boosts it
applies to a cross-encoder's scores.

It replaces the deprecated ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER``):
configure the shared ``HINDSIGHT_API_DECISION_MODEL_*`` settings instead. It is handed the
shared :class:`~hindsight_api.engine.decision_model.DecisionModel` client that
``decision_engine("recall")`` returns. The deprecated provider itself runs here too, on an
engine built from its own settings: :class:`ListwiseDecisionReranker`.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass

import numpy as np

from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion
from ..decision_rerank_interface import ALIKE_FROM

logger = logging.getLogger(__name__)

PROVIDER_NAME = "decision_model"

# What every candidate in this rerank shares — the current date and the bank's own
# framing, which recall would otherwise paste onto each one. The decision model gets it in
# the question state once instead: repeated per candidate it cannot say which candidate
# beats which, and it spends the window on the same text once per candidate. Set by
# CrossEncoderReranker.rerank().
rerank_shared_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_shared_context", default=None
)


@dataclass(frozen=True)
class DecisionRerankOptions:
    """What this one recall asks of the decision model beyond ranking.

    Per recall, not per reranker: one reranker serves every bank, and the request, then the
    bank, then the server decide (``recall_decision_model_prune`` / ``_dedup``).
    ``fact_types`` and ``ids`` are each candidate's fact type and memory id, aligned with the
    pairs handed to predict(), so dedup only compares two candidates of one type.
    ``embeddings`` is the caller's read of those ids' stored embeddings (id to vector), already
    running so it costs no round trip of its own: dedup awaits it after the ranking request. It
    resolves to ``None`` when the store has none, and then dedup asks every pair.
    """

    prune: bool = False
    dedup: bool = False
    fact_types: list[str] | None = None
    ids: list[str] | None = None
    embeddings: asyncio.Task[dict[str, np.ndarray] | None] | None = None


# Set by CrossEncoderReranker.rerank(), like rerank_shared_context.
rerank_options: contextvars.ContextVar[DecisionRerankOptions] = contextvars.ContextVar(
    "hindsight_rerank_options", default=DecisionRerankOptions()
)

# The questions themselves, their state, caps, scores, the prune cut and dedup live in
# decision_rerank_interface, which services outside hindsight-api load to rank exactly as
# this does. This class only sends the questions.


class DecisionModelReranker(CrossEncoderModel):
    """Ranks a recall pool with a decision model: one Noul per candidate, asking whether
    showing it would help rather than distract.

    Every candidate lands on one absolute scale, so the order holds all the way down to
    where the caller's budget cuts. Asking one comparative Choice over the whole pool
    instead ranks the head better (recall@1 0.94 against 0.87) but not the tail, and where
    the caller keeps dozens of memories for a generator it measured far worse end to end:
    0.59 against 0.79 on a 20-question answer-quality set.

    The scores handed back are positions, not confidences: descending from 1.0 in rank
    order. With :class:`DecisionRerankOptions` ``prune`` on, a candidate judged no help scores
    exactly 0.0, and with ``dedup`` on so does one repeating a better-ranked candidate:
    that is how recall knows to drop it. The best candidate always survives.
    """

    # Its decision-model requests are traced one by one; a rerank row would count them twice.
    traces_calls = False

    def __init__(self, client: DecisionModel, *, prune: bool = False):
        # The shared engine (decision_model.decision_engine("recall")), injected.
        self._client = client
        # Prune every recall, whatever DecisionRerankOptions.prune says: the deprecated typesafe
        # reranker's HINDSIGHT_API_RERANKER_TYPESAFE_PRUNE_CANDIDATES, when it ranks per candidate.
        self.prunes_candidates = prune

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    async def initialize(self) -> None:
        pass

    async def _rerank(
        self,
        query: str,
        docs: list[str],
        indices: list[int],
        scores: list[float],
    ) -> None:
        """Scores ``indices`` by position, best first."""
        rerank = self._client.rerank_interface()
        options = rerank_options.get()
        subset = [docs[index] for index in indices]
        state = rerank.state(query, rerank_shared_context.get())
        nouls = await self._judge(query, subset)
        # Stable on the caller's input order, so equal answers keep their RRF order.
        ranked = rerank.pruned(nouls) if options.prune or self.prunes_candidates else rerank.scores(nouls)
        if options.dedup:
            # The second request: its pairs depend on the ranking answers above.
            kinds = [options.fact_types[index] for index in indices] if options.fact_types else None
            repeats = rerank.repeat_questions(subset, ranked, kinds, await self._alike(options, indices, ranked))
            if repeats:
                found = await self._client.ask_all(
                    state,
                    {q.key: DecisionQuestion.model_validate(q.system_one()) for q in repeats},
                    scope="decision_recall_dedup",
                )
                ranked = rerank.deduped(ranked, repeats, {key: answer.noul for key, answer in found.items()})
        for slot, score in enumerate(ranked):
            scores[indices[slot]] = score

    async def _judge(self, query: str, docs: list[str]) -> list[float | None]:
        """One Noul per doc, in one batch of requests: whether showing it would help answer ``query``."""
        rerank = self._client.rerank_interface()
        asked = rerank.questions(docs)
        questions = {question.key: DecisionQuestion.model_validate(question.system_one()) for question in asked}
        # Every memory is capped (see DecisionRerank.questions), so no single question can
        # outgrow the window: a failed request now raises instead of ranking its candidates last.
        answers = await self._client.ask_all(
            rerank.state(query, rerank_shared_context.get()), questions, scope="decision_recall_rerank"
        )
        return [answers[question.key].noul if question.key in answers else None for question in asked]

    @staticmethod
    async def _alike(
        options: DecisionRerankOptions, indices: list[int], ranked: list[float]
    ) -> set[frozenset[int]] | None:
        """The pairs of slots still in the pool whose stored embeddings are alike (cosine at
        least ``ALIKE_FROM``), or None to ask every pair.

        Computed here, on the candidates the ranking kept, rather than read from the semantic
        links retain stores: retain keeps only each memory's 50 nearest above its link
        threshold, so in a large bank of alike facts a repeat is often not among them. A
        candidate with no stored embedding is not compared.
        """
        if options.embeddings is None or options.ids is None:
            return None
        vectors = await options.embeddings
        if vectors is None:
            return None
        found = [slot for slot, score in enumerate(ranked) if score > 0.0 and options.ids[indices[slot]] in vectors]
        if len(found) < 2:
            return set()
        matrix = np.stack([vectors[options.ids[indices[slot]]] for slot in found])
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        similar = np.argwhere(np.triu(matrix @ matrix.T >= ALIKE_FROM, k=1))
        return {frozenset((found[a], found[b])) for a, b in similar}

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores


class ListwiseDecisionReranker(CrossEncoderModel):
    """The deprecated ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER=typesafe``),
    on the decision engine: each query's pool ranked by one listwise Choice
    (:meth:`DecisionModel.rank`), and with ``prune`` cut where relevance ends.

    The scores are positions, not confidences, and a
    candidate past the cut scores exactly 0.0, which is how :attr:`prunes_candidates` tells
    recall to leave it out.
    """

    # Its decision-model requests are traced one by one; a rerank row would count them twice.
    traces_calls = False

    def __init__(self, client: DecisionModel, *, prune: bool = False):
        self._client = client
        self.prunes_candidates = prune

    @property
    def provider_name(self) -> str:
        return "typesafe"

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def timeout(self) -> float:
        return self._client.timeout

    async def initialize(self) -> None:
        logger.info(
            f"Reranker: initializing TypeSafe provider at {self._client.base_url} with model {self.model} "
            f"(prune_candidates={self.prunes_candidates})"
        )

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        # A Choice ranks against one question, so each query's candidates are their own pool.
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        ranked = await asyncio.gather(
            *(
                self._client.rank(query, [pairs[index][1] for index in indices], prune=self.prunes_candidates)
                for query, indices in by_query.items()
            )
        )
        scores = [0.0] * len(pairs)
        for indices, pool in zip(by_query.values(), ranked):
            for index, score in zip(indices, pool):
                scores[index] = score
        return scores
