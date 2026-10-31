"""Recall's decision-model reranker: one question per candidate, ordered by the answers."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pytest

from hindsight_api.config import DecisionModelConfig
from hindsight_api.engine import decision_rerank_interface
from hindsight_api.engine.decision_model import DecisionAnswer, DecisionModel
from hindsight_api.engine.decision_rerank_interface import MAX_MEMORY_TOKENS, MAX_SHARED_CONTEXT_TOKENS
from hindsight_api.engine.search.decision_reranker import (
    DecisionModelReranker,
    DecisionRerankOptions,
    rerank_options,
    rerank_shared_context,
)
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.token_encoding import count_tokens


class _Client:
    """Answers each candidate's question from ``keep``, by memory text; a dedup question
    is a repeat when its (Memory A, Memory B) texts are in ``repeats``."""

    def __init__(self, keep=None, repeats=()):
        self.keep = keep or {}
        self.repeats = set(repeats)
        self.asked: list[tuple[str, dict]] = []
        self.scopes: list[str] = []
        self.max_request_tokens = 26_000
        self.tokenizer = None

    rerank_interface = DecisionModel.rerank_interface

    async def ask_all(self, state, questions, *, scope):
        self.asked.append((state, questions))
        self.scopes.append(scope)
        return {key: DecisionAnswer(noul=self._answer(question.instructions)) for key, question in questions.items()}

    def _answer(self, instructions) -> float:
        if "memory_a" in instructions:
            return 0.9 if (instructions["memory_a"], instructions["memory_b"]) in self.repeats else 0.1
        return self.keep.get(instructions["memory"], 0.5)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [
            (q.instructions["memory_a"], q.instructions["memory_b"])
            for _, questions in self.asked
            for q in questions.values()
            if "memory_a" in q.instructions
        ]


def _reranker(client: _Client) -> DecisionModelReranker:
    return DecisionModelReranker(client)  # type: ignore[arg-type]


def _candidate(text: str, context: str | None = None) -> MergedCandidate:
    return MergedCandidate(
        retrieval=RetrievalResult(id=text, text=text, fact_type="world", context=context),
        rrf_score=0.0,
        source_ranks={},
    )


class TestRanking:
    async def test_the_scores_order_the_candidates(self):
        reranker = _reranker(_Client(keep={"a": 0.1, "b": 0.9, "c": 0.5}))
        scores = await reranker.predict([("q", "a"), ("q", "b"), ("q", "c")])
        assert scores[1] > scores[2] > scores[0]

    async def test_every_candidate_is_asked_about_and_the_query_is_the_state(self):
        client = _Client()
        reranker = _reranker(client)
        await reranker.predict([("what changed?", "a"), ("what changed?", "b"), ("what changed?", "c")])
        state, questions = client.asked[0]
        assert state == "Question: what changed?"
        assert [q.instructions["memory"] for q in questions.values()] == ["a", "b", "c"]
        assert all(q.type == "noul" for q in questions.values())

    async def test_candidates_scored_the_same_keep_their_input_order(self):
        """Equal scores must not reshuffle the pool: input order is the RRF order."""
        reranker = _reranker(_Client())
        scores = await reranker.predict([("q", f"candidate_{i}") for i in range(4)])
        assert scores[0] > scores[1] > scores[2] > scores[3]

    async def test_an_unanswered_candidate_is_kept_behind_the_answered(self):
        """A candidate the model gave no answer for is ranked last, not dropped: an
        unranked candidate is recoverable; a deleted one is not."""
        client = _Client(keep={"b": 0.9})
        reranker = _reranker(client)

        async def partial(state, questions, *, scope):
            return {"m1": DecisionAnswer(noul=0.9)}

        client.ask_all = partial  # type: ignore[method-assign]
        scores = await reranker.predict([("q", "a"), ("q", "b"), ("q", "c")])
        assert scores[1] > scores[0] > scores[2] > 0.0

    def test_it_is_built_for_recall_when_recall_has_a_decision_model(self, monkeypatch):
        """And replaces the configured reranker outright."""
        from hindsight_api import config as config_module

        monkeypatch.setenv("HINDSIGHT_API_DECISION_MODEL_PROVIDER", "typesafe")
        monkeypatch.setenv("HINDSIGHT_API_DECISION_MODEL_API_KEY", "k")
        monkeypatch.setattr(config_module, "get_config", config_module.HindsightConfig.from_env)
        assert isinstance(CrossEncoderReranker().cross_encoder, DecisionModelReranker)

        monkeypatch.setenv("HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER", "none")
        assert not isinstance(CrossEncoderReranker().cross_encoder, DecisionModelReranker)

    def test_from_config(self):
        config = DecisionModelConfig(
            provider="typesafe",
            api_key="k",
            model="jev-latest",
            base_url="https://example.test",
            timeout=5.0,
            max_concurrent=2,
            max_request_tokens=12_345,
        )
        assert DecisionModelReranker.from_config(config)._client.max_request_tokens == 12_345


class TestSharedState:
    """What every candidate shares goes in the question's state once, not onto each."""

    PROFILE = (
        "Conversation 1 - USER PROFILE:\n"
        "  Name: Craig Baker\n"
        "  Age: 49 years old\n"
        "  Location: Port Matthew\n"
        "  Occupation: backend developer building a personal budget tracker\n"
        "  Preferred stack: Flask, Postgres, plain CSS"
    )

    async def test_the_shared_context_goes_in_the_state_once(self):
        """Not onto each candidate: identical text cannot rank them, and it costs the
        window once per candidate."""
        client = _Client()
        reranker = _reranker(client)
        token = rerank_shared_context.set(self.PROFILE)
        try:
            await reranker.predict([("q", "The deadline is April 5"), ("q", "The team chose Postgres")])
        finally:
            rerank_shared_context.reset(token)
        state, questions = client.asked[0]
        assert state == f"{self.PROFILE}\n\nQuestion: q"
        assert [q.instructions["memory"] for q in questions.values()] == [
            "The deadline is April 5",
            "The team chose Postgres",
        ]

    async def test_a_huge_shared_context_is_capped(self):
        """It rides on every request, so uncapped it could push each one past the ceiling."""
        client = _Client()
        reranker = _reranker(client)
        token = rerank_shared_context.set("profile line\n" * 5_000)
        try:
            await reranker.predict([("q", "a"), ("q", "b")])
        finally:
            rerank_shared_context.reset(token)
        assert count_tokens(client.asked[0][0]) <= MAX_SHARED_CONTEXT_TOKENS + count_tokens("\n\nQuestion: q")

    async def test_a_huge_memory_is_capped(self):
        """One question carries one memory; uncapped, a single long one could outgrow the
        window and fail the whole rerank."""
        client = _Client()
        await _reranker(client).predict([("q", "word " * 20_000), ("q", "short")])
        memories = [q.instructions["memory"] for q in client.asked[0][1].values()]
        assert count_tokens(memories[0]) <= MAX_MEMORY_TOKENS
        assert memories[1] == "short"

    async def test_the_caps_count_in_the_models_own_vocabulary(self):
        """Capped by o200k_base, a mixed-script memory would still run past 4k by Jev's count."""
        client = _Client()
        client.tokenizer = "jev"
        await _reranker(client).predict([("q", "東京 café naïve 🎉 " * 3_000), ("q", "short")])
        memory = next(iter(client.asked[0][1].values())).instructions["memory"]
        assert count_tokens(memory, "jev") <= MAX_MEMORY_TOKENS
        assert count_tokens(memory) < MAX_MEMORY_TOKENS - 500

    async def test_a_low_request_ceiling_shrinks_the_caps(self):
        """A ceiling below the caps would otherwise fail every request."""
        client = _Client()
        client.max_request_tokens = 1_000
        reranker = _reranker(client)
        token = rerank_shared_context.set("profile line\n" * 5_000)
        try:
            await reranker.predict([("long question " * 2_000, "a"), ("long question " * 2_000, "b")])
        finally:
            rerank_shared_context.reset(token)
        assert count_tokens(client.asked[0][0]) <= 520

    async def test_recall_tells_the_model_the_question_date_and_shared_context(self):
        client = _Client()
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        await reranker.rerank(
            "q",
            [_candidate("a", self.PROFILE), _candidate("b", self.PROFILE)],
            now=datetime(2024, 5, 1, 9, 30, tzinfo=UTC),
        )
        state, questions = client.asked[0]
        assert state == f"Current date: 2024-05-01\n\n{self.PROFILE}\n\nQuestion: q"
        assert [q.instructions["memory"] for q in questions.values()] == ["a", "b"]

    async def test_a_memory_with_no_event_date_shows_the_day_it_was_stored(self):
        """Without it the model cannot tell yesterday's memory from last year's."""
        client = _Client()
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        stored = RetrievalResult(
            id="pottery",
            text="I went to a pottery class.",
            fact_type="world",
            mentioned_at=datetime(2024, 6, 14, tzinfo=UTC),
        )
        dated = RetrievalResult(
            id="concert",
            text="I went to a jazz concert.",
            fact_type="world",
            occurred_start=datetime(2024, 6, 15, tzinfo=UTC),
            mentioned_at=datetime(2024, 6, 16, tzinfo=UTC),
        )
        await reranker.rerank(
            "what did I do yesterday?",
            [MergedCandidate(retrieval=r, rrf_score=0.0, source_ranks={}) for r in (stored, dated)],
            now=datetime(2024, 6, 16, tzinfo=UTC),
        )
        assert [q.instructions["memory"] for q in client.asked[0][1].values()] == [
            "[Date: June 14, 2024 (2024-06-14)] I went to a pottery class.",
            "[Date: June 15, 2024 (2024-06-15)] I went to a jazz concert.",
        ]

    async def test_without_a_question_date_it_is_today(self):
        client = _Client()
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        before = datetime.now(UTC).date()
        await reranker.rerank("q", [_candidate("a"), _candidate("b")])
        after = datetime.now(UTC).date()
        assert client.asked[0][0] in {f"Current date: {day.isoformat()}\n\nQuestion: q" for day in (before, after)}


@contextmanager
def _options(**options) -> Iterator[None]:
    """What CrossEncoderReranker.rerank() sets for one recall."""
    token = rerank_options.set(DecisionRerankOptions(**options))
    try:
        yield
    finally:
        rerank_options.reset(token)


async def _predict(client: _Client, docs: list[str], **options) -> list[float]:
    with _options(**options):
        return await _reranker(client).predict([("q", doc) for doc in docs])


TEA, TEA_AGAIN = "Marta favourite tea is genmaicha", "Marta favorite tea is genmaicha"
SENCHA = "Marta drinks sencha"


class TestPrune:
    async def test_off_by_default_nothing_is_dropped(self):
        client = _Client({"tea is genmaicha": 0.9, "offsite in lisbon": 0.01})
        assert all(score > 0.0 for score in await _predict(client, ["tea is genmaicha", "offsite in lisbon"]))

    async def test_a_candidate_judged_no_help_is_dropped_but_the_best_always_survives(self):
        client = _Client({"tea is genmaicha": 0.9, "offsite in lisbon": 0.1})
        assert await _predict(client, ["offsite in lisbon", "tea is genmaicha"], prune=True) == [0.0, 1.0]

        # Relative to the best: 0.3 clears the floor but not 40% of 0.9.
        crowded = _Client({"tea is genmaicha": 0.9, "offsite in lisbon": 0.3})
        assert (await _predict(crowded, ["offsite in lisbon", "tea is genmaicha"], prune=True))[0] == 0.0

        lonely = _Client({"offsite in lisbon": 0.05})
        assert (await _predict(lonely, ["offsite in lisbon"], prune=True))[0] > 0.0

    async def test_pruning_asks_nothing_more_than_ranking(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.85}, repeats={(TEA, TEA_AGAIN)})
        assert all(score > 0.0 for score in await _predict(client, [TEA, TEA_AGAIN], prune=True))
        assert client.scopes == ["decision_recall_rerank"]

    async def test_recall_leaves_the_dropped_candidates_out(self):
        client = _Client({"tea is genmaicha": 0.9, "offsite in lisbon": 0.1})
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        candidates = [_candidate("offsite in lisbon"), _candidate("tea is genmaicha")]
        assert len((await reranker.rerank("q", candidates)).results) == 2
        result = await reranker.rerank("q", candidates, prune=True)
        assert [r.candidate.retrieval.text for r in result.results] == ["tea is genmaicha"]


class TestDedup:
    async def test_a_repeat_of_a_better_ranked_candidate_is_dropped_in_one_more_request(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, SENCHA: 0.7}, repeats={(TEA, TEA_AGAIN)})
        scores = await _predict(client, [TEA_AGAIN, SENCHA, TEA], dedup=True)
        assert scores[0] == 0.0 and scores[1] > 0.0 and scores[2] == 1.0
        assert client.scopes == ["decision_recall_rerank", "decision_recall_dedup"]
        # Each pair once, the better-ranked as Memory A.
        assert sorted(client.pairs) == sorted([(TEA, TEA_AGAIN), (TEA, SENCHA), (TEA_AGAIN, SENCHA)])

    async def test_off_no_second_request(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8}, repeats={(TEA, TEA_AGAIN)})
        assert all(score > 0.0 for score in await _predict(client, [TEA, TEA_AGAIN]))
        assert client.scopes == ["decision_recall_rerank"]

    async def test_only_the_candidates_left_after_the_cut_are_compared(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, "offsite in lisbon": 0.05})
        await _predict(client, [TEA, TEA_AGAIN, "offsite in lisbon"], prune=True, dedup=True)
        assert client.pairs == [(TEA, TEA_AGAIN)]

    async def test_a_repeat_of_a_dropped_candidate_stays(self):
        """Greedy in rank order, so a chain does not drop its own head."""
        a, b, c = "Acme signed the beta", "Acme signed the beta deal", "Acme signed the beta contract"
        client = _Client({a: 0.9, b: 0.8, c: 0.7}, repeats={(a, b), (b, c)})
        assert [score > 0.0 for score in await _predict(client, [a, b, c], dedup=True)] == [True, False, True]

    async def test_different_dates_and_fact_types_are_not_compared(self):
        may3 = "[Date: May 03, 2026 (2026-05-03)] Sam went climbing"
        may10 = "[Date: May 10, 2026 (2026-05-10)] Sam went climbing"
        client = _Client({may3: 0.9, may10: 0.8}, repeats={(may3, may10)})
        assert all(score > 0.0 for score in await _predict(client, [may3, may10], dedup=True))
        assert client.pairs == []

        fact, observation = "Marta leads payments", "Marta has led payments since 2022"
        client = _Client({fact: 0.9, observation: 0.8}, repeats={(fact, observation)})
        scores = await _predict(client, [fact, observation], dedup=True, fact_types=["world", "observation"])
        assert all(score > 0.0 for score in scores)
        assert client.pairs == []

    async def test_recall_leaves_the_repeat_out(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8}, repeats={(TEA, TEA_AGAIN)})
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        result = await reranker.rerank("q", [_candidate(TEA_AGAIN), _candidate(TEA)], dedup=True)
        assert [r.candidate.retrieval.text for r in result.results] == [TEA]


# Stored embeddings: the two tea memories point the same way, sencha elsewhere.
_VECTORS = {TEA: [1.0, 0.0], TEA_AGAIN: [0.95, 0.1], SENCHA: [0.0, 1.0]}


async def _found(vectors: dict[str, list[float]] | None) -> dict[str, np.ndarray] | None:
    return None if vectors is None else {key: np.array(vector) for key, vector in vectors.items()}


class TestDedupAlike:
    """With the stored embeddings, dedup asks only alike pairs, best-ranked first."""

    async def _dedup(self, client: _Client, docs: list[str], vectors: dict[str, list[float]] | None) -> list[float]:
        task = asyncio.create_task(_found(vectors))
        return await _predict(client, docs, dedup=True, ids=docs, embeddings=task)

    async def test_only_alike_pairs_are_asked(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, SENCHA: 0.7}, repeats={(TEA, TEA_AGAIN)})
        scores = await self._dedup(client, [TEA, TEA_AGAIN, SENCHA], _VECTORS)
        assert client.pairs == [(TEA, TEA_AGAIN)]
        assert scores[1] == 0.0 and scores[2] > 0.0

    async def test_no_qualifying_pair_skips_the_second_request(self):
        client = _Client({TEA: 0.9, SENCHA: 0.7})
        await self._dedup(client, [TEA, SENCHA], _VECTORS)
        assert client.scopes == ["decision_recall_rerank"]

    async def test_without_embeddings_every_pair_is_asked(self):
        """The store keeps none (or the read failed): fall back to every kept pair."""
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, SENCHA: 0.7})
        await self._dedup(client, [TEA, TEA_AGAIN, SENCHA], None)
        assert len(client.pairs) == 3

    async def test_a_candidate_without_an_embedding_is_not_compared(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, SENCHA: 0.7})
        await self._dedup(client, [TEA, TEA_AGAIN, SENCHA], {TEA: _VECTORS[TEA], SENCHA: _VECTORS[SENCHA]})
        assert client.scopes == ["decision_recall_rerank"]

    async def test_the_best_ranked_alike_pairs_are_asked_up_to_the_cap(self, monkeypatch):
        monkeypatch.setattr(decision_rerank_interface, "MAX_REPEAT_PAIRS", 1)
        closer = "Marta's favourite tea: genmaicha"
        vectors = {**_VECTORS, closer: [1.0, 0.01]}
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.8, closer: 0.7})
        await self._dedup(client, [TEA, TEA_AGAIN, closer], vectors)
        assert client.pairs == [(TEA, TEA_AGAIN)]

    async def test_only_the_kept_candidates_are_compared(self):
        client = _Client({TEA: 0.9, TEA_AGAIN: 0.05, SENCHA: 0.7})
        await _predict(
            client,
            [TEA, TEA_AGAIN, SENCHA],
            prune=True,
            dedup=True,
            ids=[TEA, TEA_AGAIN, SENCHA],
            embeddings=asyncio.create_task(_found(_VECTORS)),
        )
        assert client.scopes == ["decision_recall_rerank"]

    async def test_the_embeddings_are_read_alongside_the_ranking_request(self):
        """No round trip of its own: the ranking request is still out when the read runs."""
        read_ran = asyncio.Event()

        async def read() -> dict[str, np.ndarray] | None:
            read_ran.set()
            return await _found(_VECTORS)

        class _Ranking(_Client):
            async def ask_all(self, state, questions, *, scope):
                if scope == "decision_recall_rerank":
                    await asyncio.wait_for(read_ran.wait(), timeout=5)
                return await super().ask_all(state, questions, scope=scope)

        client = _Ranking({TEA: 0.9, TEA_AGAIN: 0.8}, repeats={(TEA, TEA_AGAIN)})
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        task = asyncio.create_task(read())
        result = await reranker.rerank("q", [_candidate(TEA), _candidate(TEA_AGAIN)], dedup=True, embeddings=task)
        assert [r.candidate.retrieval.text for r in result.results] == [TEA]


@pytest.fixture(autouse=True)
def _no_shared_context():
    assert rerank_shared_context.get() is None
