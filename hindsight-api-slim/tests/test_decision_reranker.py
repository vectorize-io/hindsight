"""Recall's decision-model reranker: one question per candidate, ordered by the answers."""

from datetime import UTC, datetime

import pytest

from hindsight_api.config import DecisionModelConfig
from hindsight_api.engine.decision_model import DecisionAnswer
from hindsight_api.engine.search.decision_reranker import (
    _MAX_MEMORY_TOKENS,
    _MAX_SHARED_CONTEXT_TOKENS,
    DecisionModelReranker,
    rerank_shared_context,
)
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.search.types import MergedCandidate, RetrievalResult
from hindsight_api.engine.token_encoding import count_tokens


class _Client:
    """Answers each candidate's question from ``keep``, by memory text."""

    def __init__(self, keep=None):
        self.keep = keep or {}
        self.asked: list[tuple[str, dict]] = []
        self.max_request_tokens = 26_000

    async def ask_all(self, state, questions, *, scope):
        self.asked.append((state, questions))
        return {
            key: DecisionAnswer(noul=self.keep.get(question.instructions["memory"], 0.5))
            for key, question in questions.items()
        }


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
        assert count_tokens(client.asked[0][0]) <= _MAX_SHARED_CONTEXT_TOKENS + count_tokens("\n\nQuestion: q")

    async def test_a_huge_memory_is_capped(self):
        """One question carries one memory; uncapped, a single long one could outgrow the
        window and fail the whole rerank."""
        client = _Client()
        await _reranker(client).predict([("q", "word " * 20_000), ("q", "short")])
        memories = [q.instructions["memory"] for q in client.asked[0][1].values()]
        assert count_tokens(memories[0]) <= _MAX_MEMORY_TOKENS
        assert memories[1] == "short"

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

    async def test_without_a_question_date_it_is_today(self):
        client = _Client()
        reranker = CrossEncoderReranker(cross_encoder=_reranker(client))
        before = datetime.now(UTC).date()
        await reranker.rerank("q", [_candidate("a"), _candidate("b")])
        after = datetime.now(UTC).date()
        assert client.asked[0][0] in {f"Current date: {day.isoformat()}\n\nQuestion: q" for day in (before, after)}


@pytest.fixture(autouse=True)
def _no_shared_context():
    assert rerank_shared_context.get() is None
