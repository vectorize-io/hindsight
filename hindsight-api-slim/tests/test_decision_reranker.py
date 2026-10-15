"""Recall's decision-model reranker: rank each candidate, then resolve conflicts."""

import pytest

from hindsight_api.config import DecisionModelConfig
from hindsight_api.engine.decision_model import DecisionAnswer
from hindsight_api.engine.search.decision_reranker import (
    _CONFLICT_INSTRUCTIONS,
    DecisionModelReranker,
    _strip_shared_prefix,
    _MAX_SHARED_CONTEXT_TOKENS,
    _ValuePayload,
    _value_payload,
    rerank_shared_context,
)
from hindsight_api.engine.search.reranking import CrossEncoderReranker
from hindsight_api.engine.token_encoding import count_tokens


class _Client:
    """Answers the rank question from ``keep`` (by memory text) and every conflict
    question with ``choice`` at ``confidence``; ``conflicts_fail`` leaves them unanswered,
    as the real client does when their request fails."""

    def __init__(self, keep=None, choice=None, confidence=0.9, conflicts_fail=False):
        self.keep = keep or {}
        self.choice = choice
        self.confidence = confidence
        self.conflicts_fail = conflicts_fail
        self.asked: list[tuple[str, dict]] = []
        self.max_request_tokens = 26_000

    async def ask_all(self, state, questions):
        self.asked.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if question.type == "noul":
                answers[key] = DecisionAnswer(noul=self.keep.get(question.instructions["memory"], 0.5))
            elif not self.conflicts_fail:
                answers[key] = DecisionAnswer(choice=self.choice, confidence=self.confidence)
        return answers

    @property
    def conflicts(self):
        return [qs for _, qs in self.asked if next(iter(qs.values())).instructions == _CONFLICT_INSTRUCTIONS]


def _reranker(client: _Client) -> DecisionModelReranker:
    return DecisionModelReranker(client)  # type: ignore[arg-type]


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
        """A failed request leaves its candidates unanswered. An unranked candidate is
        recoverable; a deleted one is not."""
        client = _Client(keep={"b": 0.9})
        reranker = _reranker(client)

        async def partial(state, questions):
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


class TestSharedPreamble:
    """Recall prefixes every candidate with its bank context, which on a single-conversation
    bank is the same profile block on all of them. The model judges and compares what is
    each candidate's own."""

    PROFILE = (
        "Conversation 1 - USER PROFILE:\n"
        "  Name: Craig Baker\n"
        "  Age: 49 years old\n"
        "  Location: Port Matthew\n"
        "  Occupation: backend developer building a personal budget tracker\n"
        "  Preferred stack: Flask, Postgres, plain CSS"
    )

    def test_a_preamble_every_candidate_shares_is_stripped(self):
        docs = [f"{self.PROFILE}\nThe deadline is April 5", f"{self.PROFILE}\nThe team chose Postgres"]
        assert _strip_shared_prefix(docs, [0, 1]) == {0: "The deadline is April 5", 1: "The team chose Postgres"}

    def test_a_preamble_only_some_share_is_kept(self):
        """Stripping needs the whole pool to agree, or it would cut real content."""
        docs = [f"{self.PROFILE}\nalpha", f"{self.PROFILE}\nbeta", "no profile here"]
        assert _strip_shared_prefix(docs, [0, 1, 2]) == {0: docs[0], 1: docs[1], 2: docs[2]}

    def test_identical_candidates_keep_their_text(self):
        """Stripping everything would leave nothing to rank."""
        assert _strip_shared_prefix(["same line", "same line"], [0, 1]) == {0: "same line", 1: "same line"}

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

    async def test_the_preamble_does_not_reach_the_rank_question(self):
        client = _Client()
        reranker = _reranker(client)
        await reranker.predict(
            [("q", f"{self.PROFILE}\nThe deadline is April 5"), ("q", f"{self.PROFILE}\nThe team chose Postgres")]
        )
        asked = [q.instructions["memory"] for q in client.asked[0][1].values()]
        assert asked == ["The deadline is April 5", "The team chose Postgres"]

    def test_the_preamble_does_not_make_unrelated_candidates_conflict(self):
        """Left in, 60 tokens of shared boilerplate put any two candidates over the overlap
        threshold, and two unrelated facts read as a disagreement."""
        reranker = _reranker(_Client())
        own = _strip_shared_prefix(
            [f"{self.PROFILE}\nThe deadline is April 5, 2024", f"{self.PROFILE}\nThe team chose Postgres"], [0, 1]
        )
        assert reranker._conflict_clusters("What is the deadline?", [own[0], own[1]], [0, 1]) == []


class TestConflictResolution:
    """Which of two candidates that disagree states the current answer.

    The pair is the shape the behaviour was built on: a corrected deadline and the stale
    claim it corrects, overlapping on almost every token and differing on one number.
    Ranking cannot separate them — only asking about them together can.
    """

    STALE = "The first sprint has a deadline of April 1, 2024 for completing the basic layout and navigation"
    CURRENT = "The first sprint has a deadline of April 5, 2024 for completing the basic layout and navigation"
    OTHER = "The team chose Postgres for storage"
    QUERY = "What is the deadline for completing the first sprint?"

    @property
    def _pairs(self):
        return [(self.QUERY, self.STALE), (self.QUERY, self.CURRENT), (self.QUERY, self.OTHER)]

    async def test_the_superseded_candidate_is_dropped_not_demoted(self):
        """Reordering is not enough: a generator reading several stale restatements and one
        correction follows the majority, so the stale ones have to leave."""
        reranker = _reranker(_Client(choice="c1"))
        scores = await reranker.predict(self._pairs)
        assert scores[0] == 0.0, "the superseded candidate must be pruned, not ranked lower"
        assert scores[1] > 0.0
        assert reranker.prunes_candidates is True

    async def test_a_restatement_is_not_deleted(self):
        """Overlap is a guess about what a candidate says, not a judgement that it is untrue.
        Only a verdict removes anything."""
        reranker = _reranker(_Client(choice=None, confidence=0.0))
        scores = await reranker.predict([(self.QUERY, self.STALE), (self.QUERY, f"{self.STALE}, as agreed")])
        assert all(score > 0.0 for score in scores)

    async def test_a_restatement_of_the_winner_with_more_detail_is_kept(self):
        """Adding a number to the current value still states it: only a candidate missing
        the winner's value is out of date."""
        reranker = _reranker(_Client(choice="c1"))
        detailed = f"{self.CURRENT}, though it may slip, on all 3 platforms"
        scores = await reranker.predict([*self._pairs, (self.QUERY, detailed)])
        assert scores[0] == 0.0
        assert scores[3] > 0.0

    def test_a_month_inside_another_word_is_not_a_month(self):
        assert _value_payload("The team marched on").months == frozenset()

    async def test_a_hedged_verdict_drops_nothing(self):
        """Below the floor the model is not resolving a conflict, and a wrong drop deletes
        the answer."""
        reranker = _reranker(_Client(choice="c1", confidence=0.5))
        assert all(score > 0.0 for score in await reranker.predict(self._pairs))

    async def test_a_question_that_does_not_ask_for_a_value_is_left_alone(self):
        client = _Client(choice="c1")
        reranker = _reranker(client)
        query = "Who owns the first sprint?"
        await reranker.predict([(query, self.STALE), (query, self.CURRENT)])
        assert client.conflicts == []

    async def test_candidates_that_are_not_about_the_question_are_left_alone(self):
        """Most candidates carry some incidental number; sharing one is not a conflict."""
        client = _Client(choice="c0")
        reranker = _reranker(client)
        await reranker.predict(
            [
                (self.QUERY, "Version 1 of the logo was approved by the design team"),
                (self.QUERY, "Version 2 of the logo was approved by the design team"),
            ]
        )
        assert client.conflicts == []

    async def test_an_unanswered_conflict_keeps_every_candidate(self):
        """Recall returning a stale candidate beside the current one is where it was
        before; failing the whole rerank over it would be worse."""
        reranker = _reranker(_Client(choice="c1", conflicts_fail=True))
        assert all(score > 0.0 for score in await reranker.predict(self._pairs))

    def test_a_terse_correction_conflicts_with_what_it_corrects(self):
        reranker = _reranker(_Client())
        assert reranker._conflict_clusters(self.QUERY, [self.STALE, self.CURRENT], [0, 1]) == [[0, 1]]

    async def test_a_confident_verdict_reaches_every_stale_restatement(self):
        """The same stale value restated many times spreads across several clusters.
        Dropping only the question's own members leaves the majority stale."""
        stale = [f"{self.STALE} (restated {n})" for n in range(5)]
        docs = [self.CURRENT, *stale]
        reranker = _reranker(_Client(choice="c0", confidence=0.95))
        superseded = await reranker._superseded(self.QUERY, docs, list(range(len(docs))))
        assert 0 not in superseded, "the winner must survive"
        assert len(superseded) == len(stale)

    def test_a_crowded_conflict_is_narrowed_not_skipped(self):
        reranker = _reranker(_Client())
        stale = [f"{self.STALE} in phase {n}" for n in range(reranker.CONFLICT_MAX_CLUSTER + 5)]
        docs = [self.CURRENT, *stale]
        clusters = reranker._conflict_clusters(self.QUERY, docs, list(range(len(docs))))
        assert clusters, "a crowded conflict must still be resolved"
        assert all(len(cluster) <= reranker.CONFLICT_MAX_CLUSTER for cluster in clusters)
        assert any(0 in cluster for cluster in clusters), "the correction has to be in the question"

    def test_a_cluster_is_the_anchor_and_its_direct_peers_only(self):
        """Grouping transitively collapsed a whole pool into one component."""
        reranker = _reranker(_Client())
        docs = [self.STALE, self.CURRENT, self.OTHER]
        assert reranker._conflict_clusters(self.QUERY, docs, [0, 1, 2]) == [[0, 1]]


class TestDateIsMetadataNotContent:
    """Recall prefixes each candidate with its date. It stays on the text the model judges
    and out of every comparison."""

    DATE = "[Date: June 05, 2024 (2024-06-05)] "

    def test_two_candidates_sharing_a_date_do_not_conflict_because_of_it(self):
        reranker = _reranker(_Client())
        docs = [f"{self.DATE}The team chose Postgres for storage", f"{self.DATE}The deadline moved to April 5"]
        assert reranker._conflict_clusters("What is the deadline?", docs, [0, 1]) == []

    def test_the_dates_digits_stay_out_of_the_value_payload(self):
        assert _value_payload(f"{self.DATE}the deadline is April 5") == _ValuePayload(
            frozenset({5}), frozenset({"april"})
        )

    def test_the_date_still_reaches_the_judged_text(self):
        own = _strip_shared_prefix([f"{self.DATE}alpha", f"{self.DATE}beta"], [0, 1])
        assert all(value.startswith("[Date:") for value in own.values())


@pytest.fixture(autouse=True)
def _no_shared_context():
    assert rerank_shared_context.get() is None
