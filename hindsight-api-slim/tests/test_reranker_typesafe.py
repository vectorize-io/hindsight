"""TypeSafe reranker: the rank question, the cut question, and what comes back.

TypeSafe answers typed questions rather than exposing a /rerank endpoint, so the
mapping from (query, doc) pairs onto questions — and from answers back onto scores —
is this provider's whole substance. The HTTP round trip is faked; what is asserted is
the requests we build and the order and cut we derive from the answers.
"""

from contextlib import asynccontextmanager
from dataclasses import fields
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import (
    _OPTION_KEY_OVERHEAD,
    TypeSafeCrossEncoder,
    create_cross_encoder_from_env,
)
from hindsight_api.engine.token_encoding import count_tokens


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status = 200

    async def json(self, content_type=None):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Answers a rank question from `ranking` and a cut question from `cut_level`.

    `ranking` maps an option key ("c0", "c1", …) to the probability the model would
    give it, so a pool ranked in several rounds is answered per round exactly as the
    real API would answer it.
    """

    def __init__(self, ranking: dict[str, float], cut_level: float = 0.0):
        self.ranking = ranking
        self.cut_level = cut_level
        self.posted: list[dict] = []
        self.urls: list[str] = []

    def get(self):
        return self

    @asynccontextmanager
    async def _post(self, url, headers=None, json=None):
        self.urls.append(url)
        self.posted.append(json)
        question_id, question = next(iter(json["questions"].items()))
        if question["type"] == "choice":
            keys = list(question["criteria"])
            answer = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {key: self.ranking.get(key, 0.0) for key in keys},
                "confidence": 0.9,
            }
        else:
            answer = {
                "type": "score",
                "score": self.cut_level,
                "legend": dict(enumerate(question["criteria"])),
                "confidence": 0.9,
            }
        yield _FakeResponse({"answers": {question_id: answer}, "usage": {"input_tokens": 1, "output_tokens": 1}})

    def post(self, url, headers=None, json=None):
        return self._post(url, headers=headers, json=json)

    @property
    def rank_requests(self) -> list[dict]:
        return [body for body in self.posted if next(iter(body["questions"].values()))["type"] == "choice"]

    @property
    def cut_requests(self) -> list[dict]:
        return [body for body in self.posted if next(iter(body["questions"].values()))["type"] == "score"]


def _encoder(
    ranking: dict[str, float],
    cut_level: float = 0.0,
    max_question_tokens: int | None = None,
    **kwargs,
):
    encoder = TypeSafeCrossEncoder(api_key="k", **kwargs)
    if max_question_tokens is not None:
        encoder.MAX_QUESTION_TOKENS = max_question_tokens
    session = _FakeSession(ranking, cut_level)
    encoder._session = session
    return encoder, session


def _make_config(**overrides) -> HindsightConfig:
    defaults: dict = {}
    for f in fields(HindsightConfig):
        if f.type == "str":
            defaults[f.name] = ""
        elif f.type == "int":
            defaults[f.name] = 0
        elif f.type == "float":
            defaults[f.name] = 0.0
        elif f.type == "bool":
            defaults[f.name] = False
        elif str(f.type).startswith("list["):
            defaults[f.name] = []
        else:
            defaults[f.name] = None
    defaults.update(overrides)
    return HindsightConfig(**defaults)


class TestRanking:
    @pytest.mark.asyncio
    async def test_the_answer_order_becomes_the_score_order(self):
        """c2 wins, then c0, then c1 — the scores must sort the same way."""
        encoder, _ = _encoder({"c0": 0.3, "c1": 0.1, "c2": 0.6})
        scores = await encoder._predict([("q", "first"), ("q", "second"), ("q", "third")])

        assert scores[2] > scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_scores_are_positions_not_the_returned_probabilities(self):
        """A Choice probability is a share of one pool, so it is not handed on as a score."""
        encoder, _ = _encoder({"c0": 0.9, "c1": 0.07, "c2": 0.03})
        scores = await encoder._predict([("q", "a"), ("q", "b"), ("q", "c")])

        assert scores == [1.0, pytest.approx(2 / 3), pytest.approx(1 / 3)]

    @pytest.mark.asyncio
    async def test_the_whole_pool_is_one_call(self):
        encoder, session = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(20)})
        await encoder._predict([("q", f"doc {i}") for i in range(20)])

        assert len(session.rank_requests) == 1
        assert len(session.cut_requests) == 0, "the cut question is only asked when pruning"

    @pytest.mark.asyncio
    async def test_candidates_are_the_options_and_the_query_is_the_question(self):
        encoder, session = _encoder({"c0": 1.0})
        await encoder._predict([("who paid?", "Alice paid the bill")])

        body = session.rank_requests[0]
        assert body["model"] == "jev-latest"
        assert "who paid?" in body["state"]
        question = body["questions"]["rank"]
        assert question["type"] == "choice"
        assert question["criteria"] == {"c0": "Alice paid the bill"}

    @pytest.mark.asyncio
    async def test_each_query_is_ranked_in_its_own_pool(self):
        """Two queries cannot share a ranking: a Choice ranks against one question."""
        encoder, session = _encoder({"c0": 0.9, "c1": 0.1})
        scores = await encoder._predict([("a", "doc-a"), ("b", "doc-b"), ("a", "doc-a2")])

        assert len(session.rank_requests) == 2
        assert len(scores) == 3
        assert scores[0] > scores[2], "query a's own two candidates keep their order"

    @pytest.mark.asyncio
    async def test_empty_pairs_make_no_request(self):
        encoder, session = _encoder({})
        assert await encoder._predict([]) == []
        assert session.posted == []


class TestChunking:
    @pytest.mark.asyncio
    async def test_a_pool_over_the_option_cap_is_ranked_in_rounds(self):
        """A Choice takes at most 255 options, so a bigger pool needs several rounds."""
        size = TypeSafeCrossEncoder.MAX_OPTIONS + 10
        encoder, session = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(size)})
        scores = await encoder._predict([("q", f"doc {i}") for i in range(size)])

        # Two rounds over the halves, then one more over their winners.
        assert len(session.rank_requests) == 3
        assert all(len(body["questions"]["rank"]["criteria"]) <= 255 for body in session.rank_requests)
        assert len(scores) == size

    @pytest.mark.asyncio
    async def test_every_candidate_still_gets_a_distinct_position(self):
        size = TypeSafeCrossEncoder.MAX_OPTIONS + 10
        encoder, _ = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(size)})
        scores = await encoder._predict([("q", f"doc {i}") for i in range(size)])

        assert len(set(scores)) == size, "positions must be distinct, not collapsed onto ties"
        assert min(scores) > 0.0, "nothing is pruned when prune_candidates is off"


class TestCut:
    @pytest.mark.asyncio
    async def test_the_cut_question_is_asked_only_when_pruning(self):
        encoder, session = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(5)}, prune_candidates=True)
        await encoder._predict([("q", f"doc {i}") for i in range(5)])

        assert len(session.cut_requests) == 1
        question = session.cut_requests[0]["questions"]["depth"]
        assert question["type"] == "score"
        assert question["criteria"] == TypeSafeCrossEncoder.CUT_LEVELS

    @pytest.mark.asyncio
    async def test_level_zero_keeps_only_the_best_candidate(self):
        encoder, _ = _encoder({"c0": 0.2, "c1": 0.7, "c2": 0.1}, cut_level=0.0, prune_candidates=True)
        scores = await encoder._predict([("q", "a"), ("q", "b"), ("q", "c")])

        assert scores[1] > 0.0
        assert scores[0] == 0.0 and scores[2] == 0.0

    @pytest.mark.asyncio
    async def test_a_deeper_level_keeps_more(self):
        encoder, _ = _encoder({"c0": 0.2, "c1": 0.7, "c2": 0.1}, cut_level=2.0, prune_candidates=True)
        scores = await encoder._predict([("q", "a"), ("q", "b"), ("q", "c")])

        assert all(score > 0.0 for score in scores), "level 2 keeps the first three"

    @pytest.mark.asyncio
    async def test_the_cut_never_empties_the_result(self):
        """There is no 'nothing is relevant' level, so the best candidate always survives."""
        encoder, _ = _encoder({"c0": 0.4, "c1": 0.35, "c2": 0.25}, cut_level=-1.0, prune_candidates=True)
        scores = await encoder._predict([("q", "a"), ("q", "b"), ("q", "c")])

        assert sum(1 for score in scores if score > 0.0) == 1

    @pytest.mark.asyncio
    async def test_the_top_level_keeps_the_whole_shortlist(self):
        encoder, _ = _encoder(
            {f"c{i}": 1.0 / (i + 1) for i in range(4)},
            cut_level=float(len(TypeSafeCrossEncoder.CUT_LEVELS) - 1),
            prune_candidates=True,
        )
        scores = await encoder._predict([("q", f"doc {i}") for i in range(4)])

        assert all(score > 0.0 for score in scores)

    @pytest.mark.asyncio
    async def test_nothing_is_pruned_when_the_flag_is_off(self):
        encoder, _ = _encoder({"c0": 0.9, "c1": 0.05, "c2": 0.05})
        scores = await encoder._predict([("q", "a"), ("q", "b"), ("q", "c")])

        assert all(score > 0.0 for score in scores)


class TestFactory:
    def test_provider_is_built_from_config(self):
        config = _make_config(
            reranker_provider="typesafe",
            reranker_typesafe_api_key="k",
            reranker_typesafe_model="jev-latest",
            reranker_typesafe_base_url="https://api.typesafe.ai",
            reranker_typesafe_max_concurrent=8,
            reranker_typesafe_prune_candidates=True,
        )
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()

        assert encoder.provider_name == "typesafe"
        assert encoder.model == "jev-latest"
        assert encoder.prunes_candidates is True

    def test_missing_api_key_names_its_env_var(self):
        config = _make_config(reranker_provider="typesafe")
        with patch("hindsight_api.config.get_config", return_value=config):
            with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY"):
                create_cross_encoder_from_env()

    def test_defaults_are_jev_and_no_pruning(self):
        config = HindsightConfig.from_env()
        assert config.reranker_typesafe_model == "jev-latest"
        assert config.reranker_typesafe_prune_candidates is False

    @pytest.mark.asyncio
    async def test_base_url_is_honoured(self):
        encoder, session = _encoder({"c0": 1.0}, base_url="https://proxy.example.com/")
        await encoder._predict([("q", "doc")])
        assert session.urls == ["https://proxy.example.com/v1/systemone"]


class TestTokenBudgetingAndOrder:
    @pytest.mark.asyncio
    async def test_single_round_fast_path_when_pool_fits(self):
        """When candidates <= 250 and within token limit, only 1 request is made (no finals)."""
        encoder, session = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(50)})
        scores = await encoder._predict([("q", f"short doc {i}") for i in range(50)])

        assert len(session.rank_requests) == 1
        assert "rank" in session.rank_requests[0]["questions"]
        assert len(scores) == 50

    @pytest.mark.asyncio
    async def test_non_finalists_preserve_relative_rrf_order_without_inversion(self):
        """Issue #4599: non-finalists preserve caller input order (RRF), discarding intra-group model ranks.

        The mock model assigns higher probabilities to later option positions, inverting the
        candidate ranking within each group relative to input RRF order.
        Group 0 finalists (top 12) are 249..238; non-finalists are 0..237.
        Group 1 finalists (top 12) are 269..258; non-finalists are 250..257.
        Under the old chunk-based concatenation, non-finalists within each group kept their
        intra-group model ranking, producing [237..0, 257..250] where candidate 237 outranked candidate 0.
        Under the new code, non-finalists fall back to initial RRF order (0 < 1 < ...), discarding
        uncalibrated intra-group probabilities.
        """
        size = TypeSafeCrossEncoder.MAX_OPTIONS + 20  # 270 candidates
        # Intra-group model preference is opposite to initial RRF order
        encoder, session = _encoder({f"c{i}": float(i + 1) for i in range(size)})
        pairs = [("q", f"candidate_{i}") for i in range(size)]
        scores = await encoder._predict(pairs)

        all_non_finalists = list(range(0, 238)) + list(range(250, 258))
        for a, b in zip(all_non_finalists[:-1], all_non_finalists[1:]):
            assert scores[a] > scores[b], f"Expected score[{a}] > score[{b}] by initial RRF order"

    @pytest.mark.asyncio
    async def test_candidate_packing_by_tokens_partitions_long_docs_into_multiple_groups(self):
        """When total candidate tokens exceed question budget, candidates are partitioned into groups."""
        max_question_tokens = 500
        long_doc = "word " * 60  # ~61 tokens + 7 key overhead = 68 tokens
        size = 8  # 8 docs: Group 0 has 6 docs (~446 tok), Group 1 has 2 docs (~174 tok)
        encoder, session = _encoder(
            {f"c{i}": 1.0 / (i + 1) for i in range(size)},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", f"{long_doc} {i}") for i in range(size)])

        # 2 group requests + 1 finals request = 3 total
        assert len(session.rank_requests) == 3
        assert len(scores) == size

    @pytest.mark.asyncio
    async def test_cut_phase_truncation_prevents_overflow(self):
        """When shortlist docs are very long, uniform capping truncates them so State tokens stay within budget."""
        max_question_tokens = 800
        long_doc = "information about the project " * 40
        encoder, session = _encoder(
            {f"c{i}": 1.0 / (i + 1) for i in range(12)},
            cut_level=2.0,
            prune_candidates=True,
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", f"{long_doc} {i}") for i in range(12)])

        assert len(session.cut_requests) == 1
        cut_body = session.cut_requests[0]
        state_tokens = count_tokens(cut_body["state"])
        assert state_tokens <= max_question_tokens, f"State tokens {state_tokens} exceeded budget {max_question_tokens}"
        assert any(s > 0.0 for s in scores)

    @pytest.mark.asyncio
    async def test_finals_round_truncation_prevents_overflow(self):
        """When finalists' total tokens exceed question budget, uniform capping truncates them."""
        max_question_tokens = 500
        long_doc = "detailed context information " * 15
        size = 16
        encoder, session = _encoder(
            {f"c{i}": 1.0 / (i + 1) for i in range(size)},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", f"{long_doc} {i}") for i in range(size)])

        assert len(scores) == size
        finals_request = session.rank_requests[-1]
        assert "rank" in finals_request["questions"]
        criteria = finals_request["questions"]["rank"]["criteria"]
        total_criteria_tokens = sum(count_tokens(text) for text in criteria.values())
        assert total_criteria_tokens <= max_question_tokens

    @pytest.mark.asyncio
    async def test_choice_questions_never_have_fewer_than_two_options(self):
        """Regression test for [P1]: Choice questions must NEVER have fewer than 2 options.

        Jev strictly rejects single-option Choice questions with 4xx ('criteria must map 2 or more options').
        Even when oversized documents force single-candidate groups in token packing,
        preliminary rounds for 1-candidate groups are skipped, and candidates are resolved
        in the finals where at least 2 options are present.
        """
        max_question_tokens = 500
        huge_doc = "word " * 400
        encoder, session = _encoder(
            {"c0": 0.8, "c1": 0.2},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", f"{huge_doc} 1"), ("q", f"{huge_doc} 2")])
        assert len(scores) == 2

        # Every Choice question sent across all requests MUST have >= 2 criteria options
        for body in session.rank_requests:
            for q_id, q_data in body["questions"].items():
                if q_data.get("type") == "choice":
                    criteria_count = len(q_data.get("criteria", {}))
                    assert criteria_count >= 2, (
                        f"Question {q_id} in request has {criteria_count} options; TypeSafe requires at least 2 options"
                    )

    @pytest.mark.asyncio
    async def test_cut_phase_obeys_token_budget_with_overlong_query(self):
        """Regression test for [P2]: _cut must share bounded query and obey token budget.

        When query has 40k tokens and prune_candidates=True, _cut must not explode
        into an 80k token request. Both _rank and _cut must operate on the identical bounded query.
        """
        max_question_tokens = 500
        overlong_query = "who was in the room? " * 1000  # ~5,000 tokens
        encoder, session = _encoder(
            {"c0": 0.8, "c1": 0.2},
            cut_level=1.0,
            prune_candidates=True,
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([(overlong_query, "doc 1"), (overlong_query, "doc 2")])
        assert len(scores) == 2

        assert len(session.rank_requests) >= 1
        assert len(session.cut_requests) == 1

        rank_state = session.rank_requests[0]["state"]
        cut_state = session.cut_requests[0]["state"]

        # Rank state and Cut prefix must use the EXACT same effective query
        assert rank_state in cut_state

        # Cut request State must strictly obey max_question_tokens
        cut_state_tokens = count_tokens(cut_state)
        assert cut_state_tokens <= max_question_tokens, (
            f"Cut state has {cut_state_tokens} tokens, exceeding budget {max_question_tokens}"
        )

    @pytest.mark.asyncio
    async def test_all_questions_strictly_obey_token_ceilings(self):
        """Invariant check: In every choice question, total tokens <= max_question_tokens."""
        max_question_tokens = 600
        doc = "sample context words " * 25
        size = 30
        encoder, session = _encoder(
            {f"c{i}": 1.0 / (i + 1) for i in range(size)},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", f"{doc} {i}") for i in range(size)])
        assert len(scores) == size

        for body in session.posted:
            state_tokens = count_tokens(body["state"])
            for q_id, q_data in body.get("questions", {}).items():
                instr_tokens = count_tokens(q_data.get("instructions", ""))
                crit_tokens = sum(count_tokens(text) for text in q_data.get("criteria", {}).values())
                envelope = len(q_data.get("criteria", {})) * _OPTION_KEY_OVERHEAD
                q_tokens = instr_tokens + crit_tokens + envelope
                assert state_tokens + q_tokens <= max_question_tokens, f"Question {q_id} exceeded question budget"

    @pytest.mark.asyncio
    async def test_single_overlong_candidate_safely_pre_truncated(self):
        """A candidate document that individually exceeds question budget is pre-truncated safely."""
        max_question_tokens = 500
        long_doc = "word " * 1000  # ~1000 tokens, far exceeding 500
        encoder, session = _encoder(
            {"c0": 0.8, "c1": 0.2},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([("q", long_doc), ("q", "short doc")])
        assert len(scores) == 2
        for body in session.rank_requests:
            state_tokens = count_tokens(body["state"])
            for q_data in body["questions"].values():
                instr_tokens = count_tokens(q_data.get("instructions", ""))
                crit_tokens = sum(count_tokens(text) for text in q_data["criteria"].values())
                envelope = len(q_data["criteria"]) * _OPTION_KEY_OVERHEAD
                assert state_tokens + instr_tokens + crit_tokens + envelope <= max_question_tokens

    @pytest.mark.asyncio
    async def test_pre_truncated_candidate_carries_truncated_text_into_finals(self):
        """When an outlier document is pre-truncated in packing, the pre-truncated text
        (not the original oversized document) must be passed into the finals round.
        """
        max_question_tokens = 500
        huge_doc = "outlier document content " * 300
        normal_doc = "normal length document content " * 15
        encoder, session = _encoder(
            {"c0": 0.6, "c1": 0.4},
            max_question_tokens=max_question_tokens,
        )
        with patch.object(encoder, "_rank_once", wraps=encoder._rank_once) as mock_rank_once:
            scores = await encoder._predict([("q", huge_doc), ("q", normal_doc)])
            assert len(scores) == 2

            # The call to _rank_once must receive pre-truncated text in effective_docs
            passed_docs = mock_rank_once.call_args[0][1]
            assert len(passed_docs[0]) < len(huge_doc)

        finals_request = session.rank_requests[-1]
        criteria = finals_request["questions"]["rank"]["criteria"]
        c0_text = criteria["c0"]
        assert len(c0_text) < len(huge_doc)
        assert count_tokens(c0_text) < max_question_tokens

    @pytest.mark.asyncio
    async def test_overlong_query_defensively_truncated_without_budget_blowup(self):
        """An overlong query is defensively capped so it never manufactures overflow budget."""
        max_question_tokens = 500
        long_query = "query word " * 800  # ~800 tokens, far exceeding question budget
        encoder, session = _encoder(
            {"c0": 0.8, "c1": 0.2},
            max_question_tokens=max_question_tokens,
        )
        scores = await encoder._predict([(long_query, "doc 1"), (long_query, "doc 2")])
        assert len(scores) == 2
        for body in session.rank_requests:
            state_tokens = count_tokens(body["state"])
            assert state_tokens <= max_question_tokens

    @pytest.mark.asyncio
    async def test_finals_never_exceed_the_option_cap_when_long_docs_make_many_groups(self):
        """Long documents split the pool into many groups; the finals is still one Choice <= MAX_OPTIONS.

        Here ~3 documents fit a group, so 30 candidates make ~10 groups whose winners, at a
        fixed 12 per group, would all reach a finals capped at 10. In production the same
        happens once a pool splits into more than 250 // 12 = 20 groups.
        """
        size = 30
        encoder, session = _encoder({f"c{i}": 1.0 / (i + 1) for i in range(size)}, max_question_tokens=400)
        encoder.MAX_OPTIONS = 10
        scores = await encoder._predict([("q", "long document words " * 40 + str(i)) for i in range(size)])

        assert len(scores) == size
        assert all(len(body["questions"]["rank"]["criteria"]) <= 10 for body in session.rank_requests)
