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
from hindsight_api.engine.cross_encoder import TypeSafeCrossEncoder, create_cross_encoder_from_env


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


def _encoder(ranking: dict[str, float], cut_level: float = 0.0, **kwargs):
    encoder = TypeSafeCrossEncoder(api_key="k", **kwargs)
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
