"""TypeSafe reranker: question shape, score mapping, and the prune_candidates flag.

TypeSafe answers typed questions rather than exposing a /rerank endpoint, so the
mapping from (query, doc) pairs onto questions — and from answers back onto scores
— is this provider's whole substance. The HTTP round trip is faked; what is
asserted is the request we build and the scores we derive.
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
    """Captures posted bodies and answers each question with a scripted verdict."""

    def __init__(self, verdicts: list[tuple[str, float]]):
        self.verdicts = verdicts
        self.posted: list[dict] = []
        self.urls: list[str] = []

    def get(self):
        return self

    @asynccontextmanager
    async def _post(self, url, headers=None, json=None):
        self.urls.append(url)
        self.posted.append(json)
        answers = {}
        for index, key in enumerate(json["questions"]):
            choice, probability = self.verdicts[len(self.posted) - 1 + index]
            answers[key] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {"relevant": probability, "related": 0.0, "irrelevant": 0.0},
                "confidence": 0.9,
            }
        yield _FakeResponse({"answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})

    def post(self, url, headers=None, json=None):
        return self._post(url, headers=headers, json=json)


def _encoder(verdicts: list[tuple[str, float]], **kwargs) -> tuple[TypeSafeCrossEncoder, _FakeSession]:
    encoder = TypeSafeCrossEncoder(api_key="k", **kwargs)
    session = _FakeSession(verdicts)
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


class TestScoring:
    @pytest.mark.asyncio
    async def test_relevant_probability_becomes_the_score(self):
        encoder, _ = _encoder([("relevant", 0.87)])
        assert await encoder._predict([("q", "doc")]) == [0.87]

    @pytest.mark.asyncio
    async def test_related_keeps_its_probability_and_is_not_dropped(self):
        """ "related" is the middle option — a partial match still ranks, never 0.0."""
        encoder, _ = _encoder([("related", 0.4)], prune_candidates=True)
        assert await encoder._predict([("q", "doc")]) == [0.4]

    @pytest.mark.asyncio
    async def test_irrelevant_scores_zero_when_dropping_is_on(self):
        encoder, _ = _encoder([("irrelevant", 0.02)], prune_candidates=True)
        assert await encoder._predict([("q", "doc")]) == [0.0]
        assert encoder.prunes_candidates is True

    @pytest.mark.asyncio
    async def test_irrelevant_keeps_its_score_when_dropping_is_off(self):
        """Off by default, an irrelevant verdict only ranks last — it is not discarded."""
        encoder, _ = _encoder([("irrelevant", 0.02)])
        assert await encoder._predict([("q", "doc")]) == [0.02]
        assert encoder.prunes_candidates is False

    @pytest.mark.asyncio
    async def test_empty_pairs_make_no_request(self):
        encoder, session = _encoder([])
        assert await encoder._predict([]) == []
        assert session.posted == []


class TestRequestShape:
    @pytest.mark.asyncio
    async def test_single_candidate_is_the_whole_state(self):
        encoder, session = _encoder([("relevant", 1.0)])
        await encoder._predict([("who paid?", "Alice paid the bill")])

        body = session.posted[0]
        assert body["state"] == "Alice paid the bill"
        assert body["model"] == "jev-latest"
        question = body["questions"]["d0"]
        assert question["type"] == "choice"
        assert "who paid?" in question["instructions"]
        assert set(question["criteria"]) == {"relevant", "related", "irrelevant"}

    @pytest.mark.asyncio
    async def test_batched_candidates_share_a_state_and_are_numbered(self):
        encoder, session = _encoder([("relevant", 0.9), ("related", 0.3)], batch_size=2)
        await encoder._predict([("q", "first"), ("q", "second")])

        assert len(session.posted) == 1, "a batch of two should cost one round trip"
        body = session.posted[0]
        assert body["state"] == "[1] first\n\n[2] second"
        assert "[1]" in body["questions"]["d0"]["instructions"]
        assert "[2]" in body["questions"]["d1"]["instructions"]

    @pytest.mark.asyncio
    async def test_scores_return_in_pair_order_across_queries(self):
        """Pairs are grouped by query and chunked, so the mapping back must hold."""
        encoder, _ = _encoder([("relevant", 0.9), ("relevant", 0.5), ("related", 0.1)])
        scores = await encoder._predict([("a", "doc-a"), ("b", "doc-b"), ("a", "doc-a2")])
        assert scores == [0.9, 0.1, 0.5]

    @pytest.mark.asyncio
    async def test_base_url_is_honoured(self):
        encoder, session = _encoder([("relevant", 1.0)], base_url="https://proxy.example.com/")
        await encoder._predict([("q", "doc")])
        assert session.urls == ["https://proxy.example.com/v1/systemone"]


class TestFactory:
    def test_provider_is_built_from_config(self):
        config = _make_config(
            reranker_provider="typesafe",
            reranker_typesafe_api_key="k",
            reranker_typesafe_model="jev-latest",
            reranker_typesafe_base_url="https://api.typesafe.ai",
            reranker_typesafe_batch_size=4,
            reranker_typesafe_max_concurrent=8,
            reranker_typesafe_prune_candidates=True,
        )
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()

        assert encoder.provider_name == "typesafe"
        assert encoder.model == "jev-latest"
        assert encoder.batch_size == 4
        assert encoder.prunes_candidates is True

    def test_missing_api_key_names_its_env_var(self):
        config = _make_config(reranker_provider="typesafe")
        with patch("hindsight_api.config.get_config", return_value=config):
            with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_TYPESAFE_API_KEY"):
                create_cross_encoder_from_env()

    def test_defaults_are_jev_and_no_dropping(self):
        config = HindsightConfig.from_env()
        assert config.reranker_typesafe_model == "jev-latest"
        assert config.reranker_typesafe_prune_candidates is False
        assert config.reranker_typesafe_batch_size == 1
