"""
Reranker failover chain: HINDSIGHT_API_RERANKER_<n>_* members.

Covers issue #3168: an unreachable reranker took the whole recall down with a 500
because the stage was a hard dependency. A chain of indexed members lets the
refinement degrade — typically to a trailing `rrf` member, which is the same
neutral-score ordering you get with no reranker configured at all.
"""

import os
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import (
    CohereCrossEncoder,
    CrossEncoderModel,
    MultiCrossEncoder,
    RemoteTEICrossEncoder,
    RRFPassthroughCrossEncoder,
    create_cross_encoder,
    create_cross_encoder_from_env,
)

PAIRS = [("q", "doc a"), ("q", "doc b")]


class _FakeCrossEncoder(CrossEncoderModel):
    """Scriptable member: records calls, optionally fails init and/or predict."""

    def __init__(
        self,
        name: str,
        scores: list[float] | None = None,
        init_error: Exception | None = None,
        predict_error: Exception | None = None,
    ) -> None:
        self._name = name
        self._scores = scores
        self._init_error = init_error
        self._predict_error = predict_error
        self.init_calls = 0
        self.predict_calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def initialize(self) -> None:
        self.init_calls += 1
        if self._init_error is not None:
            raise self._init_error

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls += 1
        if self._predict_error is not None:
            raise self._predict_error
        assert self._scores is not None
        return self._scores


# ── env parsing ────────────────────────────────────────────────────────────────


def test_no_indexed_members_by_default():
    """The default config has no fallback members — behaviour is unchanged."""
    with patch.dict(os.environ, {}, clear=False):
        config = HindsightConfig.from_env()
    assert config.reranker_members == []
    assert len(config.reranker_chain()) == 1


def test_indexed_members_are_parsed_in_order_and_stop_at_a_gap():
    """Members must be contiguous from 1; scanning stops at the first missing index."""
    env = {
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "tei",
        "HINDSIGHT_API_RERANKER_1_TEI_URL": "http://backup:8080",
        "HINDSIGHT_API_RERANKER_2_PROVIDER": "rrf",
        # index 3 missing on purpose
        "HINDSIGHT_API_RERANKER_4_PROVIDER": "cohere",
    }
    with patch.dict(os.environ, env, clear=False):
        config = HindsightConfig.from_env()

    assert [m.provider for m in config.reranker_members] == ["tei", "rrf"]
    assert [m.index for m in config.reranker_members] == [1, 2]


def test_indexed_member_reads_its_own_provider_settings():
    """Every setting of member n carries the same index."""
    env = {
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "tei",
        "HINDSIGHT_API_RERANKER_1_TEI_URL": "http://backup:8080",
        "HINDSIGHT_API_RERANKER_1_TEI_BATCH_SIZE": "7",
        "HINDSIGHT_API_RERANKER_1_TEI_HTTP_TIMEOUT": "2.5",
    }
    with patch.dict(os.environ, env, clear=False):
        member = HindsightConfig.from_env().reranker_members[0]

    assert member.tei_url == "http://backup:8080"
    assert member.tei_batch_size == 7
    assert member.tei_http_timeout == 2.5


def test_indexed_member_inherits_nothing_from_the_primary():
    """An indexed member is read in isolation — no inheritance, only built-in defaults."""
    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
        "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "primary-key",
        "HINDSIGHT_API_RERANKER_COHERE_MODEL": "rerank-multilingual-v3.0",
        "HINDSIGHT_API_RERANKER_COHERE_BASE_URL": "http://primary/rerank",
        "HINDSIGHT_API_COHERE_API_KEY": "shared-key",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "cohere",
        "HINDSIGHT_API_RERANKER_1_COHERE_API_KEY": "member-key",
    }
    with patch.dict(os.environ, env, clear=False):
        member = HindsightConfig.from_env().reranker_members[0]

    assert member.cohere_api_key == "member-key"
    assert member.cohere_base_url is None
    assert member.cohere_model == "rerank-english-v3.0"  # built-in default, not the primary's


def test_member_with_a_bad_numeric_setting_fails_fast():
    env = {
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "tei",
        "HINDSIGHT_API_RERANKER_1_TEI_BATCH_SIZE": "lots",
    }
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_1_TEI_BATCH_SIZE"):
            HindsightConfig.from_env()


# ── chain construction ─────────────────────────────────────────────────────────


def test_factory_returns_a_single_encoder_without_members():
    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
        "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "k",
    }
    with patch.dict(os.environ, env, clear=False):
        config = HindsightConfig.from_env()
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()

    assert isinstance(encoder, CohereCrossEncoder)


def test_factory_builds_the_chain_when_members_are_configured():
    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "cohere",
        "HINDSIGHT_API_RERANKER_COHERE_API_KEY": "k",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "tei",
        "HINDSIGHT_API_RERANKER_1_TEI_URL": "http://backup:8080",
        "HINDSIGHT_API_RERANKER_2_PROVIDER": "rrf",
    }
    with patch.dict(os.environ, env, clear=False):
        config = HindsightConfig.from_env()
        with patch("hindsight_api.config.get_config", return_value=config):
            encoder = create_cross_encoder_from_env()

    assert isinstance(encoder, MultiCrossEncoder)
    members = encoder._members
    assert isinstance(members[0], CohereCrossEncoder)
    assert isinstance(members[1], RemoteTEICrossEncoder)
    assert isinstance(members[2], RRFPassthroughCrossEncoder)


def test_factory_builds_the_chain_through_the_real_config_proxy():
    """The production path reads the chain via get_config()'s static proxy."""
    from hindsight_api.config import clear_config_cache

    env = {
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "rrf",
    }
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            encoder = create_cross_encoder_from_env()
    finally:
        clear_config_cache()

    assert isinstance(encoder, MultiCrossEncoder)


def test_missing_member_setting_names_the_indexed_env_var():
    """A chain misconfiguration points at the exact indexed variable."""
    env = {
        "HINDSIGHT_API_RERANKER_1_PROVIDER": "tei",  # no _1_TEI_URL
    }
    with patch.dict(os.environ, env, clear=False):
        member = HindsightConfig.from_env().reranker_members[0]

    with pytest.raises(ValueError, match="HINDSIGHT_API_RERANKER_1_TEI_URL"):
        create_cross_encoder(member)


def test_multi_requires_at_least_two_members():
    with pytest.raises(ValueError, match="at least two members"):
        MultiCrossEncoder([RRFPassthroughCrossEncoder()])


# ── failover ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_healthy_primary_serves_and_no_fallback_is_touched():
    primary = _FakeCrossEncoder("primary", scores=[0.9, 0.1])
    fallback = _FakeCrossEncoder("fallback", scores=[0.5, 0.5])
    chain = MultiCrossEncoder([primary, fallback])
    await chain.initialize()

    assert await chain.predict(PAIRS) == [0.9, 0.1]
    assert fallback.predict_calls == 0
    assert chain.provider_name == "primary"


@pytest.mark.asyncio
async def test_unreachable_primary_falls_over_to_the_next_member():
    """The issue's case: recall keeps its answer instead of raising."""
    primary = _FakeCrossEncoder("primary", predict_error=TimeoutError("connect timeout"))
    fallback = RRFPassthroughCrossEncoder()
    chain = MultiCrossEncoder([primary, fallback])
    await chain.initialize()

    scores = await chain.predict(PAIRS)

    assert scores == [0.5, 0.5]  # neutral — the fusion order survives
    assert chain.provider_name == "rrf"  # reported as passthrough for downstream scoring


@pytest.mark.asyncio
async def test_wrong_length_scores_fail_over():
    """A member that answers with the wrong number of scores is unusable, not fatal."""
    primary = _FakeCrossEncoder("primary", scores=[0.9])  # one score for two pairs
    fallback = _FakeCrossEncoder("fallback", scores=[0.4, 0.6])
    chain = MultiCrossEncoder([primary, fallback])
    await chain.initialize()

    assert await chain.predict(PAIRS) == [0.4, 0.6]


@pytest.mark.asyncio
async def test_all_members_failing_raises_the_last_error():
    primary = _FakeCrossEncoder("primary", predict_error=TimeoutError("connect timeout"))
    fallback = _FakeCrossEncoder("fallback", predict_error=RuntimeError("404 Not Found"))
    chain = MultiCrossEncoder([primary, fallback])
    await chain.initialize()

    with pytest.raises(RuntimeError, match="404 Not Found"):
        await chain.predict(PAIRS)


@pytest.mark.asyncio
async def test_a_member_that_fails_to_initialize_is_not_fatal():
    """Startup tolerates a member that is down, and retries it when it is used."""
    primary = _FakeCrossEncoder("primary", scores=[0.9, 0.1], init_error=ConnectionError("refused"))
    fallback = _FakeCrossEncoder("fallback", scores=[0.4, 0.6])
    chain = MultiCrossEncoder([primary, fallback])

    await chain.initialize()  # must not raise

    assert primary.init_calls == 1
    assert await chain.predict(PAIRS) == [0.4, 0.6]
    assert primary.init_calls == 2  # retried on use, then failed over again


@pytest.mark.asyncio
async def test_members_initialize_once_when_healthy():
    primary = _FakeCrossEncoder("primary", scores=[0.9, 0.1])
    fallback = _FakeCrossEncoder("fallback", scores=[0.4, 0.6])
    chain = MultiCrossEncoder([primary, fallback])

    await chain.initialize()
    await chain.predict(PAIRS)
    await chain.predict(PAIRS)

    assert primary.init_calls == 1
    assert fallback.init_calls == 1


@pytest.mark.asyncio
async def test_chain_handles_its_own_blocking_member_init():
    """MultiCrossEncoder offloads local members itself, so callers never thread it."""
    chain = MultiCrossEncoder([RRFPassthroughCrossEncoder(), RRFPassthroughCrossEncoder()])
    assert chain.blocking_init is False
