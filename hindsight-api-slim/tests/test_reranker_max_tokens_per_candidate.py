"""HINDSIGHT_API_RERANKER_MAX_TOKENS_PER_CANDIDATE: one per-candidate cap for every reranker.

The cap used to exist only for the LiteLLM providers; every other backend sent a long
document whole, which could blow up a CPU-only TEI server (#4977). It now lives on the
base class, so any provider honors it, and the LiteLLM-specific name is a deprecated alias.
"""

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.cross_encoder import CrossEncoderModel, create_cross_encoder

LONG_DOC = "long technical document content about machine learning reranking " * 50


class _RecordingCrossEncoder(CrossEncoderModel):
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    @property
    def provider_name(self) -> str:
        return "recording"

    async def initialize(self) -> None:
        pass

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.sent = pairs
        return [0.5] * len(pairs)


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        "HINDSIGHT_API_RERANKER_MAX_TOKENS_PER_CANDIDATE",
        "HINDSIGHT_API_RERANKER_LITELLM_MAX_TOKENS_PER_DOC",
        "HINDSIGHT_API_RERANKER_1_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_unset_means_no_truncation(clean_env):
    assert HindsightConfig.from_env().reranker_chain()[0].max_tokens_per_candidate is None


def test_generic_name_wins_over_deprecated_alias(clean_env):
    clean_env.setenv("HINDSIGHT_API_RERANKER_LITELLM_MAX_TOKENS_PER_DOC", "900")
    assert HindsightConfig.from_env().reranker_max_tokens_per_candidate == 900
    clean_env.setenv("HINDSIGHT_API_RERANKER_MAX_TOKENS_PER_CANDIDATE", "512")
    assert HindsightConfig.from_env().reranker_max_tokens_per_candidate == 512


def test_each_fallback_member_has_its_own_cap(clean_env):
    clean_env.setenv("HINDSIGHT_API_RERANKER_MAX_TOKENS_PER_CANDIDATE", "512")
    clean_env.setenv("HINDSIGHT_API_RERANKER_1_PROVIDER", "rrf")
    clean_env.setenv("HINDSIGHT_API_RERANKER_1_LITELLM_MAX_TOKENS_PER_DOC", "900")
    chain = HindsightConfig.from_env().reranker_chain()
    assert [m.max_tokens_per_candidate for m in chain] == [512, 900]
    clean_env.setenv("HINDSIGHT_API_RERANKER_1_MAX_TOKENS_PER_CANDIDATE", "256")
    assert HindsightConfig.from_env().reranker_chain()[1].max_tokens_per_candidate == 256


@pytest.mark.parametrize("provider", ["tei", "cohere", "litellm", "rrf"])
def test_factory_puts_the_cap_on_every_provider(clean_env, provider):
    clean_env.setenv("HINDSIGHT_API_RERANKER_PROVIDER", provider)
    clean_env.setenv("HINDSIGHT_API_RERANKER_TEI_URL", "http://127.0.0.1:8890")
    clean_env.setenv("HINDSIGHT_API_RERANKER_COHERE_API_KEY", "k")
    clean_env.setenv("HINDSIGHT_API_RERANKER_MAX_TOKENS_PER_CANDIDATE", "64")
    member = HindsightConfig.from_env().reranker_chain()[0]
    assert create_cross_encoder(member).max_tokens_per_candidate == 64


@pytest.mark.asyncio
async def test_predict_truncates_long_documents_only():
    encoder = _RecordingCrossEncoder()
    encoder.max_tokens_per_candidate = 64

    scores = await encoder.predict([("q", LONG_DOC), ("q", "short doc")])

    assert scores == [0.5, 0.5]
    assert encoder.sent[0][0] == "q"
    assert len(encoder.sent[0][1]) < len(LONG_DOC)
    assert encoder.sent[1] == ("q", "short doc")
