"""
TEI reranker per-document truncation: HINDSIGHT_API_RERANKER_TEI_MAX_TOKENS_PER_DOC.

Covers the fix that aligns the TEI provider with the existing LiteLLM
``max_tokens_per_doc`` behavior: when set, each candidate document is
truncated to at most N tokens (shared tiktoken encoder) before it is sent
to the TEI rerank server. Without truncation, a very long document is sent
verbatim and can blow up the rerank server's input tensor (observed on a
16 GB Mac mini running a CPU-only ONNX reranker: 37 GB physical footprint,
swap exhaustion, and a full recall stall).
"""

import pytest

from hindsight_api.config import DEFAULT_RERANKER_TEI_MAX_TOKENS_PER_DOC, HindsightConfig
from hindsight_api.engine.cross_encoder import RemoteTEICrossEncoder, create_cross_encoder

BASE_ENV = {
    "HINDSIGHT_API_RERANKER_PROVIDER": "tei",
    "HINDSIGHT_API_RERANKER_TEI_URL": "http://127.0.0.1:8890",
}


def test_tei_max_tokens_per_doc_default_is_none():
    """Default: no truncation (mirrors the LiteLLM default)."""
    with pytest.MonkeyPatch.context() as mp:
        for k, v in BASE_ENV.items():
            mp.setenv(k, v)
        mp.delenv("HINDSIGHT_API_RERANKER_TEI_MAX_TOKENS_PER_DOC", raising=False)

        config = HindsightConfig.from_env()
        member = config.reranker_chain()[0]
        assert member.tei_max_tokens_per_doc is None
        assert DEFAULT_RERANKER_TEI_MAX_TOKENS_PER_DOC is None


def test_tei_max_tokens_per_doc_env_flows_through_factory():
    """Env var reaches RerankerMemberConfig and the constructed encoder."""
    with pytest.MonkeyPatch.context() as mp:
        for k, v in BASE_ENV.items():
            mp.setenv(k, v)
        mp.setenv("HINDSIGHT_API_RERANKER_TEI_MAX_TOKENS_PER_DOC", "512")

        config = HindsightConfig.from_env()
        member = config.reranker_chain()[0]
        assert member.tei_max_tokens_per_doc == 512

        encoder = create_cross_encoder(member)
        assert isinstance(encoder, RemoteTEICrossEncoder)
        assert encoder.max_tokens_per_doc == 512


@pytest.mark.asyncio
async def test_tei_predict_truncates_long_documents_before_sending():
    """Documents over the token budget are truncated; short ones pass through."""
    encoder = RemoteTEICrossEncoder(
        base_url="http://127.0.0.1:8890",
        max_tokens_per_doc=64,
    )

    long_doc = "long technical document content about machine learning reranking " * 20  # 1300 chars > 64 tokens
    sent: dict = {}

    async def fake_rerank(self, semaphore, query, texts):
        sent["texts"] = texts
        return [(i, 0.5) for i in range(len(texts))]

    encoder._rerank_query_group = fake_rerank.__get__(encoder, type(encoder))  # type: ignore[assignment]

    scores = await encoder._predict_async([("q", long_doc), ("q", "short doc")])

    assert len(scores) == 2
    assert len(sent["texts"]) == 2
    # Long doc was truncated well below its 1300-char original.
    assert len(sent["texts"][0]) < len(long_doc)
    # Short doc is untouched.
    assert sent["texts"][1] == "short doc"
