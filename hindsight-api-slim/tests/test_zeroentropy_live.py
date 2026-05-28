"""Live ZeroEntropy API integration tests for both embeddings (zembed-1) and the reranker (zerank-2).

These tests hit the real ZeroEntropy API. They are skipped unless ZEROENTROPY_LIVE_API_KEY
is set, so CI and local default runs are unaffected. Set the env var to exercise the
full request/response path against the production endpoint.
"""

import os

import pytest

LIVE_API_KEY = os.environ.get("ZEROENTROPY_LIVE_API_KEY")
SKIP_REASON = "ZEROENTROPY_LIVE_API_KEY not set - skipping live ZeroEntropy integration test"


@pytest.mark.skipif(not LIVE_API_KEY, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_live_zeroentropy_embeddings_document_and_query():
    """zembed-1 returns 1280-dim float vectors for both document and query input types,
    and the two input types yield distinct vectors for the same text (asymmetric encoder)."""
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    assert LIVE_API_KEY is not None
    embeddings = ZeroEntropyEmbeddings(api_key=LIVE_API_KEY, dimensions=1280)
    await embeddings.initialize()

    docs = embeddings.encode_documents(["Paris is the capital of France.", "Python is a programming language."])
    assert len(docs) == 2
    assert all(len(v) == 1280 for v in docs)
    assert all(isinstance(x, float) for x in docs[0])

    queries = embeddings.encode_query(["What is the capital of France?"])
    assert len(queries) == 1
    assert len(queries[0]) == 1280

    # Asymmetric encoder: same text embedded as document vs query should differ.
    same_text = "Paris is the capital of France."
    doc_vec = embeddings.encode_documents([same_text])[0]
    query_vec = embeddings.encode_query([same_text])[0]
    assert doc_vec != query_vec


@pytest.mark.skipif(not LIVE_API_KEY, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_live_zeroentropy_embeddings_base64_matches_float():
    """The base64 response encoding decodes to the same vectors (within float tolerance)
    as the float response encoding."""
    from hindsight_api.engine.embeddings import ZeroEntropyEmbeddings

    assert LIVE_API_KEY is not None
    text = "ZeroEntropy supports Matryoshka embeddings."

    float_provider = ZeroEntropyEmbeddings(api_key=LIVE_API_KEY, dimensions=640, encoding_format="float")
    await float_provider.initialize()
    float_vec = float_provider.encode_documents([text])[0]

    base64_provider = ZeroEntropyEmbeddings(api_key=LIVE_API_KEY, dimensions=640, encoding_format="base64")
    await base64_provider.initialize()
    base64_vec = base64_provider.encode_documents([text])[0]

    assert len(float_vec) == 640
    assert len(base64_vec) == 640
    # Same text + same dimensions through different transport encodings should match within float32 precision.
    assert all(abs(a - b) < 1e-5 for a, b in zip(float_vec, base64_vec, strict=True))


@pytest.mark.skipif(not LIVE_API_KEY, reason=SKIP_REASON)
@pytest.mark.asyncio
async def test_live_zeroentropy_reranker_orders_by_relevance():
    """zerank-2 returns higher scores for more relevant passages."""
    from hindsight_api.engine.cross_encoder import ZeroEntropyCrossEncoder

    assert LIVE_API_KEY is not None
    encoder = ZeroEntropyCrossEncoder(api_key=LIVE_API_KEY)
    await encoder.initialize()

    query = "What is the capital of France?"
    pairs = [
        (query, "Paris is the capital and most populous city of France."),
        (query, "Python is a high-level programming language."),
        (query, "The Pacific Ocean is the largest body of water on Earth."),
    ]

    scores = await encoder.predict(pairs)

    assert len(scores) == 3
    # Relevant passage should outrank the unrelated ones.
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
