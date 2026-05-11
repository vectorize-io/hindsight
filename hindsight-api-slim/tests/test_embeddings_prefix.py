"""
Tests for HINDSIGHT_API_EMBEDDINGS_{QUERY,DOC}_PREFIX env vars and the
template-method prefix-aware encoding interface.

The base `Embeddings.encode(texts, purpose)` applies the configured prefix
before delegating to subclass `_encode_impl(texts)`. This is the single point
of prefix application — provider-common, so every backend honors the same env
vars without each implementing prefix logic independently.
"""

import asyncio
import os
from typing import Any

import pytest

from hindsight_api.engine.embeddings import EmbeddingPurpose, Embeddings


class FakeEmbeddings(Embeddings):
    """Test fake that records the texts passed to `_encode_impl`."""

    def __init__(self, *, query_prefix: str = "", doc_prefix: str = ""):
        super().__init__(query_prefix=query_prefix, doc_prefix=doc_prefix)
        self.last_texts: list[str] | None = None

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 4

    async def initialize(self) -> None:
        return None

    def _encode_impl(self, texts: list[str]) -> list[list[float]]:
        self.last_texts = list(texts)
        return [[0.0] * self.dimension for _ in texts]


def test_query_purpose_prepends_query_prefix():
    fake = FakeEmbeddings(query_prefix="検索クエリ: ", doc_prefix="検索文書: ")
    fake.encode(["Hermes 起動ループ", "vchord BM25"], purpose="query")
    assert fake.last_texts == ["検索クエリ: Hermes 起動ループ", "検索クエリ: vchord BM25"]


def test_document_purpose_prepends_doc_prefix():
    fake = FakeEmbeddings(query_prefix="検索クエリ: ", doc_prefix="検索文書: ")
    fake.encode(["Hermes 起動ループ"], purpose="document")
    assert fake.last_texts == ["検索文書: Hermes 起動ループ"]


def test_default_purpose_is_document():
    fake = FakeEmbeddings(query_prefix="Q: ", doc_prefix="D: ")
    fake.encode(["x"])  # no purpose kwarg
    assert fake.last_texts == ["D: x"]


def test_empty_prefixes_pass_through_unchanged():
    """Byte-identical to pre-patch behavior: empty defaults add no prefix."""
    fake = FakeEmbeddings()
    fake.encode(["raw1", "raw2"], purpose="query")
    assert fake.last_texts == ["raw1", "raw2"]
    fake.encode(["doc1"], purpose="document")
    assert fake.last_texts == ["doc1"]


def test_only_query_prefix_set_leaves_documents_unchanged():
    fake = FakeEmbeddings(query_prefix="Q: ")
    fake.encode(["x"], purpose="document")
    assert fake.last_texts == ["x"]
    fake.encode(["x"], purpose="query")
    assert fake.last_texts == ["Q: x"]


def test_async_executor_path_threads_purpose_correctly():
    """`generate_embeddings_batch` runs encode inside `run_in_executor` via
    functools.partial — the executor must receive `purpose` as a kwarg."""
    from hindsight_api.engine.retain import embedding_utils

    fake = FakeEmbeddings(query_prefix="Q: ", doc_prefix="D: ")
    results = asyncio.run(
        embedding_utils.generate_embeddings_batch(fake, ["hello", "world"], purpose="query")
    )
    assert len(results) == 2
    assert fake.last_texts == ["Q: hello", "Q: world"]

    results = asyncio.run(
        embedding_utils.generate_embeddings_batch(fake, ["a doc"], purpose="document")
    )
    assert fake.last_texts == ["D: a doc"]


def test_async_default_purpose_is_document():
    """Backwards-compatible default for callers that don't specify purpose."""
    from hindsight_api.engine.retain import embedding_utils

    fake = FakeEmbeddings(doc_prefix="D: ")
    asyncio.run(embedding_utils.generate_embeddings_batch(fake, ["x"]))
    assert fake.last_texts == ["D: x"]


def test_purpose_type_is_literal():
    """Smoke check that EmbeddingPurpose is the documented Literal."""
    assert EmbeddingPurpose is not None  # importable
    # Runtime: any string is accepted; static type checking enforces the literal.
    # We don't enforce runtime narrowing — Python's behavior matches the contract.


@pytest.fixture
def env_isolation():
    """Save/restore env vars touched by these tests."""
    from hindsight_api.config import clear_config_cache

    keys = [
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER",
        "HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_DOC_PREFIX",
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL",
    ]
    original: dict[str, Any] = {k: os.environ.get(k) for k in keys}
    clear_config_cache()
    yield
    for k, v in original.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    clear_config_cache()


def test_config_reads_prefix_env_vars(env_isolation):
    """HINDSIGHT_API_EMBEDDINGS_{QUERY,DOC}_PREFIX populate the config dataclass."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX"] = "検索クエリ: "
    os.environ["HINDSIGHT_API_EMBEDDINGS_DOC_PREFIX"] = "検索文書: "

    config = HindsightConfig.from_env()
    assert config.embeddings_query_prefix == "検索クエリ: "
    assert config.embeddings_doc_prefix == "検索文書: "


def test_config_default_prefixes_are_empty(env_isolation):
    """No env var set → empty strings → byte-identical pre-existing behavior."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX", None)
    os.environ.pop("HINDSIGHT_API_EMBEDDINGS_DOC_PREFIX", None)

    config = HindsightConfig.from_env()
    assert config.embeddings_query_prefix == ""
    assert config.embeddings_doc_prefix == ""
