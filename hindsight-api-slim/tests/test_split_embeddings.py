"""Separate query and passage embedders (#6078, Qdrant Constella).

A QUERY or PASSAGE embeddings side is the main embeddings config with any
``HINDSIGHT_API_<SIDE>_EMBEDDINGS_<X>`` laid over ``HINDSIGHT_API_EMBEDDINGS_<X>``. These
tests cover the inheritance, the routing of each input type to its side, and that the
two sides do not share a request bound — the reason #6078 asked for this.
"""

import asyncio
import threading

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.embeddings import (
    Embeddings,
    OnnxEmbeddings,
    OpenAIEmbeddings,
    RemoteTEIEmbeddings,
    SplitEmbeddings,
    create_embeddings_from_env,
    embedding_model_id,
)
from hindsight_api.engine.retain import embedding_utils


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    clear_config_cache()
    yield monkeypatch
    clear_config_cache()


def _split(embeddings: Embeddings) -> SplitEmbeddings:
    assert isinstance(embeddings, SplitEmbeddings)
    return embeddings


def test_no_side_configured_keeps_a_single_backend(env: pytest.MonkeyPatch) -> None:
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://tei")
    config = HindsightConfig.from_env()
    assert config.query_embeddings is None and config.passage_embeddings is None
    assert isinstance(create_embeddings_from_env(), RemoteTEIEmbeddings)


def test_query_base_url_only_overrides_the_url(env: pytest.MonkeyPatch) -> None:
    """#6078: same model and key, recall queries go to a second endpoint."""
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "openai")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY", "sk-shared")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", "BAAI/bge-m3")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL", "http://tei-bulk/v1")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_OPENAI_BASE_URL", "http://tei-small/v1")

    split = _split(create_embeddings_from_env())
    assert isinstance(split.query, OpenAIEmbeddings) and isinstance(split.passage, OpenAIEmbeddings)
    assert split.query.base_url == "http://tei-small/v1"
    assert split.passage.base_url == "http://tei-bulk/v1"
    # Everything not overridden is inherited from the main config.
    assert split.query.model == split.passage.model == "BAAI/bge-m3"
    assert split.query.api_key == "sk-shared"


def test_one_side_local_and_the_other_remote(env: pytest.MonkeyPatch) -> None:
    """Constella: documents from a remote Stella, queries from a small in-process model."""
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://stella")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_PROVIDER", "onnx")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_ONNX_MODEL_ID", "Qdrant/constella-nano")

    split = _split(create_embeddings_from_env())
    assert isinstance(split.query, OnnxEmbeddings)
    assert split.query.model_id == "Qdrant/constella-nano"
    assert isinstance(split.passage, RemoteTEIEmbeddings)
    # In-process backends keep the sequential default; the remote side gets the bound.
    assert split.query.max_concurrent_requests == 1
    assert split.passage.max_concurrent_requests > 1


def test_both_sides_can_be_specialized(env: pytest.MonkeyPatch) -> None:
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://main")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_TEI_URL", "http://query")
    env.setenv("HINDSIGHT_API_PASSAGE_EMBEDDINGS_TEI_URL", "http://passage")
    env.setenv("HINDSIGHT_API_PASSAGE_EMBEDDINGS_TEI_BATCH_SIZE", "7")

    split = _split(create_embeddings_from_env())
    assert isinstance(split.query, RemoteTEIEmbeddings) and isinstance(split.passage, RemoteTEIEmbeddings)
    assert (split.query.base_url, split.passage.base_url) == ("http://query", "http://passage")
    # The coalescer sizes hand-offs from the wrapper, so it must describe the passage side.
    assert split.passage.batch_size == split.batch_size == 7


def test_a_side_inherits_the_shared_key_fallback(env: pytest.MonkeyPatch) -> None:
    """Fallbacks outside the embeddings namespace (here the LLM key) apply to a side too."""
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://tei")
    env.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-llm")
    env.delenv("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY", raising=False)
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_PROVIDER", "openai")

    split = _split(create_embeddings_from_env())
    assert isinstance(split.query, OpenAIEmbeddings)
    assert split.query.api_key == "sk-llm"


def test_a_side_value_is_validated_like_the_main_one(env: pytest.MonkeyPatch) -> None:
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_TEI_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match=r"TEI_BATCH_SIZE.*query embeddings side, HINDSIGHT_API_QUERY_EMBEDDINGS_"):
        HindsightConfig.from_env()


class _Fake(Embeddings):
    def __init__(self, name: str, dimension: int = 2, hold: asyncio.Event | None = None) -> None:
        self.name = name
        self._dim = dimension
        self._hold = hold
        self.calls: list[list[str]] = []
        self.initialized = False

    @property
    def provider_name(self) -> str:
        return self.name

    @property
    def dimension(self) -> int:
        return self._dim

    async def initialize(self) -> None:
        self.initialized = True

    async def encode(self, texts: list[str]) -> list[list[float]]:
        async def run(batch: list[str]) -> list[list[float]]:
            self.calls.append(batch)
            if self._hold is not None:
                await self._hold.wait()
            return [[1.0] * self._dim for _ in batch]

        return await self._encode_batched(texts, run)


def _wrap(query: Embeddings, passage: Embeddings, query_cap: int | None = None) -> SplitEmbeddings:
    return SplitEmbeddings(
        query=query, passage=passage, query_max_input_tokens=query_cap, passage_max_input_tokens=None
    )


async def test_each_input_type_reaches_its_side() -> None:
    query, passage = _Fake("q"), _Fake("d")
    split = _wrap(query, passage)

    await embedding_utils.generate_embeddings_batch(split, ["who?"], input_type="query")
    await embedding_utils.generate_embeddings_batch(split, ["a fact"])

    assert query.calls == [["who?"]]
    assert passage.calls == [["a fact"]]


async def test_a_busy_passage_side_does_not_queue_queries() -> None:
    """The point of #6078: a held bulk request must not block a recall query."""
    release = asyncio.Event()
    query, passage = _Fake("q"), _Fake("d", hold=release)
    split = _wrap(query, passage)

    bulk = asyncio.create_task(embedding_utils.generate_embeddings_batch(split, ["bulk"]))
    await asyncio.sleep(0)
    vectors = await asyncio.wait_for(
        embedding_utils.generate_embeddings_batch(split, ["fast"], input_type="query"), timeout=2
    )
    assert vectors == [[1.0, 1.0]]
    assert not bulk.done()
    release.set()
    await bulk


async def test_initialize_fails_on_a_dimension_mismatch() -> None:
    split = _wrap(_Fake("q", dimension=384), _Fake("d", dimension=1024))
    with pytest.raises(ValueError, match="384-dim"):
        await split.initialize()


async def test_initialize_starts_both_sides() -> None:
    query, passage = _Fake("q"), _Fake("d")
    split = _wrap(query, passage)
    await split.initialize()
    assert query.initialized and passage.initialized
    assert split.dimension == 2


async def test_each_side_applies_its_own_input_token_cap() -> None:
    """A small query model takes fewer tokens than the passage model; its cap must hold."""
    query, passage = _Fake("q"), _Fake("d")
    split = _wrap(query, passage, query_cap=3)
    long_text = " ".join(["word"] * 50)

    await embedding_utils.generate_embeddings_batch(split, [long_text], input_type="query")
    await embedding_utils.generate_embeddings_batch(split, [long_text])

    assert len(query.calls[0][0].split()) <= 3
    assert passage.calls[0][0] == long_text


def test_a_side_cap_comes_from_its_own_variable(env: pytest.MonkeyPatch) -> None:
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://tei")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_MAX_INPUT_TOKENS", "512")

    split = _split(create_embeddings_from_env())
    assert split.max_input_tokens_for("query") == 512
    assert split.max_input_tokens_for("document") == HindsightConfig.from_env().embeddings_max_input_tokens


def test_a_misspelt_side_variable_is_refused(env: pytest.MonkeyPatch) -> None:
    """A typo must not quietly build a second copy of the main backend."""
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_TEI_URLL", "http://typo")
    with pytest.raises(ValueError, match="HINDSIGHT_API_QUERY_EMBEDDINGS_TEI_URLL"):
        HindsightConfig.from_env()


def test_an_empty_side_value_turns_the_side_on_and_clears_the_main_value(env: pytest.MonkeyPatch) -> None:
    env.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_TEI_URL", "http://tei")
    env.setenv("HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX", "query: ")
    env.setenv("HINDSIGHT_API_QUERY_EMBEDDINGS_QUERY_PREFIX", "")

    config = HindsightConfig.from_env()
    assert config.query_embeddings is not None
    assert config.query_embeddings.embeddings_query_prefix == ""
    assert config.embeddings_query_prefix == "query: "


async def test_a_local_side_loads_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    seen: list[int] = []

    class _Local(_Fake):
        @property
        def provider_name(self) -> str:
            return "local"

        async def initialize(self) -> None:
            seen.append(threading.get_ident())

    await _wrap(_Local("q"), _Fake("d")).initialize()
    assert seen and seen[0] != loop_thread


def test_the_written_model_is_the_passage_side() -> None:
    """A re-embed's write guard compares the model that writes stored vectors."""
    passage = _Fake("d")
    assert embedding_model_id(_wrap(_Fake("q"), passage)) == embedding_model_id(passage)
