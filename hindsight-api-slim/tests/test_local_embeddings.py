from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.engine.embeddings import LocalSTEmbeddings


async def test_local_embeddings_use_sentence_transformer_input_types():
    model = MagicMock()
    model.encode.return_value = np.array([[1.0]])
    model.encode_query.return_value = np.array([[2.0]])
    model.encode_document.return_value = np.array([[3.0]])

    embeddings = LocalSTEmbeddings()
    embeddings._model = model
    embeddings._device_type = "mps"

    with patch("hindsight_api.engine.embeddings.release_local_inference_memory") as release:
        assert await embeddings.encode(["text"]) == [[1.0]]
        assert await embeddings.encode_query(["query"]) == [[2.0]]
        assert await embeddings.encode_documents(["document"]) == [[3.0]]

    model.encode.assert_called_once_with(["text"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_query.assert_called_once_with(["query"], convert_to_numpy=True, show_progress_bar=False)
    model.encode_document.assert_called_once_with(["document"], convert_to_numpy=True, show_progress_bar=False)
    assert release.call_count == 3


def test_sentence_transformer_exposes_input_type_entry_points():
    """Guard the >=5.0 floor in local-ml.

    The test above drives a MagicMock, so it passes on any sentence-transformers
    version while the real one raises AttributeError at the first encode. These
    entry points only exist from 5.0 onwards; assert against the real class so a
    loosened pin fails here instead of at recall time.
    """
    sentence_transformers = pytest.importorskip("sentence_transformers")

    assert hasattr(sentence_transformers.SentenceTransformer, "encode_query")
    assert hasattr(sentence_transformers.SentenceTransformer, "encode_document")


@pytest.mark.parametrize(
    "methods",
    [
        # 5.4+: both exist, the old one emits a FutureWarning, so it must not be called.
        ["get_embedding_dimension", "get_sentence_embedding_dimension"],
        # 5.0-5.3 (still allowed by local-ml): only the old name exists.
        ["get_sentence_embedding_dimension"],
    ],
)
async def test_local_embeddings_read_the_dimension_on_every_supported_version(methods: list[str]):
    model = MagicMock(spec=[*methods, "encode"])
    if "get_embedding_dimension" in methods:
        model.get_embedding_dimension.return_value = 8
        model.get_sentence_embedding_dimension.side_effect = AssertionError("deprecated method called")
    else:
        model.get_sentence_embedding_dimension.return_value = 8
    model.encode.side_effect = lambda texts, **_: np.zeros((len(texts), 8), dtype=np.float32)
    embeddings = LocalSTEmbeddings(model_name="test-model", force_cpu=True)

    with (
        patch("sentence_transformers.SentenceTransformer", return_value=model),
        patch("hindsight_api.engine.embeddings.align_local_model_weights"),
        patch("hindsight_api.engine.embeddings.resolve_model_device_type", return_value="cpu"),
    ):
        await embeddings.initialize()

    assert embeddings.dimension == 8


async def test_local_embeddings_run_on_a_fixed_pool_and_keep_the_callers_context(monkeypatch):
    """#6110: every thread that runs torch keeps its own native pools for life, so
    forward passes must stay on the model's own fixed pool, however many callers come
    in at once — not spread over the default executor's threads."""
    import asyncio
    import contextvars
    import threading
    import time

    monkeypatch.setattr(LocalSTEmbeddings, "_executor", None, raising=False)
    bank: contextvars.ContextVar[str] = contextvars.ContextVar("bank")
    threads: set[int] = set()
    seen_banks: set[str] = set()

    def encode_query(texts, **_):
        threads.add(threading.get_ident())
        seen_banks.add(bank.get())
        time.sleep(0.01)  # hold the thread so callers pile up
        return np.zeros((len(texts), 1), dtype=np.float32)

    model = MagicMock()
    model.encode_query.side_effect = encode_query
    embeddings = LocalSTEmbeddings(max_concurrent=2)
    embeddings._model = model

    async def caller(i: int) -> None:
        bank.set(f"bank-{i}")
        await embeddings.encode_query([f"q{i}"])

    try:
        with patch("hindsight_api.engine.embeddings.release_local_inference_memory"):
            await asyncio.gather(*(caller(i) for i in range(20)))
    finally:
        if LocalSTEmbeddings._executor is not None:
            LocalSTEmbeddings._executor.shutdown(wait=True)

    assert len(threads) <= 2
    assert seen_banks == {f"bank-{i}" for i in range(20)}
