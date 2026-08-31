"""Tests for the ONNX Runtime embeddings provider."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.config import DEFAULT_EMBEDDINGS_ONNX_BATCH_SIZE, clear_config_cache
from hindsight_api.engine.embeddings import OnnxEmbeddings, create_embeddings_from_env


@pytest.fixture
def _fresh_config():
    """``create_embeddings_from_env`` reads the cached config; keep env-driven tests isolated."""
    clear_config_cache()
    yield
    clear_config_cache()


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        batch = len(texts)
        return {
            "input_ids": np.ones((batch, 3), dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0]] * batch, dtype=np.int64),
            "token_type_ids": np.zeros((batch, 3), dtype=np.int64),
        }


class FakeSessionOptions:
    def __init__(self):
        self.enable_cpu_mem_arena = True


class FakeOnnxSession:
    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        self.batch_sizes.append(batch)
        # Last token is masked out. Mean pooling should average first two tokens:
        # ([3, 4] + [0, 0]) / 2 = [1.5, 2.0], then normalize to [0.6, 0.8].
        token_embeddings = np.array([[[3.0, 4.0], [0.0, 0.0], [100.0, 100.0]]] * batch, dtype=np.float32)
        return [token_embeddings]


class FakeLengthTokenizer:
    """Encodes each text's length into its ids, padding to the longest text *in this call*."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.calls.append({"texts": list(texts)})
        lengths = [len(text) for text in texts]
        width = max(lengths)
        input_ids = np.zeros((len(texts), width), dtype=np.int64)
        attention_mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = length
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }


class FakeLengthOnnxSession:
    """Token embeddings read off the ids, so every text gets its own vector."""

    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        ids = inputs["input_ids"].astype(np.float32)
        self.batch_sizes.append(ids.shape[0])
        # [batch, seq, 2]: mean pooling over the unmasked tokens yields [length, 1.0].
        # Padded positions carry 0 and are masked out, which is what makes the result
        # independent of who else is in the batch.
        return [np.stack([ids, np.ones_like(ids)], axis=-1)]


def _length_embedder(batch_size: int) -> OnnxEmbeddings:
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        normalize=False,
        batch_size=batch_size,
    )
    emb._tokenizer = FakeLengthTokenizer()
    emb._session = FakeLengthOnnxSession()
    emb._dimension = 2
    return emb


class FakePooledOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        assert output_names == ["sentence_embedding"]
        return [np.array([[3.0, 4.0]] * batch, dtype=np.float32)]


def test_onnx_embeddings_mean_pooling_normalizes_and_filters_inputs():
    emb = OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", dimensions=2, max_tokens=17)
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([0.6, 0.8])]
    assert emb._tokenizer.calls[-1]["max_length"] == 17


def test_onnx_embeddings_cls_pooling_and_normalize_false():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        pooling="cls",
        normalize=False,
    )
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([3.0, 4.0])]


def test_onnx_embeddings_output_name_uses_pre_pooled_2d_output():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        output_name="sentence_embedding",
    )
    emb._tokenizer = FakeTokenizer()
    emb._session = FakePooledOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([0.6, 0.8])]


def test_onnx_embeddings_rejects_invalid_pooling_before_initialize():
    with pytest.raises(ValueError, match="pooling"):
        OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", pooling="max")


def test_onnx_embeddings_warns_when_local_model_path_has_no_tokenizer(caplog):
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/custom/onnx/model.onnx",
    )

    assert emb.tokenizer_name_or_path == "intfloat/multilingual-e5-small"
    assert "model_path is set without tokenizer_name_or_path" in caplog.text


def test_onnx_embeddings_query_and_document_prefixes_are_asymmetric():
    tokenizer = FakeTokenizer()
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        dimensions=2,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    emb._tokenizer = tokenizer
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    emb.encode_query(["weather"])
    emb.encode_documents(["weather"])

    assert tokenizer.calls[0]["texts"] == ["query: weather"]
    assert tokenizer.calls[1]["texts"] == ["passage: weather"]


@pytest.mark.asyncio
async def test_onnx_embeddings_dimension_mismatch_raises_value_error():
    emb = OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/e5/onnx/model.onnx",
        tokenizer_name_or_path="/models/e5",
        dimensions=3,
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakeTokenizer()))
    )
    fake_onnxruntime = SimpleNamespace(
        InferenceSession=MagicMock(return_value=FakeOnnxSession()),
        SessionOptions=FakeSessionOptions,
    )

    with patch.dict(sys.modules, {"transformers": fake_transformers, "onnxruntime": fake_onnxruntime}):
        with pytest.raises(ValueError, match="does not match model output"):
            await emb.initialize()


@pytest.mark.asyncio
async def test_onnx_embeddings_downloads_external_data_sidecar_when_needed():
    emb = OnnxEmbeddings(model_id="BAAI/bge-m3", onnx_file="onnx/model.onnx")
    download = MagicMock(return_value="/hf/bge-m3")
    session = MagicMock(return_value=FakeOnnxSession())
    fake_hf = SimpleNamespace(snapshot_download=download)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakeTokenizer()))
    )
    fake_onnxruntime = SimpleNamespace(InferenceSession=session, SessionOptions=FakeSessionOptions)

    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": fake_hf,
            "transformers": fake_transformers,
            "onnxruntime": fake_onnxruntime,
        },
    ):
        await emb.initialize()

    download.assert_called_once_with(
        repo_id="BAAI/bge-m3",
        allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data"],
    )
    assert session.call_count == 1
    assert session.call_args.args == ("/hf/bge-m3/onnx/model.onnx",)
    assert session.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    # The CPU memory arena caches freed activation blocks and never returns them, so
    # RSS would hold the high-water plateau for the life of the process (issue #3891).
    assert session.call_args.kwargs["sess_options"].enable_cpu_mem_arena is False


def test_create_embeddings_from_env_supports_onnx_provider():
    mock_config = MagicMock()
    mock_config.embeddings_provider = "onnx"
    mock_config.embeddings_onnx_model_id = "intfloat/multilingual-e5-small"
    mock_config.embeddings_onnx_model_path = "/models/e5/onnx/model.onnx"
    mock_config.embeddings_onnx_tokenizer_name_or_path = "/models/e5"
    mock_config.embeddings_onnx_file = "onnx/model.onnx"
    mock_config.embeddings_onnx_dimensions = 384
    mock_config.embeddings_onnx_max_tokens = 512
    mock_config.embeddings_onnx_pooling = "mean"
    mock_config.embeddings_onnx_normalize = True
    mock_config.embeddings_onnx_query_prefix = "query: "
    mock_config.embeddings_onnx_passage_prefix = "passage: "
    mock_config.embeddings_onnx_output_name = None
    mock_config.embeddings_onnx_batch_size = 16
    mock_config.embeddings_onnx_cpu_mem_arena = False

    with patch("hindsight_api.config.get_config", return_value=mock_config):
        emb = create_embeddings_from_env()

    assert isinstance(emb, OnnxEmbeddings)
    assert emb.provider_name == "onnx"
    assert emb.model_id == "intfloat/multilingual-e5-small"
    assert emb.model_path == "/models/e5/onnx/model.onnx"
    assert emb.tokenizer_name_or_path == "/models/e5"
    assert emb.dimension == 384
    assert emb.batch_size == 16
    assert emb.cpu_mem_arena is False


def test_onnx_embeddings_chunks_into_batches_and_preserves_order():
    """One forward pass per batch, with the caller's ordering restored (issue #3891)."""
    emb = _length_embedder(batch_size=2)
    texts = ["a", "bbbb", "cc", "ddddddd", "e"]

    result = emb.encode(texts)

    assert emb._session.batch_sizes == [2, 2, 1]
    # Each vector carries its own text's length, so a misplaced scatter-back shows up here.
    assert result == [pytest.approx([float(len(text)), 1.0]) for text in texts]
    # Longest first, so a single long text cannot pad an entire batch of short ones.
    assert [call["texts"] for call in emb._tokenizer.calls] == [["ddddddd", "bbbb"], ["cc", "a"], ["e"]]


def test_onnx_embeddings_single_batch_when_input_fits():
    emb = _length_embedder(batch_size=8)

    result = emb.encode(["a", "bbb", "cc"])

    assert emb._session.batch_sizes == [3]
    # Input order is untouched when nothing needs splitting.
    assert emb._tokenizer.calls[0]["texts"] == ["a", "bbb", "cc"]
    assert result == [pytest.approx([1.0, 1.0]), pytest.approx([3.0, 1.0]), pytest.approx([2.0, 1.0])]


def test_onnx_embeddings_batching_does_not_change_vectors():
    """Batch composition cannot move a vector: pooling masks padding."""
    texts = [f"text {'x' * index}" for index in range(10)]

    unbatched = _length_embedder(batch_size=len(texts))
    batched = _length_embedder(batch_size=3)

    assert batched.encode(texts) == unbatched.encode(texts)
    # The two runs really did pad to different widths — otherwise this proves nothing.
    assert unbatched._session.batch_sizes == [10]
    assert batched._session.batch_sizes == [3, 3, 3, 1]


def test_onnx_embeddings_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", batch_size=0)


def test_onnx_batch_size_and_arena_come_from_env(monkeypatch, _fresh_config):
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "onnx")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ONNX_BATCH_SIZE", "8")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ONNX_CPU_MEM_ARENA", "true")

    embeddings = create_embeddings_from_env()

    assert isinstance(embeddings, OnnxEmbeddings)
    assert embeddings.batch_size == 8
    assert embeddings.cpu_mem_arena is True


def test_onnx_batch_size_and_arena_defaults(monkeypatch, _fresh_config):
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "onnx")
    monkeypatch.delenv("HINDSIGHT_API_EMBEDDINGS_ONNX_BATCH_SIZE", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_EMBEDDINGS_ONNX_CPU_MEM_ARENA", raising=False)

    embeddings = create_embeddings_from_env()

    assert isinstance(embeddings, OnnxEmbeddings)
    assert embeddings.batch_size == DEFAULT_EMBEDDINGS_ONNX_BATCH_SIZE
    # Off by default: an enabled arena never returns freed blocks to the OS.
    assert embeddings.cpu_mem_arena is False
