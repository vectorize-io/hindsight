"""Tests for the ONNX Runtime embeddings provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.engine.embeddings import OnnxEmbeddings, create_embeddings_from_env


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


class FakeOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, inputs):
        batch = inputs["input_ids"].shape[0]
        # Last token is masked out. Mean pooling should average first two tokens:
        # ([3, 4] + [0, 0]) / 2 = [1.5, 2.0], then normalize to [0.6, 0.8].
        token_embeddings = np.array([[[3.0, 4.0], [0.0, 0.0], [100.0, 100.0]]] * batch, dtype=np.float32)
        return [token_embeddings]


def test_onnx_embeddings_mean_pooling_normalizes_and_filters_inputs():
    emb = OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", dimensions=2, max_tokens=17)
    emb._tokenizer = FakeTokenizer()
    emb._session = FakeOnnxSession()
    emb._dimension = 2

    result = emb.encode(["hello"])

    assert result == [pytest.approx([0.6, 0.8])]
    assert emb._tokenizer.calls[-1]["max_length"] == 17


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

    with patch("hindsight_api.config.get_config", return_value=mock_config):
        emb = create_embeddings_from_env()

    assert isinstance(emb, OnnxEmbeddings)
    assert emb.provider_name == "onnx"
    assert emb.model_id == "intfloat/multilingual-e5-small"
    assert emb.model_path == "/models/e5/onnx/model.onnx"
    assert emb.tokenizer_name_or_path == "/models/e5"
    assert emb.dimension == 384
