"""Tests for the ONNX Runtime reranker provider (issue #5567)."""

import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine import cross_encoder
from hindsight_api.engine.cross_encoder import OnnxCrossEncoder, RerankTimeoutError, create_cross_encoder_from_env


@pytest.fixture
def _fresh_config():
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture(autouse=True)
def _no_shared_executor():
    OnnxCrossEncoder._executor = None
    yield
    OnnxCrossEncoder._executor = None


class FakePairTokenizer:
    """Encodes each pair's document length as its ids; no token_type_ids (XLM-R style)."""

    def __call__(self, queries, docs, padding, truncation, max_length, return_tensors):
        assert len(queries) == len(docs)
        lengths = [len(doc) for doc in docs]
        width = max(lengths)
        input_ids = np.zeros((len(docs), width), dtype=np.int64)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = length
        return {"input_ids": input_ids, "attention_mask": (input_ids > 0).astype(np.int64)}


class FakeSession:
    """Scores a pair by its document length; ``labels`` picks the output head shape."""

    def __init__(self, labels: int = 1):
        self.labels = labels
        self.calls: list[dict] = []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, output_names, inputs):
        self.calls.append(inputs)
        length = inputs["input_ids"][:, 0].astype(np.float32)
        if self.labels == 1:
            return [length[:, None]]
        return [np.stack([np.zeros_like(length)] * (self.labels - 1) + [length], axis=1)]


def _encoder(labels: int = 1, **kwargs) -> OnnxCrossEncoder:
    encoder = OnnxCrossEncoder(model_id="test/model", **kwargs)
    encoder._session = FakeSession(labels)
    encoder._tokenizer = FakePairTokenizer()
    return encoder


DOCS = ["a" * n for n in (3, 9, 1, 7, 5)]
PAIRS = [("q", doc) for doc in DOCS]


def test_one_logit_scores_are_raw_and_in_input_order():
    assert _encoder()._predict_sync(PAIRS) == [3.0, 9.0, 1.0, 7.0, 5.0]


def test_two_logits_score_is_log_odds_of_relevant():
    encoder = _encoder(labels=2)
    # [0, len] heads -> len - 0; sigmoid of it equals softmax P(relevant).
    assert encoder._predict_sync(PAIRS) == [3.0, 9.0, 1.0, 7.0, 5.0]


def test_more_than_two_logits_is_rejected():
    with pytest.raises(ValueError, match="3 logits per pair"):
        _encoder(labels=3)._predict_sync(PAIRS)


def test_batches_are_bounded_and_batching_does_not_change_scores():
    small = _encoder(batch_size=2)
    assert small._predict_sync(PAIRS) == _encoder()._predict_sync(PAIRS)
    calls = small._session.calls
    assert [c["input_ids"].shape[0] for c in calls] == [2, 2, 1]
    # Longest first, so similar lengths share a pass and short pairs are not padded out.
    assert [c["input_ids"].shape[1] for c in calls] == [9, 5, 1]
    # The graph wants token_type_ids the tokenizer did not produce: zeros are supplied.
    assert all((c["token_type_ids"] == 0).all() for c in calls)


@pytest.mark.asyncio
async def test_predict_runs_on_the_executor_and_handles_empty_input():
    encoder = _encoder()
    OnnxCrossEncoder._executor = ThreadPoolExecutor(max_workers=1)
    assert await encoder.predict([]) == []
    assert await encoder.predict(PAIRS) == [3.0, 9.0, 1.0, 7.0, 5.0]
    OnnxCrossEncoder._executor.shutdown()


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"threads": -1}])
def test_invalid_settings_fail_fast(kwargs):
    with pytest.raises(ValueError):
        OnnxCrossEncoder(**kwargs)


@pytest.mark.asyncio
async def test_initialize_bounds_threads_and_disables_arena():
    entries: dict[str, str] = {}
    options = SimpleNamespace(
        enable_cpu_mem_arena=True, intra_op_num_threads=0, add_session_config_entry=entries.__setitem__
    )
    created = {}

    def fake_session(path, sess_options, providers):
        created.update(path=path, options=sess_options, providers=providers)
        return FakeSession()

    fake_ort = SimpleNamespace(SessionOptions=lambda: options, InferenceSession=fake_session)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock(return_value=FakePairTokenizer()))
    )
    encoder = OnnxCrossEncoder(model_path="/models/r/model.onnx", tokenizer_name_or_path="/models/r", threads=3)
    with patch.dict(sys.modules, {"onnxruntime": fake_ort, "transformers": fake_transformers}):
        await encoder.initialize()

    assert created["path"] == "/models/r/model.onnx"
    assert created["providers"] == ["CPUExecutionProvider"]
    assert options.enable_cpu_mem_arena is False
    assert options.intra_op_num_threads == 3
    assert entries == {"session.intra_op.allow_spinning": "0"}
    fake_transformers.AutoTokenizer.from_pretrained.assert_called_once_with("/models/r")
    assert OnnxCrossEncoder._executor is not None


def test_factory_reads_primary_and_indexed_onnx_settings(monkeypatch, _fresh_config):
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_PROVIDER", "onnx")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ONNX_MODEL_PATH", "/models/kure/model.onnx")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ONNX_TOKENIZER_NAME_OR_PATH", "/models/kure")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ONNX_BATCH_SIZE", "8")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_ONNX_THREADS", "2")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_1_PROVIDER", "onnx")
    monkeypatch.setenv("HINDSIGHT_API_RERANKER_1_ONNX_MODEL_ID", "BAAI/bge-reranker-base")

    chain = create_cross_encoder_from_env()._members
    primary, fallback = chain
    assert isinstance(primary, OnnxCrossEncoder) and isinstance(fallback, OnnxCrossEncoder)
    assert primary.model_path == "/models/kure/model.onnx"
    assert primary.tokenizer_name_or_path == "/models/kure"
    assert (primary.batch_size, primary.threads) == (8, 2)
    assert primary.retry_policy is None
    # Indexed members inherit nothing from the primary.
    assert fallback.model_id == "BAAI/bge-reranker-base"
    assert fallback.model_path is None
    assert fallback.batch_size == 32


def test_budget_exhaustion_returns_what_was_scored(monkeypatch):
    """A model too slow for the CPU hands back partial scores instead of holding recall."""
    clock = iter([0.0, 0.0, 99.0])  # deadline set, first pass in time, second pass too late
    monkeypatch.setattr(cross_encoder.time, "monotonic", lambda: next(clock))
    encoder = _encoder(batch_size=2, timeout=10.0)

    with pytest.raises(RerankTimeoutError) as exc_info:
        encoder._predict_sync(PAIRS)
    # The longest two (9 and 7 chars) were scored in the first pass; the rest were not.
    assert exc_info.value.scores == [None, 9.0, None, 7.0, None]


def test_zero_timeout_disables_the_budget():
    assert _encoder(batch_size=1, timeout=0)._predict_sync(PAIRS) == [3.0, 9.0, 1.0, 7.0, 5.0]
