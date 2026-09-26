"""CUDA opt-in, fail-fast and retry contracts without requiring GPU CI."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.embeddings import OnnxEmbeddings, create_embeddings_from_env
from tests.test_onnx_embeddings import FakeOnnxSession, FakeSessionOptions, FakeTokenizer


@pytest.fixture
def ort_runtime(monkeypatch):
    session = MagicMock(wraps=FakeOnnxSession())
    session.disable_fallback = MagicMock()
    session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    runtime = SimpleNamespace(
        InferenceSession=MagicMock(return_value=session),
        SessionOptions=FakeSessionOptions,
        get_available_providers=MagicMock(return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    tokenizer = MagicMock(return_value=FakeTokenizer())
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer))
    )
    return runtime


def embedder(**kwargs) -> OnnxEmbeddings:
    return OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path="/models/e5/onnx/model.onnx",
        tokenizer_name_or_path="/models/e5",
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("missing_module", ["onnxruntime", "transformers"])
async def test_missing_dependency_advice_matches_device(monkeypatch, ort_runtime, device, missing_module):
    monkeypatch.setitem(sys.modules, missing_module, None)
    emb = embedder(device=device)
    with pytest.raises(ImportError) as raised:
        await emb.initialize()
    message = str(raised.value)
    if device == "cuda":
        assert "onnxruntime-gpu" in message
        assert "docker/docker-compose/cuda-onnx/" in message
        assert "pip install 'hindsight-api-slim[local-onnx]'" not in message
    else:
        assert "pip install 'hindsight-api-slim[local-onnx]'" in message
    assert isinstance(raised.value.__cause__, ImportError)
    assert emb._session is None
    assert emb._tokenizer is None


@pytest.mark.parametrize("device,device_id", [("cpu", 0), ("cuda", 0), ("cuda", 2)])
@pytest.mark.parametrize("arena", [False, True])
async def test_device_selection_and_warmup(ort_runtime, caplog, device, device_id, arena):
    session = ort_runtime.InferenceSession.return_value
    if device == "cpu":
        session.get_providers.return_value = ["CPUExecutionProvider"]
    emb = embedder(device=device, cuda_device_id=device_id, cpu_mem_arena=arena)
    await emb.initialize()
    await emb.initialize()
    expected = [("CUDAExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"]
    assert ort_runtime.InferenceSession.call_args.kwargs["providers"] == (
        expected if device == "cuda" else ["CPUExecutionProvider"]
    )
    ort_runtime.InferenceSession.assert_called_once()
    options = ort_runtime.InferenceSession.call_args.kwargs["sess_options"]
    assert options is None if arena else options.enable_cpu_mem_arena is False
    assert session.disable_fallback.call_count == (1 if device == "cuda" else 0)
    assert ort_runtime.get_available_providers.call_count == (1 if device == "cuda" else 0)
    assert emb.dimension == 2
    assert await emb.encode_query(["weather"]) == [pytest.approx([0.6, 0.8])]
    assert f"device: {device}" in caplog.text
    assert "providers:" in caplog.text


async def test_unavailable_cuda_fails_before_download_or_tokenization(ort_runtime, monkeypatch):
    ort_runtime.get_available_providers.return_value = ["CPUExecutionProvider"]
    download = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    emb = OnnxEmbeddings(model_id="intfloat/multilingual-e5-small", device="cuda")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
            await emb.initialize()
    download.assert_not_called()
    sys.modules["transformers"].AutoTokenizer.from_pretrained.assert_not_called()
    ort_runtime.InferenceSession.assert_not_called()
    assert emb._session is None
    assert emb._tokenizer is None


async def test_library_failure_cannot_silently_fall_back_or_poison_retry(ort_runtime):
    ort_runtime.InferenceSession.return_value.get_providers.return_value = ["CPUExecutionProvider"]
    emb = embedder(device="cuda")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="did not activate CUDAExecutionProvider"):
            await emb.initialize()
        assert emb._session is None
        assert emb._tokenizer is None
    assert ort_runtime.InferenceSession.call_count == 2
    ort_runtime.InferenceSession.return_value.run.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_session_creation_error_preserves_cause(ort_runtime, device):
    failure = ValueError("invalid model or device")
    ort_runtime.InferenceSession.side_effect = failure
    emb = embedder(device=device)
    with pytest.raises(RuntimeError if device == "cuda" else ValueError) as raised:
        await emb.initialize()
    if device == "cuda":
        assert raised.value.__cause__ is failure
        assert "device 0" in str(raised.value)
    else:
        assert raised.value is failure
    assert emb._session is None
    assert emb._tokenizer is None


@pytest.mark.parametrize("failure", [RuntimeError("CUDA out of memory"), None])
async def test_failed_warmup_is_retried(ort_runtime, failure):
    session = ort_runtime.InferenceSession.return_value
    session.run.side_effect = failure
    emb = embedder(device="cuda", dimensions=3 if failure is None else None)
    with pytest.raises((RuntimeError, ValueError)):
        await emb.initialize()
    assert emb._session is None
    assert emb._tokenizer is None
    session.run.side_effect = None
    emb.configured_dimensions = 2
    await emb.initialize()
    assert emb.dimension == 2
    assert ort_runtime.InferenceSession.call_count == 2


async def test_inference_error_propagates_without_reinitialization(ort_runtime):
    emb = embedder(device="cuda")
    await emb.initialize()
    ort_runtime.InferenceSession.return_value.run.side_effect = RuntimeError("GPU failure")
    with pytest.raises(RuntimeError, match="GPU failure"):
        await emb.encode(["hello"])
    ort_runtime.InferenceSession.assert_called_once()
    ort_runtime.InferenceSession.return_value.disable_fallback.assert_called_once()


@pytest.mark.parametrize("device", ["xpu", "auto", ""])
def test_invalid_device(device):
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        embedder(device=device)


@pytest.mark.parametrize("device_id", [-1, 1.5, "0", True])
def test_invalid_device_id(device_id):
    with pytest.raises(ValueError, match="device ID"):
        embedder(cuda_device_id=device_id)


@pytest.mark.parametrize(
    "variable,value",
    [
        ("DEVICE", "auto"),
        ("CUDA_DEVICE_ID", "-1"),
        ("CUDA_DEVICE_ID", "1.5"),
        ("CUDA_DEVICE_ID", "abc"),
    ],
)
def test_invalid_device_environment(monkeypatch, variable, value):
    clear_config_cache()
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "onnx")
    name = f"HINDSIGHT_API_EMBEDDINGS_ONNX_{variable}"
    monkeypatch.setenv(name, value)
    try:
        with pytest.raises(ValueError, match=name):
            create_embeddings_from_env()
    finally:
        clear_config_cache()


@pytest.mark.parametrize(
    "device,device_id,expected_device,expected_id",
    [("CUDA", "2", "cuda", 2), ("cpu", "0", "cpu", 0), ("", "", "cpu", 0)],
)
def test_device_environment_normalization(monkeypatch, device, device_id, expected_device, expected_id):
    clear_config_cache()
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_PROVIDER", "onnx")
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ONNX_DEVICE", device)
    monkeypatch.setenv("HINDSIGHT_API_EMBEDDINGS_ONNX_CUDA_DEVICE_ID", device_id)
    try:
        emb = create_embeddings_from_env()
        assert emb.device == expected_device
        assert emb.cuda_device_id == expected_id
    finally:
        clear_config_cache()


def test_device_selection_is_static_configuration():
    fields = {"embeddings_onnx_device", "embeddings_onnx_cuda_device_id"}
    assert not fields & HindsightConfig.get_configurable_fields()
    assert fields <= HindsightConfig.get_static_fields()
