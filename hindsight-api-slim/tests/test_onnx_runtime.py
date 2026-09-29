"""Opt-in real model/device tests; see docker/docker-compose/cuda-onnx/README.md.

No mocks replace inference. CUDA profiling must record real GPU kernels, not
merely list an installed execution provider. Ordinary CPU CI needs no downloads.
"""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hindsight_api.engine.embeddings import OnnxEmbeddings

MODEL_DIR = os.environ.get("HINDSIGHT_TEST_ONNX_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="set HINDSIGHT_TEST_ONNX_MODEL_DIR to a local E5 ONNX snapshot")


def real_embedder(device: str, **kwargs) -> OnnxEmbeddings:
    root = Path(MODEL_DIR)
    return OnnxEmbeddings(
        model_id="intfloat/multilingual-e5-small",
        model_path=str(root / "onnx/model.onnx"),
        tokenizer_name_or_path=str(root),
        device=device,
        batch_size=2,
        **kwargs,
    )


async def test_real_cpu_batched_multilingual_embeddings():
    emb = real_embedder("cpu")
    await emb.initialize()
    texts = ["Alice lives in Berlin", "爱丽丝住在柏林", "A blue bicycle", "", "résumé 🎻"]
    batched = np.array(await emb.encode_documents(texts))
    singles = np.concatenate([await emb.encode_documents([text]) for text in texts])
    assert batched.shape == (5, 384)
    assert np.isfinite(batched).all()
    np.testing.assert_allclose(np.linalg.norm(batched, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(batched, singles, atol=1e-5, rtol=1e-4)
    assert await emb.encode([]) == []
    assert emb._session.get_providers() == ["CPUExecutionProvider"]


@pytest.mark.skipif(
    os.environ.get("HINDSIGHT_TEST_ONNX_CUDA") != "1", reason="set HINDSIGHT_TEST_ONNX_CUDA=1 on a GPU host"
)
async def test_real_cuda_parity_concurrency_and_kernel_execution(monkeypatch, tmp_path):
    import onnxruntime as ort

    # Only enable ORT's profiler; graph execution and tokenization remain real.
    session_options = ort.SessionOptions

    def profiled_options():
        options = session_options()
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / "ort")
        return options

    cpu = real_embedder("cpu")
    await cpu.initialize()
    monkeypatch.setattr(ort, "SessionOptions", profiled_options)
    gpu = real_embedder("cuda")
    await gpu.initialize()
    assert gpu._session.get_providers()[0] == "CUDAExecutionProvider"
    texts = ["Alice lives in Berlin", "爱丽丝住在柏林", "A blue bicycle", "", "résumé 🎻"]
    reference = np.array(await cpu.encode_documents(texts))
    actual = np.array(await gpu.encode_documents(texts))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
    results = await asyncio.gather(*(gpu.encode_documents(texts) for _ in range(3)))
    for result in results:
        np.testing.assert_allclose(result, actual, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(
        await gpu.encode_query(["Where does Alice live?"]),
        await cpu.encode_query(["Where does Alice live?"]),
        atol=1e-4,
        rtol=1e-3,
    )
    assert await gpu.encode([]) == []
    profile_path = Path(gpu._session.end_profiling())
    events = json.loads(profile_path.read_text())
    cuda_nodes = [event for event in events if event.get("args", {}).get("provider") == "CUDAExecutionProvider"]
    assert cuda_nodes, "CUDA was listed but no graph nodes actually executed on the GPU"
    print(f"CUDA kernel events: {len(cuda_nodes)}; max vector delta: {np.max(np.abs(actual - reference)):.8f}")


@pytest.mark.skipif(os.environ.get("HINDSIGHT_TEST_ONNX_CUDA") != "1", reason="requires explicit CUDA validation")
async def test_real_invalid_cuda_device_never_returns_cpu_session():
    emb = real_embedder("cuda", cuda_device_id=999999)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CUDA"):
            await emb.initialize()
        assert emb._session is None
        assert emb._tokenizer is None
