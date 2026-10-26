"""Local torch models download in parallel but build one at a time (issue #4420).

A transformers build flips process-global state for its whole duration
(``PreTrainedModel.tie_weights`` becomes a no-op, torch's default dtype changes).
When the embedder and the reranker built in overlapping startup threads, about half
of cold starts with two Qwen3 models died with "Cannot copy out of meta tensor" or a
float/bfloat16 mismatch in the reranker's smoke test.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _Probe:
    """Records how many fake model builds run at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def build(self, result):
        def _build(*args, **kwargs):
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.1)  # wide enough that unserialized builds would overlap
            with self._lock:
                self.active -= 1
            return result

        return _build


def _fake_embedder() -> torch.nn.Module:
    model = torch.nn.Linear(4, 4)
    model.get_sentence_embedding_dimension = lambda: 4  # type: ignore[attr-defined]
    model.encode = lambda texts: np.zeros((len(texts), 4), dtype=np.float32)  # type: ignore[attr-defined]
    return model


def _fake_reranker() -> MagicMock:
    model = MagicMock()
    model.model = torch.nn.Linear(4, 4)
    model.predict.return_value = [1.0]
    return model


async def test_startup_loads_download_together_and_build_one_at_a_time():
    from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    probe = _Probe()
    # Both downloads must reach this point before either may continue: it breaks
    # (failing the load) if a download ever runs under the build lock.
    both_downloading = threading.Barrier(2, timeout=5)

    embeddings = LocalSTEmbeddings(model_name="test-embedder", force_cpu=True)
    reranker = LocalSTCrossEncoder(model_name="test-reranker", force_cpu=True)

    def init_in_thread(model) -> None:
        asyncio.run(model.initialize())

    with (
        patch(
            "hindsight_api.engine.embeddings.prefetch_local_model", lambda name, cache_dir=None: both_downloading.wait()
        ),
        patch(
            "hindsight_api.engine.cross_encoder.prefetch_local_model",
            lambda name, cache_dir=None: both_downloading.wait(),
        ),
        patch("sentence_transformers.SentenceTransformer", side_effect=probe.build(_fake_embedder())),
        patch("sentence_transformers.CrossEncoder", side_effect=probe.build(_fake_reranker())),
    ):
        await asyncio.gather(
            asyncio.to_thread(init_in_thread, embeddings),
            asyncio.to_thread(init_in_thread, reranker),
        )

    assert probe.max_active == 1


def test_prefetch_leaves_a_local_directory_alone(tmp_path):
    from hindsight_api.engine.local_device import prefetch_local_model

    with patch("huggingface_hub.snapshot_download") as download:
        prefetch_local_model(str(tmp_path))

    download.assert_not_called()


def test_prefetch_failure_is_left_to_the_loader():
    from hindsight_api.engine.local_device import prefetch_local_model

    with patch("huggingface_hub.snapshot_download", side_effect=OSError("offline")):
        prefetch_local_model("org/model")  # must not raise


async def test_prefetch_downloads_where_sentence_transformers_reads(monkeypatch, tmp_path):
    """Prefetching into the HF cache while sentence-transformers reads from its own
    home would download every model twice."""
    from hindsight_api.engine.embeddings import LocalSTEmbeddings

    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path))
    embeddings = LocalSTEmbeddings(model_name="org/model", force_cpu=True)

    with (
        patch("huggingface_hub.snapshot_download") as download,
        patch("sentence_transformers.SentenceTransformer", return_value=_fake_embedder()),
    ):
        await embeddings.initialize()

    assert download.call_args.kwargs["cache_dir"] == str(tmp_path)


class _StopInit(Exception):
    """Raised by the first model build: everything after it is out of scope."""


@pytest.mark.parametrize("loader", ["local-embeddings", "local-reranker", "onnx-embeddings", "onnx-reranker", "t5"])
async def test_loaders_touch_transformers_only_under_the_build_lock(monkeypatch, loader):
    """transformers' lazy namespace raises ImportError for a name another thread is
    still populating, so no loader may import it (or sentence_transformers, which
    imports it) or load a tokenizer outside the build lock (issue #6069)."""
    import sys
    import types

    from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder, OnnxCrossEncoder
    from hindsight_api.engine.embeddings import LocalSTEmbeddings, OnnxEmbeddings
    from hindsight_api.engine.local_device import LOCAL_MODEL_BUILD_LOCK
    from hindsight_api.engine.query_analyzer import TransformerQueryAnalyzer

    unlocked: list[str] = []

    def touch(what: str) -> None:
        if not LOCAL_MODEL_BUILD_LOCK.locked():
            unlocked.append(what)

    class _FakeLib(types.ModuleType):
        def __getattr__(self, name: str) -> MagicMock:
            if name.startswith("__"):
                raise AttributeError(name)
            touch(f"{self.__name__}.{name}")
            cls = MagicMock(side_effect=_StopInit)  # SentenceTransformer(...) / CrossEncoder(...)

            def from_pretrained(*args, **kwargs) -> MagicMock:
                touch(f"{name}.from_pretrained")
                if name != "AutoTokenizer":
                    raise _StopInit
                return MagicMock()

            cls.from_pretrained.side_effect = from_pretrained
            return cls

    ort = MagicMock()
    ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    ort.InferenceSession.side_effect = _StopInit
    monkeypatch.setitem(sys.modules, "transformers", _FakeLib("transformers"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeLib("sentence_transformers"))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    for module in ("embeddings", "cross_encoder", "local_device"):
        monkeypatch.setattr(f"hindsight_api.engine.{module}.prefetch_local_model", lambda *a, **k: None)

    with pytest.raises(_StopInit):
        if loader == "local-embeddings":
            await LocalSTEmbeddings(model_name="org/m", force_cpu=True).initialize()
        elif loader == "local-reranker":
            await LocalSTCrossEncoder(model_name="org/m", force_cpu=True).initialize()
        elif loader == "onnx-embeddings":
            await OnnxEmbeddings(model_id="org/m", model_path="/m/model.onnx", tokenizer_name_or_path="/m").initialize()
        elif loader == "onnx-reranker":
            await OnnxCrossEncoder(model_path="/m/model.onnx", tokenizer_name_or_path="/m").initialize()
        else:
            TransformerQueryAnalyzer(model_name="org/m").load()

    assert unlocked == []
