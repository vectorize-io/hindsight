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
