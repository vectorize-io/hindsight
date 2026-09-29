"""Tests for the weight-alignment guard in engine/local_device.py.

``transformers`` maps safetensors weights zero-copy, so every parameter is a view
onto the file at ``8 + len(json_header)`` bytes. When that offset is not a multiple
of the dtype size, torch's vectorized CPU matmul reads the operand misaligned and
returns wrong values — on arm64 with torch 2.10 a float32 matmul against a
2-byte-misaligned weight corrupts ~6% of the output columns, only a few of which
show up as NaN. The default reranker (ms-marco-MiniLM-L-6-v2, 12090-byte header →
``ptr % 4 == 2``) returned NaN for every pair, which reranking sanitizes to 0.0.

Whether a model file is affected depends on its header length, so these tests build
the misaligned case explicitly rather than relying on a particular model or platform.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hindsight_api.engine import local_device  # noqa: E402
from hindsight_api.engine.local_device import (  # noqa: E402
    align_local_model_weights,
    assert_finite_local_output,
)


def _misaligned_tensor(values: "torch.Tensor", byte_offset: int = 2) -> "torch.Tensor":
    """Return a tensor holding ``values`` at a deliberate sub-element byte offset.

    torch's own APIs refuse to create this (``Tensor.view`` requires the storage
    offset to divide the element size), which is exactly why it only ever arises
    from a zero-copy map of an external buffer. numpy allows it, and
    ``torch.from_numpy`` keeps the pointer, so this reproduces what safetensors
    hands transformers.
    """
    payload = values.contiguous().numpy().tobytes()
    buffer = bytearray(byte_offset) + bytearray(payload) + bytearray(64)
    array = np.frombuffer(memoryview(buffer), dtype=np.float32, count=values.numel(), offset=byte_offset).reshape(
        values.shape
    )
    tensor = torch.from_numpy(array)
    # Keep the backing bytearray alive for as long as the tensor is.
    tensor._test_backing_buffer = buffer  # type: ignore[attr-defined]
    return tensor


class _Tiny(torch.nn.Module):
    """A one-linear-layer module whose weight/buffer alignment the test controls."""

    def __init__(self, *, misaligned: bool):
        super().__init__()
        torch.manual_seed(0)
        weight = torch.randn(64, 64) * 0.05
        scale = torch.randn(64)
        if misaligned:
            weight = _misaligned_tensor(weight)
            scale = _misaligned_tensor(scale)
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer("scale", scale)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return (x @ self.weight.T) * self.scale


def _all_tensors(module: "torch.nn.Module") -> list["torch.Tensor"]:
    return [t for _, t in list(module.named_parameters()) + list(module.named_buffers())]


class TestAlignLocalModelWeights:
    def test_misaligned_parameter_and_buffer_are_copied(self):
        model = _Tiny(misaligned=True)
        assert all(t.data_ptr() % t.element_size() != 0 for t in _all_tensors(model))

        copied = align_local_model_weights(model, label="Test")

        assert copied == 2  # the parameter and the buffer
        assert all(t.data_ptr() % t.element_size() == 0 for t in _all_tensors(model))

    def test_aligned_model_is_not_copied(self):
        """The common case must cost a pointer check, not a duplicate of the model."""
        model = _Tiny(misaligned=False)
        pointers_before = [t.data_ptr() for t in _all_tensors(model)]

        copied = align_local_model_weights(model, label="Test")

        assert copied == 0
        assert [t.data_ptr() for t in _all_tensors(model)] == pointers_before

    def test_values_are_preserved_exactly(self):
        """Copying must move the bytes, not reinterpret them."""
        model = _Tiny(misaligned=True)
        weight_before = np.array(model.weight.detach().tolist())
        scale_before = np.array(model.scale.detach().tolist())

        align_local_model_weights(model, label="Test")

        np.testing.assert_array_equal(np.array(model.weight.detach().tolist()), weight_before)
        np.testing.assert_array_equal(np.array(model.scale.detach().tolist()), scale_before)

    def test_forward_matches_reference_after_alignment(self):
        """The regression: a misaligned weight makes the CPU matmul return garbage.

        The reference is computed in float64 from the same values, which is immune to
        the misaligned-operand path. Asserting only the post-alignment result keeps
        this meaningful on platforms whose kernels tolerate the misalignment.
        """
        model = _Tiny(misaligned=True)
        x = torch.randn(16, 64)
        weight = torch.tensor(model.weight.detach().tolist(), dtype=torch.float64)
        scale = torch.tensor(model.scale.detach().tolist(), dtype=torch.float64)
        expected = ((x.double() @ weight.T) * scale).float()

        align_local_model_weights(model, label="Test")

        with torch.no_grad():
            actual = model(x)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_raises_when_alignment_cannot_be_fixed(self):
        """A model that would compute garbage must not reach production traffic."""
        model = _Tiny(misaligned=True)
        with patch.object(local_device, "_misaligned", return_value=True):
            with pytest.raises(RuntimeError, match="remain misaligned after copying"):
                align_local_model_weights(model, label="Reranker[some-model]")

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="expected a torch module"):
            align_local_model_weights(object(), label="Test")


class TestAssertFiniteLocalOutput:
    def test_accepts_finite_values(self):
        assert_finite_local_output([1.0, -11.4, 0.0], label="Test")
        assert_finite_local_output(np.array([[0.1, 0.2], [0.3, 0.4]]), label="Test")

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_values(self, bad):
        with pytest.raises(RuntimeError, match="smoke test produced"):
            assert_finite_local_output([1.0, bad, 2.0], label="Reranker[some-model]")

    def test_message_names_the_model_and_the_count(self):
        with pytest.raises(RuntimeError) as exc:
            assert_finite_local_output([float("nan"), float("nan"), 1.0], label="Embeddings[bge]")
        assert "Embeddings[bge]" in str(exc.value)
        assert "2 non-finite value(s) of 3" in str(exc.value)


class TestLoadersGuardTheirWeights:
    """The guard has to run at the call sites, not just exist."""

    async def test_reranker_aligns_and_smoke_tests(self):
        from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder

        fake = MagicMock()
        fake.model = _Tiny(misaligned=True)
        fake.predict.return_value = [2.25]
        encoder = LocalSTCrossEncoder(model_name="test-model", force_cpu=True)

        with patch("sentence_transformers.CrossEncoder", return_value=fake):
            await encoder.initialize()

        assert all(t.data_ptr() % t.element_size() == 0 for t in _all_tensors(fake.model))
        fake.predict.assert_called_once()

    async def test_reranker_refuses_a_nan_producing_model(self):
        from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder

        fake = MagicMock()
        fake.model = _Tiny(misaligned=False)
        fake.predict.return_value = [float("nan")]
        encoder = LocalSTCrossEncoder(model_name="test-model", force_cpu=True)

        with patch("sentence_transformers.CrossEncoder", return_value=fake):
            with pytest.raises(RuntimeError, match="smoke test produced"):
                await encoder.initialize()

    async def test_embeddings_align_and_smoke_test(self):
        from hindsight_api.engine.embeddings import LocalSTEmbeddings

        fake = _Tiny(misaligned=True)
        fake.get_sentence_embedding_dimension = lambda: 64  # type: ignore[method-assign]
        fake.encode = lambda texts: np.zeros((len(texts), 64), dtype=np.float32)  # type: ignore[method-assign]
        embeddings = LocalSTEmbeddings(model_name="test-model", force_cpu=True)

        with patch("sentence_transformers.SentenceTransformer", return_value=fake):
            await embeddings.initialize()

        assert all(t.data_ptr() % t.element_size() == 0 for t in _all_tensors(fake))
        assert embeddings.dimension == 64

    def test_query_analyzer_aligns_its_weights(self):
        """The T5 analyzer loads the same way; `.to("cpu")` does not launder alignment."""
        from hindsight_api.engine.query_analyzer import TransformerQueryAnalyzer

        fake = _Tiny(misaligned=True)
        analyzer = TransformerQueryAnalyzer(model_name="test-model")

        with (
            patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=fake),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        ):
            analyzer.load()

        assert all(t.data_ptr() % t.element_size() == 0 for t in _all_tensors(fake))

    async def test_embeddings_refuse_a_nan_producing_model(self):
        from hindsight_api.engine.embeddings import LocalSTEmbeddings

        fake = _Tiny(misaligned=False)
        fake.get_sentence_embedding_dimension = lambda: 64  # type: ignore[method-assign]
        fake.encode = lambda texts: np.full((len(texts), 64), np.nan, dtype=np.float32)  # type: ignore[method-assign]
        embeddings = LocalSTEmbeddings(model_name="test-model", force_cpu=True)

        with patch("sentence_transformers.SentenceTransformer", return_value=fake):
            with pytest.raises(RuntimeError, match="smoke test produced"):
                await embeddings.initialize()


class TestEveryLocalTorchLoaderIsGuarded:
    """Family guard: the defect this fixes is a loader that *doesn't* call the helper.

    A fourth local torch model added later would silently reintroduce the bug, and no
    test for it would exist by construction — so assert over the whole family rather
    than over the three loaders that happen to exist today. ONNX and MLX providers are
    deliberately out of scope: they don't map safetensors into torch tensors.
    """

    # Constructing a torch model from a HuggingFace checkpoint — the operation that
    # can hand back zero-copy, potentially misaligned weights.
    _LOADER_PATTERNS = (
        r"=\s*CrossEncoder\(",
        r"=\s*SentenceTransformer\(",
        r"AutoModel\w*\.from_pretrained\(",
    )

    def test_every_torch_model_loader_aligns_its_weights(self):
        engine_dir = Path(local_device.__file__).parent
        loaders = {
            path.name: source
            for path in sorted(engine_dir.rglob("*.py"))
            if any(re.search(p, source := path.read_text()) for p in self._LOADER_PATTERNS)
        }

        assert loaders, "found no local torch model loaders — has the detection pattern gone stale?"

        unguarded = [name for name, source in loaders.items() if "align_local_model_weights" not in source]
        assert not unguarded, (
            f"{unguarded} load a torch model without calling align_local_model_weights. "
            f"Zero-copy safetensors weights can land misaligned, which silently corrupts "
            f"the CPU matmul (see engine/local_device.py)."
        )
