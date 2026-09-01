"""
Unit tests for engine/local_device.py — device selection and post-inference
memory release for local SentenceTransformer / CrossEncoder models.

torch is faked via sys.modules so these run fast and deterministically without a
real GPU: the functions under test do ``import torch`` lazily, so the fake is
picked up.
"""

import sys
import types
from unittest.mock import patch

from hindsight_api.engine import local_device
from hindsight_api.engine.local_device import (
    _empty_gpu_cache,
    _resolve_heap_trim,
    release_local_inference_memory,
    resolve_model_device_type,
    select_local_device,
)


def _fake_torch(*, cuda=False, mps=False, xpu=False, has_xpu=None, empty_cache_log=None):
    """Build a stand-in ``torch`` module exposing just what local_device touches."""
    if has_xpu is None:
        has_xpu = xpu

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        empty_cache=lambda: (empty_cache_log.append("cuda") if empty_cache_log is not None else None),
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    torch.mps = types.SimpleNamespace(
        empty_cache=lambda: (empty_cache_log.append("mps") if empty_cache_log is not None else None),
    )
    if has_xpu:
        torch.xpu = types.SimpleNamespace(
            is_available=lambda: xpu,
            empty_cache=lambda: (empty_cache_log.append("xpu") if empty_cache_log is not None else None),
        )
    return torch


class TestSelectLocalDevice:
    def test_force_cpu_short_circuits(self):
        # force_cpu wins even if a GPU is present — no torch import needed.
        assert select_local_device(force_cpu=True, allow_mps=True) == "cpu"

    def test_cuda_auto_selects(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda=True)}):
            assert select_local_device(force_cpu=False, allow_mps=False) is None

    def test_xpu_auto_selects(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(xpu=True, has_xpu=True)}):
            assert select_local_device(force_cpu=False, allow_mps=False) is None

    def test_mps_disabled_by_default_falls_back_to_cpu(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=True)}):
            assert select_local_device(force_cpu=False, allow_mps=False) == "cpu"

    def test_mps_used_when_allowed(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(mps=True)}):
            assert select_local_device(force_cpu=False, allow_mps=True) == "mps"

    def test_cuda_preferred_over_mps_even_when_mps_allowed(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda=True, mps=True)}):
            assert select_local_device(force_cpu=False, allow_mps=True) is None

    def test_no_accelerator_is_cpu(self):
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            assert select_local_device(force_cpu=False, allow_mps=True) == "cpu"

    def test_torch_failure_falls_back_to_cpu(self):
        broken = types.ModuleType("torch")

        # Any attribute access raises — detection must swallow it and pick CPU.
        class Boom(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("no torch")

        with patch.dict(sys.modules, {"torch": Boom("torch")}):
            assert select_local_device(force_cpu=False, allow_mps=False) == "cpu"


class TestResolveModelDeviceType:
    def test_reads_model_device(self):
        model = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"))
        assert resolve_model_device_type(model) == "cuda"

    def test_falls_back_to_inner_model_device(self):
        # CrossEncoder wraps the HF model as ``.model``.
        inner = types.SimpleNamespace(device=types.SimpleNamespace(type="mps"))
        model = types.SimpleNamespace(model=inner)
        model.device = None
        assert resolve_model_device_type(model) == "mps"

    def test_defaults_to_cpu_when_unknown(self):
        assert resolve_model_device_type(object()) == "cpu"


class TestHeapTrim:
    def test_returns_callable(self):
        assert callable(_resolve_heap_trim())

    def test_callable_does_not_raise(self):
        # glibc returns an int, macOS a size_t, elsewhere None — never raises.
        result = _resolve_heap_trim()()
        assert result is None or isinstance(result, int)

    def test_module_level_trim_resolved(self):
        assert callable(local_device._heap_trim)

    def test_linux_uses_malloc_trim(self):
        with patch.object(sys, "platform", "linux"):
            assert callable(_resolve_heap_trim())

    def test_darwin_resolves_a_callable(self):
        # macOS gets malloc_zone_pressure_relief (the #1717 fix, extended to mac).
        with patch.object(sys, "platform", "darwin"):
            trim = _resolve_heap_trim()
        assert callable(trim)

    def test_unknown_platform_is_noop(self):
        with patch.object(sys, "platform", "sunos5"):
            trim = _resolve_heap_trim()
        assert trim() is None


class TestEmptyGpuCache:
    def test_cpu_is_noop(self):
        log = []
        with patch.dict(sys.modules, {"torch": _fake_torch(empty_cache_log=log)}):
            _empty_gpu_cache("cpu")
            _empty_gpu_cache(None)
        assert log == []

    def test_empties_matching_backend(self):
        log = []
        with patch.dict(sys.modules, {"torch": _fake_torch(empty_cache_log=log)}):
            _empty_gpu_cache("cuda")
            _empty_gpu_cache("mps")
        assert log == ["cuda", "mps"]

    def test_unknown_backend_is_safe(self):
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            _empty_gpu_cache("rocm")  # torch has no .rocm — must not raise


class TestReleaseLocalInferenceMemory:
    def test_release_runs_all_steps_for_gpu(self):
        log = []
        calls = []
        with (
            patch.dict(sys.modules, {"torch": _fake_torch(empty_cache_log=log)}),
            patch.object(local_device.gc, "collect", lambda: calls.append("gc")),
            patch.object(local_device, "_heap_trim", lambda: calls.append("trim")),
        ):
            release_local_inference_memory("cuda")
        assert calls == ["gc", "trim"]
        assert log == ["cuda"]

    def test_release_cpu_skips_gc_and_empty_cache_but_trims_heap(self):
        calls = []
        log = []
        with (
            patch.dict(sys.modules, {"torch": _fake_torch(empty_cache_log=log)}),
            patch.object(local_device.gc, "collect", lambda: calls.append("gc")),
            patch.object(local_device, "_heap_trim", lambda: calls.append("trim")),
        ):
            release_local_inference_memory("cpu")
        assert calls == ["trim"]
        assert log == []
