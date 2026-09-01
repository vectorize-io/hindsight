"""Device selection and post-inference memory release for local (in-process)
SentenceTransformer / CrossEncoder models.

Two concerns live here, both about keeping a local API instance's memory flat:

**1. Device selection — MPS is opt-in.**
On Apple Silicon the PyTorch **MPS** (Metal) backend caches a distinct compiled
kernel graph *and* allocator pool per unique input tensor shape, and never
releases them. Under the variable-length, high-volume recall/rerank/embed traffic
the engine generates (documents and candidate sets of every size), that per-shape
cache grows without bound. A local instance was observed idling at ~20 GB — ~9.4 GB
of Metal graphics memory plus ~8 GB of native heap, essentially all of it stale
per-shape MPS cache. CPU inference has no per-shape cache: the same workload holds
flat at a few hundred MB, with negligible latency cost for the small default
models (and MPS actually *slows down* over time as it recompiles graphs for new
shapes). So MPS is excluded from auto-detection and must be opted into explicitly;
CUDA and Intel XPU still auto-select.

This is a confirmed, still-open PyTorch bug in the MPSGraph compilation cache
(keyed on tensor shape, no eviction path). We are tracking it upstream:
  - https://github.com/pytorch/pytorch/issues/181213
    ([MPS] unbounded RSS growth with varying-shape inference — our exact case)
  - https://github.com/pytorch/pytorch/issues/164299 (graphCache identified as
    the primary leak culprit)
  - https://github.com/pytorch/pytorch/issues/182815 (proposes, but has not yet
    shipped, a torch.mps.invalidate_graph_cache() API / PYTORCH_MPS_DISABLE_GRAPH_CACHE
    env var that would let us keep MPS)
No released mitigation exists today: empty_cache(), synchronize(),
PYTORCH_MPS_HIGH_WATERMARK_RATIO, and autorelease pools were all confirmed
ineffective upstream. Revisit MPS-as-default once one of those knobs lands.

**2. Memory release after each batch.**
Local CPU inference allocates large transient numpy/tensor buffers per call. The
allocator keeps those freed pages as a high-water mark, so RSS grows monotonically
across many calls (issue #1717). We return them to the OS after each batch —
``malloc_trim`` on glibc/Linux, ``malloc_zone_pressure_relief`` on macOS (the
original #1717 fix covered only Linux). When the model ran on a GPU we also empty
that backend's allocator pool via ``torch.<backend>.empty_cache()``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import sys

logger = logging.getLogger(__name__)


def select_local_device(force_cpu: bool, allow_mps: bool) -> str | None:
    """Choose the device for a local SentenceTransformer / CrossEncoder.

    Returns a value suitable to pass as the model's ``device`` argument:

    - ``"cpu"``  — forced CPU, or the only accelerator is MPS and it is not allowed.
    - ``None``   — let sentence-transformers auto-detect (picks CUDA / XPU,
                   handling multi-GPU correctly).
    - ``"mps"``  — Apple Silicon GPU, only when ``allow_mps`` is set.

    MPS is never auto-selected because its per-shape cache leaks unbounded memory
    under the engine's variable-length workload (see the module docstring). Set the
    matching ``*_ALLOW_MPS`` config flag to opt back in.
    """
    if force_cpu:
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            return None  # auto-detect CUDA
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return None  # auto-detect Intel XPU

        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available:
            if allow_mps:
                return "mps"
            logger.info(
                "Local model: MPS (Apple Silicon GPU) is available but disabled by "
                "default because its per-shape cache leaks memory under variable-length "
                "workloads; running on CPU. Set the *_ALLOW_MPS flag to opt in."
            )
            return "cpu"
        return "cpu"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Local device detection failed, falling back to CPU: %s", e)
        return "cpu"


def resolve_model_device_type(model: object) -> str:
    """Best-effort device *type* ("cpu" / "cuda" / "mps" / "xpu") of a loaded model.

    Used to decide which GPU allocator pool to empty after inference. Falls back to
    ``"cpu"`` (the safe no-op choice for release) if the device can't be read.
    """
    device = getattr(model, "device", None)
    if device is None:
        inner = getattr(model, "model", None)  # CrossEncoder wraps the HF model
        device = getattr(inner, "device", None)
    try:
        return device.type if device is not None else "cpu"
    except Exception:  # pragma: no cover - defensive
        return "cpu"


def _resolve_heap_trim():
    """Return a callable that asks the C allocator to release freed pages to the OS.

    glibc (Linux) exposes ``malloc_trim``; macOS exposes
    ``malloc_zone_pressure_relief``. Resolved once at import; returns a no-op on
    platforms where neither is available (musl, Windows).
    """
    if sys.platform == "linux":
        libc_path = ctypes.util.find_library("c")
        if libc_path is None:
            return lambda: None
        try:
            libc = ctypes.CDLL(libc_path)
            trim = libc.malloc_trim
        except (OSError, AttributeError):
            # Not glibc (musl has no malloc_trim) or libc lookup failed.
            return lambda: None
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return lambda: trim(0)

    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
            default_zone = libc.malloc_default_zone
            default_zone.restype = ctypes.c_void_p
            relief = libc.malloc_zone_pressure_relief
            relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            relief.restype = ctypes.c_size_t
        except (OSError, AttributeError):
            return lambda: None
        # pressure_relief(zone, goal=0) reclaims as much as possible.
        return lambda: relief(default_zone(), 0)

    return lambda: None


_heap_trim = _resolve_heap_trim()


def _empty_gpu_cache(device_type: str | None) -> None:
    """Empty the allocator pool of the GPU backend the model ran on, if any."""
    if not device_type or device_type == "cpu":
        return
    try:
        import torch

        backend = getattr(torch, device_type, None)  # torch.cuda / torch.mps / torch.xpu
        if backend is not None and hasattr(backend, "empty_cache"):
            backend.empty_cache()
    except Exception:  # pragma: no cover - defensive
        pass


def release_local_inference_memory(device_type: str | None = None) -> None:
    """Release transient heap (and GPU allocator) memory after a local inference batch.

    Returns freed native pages to the OS, and empties the GPU allocator pool when
    the model ran on a GPU. Python's normal reference counting already releases
    the short-lived CPU inference buffers; a full cyclic-GC scan on every batch is
    needlessly expensive on that hot path, so CPU callers leave cyclic GC to its
    normal threshold-based schedule. GPU callers retain the existing full cleanup
    because allocator release is part of the opt-in accelerator memory policy.
    """
    if device_type != "cpu":
        gc.collect()
    _heap_trim()
    _empty_gpu_cache(device_type)
