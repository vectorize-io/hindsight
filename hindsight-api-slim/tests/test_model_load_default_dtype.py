"""
Startup must leave torch's global default dtype at float32 regardless of how the
concurrent local model loads interleave.

Covers issue #2162: transformers' dtype context manager (entered by
SentenceTransformer / CrossEncoder / from_pretrained) does a NON-thread-safe
save/restore of the *process-global* default dtype. When an fp16 embedding model
and an fp32 reranker/query-analyzer load in parallel at startup, an unlucky
interleave leaves the global default stuck at float16 — every later encode() then
emits NaN vectors that pgvector rejects ("NaN not allowed in vector") on MPS, or
raises "c10::Half != float" on CPU, non-deterministically across restarts.

MemoryEngine.initialize() loads the models in parallel (for speed) and then, once
the gather has joined every load thread, normalizes the global default dtype back
to float32 — the inference state a healthy boot already converges to. This test
simulates the poisoning by having a model load flip the default to float16, then
asserts initialize() leaves it at float32.
"""

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend


class _StopInit(Exception):
    """Sentinel to abort initialize() right after the model-load gather."""


class _PoisoningEmbeddings:
    """Local embedding stub that mimics an fp16 load poisoning the global dtype."""

    provider_name = "local"

    async def initialize(self) -> None:
        import torch

        # Reproduce the symptom of transformers' racy dtype restore: the global
        # default is left at float16 after the (parallel) load.
        torch.set_default_dtype(torch.float16)


class _NoopCrossEncoder:
    provider_name = "local"

    async def initialize(self) -> None:
        return None


class _NoopQueryAnalyzer:
    def load(self) -> None:
        return None


@pytest.mark.asyncio
async def test_global_default_dtype_restored_to_float32_after_init():
    """A load that leaves the torch default at float16 is normalized back to float32."""
    import torch

    original = torch.get_default_dtype()
    try:
        engine = MemoryEngine(
            # Non-pg0 URL so start_pg0() is a no-op and __init__ never connects.
            db_url="postgresql://u:p@localhost:5999/db",
            memory_llm_provider="none",
            memory_llm_api_key=None,
            memory_llm_model="none",
            embeddings=_PoisoningEmbeddings(),
            cross_encoder=_NoopCrossEncoder(),
            query_analyzer=_NoopQueryAnalyzer(),
            run_migrations=False,
            skip_llm_verification=True,
            lazy_reranker=False,  # load the cross-encoder eagerly, in the gather
            task_backend=SyncTaskBackend(),
        )

        # Abort right after the post-gather dtype restore, before any real DB work.
        async def _stop(*args, **kwargs):
            raise _StopInit

        engine._backend.initialize = _stop  # type: ignore[method-assign]

        with pytest.raises(_StopInit):
            await engine.initialize()

        # The embedding load poisoned the default to float16; initialize() must
        # have normalized it back so later encode() can't emit NaN vectors.
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(original)
