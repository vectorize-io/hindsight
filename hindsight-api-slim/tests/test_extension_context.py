"""The engine gives every extension it owns the one process-wide ExtensionContext.

The tenant extension and the operation validator are constructed BEFORE the engine and
handed to ``MemoryEngine.__init__``, so ``load_extension()`` never set a context on them.
Without the engine doing it, a validator hook reaching ``self.context`` (e.g.
``get_memory_engine()`` for the data-plane pool) raises "Extension context not set" --
silently, for hooks that treat context access as best-effort.
"""

import os
from unittest.mock import MagicMock, patch

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.extensions.operation_validator import (
    OperationValidatorExtension,
    ValidationResult,
)


class _Validator(OperationValidatorExtension):
    """Accepts everything -- just enough to instantiate."""

    async def validate_retain(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:
        return ValidationResult.accept()


def _build_engine(**kwargs) -> MemoryEngine:
    """Construct an engine without network, GPU or DB: __init__ never starts the pool."""
    mock_embeddings = MagicMock()
    mock_embeddings.dimension = 384

    with patch.dict(
        os.environ,
        {
            "HINDSIGHT_API_LLM_PROVIDER": "none",
            "HINDSIGHT_API_LLM_MODEL": "none",
            "HINDSIGHT_API_LLM_API_KEY": "test-key",
        },
        clear=False,
    ):
        clear_config_cache()
        engine = MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=mock_embeddings, **kwargs)

    # Drop the config cache this repopulated from the patched env, so "none"/chunks does
    # not bleed into other tests on this xdist worker.
    clear_config_cache()
    return engine


def test_validator_gets_the_engine_context() -> None:
    validator = _Validator({})
    engine = _build_engine(operation_validator=validator)

    assert validator.context is engine._ext_ctx
    assert validator.context.get_memory_engine() is engine


def test_tenant_extension_gets_the_same_context() -> None:
    engine = _build_engine()

    assert engine._tenant_extension.context is engine._ext_ctx
    assert engine._memory_defense.context is engine._ext_ctx


def test_duck_typed_validator_without_set_context_still_builds() -> None:
    class _Bare:
        async def validate_retain(self, ctx):
            return None

    engine = _build_engine(operation_validator=_Bare())  # must not raise
    assert engine._operation_validator is not None
