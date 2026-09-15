"""An operation validator must receive its ExtensionContext.

Unlike the tenant, HTTP, and memory-defense extensions, an operation validator is
passed into ``MemoryEngine.__init__`` — constructed before the engine exists — and is
not routed through ``load_extension()``, so it used to never receive an
``ExtensionContext``. Any validator hook that reaches an engine API through
``self.context`` (e.g. ``context.get_memory_engine()``) then raised
"Extension context not set".

These build a MemoryEngine without connecting (a non-pg0 URL makes ``__init__`` skip
pg0 startup, and ``set_context`` runs in ``__init__`` before any pool work), so no DB
or model load is required.
"""

from hindsight_api import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.extensions.operation_validator import (
    OperationValidatorExtension,
    ValidationResult,
)


class _MinimalValidator(OperationValidatorExtension):
    """Concrete validator that accepts everything — just enough to instantiate."""

    async def validate_retain(self, ctx) -> ValidationResult:  # noqa: D102
        return ValidationResult.accept()

    async def validate_recall(self, ctx) -> ValidationResult:  # noqa: D102
        return ValidationResult.accept()

    async def validate_reflect(self, ctx) -> ValidationResult:  # noqa: D102
        return ValidationResult.accept()


def _build_engine(validator):
    return MemoryEngine(
        # Non-pg0 URL so start_pg0() is a no-op and __init__ never connects.
        db_url="postgresql://u:p@localhost:5999/db",
        memory_llm_provider="none",
        memory_llm_api_key=None,
        memory_llm_model="none",
        run_migrations=False,
        skip_llm_verification=True,
        task_backend=SyncTaskBackend(),
        operation_validator=validator,
    )


def test_operation_validator_receives_context_with_memory_engine():
    """The validator's context is set and resolves the engine.

    Fails without the fix: ``set_context`` is never called on the validator, so
    ``validator.context`` raises "Extension context not set".
    """
    validator = _MinimalValidator({})
    engine = _build_engine(validator)

    assert validator.context is not None
    assert validator.context.get_memory_engine() is engine


def test_engine_builds_with_validator_lacking_set_context():
    """A duck-typed validator without ``set_context`` must not break construction."""

    class _Bare:
        async def validate_retain(self, ctx):
            return None

    engine = _build_engine(_Bare())  # must not raise
    assert engine is not None
