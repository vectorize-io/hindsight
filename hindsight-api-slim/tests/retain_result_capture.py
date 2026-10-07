"""Shared test helper: capture the ``RetainResult`` the engine reports to extensions.

``RetainResult`` fields like ``processed_content_tokens`` are only observable through the
post-retain hook, so several suites need the same do-nothing validator to read them. It lived
copy-pasted in three test modules before this became one import.
"""

from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RetainContext,
    RetainResult,
    ValidationResult,
)


class RetainResultCapture(OperationValidatorExtension):
    """Records each RetainResult the engine reports via ``on_retain_complete``.

    The pre-operation validators are required by the abstract base; they always accept so they
    don't interfere with the operations under test.
    """

    def __init__(self) -> None:
        self.results: list[RetainResult] = []

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()

    async def on_retain_complete(self, result: RetainResult) -> None:
        self.results.append(result)
