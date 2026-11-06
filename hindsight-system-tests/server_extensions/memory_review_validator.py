"""A test-only operation validator that reviews memories before they are written.

Drops any extracted memory or observation mentioning ``password``, and tags what it keeps:
``reviewed`` on memories, ``reviewed:observation`` on observations. Loaded by the server under
test, so it lives outside the ``hindsight_system_tests`` package (see tag_scope_validator.py).
"""

from hindsight_api.extensions import (
    ConsolidationActionsContext,
    MemoryReview,
    OperationValidatorExtension,
    RetainFactsContext,
    ValidationResult,
)

SECRET = "password"


def _review(text: str, tags: list[str], tag: str) -> MemoryReview:
    if SECRET in text.lower():
        return MemoryReview(keep=False)
    return MemoryReview(tags=[*tags, tag])


class MemoryReviewer(OperationValidatorExtension):
    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def filter_retain_facts(self, ctx: RetainFactsContext) -> list[MemoryReview]:
        return [_review(f.text, f.tags, "reviewed") for f in ctx.facts]

    async def filter_consolidation_actions(self, ctx: ConsolidationActionsContext) -> list[MemoryReview]:
        return [_review(a.text, a.tags, "reviewed:observation") for a in ctx.actions]
