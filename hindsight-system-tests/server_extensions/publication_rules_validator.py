"""A test-only operation validator holding mental-model pages to a publication rule (#5970).

A page may never say ``UNPUBLISHABLE``. A page whose name starts with ``Redraft`` gets
another draft when it does, with the reason in its prompt; any other page is refused
outright and keeps its last accepted content. Loaded by the server under test, not by the
tests, which is why it lives outside the ``hindsight_system_tests`` package.
"""

from hindsight_api.extensions import MentalModelCandidateContext, OperationValidatorExtension, ValidationResult

UNPUBLISHABLE = "UNPUBLISHABLE"
REASON = "the page must not contain the word UNPUBLISHABLE"
REDRAFT_PREFIX = "Redraft"


class PublicationRules(OperationValidatorExtension):
    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def validate_mental_model_candidate(self, ctx: MentalModelCandidateContext):
        if UNPUBLISHABLE in ctx.candidate_content:
            return ValidationResult.reject(REASON, retry=ctx.name.startswith(REDRAFT_PREFIX))
        return ValidationResult.accept()
