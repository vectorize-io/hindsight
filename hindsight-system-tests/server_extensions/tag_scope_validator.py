"""A test-only operation validator, loaded by the server under test (not by the tests).

It confines each caller to reading its own ``user:<api key>`` tag plus the shared
``kind:rule`` scope, and to writing only ``user:<api key>`` — except ``kate``, who sets the
team's rules and may also write ``kind:rule``. A request without an API key is unrestricted
(the bank's administrator). It lives
outside the ``hindsight_system_tests`` package on purpose: the server imports it, and that
package pulls in the test harness, which the server's environment does not have.
"""

from hindsight_api.engine.search.tags import TagGroupLeaf
from hindsight_api.extensions import OperationValidatorExtension, TagScopeContext, ValidationResult


class UserTagScope(OperationValidatorExtension):
    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def resolve_tag_scope(self, ctx: TagScopeContext):
        user = ctx.request_context.api_key
        if not user:
            return None
        return [TagGroupLeaf(tags=[f"user:{user}", "kind:rule"], match="any_strict")]

    async def resolve_write_tag_scope(self, ctx: TagScopeContext):
        user = ctx.request_context.api_key
        if not user:
            return None
        return [f"user:{user}", "kind:rule"] if user == "kate" else [f"user:{user}"]
