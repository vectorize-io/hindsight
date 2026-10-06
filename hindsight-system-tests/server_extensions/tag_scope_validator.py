"""A test-only operation validator, loaded by the server under test (not by the tests).

It confines each caller to reading its own ``user:<api key>`` tag plus the shared
``kind:rule`` scope, and to writing only ``user:<api key>`` — except ``kate``, who sets the
team's rules and may also write ``kind:rule``. A request without an API key is unrestricted
(the bank's administrator). A ``bank:<id>`` key instead skips the tag rules and may write only
to bank ``<id>`` — the per-key bank isolation a deployment builds on ``validate_bank_write``. It lives
outside the ``hindsight_system_tests`` package on purpose: the server imports it, and that
package pulls in the test harness, which the server's environment does not have.
"""

from hindsight_api.engine.search.tags import TagGroupLeaf
from hindsight_api.extensions import BankWriteContext, OperationValidatorExtension, TagScopeContext, ValidationResult

BANK_KEY_PREFIX = "bank:"


def _tag_user(api_key: str | None) -> str | None:
    """The key a tag scope applies to; bank-scoped keys and keyless admin calls get none."""
    if not api_key or api_key.startswith(BANK_KEY_PREFIX):
        return None
    return api_key


class UserTagScope(OperationValidatorExtension):
    async def validate_retain(self, ctx):
        return ValidationResult.accept()

    async def validate_recall(self, ctx):
        return ValidationResult.accept()

    async def validate_reflect(self, ctx):
        return ValidationResult.accept()

    async def validate_bank_write(self, ctx: BankWriteContext):
        key = ctx.request_context.api_key
        if key and key.startswith(BANK_KEY_PREFIX) and key != f"{BANK_KEY_PREFIX}{ctx.bank_id}":
            return ValidationResult.reject(f"key may not write to bank {ctx.bank_id}", status_code=403)
        return ValidationResult.accept()

    async def resolve_tag_scope(self, ctx: TagScopeContext):
        user = _tag_user(ctx.request_context.api_key)
        if not user:
            return None
        return [TagGroupLeaf(tags=[f"user:{user}", "kind:rule"], match="any_strict")]

    async def resolve_write_tag_scope(self, ctx: TagScopeContext):
        user = _tag_user(ctx.request_context.api_key)
        if not user:
            return None
        return [f"user:{user}", "kind:rule"] if user == "kate" else [f"user:{user}"]
