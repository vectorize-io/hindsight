"""Bank config writes validate and persist as one serialized unit (#3037).

Two updates that are each valid against the state they read must not be able to
commit a combination that neither of them validated. The interleaving is forced
with an ``asyncio.Barrier`` around the persistence step — both requests finish
validation before either writes — rather than with sleeps.

The helpers are shared with the Oracle suite (``test_oracle_integration.py``),
which runs the recall-budget scenarios against Oracle 23ai. The retain-chunking
scenario stays PG-only — see the comment on ``TestBankConfigAtomicity`` there.
"""

import asyncio
import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.config_resolver import ConfigResolver, apply_strategy
from hindsight_api.extensions.tenant import TenantExtension


def bank_id(prefix: str) -> str:
    return f"test-cfg-atomic-{prefix}-{uuid.uuid4().hex[:8]}"


async def run_interleaved_config_updates(
    memory: MemoryEngine,
    bank: str,
    request_context: RequestContext,
    first: dict[str, object],
    second: dict[str, object],
) -> list[BaseException]:
    """Validate both updates before either persists, then return what each raised.

    The barrier releases both requests into ``_persist_bank_config`` at the same
    moment, so they can only be ordered by the bank row lock the write takes.
    """
    resolver = memory._config_resolver
    original_persist = resolver._persist_bank_config
    barrier = asyncio.Barrier(2)

    async def gated_persist(persisted_bank_id, validated):
        if persisted_bank_id == bank:
            # A timeout rather than a bare wait: if one request is rejected
            # before it reaches the write, the other must fail the test instead
            # of hanging at the barrier forever.
            await asyncio.wait_for(barrier.wait(), timeout=30)
        return await original_persist(persisted_bank_id, validated)

    resolver._persist_bank_config = gated_persist
    try:
        results = await asyncio.gather(
            memory.update_bank_config(bank, first, request_context=request_context),
            memory.update_bank_config(bank, second, request_context=request_context),
            return_exceptions=True,
        )
    finally:
        resolver._persist_bank_config = original_persist
    return [r for r in results if isinstance(r, BaseException)]


async def assert_recall_budget_race_is_serialized(memory: MemoryEngine, request_context: RequestContext) -> None:
    """Concurrent one-sided recall-budget updates cannot cross min over max."""
    bank = bank_id("budget")
    try:
        await memory.update_bank_config(
            bank,
            {"recall_budget_min": 100, "recall_budget_max": 1000},
            request_context=request_context,
        )

        # 800 <= 1000 and 100 <= 500, but min=800 with max=500 is not a
        # configuration either request validated.
        errors = await run_interleaved_config_updates(
            memory,
            bank,
            request_context,
            {"recall_budget_min": 800},
            {"recall_budget_max": 500},
        )

        assert len(errors) == 1, f"expected exactly one rejected update, got {errors}"
        assert "recall_budget_min" in str(errors[0])
        assert isinstance(errors[0], ValueError)

        overrides = await memory._config_resolver._load_bank_config(bank)
        assert overrides["recall_budget_min"] <= overrides["recall_budget_max"]
    finally:
        await memory.delete_bank(bank, request_context=request_context)


async def assert_retain_chunking_race_is_serialized(memory: MemoryEngine, request_context: RequestContext) -> None:
    """Concurrent chunk-size and strategy updates cannot wedge the retain config.

    PostgreSQL only: the wedge depends on a ``retain_strategies`` write replacing
    the whole map, which is what ``config || $1::jsonb`` does. Oracle rewrites
    that statement to JSON_MERGEPATCH and merges into the nested object instead.
    """
    bank = bank_id("chunking")
    resolved = await memory._config_resolver.resolve_full_config(bank, request_context)
    if resolved.llm_provider == "none":
        pytest.skip("the retain completion-token budget is not enforced without an LLM provider")
    oversized_chunk = resolved.retain_max_completion_tokens + 1000

    try:
        await memory.update_bank_config(
            bank,
            {"retain_chunk_size": 3000, "retain_strategies": {"s": {"retain_chunk_size": 1000}}},
            request_context=request_context,
        )

        # The oversized bank-level chunk size is only valid while strategy "s"
        # pins its own; dropping that pin is only valid while the bank-level
        # chunk size is small. Together they wedge retain under strategy "s".
        errors = await run_interleaved_config_updates(
            memory,
            bank,
            request_context,
            {"retain_chunk_size": oversized_chunk},
            {"retain_strategies": {"s": {"retain_extraction_mode": "detailed"}}},
        )

        assert len(errors) == 1, f"expected exactly one rejected update, got {errors}"
        assert "retain_max_completion_tokens" in str(errors[0])
        assert isinstance(errors[0], ValueError)

        # The committed configuration is one retain can still run: this is the
        # call retain makes to splice a strategy onto the resolved config, and
        # it is what raised once the two updates had both landed.
        apply_strategy(await memory._config_resolver.resolve_full_config(bank, request_context), "s")
    finally:
        await memory.delete_bank(bank, request_context=request_context)


async def assert_one_sided_budget_update_sees_stored_state(
    memory: MemoryEngine, request_context: RequestContext
) -> None:
    """A single bound is validated against the bound already stored, not ignored."""
    bank = bank_id("stored")
    try:
        await memory.update_bank_config(bank, {"recall_budget_max": 100}, request_context=request_context)

        with pytest.raises(ValueError, match="recall_budget_min"):
            await memory.update_bank_config(bank, {"recall_budget_min": 500}, request_context=request_context)

        overrides = await memory._config_resolver._load_bank_config(bank)
        assert "recall_budget_min" not in overrides
    finally:
        await memory.delete_bank(bank, request_context=request_context)


@pytest.mark.asyncio
async def test_concurrent_updates_cannot_persist_invalid_recall_budget_bounds(memory, request_context):
    await assert_recall_budget_race_is_serialized(memory, request_context)


@pytest.mark.asyncio
async def test_concurrent_updates_cannot_persist_invalid_retain_chunking(memory, request_context):
    await assert_retain_chunking_race_is_serialized(memory, request_context)


@pytest.mark.asyncio
async def test_one_sided_recall_budget_update_is_validated_against_stored_state(memory, request_context):
    await assert_one_sided_budget_update_sees_stored_state(memory, request_context)


@pytest.mark.asyncio
async def test_updates_untouched_by_cross_field_constraints_still_merge(memory, request_context):
    """Two concurrent writes to unconstrained fields both survive.

    Serializing on the bank row must not turn independent field writes into a
    last-write-wins race: neither update participates in a cross-field
    constraint, so both belong in the committed config.
    """
    bank = bank_id("merge")
    try:
        await memory.update_bank_config(bank, {"enable_reranking": True}, request_context=request_context)

        errors = await run_interleaved_config_updates(
            memory,
            bank,
            request_context,
            {"enable_reranking": False},
            {"enable_observations": False},
        )

        assert errors == []
        overrides = await memory._config_resolver._load_bank_config(bank)
        assert overrides["enable_reranking"] is False
        assert overrides["enable_observations"] is False
    finally:
        await memory.delete_bank(bank, request_context=request_context)


class CountingTenantExtension(TenantExtension):
    """Tenant extension that records how often each config hook is invoked."""

    def __init__(self) -> None:
        self.tenant_config_calls = 0
        self.allowed_config_field_calls = 0

    async def authenticate(self, context):
        from hindsight_api.extensions.tenant import TenantContext

        return TenantContext(schema_name="public")

    async def list_tenants(self):
        from hindsight_api.extensions.tenant import Tenant

        return [Tenant(schema="public")]

    async def get_tenant_config(self, context):
        self.tenant_config_calls += 1
        return {}

    async def get_allowed_config_fields(self, context, bank_id):
        self.allowed_config_field_calls += 1
        return None


@pytest.mark.asyncio
async def test_config_write_calls_each_tenant_hook_once(memory, request_context):
    """Re-validating under the row lock must not replay per-request hooks.

    Permission and tenant-config hooks may reserve quota or make time-sensitive
    decisions, so the write-time re-check reuses what validation already
    resolved instead of calling them again.
    """
    bank = bank_id("hooks")
    extension = CountingTenantExtension()
    resolver = ConfigResolver(backend=memory._backend, tenant_extension=extension)
    try:
        await memory.update_bank_config(bank, {"enable_observations": True}, request_context=request_context)

        # An update that touches a cross-field constrained field takes the
        # re-check path; the unconstrained one skips it. Neither may re-run a hook.
        await resolver.update_bank_config(bank, {"recall_budget_max": 900}, request_context)
        assert extension.tenant_config_calls == 1
        assert extension.allowed_config_field_calls == 1

        await resolver.update_bank_config(bank, {"enable_reranking": False}, request_context)
        assert extension.tenant_config_calls == 1
        assert extension.allowed_config_field_calls == 2
    finally:
        await memory.delete_bank(bank, request_context=request_context)
