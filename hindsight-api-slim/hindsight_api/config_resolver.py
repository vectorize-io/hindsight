"""
Configuration resolution with hierarchical overrides.

Resolves config values through the hierarchy:
  Global (env vars) → Tenant config (via extension) → Bank config (database)

Config values are resolved on every request to ensure consistency across
multiple API servers.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, fields, replace
from functools import lru_cache
from types import UnionType
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from hindsight_api.config import (
    RECALL_BUDGET_FUNCTIONS,
    HindsightConfig,
    _get_raw_config,
    normalize_config_dict,
    validate_retain_chunking_config,
    validate_retain_completion_token_budget,
)
from hindsight_api.engine.memory_engine import fq_table
from hindsight_api.extensions.tenant import TenantExtension
from hindsight_api.models import RequestContext

if TYPE_CHECKING:
    from hindsight_api.engine.db.base import DatabaseBackend

logger = logging.getLogger(__name__)


class BankConfigPersistenceConflictError(ValueError):
    """Raised when a validated bank config update can no longer be persisted."""

    def __init__(self, bank_id: str):
        self.bank_id = bank_id
        super().__init__(f"Cannot update config for bank '{bank_id}': the bank does not exist")


# Fields that participate in a constraint spanning more than one config field.
# An update touching any of them is re-checked against the committed overrides
# while the bank row is locked (see _persist_bank_config), because two updates
# that are each valid against the old state can combine into an invalid one.
#
# An update touching none of them cannot change either side of any of these
# constraints, so the constrained state it leaves behind is exactly the state it
# found, and it needs no re-check.
_CROSS_FIELD_CONSTRAINED_FIELDS = frozenset(
    {
        # retain_max_completion_tokens (server-level) > retain_chunk_size, for the
        # bank config and for every retain strategy spliced onto it.
        "retain_chunk_size",
        "retain_structured_chunk_size",
        "retain_strategies",
        # recall_budget_min <= recall_budget_max
        "recall_budget_min",
        "recall_budget_max",
    }
)


@dataclass(frozen=True)
class ValidatedBankConfigUpdate:
    """Normalized bank overrides plus what is needed to re-validate them at write time.

    ``updates`` is ready to merge into ``banks.config``. ``parent_config`` is the
    resolved global + tenant base the projected configuration is rebuilt from
    while the bank row is locked; it is None when the update touches no
    cross-field constrained field and therefore needs no re-check. Resolving it
    during validation (rather than during persistence) keeps the tenant config
    hook on the once-per-request path and off the locked path.
    """

    updates: dict[str, Any]
    parent_config: dict[str, Any] | None = None


def _validate_retain_strategy_chunking(base_config: HindsightConfig, strategies: Any) -> None:
    """Validate retain strategy chunking with the same semantics as apply_strategy()."""
    if not isinstance(strategies, dict):
        return
    configurable = HindsightConfig.get_configurable_fields()
    for strategy_name, overrides in strategies.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Invalid retain strategy {strategy_name!r}: must be an object")
        filtered = {k: v for k, v in overrides.items() if k in configurable}
        if not filtered:
            continue
        try:
            resolved = replace(base_config, **filtered)
            validate_retain_chunking_config(
                resolved.retain_chunk_size,
                resolved.retain_structured_chunk_size,
            )
            validate_retain_completion_token_budget(
                llm_provider=resolved.llm_provider,
                retain_max_completion_tokens=resolved.retain_max_completion_tokens,
                retain_chunk_size=resolved.retain_chunk_size,
                retain_llm_model=resolved.retain_llm_model,
                llm_model=resolved.llm_model,
                retain_llm_provider=resolved.retain_llm_provider,
            )
        except ValueError as e:
            raise ValueError(f"Invalid retain strategy {strategy_name!r}: {e}") from e


def _validate_projected_bank_config(
    parent_config: dict[str, Any],
    current_overrides: dict[str, Any],
    normalized_updates: dict[str, Any],
    configurable_fields: set[str],
) -> None:
    """Validate the whole configuration an update would leave committed.

    Every constraint here spans more than one field, so it can only be checked
    against the complete projected state — never against the update alone.
    """
    projected = dict(current_overrides)
    for key, value in normalized_updates.items():
        if key not in configurable_fields:
            continue
        if value is None:
            # JSON null is the "Server Default" tombstone: it clears the override.
            projected.pop(key, None)
        else:
            projected[key] = value

    base_config = HindsightConfig(**{**parent_config, **projected})
    validate_retain_chunking_config(
        base_config.retain_chunk_size,
        base_config.retain_structured_chunk_size,
    )
    _validate_retain_strategy_chunking(base_config, base_config.retain_strategies)
    _validate_recall_budget_bounds(base_config.recall_budget_min, base_config.recall_budget_max)


class ConfigResolver:
    """Resolves hierarchical configuration with tenant/bank overrides."""

    def __init__(self, backend: "DatabaseBackend", tenant_extension: TenantExtension | None = None):
        """
        Initialize config resolver.

        Args:
            backend: Database backend for connection acquisition
            tenant_extension: Optional tenant extension for tenant-level config and permissions
        """
        self._backend = backend
        self.tenant_extension = tenant_extension
        self._global_config = _get_raw_config()
        self._configurable_fields = HindsightConfig.get_configurable_fields()
        self._credential_fields = HindsightConfig.get_credential_fields()

    async def _resolve_parent_config_dict(self, bank_id: str, context: RequestContext | None = None) -> dict[str, Any]:
        """Resolve global + tenant config before bank-level overrides."""
        config_dict = asdict(self._global_config)

        if self.tenant_extension and context:
            try:
                tenant_overrides = await self.tenant_extension.get_tenant_config(context)
                if tenant_overrides:
                    # Normalize keys and filter to configurable fields only
                    normalized_tenant = normalize_config_dict(tenant_overrides)
                    configurable_tenant = {k: v for k, v in normalized_tenant.items() if k in self._configurable_fields}
                    config_dict.update(configurable_tenant)
                    logger.debug(
                        f"Applied tenant config overrides for bank {bank_id}: {list(configurable_tenant.keys())}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load tenant config for bank {bank_id}: {e}")

        return config_dict

    async def resolve_full_config(
        self, bank_id: str, context: RequestContext | None = None, *, cached: bool = True
    ) -> HindsightConfig:
        """
        Resolve full HindsightConfig for a bank with hierarchical overrides applied.

        This is for INTERNAL USE ONLY. Returns the complete config object with all fields
        including credentials and static fields. Use get_bank_config() for API responses.

        Resolution order:
        1. Global config (from environment variables)
        2. Tenant config overrides (from TenantExtension.get_tenant_config())
        3. Bank config overrides (from banks.config JSONB)

        Args:
            bank_id: Bank identifier
            context: Request context for tenant config resolution

        Returns:
            Complete HindsightConfig with hierarchical overrides applied
        """
        config_dict = await self._resolve_parent_config_dict(bank_id, context)

        # Load bank config overrides
        bank_overrides = await self._load_bank_config(bank_id, cached=cached)
        if bank_overrides:
            config_dict.update(bank_overrides)
            logger.debug(f"Applied bank config overrides for bank {bank_id}: {list(bank_overrides.keys())}")

        # Return full config object (dataclass doesn't have __init__ that accepts kwargs, so we update the object)
        # Create a new config instance by copying the global config and updating fields
        resolved_config = HindsightConfig(**config_dict)
        # Multi-LLM chains and the reranker failover chain are static credential fields
        # (never tenant/bank-overridable), but asdict() above flattened their member
        # dataclasses into plain dicts. Restore the original typed objects from the global
        # config so the resolved object stays well-typed for any consumer that reads them.
        resolved_config = replace(
            resolved_config,
            reranker_members=self._global_config.reranker_members,
            llm_members=self._global_config.llm_members,
            llm_strategy=self._global_config.llm_strategy,
            retain_llm_members=self._global_config.retain_llm_members,
            retain_llm_strategy=self._global_config.retain_llm_strategy,
            reflect_llm_members=self._global_config.reflect_llm_members,
            reflect_llm_strategy=self._global_config.reflect_llm_strategy,
            consolidation_llm_members=self._global_config.consolidation_llm_members,
            consolidation_llm_strategy=self._global_config.consolidation_llm_strategy,
        )
        validate_retain_chunking_config(
            resolved_config.retain_chunk_size,
            resolved_config.retain_structured_chunk_size,
        )
        return resolved_config

    async def get_bank_config(
        self, bank_id: str, context: RequestContext | None = None, *, cached: bool = True
    ) -> dict[str, Any]:
        """
        Get fully resolved config for a bank (filtered by permissions).

        Resolution order:
        1. Global config (from environment variables)
        2. Tenant config overrides (from TenantExtension.get_tenant_config())
        3. Bank config overrides (from banks.config JSONB)

        ``cached`` defaults to True because this is ALSO on the hot path: ``recall_async`` and
        ``retain_batch_async`` call it per request, and forcing a read there costs a pool acquire
        each time — which is more than the query it carries, since the pool runs five
        ``set_config`` calls on checkout and a ``RESET ALL`` on release.

        The endpoint a user reads a bank's config back through passes ``cached=False``, and must:
        the cache is per PROCESS, so a cached read there answers a successful write with the
        values it just replaced, on whichever pod did not serve the write. Freshness is a property
        of THAT caller, not of this method.

        SECURITY:
        - Only returns configurable fields (excludes static/infrastructure fields)
        - Filters out ALL credential fields (API keys, base URLs, etc.)
        - Further filtered by tenant/bank permissions if extension provides them

        Args:
            bank_id: Bank identifier
            context: Request context for tenant config resolution and permissions

        Returns:
            Dict of allowed configurable fields only (never includes credentials or static fields)
        """
        # Resolve full config with all hierarchical overrides
        resolved_config = await self.resolve_full_config(bank_id, context, cached=cached)
        config_dict = asdict(resolved_config)

        # SECURITY: drop static/infrastructure + credential fields, then permission-filter.
        filtered = self._strip_static_and_credential_fields(config_dict)
        return await self._apply_permission_filter(filtered, bank_id, context)

    def _strip_static_and_credential_fields(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep only configurable, non-credential fields.

        SECURITY: excludes static/infrastructure fields and ALL credential fields
        (API keys, base URLs, etc.) so a resolved config is safe to return over the API.
        """
        return {
            k: v for k, v in config_dict.items() if k in self._configurable_fields and k not in self._credential_fields
        }

    async def _apply_permission_filter(
        self, filtered: dict[str, Any], bank_id: str, context: RequestContext | None
    ) -> dict[str, Any]:
        """Further restrict already-stripped config to the tenant/bank permission allow-list.

        On extension error, leaves ``filtered`` unchanged (parity with the historical
        single-bank path: a permissions lookup failure must not leak or drop fields).
        """
        if not (self.tenant_extension and context):
            return filtered
        try:
            allowed_fields = await self.tenant_extension.get_allowed_config_fields(context, bank_id)
            if allowed_fields is not None:  # None means "allow all"
                filtered = {k: v for k, v in filtered.items() if k in allowed_fields}
                logger.debug(
                    f"Applied permission filter for bank {bank_id}: allowed={len(allowed_fields)} fields, "
                    f"returned={len(filtered)} fields"
                )
        except Exception as e:
            logger.warning(f"Failed to load permissions for bank {bank_id}: {e}")
        return filtered

    async def get_bank_configs(
        self, bank_ids: list[str], context: RequestContext | None = None
    ) -> dict[str, dict[str, Any]]:
        """Batch variant of :meth:`get_bank_config` for many banks.

        Equivalent to calling ``get_bank_config`` per bank, but resolves the
        global + tenant base once and loads every bank's ``banks.config`` JSONB
        in a single query, instead of one config round-trip per bank. Used by
        ``list_banks`` to overlay disposition + mission without an N+1.

        Returns a mapping of bank_id -> filtered configurable-field dict. A bank
        with no config row still appears, mapped to the global+tenant base.
        """
        if not bank_ids:
            return {}

        # Global + tenant base, resolved once (tenant override is per-request, not per-bank).
        base_dict = asdict(self._global_config)
        if self.tenant_extension and context:
            try:
                tenant_overrides = await self.tenant_extension.get_tenant_config(context)
                if tenant_overrides:
                    normalized_tenant = normalize_config_dict(tenant_overrides)
                    base_dict.update({k: v for k, v in normalized_tenant.items() if k in self._configurable_fields})
            except Exception as e:
                logger.warning(f"Failed to load tenant config for bulk resolve: {e}")

        # All bank overrides in one query, then merge + strip per bank.
        bank_overrides = await self._load_bank_configs(bank_ids)
        stripped = {
            bank_id: self._strip_static_and_credential_fields({**base_dict, **bank_overrides.get(bank_id, {})})
            for bank_id in bank_ids
        }

        # Permission filter is per-bank; resolve concurrently when an extension is present.
        if not (self.tenant_extension and context):
            return stripped
        permission_filtered = await asyncio.gather(
            *(self._apply_permission_filter(stripped[bank_id], bank_id, context) for bank_id in bank_ids)
        )
        return dict(zip(bank_ids, permission_filtered, strict=True))

    async def _load_bank_config(self, bank_id: str, *, cached: bool = True) -> dict[str, Any]:
        """
        Load bank config overrides from banks.config JSONB column.

        Args:
            bank_id: Bank identifier
            cached: read through the per-process cache. True on the RETAIN path, where this runs
                once per sub-batch and a bank config that lags by one TTL changes nothing a caller
                can see. False for anything that answers a reader about the bank's own config: the
                cache is per PROCESS, so a write served by one pod is invisible to the others until
                their entry expires, and a deployment runs several. Read-your-writes on a config
                edit is not a race a user should have to lose.

        Returns:
            Dict of config overrides (only configurable fields, normalized keys)
        """
        # Cached per process (see engine/bank_info_cache). This is read on the WRITE path -- once
        # per sub-batch of every retain -- and each read takes a pooled connection, whose checkout
        # and release cost more statements than the query itself. Caching it is what lets a retain
        # into a store-owned bank, which writes nothing to Postgres, avoid the pool entirely.
        #
        # The RAW column is cached, not the resolved overrides: `_active_bank_overrides` filters
        # by what is configurable and permitted, and that answer can differ between callers of the
        # same bank. Caching the resolved form would serve one caller's permissions to another.
        from .engine import bank_info_cache

        async def _read_config_row() -> dict:
            try:
                async with self._backend.acquire() as conn:
                    row = await conn.fetchrow(
                        f"""
                        SELECT config FROM {fq_table("banks")} WHERE bank_id = $1
                        """,
                        bank_id,
                    )
                # Wrapped in a dict because the cache stores dicts; `{}` means "no row, or no
                # config", which is also what this function returns for that case anyway.
                return {"config": row["config"]} if row and row["config"] else {}
            except Exception as e:
                logger.error(f"Failed to load bank config for {bank_id}: {e}")
                # Re-raise nothing: the original swallowed this and returned {}. Returning the
                # empty dict keeps that behaviour, but it must NOT be cached as if it were an
                # answer -- bank_info_cache drops empty values for exactly this reason.
                return {}

        row = (
            await bank_info_cache.get_or_load(bank_id, "config", _read_config_row)
            if cached
            else await _read_config_row()
        )
        if row.get("config"):
            return self._active_bank_overrides(bank_id, row["config"])

        return {}

    def _active_bank_overrides(self, bank_id: str, config_data: Any) -> dict[str, Any]:
        """Parse a stored ``banks.config`` value into active, normalized overrides."""
        if not config_data:
            return {}

        # Handle case where JSONB is returned as JSON string
        if isinstance(config_data, str):
            config_data = json.loads(config_data)

        # Normalize keys (handle both env var format and Python field format)
        normalized = normalize_config_dict(config_data)

        # Only active overrides for configurable fields. JSON null is a tombstone
        # for "Server Default" in the bank-config UI and must not override defaults.
        active = {k: v for k, v in normalized.items() if k in self._configurable_fields and v is not None}
        return _coerce_stored_bank_overrides(bank_id, active)

    async def _load_bank_configs(self, bank_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Bulk variant of :meth:`_load_bank_config`: load many banks' overrides in one query.

        Returns a mapping of bank_id -> normalized active overrides. Banks with no row
        (or an empty/all-tombstone config) are simply absent from the mapping.
        """
        result: dict[str, dict[str, Any]] = {}
        if not bank_ids:
            return result
        try:
            async with self._backend.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT bank_id, config FROM {fq_table("banks")} WHERE bank_id = ANY($1)
                    """,
                    bank_ids,
                )
                for row in rows:
                    overrides = self._active_bank_overrides(row["bank_id"], row["config"])
                    if overrides:
                        result[row["bank_id"]] = overrides
        except Exception as e:
            logger.error(f"Failed to bulk-load bank configs: {e}")
        return result

    async def validate_bank_config_updates(
        self,
        bank_id: str,
        updates: dict[str, Any],
        context: RequestContext | None = None,
        *,
        projected_bank_overrides: dict[str, Any] | None = None,
        check_permissions: bool = True,
    ) -> ValidatedBankConfigUpdate:
        """
        Normalize and validate bank configuration overrides.

        Permission hooks run here, once per request. Cross-field constraints are
        checked here too, so an invalid update is rejected before the bank is
        created — but they are checked again inside ``_persist_bank_config``,
        which is where the result is actually guaranteed.

        Args:
            bank_id: Bank identifier
            updates: Dict of config field names to new values.
                    Keys can be in env var format (HINDSIGHT_API_LLM_PROVIDER)
                    or Python field format (llm_provider).
                    Only configurable fields are allowed.
            context: Request context for permission checking
            projected_bank_overrides: Bank overrides to use as the validation
                base instead of loading the current bank row.
            check_permissions: Whether client field permissions apply to these
                updates. Server-owned projected values set this to false.

        Returns:
            Normalized updates ready to persist, with the base needed to
            re-validate them at write time.

        Raises:
            ValueError: If attempting to override invalid/disallowed fields.
        """
        # Normalize keys
        normalized_updates = normalize_config_dict(updates)

        # SECURITY: Reject credential fields explicitly
        credential_attempts = set(normalized_updates.keys()) & self._credential_fields
        if credential_attempts:
            raise ValueError(
                f"Cannot set credential fields via API: {sorted(credential_attempts)}. "
                f"Credentials (API keys, base URLs) must be set at server level only."
            )

        # Validate all fields are configurable
        invalid_fields = set(normalized_updates.keys()) - self._configurable_fields
        if invalid_fields:
            static_fields = HindsightConfig.get_static_fields()
            invalid_static = invalid_fields & static_fields
            if invalid_static:
                raise ValueError(
                    f"Cannot override static (server-level) fields: {sorted(invalid_static)}. "
                    f"Only configurable fields can be overridden per-bank. "
                    f"Configurable fields include: {sorted(list(self._configurable_fields)[:10])}... "
                    f"(total: {len(self._configurable_fields)} fields)"
                )
            else:
                raise ValueError(
                    f"Unknown configuration fields: {sorted(invalid_fields)}. "
                    f"Valid configurable fields: {sorted(list(self._configurable_fields)[:10])}..."
                )

        # PERMISSIONS: Check tenant/bank permissions
        if check_permissions and self.tenant_extension and context:
            try:
                allowed_fields = await self.tenant_extension.get_allowed_config_fields(context, bank_id)
                if allowed_fields is not None:  # None means "allow all"
                    disallowed = set(normalized_updates.keys()) - allowed_fields
                    if disallowed:
                        raise ValueError(
                            f"Not allowed to modify fields: {sorted(disallowed)}. "
                            f"Your permissions allow: {sorted(list(allowed_fields)[:10])}..."
                            if allowed_fields
                            else f"Not allowed to modify fields: {sorted(disallowed)}. "
                            "Your permissions do not allow any config modifications."
                        )
            except ValueError:
                raise  # Re-raise permission errors
            except Exception as e:
                logger.warning(f"Failed to check permissions for bank {bank_id}: {e}")
                # Continue without permission check (fail open for backward compatibility)

        # Validate every value against its declared field type before the
        # field-specific checks below, so a wrong-shaped value is reported as such
        # instead of tripping a structural validator with a confusing message.
        _validate_config_value_types(normalized_updates)

        # Validate entity_labels structure
        if "entity_labels" in normalized_updates and normalized_updates["entity_labels"] is not None:
            from .engine.retain.entity_labels import parse_entity_labels

            try:
                parse_entity_labels(normalized_updates["entity_labels"])
            except Exception as e:
                raise ValueError(f"Invalid entity_labels format: {e}")

        # Validate retain_strategies: reject empty string keys
        if "retain_strategies" in normalized_updates and normalized_updates["retain_strategies"]:
            empty_keys = [k for k in normalized_updates["retain_strategies"] if not str(k).strip()]
            if empty_keys:
                raise ValueError(
                    "Strategy names must not be empty strings. Remove entries with empty names before saving."
                )
            # A strategy's overrides are applied with dataclasses.replace() at retain
            # time, so a wrong-shaped value there wedges the bank exactly as a
            # top-level one would. Same contract, same door.
            for strategy_name, strategy_overrides in normalized_updates["retain_strategies"].items():
                if not isinstance(strategy_overrides, dict):
                    raise ValueError(f"Invalid retain strategy {strategy_name!r}: must be an object")
                try:
                    _validate_config_value_types(normalize_config_dict(strategy_overrides))
                except ValueError as e:
                    raise ValueError(f"Invalid retain strategy {strategy_name!r}: {e}") from e

        # Validate recall budget fields
        _validate_recall_budget_updates(normalized_updates)

        # Validate disposition trait fields (1-5 integer scale)
        _validate_disposition_updates(normalized_updates)

        # Constraints spanning several fields need the whole projected
        # configuration, not just the update. This check is the early one, so a
        # bad update is rejected before the bank is created; _persist_bank_config
        # repeats it against the committed state under the bank row lock, which
        # is what makes the result independent of request interleaving.
        parent_config: dict[str, Any] | None = None
        if not _CROSS_FIELD_CONSTRAINED_FIELDS.isdisjoint(normalized_updates):
            parent_config = await self._resolve_parent_config_dict(bank_id, context)
            current_overrides = (
                await self._load_bank_config(bank_id)
                if projected_bank_overrides is None
                else dict(projected_bank_overrides)
            )
            _validate_projected_bank_config(
                parent_config,
                current_overrides,
                normalized_updates,
                self._configurable_fields,
            )

        return ValidatedBankConfigUpdate(updates=normalized_updates, parent_config=parent_config)

    async def update_bank_config(
        self, bank_id: str, updates: dict[str, Any], context: RequestContext | None = None
    ) -> None:
        """Validate and persist bank configuration overrides for an existing bank.

        Bank creation belongs to ``MemoryEngine``; this raises ``ValueError`` if
        the bank does not exist rather than silently discarding the overrides.
        """
        validated = await self.validate_bank_config_updates(bank_id, updates, context)
        await self._persist_bank_config(bank_id, validated)

    async def _persist_bank_config(self, bank_id: str, validated: ValidatedBankConfigUpdate) -> None:
        """Persist validated overrides, re-checking them against the committed state.

        Validation and persistence are one serialized unit: the bank row is
        locked for the whole transaction, so a concurrent config write to the
        same bank either commits before this one is validated or waits for it.
        Neither can slip a cross-field constraint past the other by validating
        against overrides that a third write has since replaced.

        The lock is pessimistic on purpose. An optimistic revision would have to
        re-run validation on retry, and validation calls tenant permission hooks
        that may reserve quota or make time-sensitive decisions; those hooks run
        exactly once, before this transaction opens. A caller that loses the race
        gets the same ValueError the losing update would have raised had the two
        writes been ordered, rather than a conflict status to retry.
        """
        async with self._backend.transaction() as conn:
            # The lock is what serializes writers; it also settles whether the
            # bank exists. Bank lifecycle belongs to MemoryEngine, so a missing
            # row means a caller skipped the engine's provisioning step — fail
            # loudly rather than update nothing and report success.
            row = await conn.fetchrow(
                f"""
                SELECT config FROM {fq_table("banks")} WHERE bank_id = $1 FOR UPDATE
                """,
                bank_id,
            )
            if row is None:
                raise BankConfigPersistenceConflictError(bank_id)

            if validated.parent_config is not None:
                _validate_projected_bank_config(
                    validated.parent_config,
                    self._active_bank_overrides(bank_id, row["config"]),
                    validated.updates,
                    self._configurable_fields,
                )

            # COALESCE guards against a NULL config column (NULL || jsonb is
            # NULL), which would drop the override.
            await conn.execute(
                f"""
                UPDATE {fq_table("banks")}
                SET config = COALESCE(config, '{{}}'::jsonb) || $1::jsonb,
                    updated_at = now()
                WHERE bank_id = $2
                """,
                json.dumps(validated.updates),
                bank_id,
            )

        # The config column just changed; the next read must not serve the entry loaded before it.
        from .engine import bank_info_cache

        await bank_info_cache.invalidate(bank_id, "config")

        logger.info(f"Updated bank config for {bank_id}: {list(validated.updates.keys())}")

    async def reset_bank_config(self, bank_id: str) -> None:
        """
        Reset bank configuration to defaults (remove all overrides).

        No row lock here, unlike _persist_bank_config: this single statement
        already blocks on the lock a concurrent config write holds, and the
        state it leaves — no overrides at all — cannot violate a cross-field
        constraint, so there is nothing to re-validate.

        Args:
            bank_id: Bank identifier
        """
        async with self._backend.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {fq_table("banks")}
                SET config = '{{}}'::jsonb,
                    updated_at = now()
                WHERE bank_id = $1
                """,
                bank_id,
            )

        # The config column just changed; the next read must not serve the entry loaded before it.
        from .engine import bank_info_cache

        await bank_info_cache.invalidate(bank_id, "config")

        logger.info(f"Reset bank config for {bank_id} to defaults")


# Fields whose accepted input shape is deliberately wider than the dataclass
# annotation, because a dedicated structural validator normalizes them later.
_WIDENED_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    # parse_entity_labels() accepts both the bare list of label groups and the
    # {"attributes": [...]} envelope, though the field is annotated `list | None`.
    "entity_labels": (list, dict),
}


def _runtime_types(declared: Any) -> tuple[type, ...]:
    """Runtime-checkable base classes for a dataclass field annotation.

    Unwraps unions (``str | None``) and generic aliases (``list[str]`` -> ``list``);
    ``None`` is dropped because callers handle the tombstone separately. Returns an
    empty tuple for anything not reducible to concrete classes, which the callers
    read as "no type contract to enforce".
    """
    if declared is type(None):
        return ()
    origin = get_origin(declared)
    if origin in (Union, UnionType):
        return tuple(t for arg in get_args(declared) for t in _runtime_types(arg))
    if origin is not None:
        return (origin,) if isinstance(origin, type) else ()
    return (declared,) if isinstance(declared, type) else ()


@lru_cache(maxsize=1)
def _configurable_field_types() -> dict[str, tuple[type, ...]]:
    """Map each configurable field to the value types it accepts."""
    configurable = HindsightConfig.get_configurable_fields()
    field_types: dict[str, tuple[type, ...]] = {}
    for field in fields(HindsightConfig):
        if field.name not in configurable:
            continue
        allowed = _WIDENED_FIELD_TYPES.get(field.name) or _runtime_types(field.type)
        if allowed:
            field_types[field.name] = allowed
    return field_types


def _value_matches_type(value: Any, allowed: tuple[type, ...]) -> bool:
    """Whether ``value`` satisfies a field's declared type contract."""
    if isinstance(value, bool):
        # bool is an int subclass; it must not slip into a numeric field.
        return bool in allowed
    if isinstance(value, int) and float in allowed:
        # JSON draws no int/float distinction: 1 is a valid ratio.
        return True
    return isinstance(value, allowed)


# Field types are reported to API clients, so name them the way the JSON payload
# reads rather than by their Python class.
_TYPE_DESCRIPTIONS: dict[type, str] = {
    bool: "a boolean",
    int: "an integer",
    float: "a number",
    str: "a string",
    list: "a list",
    dict: "an object",
}


def _describe_types(allowed: tuple[type, ...]) -> str:
    return " or ".join(dict.fromkeys(_TYPE_DESCRIPTIONS.get(t, t.__name__) for t in allowed))


def _validate_config_value_types(updates: dict[str, Any]) -> None:
    """Reject values whose type contradicts the declared HindsightConfig type.

    Without this, the bank-config API happily stores e.g. a JSON object in
    ``observations_mission``; the write succeeds and the bank then fails every
    consolidation with ``expected string or bytes-like object, got 'dict'`` from
    deep inside prompt assembly (issue #3218). Reject at the door instead, naming
    the field and the expected type.
    """
    field_types = _configurable_field_types()
    for key, value in updates.items():
        allowed = field_types.get(key)
        # None is the "clear this override" tombstone; unknown keys are rejected
        # elsewhere as non-configurable.
        if allowed is None or value is None:
            continue
        if not _value_matches_type(value, allowed):
            raise ValueError(f"{key} must be {_describe_types(allowed)}, got {type(value).__name__}")


def _coerce_stored_bank_overrides(bank_id: str, overrides: dict[str, Any], where: str = "") -> dict[str, Any]:
    """Make stored bank overrides safe to consume, tolerating pre-validation shapes.

    ``_validate_config_value_types`` rejects bad types at write time, but banks
    configured before that landed can still hold e.g. a JSON object in a
    string-typed field. Every consumer that treats such a value as text blows up
    identically on every run (``escape_for_prompt`` -> ``re.sub`` ->
    "expected string or bytes-like object, got 'dict'"), so the bank's
    consolidation never recovers on its own (issue #3218).

    String fields are JSON-encoded, which preserves the author's intent — the
    structure still reaches the prompt, as text. Anything else is dropped so the
    bank falls back to the tenant/global value rather than wedging.

    ``where`` labels the location in warnings; it is set when recursing into a
    retain strategy, whose overrides reach the same fields via ``apply_strategy``.
    """
    field_types = _configurable_field_types()
    coerced: dict[str, Any] = {}
    for key, value in overrides.items():
        allowed = field_types.get(key)
        # None passes through: the caller has already dropped top-level tombstones,
        # and inside a retain strategy a null is a deliberate override to None.
        if allowed is None or value is None or _value_matches_type(value, allowed):
            coerced[key] = value
            continue
        if str in allowed:
            coerced[key] = json.dumps(value, ensure_ascii=False)
            logger.warning(
                f"Bank {bank_id} config field '{key}'{where} holds a {type(value).__name__} but is a string field; "
                f"using its JSON encoding. Re-save this field as a string to silence this warning."
            )
        else:
            logger.warning(
                f"Bank {bank_id} config field '{key}'{where} holds a {type(value).__name__} but must be "
                f"{_describe_types(allowed)}; ignoring the override and falling back to the server default."
            )

    # Strategy overrides are spliced onto the resolved config by apply_strategy(),
    # so a bad value nested there wedges the bank just as a top-level one does.
    strategies = coerced.get("retain_strategies")
    if isinstance(strategies, dict):
        coerced["retain_strategies"] = {
            name: (
                _coerce_stored_bank_overrides(bank_id, strategy, where=f" in retain strategy {name!r}")
                if isinstance(strategy, dict)
                else strategy
            )
            for name, strategy in strategies.items()
        }
    return coerced


_RECALL_BUDGET_FIXED_KEYS = (
    "recall_budget_fixed_low",
    "recall_budget_fixed_mid",
    "recall_budget_fixed_high",
)
_RECALL_BUDGET_ADAPTIVE_KEYS = (
    "recall_budget_adaptive_low",
    "recall_budget_adaptive_mid",
    "recall_budget_adaptive_high",
)


def _validate_recall_budget_updates(updates: dict[str, Any]) -> None:
    """Validate recall budget config updates. Raises ValueError on invalid input."""
    if "recall_budget_function" in updates:
        function = updates["recall_budget_function"]
        if not isinstance(function, str) or function.lower() not in RECALL_BUDGET_FUNCTIONS:
            raise ValueError(
                f"recall_budget_function must be one of {sorted(RECALL_BUDGET_FUNCTIONS)}, got {function!r}"
            )

    for key in _RECALL_BUDGET_FIXED_KEYS:
        if key in updates:
            value = updates[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")

    for key in _RECALL_BUDGET_ADAPTIVE_KEYS:
        if key in updates:
            value = updates[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{key} must be a positive number, got {value!r}")

    for key in ("recall_budget_min", "recall_budget_max"):
        if key in updates:
            value = updates[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer, got {value!r}")

    if "recall_budget_min" in updates and "recall_budget_max" in updates:
        # The one-sided case (only one bound in the update) is checked against
        # the committed state in _validate_projected_bank_config; this catches a
        # self-contradictory update without a database read.
        _validate_recall_budget_bounds(updates["recall_budget_min"], updates["recall_budget_max"])


def _validate_recall_budget_bounds(recall_budget_min: Any, recall_budget_max: Any) -> None:
    """Validate the recall budget bounds against each other."""
    if recall_budget_min is None or recall_budget_max is None:
        return
    if recall_budget_min > recall_budget_max:
        raise ValueError(f"recall_budget_min ({recall_budget_min}) must be <= recall_budget_max ({recall_budget_max})")


_DISPOSITION_KEYS = (
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
)


def _validate_disposition_updates(updates: dict[str, Any]) -> None:
    """Validate disposition trait config updates. Raises ValueError on invalid input.

    Each trait is an integer on a 1-5 scale (or None to clear the per-bank
    override). The read overlay injects the stored value verbatim into a strict
    ``DispositionTraits(int, ge=1, le=5)``; an out-of-contract value (a float, a
    0-1 scale, or an int outside 1-5) accepted here would later 500 the whole
    bank list when any bank profile is serialized (issue #2348).
    """
    for key in _DISPOSITION_KEYS:
        if key in updates:
            value = updates[key]
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 5):
                raise ValueError(f"{key} must be an integer between 1 and 5, got {value!r}")


def apply_strategy(config: HindsightConfig, strategy_name: str) -> HindsightConfig:
    """
    Apply a named retain strategy's overrides on top of a resolved config.

    A strategy is a named set of hierarchical field overrides stored in
    config.retain_strategies. Any field in _HIERARCHICAL_FIELDS can be
    overridden, including retain_extraction_mode, retain_chunk_size,
    retain_structured_chunk_size, entity_labels,
    entities_allow_free_form, etc.

    Unknown strategy names log a warning and return config unchanged.
    Unknown or non-hierarchical fields in the strategy are silently ignored.
    """
    strategies = config.retain_strategies or {}
    if strategy_name not in strategies:
        logger.warning(f"Unknown retain strategy '{strategy_name}', using resolved config as-is")
        return config

    overrides = strategies[strategy_name]
    if not isinstance(overrides, dict):
        logger.warning(f"Retain strategy '{strategy_name}' is not a dict, skipping")
        return config

    configurable = HindsightConfig.get_configurable_fields()
    filtered = {k: v for k, v in overrides.items() if k in configurable}

    if not filtered:
        return config

    logger.debug(f"Applying retain strategy '{strategy_name}': {list(filtered.keys())}")
    resolved = replace(config, **filtered)
    validate_retain_chunking_config(
        resolved.retain_chunk_size,
        resolved.retain_structured_chunk_size,
    )
    validate_retain_completion_token_budget(
        llm_provider=resolved.llm_provider,
        retain_max_completion_tokens=resolved.retain_max_completion_tokens,
        retain_chunk_size=resolved.retain_chunk_size,
        retain_llm_model=resolved.retain_llm_model,
        llm_model=resolved.llm_model,
        retain_llm_provider=resolved.retain_llm_provider,
    )
    return resolved
