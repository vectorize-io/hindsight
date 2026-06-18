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
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

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

    async def resolve_full_config(self, bank_id: str, context: RequestContext | None = None) -> HindsightConfig:
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
        bank_overrides = await self._load_bank_config(bank_id)
        if bank_overrides:
            config_dict.update(bank_overrides)
            logger.debug(f"Applied bank config overrides for bank {bank_id}: {list(bank_overrides.keys())}")

        # Return full config object (dataclass doesn't have __init__ that accepts kwargs, so we update the object)
        # Create a new config instance by copying the global config and updating fields
        resolved_config = HindsightConfig(**config_dict)
        validate_retain_chunking_config(
            resolved_config.retain_chunk_size,
            resolved_config.retain_structured_chunk_size,
        )
        return resolved_config

    async def get_bank_config(self, bank_id: str, context: RequestContext | None = None) -> dict[str, Any]:
        """
        Get fully resolved config for a bank (filtered by permissions).

        Resolution order:
        1. Global config (from environment variables)
        2. Tenant config overrides (from TenantExtension.get_tenant_config())
        3. Bank config overrides (from banks.config JSONB)

        Note: Config is resolved on every call (not cached) to ensure consistency
        across multiple API servers.

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
        resolved_config = await self.resolve_full_config(bank_id, context)
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

    async def _load_bank_config(self, bank_id: str) -> dict[str, Any]:
        """
        Load bank config overrides from banks.config JSONB column.

        Args:
            bank_id: Bank identifier

        Returns:
            Dict of config overrides (only configurable fields, normalized keys)
        """
        try:
            async with self._backend.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT config FROM {fq_table("banks")} WHERE bank_id = $1
                    """,
                    bank_id,
                )

                if row and row["config"]:
                    config_data = row["config"]

                    # Handle case where JSONB is returned as JSON string
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)

                    # Normalize keys (handle both env var format and Python field format)
                    normalized = normalize_config_dict(config_data)

                    # Only return active overrides for configurable fields. JSON null is a tombstone
                    # for "Server Default" in the bank-config UI and should not override defaults.
                    return {k: v for k, v in normalized.items() if k in self._configurable_fields and v is not None}
        except Exception as e:
            logger.error(f"Failed to load bank config for {bank_id}: {e}")

        return {}

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
                    config_data = row["config"]
                    if not config_data:
                        continue
                    # Handle case where JSONB is returned as JSON string
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)

                    # Normalize keys (handle both env var format and Python field format)
                    normalized = normalize_config_dict(config_data)

                    # Only active overrides for configurable fields. JSON null is a tombstone
                    # for "Server Default" in the bank-config UI and must not override defaults.
                    overrides = {
                        k: v for k, v in normalized.items() if k in self._configurable_fields and v is not None
                    }
                    if overrides:
                        result[row["bank_id"]] = overrides
        except Exception as e:
            logger.error(f"Failed to bulk-load bank configs: {e}")
        return result

    async def update_bank_config(
        self, bank_id: str, updates: dict[str, Any], context: RequestContext | None = None
    ) -> None:
        """
        Update bank configuration overrides (with permission checking).

        Args:
            bank_id: Bank identifier
            updates: Dict of config field names to new values.
                    Keys can be in env var format (HINDSIGHT_API_LLM_PROVIDER)
                    or Python field format (llm_provider).
                    Only configurable fields are allowed.
            context: Request context for permission checking

        Raises:
            ValueError: If attempting to override invalid/disallowed fields
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
        if self.tenant_extension and context:
            try:
                allowed_fields = await self.tenant_extension.get_allowed_config_fields(context, bank_id)
                if allowed_fields is not None:  # None means "allow all"
                    disallowed = set(normalized_updates.keys()) - allowed_fields
                    if disallowed:
                        raise ValueError(
                            f"Not allowed to modify fields: {sorted(disallowed)}. "
                            f"Your permissions allow: {sorted(list(allowed_fields)[:10])}..."
                            if allowed_fields
                            else "Not allowed to modify fields: {sorted(disallowed)}. "
                            "Your permissions do not allow any config modifications."
                        )
            except ValueError:
                raise  # Re-raise permission errors
            except Exception as e:
                logger.warning(f"Failed to check permissions for bank {bank_id}: {e}")
                # Continue without permission check (fail open for backward compatibility)

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

        # Validate recall budget fields
        _validate_recall_budget_updates(normalized_updates)

        chunking_fields_updated = (
            "retain_chunk_size" in normalized_updates
            or "retain_structured_chunk_size" in normalized_updates
            or "retain_strategies" in normalized_updates
        )
        if chunking_fields_updated:
            config_dict = await self._resolve_parent_config_dict(bank_id, context)
            active_bank_overrides = await self._load_bank_config(bank_id)
            for key, value in normalized_updates.items():
                if key not in self._configurable_fields:
                    continue
                if value is None:
                    active_bank_overrides.pop(key, None)
                else:
                    active_bank_overrides[key] = value
            config_dict.update(active_bank_overrides)
            base_config = HindsightConfig(**config_dict)
            validate_retain_chunking_config(
                base_config.retain_chunk_size,
                base_config.retain_structured_chunk_size,
            )
            _validate_retain_strategy_chunking(base_config, base_config.retain_strategies)

        # Persist the override. Banks are created lazily (on first retain), so a
        # PATCH that precedes any ingestion would otherwise UPDATE zero rows and
        # silently no-op while returning 200. Ensure the bank row exists first
        # (this also creates its per-bank vector indexes), then merge defensively:
        # COALESCE guards against a NULL config column (NULL || jsonb is NULL),
        # which would drop the override even when a row is updated.
        from .engine.retain.fact_storage import ensure_bank_exists

        async with self._backend.acquire() as conn:
            await ensure_bank_exists(conn, bank_id, ops=self._backend.ops)
            await conn.execute(
                f"""
                UPDATE {fq_table("banks")}
                SET config = COALESCE(config, '{{}}'::jsonb) || $1::jsonb,
                    updated_at = now()
                WHERE bank_id = $2
                """,
                json.dumps(normalized_updates),
                bank_id,
            )

        logger.info(f"Updated bank config for {bank_id}: {list(normalized_updates.keys())}")

    async def reset_bank_config(self, bank_id: str) -> None:
        """
        Reset bank configuration to defaults (remove all overrides).

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

        logger.info(f"Reset bank config for {bank_id} to defaults")


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
        if updates["recall_budget_min"] > updates["recall_budget_max"]:
            raise ValueError(
                f"recall_budget_min ({updates['recall_budget_min']}) must be <= "
                f"recall_budget_max ({updates['recall_budget_max']})"
            )


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
