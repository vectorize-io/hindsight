"""
Configuration resolution with hierarchical overrides.

Resolves config values through the hierarchy:
  Global (env vars) → Tenant config (via extension) → Bank config (database)

Config values are resolved on every request to ensure consistency across
multiple API servers. LLM provider instances are pooled separately.
"""

import json
import logging
from dataclasses import asdict
from typing import Any

import asyncpg

from hindsight_api.config import HindsightConfig, _get_raw_config, normalize_config_dict
from hindsight_api.extensions.tenant import TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)


class ConfigResolver:
    """Resolves hierarchical configuration with tenant/bank overrides."""

    def __init__(self, pool: asyncpg.Pool, tenant_extension: TenantExtension | None = None):
        """
        Initialize config resolver.

        Args:
            pool: Database connection pool
            tenant_extension: Optional tenant extension for tenant-level config and permissions
        """
        self.pool = pool
        self.tenant_extension = tenant_extension
        self._global_config = _get_raw_config()
        self._configurable_fields = HindsightConfig.get_configurable_fields()
        self._credential_fields = HindsightConfig.get_credential_fields()

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
        # Start with global config (all fields)
        config_dict = asdict(self._global_config)

        # Load tenant config overrides (if tenant extension available)
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

        # Load bank config overrides
        bank_overrides = await self._load_bank_config(bank_id)
        if bank_overrides:
            config_dict.update(bank_overrides)
            logger.debug(f"Applied bank config overrides for bank {bank_id}: {list(bank_overrides.keys())}")

        # Return full config object (dataclass doesn't have __init__ that accepts kwargs, so we update the object)
        # Create a new config instance by copying the global config and updating fields
        resolved_config = HindsightConfig(**config_dict)
        return resolved_config

    async def get_bank_config(self, bank_id: str, context: RequestContext | None = None) -> dict[str, Any]:
        """
        Get fully resolved config for a bank (filtered by permissions).

        Resolution order:
        1. Global config (from environment variables)
        2. Tenant config overrides (from TenantExtension.get_tenant_config())
        3. Bank config overrides (from banks.config JSONB)

        Note: Config is resolved on every call (not cached) to ensure consistency
        across multiple API servers. LLM provider instances are pooled separately.

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
        # Start with global config (all fields)
        config_dict = asdict(self._global_config)

        # Load tenant config overrides (if tenant extension available)
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

        # Load bank config overrides
        bank_overrides = await self._load_bank_config(bank_id)
        if bank_overrides:
            config_dict.update(bank_overrides)
            logger.debug(f"Applied bank config overrides for bank {bank_id}: {list(bank_overrides.keys())}")

        # SECURITY: Filter to only configurable fields (exclude static/infrastructure)
        filtered = {k: v for k, v in config_dict.items() if k in self._configurable_fields}

        # SECURITY: Remove ALL credential fields (API keys, base URLs, etc.)
        filtered = {k: v for k, v in filtered.items() if k not in self._credential_fields}

        # PERMISSIONS: Further filter based on tenant/bank permissions
        if self.tenant_extension and context:
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

    async def _load_bank_config(self, bank_id: str) -> dict[str, Any]:
        """
        Load bank config overrides from banks.config JSONB column.

        Args:
            bank_id: Bank identifier

        Returns:
            Dict of config overrides (only configurable fields, normalized keys)
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT config FROM banks WHERE bank_id = $1
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

                    # Only return overrides for configurable fields
                    return {k: v for k, v in normalized.items() if k in self._configurable_fields}
        except Exception as e:
            logger.error(f"Failed to load bank config for {bank_id}: {e}")

        return {}

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

        # Merge with existing config (JSONB || operator)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE banks
                SET config = config || $1::jsonb,
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
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE banks
                SET config = '{}'::jsonb,
                    updated_at = now()
                WHERE bank_id = $1
                """,
                bank_id,
            )

        logger.info(f"Reset bank config for {bank_id} to defaults")
