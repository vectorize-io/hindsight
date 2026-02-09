"""
Configuration resolution with hierarchical overrides.

Resolves config values through the hierarchy:
  Global (env vars) → Tenant config (via extension) → Bank config (database)

Implements LRU caching (max 1000 banks) for performance.
"""

import json
import logging
from collections import OrderedDict
from dataclasses import asdict
from typing import Any

import asyncpg

from hindsight_api.config import HindsightConfig, get_config, normalize_config_dict
from hindsight_api.extensions.tenant import TenantExtension
from hindsight_api.models import RequestContext

logger = logging.getLogger(__name__)

# LRU cache for resolved bank configs
# Structure: OrderedDict{bank_id: resolved_config_dict}
# Max 1000 entries, oldest evicted when limit exceeded
_CONFIG_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_MAX_SIZE = 1000


class ConfigResolver:
    """Resolves hierarchical configuration with tenant/bank overrides."""

    def __init__(self, pool: asyncpg.Pool, tenant_extension: TenantExtension | None = None):
        """
        Initialize config resolver.

        Args:
            pool: Database connection pool
            tenant_extension: Optional tenant extension for tenant-level config
        """
        self.pool = pool
        self.tenant_extension = tenant_extension
        self._global_config = get_config()
        self._hierarchical_fields = HindsightConfig.get_hierarchical_fields()

    async def get_bank_config(self, bank_id: str, context: RequestContext | None = None) -> dict[str, Any]:
        """
        Get fully resolved config for a bank.

        Resolution order:
        1. Global config (from environment variables)
        2. Tenant config overrides (from TenantExtension.get_tenant_config())
        3. Bank config overrides (from banks.config JSONB)

        Args:
            bank_id: Bank identifier
            context: Request context for tenant config resolution

        Returns:
            Dict of all config values with hierarchical overrides applied
        """
        # Check LRU cache
        if bank_id in _CONFIG_CACHE:
            logger.debug(f"Config cache hit for bank {bank_id}")
            # Move to end (mark as recently used)
            _CONFIG_CACHE.move_to_end(bank_id)
            return _CONFIG_CACHE[bank_id]

        logger.debug(f"Config cache miss for bank {bank_id}, loading from database")

        # Start with global config (all fields)
        config_dict = asdict(self._global_config)

        # Load tenant config overrides (if tenant extension available)
        if self.tenant_extension and context:
            try:
                tenant_overrides = await self.tenant_extension.get_tenant_config(context)
                if tenant_overrides:
                    # Normalize keys and filter to hierarchical fields only
                    normalized_tenant = normalize_config_dict(tenant_overrides)
                    hierarchical_tenant = {k: v for k, v in normalized_tenant.items() if k in self._hierarchical_fields}
                    config_dict.update(hierarchical_tenant)
                    logger.debug(
                        f"Applied tenant config overrides for bank {bank_id}: {list(hierarchical_tenant.keys())}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load tenant config for bank {bank_id}: {e}")

        # Load bank config overrides
        bank_overrides = await self._load_bank_config(bank_id)
        if bank_overrides:
            config_dict.update(bank_overrides)
            logger.debug(f"Applied bank config overrides for bank {bank_id}: {list(bank_overrides.keys())}")

        # Add to LRU cache
        _CONFIG_CACHE[bank_id] = config_dict

        # Evict oldest entry if cache exceeds max size
        if len(_CONFIG_CACHE) > _CACHE_MAX_SIZE:
            evicted_bank_id = next(iter(_CONFIG_CACHE))  # First item (oldest)
            del _CONFIG_CACHE[evicted_bank_id]
            logger.debug(
                f"Config cache full, evicted oldest entry: {evicted_bank_id} (cache size: {len(_CONFIG_CACHE)})"
            )

        return config_dict

    async def _load_bank_config(self, bank_id: str) -> dict[str, Any]:
        """
        Load bank config overrides from banks.config JSONB column.

        Args:
            bank_id: Bank identifier

        Returns:
            Dict of config overrides (only hierarchical fields, normalized keys)
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

                    # Only return overrides for hierarchical fields
                    return {k: v for k, v in normalized.items() if k in self._hierarchical_fields}
        except Exception as e:
            logger.error(f"Failed to load bank config for {bank_id}: {e}")

        return {}

    async def update_bank_config(self, bank_id: str, updates: dict[str, Any]) -> None:
        """
        Update bank configuration overrides.

        Args:
            bank_id: Bank identifier
            updates: Dict of config field names to new values.
                    Keys can be in env var format (HINDSIGHT_API_LLM_PROVIDER)
                    or Python field format (llm_provider).
                    Only hierarchical fields are allowed.

        Raises:
            ValueError: If attempting to override static fields
        """
        # Normalize keys
        normalized_updates = normalize_config_dict(updates)

        # Validate all fields are hierarchical
        invalid_fields = set(normalized_updates.keys()) - self._hierarchical_fields
        if invalid_fields:
            static_fields = HindsightConfig.get_static_fields()
            invalid_static = invalid_fields & static_fields
            if invalid_static:
                raise ValueError(
                    f"Cannot override static (server-level) fields: {sorted(invalid_static)}. "
                    f"Only hierarchical fields can be overridden per-bank. "
                    f"Hierarchical fields include: {sorted(list(self._hierarchical_fields)[:10])}... "
                    f"(total: {len(self._hierarchical_fields)} fields)"
                )
            else:
                raise ValueError(
                    f"Unknown configuration fields: {sorted(invalid_fields)}. "
                    f"Valid hierarchical fields: {sorted(list(self._hierarchical_fields)[:10])}..."
                )

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

        # Invalidate cache
        self.invalidate_cache(bank_id)
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

        # Invalidate cache
        self.invalidate_cache(bank_id)
        logger.info(f"Reset bank config for {bank_id} to defaults")

    def invalidate_cache(self, bank_id: str) -> None:
        """
        Remove bank from LRU cache.

        Called after config updates to ensure fresh data is loaded.

        Args:
            bank_id: Bank identifier to invalidate
        """
        if bank_id in _CONFIG_CACHE:
            del _CONFIG_CACHE[bank_id]
            logger.debug(f"Invalidated config cache for bank {bank_id}")

    @staticmethod
    def clear_all_cache() -> None:
        """Clear the entire config cache. Useful for testing."""
        global _CONFIG_CACHE
        _CONFIG_CACHE.clear()
        logger.info("Cleared entire config cache")
