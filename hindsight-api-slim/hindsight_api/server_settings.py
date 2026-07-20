"""Instance-wide settings persisted in the database.

The control plane can write these at runtime so a self-hosted operator configures
the instance from the browser instead of only via environment variables. Today
the only setting is the base LLM config (provider / model / api_key / base_url),
stored as a single JSON row keyed ``llm_config`` in the ``server_settings`` table.

Security: the api_key is a credential. It is never returned over the config-read
API (the caller only learns whether one is set). At rest it is stored plaintext
by default; if ``HINDSIGHT_API_SETTINGS_ENC_KEY`` is set, it is encrypted with a
Fernet key derived from that secret.
"""

import base64
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .engine.llm_wrapper import VALID_LLM_PROVIDERS
from .engine.schema import fq_table

if TYPE_CHECKING:
    from .engine.db.base import DatabaseBackend

logger = logging.getLogger(__name__)

_LLM_CONFIG_KEY = "llm_config"
# Marks an api_key value that was encrypted before being written to the DB, so
# reads can tell an encrypted token from a plaintext key.
_ENCRYPTED_PREFIX = "enc:v1:"


@dataclass
class ServerLlmConfig:
    """The persisted base LLM configuration. ``api_key`` is decrypted on load."""

    provider: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None


class ServerSettingsStore:
    """Read/write access to the ``server_settings`` table.

    SQL is written PostgreSQL-style; the Oracle backend transparently rewrites
    it (``$n`` params, ``ON CONFLICT`` → ``MERGE``), matching the rest of the
    engine's queries.
    """

    def __init__(self, backend: "DatabaseBackend", enc_key: str | None = None) -> None:
        self._backend = backend
        self._fernet = self._build_fernet(enc_key) if enc_key else None

    @staticmethod
    def _build_fernet(passphrase: str):
        # Accept any passphrase by deriving a valid 32-byte Fernet key from it.
        from cryptography.fernet import Fernet

        digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
        return Fernet(base64.urlsafe_b64encode(digest))

    def _encrypt_api_key(self, api_key: str | None) -> str | None:
        if not api_key or self._fernet is None:
            return api_key
        token = self._fernet.encrypt(api_key.encode("utf-8")).decode("utf-8")
        return f"{_ENCRYPTED_PREFIX}{token}"

    def _decrypt_api_key(self, stored: str | None) -> str | None:
        if not stored or not stored.startswith(_ENCRYPTED_PREFIX):
            return stored
        if self._fernet is None:
            logger.warning(
                "server_settings: api_key is encrypted but HINDSIGHT_API_SETTINGS_ENC_KEY is not set; "
                "cannot decrypt. Re-save the LLM config to store a usable key."
            )
            return None
        token = stored[len(_ENCRYPTED_PREFIX) :]
        return self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")

    async def load_llm_config(self) -> ServerLlmConfig | None:
        """Return the persisted LLM config, or ``None`` if none is stored."""
        async with self._backend.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT value FROM {fq_table('server_settings')} WHERE setting_key = $1",
                _LLM_CONFIG_KEY,
            )
            if not row:
                return None
            value = conn.parse_json(row["value"])
        if not isinstance(value, dict) or not value.get("provider"):
            return None
        return ServerLlmConfig(
            provider=value["provider"],
            model=value.get("model") or None,
            api_key=self._decrypt_api_key(value.get("api_key")),
            base_url=value.get("base_url") or None,
        )

    async def save_llm_config(
        self,
        *,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Persist the base LLM config (full write; the caller supplies the final api_key)."""
        provider = (provider or "").lower()
        if provider not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Invalid LLM provider: {provider!r}. Must be one of: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
            )
        payload = json.dumps(
            {
                "provider": provider,
                "model": model or None,
                "api_key": self._encrypt_api_key(api_key),
                "base_url": base_url or None,
            }
        )
        async with self._backend.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {fq_table("server_settings")} (setting_key, value)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (setting_key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = now()
                """,
                _LLM_CONFIG_KEY,
                payload,
            )

    async def clear_llm_config(self) -> None:
        """Remove the persisted LLM config (revert to env-level defaults)."""
        async with self._backend.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {fq_table('server_settings')} WHERE setting_key = $1",
                _LLM_CONFIG_KEY,
            )
