"""
OpenClaw auth-profile credential resolver for Hindsight.

Reads LLM credentials from OpenClaw's auth-profile system so the Hindsight
daemon can reuse existing provider credentials without separate API keys
or CLI installations.

OpenClaw stores credentials in three profile types:
- api_key: standard platform keys (key field)
- token: static bearer/PAT tokens (token field)
- oauth: refreshable OAuth with access/refresh/expires (access field)

Config metadata (auth.profiles, auth.order) lives in openclaw.json.
Actual secrets live in agents/{agentId}/agent/auth-profiles.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jwt as pyjwt

logger = logging.getLogger(__name__)

ProfileType = Literal["api_key", "token", "oauth", ""]

ANTHROPIC_OAT_PREFIX = "sk-ant-oat"
ANTHROPIC_OAUTH_BETA_HEADER = "oauth-2025-04-20"
CODEX_PROVIDER = "openai-codex"
ANTHROPIC_PROVIDER = "anthropic"
CODEX_JWT_AUTH_CLAIM = "https://api.openai.com/auth"

PROVIDER_ALIASES: dict[str, list[str]] = {
    "gemini": ["google", "google-gemini-cli"],
    "volcano": ["volcengine", "bytedance", "doubao"],
}


@dataclass
class OpenClawCredential:
    """Resolved credential from OpenClaw auth-profiles."""

    api_key: str | None = None
    auth_token: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    account_id: str | None = None
    profile_type: ProfileType = ""
    profile_id: str = ""


@dataclass
class OpenClawAuthConfig:
    """Paths needed for OpenClaw credential resolution."""

    config_path: str
    auth_profiles_path: str


def resolve_openclaw_credentials(provider: str, config: OpenClawAuthConfig) -> OpenClawCredential:
    """
    Resolve LLM credentials from OpenClaw's auth-profile files.

    Resolution chain:
    1. openclaw.json auth.order[provider] → preferred profile ID
    2. openclaw.json auth.profiles → first matching provider
    3. auth-profiles.json direct scan → match by provider field or key prefix
    """
    openclaw_config = _load_json(config.config_path)
    auth_profiles = _load_auth_profiles(config.auth_profiles_path)

    if not auth_profiles:
        raise ValueError(
            f"No auth profiles found at {config.auth_profiles_path}. "
            "Ensure OpenClaw has at least one provider authenticated."
        )

    provider_names = _resolve_provider_names(provider)
    profile_id = _find_profile_id(provider_names, openclaw_config, auth_profiles)

    if not profile_id:
        raise ValueError(
            f"No OpenClaw auth profile found for provider '{provider}' "
            f"(also tried: {provider_names[1:]}). "
            f"Available profiles: {list(auth_profiles.keys())}"
        )

    profile = auth_profiles[profile_id]
    return _build_credential(profile, profile_id, provider)


def reload_credentials(provider: str, config: OpenClawAuthConfig) -> OpenClawCredential:
    """Re-read credentials from disk. Called on 401/403 to pick up refreshed tokens."""
    logger.info(f"Re-reading OpenClaw auth-profiles for provider '{provider}' after auth failure")
    return resolve_openclaw_credentials(provider, config)


def _resolve_provider_names(provider: str) -> list[str]:
    """Return the provider name plus any known aliases."""
    names = [provider.lower()]
    for alias in PROVIDER_ALIASES.get(provider.lower(), []):
        if alias not in names:
            names.append(alias)
    return names


def _find_profile_id(
    provider_names: list[str],
    openclaw_config: dict,
    auth_profiles: dict,
) -> str | None:
    """
    Find the best matching profile ID using OpenClaw's resolution chain.

    Mirrors OpenClaw's own auth resolution: auth.order → auth.profiles → direct scan.
    """
    auth_config = openclaw_config.get("auth", {})

    # Step 1: auth.order — explicit preferred profile per provider
    auth_order = auth_config.get("order", {})
    for name in provider_names:
        ordered = auth_order.get(name, [])
        for profile_id in ordered:
            if profile_id in auth_profiles:
                return profile_id

    # Step 2: auth.profiles metadata — find first matching provider
    auth_profiles_meta = auth_config.get("profiles", {})
    for name in provider_names:
        for profile_id, meta in auth_profiles_meta.items():
            if isinstance(meta, dict) and meta.get("provider") == name and profile_id in auth_profiles:
                return profile_id

    # Step 3: Direct scan of auth-profiles.json — match by provider field or
    # key prefix (e.g. "anthropic:default" starts with "anthropic:").
    for name in provider_names:
        prefix = f"{name}:"
        for profile_id, profile in auth_profiles.items():
            provider_field = profile.get("provider") if isinstance(profile, dict) else None
            if provider_field == name or profile_id.startswith(prefix):
                return profile_id

    return None


def _build_credential(profile: dict, profile_id: str, hindsight_provider: str) -> OpenClawCredential:
    """Extract the secret from a profile and apply provider-specific handling."""
    profile_type = profile.get("type", "")
    secret = _extract_secret(profile, profile_type)

    if not secret:
        raise ValueError(
            f"OpenClaw profile '{profile_id}' (type={profile_type}) has no secret value. "
            "The credential may be stored as a SecretRef that requires OpenClaw runtime resolution."
        )

    provider_lower = hindsight_provider.lower()

    # Anthropic OATs (`sk-ant-oat*`) require auth_token + the OAuth beta header;
    # the standard API-key path doesn't accept them.
    if provider_lower == ANTHROPIC_PROVIDER and secret.startswith(ANTHROPIC_OAT_PREFIX):
        return OpenClawCredential(
            auth_token=secret,
            extra_headers={"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER},
            profile_type=profile_type,
            profile_id=profile_id,
        )

    if provider_lower == CODEX_PROVIDER and profile_type == "oauth":
        account_id = _decode_codex_account_id(secret)
        if not account_id:
            raise ValueError(
                f"OpenClaw profile '{profile_id}' is missing chatgpt_account_id in its "
                "JWT claims. Re-authenticate the codex provider in OpenClaw to refresh "
                "the OAuth token."
            )
        return OpenClawCredential(
            api_key=secret,
            account_id=account_id,
            profile_type=profile_type,
            profile_id=profile_id,
        )

    return OpenClawCredential(
        api_key=secret,
        profile_type=profile_type,
        profile_id=profile_id,
    )


def _extract_secret(profile: dict, profile_type: str) -> str | None:
    """Read the correct secret field based on profile type."""
    field_by_type = {"api_key": "key", "token": "token", "oauth": "access"}
    field = field_by_type.get(profile_type)
    if field is None:
        return None
    # Reject non-string secrets (e.g. SecretRef-shaped dicts that need
    # OpenClaw runtime resolution before they're usable here).
    value = profile.get(field)
    return value if isinstance(value, str) else None


def _decode_codex_account_id(jwt_token: str) -> str | None:
    """Read chatgpt_account_id from the JWT claims without verifying the signature.

    The signing key isn't shipped with the token, and we only need the claim to
    forward to the Codex API — the access token itself remains the auth credential.
    """
    try:
        claims = pyjwt.decode(jwt_token, options={"verify_signature": False})
    except pyjwt.PyJWTError as e:
        logger.warning(f"Failed to decode Codex JWT for account_id: {e}")
        return None
    auth_claims = claims.get(CODEX_JWT_AUTH_CLAIM)
    if not isinstance(auth_claims, dict):
        logger.warning("Codex JWT auth claim is missing or not an object")
        return None
    account_id = auth_claims.get("chatgpt_account_id")
    if not account_id:
        logger.warning("Codex JWT missing chatgpt_account_id claim")
    return account_id


def _load_json(path: str) -> dict:
    """Load a JSON file, returning empty dict if missing or invalid."""
    try:
        with open(Path(path).expanduser()) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def _load_auth_profiles(path: str) -> dict:
    """Load auth-profiles.json and return the profiles dict."""
    data = _load_json(path)
    profiles = data.get("profiles", {})
    return profiles if isinstance(profiles, dict) else {}
