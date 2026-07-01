"""Google OAuth 2.0 (read-only) config + token storage design.

Secrets policy: client secret, access tokens, and refresh tokens come from the
environment or the token store ONLY — never hardcoded, never logged. ``redact``
collapses any token-bearing dict for safe logging/audit.

The real OAuth exchange (google-auth-oauthlib) is imported lazily; ``build_flow``
raises if the optional extra or config is missing. Token persistence is modeled
by ``TokenStore`` — v0.1 keeps an in-process store; production swaps in an
encrypted secret backend (the interface is the contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings

# Scopes are fixed and read-only. Guard against accidental privilege widening.
_FORBIDDEN_SCOPE_HINTS = ("drive.file", "drive ", "auth/drive\t")
WRITE_SCOPE_DENYLIST = (
    "https://www.googleapis.com/auth/drive",  # full read/write
    "https://www.googleapis.com/auth/drive.appdata",
)


class OAuthConfigError(RuntimeError):
    """Raised when OAuth is misconfigured or would request unsafe scopes."""


def validate_scopes(scopes: tuple[str, ...]) -> None:
    """Fail closed if any non-read-only scope sneaks in."""
    for scope in scopes:
        if scope in WRITE_SCOPE_DENYLIST:
            raise OAuthConfigError(f"refusing non-read-only scope: {scope}")
        if not scope.endswith("readonly"):
            raise OAuthConfigError(f"scope is not read-only: {scope}")


def oauth_status() -> dict[str, Any]:
    """Non-secret view of OAuth config (safe to return from the API)."""
    return {
        "configured": settings.google_configured,
        "redirect_uri": settings.google_redirect_uri,
        "scopes": list(settings.google_scopes),
        "read_only": True,
    }


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace any token/secret-bearing field with [REDACTED] for logs/audit."""
    secret_keys = {"access_token", "refresh_token", "token", "client_secret", "id_token"}
    return {k: ("[REDACTED]" if k in secret_keys and v else v) for k, v in payload.items()}


def build_flow(state: str | None = None):  # pragma: no cover - needs optional extra + creds
    """Create a google-auth-oauthlib Flow for the read-only scopes."""
    if not settings.google_configured:
        raise OAuthConfigError("google client_id/client_secret not configured")
    validate_scopes(settings.google_scopes)
    try:
        from google_auth_oauthlib.flow import Flow  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OAuthConfigError("install the 'gdrive' extra for real OAuth") from exc
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [settings.google_redirect_uri],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=list(settings.google_scopes), state=state)
    flow.redirect_uri = settings.google_redirect_uri
    return flow


@dataclass
class StoredToken:
    """Per-connector token reference. Never logged raw — use ``redact``."""

    connector_id: str
    refresh_token: str
    account_email: str | None = None
    scopes: tuple[str, ...] = field(default_factory=tuple)


class TokenStore:
    """v0.1 in-process token store. Production: encrypted secret backend.

    The interface is the durable contract: ``save``/``get``/``delete`` by
    connector id. Tokens are never returned in API responses.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, StoredToken] = {}

    def save(self, token: StoredToken) -> None:
        self._tokens[token.connector_id] = token

    def get(self, connector_id: str) -> StoredToken | None:
        return self._tokens.get(connector_id)

    def delete(self, connector_id: str) -> bool:
        return self._tokens.pop(connector_id, None) is not None

    def has(self, connector_id: str) -> bool:
        return connector_id in self._tokens


# Process-wide store for v0.1.
token_store = TokenStore()
