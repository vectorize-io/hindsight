"""OAuth config / scope-safety / secret-redaction tests (no real creds)."""

import pytest

from app.connectors.google_drive.oauth import (
    OAuthConfigError,
    StoredToken,
    TokenStore,
    oauth_status,
    redact,
    validate_scopes,
)

READONLY = (
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
)


def test_validate_scopes_accepts_readonly():
    validate_scopes(READONLY)  # should not raise


def test_validate_scopes_rejects_full_drive():
    with pytest.raises(OAuthConfigError):
        validate_scopes(("https://www.googleapis.com/auth/drive",))


def test_validate_scopes_rejects_non_readonly():
    with pytest.raises(OAuthConfigError):
        validate_scopes(("https://www.googleapis.com/auth/drive.file",))


def test_redact_hides_secrets():
    out = redact({"access_token": "ya29.secret", "refresh_token": "rt", "client_secret": "cs",
                  "account_email": "a@x.com", "empty": ""})
    assert out["access_token"] == "[REDACTED]"
    assert out["refresh_token"] == "[REDACTED]"
    assert out["client_secret"] == "[REDACTED]"
    assert out["account_email"] == "a@x.com"  # non-secret preserved
    assert out["empty"] == ""  # empty values not masked


def test_oauth_status_shape_is_non_secret():
    s = oauth_status()
    assert set(s) >= {"configured", "redirect_uri", "scopes", "read_only"}
    assert s["read_only"] is True
    assert "client_secret" not in s


def test_token_store_roundtrip():
    store = TokenStore()
    store.save(StoredToken(connector_id="c1", refresh_token="rt", account_email="a@x.com"))
    assert store.has("c1")
    assert store.get("c1").refresh_token == "rt"
    assert store.delete("c1") is True
    assert not store.has("c1")
