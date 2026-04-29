"""
Tests for OpenClaw auth-profile credential resolver.

Verifies the resolution chain (auth.order → auth.profiles → direct scan),
provider name aliases, OAT detection, JWT decoding, and credential re-read.
"""

import base64
import json
import tempfile
from pathlib import Path

import pytest

from hindsight_api.engine.providers.openclaw_auth import (
    PROVIDER_ALIASES,
    OpenClawAuthConfig,
    OpenClawCredential,
    _decode_codex_account_id,
    _resolve_provider_names,
    reload_credentials,
    resolve_openclaw_credentials,
)


def _make_jwt(claims: dict) -> str:
    """Build a minimal JWT with the given payload claims (no signature verification)."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.fake-signature"


@pytest.fixture()
def auth_dir(tmp_path: Path):
    """Create a temp dir with openclaw.json and auth-profiles.json."""

    auth_profiles = {
        "version": 1,
        "profiles": {
            "openai-codex:user@example.com": {
                "type": "oauth",
                "provider": "openai-codex",
                "access": _make_jwt(
                    {
                        "https://api.openai.com/auth": {
                            "chatgpt_account_id": "acct-123",
                        },
                    }
                ),
                "refresh": "rt_fake",
                "expires": 9999999999999,
                "email": "user@example.com",
            },
            "anthropic:default": {
                "type": "token",
                "provider": "anthropic",
                "token": "sk-ant-oat01-fake-oat-token",
            },
            "openai:default": {
                "type": "api_key",
                "provider": "openai",
                "key": "sk-openai-fake-key",
            },
            "groq:default": {
                "type": "api_key",
                "provider": "groq",
                "key": "gsk-groq-fake-key",
            },
            "google:default": {
                "type": "api_key",
                "provider": "google",
                "key": "gemini-fake-key",
            },
        },
    }

    openclaw_config = {
        "auth": {
            "profiles": {
                "openai-codex:user@example.com": {"provider": "openai-codex", "mode": "oauth"},
                "anthropic:default": {"provider": "anthropic", "mode": "token"},
                "openai:default": {"provider": "openai", "mode": "api_key"},
            },
            "order": {
                "openai-codex": ["openai-codex:user@example.com"],
                "anthropic": ["anthropic:default"],
            },
        },
    }

    config_path = tmp_path / "openclaw.json"
    config_path.write_text(json.dumps(openclaw_config))

    profiles_path = tmp_path / "auth-profiles.json"
    profiles_path.write_text(json.dumps(auth_profiles))

    return OpenClawAuthConfig(
        config_path=str(config_path),
        auth_profiles_path=str(profiles_path),
    )


class TestResolutionChain:
    """Test the three-step resolution chain."""

    def test_resolves_via_auth_order(self, auth_dir: OpenClawAuthConfig):
        """Step 1: auth.order provides preferred profile ID."""
        cred = resolve_openclaw_credentials("anthropic", auth_dir)
        assert cred.profile_id == "anthropic:default"
        assert cred.auth_token == "sk-ant-oat01-fake-oat-token"

    def test_resolves_via_auth_profiles_metadata(self, auth_dir: OpenClawAuthConfig):
        """Step 2: auth.profiles metadata matches when auth.order has no entry."""
        cred = resolve_openclaw_credentials("openai", auth_dir)
        assert cred.profile_id == "openai:default"
        assert cred.api_key == "sk-openai-fake-key"

    def test_resolves_via_direct_scan(self, auth_dir: OpenClawAuthConfig):
        """Step 3: direct scan of auth-profiles.json when neither auth.order nor auth.profiles match."""
        cred = resolve_openclaw_credentials("groq", auth_dir)
        assert cred.profile_id == "groq:default"
        assert cred.api_key == "gsk-groq-fake-key"

    def test_resolves_with_empty_config(self, tmp_path: Path):
        """Step 3 fallback: works even with no openclaw.json."""
        profiles = {
            "version": 1,
            "profiles": {
                "deepseek:default": {
                    "type": "api_key",
                    "provider": "deepseek",
                    "key": "ds-fake-key",
                },
            },
        }
        profiles_path = tmp_path / "auth-profiles.json"
        profiles_path.write_text(json.dumps(profiles))

        config = OpenClawAuthConfig(
            config_path=str(tmp_path / "nonexistent.json"),
            auth_profiles_path=str(profiles_path),
        )
        cred = resolve_openclaw_credentials("deepseek", config)
        assert cred.api_key == "ds-fake-key"

    def test_missing_provider_raises(self, auth_dir: OpenClawAuthConfig):
        with pytest.raises(ValueError, match="No OpenClaw auth profile found for provider 'bedrock'"):
            resolve_openclaw_credentials("bedrock", auth_dir)


class TestProviderAliases:
    """Test provider name mapping (Hindsight → OpenClaw)."""

    def test_gemini_resolves_to_google(self, auth_dir: OpenClawAuthConfig):
        """Hindsight's 'gemini' should find OpenClaw's 'google' profile."""
        cred = resolve_openclaw_credentials("gemini", auth_dir)
        assert cred.profile_id == "google:default"
        assert cred.api_key == "gemini-fake-key"

    def test_resolve_provider_names(self):
        names = _resolve_provider_names("gemini")
        assert names == ["gemini", "google", "google-gemini-cli"]

    def test_resolve_provider_names_no_alias(self):
        names = _resolve_provider_names("openai")
        assert names == ["openai"]


class TestAnthropicOat:
    """Test Anthropic OAuth Access Token (OAT) detection."""

    def test_oat_detected(self, auth_dir: OpenClawAuthConfig):
        cred = resolve_openclaw_credentials("anthropic", auth_dir)
        assert cred.auth_token is not None
        assert cred.api_key is None
        assert "anthropic-beta" in cred.extra_headers
        assert cred.extra_headers["anthropic-beta"] == "oauth-2025-04-20"

    def test_standard_api_key_not_oat(self, tmp_path: Path):
        """Standard Anthropic API keys should use api_key, not auth_token."""
        profiles = {
            "version": 1,
            "profiles": {
                "anthropic:default": {
                    "type": "api_key",
                    "provider": "anthropic",
                    "key": "sk-ant-api03-standard-key",
                },
            },
        }
        profiles_path = tmp_path / "auth-profiles.json"
        profiles_path.write_text(json.dumps(profiles))
        config = OpenClawAuthConfig(
            config_path=str(tmp_path / "nonexistent.json"),
            auth_profiles_path=str(profiles_path),
        )

        cred = resolve_openclaw_credentials("anthropic", config)
        assert cred.api_key == "sk-ant-api03-standard-key"
        assert cred.auth_token is None
        assert cred.extra_headers == {}


class TestCodexOauth:
    """Test Codex OAuth JWT decoding."""

    def test_codex_account_id_decoded(self, auth_dir: OpenClawAuthConfig):
        cred = resolve_openclaw_credentials("openai-codex", auth_dir)
        assert cred.account_id == "acct-123"
        assert cred.api_key is not None

    def test_decode_codex_account_id(self):
        jwt = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "test-acct"}})
        assert _decode_codex_account_id(jwt) == "test-acct"

    def test_decode_codex_account_id_missing_claim(self):
        jwt = _make_jwt({"sub": "user-123"})
        assert _decode_codex_account_id(jwt) is None

    def test_decode_codex_account_id_invalid_jwt(self):
        assert _decode_codex_account_id("not-a-jwt") is None


class TestReload:
    """Test credential re-read (for 401 recovery)."""

    def test_reload_picks_up_updated_token(self, tmp_path: Path):
        profiles = {
            "version": 1,
            "profiles": {
                "openai:default": {"type": "api_key", "provider": "openai", "key": "old-key"},
            },
        }
        profiles_path = tmp_path / "auth-profiles.json"
        profiles_path.write_text(json.dumps(profiles))
        config = OpenClawAuthConfig(
            config_path=str(tmp_path / "nonexistent.json"),
            auth_profiles_path=str(profiles_path),
        )

        cred1 = resolve_openclaw_credentials("openai", config)
        assert cred1.api_key == "old-key"

        # Simulate OpenClaw refreshing the token
        profiles["profiles"]["openai:default"]["key"] = "new-key"
        profiles_path.write_text(json.dumps(profiles))

        cred2 = reload_credentials("openai", config)
        assert cred2.api_key == "new-key"


class TestAutoResolveInInit:
    """LLMProvider.__init__ auto-resolves when the env var is set.

    Covers callers that build LLMProvider directly (e.g. MemoryEngine) and
    therefore bypass from_env(). Without this, providers fall back to local
    CLI auth files that don't exist on a customer VPS.
    """

    @staticmethod
    def _patch_create(monkeypatch: pytest.MonkeyPatch) -> dict:
        """Capture create_llm_provider kwargs and skip real provider construction."""
        captured: dict = {}

        class _Stub:
            """Empty class so LLMProvider.__init__ can set attrs on it post-construction."""

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _Stub()

        monkeypatch.setattr("hindsight_api.engine.llm_wrapper.create_llm_provider", fake_create)
        return captured

    @staticmethod
    def _set_openclaw_env(monkeypatch: pytest.MonkeyPatch, auth_dir: OpenClawAuthConfig) -> None:
        monkeypatch.setenv("HINDSIGHT_API_LLM_AUTH_SOURCE", "openclaw")
        monkeypatch.setenv("HINDSIGHT_API_LLM_OPENCLAW_CONFIG_PATH", auth_dir.config_path)
        monkeypatch.setenv("HINDSIGHT_API_LLM_OPENCLAW_AUTH_PROFILES_PATH", auth_dir.auth_profiles_path)

    def test_codex_credentials_resolved_in_init(self, auth_dir, monkeypatch):
        from hindsight_api.engine.llm_wrapper import LLMProvider

        self._set_openclaw_env(monkeypatch, auth_dir)
        captured = self._patch_create(monkeypatch)

        provider = LLMProvider(provider="openai-codex", api_key="", base_url="", model="gpt-5.2-codex")

        assert provider._openclaw_credential is not None
        assert provider._openclaw_credential.account_id == "acct-123"
        assert captured["openclaw_account_id"] == "acct-123"
        assert captured["openclaw_access_token"].startswith("eyJ")

    def test_anthropic_oat_resolved_in_init(self, auth_dir, monkeypatch):
        from hindsight_api.engine.llm_wrapper import LLMProvider

        self._set_openclaw_env(monkeypatch, auth_dir)
        captured = self._patch_create(monkeypatch)

        provider = LLMProvider(provider="anthropic", api_key="", base_url="", model="claude-sonnet-4-5")

        assert provider._openclaw_credential is not None
        assert provider._openclaw_credential.auth_token == "sk-ant-oat01-fake-oat-token"
        assert captured["auth_token"] == "sk-ant-oat01-fake-oat-token"
        assert captured["extra_headers"]["anthropic-beta"] == "oauth-2025-04-20"

    def test_api_key_populated_when_caller_left_it_empty(self, auth_dir, monkeypatch):
        """Caller passes api_key='' for an api_key-style provider; we fill it from auth-profiles."""
        from hindsight_api.engine.llm_wrapper import LLMProvider

        self._set_openclaw_env(monkeypatch, auth_dir)
        captured = self._patch_create(monkeypatch)

        provider = LLMProvider(provider="openai", api_key="", base_url="", model="gpt-4o")

        assert provider.api_key == "sk-openai-fake-key"
        assert captured["api_key"] == "sk-openai-fake-key"

    def test_skipped_when_env_var_absent(self, monkeypatch):
        from hindsight_api.engine.llm_wrapper import LLMProvider

        monkeypatch.delenv("HINDSIGHT_API_LLM_AUTH_SOURCE", raising=False)
        captured = self._patch_create(monkeypatch)

        provider = LLMProvider(provider="openai", api_key="explicit-key", base_url="", model="gpt-4o")

        assert provider._openclaw_credential is None
        assert provider.api_key == "explicit-key"
        assert captured["api_key"] == "explicit-key"

    def test_skipped_when_env_var_is_env(self, monkeypatch):
        from hindsight_api.engine.llm_wrapper import LLMProvider

        monkeypatch.setenv("HINDSIGHT_API_LLM_AUTH_SOURCE", "env")
        captured = self._patch_create(monkeypatch)

        provider = LLMProvider(provider="openai", api_key="explicit-key", base_url="", model="gpt-4o")

        assert provider._openclaw_credential is None
        assert captured["api_key"] == "explicit-key"

    def test_explicit_credential_not_overridden(self, auth_dir, monkeypatch):
        """If a credential is passed in (e.g. by from_env), don't re-resolve."""
        from hindsight_api.engine.llm_wrapper import LLMProvider

        self._set_openclaw_env(monkeypatch, auth_dir)
        self._patch_create(monkeypatch)
        prebuilt = OpenClawCredential(api_key="prebuilt-key", profile_id="prebuilt:fixture", profile_type="api_key")

        provider = LLMProvider(
            provider="openai",
            api_key="prebuilt-key",
            base_url="",
            model="gpt-4o",
            _openclaw_credential=prebuilt,
        )

        assert provider._openclaw_credential is prebuilt
