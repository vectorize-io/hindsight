"""Tests for ``CODEX_HOME`` resolution of the Codex ``auth.json`` location.

Codex stores its OAuth credentials under a configurable home directory. The
canonical ``@openai/codex`` CLI honors the ``CODEX_HOME`` environment variable
and falls back to ``~/.codex``. Hindsight's Codex auth/LLM/embeddings paths
must resolve the same way so that a user who relocates ``CODEX_HOME`` is still
authenticated.
"""

import json
from pathlib import Path

from hindsight_api.engine.providers.codex_auth import (
    CodexAuthManager,
    default_codex_auth_file,
)
from hindsight_api.engine.providers.codex_llm import CodexLLM


def _write_auth(auth_dir: Path, access_token: str = "at-test") -> Path:
    """Write a minimal chatgpt-mode auth.json under ``auth_dir``."""
    auth_dir.mkdir(parents=True, exist_ok=True)
    auth_file = auth_dir / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "rt-test",
                    "account_id": "acct-test",
                },
            }
        )
    )
    return auth_file


# ---------------------------------------------------------------------------
# default_codex_auth_file()
# ---------------------------------------------------------------------------


def test_default_auth_file_falls_back_to_home_codex_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert default_codex_auth_file() == tmp_path / ".codex" / "auth.json"


def test_default_auth_file_honors_codex_home_when_set(tmp_path, monkeypatch):
    codex_home = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert default_codex_auth_file() == codex_home / "auth.json"


def test_default_auth_file_empty_codex_home_falls_back(tmp_path, monkeypatch):
    """An empty ``CODEX_HOME`` is treated as unset (matches shell semantics)."""
    monkeypatch.setenv("CODEX_HOME", "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert default_codex_auth_file() == tmp_path / ".codex" / "auth.json"


def test_default_auth_file_resolved_lazily(tmp_path, monkeypatch):
    """The env var is read on each call, not cached at import time."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "a"))
    assert default_codex_auth_file() == tmp_path / "a" / "auth.json"

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "b"))
    assert default_codex_auth_file() == tmp_path / "b" / "auth.json"


# ---------------------------------------------------------------------------
# CodexAuthManager.from_file() — honors CODEX_HOME by default
# ---------------------------------------------------------------------------


def test_auth_manager_from_file_uses_codex_home(tmp_path, monkeypatch):
    codex_home = tmp_path / "custom-codex"
    _write_auth(codex_home, access_token="at-from-codex-home")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    mgr = CodexAuthManager.from_file()

    assert mgr.access_token == "at-from-codex-home"
    assert mgr._auth_file == codex_home / "auth.json"


# ---------------------------------------------------------------------------
# CodexLLM — loads credentials from CODEX_HOME
# ---------------------------------------------------------------------------


def test_codex_llm_loads_from_codex_home(tmp_path, monkeypatch):
    codex_home = tmp_path / "custom-codex"
    _write_auth(codex_home, access_token="at-llm")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    llm = CodexLLM(
        provider="codex",
        api_key="ignored",
        base_url="",
        model="gpt-5-codex",
    )

    assert llm.access_token == "at-llm"
    assert llm._auth_file == codex_home / "auth.json"


# ---------------------------------------------------------------------------
# Explicit ``codex_home`` — per-provider credential store (issue #3793)
# ---------------------------------------------------------------------------


def test_default_auth_file_explicit_home_wins_over_env(tmp_path, monkeypatch):
    """An explicit ``codex_home`` overrides the process-wide ``CODEX_HOME``."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "env-home"))

    assert default_codex_auth_file(str(tmp_path / "explicit")) == tmp_path / "explicit" / "auth.json"


def test_default_auth_file_empty_explicit_home_falls_back_to_env(tmp_path, monkeypatch):
    """An empty string is treated as unset, like the env var itself."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "env-home"))

    assert default_codex_auth_file("") == tmp_path / "env-home" / "auth.json"


def test_codex_llm_loads_from_explicit_codex_home(tmp_path, monkeypatch):
    """``codex_home=`` selects the auth store even when ``CODEX_HOME`` differs."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "env-home"))
    profile = tmp_path / "profile-b"
    _write_auth(profile, access_token="at-profile-b")

    llm = CodexLLM(
        provider="codex",
        api_key="ignored",
        base_url="",
        model="gpt-5-codex",
        codex_home=str(profile),
    )

    assert llm.access_token == "at-profile-b"
    assert llm._auth_file == profile / "auth.json"


def test_two_codex_llms_hold_independent_profiles(tmp_path, monkeypatch):
    """Two instances in one process authenticate as two different accounts.

    This is what makes a Codex-to-Codex failover chain meaningful: without a
    per-instance store both members would read the same ``auth.json`` and a
    quota-exhausted profile would just be retried.
    """
    monkeypatch.delenv("CODEX_HOME", raising=False)
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_auth(primary, access_token="at-primary")
    _write_auth(secondary, access_token="at-secondary")

    def _build(home):
        return CodexLLM(
            provider="codex",
            api_key="ignored",
            base_url="",
            model="gpt-5-codex",
            codex_home=str(home),
        )

    a, b = _build(primary), _build(secondary)

    assert (a.access_token, b.access_token) == ("at-primary", "at-secondary")
    assert a._auth_file != b._auth_file
