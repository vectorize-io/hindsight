"""Every copied hook LLM registry must know every no-API-key provider.

The registries under ``hindsight-integrations/**/llm.py`` are copies of one table,
one per harness, plus two TypeScript twins in openclaw. A provider that authenticates
through a subscription CLI has no API key, so a registry that has not heard of it
refuses to start: ``HINDSIGHT_API_LLM_PROVIDER is set to "<id>" but
HINDSIGHT_API_LLM_API_KEY is not set``.

The test is parametrized over the family rather than pinned to one id on purpose.
It used to assert ``github-copilot`` alone, which is why adding ``cursor`` could
(and did) miss all six Python copies while updating both TypeScript ones — the
asymmetry a single-id test cannot see.
"""

from pathlib import Path
from runpy import run_path

import pytest

#: Providers that authenticate via a subscription CLI or run locally — no API key.
#: Mirrors ``_PROVIDERS_WITHOUT_API_KEY`` in ``engine/provider_auth.py``, limited to
#: the ids the hook registries actually offer (they do not carry the server-only
#: backends like ``bedrock`` or ``litellm``).
NO_KEY_PROVIDERS = ["ollama", "openai-codex", "claude-code", "cursor", "github-copilot"]

DEFAULT_MODELS = {
    "ollama": "gemma3:12b",
    "openai-codex": "gpt-5.4-mini",
    "claude-code": "claude-sonnet-4-5-20250929",
    "cursor": "auto",
    "github-copilot": "gpt-5.6-terra",
}


def _registry_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "hindsight-integrations").glob("**/llm.py"))
    assert files, "no hook LLM registries found"
    return files


@pytest.mark.parametrize("provider", NO_KEY_PROVIDERS)
def test_all_copied_llm_registries_allow_no_key_providers(provider, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", provider)
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", DEFAULT_MODELS[provider])
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)

    for path in _registry_files():
        namespace = run_path(str(path))
        no_key_required = namespace.get("NO_KEY_REQUIRED")
        assert isinstance(no_key_required, set), f"{path} has no no-key provider registry"
        assert provider in no_key_required, f"{path} does not register {provider}"
        detected = namespace["detect_llm_config"]({})
        assert detected["provider"] == provider
        assert detected["api_key"] == ""
        assert detected["model"] == DEFAULT_MODELS[provider]


@pytest.mark.parametrize("provider", NO_KEY_PROVIDERS)
def test_all_copied_llm_registries_list_no_key_providers_for_detection(provider):
    """A provider absent from PROVIDER_DETECTION is never auto-selected, only accepted."""
    for path in _registry_files():
        namespace = run_path(str(path))
        names = {entry["name"] for entry in namespace["PROVIDER_DETECTION"]}
        assert provider in names, f"{path} does not list {provider} in PROVIDER_DETECTION"


@pytest.mark.parametrize("provider", ["claude-code", "cursor", "github-copilot"])
def test_openclaw_no_key_registries_include_subscription_providers(provider):
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "hindsight-integrations" / "openclaw" / "src" / "index.ts",
        root / "hindsight-integrations" / "openclaw" / "src" / "setup-lib.ts",
    ]

    for path in paths:
        assert f'"{provider}"' in path.read_text(encoding="utf-8"), f"{path} does not register {provider}"
