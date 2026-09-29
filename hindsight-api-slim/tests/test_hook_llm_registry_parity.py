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

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.provider_auth import _PROVIDERS_WITHOUT_API_KEY

#: Server-only backends: no API key either, but a coding-agent hook never runs the
#: model locally through them, so the hook registries deliberately do not offer them.
#: Every other no-key provider MUST appear in all of them. Derived rather than typed
#: out, so adding the next subscription provider to ``_PROVIDERS_WITHOUT_API_KEY``
#: fails here until its six registries are updated too — the failure mode that let
#: ``cursor`` reach all six TypeScript-updated-but-Python-forgotten registries.
HOOK_EXEMPT_PROVIDERS = frozenset(
    {
        "lmstudio",
        "llamacpp",
        "vertexai",
        "bedrock",
        "litellm",
        "litellmrouter",
        "nous",
        "xai-oauth",
        "mock",
        "none",
    }
)

NO_KEY_PROVIDERS = sorted(_PROVIDERS_WITHOUT_API_KEY - HOOK_EXEMPT_PROVIDERS)


def test_exemption_list_is_not_stale():
    """An exempted id that no longer exists hides a provider the registries must carry."""
    unknown = HOOK_EXEMPT_PROVIDERS - _PROVIDERS_WITHOUT_API_KEY
    assert not unknown, f"exempted providers that are no longer no-key: {sorted(unknown)}"


def _registry_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "hindsight-integrations").glob("**/llm.py"))
    assert files, "no hook LLM registries found"
    return files


@pytest.mark.parametrize("provider", NO_KEY_PROVIDERS)
def test_all_copied_llm_registries_allow_no_key_providers(provider, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", provider)
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", PROVIDER_DEFAULT_MODELS[provider])
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)

    for path in _registry_files():
        namespace = run_path(str(path))
        no_key_required = namespace.get("NO_KEY_REQUIRED")
        assert isinstance(no_key_required, set), f"{path} has no no-key provider registry"
        assert provider in no_key_required, f"{path} does not register {provider}"
        detected = namespace["detect_llm_config"]({})
        assert detected["provider"] == provider
        assert detected["api_key"] == ""
        assert detected["model"] == PROVIDER_DEFAULT_MODELS[provider]


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
