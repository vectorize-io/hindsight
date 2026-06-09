"""LLM provider detection for Hindsight's fact extraction.

Port of: detectLLMConfig() in index.js

When running hindsight-embed locally (daemon mode), it needs an LLM to
extract facts from retained conversations. This module detects the LLM
config using the same priority chain as Openclaw:

  1. HINDSIGHT_API_LLM_* environment variables (highest priority)
  2. Plugin config (llmProvider, llmModel, llmApiKeyEnv)
  3. Auto-detect from standard provider env vars
  4. External API mode (server-side LLM, no local config needed)
"""

import os

# Provider detection table — same order as Openclaw
PROVIDER_DETECTION = [
    {"name": "openai", "key_env": "OPENAI_API_KEY"},
    {"name": "anthropic", "key_env": "ANTHROPIC_API_KEY"},
    {"name": "gemini", "key_env": "GEMINI_API_KEY"},
    {"name": "groq", "key_env": "GROQ_API_KEY"},
    {"name": "ollama", "key_env": ""},
    {"name": "openai-codex", "key_env": ""},
    {"name": "claude-code", "key_env": ""},
]

# Providers that don't require an API key
NO_KEY_REQUIRED = {"ollama", "openai-codex", "claude-code"}


def _find_provider(name):
    """Find a provider entry by name."""
    for p in PROVIDER_DETECTION:
        if p["name"] == name:
            return p
    return None


def detect_llm_config(config: dict) -> dict:
    """Detect LLM configuration.

    Returns dict with: provider, api_key, model, base_url, source.
    Returns None values for external API mode (server handles LLM).
    Raises RuntimeError if no configuration found and not in external API mode.
    """
    override_provider = os.environ.get("HINDSIGHT_API_LLM_PROVIDER")
    override_model = os.environ.get("HINDSIGHT_API_LLM_MODEL")
    override_key = os.environ.get("HINDSIGHT_API_LLM_API_KEY")
    override_base_url = os.environ.get("HINDSIGHT_API_LLM_BASE_URL")

    # Priority 1: HINDSIGHT_API_LLM_PROVIDER env var
    if override_provider:
        if not override_key and override_provider not in NO_KEY_REQUIRED:
            raise RuntimeError(
                f'HINDSIGHT_API_LLM_PROVIDER is set to "{override_provider}" but HINDSIGHT_API_LLM_API_KEY is not set.'
            )
        pinfo = _find_provider(override_provider)
        return {
            "provider": override_provider,
            "api_key": override_key or "",
            "model": override_model,
            "base_url": override_base_url,
            "source": "HINDSIGHT_API_LLM_PROVIDER override",
        }

    # Priority 2: Plugin config llmProvider/llmModel
    cfg_provider = config.get("llmProvider")
    if cfg_provider:
        pinfo = _find_provider(cfg_provider)
        api_key = ""
        key_env_name = config.get("llmApiKeyEnv")
        if key_env_name:
            api_key = os.environ.get(key_env_name, "")
        elif pinfo and pinfo["key_env"]:
            api_key = os.environ.get(pinfo["key_env"], "")

        if not api_key and cfg_provider not in NO_KEY_REQUIRED:
            key_source = key_env_name or (pinfo["key_env"] if pinfo else "unknown")
            raise RuntimeError(
                f'Plugin config llmProvider is "{cfg_provider}" but no API key found. Expected env var: {key_source}'
            )
        return {
            "provider": cfg_provider,
            "api_key": api_key,
            "model": config.get("llmModel") or override_model,
            "base_url": override_base_url,
            "source": "plugin config",
        }

    # Priority 3: Auto-detect from standard provider env vars
    for pinfo in PROVIDER_DETECTION:
        if pinfo["name"] in NO_KEY_REQUIRED:
            continue  # Must be explicitly requested
        if not pinfo["key_env"]:
            continue
        api_key = os.environ.get(pinfo["key_env"], "")
        if api_key:
            return {
                "provider": pinfo["name"],
                "api_key": api_key,
                "model": override_model,
                "base_url": override_base_url,
                "source": f"auto-detected from {pinfo['key_env']}",
            }

    # Priority 4: External API mode — server handles LLM
    if config.get("hindsightApiUrl"):
        return {
            "provider": None,
            "api_key": None,
            "model": None,
            "base_url": None,
            "source": "external-api-mode-no-llm",
        }

    raise RuntimeError(
        "No LLM configuration found for Hindsight.\n\n"
        "Option 1: Set a standard provider API key (auto-detect):\n"
        "  export OPENAI_API_KEY=sk-your-key\n"
        "  export ANTHROPIC_API_KEY=your-key\n\n"
        "Option 2: Override with Hindsight-specific env vars:\n"
        "  export HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "  export HINDSIGHT_API_LLM_API_KEY=sk-your-key\n\n"
        "Option 3: Use an external Hindsight API (server-side LLM):\n"
        "  Set hindsightApiUrl in settings.json or HINDSIGHT_API_URL env var\n\n"
        "The model will be selected automatically by Hindsight. To override: export HINDSIGHT_API_LLM_MODEL=your-model"
    )


def get_llm_env_vars(llm_config: dict) -> dict:
    """Build environment variables for hindsight-embed daemon from LLM config.

    These are passed to the daemon subprocess so it knows which LLM to use
    for fact extraction.
    """
    env = {}
    if llm_config.get("provider"):
        env["HINDSIGHT_API_LLM_PROVIDER"] = llm_config["provider"]
    if llm_config.get("api_key"):
        env["HINDSIGHT_API_LLM_API_KEY"] = llm_config["api_key"]
    if llm_config.get("model"):
        env["HINDSIGHT_API_LLM_MODEL"] = llm_config["model"]
    if llm_config.get("base_url"):
        env["HINDSIGHT_API_LLM_BASE_URL"] = llm_config["base_url"]
    return env
