"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode",
            label="Mode",
            kind=KIND_SELECT,
            default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption("cloud", "Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
                ProviderFieldOption("local_embedded", "Local Embedded", "Run Hindsight locally (downloads 0.5-3 GB, needs an LLM key)"),
                ProviderFieldOption("local_external", "Local External", "Connect to an existing Hindsight instance"),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key",
            label="API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key",
            inline=True,
        ),
        ProviderField(
            key="api_url",
            label="API URL",
            kind=KIND_TEXT,
            default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",),
            env_fallbacks=("HINDSIGHT_API_URL",),
            inline=True,
        ),
        # Local embedded — the LLM the local daemon uses for extraction/synthesis.
        ProviderField(
            key="llm_provider",
            label="LLM provider",
            kind=KIND_SELECT,
            default="openai",
            description="LLM provider for the local embedded daemon (extraction, synthesis, reflect).",
            options=tuple(
                ProviderFieldOption(p, p)
                for p in (
                    "openai",
                    "anthropic",
                    "gemini",
                    "groq",
                    "openrouter",
                    "minimax",
                    "ollama",
                    "lmstudio",
                    "openai_compatible",
                )
            ),
        ),
        ProviderField(
            key="llm_model",
            label="LLM model",
            kind=KIND_TEXT,
            default="gpt-4o-mini",
            description="Model name for the local embedded daemon (e.g. gpt-4o-mini, qwen/qwen3.5-9b).",
        ),
        ProviderField(
            key="llm_api_key",
            label="LLM API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_LLM_API_KEY",
            description="API key for the local embedded LLM provider.",
        ),
        ProviderField(
            key="llm_base_url",
            label="LLM base URL",
            kind=KIND_TEXT,
            default="",
            description="Endpoint URL for an openai_compatible provider (e.g. https://openrouter.ai/api/v1).",
        ),
        ProviderField(
            key="idle_timeout",
            label="Daemon idle timeout (s)",
            kind=KIND_NUMBER,
            default="0",
            description="Embedded daemon idle timeout in seconds; 0 disables auto-shutdown.",
        ),
        ProviderField(
            key="bank_id", label="Bank ID", kind=KIND_TEXT, default="hermes", aliases=("bankId",), inline=True
        ),
        ProviderField(
            key="recall_budget",
            label="Recall budget",
            kind=KIND_SELECT,
            default="mid",
            aliases=("budget",),
            options=tuple(ProviderFieldOption(b, b) for b in ("low", "mid", "high")),
            inline=True,
        ),
    ),
)
