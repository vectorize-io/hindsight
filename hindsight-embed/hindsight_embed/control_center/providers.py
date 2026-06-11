"""LLM provider catalog for the control center's config wizard.

Just the list of providers shown in the dropdown; the daemon owns actual
provider validation and defaults.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """A provider the wizard can configure."""

    id: str
    label: str
    needs_api_key: bool
    default_base_url: str | None = None


# Ordered for display in the wizard dropdown. Mirrors the providers that
# hindsight-api's PROVIDER_DEFAULT_MODELS supports.
PROVIDER_CATALOG: tuple[ProviderInfo, ...] = (
    ProviderInfo("openai", "OpenAI", needs_api_key=True, default_base_url="https://api.openai.com/v1"),
    ProviderInfo("anthropic", "Anthropic", needs_api_key=True, default_base_url="https://api.anthropic.com"),
    ProviderInfo("gemini", "Google Gemini", needs_api_key=True),
    ProviderInfo("groq", "Groq", needs_api_key=True, default_base_url="https://api.groq.com/openai/v1"),
    ProviderInfo("ollama", "Ollama (local)", needs_api_key=False, default_base_url="http://localhost:11434"),
    ProviderInfo("lmstudio", "LM Studio (local)", needs_api_key=False, default_base_url="http://localhost:1234/v1"),
    ProviderInfo("vertexai", "Google Vertex AI", needs_api_key=False),
    ProviderInfo("deepseek", "DeepSeek", needs_api_key=True),
    ProviderInfo("minimax", "MiniMax", needs_api_key=True),
    ProviderInfo("zai", "Z.ai", needs_api_key=True),
    ProviderInfo("volcano", "Volcano", needs_api_key=True),
)
