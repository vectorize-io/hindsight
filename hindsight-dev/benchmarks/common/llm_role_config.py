"""Build the benchmark-role LLM configs (answer generator, judge).

Benchmarks configure two LLMs independently of the API's own config system: the
answer generator (``HINDSIGHT_API_ANSWER_LLM_*``) and the judge
(``HINDSIGHT_API_JUDGE_LLM_*``), each falling back to the shared
``HINDSIGHT_API_LLM_*`` values.

Each call site used to build its own ``LLMConfig`` from four env vars —
provider, api_key, base_url, model — and nothing else. ``LLMConfig.__init__``
deliberately does not read the environment for provider-specific settings (its
docstring says the caller resolves them; ``LLMConfig.from_env`` is where the API
does it), so those hand-rolled configs silently dropped every Vertex AI setting.
With ``provider=vertexai`` the constructor then raised
"HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required" before a single question ran,
which is what broke the scheduled LoComo benchmark.
"""

from __future__ import annotations

import os

from hindsight_api.engine.llm_wrapper import LLMConfig

# Roles a benchmark can configure separately from the engine's own LLM.
ANSWER_ROLE = "ANSWER"
JUDGE_ROLE = "JUDGE"


def _role_env(role: str, suffix: str, default: str = "") -> str:
    """Read ``HINDSIGHT_API_<ROLE>_LLM_<SUFFIX>``, falling back to the shared var."""
    return os.getenv(
        f"HINDSIGHT_API_{role}_LLM_{suffix}",
        os.getenv(f"HINDSIGHT_API_LLM_{suffix}", default),
    )


def build_role_llm_config(
    role: str,
    *,
    default_model: str = "gpt-4o-mini",
    reasoning_effort: str = "high",
) -> LLMConfig:
    """Build the ``LLMConfig`` for a benchmark role.

    Args:
        role: ``ANSWER_ROLE`` or ``JUDGE_ROLE`` — selects the env var prefix.
        default_model: Model to use when neither the role nor the shared var is set.
        reasoning_effort: Passed straight through to ``LLMConfig``.

    Returns:
        A config carrying the provider-specific settings as well as the common
        ones, so providers that need more than an api_key (currently Vertex AI)
        are usable for benchmark roles.
    """
    return LLMConfig(
        provider=_role_env(role, "PROVIDER", "openai"),
        api_key=_role_env(role, "API_KEY"),
        base_url=_role_env(role, "BASE_URL"),
        model=_role_env(role, "MODEL", default_model),
        reasoning_effort=reasoning_effort,
        # Vertex AI authenticates with a project/region/service account rather
        # than an api_key, and LLMConfig requires the project id up front.
        vertexai_project_id=_role_env(role, "VERTEXAI_PROJECT_ID") or None,
        vertexai_region=_role_env(role, "VERTEXAI_REGION") or None,
        vertexai_service_account_key=_role_env(role, "VERTEXAI_SERVICE_ACCOUNT_KEY") or None,
    )
