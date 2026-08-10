"""Unit tests for benchmark role LLM config (no network, no credentials)."""

import pytest

from benchmarks.common.llm_role_config import ANSWER_ROLE, JUDGE_ROLE, build_role_llm_config


def test_role_vars_win_over_shared_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "shared-model")
    monkeypatch.setenv("HINDSIGHT_API_ANSWER_LLM_MODEL", "answer-model")

    config = build_role_llm_config(ANSWER_ROLE)

    assert config.provider == "openai"
    assert config.model == "answer-model"


def test_falls_back_to_shared_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HINDSIGHT_API_JUDGE_LLM_MODEL", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "shared-model")

    config = build_role_llm_config(JUDGE_ROLE)

    assert config.model == "shared-model"


@pytest.mark.parametrize("role", [ANSWER_ROLE, JUDGE_ROLE])
def test_vertexai_project_id_reaches_the_config(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the scheduled LoComo run died before its first question.

    Each benchmark role built its LLMConfig from provider/api_key/base_url/model
    alone. LLMConfig does not read the environment for provider-specific
    settings, so every Vertex AI value was dropped and the constructor raised
    "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required" — even though the
    workflow exports it. Vertex AI authenticates with a project rather than an
    api_key, so this asserts the passthrough, not just that construction works.
    """
    monkeypatch.setenv(f"HINDSIGHT_API_{role}_LLM_PROVIDER", "vertexai")
    monkeypatch.setenv(f"HINDSIGHT_API_{role}_LLM_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setenv("HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID", "test-project")
    monkeypatch.setenv("HINDSIGHT_API_LLM_VERTEXAI_REGION", "europe-west4")

    config = build_role_llm_config(role)

    assert config.provider == "vertexai"
    # LLMConfig strips the google/ prefix for the native Vertex SDK.
    assert config.model == "gemini-2.5-flash"


def test_vertexai_without_project_id_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The passthrough must not paper over genuinely missing configuration."""
    monkeypatch.setenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", "vertexai")
    monkeypatch.delenv("HINDSIGHT_API_ANSWER_LLM_VERTEXAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID", raising=False)

    with pytest.raises(ValueError, match="VERTEXAI_PROJECT_ID"):
        build_role_llm_config(ANSWER_ROLE)
