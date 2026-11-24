"""A/B switches reach the eval server, with or without an LLM."""

from __future__ import annotations

import pytest

from hindsight_system_evals.server import provider_environment


@pytest.mark.parametrize("provider", ["none", "gemini"])
def test_eval_set_switches_reach_the_server(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """``none`` used to return before forwarding them, so a no-LLM reranker A/B ran the default."""
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_MODEL", "m")
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_API_KEY", "k")
    monkeypatch.setenv("HINDSIGHT_EVAL_SET_RERANKER_PROVIDER", "rrf")

    assert provider_environment()["HINDSIGHT_API_RERANKER_PROVIDER"] == "rrf"


@pytest.mark.parametrize("provider", ["cursor", "claude-code", "ollama"])
def test_keyless_providers_start_without_a_key(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    """The server accepts these with no key (issue #6831), so the runner must too."""
    for name in ("HINDSIGHT_EVAL_LLM_API_KEY", "HINDSIGHT_API_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_MODEL", "m")

    assert "HINDSIGHT_API_LLM_API_KEY" not in provider_environment()


@pytest.mark.parametrize("provider", ["gemini", "mock"])
def test_keyed_providers_and_the_stub_still_refuse(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    for name in ("HINDSIGHT_EVAL_LLM_API_KEY", "HINDSIGHT_API_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_PROVIDER", provider)
    monkeypatch.setenv("HINDSIGHT_EVAL_LLM_MODEL", "m")

    with pytest.raises(RuntimeError, match="real model"):
        provider_environment()
