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
