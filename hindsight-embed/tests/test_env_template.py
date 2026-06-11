"""Tests for env_template — seeding embed configs from .env.example."""

from pathlib import Path

from hindsight_embed.env_template import load_template, render_config


def _active_keys(text: str) -> dict[str, str]:
    """Parse a rendered config the same way the daemon loader does."""
    config: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def test_user_values_are_active():
    text = render_config(
        {
            "HINDSIGHT_API_LLM_PROVIDER": "anthropic",
            "HINDSIGHT_API_LLM_API_KEY": "sk-secret",
            "HINDSIGHT_API_LLM_MODEL": "gpt-4o-mini",
        }
    )
    active = _active_keys(text)
    assert active["HINDSIGHT_API_LLM_PROVIDER"] == "anthropic"
    assert active["HINDSIGHT_API_LLM_API_KEY"] == "sk-secret"
    assert active["HINDSIGHT_API_LLM_MODEL"] == "gpt-4o-mini"


def test_only_user_values_are_active():
    """Backwards-compatibility guarantee: the active (uncommented) keys must be
    EXACTLY the values passed in — nothing more. The old bare config wrote only
    the user's keys; everything else in the template must stay commented so the
    daemon sees the identical active config and never inherits the template's
    api-server defaults (port 8888, OpenAI base url, gpt-4o-mini, host)."""
    values = {
        "HINDSIGHT_API_LLM_PROVIDER": "anthropic",
        "HINDSIGHT_API_LLM_API_KEY": "sk-secret",
        "HINDSIGHT_API_LLM_MODEL": "gpt-4o-mini",
    }
    assert _active_keys(render_config(values)) == values

    # Minimal case: provider only — no other key leaks in active.
    assert _active_keys(render_config({"HINDSIGHT_API_LLM_PROVIDER": "gemini"})) == {
        "HINDSIGHT_API_LLM_PROVIDER": "gemini"
    }


def test_documentation_is_preserved():
    """Commented example blocks survive so the file is self-documenting."""
    text = render_config({"HINDSIGHT_API_LLM_PROVIDER": "openai"})
    assert "# Example: Anthropic Claude configuration" in text
    assert "Supported providers:" in text


def test_unknown_keys_are_appended():
    text = render_config({"HINDSIGHT_API_LLM_PROVIDER": "openai", "KEY": "value"})
    assert _active_keys(text)["KEY"] == "value"


def test_falls_back_to_minimal_without_template():
    text = render_config({"HINDSIGHT_API_LLM_PROVIDER": "openai"}, template="")
    assert _active_keys(text) == {"HINDSIGHT_API_LLM_PROVIDER": "openai"}


def test_bundled_template_matches_repo_root():
    """The package ships a copy of the repo-root .env.example; keep them in
    sync so installed users get the same documented options. Skips when the
    repo root isn't present (installed-package context)."""
    bundled = load_template()
    assert bundled is not None
    repo_root_example = Path(__file__).resolve().parents[2] / ".env.example"
    if not repo_root_example.is_file():
        return
    assert bundled == repo_root_example.read_text(), (
        "hindsight_embed/env.example is out of sync with the repo-root .env.example — re-copy it."
    )
