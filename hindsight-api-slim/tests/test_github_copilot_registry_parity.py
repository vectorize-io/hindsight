from pathlib import Path
from runpy import run_path


def test_all_copied_llm_registries_allow_github_copilot_without_an_api_key(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    registry_files = sorted((root / "hindsight-integrations").glob("**/llm.py"))

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "github-copilot")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "gpt-5.6-terra")
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)

    assert registry_files
    for path in registry_files:
        namespace = run_path(str(path))
        no_key_required = namespace.get("NO_KEY_REQUIRED")
        assert isinstance(no_key_required, set), f"{path} has no no-key provider registry"
        assert "github-copilot" in no_key_required, f"{path} does not register github-copilot"
        detected = namespace["detect_llm_config"]({})
        assert detected["provider"] == "github-copilot"
        assert detected["api_key"] == ""
        assert detected["model"] == "gpt-5.6-terra"


def test_openclaw_no_key_registries_include_github_copilot():
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "hindsight-integrations" / "openclaw" / "src" / "index.ts",
        root / "hindsight-integrations" / "openclaw" / "src" / "setup-lib.ts",
    ]

    for path in paths:
        assert '"github-copilot"' in path.read_text(encoding="utf-8")
