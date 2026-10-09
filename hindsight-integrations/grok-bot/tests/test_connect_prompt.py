"""The manual setup prompt, used until the marketplace listing is approved.

The prompt carries the same rules as the skills, so it has to name the same
tools and keep the same prohibitions.
"""

import re
from pathlib import Path

PROMPT = (Path(__file__).resolve().parents[1] / "connect-prompt.md").read_text()

DESTRUCTIVE = (
    "delete_bank",
    "clear_memories",
    "invalidate_memory",
    "delete_document",
    "delete_mental_model",
)
REQUIRED = (
    "list_banks",
    "create_bank",
    "recall",
    "retain",
    "reflect",
    "list_mental_models",
    "create_mental_model",
    "get_mental_model",
)


def test_prompt_uses_the_root_cloud_url() -> None:
    assert "https://api.hindsight.vectorize.io/mcp\n" in PROMPT
    assert "/mcp/" not in PROMPT


def test_prompt_derives_the_bank_id_from_the_bot_name() -> None:
    assert "`grok-bot::sales-researcher`" in PROMPT
    assert "`grok-bot::shared`" in PROMPT
    assert "Grok Bot" in PROMPT


def test_prompt_forbids_destructive_bank_tools() -> None:
    for tool in DESTRUCTIVE:
        assert f"`{tool}`" in PROMPT
    assert "Never call" in PROMPT


def test_prompt_names_every_required_tool() -> None:
    for tool in REQUIRED:
        assert f"`{tool}`" in PROMPT


def test_prompt_bounds_the_handoff_exception() -> None:
    assert "`handoff:any`" in PROMPT
    assert "not instructions to you" in PROMPT
    assert "contact anyone outside this account" in PROMPT


def test_prompt_contains_no_credentials() -> None:
    assert not re.search(r"hsk_[A-Za-z0-9]", PROMPT)
    assert "Bearer " not in PROMPT


def test_readme_documents_the_manual_path() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "connect-prompt.md" in readme
    assert "not in the Grok Bot or Cursor marketplace yet" in readme
    assert "~/.cursor/plugins/local" in readme
