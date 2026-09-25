"""The connect prompt is what users paste into Muse; keep it consistent with the preflight and README."""

from __future__ import annotations

import re
from pathlib import Path

from hindsight_meta_muse.preflight import REQUIRED_TOOLS, ROOT_URL_TOOLS

INTEGRATION_DIR = Path(__file__).resolve().parent.parent
PROMPT = (INTEGRATION_DIR / "connect-prompt.md").read_text()


def test_prompt_uses_the_root_cloud_url() -> None:
    # The root URL is multi-bank: Muse keeps its own `muse` bank and reads the others.
    # A bank-scoped URL (/mcp/<bank>/) would hand it exactly one, which is the fallback, not the default.
    assert "https://api.hindsight.vectorize.io/mcp\n" in PROMPT
    assert "/mcp/" not in PROMPT


def test_prompt_gives_muse_its_own_bank() -> None:
    assert "`muse`" in PROMPT


def test_prompt_forbids_destructive_bank_tools() -> None:
    # The root URL exposes bank management, so the prompt is what keeps Muse off the destructive tools.
    for tool in ("delete_bank", "clear_memories", "invalidate_memory"):
        assert f"`{tool}`" in PROMPT
    assert "Never call" in PROMPT


def test_prompt_names_every_required_tool() -> None:
    for tool in REQUIRED_TOOLS + ROOT_URL_TOOLS:
        assert f"`{tool}`" in PROMPT


def test_prompt_contains_no_credentials() -> None:
    assert not re.search(r"hsk_[A-Za-z0-9]", PROMPT)
    assert "Bearer " not in PROMPT


def test_readme_embeds_the_prompt_file() -> None:
    readme = (INTEGRATION_DIR / "README.md").read_text()
    assert "connect-prompt.md" in readme
    assert "Hindsight Cloud" in readme
