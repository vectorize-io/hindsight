"""Tests for install.py — Roo Code Hindsight integration installer."""

import json
from pathlib import Path

import pytest

from install import build_mcp_entry, get_roo_dir, install_mcp, install_rules


# ---------------------------------------------------------------------------
# build_mcp_entry
# ---------------------------------------------------------------------------


def test_mcp_entry_url_includes_mcp_path() -> None:
    entry = build_mcp_entry("http://localhost:8888")
    assert entry["url"] == "http://localhost:8888/mcp"


def test_mcp_entry_trailing_slash_stripped() -> None:
    entry = build_mcp_entry("http://localhost:8888/")
    assert entry["url"] == "http://localhost:8888/mcp"


def test_mcp_entry_always_allow_tools() -> None:
    entry = build_mcp_entry("http://localhost:8888")
    assert "recall" in entry["alwaysAllow"]
    assert "retain" in entry["alwaysAllow"]


# ---------------------------------------------------------------------------
# get_roo_dir
# ---------------------------------------------------------------------------


def test_get_roo_dir_project(tmp_path: Path) -> None:
    roo = get_roo_dir(tmp_path, global_install=False)
    assert roo == tmp_path / ".roo"


def test_get_roo_dir_global(tmp_path: Path) -> None:
    roo = get_roo_dir(tmp_path, global_install=True)
    assert roo == Path.home() / ".roo"


# ---------------------------------------------------------------------------
# install_mcp — fresh install
# ---------------------------------------------------------------------------


def test_fresh_install_creates_mcp_json(tmp_project: Path, hindsight_url: str) -> None:
    roo_dir = tmp_project / ".roo"
    install_mcp(roo_dir, hindsight_url)

    mcp_path = roo_dir / "mcp.json"
    assert mcp_path.exists()

    data = json.loads(mcp_path.read_text())
    assert "hindsight" in data["mcpServers"]
    assert data["mcpServers"]["hindsight"]["url"] == f"{hindsight_url}/mcp"


def test_fresh_install_creates_roo_dir(tmp_project: Path, hindsight_url: str) -> None:
    roo_dir = tmp_project / ".roo"
    assert not roo_dir.exists()
    install_mcp(roo_dir, hindsight_url)
    assert roo_dir.is_dir()


# ---------------------------------------------------------------------------
# install_mcp — merge with existing servers
# ---------------------------------------------------------------------------


def test_merge_preserves_existing_servers(
    project_with_existing_mcp: Path, hindsight_url: str
) -> None:
    roo_dir = project_with_existing_mcp / ".roo"
    install_mcp(roo_dir, hindsight_url)

    data = json.loads((roo_dir / "mcp.json").read_text())
    # Both servers present
    assert "other-server" in data["mcpServers"]
    assert "hindsight" in data["mcpServers"]


def test_merge_does_not_duplicate(
    project_with_existing_mcp: Path, hindsight_url: str
) -> None:
    roo_dir = project_with_existing_mcp / ".roo"
    install_mcp(roo_dir, hindsight_url)
    install_mcp(roo_dir, hindsight_url)  # second call

    data = json.loads((roo_dir / "mcp.json").read_text())
    assert list(data["mcpServers"].keys()).count("hindsight") == 1


# ---------------------------------------------------------------------------
# install_mcp — custom API URL
# ---------------------------------------------------------------------------


def test_custom_api_url_written_to_mcp(tmp_project: Path) -> None:
    roo_dir = tmp_project / ".roo"
    install_mcp(roo_dir, "https://my-hindsight.example.com")

    data = json.loads((roo_dir / "mcp.json").read_text())
    assert data["mcpServers"]["hindsight"]["url"] == "https://my-hindsight.example.com/mcp"


# ---------------------------------------------------------------------------
# install_rules
# ---------------------------------------------------------------------------


def test_install_rules_creates_file(tmp_project: Path) -> None:
    roo_dir = tmp_project / ".roo"
    install_rules(roo_dir)

    rules_path = roo_dir / "rules" / "hindsight-memory.md"
    assert rules_path.exists()
    assert rules_path.stat().st_size > 0


def test_install_rules_creates_rules_dir(tmp_project: Path) -> None:
    roo_dir = tmp_project / ".roo"
    install_rules(roo_dir)
    assert (roo_dir / "rules").is_dir()


def test_install_rules_idempotent(tmp_project: Path) -> None:
    roo_dir = tmp_project / ".roo"
    install_rules(roo_dir)
    install_rules(roo_dir)  # should not raise
    assert (roo_dir / "rules" / "hindsight-memory.md").exists()
