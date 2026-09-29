"""Tests for the hindsight-roo-code CLI entry point."""

import json
from pathlib import Path

import pytest

from hindsight_roo_code.cli import main
from hindsight_roo_code.install import DEFAULT_API_URL


def test_install_writes_both_files(tmp_path: Path) -> None:
    rc = main(["install", "--project-dir", str(tmp_path)])
    assert rc == 0

    roo_dir = tmp_path / ".roo"
    assert (roo_dir / "mcp.json").exists()
    assert (roo_dir / "rules" / "hindsight-memory.md").exists()


def test_install_defaults_to_cloud_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HINDSIGHT_API_URL", raising=False)
    main(["install", "--project-dir", str(tmp_path)])

    data = json.loads((tmp_path / ".roo" / "mcp.json").read_text())
    assert data["mcpServers"]["hindsight"]["url"] == f"{DEFAULT_API_URL}/mcp"


def test_install_custom_api_url(tmp_path: Path) -> None:
    main(
        [
            "install",
            "--project-dir",
            str(tmp_path),
            "--api-url",
            "http://localhost:8888",
        ]
    )

    data = json.loads((tmp_path / ".roo" / "mcp.json").read_text())
    assert data["mcpServers"]["hindsight"]["url"] == "http://localhost:8888/mcp"


def test_api_url_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_URL", "http://env-url:9999")
    main(["install", "--project-dir", str(tmp_path)])

    data = json.loads((tmp_path / ".roo" / "mcp.json").read_text())
    assert data["mcpServers"]["hindsight"]["url"] == "http://env-url:9999/mcp"


def test_no_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
