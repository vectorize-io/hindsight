"""Shared fixtures for Roo Code Hindsight integration tests."""

import json
import sys
import os
from pathlib import Path

import pytest

# Make the integration root importable so tests can import install.py helpers
INTEGRATION_DIR = Path(__file__).parent.parent
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Empty temporary project directory."""
    return tmp_path


@pytest.fixture()
def project_with_existing_mcp(tmp_path: Path) -> Path:
    """Project with a pre-existing .roo/mcp.json that has another server."""
    roo_dir = tmp_path / ".roo"
    roo_dir.mkdir()
    (roo_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "other-server": {
                        "url": "http://other-server/mcp",
                        "timeout": 5000,
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )
    return tmp_path


@pytest.fixture()
def hindsight_url() -> str:
    return "http://localhost:8888"
