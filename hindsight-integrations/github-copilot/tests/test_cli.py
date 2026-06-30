"""Tests for the CLI (init/status/uninstall)."""

import json

from hindsight_copilot.cli import build_install, main
from hindsight_copilot.config import CopilotConfig
from hindsight_copilot.mcp_config import SERVER_NAME, is_installed as server_installed


class TestBuildInstall:
    def test_writes_mcp_and_rule(self, tmp_path):
        mcp = tmp_path / "mcp.json"
        instr = tmp_path / "copilot-instructions.md"
        cfg = CopilotConfig(
            hindsight_api_url="https://api.hindsight.vectorize.io", hindsight_api_token="k", bank_id="proj"
        )
        outcome = build_install(cfg, mcp, instr)
        assert outcome.mcp.action == "created"
        server = json.loads(mcp.read_text())["servers"][SERVER_NAME]
        assert server["url"] == "https://api.hindsight.vectorize.io/mcp/proj/"
        assert server["headers"]["Authorization"] == "Bearer k"
        assert "HINDSIGHT:BEGIN" in instr.read_text()


class TestMain:
    def _common(self, tmp_path):
        return [
            "--mcp-path",
            str(tmp_path / "mcp.json"),
            "--instructions-path",
            str(tmp_path / "copilot-instructions.md"),
            "--user-config-path",
            str(tmp_path / "user.json"),
        ]

    def test_init_status_uninstall(self, tmp_path, capsys):
        common = self._common(tmp_path)
        assert main(["init", "--api-url", "http://localhost:8888", "--bank-id", "b", *common]) == 0
        assert server_installed(tmp_path / "mcp.json")
        main(["status", *common])
        assert "installed" in capsys.readouterr().out
        main(["uninstall", *common])
        assert not server_installed(tmp_path / "mcp.json")
        assert not (tmp_path / "copilot-instructions.md").exists()

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        mcp = tmp_path / "mcp.json"
        instr = tmp_path / "copilot-instructions.md"
        rc = main(
            [
                "init",
                "--print-only",
                "--api-url",
                "http://localhost:8888",
                "--mcp-path",
                str(mcp),
                "--instructions-path",
                str(instr),
                "--user-config-path",
                str(tmp_path / "user.json"),
            ]
        )
        assert rc == 0
        assert not mcp.exists() and not instr.exists()
        assert "servers" in capsys.readouterr().out

    def test_no_command_returns_1(self):
        assert main([]) == 1
