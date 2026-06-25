"""Tests for the CLI (init/status/uninstall)."""

import json

from hindsight_windsurf.cli import build_install, main
from hindsight_windsurf.config import WindsurfConfig
from hindsight_windsurf.mcp_config import SERVER_NAME
from hindsight_windsurf.mcp_config import is_installed as server_installed
from hindsight_windsurf.rules import is_installed as rule_installed


class TestBuildInstall:
    def test_writes_mcp_and_rule(self, tmp_path):
        mcp = tmp_path / "mcp_config.json"
        rules = tmp_path / "rules" / "hindsight.md"
        cfg = WindsurfConfig(
            hindsight_api_url="https://api.hindsight.vectorize.io", hindsight_api_token="k", bank_id="proj"
        )
        outcome = build_install(cfg, mcp, rules)
        assert outcome.mcp.action == "created"
        server = json.loads(mcp.read_text())["mcpServers"][SERVER_NAME]
        assert server["serverUrl"] == "https://api.hindsight.vectorize.io/mcp/proj/"
        assert server["headers"]["Authorization"] == "Bearer k"
        assert rule_installed(rules)


class TestMain:
    def _common(self, tmp_path):
        return [
            "--mcp-path",
            str(tmp_path / "mcp_config.json"),
            "--rules-path",
            str(tmp_path / "rules" / "hindsight.md"),
            "--user-config-path",
            str(tmp_path / "user.json"),
        ]

    def test_init_status_uninstall(self, tmp_path, capsys):
        common = self._common(tmp_path)
        assert main(["init", "--api-url", "http://localhost:8888", "--bank-id", "b", *common]) == 0
        assert server_installed(tmp_path / "mcp_config.json")
        assert rule_installed(tmp_path / "rules" / "hindsight.md")
        main(["status", *common])
        assert "installed" in capsys.readouterr().out
        main(["uninstall", *common])
        assert not server_installed(tmp_path / "mcp_config.json")
        assert not (tmp_path / "rules" / "hindsight.md").exists()

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        mcp = tmp_path / "mcp_config.json"
        rules = tmp_path / "rules" / "hindsight.md"
        rc = main(
            [
                "init",
                "--print-only",
                "--api-url",
                "http://localhost:8888",
                "--mcp-path",
                str(mcp),
                "--rules-path",
                str(rules),
                "--user-config-path",
                str(tmp_path / "user.json"),
            ]
        )
        assert rc == 0
        assert not mcp.exists() and not rules.exists()
        assert "mcpServers" in capsys.readouterr().out

    def test_no_command_returns_1(self):
        assert main([]) == 1
