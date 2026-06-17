"""Tests for the CLI (init/status/uninstall over settings + rule files)."""

import json

from hindsight_zed.cli import build_install, main
from hindsight_zed.config import ZedConfig
from hindsight_zed.zed_settings import SERVER_NAME, is_installed as server_installed


class TestBuildInstall:
    def test_writes_settings_and_rule(self, tmp_path):
        settings = tmp_path / "settings.json"
        rules = tmp_path / "AGENTS.md"
        cfg = ZedConfig(hindsight_api_url="https://api.hindsight.vectorize.io", hindsight_api_token="k", bank_id="proj")
        outcome = build_install(cfg, settings, rules)

        assert outcome.settings.action == "created"
        server = json.loads(settings.read_text())["context_servers"][SERVER_NAME]
        assert "https://api.hindsight.vectorize.io/mcp/proj/" in server["args"]
        assert "Authorization: Bearer k" in server["args"]
        assert rules.read_text().count("HINDSIGHT:BEGIN") == 1


class TestMainCommands:
    def test_init_then_status_then_uninstall(self, tmp_path, capsys):
        settings = str(tmp_path / "settings.json")
        rules = str(tmp_path / "AGENTS.md")
        config = str(tmp_path / "zed.json")
        common = ["--settings-path", settings, "--rules-path", rules, "--config-path", config]

        rc = main(["init", "--api-url", "http://localhost:8888", "--bank-id", "b", *common])
        assert rc == 0
        assert server_installed(tmp_path / "settings.json")

        main(["status", *common])
        out = capsys.readouterr().out
        assert "installed" in out

        main(["uninstall", *common])
        # settings entry + rule both gone
        assert not server_installed(tmp_path / "settings.json")
        assert not (tmp_path / "AGENTS.md").exists()

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        settings = tmp_path / "settings.json"
        rules = tmp_path / "AGENTS.md"
        rc = main(
            [
                "init",
                "--print-only",
                "--api-url",
                "http://localhost:8888",
                "--settings-path",
                str(settings),
                "--rules-path",
                str(rules),
                "--config-path",
                str(tmp_path / "zed.json"),
            ]
        )
        assert rc == 0
        assert not settings.exists()
        assert not rules.exists()
        assert "context_servers" in capsys.readouterr().out

    def test_no_command_prints_help_and_returns_1(self, capsys):
        assert main([]) == 1
