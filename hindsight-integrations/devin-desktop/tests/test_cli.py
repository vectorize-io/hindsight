"""Tests for the CLI (init/status/uninstall)."""

import json
import subprocess

from hindsight_devin_desktop.cli import Resolved, build_install, main
from hindsight_devin_desktop.global_rules import is_installed as global_rule_installed
from hindsight_devin_desktop.mcp_config import SERVER_NAME
from hindsight_devin_desktop.mcp_config import is_installed as server_installed
from hindsight_devin_desktop.rules import is_installed as rule_installed


class TestBuildInstall:
    def test_writes_mcp_rule_and_global_rule(self, tmp_path):
        mcp_a = tmp_path / "a" / "mcp_config.json"
        mcp_b = tmp_path / "b" / "mcp_config.json"
        rules = tmp_path / "rules" / "hindsight.md"
        global_rules = tmp_path / "memories" / "global_rules.md"
        resolved = Resolved(
            api_url="https://api.hindsight.vectorize.io",
            api_token="k",
            global_bank="devin-desktop",
            project_bank="devin-desktop-acme-web",
            project_source="test",
        )
        outcome = build_install(resolved, [mcp_a, mcp_b], rules, global_rules)

        assert [r.action for r in outcome.mcp] == ["created", "created"]
        for mcp in (mcp_a, mcp_b):
            server = json.loads(mcp.read_text())["mcpServers"][SERVER_NAME]
            # Multi-bank endpoint + global bank as the default header.
            assert server["serverUrl"] == "https://api.hindsight.vectorize.io/mcp/"
            assert server["headers"]["Authorization"] == "Bearer k"
            assert server["headers"]["X-Bank-Id"] == "devin-desktop"
        rule_text = rules.read_text()
        assert "devin-desktop-acme-web" in rule_text and "devin-desktop" in rule_text
        assert global_rule_installed(global_rules)


class TestMain:
    def _common(self, tmp_path):
        return [
            "--mcp-path",
            str(tmp_path / "mcp_config.json"),
            "--rules-path",
            str(tmp_path / "rules" / "hindsight.md"),
            "--global-rules-path",
            str(tmp_path / "global_rules.md"),
            "--user-config-path",
            str(tmp_path / "user.json"),
        ]

    def test_init_status_uninstall(self, tmp_path, capsys):
        common = self._common(tmp_path)
        rc = main(["init", "--api-url", "http://localhost:8888", "--bank-id", "proj", *common])
        assert rc == 0
        assert server_installed(tmp_path / "mcp_config.json")
        assert rule_installed(tmp_path / "rules" / "hindsight.md")
        assert global_rule_installed(tmp_path / "global_rules.md")

        main(["status", *common])
        out = capsys.readouterr().out
        assert "installed" in out and "proj" in out

        main(["uninstall", *common])
        assert not server_installed(tmp_path / "mcp_config.json")
        assert not (tmp_path / "rules" / "hindsight.md").exists()
        assert not global_rule_installed(tmp_path / "global_rules.md")

    def test_init_derives_project_bank_from_git(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "remote", "add", "origin", "git@github.com:acme/web.git"], cwd=repo, check=True)
        rc = main(
            [
                "init",
                "--api-url",
                "http://localhost:8888",
                "--project-dir",
                str(repo),
                *self._common(tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "devin-desktop-acme-web" in out
        assert "devin-desktop-acme-web" in (tmp_path / "rules" / "hindsight.md").read_text()

    def test_init_prints_refresh_reminder(self, tmp_path, capsys):
        main(["init", "--api-url", "http://localhost:8888", "--bank-id", "proj", *self._common(tmp_path)])
        out = capsys.readouterr().out.lower()
        assert "refresh" in out

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        mcp = tmp_path / "mcp_config.json"
        rc = main(
            [
                "init",
                "--print-only",
                "--api-url",
                "http://localhost:8888",
                "--bank-id",
                "proj",
                *self._common(tmp_path),
            ]
        )
        assert rc == 0
        assert not mcp.exists()
        assert not (tmp_path / "rules" / "hindsight.md").exists()
        assert "mcpServers" in capsys.readouterr().out

    def test_no_command_returns_1(self):
        assert main([]) == 1
