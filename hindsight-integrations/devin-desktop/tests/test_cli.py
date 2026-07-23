"""Tests for the CLI (init/status/uninstall) — covers both agents."""

import json
import subprocess

from hindsight_devin_desktop import devin_local
from hindsight_devin_desktop.cli import Paths, Resolved, build_install, main
from hindsight_devin_desktop.global_rules import is_installed as global_rule_installed
from hindsight_devin_desktop.mcp_config import SERVER_NAME
from hindsight_devin_desktop.mcp_config import is_installed as server_installed
from hindsight_devin_desktop.rules import is_installed as rule_installed


def _paths(tmp_path):
    return Paths(
        mcp=[tmp_path / "a" / "mcp_config.json", tmp_path / "b" / "mcp_config.json"],
        rules=tmp_path / "rules" / "hindsight.md",
        global_rules=tmp_path / "memories" / "global_rules.md",
        cascade_hooks=tmp_path / "windsurf" / "hooks.json",
        devin_config=tmp_path / "devin" / "config.json",
        devin_global_agents=tmp_path / "devin" / "AGENTS.md",
        devin_project_agents=tmp_path / "proj" / "AGENTS.md",
    )


class TestBuildInstall:
    def test_wires_both_agents(self, tmp_path):
        paths = _paths(tmp_path)
        resolved = Resolved(
            api_url="https://api.hindsight.vectorize.io",
            api_token="k",
            global_bank="devin-desktop",
            project_bank="devin-desktop-acme-web",
            project_source="test",
        )
        outcome = build_install(resolved, paths)

        # Cascade MCP (serverUrl) in both locations
        assert [r.action for r in outcome.mcp] == ["created", "created"]
        for mcp in paths.mcp:
            server = json.loads(mcp.read_text())["mcpServers"][SERVER_NAME]
            assert server["serverUrl"] == "https://api.hindsight.vectorize.io/mcp/"
            assert server["headers"]["X-Bank-Id"] == "devin-desktop"
        # Cascade rules + visibility banner
        assert "devin-desktop-acme-web" in paths.rules.read_text()
        assert global_rule_installed(paths.global_rules)
        assert outcome.cascade_banner.action == "created"
        assert "post_mcp_tool_use" in json.loads(paths.cascade_hooks.read_text())["hooks"]

        # Devin Local MCP (url + transport) + auto-approve permission
        assert outcome.devin_config.action == "created"
        dl = json.loads(paths.devin_config.read_text())
        dserver = dl["mcpServers"][SERVER_NAME]
        assert dserver["url"] == "https://api.hindsight.vectorize.io/mcp/"
        assert dserver["transport"] == "http"
        assert dserver["headers"]["Authorization"] == "Bearer k"
        assert dserver["headers"]["X-Bank-Id"] == "devin-desktop"
        assert dl["permissions"]["allow"] == ["mcp__hindsight__*"]
        # Devin Local AGENTS.md (project names both banks; global names the global bank)
        assert "devin-desktop-acme-web" in paths.devin_project_agents.read_text()
        assert devin_local.agents_installed(paths.devin_global_agents)


class TestMain:
    def _common(self, tmp_path):
        return [
            "--mcp-path",
            str(tmp_path / "mcp_config.json"),
            "--rules-path",
            str(tmp_path / "rules" / "hindsight.md"),
            "--global-rules-path",
            str(tmp_path / "global_rules.md"),
            "--cascade-hooks-path",
            str(tmp_path / "hooks.json"),
            "--devin-config-path",
            str(tmp_path / "devin" / "config.json"),
            "--devin-global-agents-path",
            str(tmp_path / "devin" / "AGENTS.md"),
            "--project-agents-path",
            str(tmp_path / "AGENTS.md"),
            "--user-config-path",
            str(tmp_path / "user.json"),
        ]

    def test_init_status_uninstall(self, tmp_path, capsys):
        common = self._common(tmp_path)
        assert main(["init", "--api-url", "http://localhost:8888", "--bank-id", "proj", *common]) == 0
        # both agents configured
        assert server_installed(tmp_path / "mcp_config.json")  # Cascade
        assert rule_installed(tmp_path / "rules" / "hindsight.md")
        assert global_rule_installed(tmp_path / "global_rules.md")
        assert devin_local.is_installed(tmp_path / "devin" / "config.json")  # Devin Local
        assert devin_local.agents_installed(tmp_path / "AGENTS.md")
        assert devin_local.agents_installed(tmp_path / "devin" / "AGENTS.md")

        main(["status", *common])
        out = capsys.readouterr().out
        assert "Cascade" in out and "Devin Local" in out and "proj" in out

        main(["uninstall", *common])
        assert not server_installed(tmp_path / "mcp_config.json")
        assert not devin_local.is_installed(tmp_path / "devin" / "config.json")
        assert not devin_local.agents_installed(tmp_path / "AGENTS.md")

    def test_init_derives_project_bank_from_git(self, tmp_path, capsys):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "remote", "add", "origin", "git@github.com:acme/web.git"], cwd=repo, check=True)
        rc = main(["init", "--api-url", "http://localhost:8888", "--project-dir", str(repo), *self._common(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "devin-desktop-acme-web" in out
        # both agents' project rules carry the derived bank
        assert "devin-desktop-acme-web" in (tmp_path / "rules" / "hindsight.md").read_text()
        assert "devin-desktop-acme-web" in (tmp_path / "AGENTS.md").read_text()

    def test_init_prints_activation_steps(self, tmp_path, capsys):
        main(["init", "--api-url", "http://localhost:8888", "--bank-id", "proj", *self._common(tmp_path)])
        out = capsys.readouterr().out.lower()
        assert "refresh" in out  # Cascade activation
        assert "connect" in out  # Devin Local activation

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        rc = main(
            ["init", "--print-only", "--api-url", "http://localhost:8888", "--bank-id", "proj", *self._common(tmp_path)]
        )
        assert rc == 0
        assert not (tmp_path / "mcp_config.json").exists()
        assert not (tmp_path / "devin" / "config.json").exists()
        assert not (tmp_path / "AGENTS.md").exists()
        out = capsys.readouterr().out
        assert "mcpServers" in out and "Devin Local" in out

    def test_local_only_mode(self, tmp_path):
        common = self._common(tmp_path)
        assert (
            main(["init", "--api-url", "http://localhost:8888", "--bank-id", "proj", "--no-global-bank", *common]) == 0
        )
        # No shared global rule files
        assert not global_rule_installed(tmp_path / "global_rules.md")
        assert not devin_local.agents_installed(tmp_path / "devin" / "AGENTS.md")
        # Project rule is single-bank (mentions the project bank, no separate global)
        txt = (tmp_path / "AGENTS.md").read_text()
        assert "proj" in txt and "one bank" in txt.lower()
        # Devin Local hooks carry --local-only
        cfg = json.loads((tmp_path / "devin" / "config.json").read_text())
        cmd = cfg["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert "--local-only" in cmd

    def test_no_command_returns_1(self):
        assert main([]) == 1
