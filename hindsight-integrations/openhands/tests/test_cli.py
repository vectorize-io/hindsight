"""Tests for the CLI (init/status/uninstall over config.toml + AGENTS.md)."""

import tomlkit

from hindsight_openhands.cli import build_install, main
from hindsight_openhands.config import OpenHandsConfig
from hindsight_openhands.openhands_config import build_shttp_server, is_installed as server_installed


class TestBuildInstall:
    def test_writes_config_and_rule(self, tmp_path):
        config = tmp_path / "config.toml"
        agents = tmp_path / "AGENTS.md"
        cfg = OpenHandsConfig(
            hindsight_api_url="https://api.hindsight.vectorize.io", hindsight_api_token="k", bank_id="proj"
        )
        outcome = build_install(cfg, config, agents)
        assert outcome.config.action == "created"
        doc = tomlkit.parse(config.read_text())
        entry = doc["mcp"]["shttp_servers"][0]
        assert entry["url"] == "https://api.hindsight.vectorize.io/mcp/proj/"
        assert entry["api_key"] == "k"
        assert "HINDSIGHT:BEGIN" in agents.read_text()


class TestMain:
    def _common(self, tmp_path):
        return [
            "--config-path",
            str(tmp_path / "config.toml"),
            "--agents-path",
            str(tmp_path / "AGENTS.md"),
            "--user-config-path",
            str(tmp_path / "user.json"),
        ]

    def test_init_status_uninstall(self, tmp_path, capsys):
        common = self._common(tmp_path)
        s = build_shttp_server("http://localhost:8888", None, "b")

        assert main(["init", "--api-url", "http://localhost:8888", "--bank-id", "b", *common]) == 0
        assert server_installed(tmp_path / "config.toml", s)

        main(["status", "--api-url", "http://localhost:8888", "--bank-id", "b", *common])
        assert "installed" in capsys.readouterr().out

        main(["uninstall", "--api-url", "http://localhost:8888", "--bank-id", "b", *common])
        assert not server_installed(tmp_path / "config.toml", s)
        assert not (tmp_path / "AGENTS.md").exists()

    def test_print_only_writes_nothing(self, tmp_path, capsys):
        config = tmp_path / "config.toml"
        agents = tmp_path / "AGENTS.md"
        rc = main(
            [
                "init",
                "--print-only",
                "--api-url",
                "http://localhost:8888",
                "--config-path",
                str(config),
                "--agents-path",
                str(agents),
                "--user-config-path",
                str(tmp_path / "user.json"),
            ]
        )
        assert rc == 0
        assert not config.exists() and not agents.exists()
        assert "shttp_servers" in capsys.readouterr().out

    def test_no_command_returns_1(self):
        assert main([]) == 1
