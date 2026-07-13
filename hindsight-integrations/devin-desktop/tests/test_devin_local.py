"""Tests for the Devin Local agent wiring (config.json + AGENTS.md)."""

import json

from hindsight_devin_desktop.devin_local import (
    ALLOW_PATTERN,
    SERVER_NAME,
    agents_installed,
    apply_to_config,
    build_http_server,
    clear_agents,
    default_config_path,
    default_global_agents_path,
    default_project_agents_path,
    is_installed,
    remove_from_config,
    write_global_agents,
    write_project_agents,
)


class TestBuildServer:
    def test_uses_url_transport_and_headers(self):
        s = build_http_server("https://api.hindsight.vectorize.io", "hsk_x", "devin-desktop")
        assert s["url"] == "https://api.hindsight.vectorize.io/mcp/"
        assert s["transport"] == "http"
        assert s["headers"]["Authorization"] == "Bearer hsk_x"
        assert s["headers"]["X-Bank-Id"] == "devin-desktop"
        assert "serverUrl" not in s  # that's Cascade's field

    def test_open_server_omits_auth_keeps_bank(self):
        s = build_http_server("http://localhost:8888", None, "devin-desktop")
        assert s["url"] == "http://localhost:8888/mcp/"
        assert s["headers"] == {"X-Bank-Id": "devin-desktop"}


class TestDefaultPaths:
    def test_paths(self):
        assert str(default_config_path()).endswith("/.config/devin/config.json")
        assert str(default_global_agents_path()).endswith("/.config/devin/AGENTS.md")
        assert default_project_agents_path().name == "AGENTS.md"


class TestApplyConfig:
    def test_creates_with_server_and_allow(self, tmp_path):
        p = tmp_path / "config.json"
        s = build_http_server("http://localhost:8888", "k", "b")
        result = apply_to_config(p, s)
        assert result.action == "created"
        data = json.loads(p.read_text())
        assert data["mcpServers"][SERVER_NAME] == s
        assert ALLOW_PATTERN in data["permissions"]["allow"]

    def test_preserves_existing_keys(self, tmp_path):
        p = tmp_path / "config.json"
        # e.g. a real Devin Local config that already has a version + a user allow rule
        p.write_text(json.dumps({"version": 1, "permissions": {"allow": ["mcp__other__*"]}}))
        s = build_http_server("http://localhost:8888", "k", "b")
        result = apply_to_config(p, s)
        assert result.action == "merged"
        data = json.loads(p.read_text())
        assert data["version"] == 1  # preserved
        assert "mcp__other__*" in data["permissions"]["allow"]  # user's rule kept
        assert ALLOW_PATTERN in data["permissions"]["allow"]  # ours added

    def test_unchanged_when_identical(self, tmp_path):
        p = tmp_path / "config.json"
        s = build_http_server("http://localhost:8888", "k", "b")
        apply_to_config(p, s)
        assert apply_to_config(p, s).action == "unchanged"

    def test_non_json_returns_manual(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("{ not json")
        s = build_http_server("http://localhost:8888", "k", "b")
        result = apply_to_config(p, s)
        assert result.action == "manual"
        assert result.snippet and SERVER_NAME in result.snippet
        assert p.read_text() == "{ not json"  # untouched

    def test_is_installed(self, tmp_path):
        p = tmp_path / "config.json"
        assert is_installed(p) is False
        apply_to_config(p, build_http_server("http://localhost:8888", "k", "b"))
        assert is_installed(p) is True


class TestRemoveConfig:
    def test_removes_server_and_allow_keeps_rest(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"version": 1, "permissions": {"allow": ["mcp__other__*"]}}))
        apply_to_config(p, build_http_server("http://localhost:8888", "k", "b"))
        result = remove_from_config(p)
        assert result.action == "removed"
        data = json.loads(p.read_text())
        assert "mcpServers" not in data
        assert data["version"] == 1  # preserved
        assert data["permissions"]["allow"] == ["mcp__other__*"]  # only ours removed

    def test_unchanged_when_absent(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"version": 1}))
        assert remove_from_config(p).action == "unchanged"


class TestAgentsMd:
    def test_project_names_both_banks(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        write_project_agents(p, "devin-desktop-acme-web", "devin-desktop")
        text = p.read_text()
        assert "devin-desktop-acme-web" in text and "devin-desktop" in text
        assert agents_installed(p)

    def test_global_names_global_bank(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        write_global_agents(p, "devin-desktop")
        assert "devin-desktop" in p.read_text()
        assert agents_installed(p)

    def test_clear(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        write_global_agents(p, "devin-desktop")
        clear_agents(p)
        assert not agents_installed(p)
