"""Tests for the .vscode/mcp.json servers writer."""

import json

from hindsight_copilot.mcp_config import (
    SERVER_NAME,
    apply_to_mcp,
    build_http_server,
    is_installed,
    mcp_endpoint_url,
    remove_from_mcp,
    render_snippet,
)


class TestBuildServer:
    def test_endpoint_url_embeds_bank(self):
        assert mcp_endpoint_url("https://api.hindsight.vectorize.io", "proj") == (
            "https://api.hindsight.vectorize.io/mcp/proj/"
        )
        assert mcp_endpoint_url("http://localhost:8888/", "b") == "http://localhost:8888/mcp/b/"

    def test_cloud_server_http_with_auth_header(self):
        s = build_http_server("https://api.hindsight.vectorize.io", "hsk_abc", "proj")
        assert s["type"] == "http"
        assert s["url"] == "https://api.hindsight.vectorize.io/mcp/proj/"
        assert s["headers"] == {"Authorization": "Bearer hsk_abc"}

    def test_open_server_omits_headers(self):
        s = build_http_server("http://localhost:8888", None, "proj")
        assert s == {"type": "http", "url": "http://localhost:8888/mcp/proj/"}
        assert "headers" not in s


class TestApply:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "mcp.json"
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_mcp(path, s)
        assert result.action == "created"
        assert json.loads(path.read_text())["servers"][SERVER_NAME] == s

    def test_merges_preserves_other_servers_and_inputs(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"inputs": [{"id": "tok"}], "servers": {"other": {"type": "stdio"}}}))
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_mcp(path, s)
        assert result.action == "merged"
        data = json.loads(path.read_text())
        assert data["inputs"] == [{"id": "tok"}]  # untouched
        assert data["servers"]["other"] == {"type": "stdio"}  # untouched
        assert data["servers"][SERVER_NAME] == s

    def test_unchanged_when_identical(self, tmp_path):
        path = tmp_path / "mcp.json"
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_mcp(path, s)
        assert apply_to_mcp(path, s).action == "unchanged"

    def test_jsonc_returns_manual(self, tmp_path):
        path = tmp_path / "mcp.json"
        original = '{\n  // comment\n  "servers": {}\n}\n'
        path.write_text(original)
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_mcp(path, s)
        assert result.action == "manual"
        assert result.snippet and SERVER_NAME in result.snippet
        assert path.read_text() == original  # untouched


class TestRemoveAndStatus:
    def test_remove_only_our_entry(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"servers": {"other": {"type": "stdio"}, SERVER_NAME: {"type": "http"}}}))
        result = remove_from_mcp(path)
        assert result.action == "removed"
        servers = json.loads(path.read_text())["servers"]
        assert SERVER_NAME not in servers and "other" in servers

    def test_remove_drops_empty_servers(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text(json.dumps({"inputs": [], "servers": {SERVER_NAME: {"type": "http"}}}))
        remove_from_mcp(path)
        data = json.loads(path.read_text())
        assert "servers" not in data
        assert "inputs" in data

    def test_is_installed(self, tmp_path):
        path = tmp_path / "mcp.json"
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        assert is_installed(path) is False
        apply_to_mcp(path, s)
        assert is_installed(path) is True

    def test_render_snippet_valid_json(self):
        s = build_http_server("https://api.hindsight.vectorize.io", "k", "b")
        assert json.loads(render_snippet(s))["servers"][SERVER_NAME] == s
