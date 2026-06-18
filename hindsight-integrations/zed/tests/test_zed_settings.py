"""Tests for the Zed settings.json context_servers writer."""

import json

from hindsight_zed.zed_settings import (
    SERVER_NAME,
    apply_to_settings,
    build_context_server,
    is_installed,
    mcp_endpoint_url,
    remove_from_settings,
    render_snippet,
)


class TestBuildContextServer:
    def test_endpoint_url_embeds_bank(self):
        assert mcp_endpoint_url("https://api.hindsight.vectorize.io", "proj") == (
            "https://api.hindsight.vectorize.io/mcp/proj/"
        )
        # Trailing slash on the api url is normalized.
        assert mcp_endpoint_url("http://localhost:8888/", "b") == "http://localhost:8888/mcp/b/"

    def test_cloud_server_has_auth_header(self):
        server = build_context_server("https://api.hindsight.vectorize.io", "hsk_abc", "proj")
        assert server["command"] == "npx"
        assert "mcp-remote" in server["args"]
        assert "https://api.hindsight.vectorize.io/mcp/proj/" in server["args"]
        assert "--header" in server["args"]
        assert "Authorization: Bearer hsk_abc" in server["args"]

    def test_open_server_omits_auth_header(self):
        server = build_context_server("http://localhost:8888", None, "proj")
        assert "--header" not in server["args"]


class TestApplyToSettings:
    def test_creates_file_when_absent(self, tmp_path):
        path = tmp_path / "settings.json"
        server = build_context_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_settings(path, server)
        assert result.action == "created"
        data = json.loads(path.read_text())
        assert data["context_servers"][SERVER_NAME] == server

    def test_merges_into_existing_and_preserves_other_keys(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"theme": "One Dark", "context_servers": {"other": {"command": "x"}}}))
        server = build_context_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_settings(path, server)
        assert result.action == "merged"
        data = json.loads(path.read_text())
        assert data["theme"] == "One Dark"  # untouched
        assert data["context_servers"]["other"] == {"command": "x"}  # untouched
        assert data["context_servers"][SERVER_NAME] == server

    def test_unchanged_when_identical(self, tmp_path):
        path = tmp_path / "settings.json"
        server = build_context_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_settings(path, server)
        result = apply_to_settings(path, server)
        assert result.action == "unchanged"

    def test_jsonc_file_returns_manual_and_is_not_rewritten(self, tmp_path):
        path = tmp_path / "settings.json"
        original = '{\n  // my comment\n  "theme": "One Dark",\n}\n'
        path.write_text(original)
        server = build_context_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_settings(path, server)
        assert result.action == "manual"
        assert result.snippet is not None and SERVER_NAME in result.snippet
        assert path.read_text() == original  # never touched the commented file


class TestRemoveAndStatus:
    def test_remove_deletes_only_our_entry(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"context_servers": {"other": {"command": "x"}, SERVER_NAME: {"command": "npx"}}}))
        result = remove_from_settings(path)
        assert result.action == "removed"
        data = json.loads(path.read_text())
        assert SERVER_NAME not in data["context_servers"]
        assert "other" in data["context_servers"]

    def test_remove_drops_empty_context_servers_key(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"theme": "x", "context_servers": {SERVER_NAME: {"command": "npx"}}}))
        remove_from_settings(path)
        data = json.loads(path.read_text())
        assert "context_servers" not in data
        assert data["theme"] == "x"

    def test_is_installed(self, tmp_path):
        path = tmp_path / "settings.json"
        assert is_installed(path) is False
        apply_to_settings(path, build_context_server("https://api.hindsight.vectorize.io", "k", "b"))
        assert is_installed(path) is True

    def test_render_snippet_is_valid_json(self, tmp_path):
        server = build_context_server("https://api.hindsight.vectorize.io", "k", "b")
        snippet = render_snippet(server)
        assert json.loads(snippet)["context_servers"][SERVER_NAME] == server
