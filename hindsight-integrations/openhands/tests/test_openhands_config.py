"""Tests for the OpenHands config.toml [mcp] writer."""

import tomlkit

from hindsight_openhands.openhands_config import (
    apply_to_config,
    build_shttp_server,
    is_installed,
    mcp_endpoint_url,
    remove_from_config,
    render_snippet,
)


class TestBuildServer:
    def test_endpoint_url_embeds_bank(self):
        assert mcp_endpoint_url("https://api.hindsight.vectorize.io", "proj") == (
            "https://api.hindsight.vectorize.io/mcp/proj/"
        )
        assert mcp_endpoint_url("http://localhost:8888/", "b") == "http://localhost:8888/mcp/b/"

    def test_cloud_server_has_api_key(self):
        s = build_shttp_server("https://api.hindsight.vectorize.io", "hsk_abc", "proj")
        assert s == {"url": "https://api.hindsight.vectorize.io/mcp/proj/", "api_key": "hsk_abc"}

    def test_open_server_omits_api_key(self):
        s = build_shttp_server("http://localhost:8888", None, "proj")
        assert s == {"url": "http://localhost:8888/mcp/proj/"}
        assert "api_key" not in s


class TestApply:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "config.toml"
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_config(path, s)
        assert result.action == "created"
        doc = tomlkit.parse(path.read_text())
        assert doc["mcp"]["shttp_servers"][0]["url"] == s["url"]
        assert doc["mcp"]["shttp_servers"][0]["api_key"] == "k"

    def test_merges_and_preserves_existing(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('# header comment\n[core]\nworkspace_base = "./ws"\n\n[llm]\nmodel = "anthropic/claude"\n')
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_config(path, s)
        assert result.action == "merged"
        text = path.read_text()
        assert "# header comment" in text  # comment preserved
        assert 'model = "anthropic/claude"' in text  # other config preserved
        doc = tomlkit.parse(text)
        assert doc["mcp"]["shttp_servers"][0]["url"] == s["url"]

    def test_appends_alongside_other_mcp_servers(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[mcp]\nshttp_servers = [{url = "https://other/mcp/x"}]\n')
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_config(path, s)
        doc = tomlkit.parse(path.read_text())
        urls = [e["url"] for e in doc["mcp"]["shttp_servers"]]
        assert "https://other/mcp/x" in urls  # untouched
        assert s["url"] in urls

    def test_unchanged_when_identical(self, tmp_path):
        path = tmp_path / "config.toml"
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_config(path, s)
        assert apply_to_config(path, s).action == "unchanged"

    def test_updates_when_url_matches_but_token_differs(self, tmp_path):
        path = tmp_path / "config.toml"
        apply_to_config(path, build_shttp_server("https://api.hindsight.vectorize.io", "old", "b"))
        result = apply_to_config(path, build_shttp_server("https://api.hindsight.vectorize.io", "new", "b"))
        assert result.action == "merged"
        doc = tomlkit.parse(path.read_text())
        assert len(doc["mcp"]["shttp_servers"]) == 1  # not duplicated
        assert doc["mcp"]["shttp_servers"][0]["api_key"] == "new"

    def test_invalid_toml_returns_manual(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("this is = = not valid toml ][\n")
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        result = apply_to_config(path, s)
        assert result.action == "manual"
        assert result.snippet and "shttp_servers" in result.snippet


class TestRemoveAndStatus:
    def test_remove_only_our_entry(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[mcp]\nshttp_servers = [{url = "https://other/mcp/x"}]\n')
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_config(path, s)
        result = remove_from_config(path, s)
        assert result.action == "removed"
        doc = tomlkit.parse(path.read_text())
        urls = [e["url"] for e in doc["mcp"]["shttp_servers"]]
        assert s["url"] not in urls
        assert "https://other/mcp/x" in urls

    def test_remove_drops_empty_mcp(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("[core]\nx = 1\n")
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        apply_to_config(path, s)
        remove_from_config(path, s)
        doc = tomlkit.parse(path.read_text())
        assert "mcp" not in doc
        assert doc["core"]["x"] == 1

    def test_is_installed(self, tmp_path):
        path = tmp_path / "config.toml"
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        assert is_installed(path, s) is False
        apply_to_config(path, s)
        assert is_installed(path, s) is True

    def test_render_snippet_parses(self):
        s = build_shttp_server("https://api.hindsight.vectorize.io", "k", "b")
        doc = tomlkit.parse(render_snippet(s))
        assert doc["mcp"]["shttp_servers"][0]["url"] == s["url"]
