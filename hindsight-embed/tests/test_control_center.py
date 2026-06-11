"""Tests for the hindsight-embed control center.

Covers the service layer (config read/write, alias stripping, key handling)
and the HTTP server (token enforcement, routing) with the daemon lifecycle
mocked so no real daemon is spawned.
"""

import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from hindsight_embed.control_center import lifecycle, providers, service
from hindsight_embed.control_center.server import ControlCenterHandler


@pytest.fixture
def temp_hindsight_dir(tmp_path, monkeypatch):
    """Isolate ~/.hindsight into a temp HOME (matches test_profile_manager)."""
    from hindsight_embed.cli import set_cli_profile_override

    set_cli_profile_override(None)
    monkeypatch.delenv("HINDSIGHT_EMBED_PROFILE", raising=False)
    temp_home = tmp_path / "home"
    temp_home.mkdir()
    monkeypatch.setenv("HOME", str(temp_home))
    monkeypatch.setenv("USERPROFILE", str(temp_home))
    (temp_home / ".hindsight").mkdir()
    return temp_home / ".hindsight"


# --------------------------------------------------------------------------
# service layer
# --------------------------------------------------------------------------
class TestService:
    def test_save_and_read_default_profile(self, temp_hindsight_dir):
        view = service.save_llm_config("", "openai", "sk-abcdef123456", "gpt-4o-mini", "")
        assert view.provider == "openai"
        assert view.model == "gpt-4o-mini"
        assert view.has_api_key
        assert view.api_key_masked and "sk-" in view.api_key_masked
        # round-trips through the "default" URL alias
        assert service.get_profile_config("default").provider == "openai"

    def test_env_file_has_no_alias_keys(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-key1234567890", "m", "")
        raw = service._read_raw_env("")
        assert raw  # not empty
        assert all(not k.islower() for k in raw), f"alias keys leaked: {raw}"

    def test_api_key_unchanged_preserves_key(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-original12345", "m", "")
        after = service.save_llm_config("", "openai", service.API_KEY_UNCHANGED, "", "")
        assert after.has_api_key, "key must survive an unchanged save"
        assert after.model is None, "empty model clears the override"

    def test_empty_api_key_clears_it(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-original12345", "m", "")
        after = service.save_llm_config("", "openai", "", "m", "")
        assert not after.has_api_key

    def test_base_url_preserved_when_not_sent(self, temp_hindsight_dir):
        # base_url is no longer in the wizard; saving without it (None) must keep
        # an existing override, while "" still clears it.
        service.save_llm_config("w", "openai", "sk-key1234567890", "", "https://custom.example/v1")
        service.save_llm_config("w", "openai", service.API_KEY_UNCHANGED, "", None)
        assert service._read_raw_env("w")["HINDSIGHT_API_LLM_BASE_URL"] == "https://custom.example/v1"
        service.save_llm_config("w", "openai", service.API_KEY_UNCHANGED, "", "")
        assert "HINDSIGHT_API_LLM_BASE_URL" not in service._read_raw_env("w")

    def test_named_profile_write_creates_file(self, temp_hindsight_dir):
        service.save_llm_config("work", "groq", "gsk-key1234567890", "", "")
        cfg = service.get_profile_config("work")
        assert cfg.provider == "groq"
        assert (temp_hindsight_dir / "profiles" / "work.env").exists()

    def test_list_profiles_includes_default(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-key1234567890", "", "")
        names = {p.display_name for p in service.list_profiles()}
        assert "default" in names

    def test_save_requires_provider(self, temp_hindsight_dir):
        with pytest.raises(ValueError):
            service.save_llm_config("", "", "k", "", "")

    def test_raw_env_roundtrip(self, temp_hindsight_dir):
        service.write_env_file("", "HINDSIGHT_API_LLM_PROVIDER=groq\nHINDSIGHT_API_LLM_API_KEY=gsk-x\n")
        view = service.read_env_file("default")
        assert view.exists and "groq" in view.content
        # raw edit is reflected in the structured view
        assert service.get_profile_config("").provider == "groq"

    def test_write_env_adds_trailing_newline(self, temp_hindsight_dir):
        view = service.write_env_file("", "HINDSIGHT_API_LLM_PROVIDER=openai")
        assert view.content.endswith("\n")

    def test_profile_paths_references(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-key1234567890", "", "")
        paths = service.get_profile_paths("")
        assert paths.config_path.endswith("embed")
        # URLs are presented as localhost everywhere (not 127.0.0.1 / 0.0.0.0)
        assert paths.daemon_url.startswith("http://localhost:")
        assert "127.0.0.1" not in paths.ui_url and "0.0.0.0" not in paths.ui_url

    def test_tail_log_missing_file(self, temp_hindsight_dir):
        view = service.tail_log("", 10)
        assert view.exists is False and view.content == ""

    def test_tail_log_source_picks_daemon_or_ui(self, temp_hindsight_dir):
        daemon = service.tail_log("", 10, "daemon")
        ui = service.tail_log("", 10, "ui")
        assert daemon.path.endswith(".log") and not daemon.path.endswith(".ui.log")
        assert ui.path.endswith(".ui.log")

    def test_delete_named_profile(self, temp_hindsight_dir):
        service.save_llm_config("scratch", "openai", "sk-key1234567890", "", "")
        assert (temp_hindsight_dir / "profiles" / "scratch.env").exists()
        result = service.delete_profile("scratch")
        assert result.ok
        assert not (temp_hindsight_dir / "profiles" / "scratch.env").exists()

    def test_delete_default_profile_refused(self, temp_hindsight_dir):
        service.save_llm_config("", "openai", "sk-key1234567890", "", "")
        result = service.delete_profile("default")
        assert result.ok is False and "default" in result.message.lower()

    def test_config_exposes_ports(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "")
        cfg = service.get_profile_config("work")
        assert 8889 <= cfg.api_port <= 9888
        assert cfg.ui_port == cfg.api_port + 10000
        assert cfg.ui_port_is_default

    def test_save_pins_ports(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "")
        cfg = service.save_llm_config(
            "work", "openai", service.API_KEY_UNCHANGED, "", "", api_port="9500", ui_port="25000"
        )
        assert cfg.api_port == 9500 and cfg.ui_port == 25000 and cfg.ui_port_is_default is False
        raw = service._read_raw_env("work")
        assert raw["HINDSIGHT_API_PORT"] == "9500" and raw["HINDSIGHT_EMBED_CP_PORT"] == "25000"

    def test_blank_ui_port_derives_from_api(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "", api_port="9500", ui_port="25000")
        cfg = service.save_llm_config("work", "openai", service.API_KEY_UNCHANGED, "", "", api_port="9500", ui_port="")
        assert cfg.ui_port == 19500 and cfg.ui_port_is_default
        assert "HINDSIGHT_EMBED_CP_PORT" not in service._read_raw_env("work")

    def test_versions_default_to_none(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "")
        cfg = service.get_profile_config("work")
        assert cfg.api_version is None and cfg.cp_version is None

    def test_save_pins_versions(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "")
        cfg = service.save_llm_config(
            "work", "openai", service.API_KEY_UNCHANGED, "", "", api_version="0.7.0", cp_version="0.8.1"
        )
        assert cfg.api_version == "0.7.0" and cfg.cp_version == "0.8.1"
        raw = service._read_raw_env("work")
        assert raw["HINDSIGHT_EMBED_API_VERSION"] == "0.7.0" and raw["HINDSIGHT_EMBED_CP_VERSION"] == "0.8.1"

    def test_blank_version_removes_override(self, temp_hindsight_dir):
        service.save_llm_config("work", "openai", "sk-key1234567890", "", "", api_version="0.7.0", cp_version="0.8.1")
        cfg = service.save_llm_config(
            "work", "openai", service.API_KEY_UNCHANGED, "", "", api_version="", cp_version=""
        )
        assert cfg.api_version is None and cfg.cp_version is None
        raw = service._read_raw_env("work")
        assert "HINDSIGHT_EMBED_API_VERSION" not in raw and "HINDSIGHT_EMBED_CP_VERSION" not in raw

    def test_health_reports_down_when_no_daemon(self, temp_hindsight_dir, monkeypatch):
        from hindsight_embed import daemon_client

        monkeypatch.setattr(service, "_http_get", lambda *a, **k: None)
        monkeypatch.setattr(daemon_client, "is_ui_running", lambda *a, **k: False)
        h = service.health("default")
        assert h.api_ok is False and h.ui_ok is False
        assert h.api_detail == "unreachable"

    def test_health_reports_healthy(self, temp_hindsight_dir, monkeypatch):
        from hindsight_embed import daemon_client

        class _Resp:
            status_code = 200

            def json(self):
                return {"status": "healthy", "database": "connected"}

        monkeypatch.setattr(service, "_http_get", lambda *a, **k: _Resp())
        monkeypatch.setattr(daemon_client, "is_ui_running", lambda *a, **k: True)
        h = service.health("default")
        assert h.api_ok and h.ui_ok and "connected" in h.api_detail


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
class TestProviders:
    def test_catalog_has_common_providers(self):
        ids = {p.id for p in providers.PROVIDER_CATALOG}
        assert {"openai", "anthropic", "gemini", "groq", "ollama"} <= ids


# --------------------------------------------------------------------------
# lifecycle token
# --------------------------------------------------------------------------
class TestLifecycle:
    def test_token_is_generated_and_stable(self, temp_hindsight_dir):
        t1 = lifecycle.get_or_create_token()
        assert len(t1) > 20
        assert lifecycle.read_token() == t1
        assert lifecycle.get_or_create_token() == t1  # idempotent

    def test_resolve_port_default_and_override(self, temp_hindsight_dir, monkeypatch):
        monkeypatch.delenv(lifecycle.ENV_CONTROL_PORT, raising=False)
        assert lifecycle.resolve_control_port() == lifecycle.CONTROL_PORT_DEFAULT
        monkeypatch.setenv(lifecycle.ENV_CONTROL_PORT, "9191")
        assert lifecycle.resolve_control_port() == 9191
        monkeypatch.setenv(lifecycle.ENV_CONTROL_PORT, "not-a-port")
        assert lifecycle.resolve_control_port() == lifecycle.CONTROL_PORT_DEFAULT


# --------------------------------------------------------------------------
# HTTP server (real socket, mocked daemon lifecycle)
# --------------------------------------------------------------------------
@pytest.fixture
def server(temp_hindsight_dir):
    """A running control-center server on an ephemeral port with a known token."""
    ControlCenterHandler.token = "test-token-secret"
    ControlCenterHandler.version = "test"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ControlCenterHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _auth(token="test-token-secret"):
    return {"X-Hindsight-Control-Token": token}


class TestServer:
    def test_health_needs_no_token(self, server):
        r = httpx.get(f"{server}/api/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    def test_index_served(self, server):
        r = httpx.get(f"{server}/")
        assert r.status_code == 200 and "Hindsight Embed Control Center" in r.text

    def test_api_requires_token(self, server):
        assert httpx.get(f"{server}/api/profiles").status_code == 401
        assert httpx.get(f"{server}/api/profiles", headers=_auth("wrong")).status_code == 401

    def test_providers_endpoint(self, server):
        r = httpx.get(f"{server}/api/providers", headers=_auth())
        assert r.status_code == 200
        assert any(p["id"] == "openai" for p in r.json()["providers"])

    def test_config_roundtrip_over_http(self, server):
        post = httpx.post(
            f"{server}/api/profiles/default/config",
            headers=_auth(),
            json={"provider": "openai", "api_key": "sk-http1234567890", "model": "gpt-4o-mini"},
        )
        assert post.status_code == 200
        get = httpx.get(f"{server}/api/profiles/default/config", headers=_auth())
        assert get.json()["provider"] == "openai"
        assert get.json()["has_api_key"] is True

    def test_save_missing_provider_is_400(self, server):
        r = httpx.post(f"{server}/api/profiles/default/config", headers=_auth(), json={"provider": ""})
        assert r.status_code == 400

    def test_daemon_action_mocked(self, server, monkeypatch):
        from hindsight_embed import daemon_client

        monkeypatch.setattr(daemon_client, "ensure_daemon_running", lambda *a, **k: True)
        monkeypatch.setattr(daemon_client, "is_daemon_running", lambda profile=None: True)
        monkeypatch.setattr(daemon_client, "get_daemon_url", lambda profile=None: "http://127.0.0.1:8888")
        r = httpx.post(f"{server}/api/profiles/default/daemon/start", headers=_auth())
        assert r.status_code == 200
        assert r.json()["running"] is True
        # the daemon URL is presented as localhost
        assert r.json()["url"] == "http://localhost:8888"

    def test_unknown_daemon_action_404(self, server, monkeypatch):
        r = httpx.post(f"{server}/api/profiles/default/daemon/frobnicate", headers=_auth())
        assert r.status_code == 404

    def test_logo_served_without_token(self, server):
        r = httpx.get(f"{server}/logo.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"

    def test_self_hosted_fonts_served(self, server):
        # Fonts are self-hosted (bundled @font-face references /fonts/*) so the
        # UI renders offline; the woff2 files are served from the static dir.
        woff = httpx.get(f"{server}/fonts/SpaceGrotesk-700.woff2")
        assert woff.status_code == 200 and woff.headers["content-type"] == "font/woff2"

    def test_static_unknown_extension_not_served(self, server):
        # Only whitelisted asset types are served; anything else falls through.
        assert httpx.get(f"{server}/index.html.bak").status_code in (404, 401)

    def test_raw_env_roundtrip_over_http(self, server):
        body = "HINDSIGHT_API_LLM_PROVIDER=groq\nHINDSIGHT_API_LLM_API_KEY=gsk-http\n"
        post = httpx.post(f"{server}/api/profiles/default/env", headers=_auth(), json={"content": body})
        assert post.status_code == 200 and "groq" in post.json()["content"]
        get = httpx.get(f"{server}/api/profiles/default/env", headers=_auth())
        assert "groq" in get.json()["content"]

    def test_paths_endpoint(self, server):
        r = httpx.get(f"{server}/api/profiles/default/paths", headers=_auth())
        assert r.status_code == 200 and r.json()["config_path"].endswith("embed")

    def test_logs_endpoint_respects_lines_param(self, server):
        r = httpx.get(f"{server}/api/profiles/default/logs?lines=10", headers=_auth())
        assert r.status_code == 200 and "exists" in r.json()

    def test_ui_status_needs_token(self, server):
        assert httpx.get(f"{server}/api/profiles/default/ui").status_code == 401
        r = httpx.get(f"{server}/api/profiles/default/ui", headers=_auth())
        assert r.status_code == 200 and "running" in r.json()

    def test_health_endpoint(self, server, monkeypatch):
        from hindsight_embed import daemon_client

        monkeypatch.setattr(service, "_http_get", lambda *a, **k: None)
        monkeypatch.setattr(daemon_client, "is_ui_running", lambda *a, **k: False)
        r = httpx.get(f"{server}/api/profiles/default/health", headers=_auth())
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"api_ok", "api_detail", "ui_ok"}
        assert body["api_ok"] is False  # probe mocked as unreachable

    def test_delete_profile_over_http(self, server):
        httpx.post(
            f"{server}/api/profiles/throwaway/config",
            headers=_auth(),
            json={"provider": "openai", "api_key": "sk-x1234567890"},
        )
        r = httpx.post(f"{server}/api/profiles/throwaway/delete", headers=_auth())
        assert r.status_code == 200 and r.json()["ok"] is True
        # gone from the list
        names = {p["name"] for p in httpx.get(f"{server}/api/profiles", headers=_auth()).json()["profiles"]}
        assert "throwaway" not in names

    def test_responses_are_no_store(self, server):
        assert httpx.get(f"{server}/").headers.get("cache-control") == "no-store"
        assert httpx.get(f"{server}/api/health").headers.get("cache-control") == "no-store"
