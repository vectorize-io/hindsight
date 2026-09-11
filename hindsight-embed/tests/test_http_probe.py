"""The aiohttp liveness probes, against a real loopback server.

Covers the probe itself (status, body, refused, timeout, malformed URL, the
running-loop guard) and the callers that interpret its answer.
"""

import asyncio
import time

import pytest

from hindsight_embed._http_probe import ProbeResponse, aprobe_get, probe_get
from hindsight_embed.control_center import lifecycle
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager
from hindsight_embed.profile_manager import ProfileManager

from .http_stub import closed_port, reply, serve


class TestProbe:
    def test_returns_status_and_json_body(self):
        with serve(reply(200, {"status": "healthy"})) as stub:
            resp = probe_get(f"{stub.base_url}/health", read_timeout=2.0)
        assert resp is not None
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}
        assert stub.paths == ["/health"]

    def test_non_200_is_still_an_answer(self):
        with serve(reply(503, {"status": "starting"})) as stub:
            resp = probe_get(f"{stub.base_url}/health", read_timeout=2.0)
        assert resp is not None and resp.status_code == 503

    def test_non_json_body_raises_only_on_json(self):
        with serve(reply(200, text="<html>not json</html>")) as stub:
            resp = probe_get(f"{stub.base_url}/health", read_timeout=2.0)
        assert resp is not None and resp.status_code == 200
        with pytest.raises(ValueError):
            resp.json()

    def test_refused_connection_is_none(self):
        assert probe_get(f"http://127.0.0.1:{closed_port()}/health", read_timeout=2.0) is None

    def test_read_timeout_is_none_and_bounded(self):
        with serve(reply(200, {}, delay=1.5)) as stub:
            started = time.monotonic()
            resp = probe_get(f"{stub.base_url}/health", read_timeout=0.2, connect_timeout=0.2)
            elapsed = time.monotonic() - started
        assert resp is None
        assert elapsed < 1.2

    def test_malformed_url_is_none(self):
        assert probe_get("http://127.0.0.1:notaport/health", read_timeout=1.0) is None

    def test_sync_entry_point_works_inside_a_running_loop(self):
        """hindsight-all's sync _ensure_started is reached from its async methods too."""

        async def inside_loop() -> ProbeResponse | None:
            return probe_get(f"{stub.base_url}/api/health", read_timeout=2.0)

        with serve(reply(200, {"status": "ok"})) as stub:
            resp = asyncio.run(inside_loop())
        assert resp == ProbeResponse(status_code=200, text='{"status": "ok"}')

    def test_async_entry_point_works_inside_a_loop(self):
        with serve(reply(200, {"status": "ok"})) as stub:
            resp = asyncio.run(aprobe_get(f"{stub.base_url}/api/health", read_timeout=2.0))
        assert resp == ProbeResponse(status_code=200, text='{"status": "ok"}')


class TestPortHealthOk:
    """DaemonEmbedManager._port_health_ok: only Hindsight's initialized payload counts."""

    def test_initialized_hindsight(self):
        with serve(reply(200, {"status": "healthy", "database": "connected"})) as stub:
            assert DaemonEmbedManager._port_health_ok(stub.port) is True

    @pytest.mark.parametrize(
        "handler",
        [
            reply(200, {"status": "healthy", "database": "disconnected"}),
            reply(200, {"status": "ok"}),
            reply(200, ["not", "an", "object"]),
            reply(200, text="plain text"),
            reply(503, {"status": "healthy", "database": "connected"}),
        ],
        ids=["db-down", "foreign-200", "json-array", "not-json", "503"],
    )
    def test_anything_else_is_not_healthy(self, handler):
        with serve(handler) as stub:
            assert DaemonEmbedManager._port_health_ok(stub.port) is False

    def test_nothing_listening(self):
        assert DaemonEmbedManager._port_health_ok(closed_port()) is False


class TestControlCenterHealth:
    """lifecycle._health_ok: 200 with {"status": "ok"} and nothing else."""

    def test_ok(self):
        with serve(reply(200, {"status": "ok"})) as stub:
            assert lifecycle._health_ok(stub.port) is True
            assert stub.paths == ["/api/health"]

    @pytest.mark.parametrize(
        "handler",
        [reply(200, {"status": "starting"}), reply(200, text="nope"), reply(200, [1]), reply(500, {"status": "ok"})],
        ids=["wrong-status", "not-json", "json-array", "500"],
    )
    def test_not_ok(self, handler):
        with serve(handler) as stub:
            assert lifecycle._health_ok(stub.port) is False

    def test_down(self):
        assert lifecycle._health_ok(closed_port()) is False


class TestProfileDaemonCheck:
    """ProfileManager._check_daemon_running: any 200 from /health."""

    def test_200(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with serve(reply(200, {"status": "starting"})) as stub:
            assert ProfileManager()._check_daemon_running(stub.port) is True
            assert stub.paths == ["/health"]

    def test_non_200(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        with serve(reply(503)) as stub:
            assert ProfileManager()._check_daemon_running(stub.port) is False

    def test_down(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert ProfileManager()._check_daemon_running(closed_port()) is False
