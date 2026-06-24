"""Tests for container-runtime detection used to warn about unstable worker ids."""

import builtins

from hindsight_api.utils import detect_container_runtime, warn_if_container_default_worker_id


def test_detects_kubernetes_via_env(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    assert detect_container_runtime() == "kubernetes"


def test_detects_docker_via_dockerenv(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: p == "/.dockerenv")
    assert detect_container_runtime() == "docker"


def test_detects_docker_via_cgroup(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: False)

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/1/cgroup":
            import io

            return io.StringIO("12:devices:/docker/abcdef123456\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert detect_container_runtime() == "docker"


def test_returns_none_when_not_containerized(monkeypatch):
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setattr("os.path.exists", lambda p: False)

    def fake_open(path, *args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr("builtins.open", fake_open)
    assert detect_container_runtime() is None


def test_warns_when_default_worker_id_is_used_in_container(monkeypatch, caplog):
    monkeypatch.setattr("hindsight_api.utils.detect_container_runtime", lambda: "docker")

    warn_if_container_default_worker_id(None)

    assert "HINDSIGHT_API_WORKER_ID is not set" in caplog.text
    assert "appears to be running inside docker" in caplog.text


def test_skips_warning_when_worker_id_is_explicit(monkeypatch, caplog):
    monkeypatch.setattr("hindsight_api.utils.detect_container_runtime", lambda: "docker")

    warn_if_container_default_worker_id("worker-1")

    assert caplog.text == ""


def test_skips_warning_outside_containers(monkeypatch, caplog):
    monkeypatch.setattr("hindsight_api.utils.detect_container_runtime", lambda: None)

    warn_if_container_default_worker_id(None)

    assert caplog.text == ""
