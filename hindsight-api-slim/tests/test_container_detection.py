"""Tests for container-runtime detection used to warn about unstable worker ids."""

import builtins

from hindsight_api.utils import detect_container_runtime


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
