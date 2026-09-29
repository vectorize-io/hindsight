"""The one piece of logic here that can be wrong without anyone noticing.

A remote target that quietly falls back to starting a local server would still
produce a green run — against the wrong system. So: given a URL, nothing is
started, and the key travels with it.
"""

from __future__ import annotations

import pytest

from hindsight_system_evals.target import ENV_API_KEY, ENV_API_URL, eval_target


def test_explicit_url_starts_no_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_KEY, "k")
    monkeypatch.delenv(ENV_API_URL, raising=False)
    with eval_target("https://api.dev.example/") as target:
        assert target.url == "https://api.dev.example"  # trailing slash would double up in paths
        assert target.api_key == "k"
        assert target.is_remote


def test_env_url_is_the_same_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_URL, "http://localhost:8888")
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with eval_target() as target:
        assert target.url == "http://localhost:8888"
        assert target.api_key is None
        assert target.is_remote


class _FakeServer:
    url = "http://127.0.0.1:1"

    def __init__(self) -> None:
        self.stopped = False

    def logs(self) -> str:
        return ""

    def stop(self) -> None:
        self.stopped = True


def test_empty_url_falls_through_to_a_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable exported as "" must not read as "point at nothing"."""
    monkeypatch.setenv(ENV_API_URL, "")
    server = _FakeServer()
    monkeypatch.setattr("hindsight_system_evals.target.start_eval_server", lambda **_: server)
    with eval_target() as target:
        assert target.url == server.url
        assert not target.is_remote
    assert server.stopped, "a server we started has to be stopped, green or red"
