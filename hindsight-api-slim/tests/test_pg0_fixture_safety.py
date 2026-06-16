"""Regression tests for the shared pg0 pytest fixture safety guard."""

from pathlib import Path

import pytest

from tests import conftest


class _TmpPathFactory:
    def __init__(self, base: Path) -> None:
        self._base = base

    def getbasetemp(self) -> Path:
        return self._base


@pytest.mark.parametrize(
    "db_url",
    ["pg0", "pg0://", "pg0://hindsight", "pg0://hindsight:5556"],
)
def test_pg0_db_url_redirects_live_hindsight_aliases_to_test_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_url: str
) -> None:
    """Bare/default pg0 must resolve to the shared test instance, never the live one."""

    cleanup_calls: list[str] = []

    class RecordingEmbeddedPostgres:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.name = kwargs.get("name")
            self.port = kwargs.get("port")

        async def ensure_running(self) -> str:
            assert self.name == conftest.DEFAULT_PG0_INSTANCE_NAME
            assert self.port == conftest.DEFAULT_PG0_PORT
            return "postgresql://test-instance"

    monkeypatch.setattr(conftest, "EmbeddedPostgres", RecordingEmbeddedPostgres)
    monkeypatch.setattr(conftest, "_cleanup_stale_test_data", cleanup_calls.append)
    monkeypatch.setattr("hindsight_api.migrations.run_migrations", lambda _url: None)

    url = conftest.pg0_db_url.__wrapped__(db_url, _TmpPathFactory(tmp_path), "master")
    assert url == "postgresql://test-instance"
    assert cleanup_calls == ["postgresql://test-instance"]


def test_resolve_test_pg0_target_allows_named_test_instance() -> None:
    target = conftest._resolve_test_pg0_target("pg0://hindsight-test:5557")

    assert target.name == "hindsight-test"
    assert target.port == 5557


def test_resolve_test_pg0_target_defaults_to_test_instance_when_env_unset() -> None:
    target = conftest._resolve_test_pg0_target(None)

    assert target.name == "hindsight-test"
    assert target.port == conftest.DEFAULT_PG0_PORT
