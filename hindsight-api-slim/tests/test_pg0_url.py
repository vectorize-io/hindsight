"""Unit tests for pg0 URL parsing and MemoryEngine startup."""

from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.pg0 import Pg0Url, parse_pg0_url


class _StopInitialization(Exception):
    """Abort startup after pg0 is constructed, before opening a database pool."""


class _NoopEmbeddings:
    provider_name = "test"

    async def initialize(self) -> None:
        return None


class _NoopCrossEncoder:
    provider_name = "test"

    async def initialize(self) -> None:
        return None


class _NoopQueryAnalyzer:
    def load(self) -> None:
        return None


def test_bare_pg0():
    assert parse_pg0_url("pg0") == Pg0Url(is_pg0=True, instance_name="hindsight")


def test_named_instance():
    assert parse_pg0_url("pg0://mydb") == Pg0Url(is_pg0=True, instance_name="mydb")


def test_named_instance_with_port():
    assert parse_pg0_url("pg0://mydb:5544") == Pg0Url(is_pg0=True, instance_name="mydb", port=5544)


def test_empty_instance_falls_back_to_default():
    assert parse_pg0_url("pg0://").instance_name == "hindsight"
    assert parse_pg0_url("pg0://:5544") == Pg0Url(is_pg0=True, instance_name="hindsight", port=5544)


def test_non_pg0_url_passthrough():
    parsed = parse_pg0_url("postgresql://user:pwd@localhost:5432/db")
    assert parsed == Pg0Url(is_pg0=False)


def test_credentials_user_and_password():
    assert parse_pg0_url("pg0://alice:s3cret@mydb:5544") == Pg0Url(
        is_pg0=True,
        instance_name="mydb",
        port=5544,
        username="alice",
        password="s3cret",
    )


def test_credentials_user_only():
    assert parse_pg0_url("pg0://alice@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password=None
    )


def test_credentials_without_port():
    assert parse_pg0_url("pg0://alice:s3cret@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password="s3cret"
    )


def test_password_may_contain_at_sign():
    # rsplit on the last "@" keeps an "@" inside the password intact.
    assert parse_pg0_url("pg0://alice:p@ss@mydb:5544") == Pg0Url(
        is_pg0=True,
        instance_name="mydb",
        port=5544,
        username="alice",
        password="p@ss",
    )


def test_empty_password_after_colon_is_empty_string():
    # "user:" explicitly sets an empty password (distinct from omitting it).
    assert parse_pg0_url("pg0://alice:@mydb") == Pg0Url(
        is_pg0=True, instance_name="mydb", username="alice", password=""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("db_url", "expected_username", "expected_password"),
    [
        ("pg0://alice:s3cret@mydb:5544", "alice", "s3cret"),
        ("pg0://alice:@mydb:5544", "alice", ""),
        ("pg0://mydb:5544", None, None),
    ],
)
async def test_memory_engine_forwards_pg0_credentials(
    db_url: str,
    expected_username: str | None,
    expected_password: str | None,
) -> None:
    """The primary server startup must honor the same URL contract as the parser."""
    with patch("hindsight_api.engine.memory_engine.EmbeddedPostgres") as embedded_postgres:
        pg0 = embedded_postgres.return_value
        pg0.is_running = AsyncMock(return_value=True)
        pg0.ensure_running = AsyncMock(return_value="postgresql://resolved")

        engine = MemoryEngine(
            db_url=db_url,
            memory_llm_provider="none",
            memory_llm_model="none",
            embeddings=_NoopEmbeddings(),
            cross_encoder=_NoopCrossEncoder(),
            query_analyzer=_NoopQueryAnalyzer(),
            run_migrations=False,
            task_backend=SyncTaskBackend(),
            skip_llm_verification=True,
        )
        assert engine._backend is not None
        engine._backend.initialize = AsyncMock(side_effect=_StopInitialization)  # type: ignore[method-assign]

        with pytest.raises(_StopInitialization):
            await engine.initialize()

    if expected_username is None:
        embedded_postgres.assert_called_once_with(name="mydb", port=5544)
    else:
        embedded_postgres.assert_called_once_with(
            name="mydb",
            port=5544,
            username=expected_username,
            password=expected_password,
        )
