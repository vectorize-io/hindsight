"""Tests for ``hindsight_api.db_url.to_libpq_url``.

Covers backward compatibility (existing configs must pass through unchanged)
and the two transformations needed to support external PostgreSQL deployments
that use SQLAlchemy-style ``postgresql+asyncpg://...?ssl=require`` URLs:

1. strip the ``+asyncpg`` dialect suffix,
2. rename the ``ssl=`` query parameter to ``sslmode=``.
"""

from __future__ import annotations

import pytest

from hindsight_api.db_url import to_libpq_url


class TestPassthrough:
    """Inputs that must be returned unchanged — protects existing configs."""

    @pytest.mark.parametrize(
        "url",
        [
            "pg0",
            "",
            "postgresql://user:pass@host:5432/db",
            "postgresql://user:pass@host:5432/db?sslmode=require",
            "postgresql://user:pass@host/db?sslmode=verify-full&connect_timeout=10",
            "sqlite:///./test.db",
            "postgresql+psycopg2://user:pass@host/db",
        ],
    )
    def test_unchanged(self, url: str) -> None:
        assert to_libpq_url(url) == url


class TestSchemeNormalization:
    def test_asyncpg_scheme_stripped(self) -> None:
        assert to_libpq_url("postgresql+asyncpg://user:pass@host:5432/db") == "postgresql://user:pass@host:5432/db"

    def test_postgres_asyncpg_scheme_normalized(self) -> None:
        assert to_libpq_url("postgres+asyncpg://user:pass@host/db") == "postgresql://user:pass@host/db"

    def test_bare_postgres_scheme_normalized_to_postgresql(self) -> None:
        assert to_libpq_url("postgres://user:pass@host/db") == "postgresql://user:pass@host/db"


class TestSslParamRename:
    def test_ssl_require_to_sslmode_require(self) -> None:
        assert (
            to_libpq_url("postgresql+asyncpg://user:pass@host:5432/db?ssl=require")
            == "postgresql://user:pass@host:5432/db?sslmode=require"
        )

    @pytest.mark.parametrize("mode", ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"])
    def test_all_ssl_modes_translated(self, mode: str) -> None:
        result = to_libpq_url(f"postgresql+asyncpg://h/d?ssl={mode}")
        assert result == f"postgresql://h/d?sslmode={mode}"

    def test_ssl_rename_on_libpq_url(self) -> None:
        """Someone accidentally using SQLAlchemy-style ssl= on a libpq URL is also fixed."""
        assert to_libpq_url("postgresql://h/d?ssl=require") == "postgresql://h/d?sslmode=require"

    def test_ssl_param_preserved_among_other_params(self) -> None:
        result = to_libpq_url("postgresql+asyncpg://h/d?ssl=require&application_name=hindsight&connect_timeout=10")
        assert result.startswith("postgresql://h/d?")
        # Query order should be preserved; ssl renamed, others untouched.
        assert "sslmode=require" in result
        assert "application_name=hindsight" in result
        assert "connect_timeout=10" in result
        assert "ssl=" not in result.split("?", 1)[1].replace("sslmode=", "")

    def test_sslmode_not_double_renamed(self) -> None:
        """An already-correct sslmode= param must not be altered."""
        assert to_libpq_url("postgresql+asyncpg://h/d?sslmode=require") == "postgresql://h/d?sslmode=require"


class TestProductionConfigs:
    """Regression guard: current production URL shapes must pass through unchanged.

    These are the exact shapes currently set for HINDSIGHT_API_DATABASE_URL,
    HINDSIGHT_API_CONTROL_DATABASE_URL and HINDSIGHT_API_MIGRATION_DATABASE_URL
    in production. The helper must be a pure no-op for them so this change is
    truly backward-compatible.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql://app:pw@pg-pooler.example:5432/appdb?sslmode=disable",
            "postgresql://app:pw@pg-primary.example:5432/appdb_control?sslmode=disable",
            "postgresql://app:pw@pg-primary.example:5432/appdb?sslmode=disable",
        ],
    )
    def test_prod_urls_object_identical(self, url: str) -> None:
        # Not just equal — must be the exact same object (early-out path),
        # guaranteeing no parse/reassembly and no subtle mutation.
        assert to_libpq_url(url) is url


class TestEdgeCases:
    def test_idempotent(self) -> None:
        original = "postgresql+asyncpg://user:pass@host:5432/db?ssl=require"
        once = to_libpq_url(original)
        twice = to_libpq_url(once)
        assert once == twice

    def test_password_with_plus_is_preserved(self) -> None:
        """A naive str.replace('+asyncpg', ...) would corrupt passwords containing '+'.

        urllib.parse operates on the parsed scheme only, so this stays safe.
        """
        url = "postgresql+asyncpg://user:pa%2Bsswd@host/db?ssl=require"
        result = to_libpq_url(url)
        assert result == "postgresql://user:pa%2Bsswd@host/db?sslmode=require"

    def test_password_literal_asyncpg_in_password(self) -> None:
        """Even a password that literally contains '+asyncpg' must survive."""
        url = "postgresql+asyncpg://user:my%2Basyncpgpass@host/db"
        result = to_libpq_url(url)
        assert result == "postgresql://user:my%2Basyncpgpass@host/db"

    def test_url_without_query_string(self) -> None:
        assert to_libpq_url("postgresql+asyncpg://user:pass@host/db") == "postgresql://user:pass@host/db"

    def test_url_with_port_and_path_only(self) -> None:
        assert to_libpq_url("postgresql+asyncpg://host:5432/db") == "postgresql://host:5432/db"
