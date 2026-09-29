"""Config parsing for HINDSIGHT_API_DB_MAX_PARALLEL_WORKERS_PER_GATHER.

The flag is optional and static (server-level): unset means "leave the
Postgres server default untouched"; 0 is a meaningful value (disable planner
parallelism on this process's pool connections). These tests pin the parse
semantics so a refactor can't silently turn "unset" into 0 or reject 0.
"""

import pytest

from hindsight_api.config import (
    ENV_DB_MAX_PARALLEL_WORKERS_PER_GATHER,
    HindsightConfig,
    _parse_optional_non_negative_int,
)


class TestParseOptionalNonNegativeInt:
    def test_unset_returns_none(self):
        assert _parse_optional_non_negative_int("X", None) is None

    def test_empty_returns_none(self):
        assert _parse_optional_non_negative_int("X", "") is None

    def test_zero_is_valid(self):
        # 0 = disable planner parallelism; must NOT be treated as unset.
        assert _parse_optional_non_negative_int("X", "0") == 0

    def test_positive_is_valid(self):
        assert _parse_optional_non_negative_int("X", "2") == 2

    def test_negative_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            _parse_optional_non_negative_int("X", "-1")

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_optional_non_negative_int("X", "two")


class TestFromEnv:
    def test_default_is_none(self, monkeypatch):
        monkeypatch.delenv(ENV_DB_MAX_PARALLEL_WORKERS_PER_GATHER, raising=False)
        config = HindsightConfig.from_env()
        assert config.db_max_parallel_workers_per_gather is None

    def test_env_zero(self, monkeypatch):
        monkeypatch.setenv(ENV_DB_MAX_PARALLEL_WORKERS_PER_GATHER, "0")
        config = HindsightConfig.from_env()
        assert config.db_max_parallel_workers_per_gather == 0

    def test_env_positive(self, monkeypatch):
        monkeypatch.setenv(ENV_DB_MAX_PARALLEL_WORKERS_PER_GATHER, "1")
        config = HindsightConfig.from_env()
        assert config.db_max_parallel_workers_per_gather == 1

    def test_env_invalid_fails_fast(self, monkeypatch):
        monkeypatch.setenv(ENV_DB_MAX_PARALLEL_WORKERS_PER_GATHER, "-2")
        with pytest.raises(ValueError):
            HindsightConfig.from_env()
