"""Config parsing for HINDSIGHT_API_ENTITY_TRGM_SIMILARITY_THRESHOLD.

Static (server-level) float applied as ``SET pg_trgm.similarity_threshold`` on
every pool connection. pg_trgm only accepts a threshold in (0, 1], so an
out-of-range value must fail fast at config load rather than break every
connection's setup. These tests pin the default, the parse, and the bounds.
"""

import pytest

from hindsight_api.config import (
    DEFAULT_ENTITY_TRGM_SIMILARITY_THRESHOLD,
    ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD,
    HindsightConfig,
)


class TestEntityTrgmThresholdConfig:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, raising=False)
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == DEFAULT_ENTITY_TRGM_SIMILARITY_THRESHOLD

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, "0.3")
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == 0.3

    def test_upper_bound_one_is_valid(self, monkeypatch):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, "1.0")
        config = HindsightConfig.from_env()
        assert config.entity_trgm_similarity_threshold == 1.0

    @pytest.mark.parametrize("value", ["0", "0.0", "-0.1", "1.5"])
    def test_out_of_range_fails_fast(self, monkeypatch, value):
        monkeypatch.setenv(ENV_ENTITY_TRGM_SIMILARITY_THRESHOLD, value)
        with pytest.raises(ValueError, match="entity_trgm_similarity_threshold"):
            HindsightConfig.from_env()
