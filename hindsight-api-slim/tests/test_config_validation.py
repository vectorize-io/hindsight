"""
Tests for configuration validation.

Verifies that config validation catches invalid parameter combinations.
"""

import logging
import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from hindsight_api.config import clear_config_cache

    # Save original environment values
    env_vars_to_save = [
        "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS",
        "HINDSIGHT_API_RETAIN_CHUNK_SIZE",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
        "HINDSIGHT_API_DATABASE_URL",
        "HINDSIGHT_API_MIGRATION_DATABASE_URL",
    ]

    # Save original values
    original_values = {}
    for key in env_vars_to_save:
        original_values[key] = os.environ.get(key)

    clear_config_cache()

    yield

    # Restore original environment
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_retain_max_completion_tokens_must_be_greater_than_chunk_size():
    """Test that RETAIN_MAX_COMPLETION_TOKENS > RETAIN_CHUNK_SIZE validation works."""
    from hindsight_api.config import HindsightConfig

    # Set invalid config: max_completion_tokens <= chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "1000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "2000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)

    # Verify error message contains helpful information
    assert "HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "1000" in error_message
    assert "HINDSIGHT_API_RETAIN_CHUNK_SIZE" in error_message
    assert "2000" in error_message
    assert "must be greater than" in error_message
    assert "You have two options to fix this:" in error_message
    assert "Increase HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS" in error_message
    assert "Use a model that supports" in error_message


def test_retain_max_completion_tokens_equal_to_chunk_size_fails():
    """Test that RETAIN_MAX_COMPLETION_TOKENS == RETAIN_CHUNK_SIZE also fails."""
    from hindsight_api.config import HindsightConfig

    # Set invalid config: max_completion_tokens == chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "3000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "must be greater than" in error_message


def test_valid_retain_config_succeeds():
    """Test that valid config with max_completion_tokens > chunk_size works."""
    from hindsight_api.config import HindsightConfig

    # Set valid config: max_completion_tokens > chunk_size
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    # Should not raise
    config = HindsightConfig.from_env()
    assert config.retain_max_completion_tokens == 64000
    assert config.retain_chunk_size == 3000


def test_log_config_masks_database_urls(caplog):
    """Config startup logs must not expose database credentials."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_DATABASE_URL"] = "postgresql://hindsight_user:plain-password@db:5432/hindsight_db"
    os.environ["HINDSIGHT_API_MIGRATION_DATABASE_URL"] = (
        "postgresql://migration_user:migration-password@db-admin:5432/hindsight_db"
    )
    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    caplog.set_level(logging.INFO, logger="hindsight_api.config")

    config = HindsightConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "hindsight_user" not in log_output
    assert "plain-password" not in log_output
    assert "migration_user" not in log_output
    assert "migration-password" not in log_output
    assert "postgresql://***:***@db:5432/hindsight_db" in log_output
    assert "postgresql://***:***@db-admin:5432/hindsight_db" in log_output


def test_read_database_url_defaults_to_none_when_unset(monkeypatch):
    """Without HINDSIGHT_API_READ_DATABASE_URL, the field is None — engine
    will alias the read backend to the primary, preserving today's
    single-pool behaviour byte-for-bit. This is the most important guarantee
    of the change: zero-config means zero behaviour change.
    """
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_READ_DATABASE_URL", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.read_database_url is None


def test_read_database_url_is_loaded_when_set(monkeypatch):
    """When HINDSIGHT_API_READ_DATABASE_URL is set, the value flows into
    config so MemoryEngine.initialize() will open a second backend against
    that URL for recall queries.
    """
    from hindsight_api.config import HindsightConfig

    read_url = "postgresql://reader:secret@replica.example:5432/hindsight"
    monkeypatch.setenv("HINDSIGHT_API_READ_DATABASE_URL", read_url)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.read_database_url == read_url


def test_read_database_url_empty_string_is_treated_as_unset(monkeypatch):
    """Helm sometimes renders an unset env var as the empty string. Treat it
    the same as unset so deployments that conditionally set the var don't
    accidentally try to open a pool against `''`.
    """
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_READ_DATABASE_URL", "")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.read_database_url is None


def test_log_config_masks_read_database_url(monkeypatch, caplog):
    """Read-replica URL credentials must be masked in startup logs, same as
    the primary URL.
    """
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_DATABASE_URL", "postgresql://hindsight:pw@primary:5432/db")
    monkeypatch.setenv("HINDSIGHT_API_READ_DATABASE_URL", "postgresql://reader:replica-secret@replica:5432/db")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS", "64000")
    monkeypatch.setenv("HINDSIGHT_API_RETAIN_CHUNK_SIZE", "3000")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    caplog.set_level(logging.INFO, logger="hindsight_api.config")

    config = HindsightConfig.from_env()
    config.log_config()

    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert "reader" not in log_output
    assert "replica-secret" not in log_output
    assert "Read database" in log_output
    assert "postgresql://***:***@replica:5432/db" in log_output


# Note: The BadRequestError wrapping is implemented in fact_extraction.py
# but requires a complex integration test setup. The functionality is
# straightforward: when a BadRequestError containing keywords like
# "max_tokens", "max_completion_tokens", or "maximum context" is caught,
# it's wrapped in a ValueError with helpful guidance.
#
# The config validation tests above ensure users get early feedback
# about invalid configurations before runtime errors occur.


# ---------------------------------------------------------------------------
# Multilingual BM25 configuration
# ---------------------------------------------------------------------------


def test_native_language_defaults_to_english(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_native_language == "english"


def test_native_language_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", "french")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_native_language == "french"


def test_native_language_lowercased(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", "Spanish")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_native_language == "spanish"


@pytest.mark.parametrize(
    "bad_value",
    ["en glish", "english;DROP TABLE", "english'", "1english", "english-extra", ""],
)
def test_native_language_rejects_invalid_identifiers(monkeypatch, bad_value):
    """text_search_extension_native_language is embedded into raw SQL — non-identifiers must be rejected."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_NATIVE_LANGUAGE", bad_value)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match="Invalid text_search_extension_native_language"):
        HindsightConfig.from_env()


def test_text_search_extension_accepts_pgroonga(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION", "pgroonga")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension == "pgroonga"


def test_text_search_extension_rejects_unknown(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION", "bogus")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match="Invalid text_search_extension"):
        HindsightConfig.from_env()


def test_pg_search_tokenizer_defaults_to_empty(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_pg_search_tokenizer == ""


def test_pg_search_tokenizer_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER", "Jieba")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_pg_search_tokenizer == "jieba"


def test_pg_search_tokenizer_accepts_lindera_alias(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER", "chinese_lindera")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.text_search_extension_pg_search_tokenizer == "lindera(chinese)"


@pytest.mark.parametrize("bad_value", ["jieba;DROP TABLE", "ngram(3,2)", "unknown", "pdb.jieba"])
def test_pg_search_tokenizer_rejects_invalid_values(monkeypatch, bad_value):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER", bad_value)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match="Invalid HINDSIGHT_API_TEXT_SEARCH_EXTENSION_PG_SEARCH_TOKENIZER"):
        HindsightConfig.from_env()


def test_pg_search_bm25_columns_apply_tokenizer():
    from hindsight_api._pg_search import pg_search_bm25_columns

    assert pg_search_bm25_columns("id", ("text", "context"), "") == "id, text, context"
    assert pg_search_bm25_columns("id", ("text", "context"), "jieba") == "id, (text::pdb.jieba), (context::pdb.jieba)"
    assert pg_search_bm25_columns("id", ("text",), "ngram(2, 3)") == "id, (text::pdb.ngram(2,3))"
    assert pg_search_bm25_columns("id", ("text",), "edge_ngram(2, 5)") == "id, (text::pdb.edge_ngram(2,5))"


def test_llm_output_language_defaults_to_none(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_LLM_OUTPUT_LANGUAGE", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_output_language is None


def test_llm_output_language_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_OUTPUT_LANGUAGE", "Japanese")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_output_language == "Japanese"


def test_llm_output_language_empty_string_is_unset(monkeypatch):
    """Empty env var (e.g. from Helm) should be treated as unset, not literal ''."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_OUTPUT_LANGUAGE", "")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_output_language is None
