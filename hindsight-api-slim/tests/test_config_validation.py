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
        "HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS",
        "HINDSIGHT_API_RETAIN_CHUNK_SIZE",
        "HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
        "HINDSIGHT_API_LLM_REASONING_EFFORT",
        "HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER",
        "HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER",
        "HINDSIGHT_API_SEMANTIC_MIN_SIMILARITY",
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
    assert config.retain_structured_chunk_size is None


def test_retain_structured_chunk_size_reads_from_env():
    """Structured JSONL/conversation units can have an explicit character cap."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE"] = "9000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()
    assert config.retain_structured_chunk_size == 9000


def test_fail_on_extraction_errors_defaults_to_false(monkeypatch):
    """Silent-success behavior is preserved by default (issue #2700)."""
    from hindsight_api.config import ENV_FAIL_ON_EXTRACTION_ERRORS, HindsightConfig

    monkeypatch.delenv(ENV_FAIL_ON_EXTRACTION_ERRORS, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.fail_on_extraction_errors is False


def test_fail_on_extraction_errors_reads_true_from_env(monkeypatch):
    """The opt-in escape hatch parses truthy values from the environment."""
    from hindsight_api.config import ENV_FAIL_ON_EXTRACTION_ERRORS, HindsightConfig

    monkeypatch.setenv(ENV_FAIL_ON_EXTRACTION_ERRORS, "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.fail_on_extraction_errors is True


def test_llm_ollama_num_ctx_defaults_to_none(monkeypatch):
    """Unset Ollama num_ctx override lets Ollama use its model/server default."""
    from hindsight_api.config import ENV_LLM_OLLAMA_NUM_CTX, HindsightConfig

    monkeypatch.delenv(ENV_LLM_OLLAMA_NUM_CTX, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.llm_ollama_num_ctx is None


def test_llm_ollama_num_ctx_keeps_direct_construction_default():
    """Direct HindsightConfig construction should not require the new field."""
    from dataclasses import fields

    from hindsight_api.config import HindsightConfig

    config_field = next(item for item in fields(HindsightConfig) if item.name == "llm_ollama_num_ctx")

    assert config_field.default is None
    assert config_field.kw_only


def test_llm_ollama_num_ctx_reads_positive_int(monkeypatch):
    """The native Ollama context override is parsed as a positive integer."""
    from hindsight_api.config import ENV_LLM_OLLAMA_NUM_CTX, HindsightConfig

    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "65536")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.llm_ollama_num_ctx == 65536


def test_llm_ollama_num_ctx_rejects_non_positive_values(monkeypatch):
    """Zero would be accepted by neither Ollama nor downstream range logic."""
    from hindsight_api.config import ENV_LLM_OLLAMA_NUM_CTX, HindsightConfig

    monkeypatch.setenv(ENV_LLM_OLLAMA_NUM_CTX, "0")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_LLM_OLLAMA_NUM_CTX):
        HindsightConfig.from_env()


def test_retain_structured_chunk_size_can_be_less_than_chunk_size():
    """Structured-chunk cap can be smaller than the retain chunk target."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_RETAIN_STRUCTURED_CHUNK_SIZE"] = "2000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()
    assert config.retain_chunk_size == 3000
    assert config.retain_structured_chunk_size == 2000


def test_retain_strategy_structured_chunk_size_validation():
    """Retain strategies allow structured-chunk caps below chunk size."""
    from hindsight_api.config import HindsightConfig
    from hindsight_api.config_resolver import apply_strategy

    os.environ["HINDSIGHT_API_RETAIN_MAX_COMPLETION_TOKENS"] = "64000"
    os.environ["HINDSIGHT_API_RETAIN_CHUNK_SIZE"] = "3000"
    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"

    config = HindsightConfig.from_env()
    config.retain_strategies = {
        "jsonl": {
            "retain_structured_chunk_size": 2000,
        }
    }

    resolved = apply_strategy(config, "jsonl")
    assert resolved.retain_structured_chunk_size == 2000


def test_semantic_min_similarity_reads_from_env():
    """Semantic retrieval min similarity can be configured at the server level."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_SEMANTIC_MIN_SIMILARITY"] = "0.58"

    config = HindsightConfig.from_env()
    assert config.semantic_min_similarity == 0.58


def test_semantic_min_similarity_must_be_between_zero_and_one():
    """Invalid semantic min similarity fails fast during configuration loading."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_SEMANTIC_MIN_SIMILARITY"] = "1.5"

    with pytest.raises(ValueError, match="semantic_min_similarity"):
        HindsightConfig.from_env()


def test_consolidation_max_completion_tokens_defaults_to_unset():
    """By default consolidation sends no explicit output budget (backwards compatible)."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ.pop("HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS", None)

    config = HindsightConfig.from_env()
    assert config.consolidation_max_completion_tokens is None


def test_consolidation_max_completion_tokens_env_override():
    """HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS controls consolidation LLM output budget."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    os.environ["HINDSIGHT_API_CONSOLIDATION_MAX_COMPLETION_TOKENS"] = "8192"

    config = HindsightConfig.from_env()
    assert config.consolidation_max_completion_tokens == 8192


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


def test_markitdown_ocr_defaults_disabled(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.file_parser_markitdown_ocr_enabled is False


def test_markitdown_ocr_does_not_fall_back_to_main_llm_config(monkeypatch):
    from hindsight_api.config import DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT, HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_ENABLED", "true")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "main-key")
    monkeypatch.setenv("HINDSIGHT_API_LLM_BASE_URL", "https://main.example/v1")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "main-vision-model")

    config = HindsightConfig.from_env()
    assert config.file_parser_markitdown_ocr_enabled is True
    assert config.file_parser_markitdown_ocr_api_key is None
    assert config.file_parser_markitdown_ocr_base_url is None
    assert config.file_parser_markitdown_ocr_model is None
    assert config.file_parser_markitdown_ocr_prompt == DEFAULT_FILE_PARSER_MARKITDOWN_OCR_PROMPT


def test_markitdown_ocr_uses_explicit_config(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_ENABLED", "true")
    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_API_KEY", "parser-key")
    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_BASE_URL", "https://parser.example/v1")
    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_MODEL", "parser-vision-model")
    monkeypatch.setenv("HINDSIGHT_API_FILE_PARSER_MARKITDOWN_OCR_PROMPT", "Extract this document exactly.")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "main-key")
    monkeypatch.setenv("HINDSIGHT_API_LLM_BASE_URL", "https://main.example/v1")
    monkeypatch.setenv("HINDSIGHT_API_LLM_MODEL", "main-vision-model")

    config = HindsightConfig.from_env()
    assert config.file_parser_markitdown_ocr_enabled is True
    assert config.file_parser_markitdown_ocr_api_key == "parser-key"
    assert config.file_parser_markitdown_ocr_base_url == "https://parser.example/v1"
    assert config.file_parser_markitdown_ocr_model == "parser-vision-model"
    assert config.file_parser_markitdown_ocr_prompt == "Extract this document exactly."


def test_llm_reasoning_effort_defaults_to_low(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_LLM_REASONING_EFFORT", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_reasoning_effort == "low"


def test_llm_reasoning_effort_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_reasoning_effort == "xhigh"


# ---------------------------------------------------------------------------
# Recall candidate gating (BM25 score floor + per-source cap) — issue #1707
# ---------------------------------------------------------------------------


def test_bm25_min_score_defaults_to_zero(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_BM25_MIN_SCORE", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.bm25_min_score == 0.0


def test_bm25_min_score_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_BM25_MIN_SCORE", "1.5")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.bm25_min_score == 1.5


def test_recall_max_candidates_per_source_defaults_to_disabled(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_RECALL_MAX_CANDIDATES_PER_SOURCE", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.recall_max_candidates_per_source == 0


def test_recall_max_candidates_per_source_loaded_from_env(monkeypatch):
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_RECALL_MAX_CANDIDATES_PER_SOURCE", "150")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.recall_max_candidates_per_source == 150


# ---------------------------------------------------------------------------
# Bedrock service tier (HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER)
# ---------------------------------------------------------------------------


def test_bedrock_service_tier_defaults_to_none(monkeypatch):
    """Bedrock service tier defaults to None (standard tier) when unset."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_bedrock_service_tier is None


def test_bedrock_service_tier_flex(monkeypatch):
    """Flex tier (50% cost savings) is accepted."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER", "flex")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_bedrock_service_tier == "flex"


def test_bedrock_service_tier_priority(monkeypatch):
    """Priority tier (guaranteed throughput) is accepted."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER", "priority")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_bedrock_service_tier == "priority"


def test_bedrock_service_tier_reserved(monkeypatch):
    """Reserved tier (provisioned capacity) is accepted."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER", "reserved")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_bedrock_service_tier == "reserved"


def test_bedrock_service_tier_rejects_invalid_value(monkeypatch):
    """ "standard" is not a valid Bedrock service tier and must be rejected."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER", "standard")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "HINDSIGHT_API_LLM_BEDROCK_SERVICE_TIER" in error_message
    assert "standard" in error_message
    assert "'standard' is not a valid Bedrock service tier" in error_message


# ---------------------------------------------------------------------------
# Gemini service tier (HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER)
# ---------------------------------------------------------------------------


def test_gemini_service_tier_defaults_to_none(monkeypatch):
    """Gemini service tier defaults to None (standard tier) when unset."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.delenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_gemini_service_tier is None


def test_gemini_service_tier_flex(monkeypatch):
    """Flex tier is accepted for Gemini."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", "flex")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "fake-key")

    config = HindsightConfig.from_env()
    assert config.llm_gemini_service_tier == "flex"


def test_gemini_service_tier_accepts_mixed_case_provider(monkeypatch):
    """Gemini tier parsing follows provider's case-insensitive handling."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", "flex")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "Gemini")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "fake-key")

    config = HindsightConfig.from_env()
    assert config.llm_gemini_service_tier == "flex"


def test_gemini_service_tier_rejects_invalid_value(monkeypatch):
    """Unknown Gemini service tiers are rejected early."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", "standard")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "fake-key")

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER" in error_message
    assert "standard" in error_message


def test_gemini_service_tier_ignored_for_non_gemini_provider(monkeypatch):
    """Invalid Gemini-only tiers do not break unrelated providers."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")
    monkeypatch.setenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", "standard")

    config = HindsightConfig.from_env()
    assert config.llm_gemini_service_tier is None


def test_gemini_service_tier_empty_env_is_unset(monkeypatch):
    """Empty env values are treated as unset for templated deployments."""
    from hindsight_api.config import HindsightConfig

    monkeypatch.setenv("HINDSIGHT_API_LLM_GEMINI_SERVICE_TIER", "")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()
    assert config.llm_gemini_service_tier is None


def test_operation_retention_defaults(monkeypatch):
    from hindsight_api.config import (
        ENV_OPERATION_CLEANUP_BATCH_SIZE,
        ENV_OPERATION_RETENTION_DAYS,
        HindsightConfig,
    )

    monkeypatch.delenv(ENV_OPERATION_RETENTION_DAYS, raising=False)
    monkeypatch.delenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.operation_retention_days == 0
    assert config.operation_cleanup_batch_size == 1000
    assert "operation_retention_days" in HindsightConfig.get_static_fields()
    assert "operation_cleanup_batch_size" in HindsightConfig.get_static_fields()


def test_operation_retention_env_overrides(monkeypatch):
    from hindsight_api.config import (
        ENV_OPERATION_CLEANUP_BATCH_SIZE,
        ENV_OPERATION_RETENTION_DAYS,
        HindsightConfig,
    )

    monkeypatch.setenv(ENV_OPERATION_RETENTION_DAYS, "0")
    monkeypatch.setenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, "37")
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    config = HindsightConfig.from_env()

    assert config.operation_retention_days == 0
    assert config.operation_cleanup_batch_size == 37


@pytest.mark.parametrize("raw", ["-1", "not-an-int"])
def test_operation_retention_rejects_invalid_values(monkeypatch, raw):
    from hindsight_api.config import ENV_OPERATION_RETENTION_DAYS, HindsightConfig

    monkeypatch.setenv(ENV_OPERATION_RETENTION_DAYS, raw)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_OPERATION_RETENTION_DAYS):
        HindsightConfig.from_env()


@pytest.mark.parametrize("raw", ["0", "-1", "not-an-int"])
def test_operation_cleanup_batch_size_requires_positive_integer(monkeypatch, raw):
    from hindsight_api.config import ENV_OPERATION_CLEANUP_BATCH_SIZE, HindsightConfig

    monkeypatch.setenv(ENV_OPERATION_CLEANUP_BATCH_SIZE, raw)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "mock")

    with pytest.raises(ValueError, match=ENV_OPERATION_CLEANUP_BATCH_SIZE):
        HindsightConfig.from_env()
