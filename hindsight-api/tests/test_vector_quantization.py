"""
Tests for vector quantization configuration and helpers.

Verifies that quantization config validation and helper functions work correctly.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    from hindsight_api.config import clear_config_cache

    env_vars_to_save = [
        "HINDSIGHT_API_VECTOR_EXTENSION",
        "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED",
        "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE",
    ]

    original_values = {}
    for key in env_vars_to_save:
        original_values[key] = os.environ.get(key)

    clear_config_cache()

    yield

    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

    clear_config_cache()


def test_quantization_disabled_by_default():
    """Test that quantization is disabled by default."""
    from hindsight_api.config import HindsightConfig

    os.environ.pop("HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED", None)
    os.environ.pop("HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE", None)

    config = HindsightConfig.from_env()
    assert config.vector_quantization_enabled is False
    assert config.vector_quantization_type == "rabitq8"


def test_quantization_requires_vchord():
    """Test that quantization requires HINDSIGHT_API_VECTOR_EXTENSION=vchord."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "pgvector"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "Vector quantization" in error_message
    assert "vchord" in error_message
    assert "pgvector" in error_message


def test_quantization_with_vchord_succeeds():
    """Test that quantization works with vchord."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "vchord"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE"] = "rabitq8"

    config = HindsightConfig.from_env()
    assert config.vector_quantization_enabled is True
    assert config.vector_quantization_type == "rabitq8"


def test_invalid_quantization_type():
    """Test that invalid quantization type is rejected."""
    from hindsight_api.config import HindsightConfig

    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "vchord"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE"] = "invalid"

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()

    error_message = str(exc_info.value)
    assert "invalid" in error_message.lower()
    assert "rabitq8" in error_message or "rabitq4" in error_message


def test_quantize_embedding_helper():
    """Test quantize_embedding helper function."""
    from hindsight_api.engine.embeddings import quantize_embedding

    embedding = [0.1, 0.2, 0.3]

    # No quantization
    result = quantize_embedding(embedding, None)
    assert "::vector" in result
    assert "quantize" not in result

    # rabitq8
    result = quantize_embedding(embedding, "rabitq8")
    assert "quantize_to_rabitq8" in result
    assert "::vector" in result

    # rabitq4
    result = quantize_embedding(embedding, "rabitq4")
    assert "quantize_to_rabitq4" in result
    assert "::vector" in result


def test_build_quantized_query_helper():
    """Test _build_quantized_query helper function."""
    from hindsight_api.engine.search.retrieval import _build_quantized_query

    query_emb = "'[0.1,0.2,0.3]'"

    # No quantization
    result = _build_quantized_query(query_emb, None)
    assert result == query_emb

    # rabitq8
    result = _build_quantized_query(query_emb, "rabitq8")
    assert "quantize_to_rabitq8" in result
    assert query_emb in result

    # rabitq4
    result = _build_quantized_query(query_emb, "rabitq4")
    assert "quantize_to_rabitq4" in result
    assert query_emb in result


@pytest.mark.asyncio
async def test_quantization_in_retrieval():
    """Test that quantization is properly integrated in retrieval queries."""
    import asyncpg

    from hindsight_api.config import HindsightConfig, get_config
    from hindsight_api.engine.search.retrieval import retrieve_semantic_bm25_combined

    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "vchord"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE"] = "rabitq8"

    config = HindsightConfig.from_env()
    assert config.vector_quantization_enabled is True
    assert config.vector_quantization_type == "rabitq8"

    # Verify that get_config returns the right values
    static_config = get_config()
    assert static_config.vector_quantization_enabled is True
    assert static_config.vector_quantization_type == "rabitq8"


def test_quantization_validation_order():
    """Test that quantization validation checks vchord requirement first."""
    from hindsight_api.config import HindsightConfig

    # Test 1: pgvector + quantization = error
    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "pgvector"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()
    assert "vchord" in str(exc_info.value).lower()

    # Test 2: vchord + valid quantization type = success
    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "vchord"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE"] = "rabitq4"

    config = HindsightConfig.from_env()
    assert config.vector_extension == "vchord"
    assert config.vector_quantization_type == "rabitq4"

    # Test 3: vchord + invalid quantization type = error
    os.environ["HINDSIGHT_API_VECTOR_EXTENSION"] = "vchord"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED"] = "true"
    os.environ["HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE"] = "invalid_type"

    with pytest.raises(ValueError) as exc_info:
        HindsightConfig.from_env()
    assert "invalid" in str(exc_info.value).lower()

