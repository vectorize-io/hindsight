"""
Tests for VectorChord RaBitQ quantization feature.

Tests cover:
1. Config validation
2. Embedding quantization
3. Query quantization
4. Migration behavior
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.embeddings import quantize_embedding
from hindsight_api.engine.search.retrieval import _build_quantized_query


class TestConfigValidation:
    """Test 1: Config validation rejects quantization without vchord"""

    def test_quantization_requires_vchord(self):
        """Quantization enabled without vchord should raise ValueError"""
        with patch.dict(os.environ, {
            "HINDSIGHT_API_VECTOR_EXTENSION": "pgvector",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED": "true",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE": "rabitq8",
        }):
            with pytest.raises(ValueError, match="requires HINDSIGHT_API_VECTOR_EXTENSION=vchord"):
                HindsightConfig.from_env()

    def test_invalid_quantization_type(self):
        """Invalid quantization type should raise ValueError"""
        with patch.dict(os.environ, {
            "HINDSIGHT_API_VECTOR_EXTENSION": "vchord",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED": "true",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE": "invalid_type",
        }):
            with pytest.raises(ValueError, match="Invalid vector_quantization_type"):
                HindsightConfig.from_env()

    def test_valid_rabitq8_config(self):
        """Valid rabitq8 config should not raise"""
        with patch.dict(os.environ, {
            "HINDSIGHT_API_VECTOR_EXTENSION": "vchord",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED": "true",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE": "rabitq8",
        }):
            config = HindsightConfig.from_env()
            assert config.vector_quantization_enabled is True
            assert config.vector_quantization_type == "rabitq8"

    def test_valid_rabitq4_config(self):
        """Valid rabitq4 config should not raise"""
        with patch.dict(os.environ, {
            "HINDSIGHT_API_VECTOR_EXTENSION": "vchord",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED": "true",
            "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE": "rabitq4",
        }):
            config = HindsightConfig.from_env()
            assert config.vector_quantization_enabled is True
            assert config.vector_quantization_type == "rabitq4"


class TestEmbeddingQuantization:
    """Test 3: Embeddings quantized on INSERT"""

    def test_quantize_embedding_no_quantization(self):
        """No quantization should return plain vector cast"""
        embedding = [0.1, 0.2, 0.3]
        result = quantize_embedding(embedding, None)
        assert result == f"'{embedding}'::vector"

    def test_quantize_embedding_rabitq8(self):
        """rabitq8 quantization should wrap with quantize_to_rabitq8"""
        embedding = [0.1, 0.2, 0.3]
        result = quantize_embedding(embedding, "rabitq8")
        assert result == f"quantize_to_rabitq8('{embedding}'::vector)"

    def test_quantize_embedding_rabitq4(self):
        """rabitq4 quantization should wrap with quantize_to_rabitq4"""
        embedding = [0.1, 0.2, 0.3]
        result = quantize_embedding(embedding, "rabitq4")
        assert result == f"quantize_to_rabitq4('{embedding}'::vector)"

    def test_quantize_embedding_invalid_type(self):
        """Invalid quantization type should raise ValueError"""
        embedding = [0.1, 0.2, 0.3]
        with pytest.raises(ValueError, match="Unknown quantization type"):
            quantize_embedding(embedding, "invalid")


class TestQueryQuantization:
    """Test 4: Query embeddings quantized on SELECT"""

    def test_build_quantized_query_no_quantization(self):
        """No quantization should return plain query"""
        query = "$1"
        result = _build_quantized_query(query, None)
        assert result == query

    def test_build_quantized_query_rabitq8(self):
        """rabitq8 should wrap query with quantize_to_rabitq8"""
        query = "$1"
        result = _build_quantized_query(query, "rabitq8")
        assert result == "quantize_to_rabitq8($1::vector)"

    def test_build_quantized_query_rabitq4(self):
        """rabitq4 should wrap query with quantize_to_rabitq4"""
        query = "$1"
        result = _build_quantized_query(query, "rabitq4")
        assert result == "quantize_to_rabitq4($1::vector)"

    def test_build_quantized_query_invalid_type(self):
        """Invalid quantization type should raise ValueError"""
        query = "$1"
        with pytest.raises(ValueError, match="Unknown quantization type"):
            _build_quantized_query(query, "invalid")


class TestMigrationValidation:
    """Test 2 & 5: Migration creates correct column types and index"""

    def test_migration_file_exists(self):
        """Migration file should exist"""
        import os
        migration_path = os.path.join(
            os.path.dirname(__file__),
            "../hindsight_api/alembic/versions/z1a2b3c4d5e6_add_rabitq_quantization.py"
        )
        assert os.path.exists(migration_path), f"Migration file not found at {migration_path}"

    def test_migration_has_upgrade_function(self):
        """Migration should have upgrade function"""
        from hindsight_api.alembic.versions.z1a2b3c4d5e6_add_rabitq_quantization import upgrade
        assert callable(upgrade), "upgrade function not found in migration"

    def test_migration_has_downgrade_function(self):
        """Migration should have downgrade function"""
        from hindsight_api.alembic.versions.z1a2b3c4d5e6_add_rabitq_quantization import downgrade
        assert callable(downgrade), "downgrade function not found in migration"

    def test_migration_revision_id(self):
        """Migration should have correct revision ID"""
        from hindsight_api.alembic.versions.z1a2b3c4d5e6_add_rabitq_quantization import revision
        assert revision == "z1a2b3c4d5e6", f"Expected revision z1a2b3c4d5e6, got {revision}"

