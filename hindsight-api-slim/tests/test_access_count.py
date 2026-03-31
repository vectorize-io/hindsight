"""
Tests for retrieval frequency tracking (access_count) feature.

Tests the response model, migration structure, and API response format
for the access_count field on memory units.
"""

import importlib
from collections.abc import Sequence
from pathlib import Path

import pytest


class TestAccessCountResponseModels:
    """Test access_count is properly defined in response models."""

    def test_memory_fact_has_access_count_field(self):
        """MemoryFact core model should have access_count field."""
        from hindsight_api.engine.response_models import MemoryFact

        # Create a MemoryFact with access_count
        fact = MemoryFact(
            id="test-id",
            text="some memory",
            fact_type="world",
            access_count=5,
        )
        assert fact.access_count == 5

    def test_memory_fact_access_count_defaults_to_none(self):
        """access_count should default to None when not provided."""
        from hindsight_api.engine.response_models import MemoryFact

        fact = MemoryFact(
            id="test-id",
            text="some memory",
            fact_type="world",
        )
        assert fact.access_count is None

    def test_memory_fact_access_count_in_json_schema(self):
        """access_count should appear in the JSON schema."""
        from hindsight_api.engine.response_models import MemoryFact

        schema = MemoryFact.model_json_schema()
        assert "access_count" in schema["properties"]
        prop = schema["properties"]["access_count"]
        assert "integer" in str(prop.get("anyOf", prop.get("type", "")))

    def test_memory_fact_serialization_includes_access_count(self):
        """access_count should be included in model_dump output."""
        from hindsight_api.engine.response_models import MemoryFact

        fact = MemoryFact(
            id="test-id",
            text="some memory",
            fact_type="world",
            access_count=42,
        )
        dumped = fact.model_dump()
        assert dumped["access_count"] == 42

    def test_memory_fact_serialization_includes_null_access_count(self):
        """access_count=None should still be present in serialized output."""
        from hindsight_api.engine.response_models import MemoryFact

        fact = MemoryFact(
            id="test-id",
            text="some memory",
            fact_type="world",
        )
        dumped = fact.model_dump()
        assert "access_count" in dumped
        assert dumped["access_count"] is None


class TestAccessCountAPIModel:
    """Test access_count in the HTTP API response model."""

    def test_api_recall_result_has_access_count(self):
        """HTTP API RecallResult should have access_count field."""
        from hindsight_api.api.http import RecallResult

        result = RecallResult(
            id="test-id",
            text="some memory",
            type="world",
            access_count=10,
        )
        assert result.access_count == 10

    def test_api_recall_result_access_count_defaults_to_none(self):
        """HTTP API RecallResult access_count should default to None."""
        from hindsight_api.api.http import RecallResult

        result = RecallResult(
            id="test-id",
            text="some memory",
            type="world",
        )
        assert result.access_count is None


class TestAccessCountMigration:
    """Test the migration file structure."""

    def test_migration_file_exists(self):
        """Migration file for access_count should exist."""
        migration_dir = (
            Path(__file__).parent.parent
            / "hindsight_api"
            / "alembic"
            / "versions"
        )
        migration_file = migration_dir / "f6a7b8c9d0e1_add_access_count_to_memory_units.py"
        assert migration_file.exists(), f"Migration file not found: {migration_file}"

    def test_migration_has_correct_revision(self):
        """Migration should have the expected revision ID."""
        from hindsight_api.alembic.versions.f6a7b8c9d0e1_add_access_count_to_memory_units import (
            revision,
            down_revision,
        )

        assert revision == "f6a7b8c9d0e1"
        # Should depend on existing heads (merge migration)
        assert isinstance(down_revision, (tuple, list))

    def test_migration_has_upgrade_and_downgrade(self):
        """Migration should have both upgrade() and downgrade() functions."""
        from hindsight_api.alembic.versions import (
            f6a7b8c9d0e1_add_access_count_to_memory_units as migration,
        )

        assert hasattr(migration, "upgrade")
        assert hasattr(migration, "downgrade")
        assert callable(migration.upgrade)
        assert callable(migration.downgrade)


class TestAccessCountORMModel:
    """Test access_count in the SQLAlchemy ORM model."""

    def test_memory_unit_model_has_access_count(self):
        """MemoryUnit ORM model should have access_count column."""
        from hindsight_api.models import MemoryUnit

        # Check the column exists in the model
        assert hasattr(MemoryUnit, "access_count")

    def test_memory_unit_access_count_column_properties(self):
        """access_count column should have correct properties."""
        from hindsight_api.models import MemoryUnit

        col = MemoryUnit.__table__.columns["access_count"]
        assert not col.nullable
        assert str(col.server_default.arg) in ("0", "'0'")

    def test_memory_unit_has_access_count_index(self):
        """MemoryUnit should have an index on access_count."""
        from hindsight_api.models import MemoryUnit

        index_names = [idx.name for idx in MemoryUnit.__table__.indexes]
        assert "idx_memory_units_access_count" in index_names
