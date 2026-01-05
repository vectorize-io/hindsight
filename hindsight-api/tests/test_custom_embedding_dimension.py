"""
Tests for custom embedding dimensions and automatic dimension detection.

Uses isolated PostgreSQL schemas to avoid affecting other tests.
Includes tests for:
- Automatic embedding dimension detection and database schema adjustment
- OpenAI embeddings provider with 1536 dimensions
"""

import asyncio
import os
import pytest
from datetime import datetime
from sqlalchemy import create_engine, text

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.embeddings import LocalSTEmbeddings, OpenAIEmbeddings
from hindsight_api.engine.cross_encoder import LocalSTCrossEncoder
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.extensions import TenantExtension, TenantContext
from hindsight_api.migrations import run_migrations, ensure_embedding_dimension


# =============================================================================
# Shared Utilities
# =============================================================================


class SchemaTenantExtension(TenantExtension):
    """Tenant extension that routes all requests to a specific schema (for testing)."""

    def __init__(self, schema_name: str):
        self.schema_name = schema_name

    async def authenticate(self, request_context: RequestContext) -> TenantContext:
        return TenantContext(schema_name=self.schema_name)


def get_test_schema(prefix: str, worker_id: str) -> str:
    """Get unique schema name per xdist worker."""
    if worker_id == "master" or not worker_id:
        return prefix
    return f"{prefix}_{worker_id}"


def create_isolated_schema(db_url: str, schema_name: str, dimension: int | None = None):
    """Create an isolated schema with migrations and optional dimension adjustment."""
    engine = create_engine(db_url)

    # Create schema (drop first if exists from previous failed run)
    with engine.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {schema_name}"))
        conn.commit()

    # Run migrations in the isolated schema
    run_migrations(db_url, schema=schema_name)

    # Adjust embedding dimension if specified
    if dimension is not None:
        ensure_embedding_dimension(db_url, dimension, schema=schema_name)


def drop_schema(db_url: str, schema_name: str):
    """Drop an isolated schema."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        conn.commit()


def get_column_dimension(db_url: str, schema: str = "public") -> int | None:
    """Get the current embedding column dimension from the database."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = :schema
                  AND c.relname = 'memory_units'
                  AND a.attname = 'embedding'
            """),
            {"schema": schema},
        ).scalar()
        return result


def get_row_count(db_url: str, schema: str = "public") -> int:
    """Get the number of rows with embeddings in memory_units."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {schema}.memory_units WHERE embedding IS NOT NULL")
        ).scalar()


def insert_test_embedding(db_url: str, schema: str, dimension: int):
    """Insert a test row with a dummy embedding."""
    engine = create_engine(db_url)
    embedding = [0.1] * dimension
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    with engine.connect() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {schema}.memory_units (bank_id, text, embedding, event_date, fact_type)
                VALUES ('test-bank', 'test text', '{embedding_str}'::vector, NOW(), 'world')
            """)
        )
        conn.commit()


def clear_embeddings(db_url: str, schema: str):
    """Clear all rows from memory_units."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {schema}.memory_units"))
        conn.commit()


# =============================================================================
# Embedding Dimension Tests (Local Embeddings)
# =============================================================================


@pytest.fixture(scope="class")
def dimension_test_schema(pg0_db_url, worker_id):
    """Create an isolated schema for dimension tests."""
    schema_name = get_test_schema("test_embed_dim", worker_id)
    create_isolated_schema(pg0_db_url, schema_name)
    yield pg0_db_url, schema_name
    drop_schema(pg0_db_url, schema_name)


class TestEmbeddingDimension:
    """Tests for embedding dimension detection and adjustment."""

    def test_dimension_matches_no_change(self, dimension_test_schema):
        """When dimension matches, no changes should be made."""
        db_url, schema = dimension_test_schema

        # Get initial dimension (should be 384 from migration)
        initial_dim = get_column_dimension(db_url, schema)
        assert initial_dim == 384, f"Expected 384, got {initial_dim}"

        # Call ensure_embedding_dimension with matching dimension
        ensure_embedding_dimension(db_url, 384, schema=schema)

        # Dimension should still be 384
        assert get_column_dimension(db_url, schema) == 384

    def test_dimension_change_empty_table(self, dimension_test_schema):
        """When table is empty, dimension can be changed."""
        db_url, schema = dimension_test_schema

        # Ensure table is empty
        clear_embeddings(db_url, schema)
        assert get_row_count(db_url, schema) == 0

        # Change dimension to 768
        ensure_embedding_dimension(db_url, 768, schema=schema)

        # Verify dimension changed
        new_dim = get_column_dimension(db_url, schema)
        assert new_dim == 768, f"Expected 768, got {new_dim}"

        # Change back to 384 for other tests
        ensure_embedding_dimension(db_url, 384, schema=schema)
        assert get_column_dimension(db_url, schema) == 384

    def test_dimension_change_blocked_with_data(self, dimension_test_schema):
        """When table has data, dimension change should be blocked."""
        db_url, schema = dimension_test_schema

        # Ensure table is empty first
        clear_embeddings(db_url, schema)

        # Insert a test row with 384-dim embedding
        insert_test_embedding(db_url, schema, 384)
        assert get_row_count(db_url, schema) == 1

        # Try to change dimension - should raise error
        with pytest.raises(RuntimeError) as exc_info:
            ensure_embedding_dimension(db_url, 768, schema=schema)

        assert "Cannot change embedding dimension" in str(exc_info.value)
        assert "1 rows with embeddings" in str(exc_info.value)

        # Dimension should be unchanged
        assert get_column_dimension(db_url, schema) == 384

        # Cleanup
        clear_embeddings(db_url, schema)

    def test_local_embeddings_dimension_detection(self, embeddings):
        """Test that LocalSTEmbeddings correctly detects dimension."""
        # Initialize embeddings if not already done
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(embeddings.initialize())
        finally:
            loop.close()

        # bge-small-en-v1.5 produces 384-dim embeddings
        assert embeddings.dimension == 384

        # Verify by generating an actual embedding
        result = embeddings.encode(["test"])
        assert len(result) == 1
        assert len(result[0]) == 384


# =============================================================================
# OpenAI Embeddings Tests
# =============================================================================


def has_openai_api_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"))


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    return os.environ.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY", "")


@pytest.fixture(scope="module")
def openai_embeddings():
    """Create OpenAI embeddings instance."""
    if not has_openai_api_key():
        pytest.skip("OpenAI API key not available (set HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY)")

    embeddings = OpenAIEmbeddings(
        api_key=get_openai_api_key(),
        model="text-embedding-3-small",
    )
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(embeddings.initialize())
    finally:
        loop.close()
    return embeddings


@pytest.fixture(scope="module")
def openai_test_schema(pg0_db_url, worker_id, openai_embeddings):
    """Create an isolated schema for OpenAI embedding tests."""
    schema_name = get_test_schema("test_openai_embed", worker_id)
    create_isolated_schema(pg0_db_url, schema_name, dimension=openai_embeddings.dimension)
    yield pg0_db_url, schema_name
    drop_schema(pg0_db_url, schema_name)


@pytest.fixture
def cross_encoder():
    """Provide a cross encoder for tests."""
    return LocalSTCrossEncoder()


@pytest.fixture
def query_analyzer():
    """Provide a query analyzer for tests."""
    return DateparserQueryAnalyzer()


@pytest.fixture
def test_bank_id():
    """Provide a unique bank ID for this test run."""
    return f"openai_test_{datetime.now().timestamp()}"


@pytest.fixture
def request_context():
    """Provide a default RequestContext for tests."""
    return RequestContext()


class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings provider."""

    def test_openai_embeddings_initialization(self, openai_embeddings):
        """Test that OpenAI embeddings initializes correctly."""
        assert openai_embeddings.dimension == 1536
        assert openai_embeddings.provider_name == "openai"

    def test_openai_embeddings_encode(self, openai_embeddings):
        """Test that OpenAI embeddings can encode text."""
        texts = ["Hello, world!", "This is a test."]
        embeddings = openai_embeddings.encode(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        assert all(isinstance(x, float) for x in embeddings[0])

    @pytest.mark.asyncio
    async def test_openai_embeddings_retain_recall(
        self,
        openai_test_schema,
        openai_embeddings,
        cross_encoder,
        query_analyzer,
        test_bank_id,
        request_context,
    ):
        """Test retain and recall operations with OpenAI embeddings."""
        db_url, schema_name = openai_test_schema

        memory = MemoryEngine(
            db_url=db_url,
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
            embeddings=openai_embeddings,
            cross_encoder=cross_encoder,
            query_analyzer=query_analyzer,
            pool_min_size=1,
            pool_max_size=3,
            run_migrations=False,
            tenant_extension=SchemaTenantExtension(schema_name),
        )

        try:
            await memory.initialize()

            # Store some memories
            await memory.retain_async(
                bank_id=test_bank_id,
                content="Alice works as a software engineer at Google.",
                context="career discussion",
                request_context=request_context,
            )

            await memory.retain_async(
                bank_id=test_bank_id,
                content="Bob is a data scientist specializing in machine learning.",
                context="team introductions",
                request_context=request_context,
            )

            # Recall memories
            result = await memory.recall_async(
                bank_id=test_bank_id,
                query="Who works in technology?",
                request_context=request_context,
            )

            assert result is not None
            assert len(result.results) > 0

            memory_texts = [m.text for m in result.results]
            assert any(
                "Alice" in text or "Bob" in text or "software" in text or "data scientist" in text
                for text in memory_texts
            ), f"Expected to find relevant memories, got: {memory_texts}"

        finally:
            try:
                if memory._pool and not memory._pool._closing:
                    await memory.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_openai_embeddings_batch_retain(
        self,
        openai_test_schema,
        openai_embeddings,
        cross_encoder,
        query_analyzer,
        test_bank_id,
        request_context,
    ):
        """Test batch retain with OpenAI embeddings."""
        db_url, schema_name = openai_test_schema

        memory = MemoryEngine(
            db_url=db_url,
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
            embeddings=openai_embeddings,
            cross_encoder=cross_encoder,
            query_analyzer=query_analyzer,
            pool_min_size=1,
            pool_max_size=3,
            run_migrations=False,
            tenant_extension=SchemaTenantExtension(schema_name),
        )

        try:
            await memory.initialize()

            contents = [
                {"content": "Python is my favorite programming language.", "context": "preferences"},
                {"content": "I prefer dark mode for all my applications.", "context": "preferences"},
                {"content": "Coffee is essential for morning productivity.", "context": "habits"},
            ]

            result = await memory.retain_batch_async(
                bank_id=test_bank_id,
                contents=contents,
                request_context=request_context,
            )

            assert len(result) == 3

            recall_result = await memory.recall_async(
                bank_id=test_bank_id,
                query="What are my preferences?",
                request_context=request_context,
            )

            assert recall_result is not None
            assert len(recall_result.results) > 0

        finally:
            try:
                if memory._pool and not memory._pool._closing:
                    await memory.close()
            except Exception:
                pass
