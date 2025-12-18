"""
Tests for multi-tenant schema isolation.

Verifies that concurrent retain operations from different tenants
are properly isolated in their respective PostgreSQL schemas.
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

from hindsight_api.extensions import RequestContext, TenantContext, TenantExtension
from hindsight_api.engine.memory_engine import _current_schema, fq_table
from hindsight_api.migrations import run_migrations


class MultiSchemaTestTenantExtension(TenantExtension):
    """
    Test tenant extension that maps API keys to schema names.

    API keys are in format: "key-{schema_name}"
    Provisions schemas on first access using run_migrations(schema=name).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.db_url = config.get("db_url")
        # Pre-configured valid schemas for test
        self.valid_schemas = config.get("valid_schemas", set())
        # Track provisioned schemas
        self._provisioned: set[str] = set()

    async def authenticate(self, context: RequestContext) -> TenantContext:
        if not context.api_key:
            from hindsight_api.extensions import AuthenticationError

            raise AuthenticationError("API key required")

        # Parse schema from API key (format: "key-{schema}")
        if context.api_key.startswith("key-"):
            schema = context.api_key[4:]  # Remove "key-" prefix
            if schema in self.valid_schemas:
                # Provision schema on first access
                if schema not in self._provisioned and self.db_url:
                    run_migrations(self.db_url, schema=schema)
                    self._provisioned.add(schema)
                return TenantContext(schema_name=schema)

        from hindsight_api.extensions import AuthenticationError

        raise AuthenticationError(f"Unknown API key: {context.api_key}")


async def drop_schema(conn, schema_name: str) -> None:
    """Drop a schema and all its contents."""
    await conn.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')


async def count_memories_in_schema(conn, schema_name: str, bank_id: str) -> int:
    """Count memory units in a specific schema for a bank."""
    result = await conn.fetchval(
        f'SELECT COUNT(*) FROM "{schema_name}".memory_units WHERE bank_id = $1',
        bank_id,
    )
    return result or 0


async def get_memory_texts_in_schema(conn, schema_name: str, bank_id: str) -> list[str]:
    """Get all memory texts in a specific schema for a bank."""
    rows = await conn.fetch(
        f'SELECT text FROM "{schema_name}".memory_units WHERE bank_id = $1 ORDER BY text',
        bank_id,
    )
    return [row["text"] for row in rows]


class TestSchemaIsolation:
    """Tests for multi-tenant schema isolation."""

    @pytest.mark.asyncio
    async def test_concurrent_inserts_isolated_by_schema(self, memory, pg0_db_url):
        """
        Multiple concurrent database operations from different tenants
        should store data in their respective schemas without cross-contamination.

        Uses run_migrations(schema=x) to provision schemas like a real extension.
        """
        import asyncpg

        # Test schemas
        schemas = ["tenant_alpha", "tenant_beta", "tenant_gamma"]
        bank_id = f"test-isolation-{uuid.uuid4().hex[:8]}"

        # Clean up any existing schemas
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                await drop_schema(conn, schema)
        finally:
            await conn.close()

        # Configure tenant extension that provisions schemas via run_migrations
        tenant_ext = MultiSchemaTestTenantExtension({
            "db_url": pg0_db_url,
            "valid_schemas": set(schemas),
        })
        memory._tenant_extension = tenant_ext

        # Define concurrent insert tasks for each tenant
        async def insert_for_tenant(schema_name: str, content_prefix: str):
            """Insert memories for a specific tenant using schema context."""
            # Authenticate to set the schema context
            tenant_request = RequestContext(api_key=f"key-{schema_name}")
            await memory._authenticate_tenant(tenant_request)

            # Now fq_table will use the correct schema
            pool = await memory._get_pool()
            from hindsight_api.engine.db_utils import acquire_with_retry

            async with acquire_with_retry(pool) as conn:
                # Insert 3 memories for this tenant
                for i in range(3):
                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table('memory_units')} (bank_id, text, event_date, fact_type)
                        VALUES ($1, $2, now(), 'world')
                        """,
                        bank_id,
                        f"MARKER_{content_prefix}_DOC{i}: Memory for {schema_name}",
                    )

        # Run concurrent inserts for all tenants
        await asyncio.gather(
            insert_for_tenant("tenant_alpha", "ALPHA"),
            insert_for_tenant("tenant_beta", "BETA"),
            insert_for_tenant("tenant_gamma", "GAMMA"),
        )

        # Verify isolation - each schema should only have its own data
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                texts = await get_memory_texts_in_schema(conn, schema, bank_id)
                prefix = schema.replace("tenant_", "").upper()

                # Should have exactly 3 memories
                assert len(texts) == 3, f"Schema {schema} should have 3 memories, got {len(texts)}"

                # All texts should contain the schema's marker
                for text in texts:
                    assert f"MARKER_{prefix}" in text, (
                        f"Memory in {schema} missing its marker: {text}"
                    )

                # Should NOT contain other tenants' markers
                other_prefixes = ["ALPHA", "BETA", "GAMMA"]
                other_prefixes.remove(prefix)
                for other in other_prefixes:
                    for text in texts:
                        assert f"MARKER_{other}" not in text, (
                            f"Cross-contamination! Schema {schema} has {other}'s marker: {text}"
                        )

        finally:
            # Cleanup
            for schema in schemas:
                await drop_schema(conn, schema)
            await conn.close()

        # Reset tenant extension
        memory._tenant_extension = None
        _current_schema.set("public")

    @pytest.mark.asyncio
    async def test_schema_context_isolation_in_concurrent_tasks(self, pg0_db_url):
        """
        Verify that _current_schema contextvar is properly isolated
        between concurrent async tasks.
        """
        results = {}
        errors = []

        async def check_schema_context(schema_name: str, delay: float):
            """Set schema context, wait, then verify it's still correct."""
            try:
                # Set the schema
                _current_schema.set(schema_name)

                # Small delay to allow interleaving
                await asyncio.sleep(delay)

                # Verify schema is still correct
                current = _current_schema.get()
                if current != schema_name:
                    errors.append(f"Expected {schema_name}, got {current}")

                # Verify fq_table uses correct schema
                table = fq_table("memory_units")
                expected = f"{schema_name}.memory_units"
                if table != expected:
                    errors.append(f"Expected {expected}, got {table}")

                results[schema_name] = current

            except Exception as e:
                errors.append(f"Error in {schema_name}: {e}")

        # Run many concurrent tasks with different schemas
        tasks = []
        for i in range(10):
            for schema in ["schema_a", "schema_b", "schema_c"]:
                # Vary delays to create interleaving
                delay = 0.01 * (i % 3)
                tasks.append(check_schema_context(f"{schema}_{i}", delay))

        await asyncio.gather(*tasks)

        # No errors should have occurred
        assert not errors, f"Schema context isolation errors: {errors}"

    @pytest.mark.asyncio
    async def test_list_memories_respects_schema(self, memory, pg0_db_url):
        """
        list_memory_units should only return memories from the current schema.

        Uses run_migrations(schema=x) to provision schemas.
        """
        import asyncpg

        schemas = ["tenant_list_a", "tenant_list_b"]
        bank_id = f"test-list-{uuid.uuid4().hex[:8]}"

        # Clean up any existing schemas and provision via migrations
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                await drop_schema(conn, schema)
        finally:
            await conn.close()

        # Provision schemas using run_migrations
        for schema in schemas:
            run_migrations(pg0_db_url, schema=schema)

        # Insert test data directly into each schema
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                await conn.execute(
                    f"""
                    INSERT INTO "{schema}".memory_units (bank_id, text, event_date, fact_type)
                    VALUES ($1, $2, now(), 'world')
                    """,
                    bank_id,
                    f"Direct insert for {schema}",
                )
        finally:
            await conn.close()

        # Configure tenant extension
        tenant_ext = MultiSchemaTestTenantExtension({
            "db_url": pg0_db_url,
            "valid_schemas": set(schemas),
        })
        memory._tenant_extension = tenant_ext

        try:
            # Query as tenant_list_a - should only see tenant_list_a's data
            tenant_a_request = RequestContext(api_key="key-tenant_list_a")
            await memory._authenticate_tenant(tenant_a_request)

            result_a = await memory.list_memory_units(bank_id=bank_id, request_context=tenant_a_request)
            texts_a = [item["text"] for item in result_a.get("items", [])]

            assert len(texts_a) == 1, f"Expected 1 memory for tenant_list_a, got {len(texts_a)}"
            assert "tenant_list_a" in texts_a[0], f"Wrong content: {texts_a[0]}"

            # Query as tenant_list_b - should only see tenant_list_b's data
            tenant_b_request = RequestContext(api_key="key-tenant_list_b")
            await memory._authenticate_tenant(tenant_b_request)

            result_b = await memory.list_memory_units(bank_id=bank_id, request_context=tenant_b_request)
            texts_b = [item["text"] for item in result_b.get("items", [])]

            assert len(texts_b) == 1, f"Expected 1 memory for tenant_list_b, got {len(texts_b)}"
            assert "tenant_list_b" in texts_b[0], f"Wrong content: {texts_b[0]}"

        finally:
            # Cleanup
            conn = await asyncpg.connect(pg0_db_url)
            try:
                for schema in schemas:
                    await drop_schema(conn, schema)
            finally:
                await conn.close()

            memory._tenant_extension = None
            _current_schema.set("public")

    @pytest.mark.asyncio
    async def test_high_concurrency_schema_isolation(self, memory, pg0_db_url):
        """
        Stress test: Many concurrent operations across multiple schemas
        should maintain perfect isolation.

        Uses run_migrations(schema=x) to provision schemas like a real extension.
        """
        import asyncpg

        # Create more schemas for stress test
        num_schemas = 5
        ops_per_schema = 10
        schemas = [f"stress_tenant_{i}" for i in range(num_schemas)]
        bank_id = f"test-stress-{uuid.uuid4().hex[:8]}"

        # Clean up any existing schemas first
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                await drop_schema(conn, schema)
        finally:
            await conn.close()

        # Provision schemas using run_migrations
        for schema in schemas:
            run_migrations(pg0_db_url, schema=schema)

        # Configure tenant extension (schemas already provisioned)
        tenant_ext = MultiSchemaTestTenantExtension({
            "db_url": pg0_db_url,
            "valid_schemas": set(schemas),
        })
        # Mark schemas as already provisioned so extension doesn't re-run migrations
        tenant_ext._provisioned = set(schemas)
        memory._tenant_extension = tenant_ext

        errors = []

        async def insert_one(schema: str, item_id: int):
            """Single insert operation for tracking."""
            try:
                # Authenticate to set the schema context
                tenant_request = RequestContext(api_key=f"key-{schema}")
                await memory._authenticate_tenant(tenant_request)

                # Insert using fq_table
                pool = await memory._get_pool()
                from hindsight_api.engine.db_utils import acquire_with_retry

                async with acquire_with_retry(pool) as conn:
                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table('memory_units')} (bank_id, text, event_date, fact_type)
                        VALUES ($1, $2, now(), 'world')
                        """,
                        bank_id,
                        f"STRESS_MARKER_{schema}_ITEM{item_id}: Memory for {schema}",
                    )
            except Exception as e:
                errors.append(f"Insert error for {schema}: {e}")

        # Run many concurrent operations
        tasks = []
        for i in range(ops_per_schema):
            for schema in schemas:
                tasks.append(insert_one(schema, i))

        await asyncio.gather(*tasks)

        # Check for errors during insert
        assert not errors, f"Errors during insert: {errors}"

        # Verify no cross-contamination
        conn = await asyncpg.connect(pg0_db_url)
        try:
            for schema in schemas:
                texts = await get_memory_texts_in_schema(conn, schema, bank_id)

                # Should have exactly ops_per_schema memories
                assert len(texts) == ops_per_schema, (
                    f"Schema {schema} should have {ops_per_schema} memories, got {len(texts)}"
                )

                # All memories should reference this schema only
                for text in texts:
                    # Check it contains our schema marker
                    assert f"STRESS_MARKER_{schema}" in text, (
                        f"Memory in {schema} doesn't contain schema marker: {text}"
                    )

                    # Check it doesn't contain other schema markers
                    for other_schema in schemas:
                        if other_schema != schema:
                            assert f"STRESS_MARKER_{other_schema}" not in text, (
                                f"Cross-contamination! {schema} has {other_schema}'s data: {text}"
                            )
        finally:
            # Cleanup
            for schema in schemas:
                await drop_schema(conn, schema)
            await conn.close()

        memory._tenant_extension = None
        _current_schema.set("public")
