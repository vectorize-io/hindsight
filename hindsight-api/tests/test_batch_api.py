"""
Test OpenAI Batch API integration for retain fact extraction.

Tests cover:
- Normal batch API flow (submit, poll, complete)
- Crash recovery (resume from existing batch_id)
- Provider fallback (when batch API not supported)
- Worker recovery on restart
"""
import pytest
import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from hindsight_api import RequestContext
from hindsight_api.engine.retain.fact_extraction import (
    extract_facts_from_contents_batch_api,
    extract_facts_from_contents,
    RetainContent,
)
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.worker.poller import WorkerPoller

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM config with batch API support."""
    mock = MagicMock()
    mock.provider = "openai"
    mock.model = "gpt-4o-mini"
    mock._provider_impl = AsyncMock()
    return mock


@pytest.fixture
def test_contents():
    """Create test content for fact extraction."""
    return [
        RetainContent(
            content="Alice is a senior software engineer at TechCorp. She specializes in distributed systems.",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            context="team overview",
        ),
        RetainContent(
            content="Bob joined the team last month as a junior developer. He is learning React.",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            context="team overview",
        ),
    ]


@pytest.fixture
def hindsight_config():
    """Create test config with batch API enabled."""
    config = HindsightConfig.from_env()
    config.retain_use_batch_api = True
    config.retain_batch_poll_interval = 1  # Fast polling for tests
    config.retain_chunk_size = 4000
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    return config


@pytest.mark.asyncio
async def test_batch_api_normal_flow(mock_llm_config, test_contents, hindsight_config, memory, request_context):
    """Test normal batch API flow: submit, poll, complete."""
    bank_id = f"test_batch_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Mock batch API responses
        batch_id = "batch_test123"

        # Mock supports_batch_api
        mock_llm_config._provider_impl.supports_batch_api = AsyncMock(return_value=True)

        # Mock submit_batch - returns batch metadata
        mock_llm_config._provider_impl.submit_batch = AsyncMock(
            return_value={
                "batch_id": batch_id,
                "status": "validating",
                "request_counts": {"total": 2, "completed": 0, "failed": 0},
            }
        )

        # Mock get_batch_status - simulate polling sequence
        status_sequence = [
            {"status": "in_progress", "request_counts": {"total": 2, "completed": 1, "failed": 0}},
            {"status": "completed", "request_counts": {"total": 2, "completed": 2, "failed": 0}},
        ]
        mock_llm_config._provider_impl.get_batch_status = AsyncMock(side_effect=status_sequence)

        # Mock retrieve_batch_results - returns fact extraction results
        mock_results = [
            {
                "custom_id": "chunk_0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({
                                        "facts": [
                                            {
                                                "what": "Alice is a senior software engineer at TechCorp",
                                                "when": "present",
                                                "where": "TechCorp",
                                                "who": "Alice",
                                                "why": "Professional background information",
                                                "fact_type": "world",
                                                "fact_kind": "conversation",
                                            }
                                        ]
                                    })
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    }
                },
            },
            {
                "custom_id": "chunk_1",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({
                                        "facts": [
                                            {
                                                "what": "Bob joined the team last month as a junior developer",
                                                "when": "last month",
                                                "where": "team",
                                                "who": "Bob",
                                                "why": "New team member information",
                                                "fact_type": "world",
                                                "fact_kind": "conversation",
                                            }
                                        ]
                                    })
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    }
                },
            },
        ]
        mock_llm_config._provider_impl.retrieve_batch_results = AsyncMock(return_value=mock_results)

        # Call batch API extraction
        facts, chunks, usage = await extract_facts_from_contents_batch_api(
            contents=test_contents,
            llm_config=mock_llm_config,
            agent_name="test_agent",
            config=hindsight_config,
            pool=None,  # No DB pool for this test
            operation_id=None,
            schema=None,
        )

        # Verify results
        assert len(facts) == 2, "Should extract 2 facts (one per chunk)"
        # Facts are ExtractedFact objects with .fact_text field
        assert "Alice" in facts[0].fact_text and "senior software engineer" in facts[0].fact_text
        assert "Bob" in facts[1].fact_text and "junior developer" in facts[1].fact_text

        # Verify chunks metadata
        assert len(chunks) == 2, "Should have 2 chunks metadata"
        assert chunks[0].fact_count == 1
        assert chunks[1].fact_count == 1

        # Verify token usage
        assert usage.input_tokens == 200  # 100 per chunk
        assert usage.output_tokens == 100  # 50 per chunk
        assert usage.total_tokens == 300

        # Verify API calls
        mock_llm_config._provider_impl.submit_batch.assert_called_once()
        assert mock_llm_config._provider_impl.get_batch_status.call_count == 2
        mock_llm_config._provider_impl.retrieve_batch_results.assert_called_once_with(batch_id)

        logger.info("✅ Normal batch API flow test passed")

    finally:
        # Cleanup
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_batch_api_crash_recovery(mock_llm_config, test_contents, hindsight_config, memory, request_context):
    """Test crash recovery: resume polling from existing batch_id."""
    bank_id = f"test_crash_{datetime.now(timezone.utc).timestamp()}"
    operation_id = str(uuid.uuid4())  # Must be UUID for async_operations table

    try:
        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        # Setup: Store batch_id in async_operations table (simulates partial execution)
        batch_id = "batch_recovered_456"
        pool = memory._pool
        schema = request_context.tenant_id

        from hindsight_api.engine.task_backend import fq_table
        table = fq_table("async_operations", schema)

        # Create operation with batch_id already stored
        await pool.execute(
            f"""
            INSERT INTO {table} (operation_id, operation_type, bank_id, status, result_metadata)
            VALUES ($1, 'retain', $2, 'processing', $3::jsonb)
            """,
            operation_id,
            bank_id,
            json.dumps({
                "batch_id": batch_id,
                "batch_provider": "openai",
                "chunk_count": 2,
            }),
        )

        # Mock batch API responses for resume scenario
        mock_llm_config._provider_impl.supports_batch_api = AsyncMock(return_value=True)

        # Mock get_batch_status - batch already in progress
        mock_llm_config._provider_impl.get_batch_status = AsyncMock(
            return_value={
                "status": "completed",
                "request_counts": {"total": 2, "completed": 2, "failed": 0},
            }
        )

        # Mock retrieve_batch_results
        mock_results = [
            {
                "custom_id": "chunk_0",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({
                                        "facts": [
                                            {
                                                "what": "Alice is a senior software engineer",
                                                "when": "present",
                                                "where": "TechCorp",
                                                "who": "Alice",
                                                "why": "Background",
                                                "fact_type": "world",
                                                "fact_kind": "conversation",
                                            }
                                        ]
                                    })
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    }
                },
            },
            {
                "custom_id": "chunk_1",
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps({
                                        "facts": [
                                            {
                                                "what": "Bob is a junior developer",
                                                "when": "last month",
                                                "where": "team",
                                                "who": "Bob",
                                                "why": "New member",
                                                "fact_type": "world",
                                                "fact_kind": "conversation",
                                            }
                                        ]
                                    })
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    }
                },
            },
        ]
        mock_llm_config._provider_impl.retrieve_batch_results = AsyncMock(return_value=mock_results)

        # Call batch API extraction with operation_id (crash recovery scenario)
        facts, chunks, usage = await extract_facts_from_contents_batch_api(
            contents=test_contents,
            llm_config=mock_llm_config,
            agent_name="test_agent",
            config=hindsight_config,
            pool=pool,
            operation_id=operation_id,  # Provides crash recovery context
            schema=schema,
        )

        # Verify results
        assert len(facts) == 2, "Should extract 2 facts after recovery"

        # CRITICAL: Verify submit_batch was NOT called (because batch_id already exists)
        mock_llm_config._provider_impl.submit_batch.assert_not_called()

        # Verify get_batch_status WAS called (polling resumed)
        mock_llm_config._provider_impl.get_batch_status.assert_called()

        # Verify retrieve_batch_results was called with the recovered batch_id
        mock_llm_config._provider_impl.retrieve_batch_results.assert_called_once_with(batch_id)

        logger.info("✅ Crash recovery test passed - resumed polling without re-submission")

    finally:
        # Cleanup
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_batch_api_fallback_unsupported_provider(mock_llm_config, test_contents, hindsight_config):
    """Test fallback to sync mode when provider doesn't support batch API."""

    # Mock provider that doesn't support batch API
    mock_llm_config._provider_impl.supports_batch_api = AsyncMock(return_value=False)
    mock_llm_config.provider = "groq"  # Example of provider

    # Patch the sync mode function to verify it's called
    with patch(
        "hindsight_api.engine.retain.fact_extraction.extract_facts_from_contents"
    ) as mock_sync_extract:
        mock_sync_extract.return_value = ([], [], MagicMock())

        # Call batch API extraction (should fallback to sync)
        await extract_facts_from_contents_batch_api(
            contents=test_contents,
            llm_config=mock_llm_config,
            agent_name="test_agent",
            config=hindsight_config,
            pool=None,
            operation_id=None,
            schema=None,
        )

        # Verify fallback occurred
        mock_sync_extract.assert_called_once()

        # Verify batch API methods were NOT called
        mock_llm_config._provider_impl.submit_batch.assert_not_called()

        logger.info("✅ Fallback to sync mode test passed")


@pytest.mark.asyncio
async def test_worker_batch_recovery(memory, request_context):
    """Test that WorkerPoller._recover_batch_operations finds and resets orphaned batches."""
    bank_id = f"test_worker_recovery_{datetime.now(timezone.utc).timestamp()}"
    operation_id = str(uuid.uuid4())  # Must be UUID for async_operations table

    try:
        # Ensure bank exists
        await memory.get_bank_profile(bank_id, request_context=request_context)

        pool = memory._pool
        schema = request_context.tenant_id

        from hindsight_api.engine.task_backend import fq_table
        table = fq_table("async_operations", schema)

        # Create orphaned batch operation (simulates worker crash during polling)
        batch_id = "batch_orphaned_999"
        task_payload = {
            "operation_type": "retain",
            "bank_id": bank_id,
            "contents": [{"content": "test", "event_date": "2024-01-15T00:00:00Z"}],
        }

        await pool.execute(
            f"""
            INSERT INTO {table} (operation_id, operation_type, bank_id, status, worker_id, result_metadata, task_payload)
            VALUES ($1, 'retain', $2, 'processing', 'worker_crashed', $3::jsonb, $4::jsonb)
            """,
            operation_id,
            bank_id,
            json.dumps({
                "batch_id": batch_id,
                "batch_provider": "openai",
                "chunk_count": 1,
            }),
            json.dumps(task_payload),
        )

        # Create WorkerPoller
        from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
        tenant_extension = DefaultTenantExtension(config={"schema": schema} if schema else {})

        poller = WorkerPoller(
            pool=pool,
            worker_id="test_worker_recovery",
            executor=memory,
            poll_interval_ms=100,
            max_retries=3,
            schema=schema,
            tenant_extension=tenant_extension,
            max_slots=5,
            consolidation_max_slots=2,
        )

        # Run recovery
        recovered_count = await poller._recover_batch_operations(schema)

        # Verify recovery
        assert recovered_count == 1, "Should recover 1 batch operation"

        # Verify operation was reset to pending
        row = await pool.fetchrow(
            f"SELECT status, worker_id FROM {table} WHERE operation_id = $1",
            operation_id,
        )

        assert row["status"] == "pending", "Operation should be reset to pending"
        assert row["worker_id"] is None, "Worker ID should be cleared"

        logger.info("✅ Worker batch recovery test passed")

    finally:
        # Cleanup
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_batch_api_via_extract_facts_from_contents(
    mock_llm_config, test_contents, hindsight_config, memory, request_context
):
    """Test that extract_facts_from_contents routes to batch API when enabled."""
    bank_id = f"test_routing_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Enable batch API in config
        hindsight_config.retain_use_batch_api = True

        # Mock batch API support
        mock_llm_config._provider_impl.supports_batch_api = AsyncMock(return_value=True)
        mock_llm_config._provider_impl.submit_batch = AsyncMock(
            return_value={"batch_id": "batch_123", "status": "validating", "request_counts": {}}
        )
        mock_llm_config._provider_impl.get_batch_status = AsyncMock(
            return_value={"status": "completed", "request_counts": {"total": 1, "completed": 1, "failed": 0}}
        )
        mock_llm_config._provider_impl.retrieve_batch_results = AsyncMock(
            return_value=[
                {
                    "custom_id": "chunk_0",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps({"facts": []})
                                    }
                                }
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        }
                    },
                }
            ]
        )

        # Call main extract_facts_from_contents (should route to batch API)
        facts, chunks, usage = await extract_facts_from_contents(
            contents=test_contents,
            llm_config=mock_llm_config,
            agent_name="test_agent",
            config=hindsight_config,
            pool=None,
            operation_id=None,
            schema=None,
        )

        # Verify batch API was called
        mock_llm_config._provider_impl.submit_batch.assert_called_once()

        logger.info("✅ Routing to batch API test passed")

    finally:
        # Cleanup
        try:
            await memory.delete_bank(bank_id, request_context=request_context)
        except Exception:
            pass
