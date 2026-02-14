"""
Tests for the exclude_system_events configuration and filtering.

Verifies that system events (role: system, tool, tool_result, function)
are filtered from retain requests when HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS is enabled.
"""

import os

import pytest
import pytest_asyncio
import httpx
from datetime import datetime, timezone

from hindsight_api.api import create_app
from hindsight_api.config import (
    DEFAULT_EXCLUDE_SYSTEM_EVENTS,
    SYSTEM_EVENT_ROLES,
    HindsightConfig,
    clear_config_cache,
)


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment for each test, restoring original values after."""
    env_vars_to_save = [
        "HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS",
        "HINDSIGHT_API_LLM_PROVIDER",
        "HINDSIGHT_API_LLM_MODEL",
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


# ================================================================
# Config Tests
# ================================================================


class TestExcludeSystemEventsConfig:
    """Tests for the exclude_system_events configuration field."""

    def test_default_is_false(self):
        """exclude_system_events should default to False."""
        assert DEFAULT_EXCLUDE_SYSTEM_EVENTS is False

    def test_system_event_roles_defined(self):
        """SYSTEM_EVENT_ROLES should contain expected roles."""
        assert "system" in SYSTEM_EVENT_ROLES
        assert "tool" in SYSTEM_EVENT_ROLES
        assert "tool_result" in SYSTEM_EVENT_ROLES
        assert "function" in SYSTEM_EVENT_ROLES

    def test_config_from_env_default(self):
        """Config should default to exclude_system_events=False."""
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        config = HindsightConfig.from_env()
        assert config.exclude_system_events is False

    def test_config_from_env_enabled(self):
        """Config should read HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS=true."""
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "true"
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        config = HindsightConfig.from_env()
        assert config.exclude_system_events is True

    def test_config_from_env_enabled_with_1(self):
        """Config should accept '1' as truthy."""
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "1"
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        config = HindsightConfig.from_env()
        assert config.exclude_system_events is True

    def test_config_from_env_disabled(self):
        """Config should read HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS=false."""
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "false"
        os.environ["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
        config = HindsightConfig.from_env()
        assert config.exclude_system_events is False

    def test_field_is_configurable(self):
        """exclude_system_events should be in the configurable fields set."""
        configurable = HindsightConfig.get_configurable_fields()
        assert "exclude_system_events" in configurable


# ================================================================
# HTTP API Filtering Tests
# ================================================================


@pytest_asyncio.fixture
async def api_client(memory):
    """Create an async test client for the FastAPI app."""
    app = create_app(memory, initialize_memory=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_retain_filters_system_events_when_enabled(api_client, request_context):
    """When exclude_system_events is enabled, system role items should be filtered."""
    bank_id = f"test_exclude_sys_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Enable exclude_system_events for this bank via bank config
        # First, enable the bank config API
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "true"
        clear_config_cache()

        # Send retain request with mixed items
        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": "You are a helpful assistant.",
                        "metadata": {"role": "system"},
                    },
                    {
                        "content": "Alice likes Python programming.",
                        "metadata": {"role": "user"},
                    },
                    {
                        "content": "search_web({\"query\": \"Python tutorials\"})",
                        "metadata": {"role": "tool"},
                    },
                    {
                        "content": "Here are the results of the search.",
                        "metadata": {"role": "tool_result"},
                    },
                    {
                        "content": "Bob enjoys hiking on weekends.",
                        "metadata": {"role": "assistant"},
                    },
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        # Only user and assistant items should be retained (3 system events filtered)
        assert result["items_count"] == 2

    finally:
        # Clean up
        await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        clear_config_cache()


@pytest.mark.asyncio
async def test_retain_keeps_all_items_when_disabled(api_client, request_context):
    """When exclude_system_events is disabled (default), all items should be retained."""
    bank_id = f"test_keep_sys_{datetime.now(timezone.utc).timestamp()}"

    try:
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        clear_config_cache()

        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": "You are a helpful assistant.",
                        "metadata": {"role": "system"},
                    },
                    {
                        "content": "Alice likes Python programming.",
                        "metadata": {"role": "user"},
                    },
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        # Both items should be retained when filtering is off
        assert result["items_count"] == 2

    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}/memories")


@pytest.mark.asyncio
async def test_retain_handles_items_without_metadata(api_client, request_context):
    """Items without metadata should not be filtered, even when exclude_system_events is enabled."""
    bank_id = f"test_no_meta_{datetime.now(timezone.utc).timestamp()}"

    try:
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "true"
        clear_config_cache()

        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {"content": "Alice likes Python programming."},
                    {"content": "Bob enjoys hiking."},
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        # Both items should be retained (no metadata to filter on)
        assert result["items_count"] == 2

    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        clear_config_cache()


@pytest.mark.asyncio
async def test_retain_returns_empty_when_all_filtered(api_client, request_context):
    """When all items are system events, should return success with items_count=0."""
    bank_id = f"test_all_filtered_{datetime.now(timezone.utc).timestamp()}"

    try:
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "true"
        clear_config_cache()

        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": "You are a helpful assistant.",
                        "metadata": {"role": "system"},
                    },
                    {
                        "content": "search_web result",
                        "metadata": {"role": "tool"},
                    },
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["items_count"] == 0

    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        clear_config_cache()


@pytest.mark.asyncio
async def test_retain_filters_function_role(api_client, request_context):
    """Legacy 'function' role should also be filtered."""
    bank_id = f"test_func_role_{datetime.now(timezone.utc).timestamp()}"

    try:
        os.environ["HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS"] = "true"
        clear_config_cache()

        response = await api_client.post(
            f"/v1/default/banks/{bank_id}/memories",
            json={
                "items": [
                    {
                        "content": "Function call result: {\"result\": \"success\"}",
                        "metadata": {"role": "function"},
                    },
                    {
                        "content": "Alice completed the task successfully.",
                        "metadata": {"role": "user"},
                    },
                ]
            },
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["items_count"] == 1

    finally:
        await api_client.delete(f"/v1/default/banks/{bank_id}/memories")
        os.environ.pop("HINDSIGHT_API_EXCLUDE_SYSTEM_EVENTS", None)
        clear_config_cache()
