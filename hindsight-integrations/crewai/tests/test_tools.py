"""Unit tests for HindsightReflectTool."""

from unittest.mock import MagicMock, patch

from hindsight_crewai import HindsightReflectTool, configure, reset_config
from hindsight_crewai.errors import HindsightError

import pytest


def _passthrough(fn, *args, **kwargs):
    """Replace call_sync with direct call for testing."""
    return fn(*args, **kwargs)


class TestReflectTool:
    def setup_method(self):
        reset_config()
        configure(hindsight_api_url="http://localhost:8888")

    def teardown_method(self):
        reset_config()

    @patch("hindsight_crewai.tools.call_sync", side_effect=_passthrough)
    def test_reflect_returns_text(self, _mock_cs):
        tool = HindsightReflectTool(bank_id="test-bank")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "The user is a Python developer who..."
        mock_client.reflect.return_value = mock_result
        tool._local.client = mock_client

        result = tool._run("What do you know about the user?")

        assert result == "The user is a Python developer who..."
        mock_client.reflect.assert_called_once()

    @patch("hindsight_crewai.tools.call_sync", side_effect=_passthrough)
    def test_reflect_empty_returns_fallback(self, _mock_cs):
        tool = HindsightReflectTool(bank_id="test-bank")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = ""
        mock_client.reflect.return_value = mock_result
        tool._local.client = mock_client

        result = tool._run("anything")
        assert "No relevant memories" in result

    @patch("hindsight_crewai.tools.call_sync", side_effect=_passthrough)
    def test_reflect_passes_context(self, _mock_cs):
        tool = HindsightReflectTool(
            bank_id="test-bank",
            reflect_context="Agent is a delivery robot.",
        )
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Synthesized answer"
        mock_client.reflect.return_value = mock_result
        tool._local.client = mock_client

        tool._run("Where is Alice?")

        call_kwargs = mock_client.reflect.call_args[1]
        assert call_kwargs["context"] == "Agent is a delivery robot."
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["budget"] == "mid"

    @patch("hindsight_crewai.tools.call_sync", side_effect=_passthrough)
    def test_reflect_passes_budget(self, _mock_cs):
        tool = HindsightReflectTool(bank_id="test-bank", budget="high")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "answer"
        mock_client.reflect.return_value = mock_result
        tool._local.client = mock_client

        tool._run("query")

        call_kwargs = mock_client.reflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    @patch("hindsight_crewai.tools.call_sync", side_effect=_passthrough)
    def test_reflect_raises_hindsight_error_on_failure(self, _mock_cs):
        tool = HindsightReflectTool(bank_id="test-bank")
        mock_client = MagicMock()
        mock_client.reflect.side_effect = RuntimeError("timeout")
        tool._local.client = mock_client

        with pytest.raises(HindsightError, match="Reflect failed"):
            tool._run("query")

    def test_tool_metadata(self):
        tool = HindsightReflectTool(bank_id="test-bank")
        assert tool.name == "hindsight_reflect"
        assert "reflect" in tool.description.lower()
        assert "memories" in tool.description.lower()
