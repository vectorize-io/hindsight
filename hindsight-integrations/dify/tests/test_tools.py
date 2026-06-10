"""Tests for Retain, Recall, and Reflect tool implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.recall import RecallTool
from tools.reflect import ReflectTool
from tools.retain import RetainTool


@dataclass
class FakeMessage:
    """Minimal message container for test assertions."""

    type: str
    message: Any


def _make_tool(cls):
    """Construct a tool instance without invoking the dify_plugin base init."""
    tool = cls.__new__(cls)
    tool.runtime = MagicMock()
    tool.runtime.credentials = {"api_url": "http://localhost:8888", "api_key": "hsk_test"}
    tool.create_text_message = lambda text: FakeMessage(type="text", message=text)
    tool.create_json_message = lambda json: FakeMessage(type="json", message=json)
    return tool


class TestRetainTool:
    def test_missing_bank_id(self):
        tool = _make_tool(RetainTool)
        msgs = list(tool._invoke({"bank_id": "", "content": "hello"}))
        assert len(msgs) == 1
        assert "bank_id is required" in msgs[0].message

    def test_missing_content(self):
        tool = _make_tool(RetainTool)
        msgs = list(tool._invoke({"bank_id": "b1", "content": ""}))
        assert len(msgs) == 1
        assert "content is required" in msgs[0].message

    @patch("tools.retain.build_client")
    def test_successful_retain(self, mock_build):
        mock_client = MagicMock()
        mock_client.retain.return_value = MagicMock(success=True)
        mock_build.return_value = mock_client

        tool = _make_tool(RetainTool)
        msgs = list(tool._invoke({"bank_id": "b1", "content": "hello world", "tags": "a, b"}))

        mock_client.retain.assert_called_once_with(bank_id="b1", content="hello world", tags=["a", "b"])
        assert len(msgs) == 2
        # First message is JSON
        assert msgs[0].message["success"] is True
        # Second message is text confirmation
        assert "Retained 1 memory" in msgs[1].message

    @patch("tools.retain.build_client")
    def test_retain_error(self, mock_build):
        mock_client = MagicMock()
        mock_client.retain.side_effect = RuntimeError("connection refused")
        mock_build.return_value = mock_client

        tool = _make_tool(RetainTool)
        msgs = list(tool._invoke({"bank_id": "b1", "content": "hello"}))

        assert len(msgs) == 1
        assert "retain failed" in msgs[0].message


class TestRecallTool:
    def test_missing_bank_id(self):
        tool = _make_tool(RecallTool)
        msgs = list(tool._invoke({"bank_id": "", "query": "test"}))
        assert len(msgs) == 1
        assert "bank_id is required" in msgs[0].message

    def test_missing_query(self):
        tool = _make_tool(RecallTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": ""}))
        assert len(msgs) == 1
        assert "query is required" in msgs[0].message

    @patch("tools.recall.build_client")
    def test_successful_recall_with_results(self, mock_build):
        mock_memory = MagicMock()
        mock_memory.model_dump.return_value = {"id": "m1", "text": "fact one", "type": "world"}

        mock_client = MagicMock()
        mock_client.recall.return_value = MagicMock(results=[mock_memory])
        mock_build.return_value = mock_client

        tool = _make_tool(RecallTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "what happened"}))

        mock_client.recall.assert_called_once_with(
            bank_id="b1", query="what happened", budget="mid", max_tokens=4096, tags=None
        )
        assert len(msgs) == 2
        assert msgs[0].message["count"] == 1
        assert "Recalled 1 memories" in msgs[1].message

    @patch("tools.recall.build_client")
    def test_successful_recall_empty(self, mock_build):
        mock_client = MagicMock()
        mock_client.recall.return_value = MagicMock(results=[])
        mock_build.return_value = mock_client

        tool = _make_tool(RecallTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "nothing"}))

        assert len(msgs) == 2
        assert msgs[0].message["count"] == 0
        assert "No memories found" in msgs[1].message

    @patch("tools.recall.build_client")
    def test_recall_error(self, mock_build):
        mock_client = MagicMock()
        mock_client.recall.side_effect = RuntimeError("timeout")
        mock_build.return_value = mock_client

        tool = _make_tool(RecallTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "test"}))

        assert len(msgs) == 1
        assert "recall failed" in msgs[0].message


class TestReflectTool:
    def test_missing_bank_id(self):
        tool = _make_tool(ReflectTool)
        msgs = list(tool._invoke({"bank_id": "", "query": "test"}))
        assert len(msgs) == 1
        assert "bank_id is required" in msgs[0].message

    def test_missing_query(self):
        tool = _make_tool(ReflectTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": ""}))
        assert len(msgs) == 1
        assert "query is required" in msgs[0].message

    @patch("tools.reflect.build_client")
    def test_successful_reflect(self, mock_build):
        mock_client = MagicMock()
        mock_client.reflect.return_value = MagicMock(text="Ben tested Dify integration.")
        mock_build.return_value = mock_client

        tool = _make_tool(ReflectTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "what do we know about Ben?"}))

        mock_client.reflect.assert_called_once_with(bank_id="b1", query="what do we know about Ben?", budget="low")
        assert len(msgs) == 2
        assert msgs[0].message["text"] == "Ben tested Dify integration."
        assert "Ben tested Dify integration." in msgs[1].message

    @patch("tools.reflect.build_client")
    def test_reflect_empty_response(self, mock_build):
        mock_client = MagicMock()
        mock_client.reflect.return_value = MagicMock(text="")
        mock_build.return_value = mock_client

        tool = _make_tool(ReflectTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "unknown"}))

        assert len(msgs) == 2
        assert msgs[0].message["text"] == ""
        assert "(no answer)" in msgs[1].message

    @patch("tools.reflect.build_client")
    def test_reflect_error(self, mock_build):
        mock_client = MagicMock()
        mock_client.reflect.side_effect = RuntimeError("server error")
        mock_build.return_value = mock_client

        tool = _make_tool(ReflectTool)
        msgs = list(tool._invoke({"bank_id": "b1", "query": "test"}))

        assert len(msgs) == 1
        assert "reflect failed" in msgs[0].message
