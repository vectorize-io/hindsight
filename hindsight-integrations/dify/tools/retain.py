"""Hindsight Retain tool — store content in a memory bank."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from ._client import build_client, parse_tags


class RetainTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage, None, None]:
        bank_id = tool_parameters.get("bank_id")
        content = tool_parameters.get("content")

        if not bank_id:
            yield self.create_text_message("bank_id is required")
            return
        if not content:
            yield self.create_text_message("content is required")
            return

        client = build_client(self.runtime.credentials)
        tags = parse_tags(tool_parameters.get("tags"))

        try:
            response = client.retain(bank_id=bank_id, content=content, tags=tags)
        except Exception as e:
            yield self.create_text_message(f"Hindsight retain failed: {e}")
            return

        result: dict[str, Any] = {
            "success": getattr(response, "success", True),
            "bank_id": bank_id,
        }
        yield self.create_json_message(result)
        yield self.create_text_message(f"Retained 1 memory in bank '{bank_id}'.")
