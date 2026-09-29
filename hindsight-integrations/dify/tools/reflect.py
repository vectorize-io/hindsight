"""Hindsight Reflect tool — synthesized answer over a memory bank."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from ._client import build_client


class ReflectTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage, None, None]:
        bank_id = tool_parameters.get("bank_id")
        query = tool_parameters.get("query")

        if not bank_id:
            yield self.create_text_message("bank_id is required")
            return
        if not query:
            yield self.create_text_message("query is required")
            return

        client = build_client(self.runtime.credentials)
        # Reflect defaults to "low" budget since it involves LLM synthesis (more expensive)
        budget = tool_parameters.get("budget") or "low"

        try:
            response = client.reflect(bank_id=bank_id, query=query, budget=budget)
        except Exception as e:
            yield self.create_text_message(f"Hindsight reflect failed: {e}")
            return

        text = getattr(response, "text", "") or ""
        yield self.create_json_message({"text": text})
        yield self.create_text_message(text or "(no answer)")
