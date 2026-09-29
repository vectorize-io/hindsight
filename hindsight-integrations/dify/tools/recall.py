"""Hindsight Recall tool — search a memory bank by query."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from ._client import build_client, parse_tags


class RecallTool(Tool):
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
        budget = tool_parameters.get("budget") or "mid"
        max_tokens = int(tool_parameters.get("max_tokens") or 4096)
        tags = parse_tags(tool_parameters.get("tags"))

        try:
            response = client.recall(
                bank_id=bank_id,
                query=query,
                budget=budget,
                max_tokens=max_tokens,
                tags=tags,
            )
        except Exception as e:
            yield self.create_text_message(f"Hindsight recall failed: {e}")
            return

        results = [_memory_to_dict(m) for m in (response.results or [])]

        yield self.create_json_message(
            {
                "results": results,
                "count": len(results),
            }
        )

        if results:
            lines = [f"Recalled {len(results)} memories:"]
            for idx, m in enumerate(results, 1):
                lines.append(f"{idx}. {m.get('text', '')}")
            yield self.create_text_message("\n".join(lines))
        else:
            yield self.create_text_message("No memories found.")


def _memory_to_dict(memory: Any) -> dict[str, Any]:
    """Convert a Memory result to a JSON-serializable dict."""
    if hasattr(memory, "model_dump"):
        return memory.model_dump(exclude_none=True)
    return {
        "id": getattr(memory, "id", None),
        "text": getattr(memory, "text", None),
        "type": getattr(memory, "type", None),
    }
