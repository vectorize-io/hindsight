import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "hindsight_api"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


ensure_package("hindsight_api", PACKAGE_ROOT)
ensure_package("hindsight_api.engine", PACKAGE_ROOT / "engine")
ensure_package("hindsight_api.engine.providers", PACKAGE_ROOT / "engine" / "providers")
fake_metrics = types.ModuleType("hindsight_api.metrics")
fake_metrics.get_metrics_collector = lambda: types.SimpleNamespace(record_llm_call=lambda **kwargs: None)
sys.modules["hindsight_api.metrics"] = fake_metrics

from hindsight_api.engine.providers.codex_llm import CodexLLM


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def build_llm() -> CodexLLM:
    with patch.object(CodexLLM, "_load_codex_auth", return_value=("token", "account")):
        return CodexLLM(
            provider="openai-codex",
            api_key="ignored",
            base_url="https://chatgpt.com/backend-api",
            model="gpt-5.4-mini",
        )


class CodexToolChoiceTests(unittest.TestCase):
    def test_codex_normalizes_legacy_named_tool_choice_shape(self) -> None:
        async def scenario() -> dict:
            llm = build_llm()
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = response
                with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
                    mock_parse.return_value = (None, [])
                    await llm.call_with_tools(
                        messages=[{"role": "user", "content": "recall the memory"}],
                        tools=TOOLS,
                        tool_choice={"type": "function", "function": {"name": "recall"}},
                        max_retries=0,
                    )
                return mock_post.call_args.kwargs["json"]

        sent_payload = asyncio.run(scenario())
        self.assertEqual(sent_payload["tool_choice"], {"type": "function", "name": "recall"})

    def test_codex_forced_tool_choice_still_yields_tool_calls(self) -> None:
        async def scenario() -> tuple[object, dict]:
            llm = build_llm()
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            tool_call = {"id": "call-1", "name": "recall", "arguments": {"query": "memory"}}
            with patch.object(llm._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = response
                with patch.object(llm, "_parse_sse_tool_stream", new_callable=AsyncMock) as mock_parse:
                    mock_parse.return_value = (None, [tool_call])
                    result = await llm.call_with_tools(
                        messages=[{"role": "user", "content": "recall the memory"}],
                        tools=TOOLS,
                        tool_choice={"type": "function", "function": {"name": "recall"}},
                        max_retries=0,
                    )
                return result, mock_post.call_args.kwargs["json"]

        result, sent_payload = asyncio.run(scenario())
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "recall")
        self.assertEqual(sent_payload["tool_choice"], {"type": "function", "name": "recall"})


if __name__ == "__main__":
    unittest.main()
