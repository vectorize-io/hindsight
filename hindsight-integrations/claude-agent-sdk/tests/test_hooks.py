"""Unit tests for Hindsight Claude Agent SDK hooks."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hindsight_claude_agent_sdk import (
    MemoryHookConfig,
    create_memory_hooks,
    reset_config,
)
from hindsight_claude_agent_sdk.errors import HindsightError


def _mock_client():
    """Create a mock Hindsight client with async methods."""
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock()
    client.areflect = AsyncMock()
    return client


def _mock_recall_response(texts: list[str]):
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


def _write_transcript(tmp_path, *entries: dict) -> str:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    return str(transcript)


class TestCreateMemoryHooks:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_default_hooks_include_recall_and_retain(self):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        assert "UserPromptSubmit" in hooks
        assert "Stop" in hooks

    def test_disable_auto_recall(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(auto_recall=False),
        )
        assert "UserPromptSubmit" not in hooks
        assert "Stop" in hooks

    def test_disable_auto_retain(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(auto_retain=False),
        )
        assert "UserPromptSubmit" in hooks
        assert "Stop" not in hooks

    def test_disable_both(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(auto_recall=False, auto_retain=False),
        )
        assert len(hooks) == 0

    def test_retain_on_tools_adds_post_tool_use(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_on_tools=["Bash"]),
        )
        assert "PostToolUse" in hooks
        assert hooks["PostToolUse"][0].matcher == "Bash"

    def test_retain_on_tools_multiple_patterns(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_on_tools=["Bash", "Read"]),
        )
        assert hooks["PostToolUse"][0].matcher == "Bash|Read"

    def test_defaults_to_cloud_without_config(self, monkeypatch):
        """With no client, config, or explicit URL, hooks default to the cloud URL."""
        from hindsight_claude_agent_sdk.config import DEFAULT_HINDSIGHT_API_URL

        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        with patch("hindsight_claude_agent_sdk._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            hooks = create_memory_hooks(bank_id="test")
            assert "UserPromptSubmit" in hooks
            assert mock_cls.call_args.kwargs["base_url"] == DEFAULT_HINDSIGHT_API_URL
            assert "api_key" not in mock_cls.call_args.kwargs

    def test_reads_api_key_from_env_without_config(self, monkeypatch):
        """create_memory_hooks honours HINDSIGHT_API_KEY even without configure()."""
        monkeypatch.setenv("HINDSIGHT_API_KEY", "sk-from-env")
        with patch("hindsight_claude_agent_sdk._client.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            create_memory_hooks(bank_id="test")
            assert mock_cls.call_args.kwargs["api_key"] == "sk-from-env"


class TestRecallHook:
    @pytest.mark.asyncio
    async def test_injects_memories_as_system_message(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook(
            {"prompt": "What do you know about me?"},
            None,
            None,
        )

        assert "systemMessage" in result
        assert "1. User likes Python" in result["systemMessage"]
        assert "2. User is in NYC" in result["systemMessage"]

    @pytest.mark.asyncio
    async def test_passes_prompt_as_query(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        await recall_hook({"prompt": "my preferences"}, None, None)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["query"] == "my preferences"
        assert call_kwargs["bank_id"] == "test"

    @pytest.mark.asyncio
    async def test_custom_recall_query(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(recall_query="user context"),
        )
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        await recall_hook({"prompt": "something else"}, None, None)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["query"] == "user context"

    @pytest.mark.asyncio
    async def test_respects_max_results(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["a", "b", "c", "d", "e"])
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(recall_max_results=3),
        )
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook({"prompt": "query"}, None, None)

        assert "3. c" in result["systemMessage"]
        assert "4." not in result["systemMessage"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_memories(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response([])
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook({"prompt": "query"}, None, None)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        client = _mock_client()
        client.arecall.side_effect = RuntimeError("connection refused")
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook({"prompt": "query"}, None, None)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_prompt(self):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook({"prompt": ""}, None, None)

        assert result == {}
        client.arecall.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_budget_and_tags(self):
        client = _mock_client()
        client.arecall.return_value = _mock_recall_response(["fact"])
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            budget="low",
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        await recall_hook({"prompt": "query"}, None, None)

        call_kwargs = client.arecall.call_args[1]
        assert call_kwargs["budget"] == "low"
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"


class TestRetainHook:
    @pytest.mark.asyncio
    async def test_retains_result_on_stop(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Working on it"}]}},
            {
                "type": "result",
                "result": "I refactored the auth module to use JWT tokens instead of sessions.",
            },
        )

        result = await retain_hook(
            {"transcript_path": transcript_path, "stop_hook_active": False},
            None,
            None,
        )

        assert result == {}
        client.aretain.assert_called_once()
        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["bank_id"] == "test"
        assert "refactored the auth module" in call_kwargs["content"]
        assert call_kwargs["tags"] == ["source:claude-agent-sdk"]

    @pytest.mark.asyncio
    async def test_skips_short_results(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "result", "result": "OK"},
        )

        await retain_hook({"transcript_path": transcript_path, "stop_hook_active": False}, None, None)

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_empty_results(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "result", "result": ""},
        )

        await retain_hook({"transcript_path": transcript_path, "stop_hook_active": False}, None, None)

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_long_results(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]

        long_result = "x" * 10000
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "result", "result": long_result},
        )
        await retain_hook({"transcript_path": transcript_path, "stop_hook_active": False}, None, None)

        call_kwargs = client.aretain.call_args[1]
        # 4000 chars of result + prefix
        assert len(call_kwargs["content"]) < 5000

    @pytest.mark.asyncio
    async def test_custom_retain_tags(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_tags=["env:prod", "team:backend"]),
        )
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "result", "result": "Completed the deployment pipeline changes."},
        )

        await retain_hook(
            {"transcript_path": transcript_path, "stop_hook_active": False},
            None,
            None,
        )

        call_kwargs = client.aretain.call_args[1]
        assert call_kwargs["tags"] == ["env:prod", "team:backend"]

    @pytest.mark.asyncio
    async def test_retain_error_is_non_fatal(self, tmp_path):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {"type": "result", "result": "Some meaningful result text here."},
        )

        result = await retain_hook(
            {"transcript_path": transcript_path, "stop_hook_active": False},
            None,
            None,
        )

        assert result == {}

    @pytest.mark.asyncio
    async def test_falls_back_to_final_assistant_text_when_result_missing(self, tmp_path):
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]
        transcript_path = _write_transcript(
            tmp_path,
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "I updated the deployment pipeline and added a rollback check.",
                        }
                    ]
                },
            },
        )

        await retain_hook({"transcript_path": transcript_path, "stop_hook_active": False}, None, None)

        call_kwargs = client.aretain.call_args[1]
        assert "deployment pipeline" in call_kwargs["content"]


class TestToolRetainHook:
    @pytest.mark.asyncio
    async def test_retains_tool_result(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_on_tools=["Bash"]),
        )
        tool_hook = hooks["PostToolUse"][0].hooks[0]

        await tool_hook(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest tests/ -v"},
                "tool_response": "PASSED 42 tests in 3.2s",
            },
            "tool-123",
            None,
        )

        client.aretain.assert_called_once()
        call_kwargs = client.aretain.call_args[1]
        assert "Bash" in call_kwargs["content"]
        assert "pytest" in call_kwargs["content"]
        assert "PASSED" in call_kwargs["content"]
        assert "tool:Bash" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_skips_short_tool_results(self):
        client = _mock_client()
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_on_tools=["Bash"]),
        )
        tool_hook = hooks["PostToolUse"][0].hooks[0]

        await tool_hook(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}, "tool_response": "ok"},
            "tool-123",
            None,
        )

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_retain_error_is_non_fatal(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("connection refused")
        hooks = create_memory_hooks(
            bank_id="test",
            client=client,
            hook_config=MemoryHookConfig(retain_on_tools=["Bash"]),
        )
        tool_hook = hooks["PostToolUse"][0].hooks[0]

        result = await tool_hook(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "make build"},
                "tool_response": "Build failed with error XYZ...",
            },
            "tool-123",
            None,
        )

        assert result == {}


class TestRecallHookMissingPrompt:
    @pytest.mark.asyncio
    async def test_returns_empty_when_prompt_key_absent(self):
        """Recall hook should handle input_data with no 'prompt' key."""
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        recall_hook = hooks["UserPromptSubmit"][0].hooks[0]

        result = await recall_hook({}, None, None)

        assert result == {}
        client.arecall.assert_not_called()


class TestExtractResultFromTranscript:
    @pytest.mark.asyncio
    async def test_missing_transcript_file(self, tmp_path):
        """Retain hook should skip when transcript file doesn't exist."""
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]

        missing_path = str(tmp_path / "nonexistent.jsonl")
        await retain_hook(
            {"transcript_path": missing_path, "stop_hook_active": False},
            None,
            None,
        )

        client.aretain.assert_not_called()

    @pytest.mark.asyncio
    async def test_corrupt_transcript_file(self, tmp_path):
        """Retain hook should skip when transcript contains only invalid JSON."""
        client = _mock_client()
        hooks = create_memory_hooks(bank_id="test", client=client)
        retain_hook = hooks["Stop"][0].hooks[0]

        corrupt_file = tmp_path / "corrupt.jsonl"
        corrupt_file.write_text("not valid json\nalso not json\n", encoding="utf-8")
        await retain_hook(
            {"transcript_path": str(corrupt_file), "stop_hook_active": False},
            None,
            None,
        )

        client.aretain.assert_not_called()
