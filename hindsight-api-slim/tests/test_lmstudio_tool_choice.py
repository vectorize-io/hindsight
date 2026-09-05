"""
Reproduce issue #520: Reflect fails with LM Studio due to unsupported tool_choice format.

The reflect agent forces tool selection via named tool_choice dicts on the first few iterations:
  {"type": "function", "function": {"name": "search_mental_models"}}

LM Studio (and Ollama) reject this format with HTTP 400:
  "Tool choice of type 'function' is not supported. Use 'auto', 'none', or 'required'."

For providers that drop tool_choice="required" (lmstudio/ollama), keep the full
tools list byte-identical and append a user-message suffix forcing the named
tool (#3865). Other providers still convert named tool_choice to "required" + a
filtered tools list.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

# Reflect agent tools (subset matching what agent.py uses)
REFLECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_mental_models",
            "description": "Search consolidated mental models",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search raw observations",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
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
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Finish and return the answer",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]


def _make_lmstudio_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="lmstudio",
        api_key="local",
        base_url="http://localhost:1234/v1",
        model="openai/gpt-oss-20b",
    )


def _lmstudio_400_error(
    msg: str = "Tool choice of type 'function' is not supported. Use 'auto', 'none', or 'required'.",
) -> APIStatusError:
    """Simulate the HTTP 400 LM Studio returns for unsupported tool_choice format."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {}
    return APIStatusError(
        message=msg,
        response=mock_response,
        body={"error": {"message": msg, "type": "invalid_request_error"}},
    )


def _make_tool_call_response(tool_name: str, arguments: dict) -> MagicMock:
    """Build a mock successful tool call response from the LLM API."""
    mock_tc = MagicMock()
    mock_tc.id = "call_abc123"
    mock_tc.function.name = tool_name
    mock_tc.function.arguments = json.dumps(arguments)

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 40
    mock_response.usage.total_tokens = 160
    # Explicit None: an auto-MagicMock here is truthy, so the reasoning-token
    # accounting (#2378) would do arithmetic on a MagicMock and crash.
    mock_response.usage.completion_tokens_details = None
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tc]
    return mock_response


class TestLMStudioNamedToolChoiceBug:
    """
    Reproduces issue #520.

    The reflect agent (agent.py lines 546-555) sets tool_choice to a named dict
    on the first iterations to force sequential retrieval:

        iteration=0, has_mental_models=True  → {"type": "function", "function": {"name": "search_mental_models"}}
        iteration=0, has_mental_models=False → {"type": "function", "function": {"name": "search_observations"}}
        iteration=1, has_mental_models=True  → {"type": "function", "function": {"name": "search_observations"}}
        iteration=1 or (2 with models)       → {"type": "function", "function": {"name": "recall"}}

    LM Studio rejects these dict formats with HTTP 400.
    """

    @pytest.mark.asyncio
    async def test_lmstudio_named_tool_choice_no_longer_causes_400(self):
        """
        Regression test for issue #520: named tool_choice dict is converted to
        "required" + filtered tools before the API call, so LM Studio never
        sees the unsupported format and the 400 error no longer occurs.
        """
        llm = _make_lmstudio_llm()
        named_tool_choice = LLMToolChoice.named("search_mental_models")
        success_response = _make_tool_call_response("search_mental_models", {"query": "user name"})

        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = success_response

            # Should succeed — no 400 because the dict is converted before sending
            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "What is the user's name?"}],
                tools=REFLECT_TOOLS,
                tool_choice=named_tool_choice,
                max_retries=0,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_mental_models"

        sent_kwargs = mock_create.call_args.kwargs
        # "required" is omitted because LM Studio silently drops it
        # (#1563/#1179/#1877). Tools stay full-schema; a user suffix forces the
        # named tool (#3865). See test_tool_choice_required_downgrade.py.
        assert "tool_choice" not in sent_kwargs
        assert [t["function"]["name"] for t in sent_kwargs["tools"]] == [
            "search_mental_models",
            "search_observations",
            "recall",
            "done",
        ]
        assert sent_kwargs["messages"][-1]["content"] == "You must call `search_mental_models` now."

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "forced_tool_name",
        ["search_mental_models", "search_observations", "recall"],
    )
    async def test_all_reflect_forced_tools_fail_on_lmstudio(self, forced_tool_name: str):
        """
        Each named tool_choice the reflect agent uses on iterations 0-2 triggers
        the same 400 error on LM Studio.
        """
        llm = _make_lmstudio_llm()
        named_tool_choice = LLMToolChoice.named(forced_tool_name)

        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = _lmstudio_400_error()

            with pytest.raises(APIStatusError) as exc_info:
                await llm.call_with_tools(
                    messages=[{"role": "user", "content": "Test query"}],
                    tools=REFLECT_TOOLS,
                    tool_choice=named_tool_choice,
                    max_retries=0,
                )

            assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_lmstudio_string_required_is_downgraded(self):
        """
        LM Studio does not honour the string ``tool_choice="required"`` either:
        it silently returns an empty tool_calls array (#1563/#1179/#1877). So
        "required" is downgraded to auto (omitted) for LM Studio. See the
        dedicated suite in test_tool_choice_required_downgrade.py.
        """
        llm = _make_lmstudio_llm()
        success_response = _make_tool_call_response("search_mental_models", {"query": "user name"})

        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = success_response

            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "What is the user's name?"}],
                tools=REFLECT_TOOLS,
                tool_choice=LLM_TOOL_CHOICE_REQUIRED,
                max_retries=0,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_mental_models"

        # "required" is omitted (downgraded to auto) rather than sent verbatim.
        sent_kwargs = mock_create.call_args.kwargs
        assert "tool_choice" not in sent_kwargs


class TestExpectedFixBehavior:
    """
    Tests that document the EXPECTED behavior after the fix is applied.

    For lmstudio (and ollama) providers, when tool_choice is a named dict:
      {"type": "function", "function": {"name": "search_mental_models"}}

    The fix should:
      1. Omit tool_choice (these servers drop "required")
      2. Keep the full tools list
      3. Append a user suffix forcing the named tool
    """

    @pytest.mark.asyncio
    async def test_fix_converts_named_tool_choice_and_downgrades(self):
        """
        Named tool_choice dict → normalized to "required" + a single filtered
        tool → "required" downgraded to auto (omitted) for lmstudio. The API
        never sees the unsupported dict nor the silently-dropped "required".
        """
        llm = _make_lmstudio_llm()
        named_tool_choice = LLMToolChoice.named("search_mental_models")
        success_response = _make_tool_call_response("search_mental_models", {"query": "user name"})

        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = success_response

            result = await llm.call_with_tools(
                messages=[{"role": "user", "content": "What is the user's name?"}],
                tools=REFLECT_TOOLS,
                tool_choice=named_tool_choice,
                max_retries=0,
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_mental_models"

        sent_kwargs = mock_create.call_args.kwargs
        # Fix: "required" omitted; tools stay full; suffix forces the named tool
        assert "tool_choice" not in sent_kwargs
        assert len(sent_kwargs["tools"]) == len(REFLECT_TOOLS)
        assert sent_kwargs["messages"][-1]["content"] == "You must call `search_mental_models` now."

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "forced_tool_name",
        ["search_mental_models", "search_observations", "recall"],
    )
    async def test_fix_keeps_full_tools_and_appends_force_suffix(self, forced_tool_name: str):
        """
        After fix (#3865): tools list stays the full reflect schema and a user
        suffix asks for the forced tool (equivalent to named tool_choice without
        mutating tools[]).
        """
        llm = _make_lmstudio_llm()
        named_tool_choice = LLMToolChoice.named(forced_tool_name)
        success_response = _make_tool_call_response(forced_tool_name, {"query": "test"})

        with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = success_response

            await llm.call_with_tools(
                messages=[{"role": "user", "content": "Test query"}],
                tools=REFLECT_TOOLS,
                tool_choice=named_tool_choice,
                max_retries=0,
            )

        sent_kwargs = mock_create.call_args.kwargs
        # "required" is omitted for lmstudio; tools stay full; suffix forces.
        assert "tool_choice" not in sent_kwargs
        assert len(sent_kwargs["tools"]) == len(REFLECT_TOOLS)
        assert sent_kwargs["messages"][-1]["content"] == f"You must call `{forced_tool_name}` now."

    @pytest.mark.asyncio
    async def test_fix_also_applies_to_openai_provider(self):
        """
        The fix is generalized: all providers convert named tool_choice to
        "required" + filtered tools.  OpenAI natively supports the dict format
        too, so the behaviour is semantically identical either way. The real
        OpenAI API (no base_url override) honours "required", so unlike the
        self-hosted providers it is NOT downgraded.
        """
        from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

        openai_llm = OpenAICompatibleLLM(
            provider="openai",
            api_key="sk-test",
            base_url="",
            model="gpt-4o-mini",
        )

        named_tool_choice = LLMToolChoice.named("search_mental_models")
        success_response = _make_tool_call_response("search_mental_models", {"query": "test"})

        with patch.object(openai_llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = success_response

            await openai_llm.call_with_tools(
                messages=[{"role": "user", "content": "Test"}],
                tools=REFLECT_TOOLS,
                tool_choice=named_tool_choice,
                max_retries=0,
            )

        sent_kwargs = mock_create.call_args.kwargs
        # Generalized fix applies to OpenAI too
        assert sent_kwargs["tool_choice"] == "required"
        assert len(sent_kwargs["tools"]) == 1
        assert sent_kwargs["tools"][0]["function"]["name"] == "search_mental_models"
