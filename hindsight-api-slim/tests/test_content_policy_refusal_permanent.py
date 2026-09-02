"""Regression tests for vectorize-io/hindsight#3690.

A provider content-policy (AUP) refusal is a deterministic response to the
content, not a transient fault. Before this fix it was treated as an ordinary
error: the Claude Code provider replayed it through its full transport retry
budget, ``extract_facts_from_text`` wrapped it in a generic ``RuntimeError``,
and the worker rescheduled the whole task — which replayed the identical
refusal all over again before finally failing the operation.

These tests pin the permanence at all three layers:

1. the provider raises ``ProviderContentPolicyError`` on the first attempt and
   does not retry it (while ordinary errors are still retried),
2. ``extract_facts_from_text`` re-raises the permanent type when any chunk was
   refused, instead of flattening it into ``RuntimeError``,
3. ``_is_non_retryable_task_error`` classifies it, so ``execute_task`` marks the
   operation failed instead of raising ``RetryTaskAt``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.engine.llm_interface import ProviderContentPolicyError, ProviderRateLimitResetError
from hindsight_api.engine.memory_engine import _is_non_retryable_task_error
from hindsight_api.worker.exceptions import RetryTaskAt

# Verbatim shape of the refusal from the issue's worker log.
REFUSAL_TEXT = (
    "API Error: Sonnet 4.5 can't help with this. Start a new session to continue.\n\n"
    "Learn more: https://www.anthropic.com/legal/aup\n\n"
    "Request ID: req_011CeEwAh8ESFmbMGca7ZkrG"
)
TRANSIENT_TEXT = "API Error: 500 Internal Server Error"


# ---------------------------------------------------------------------------
# Layer 1: the provider does not retry a refusal
# ---------------------------------------------------------------------------


@dataclass
class _FakeOptions:
    """Stand-in for ClaudeAgentOptions; captures kwargs without importing the SDK."""

    system_prompt: str | None = None
    max_turns: int | None = None
    allowed_tools: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class _FakeAssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResultMessage:
    def __init__(self, subtype: str, is_error: bool, result: str | None) -> None:
        self.subtype = subtype
        self.is_error = is_error
        self.result = result


def _install_fake_sdk(monkeypatch, error_text: str, attempts: list[int]) -> None:
    """Patch the Agent SDK so every query yields one is_error ResultMessage."""
    import claude_agent_sdk

    async def fake_query(prompt: str, options: _FakeOptions):
        attempts.append(1)
        yield _FakeResultMessage(subtype="success", is_error=True, result=error_text)

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)


def _instantiate_provider():
    from hindsight_api.engine.providers.claude_code_llm import ClaudeCodeLLM

    return ClaudeCodeLLM(
        provider="claude-code",
        api_key="",
        base_url="",
        model="claude-haiku-4-5",
        reasoning_effort="low",
    )


@pytest.mark.asyncio
async def test_call_raises_permanent_error_without_retrying(monkeypatch):
    """A refusal must fail on attempt 1 with the permanent type, not burn the budget."""
    attempts: list[int] = []
    _install_fake_sdk(monkeypatch, REFUSAL_TEXT, attempts)

    with pytest.raises(ProviderContentPolicyError) as excinfo:
        await _instantiate_provider().call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=3,
            initial_backoff=0.0,
            max_backoff=0.0,
            scope="retain_extract_facts",
        )

    assert "anthropic.com/legal/aup" in str(excinfo.value)
    assert len(attempts) == 1, f"refusal must not be retried, but the SDK was queried {len(attempts)} times"


@pytest.mark.asyncio
async def test_call_still_retries_ordinary_errors(monkeypatch):
    """The permanence guard must be narrow: transient errors keep their retries."""
    attempts: list[int] = []
    _install_fake_sdk(monkeypatch, TRANSIENT_TEXT, attempts)

    with pytest.raises(RuntimeError) as excinfo:
        await _instantiate_provider().call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            initial_backoff=0.0,
            max_backoff=0.0,
            scope="retain_extract_facts",
        )

    assert not isinstance(excinfo.value, ProviderContentPolicyError)
    assert len(attempts) == 3, f"expected 1 initial attempt + 2 retries, got {len(attempts)}"


@pytest.mark.asyncio
async def test_call_with_tools_raises_permanent_error_without_retrying(monkeypatch):
    """call_with_tools() classifies the refusal the same way call() does."""
    import claude_agent_sdk

    attempts: list[int] = []

    class _FakeClient:
        def __init__(self, options: _FakeOptions) -> None:
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, prompt: str) -> None:
            return None

        async def receive_response(self):
            attempts.append(1)
            yield _FakeResultMessage(subtype="success", is_error=True, result=REFUSAL_TEXT)

    @dataclass
    class _FakeSdkMcpTool:
        name: str
        description: str
        input_schema: dict[str, Any]
        handler: Any

    def fake_create_sdk_mcp_server(name: str, version: str, tools=None):
        return {"name": name, "version": version, "tools": tools}

    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", _FakeOptions)
    monkeypatch.setattr(claude_agent_sdk, "AssistantMessage", _FakeAssistantMessage)
    monkeypatch.setattr(claude_agent_sdk, "TextBlock", _FakeTextBlock)
    monkeypatch.setattr(claude_agent_sdk, "ResultMessage", _FakeResultMessage)
    monkeypatch.setattr(claude_agent_sdk, "ToolUseBlock", type("ToolUseBlock", (), {}))
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr(claude_agent_sdk, "SdkMcpTool", _FakeSdkMcpTool)
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", fake_create_sdk_mcp_server)

    with pytest.raises(ProviderContentPolicyError):
        await _instantiate_provider().call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    }
                }
            ],
            max_retries=3,
            initial_backoff=0.0,
            max_backoff=0.0,
            scope="test",
        )

    assert len(attempts) == 1, f"refusal must not be retried, but the SDK was queried {len(attempts)} times"


# ---------------------------------------------------------------------------
# Layer 2: extract_facts_from_text keeps the permanent type
# ---------------------------------------------------------------------------


def _extraction_config() -> SimpleNamespace:
    """Only the two fields extract_facts_from_text reads before chunk dispatch."""
    return SimpleNamespace(retain_chunk_size=1000, retain_structured_chunk_size=1000)


async def _run_extraction(chunk_errors: dict[int, Exception]):
    """Extract from a 2-chunk text where the given chunk indices raise."""
    from hindsight_api.engine.retain import fact_extraction

    async def fake_chunk_extract(*, chunk_index: int, **_kwargs):
        error = chunk_errors.get(chunk_index)
        if error is not None:
            raise error
        return [], fact_extraction.TokenUsage()

    with patch.object(fact_extraction, "_extract_facts_with_auto_split", side_effect=fake_chunk_extract):
        return await fact_extraction.extract_facts_from_text(
            text="First sentence. " * 100 + "\n\n" + "Second sentence. " * 100,
            event_date=None,
            llm_config=SimpleNamespace(),
            agent_name="agent",
            config=_extraction_config(),
        )


@pytest.mark.asyncio
async def test_refused_chunk_raises_permanent_error():
    """One refused chunk makes the whole extraction permanently failed, not retryable."""
    with pytest.raises(ProviderContentPolicyError) as excinfo:
        await _run_extraction({0: ProviderContentPolicyError(f"Claude Code reported an error: {REFUSAL_TEXT}")})

    assert "content policy" in str(excinfo.value).lower()
    assert "chunk 0" in str(excinfo.value)
    assert _is_non_retryable_task_error(excinfo.value) is True


@pytest.mark.asyncio
async def test_refusal_wins_over_a_sibling_quota_defer():
    """A refusal alongside a deferrable quota error still fails permanently.

    Deferring would reschedule the task for a batch that can never succeed: the
    refused chunk is refused again whenever the quota reopens.
    """
    errors = {
        0: ProviderRateLimitResetError(retry_at=datetime.now(UTC) + timedelta(hours=1), message="weekly limit"),
        1: ProviderContentPolicyError(f"Claude Code reported an error: {REFUSAL_TEXT}"),
    }
    with pytest.raises(ProviderContentPolicyError):
        await _run_extraction(errors)


@pytest.mark.asyncio
async def test_transient_chunk_failure_stays_retryable():
    """Without a refusal the existing generic failure (and its retry) is unchanged."""
    with pytest.raises(RuntimeError) as excinfo:
        await _run_extraction({1: RuntimeError("connection reset by peer")})

    assert not isinstance(excinfo.value, ProviderContentPolicyError)
    assert _is_non_retryable_task_error(excinfo.value) is False


# ---------------------------------------------------------------------------
# Layer 3: the worker fails the operation instead of rescheduling it
# ---------------------------------------------------------------------------


async def _ensure_bank(pool, bank_id: str) -> None:
    await pool.execute(
        "INSERT INTO banks (bank_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        bank_id,
        bank_id,
    )


async def _create_pending_operation(pool, bank_id: str, operation_id: uuid.UUID) -> None:
    payload = json.dumps(
        {
            "type": "batch_retain",
            "operation_id": str(operation_id),
            "bank_id": bank_id,
            "contents": [{"content": "test", "document_id": "doc-1"}],
        }
    )
    await pool.execute(
        """
        INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
        VALUES ($1, $2, 'retain', 'pending', $3::jsonb)
        """,
        operation_id,
        bank_id,
        payload,
    )


@pytest.mark.asyncio
async def test_execute_task_marks_refused_retain_failed_without_retry(memory):
    """execute_task must fail the operation on the first refusal, not RetryTaskAt."""
    bank_id = f"test-worker-{uuid.uuid4().hex[:8]}"
    operation_id = uuid.uuid4()

    pool = await memory._get_pool()
    await _ensure_bank(pool, bank_id)
    await _create_pending_operation(pool, bank_id, operation_id)

    refusal = ProviderContentPolicyError(
        "Fact extraction refused by provider content policy: 1 of 1 failed chunks (1 total) were refused; "
        f"retrying cannot succeed. First failures: chunk 0: ProviderContentPolicyError: {REFUSAL_TEXT}"
    )

    task_dict = {
        "type": "batch_retain",
        "operation_id": str(operation_id),
        "bank_id": bank_id,
        "contents": [{"content": "test", "document_id": "doc-1"}],
    }

    with patch.object(memory, "_handle_batch_retain", side_effect=refusal):
        try:
            await memory.execute_task(task_dict)
        except RetryTaskAt as exc:
            pytest.fail(f"A content-policy refusal must not be retried, but execute_task raised {exc!r}")

    row = await pool.fetchrow(
        "SELECT status, error_message FROM async_operations WHERE operation_id = $1",
        operation_id,
    )
    assert row is not None, "Operation row disappeared"
    assert row["status"] == "failed", f"Expected status='failed' after a refusal, got {row['status']!r}"
    assert "content policy" in (row["error_message"] or "").lower()

    await pool.execute("DELETE FROM async_operations WHERE operation_id = $1", operation_id)
    await pool.execute("DELETE FROM banks WHERE bank_id = $1", bank_id)
