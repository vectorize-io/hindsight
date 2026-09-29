"""HindsightProvider: recall injection, retain, feedback-loop avoidance.

These drive the *real* Agent Framework ContextProvider / SessionContext (only
the Hindsight client is mocked), so any drift in the framework's hook API or
the SessionContext contract fails these tests loudly.
"""

import pytest
from conftest import fake_client, msg, session_context

from hindsight_agent_framework import HindsightProvider


def _provider(client, **kwargs) -> HindsightProvider:
    return HindsightProvider(bank_id="test-bank", client=client, **kwargs)


# ── before_run (recall) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_before_run_injects_memories():
    client = fake_client("User prefers vegetarian food")
    provider = _provider(client)
    ctx = session_context(input_texts=("suggest a recipe",))

    await provider.before_run(agent=None, session=None, context=ctx, state={})

    client.arecall.assert_awaited_once()
    assert client.arecall.await_args.kwargs["bank_id"] == "test-bank"
    assert client.arecall.await_args.kwargs["query"] == "suggest a recipe"
    injected = "\n".join(ctx.instructions)
    assert "## Memories" in injected
    assert "User prefers vegetarian food" in injected


@pytest.mark.asyncio
async def test_before_run_no_memories_no_injection():
    client = fake_client()  # empty results
    provider = _provider(client)
    ctx = session_context()
    await provider.before_run(agent=None, session=None, context=ctx, state={})
    assert ctx.instructions == []


@pytest.mark.asyncio
async def test_before_run_disabled_skips_recall():
    client = fake_client("x")
    provider = _provider(client, auto_recall=False)
    ctx = session_context()
    await provider.before_run(agent=None, session=None, context=ctx, state={})
    client.arecall.assert_not_awaited()
    assert ctx.instructions == []


@pytest.mark.asyncio
async def test_before_run_recall_failure_is_silent():
    client = fake_client()
    client.arecall.side_effect = RuntimeError("server down")
    provider = _provider(client)
    ctx = session_context()
    # Must not raise — a recall failure can never block the agent.
    await provider.before_run(agent=None, session=None, context=ctx, state={})
    assert ctx.instructions == []


@pytest.mark.asyncio
async def test_query_built_from_user_messages_only():
    client = fake_client("m")
    provider = _provider(client)
    ctx = session_context(input_texts=("first part", "second part"))
    await provider.before_run(agent=None, session=None, context=ctx, state={})
    assert client.arecall.await_args.kwargs["query"] == "first part\nsecond part"


# ── after_run (retain) ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_after_run_retains_user_and_assistant():
    client = fake_client()
    provider = _provider(client)
    ctx = session_context(input_texts=("build the parser",), response_text="parser built with PLY")

    await provider.after_run(agent=None, session=None, context=ctx, state={})

    client.aretain.assert_awaited_once()
    kwargs = client.aretain.await_args.kwargs
    assert kwargs["bank_id"] == "test-bank"
    assert kwargs["context"] == "agent-framework"
    assert "[user]" in kwargs["content"] and "build the parser" in kwargs["content"]
    assert "[assistant]" in kwargs["content"] and "parser built with PLY" in kwargs["content"]


@pytest.mark.asyncio
async def test_after_run_disabled_skips_retain():
    client = fake_client()
    provider = _provider(client, auto_retain=False)
    ctx = session_context(response_text="hi")
    await provider.after_run(agent=None, session=None, context=ctx, state={})
    client.aretain.assert_not_awaited()


@pytest.mark.asyncio
async def test_after_run_empty_transcript_skips_retain():
    client = fake_client()
    provider = _provider(client)
    ctx = session_context(input_texts=("   ",))  # nothing meaningful
    await provider.after_run(agent=None, session=None, context=ctx, state={})
    client.aretain.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_feedback_loop_injected_memories_not_retained():
    client = fake_client("Secret recalled memory")
    provider = _provider(client)
    ctx = session_context(input_texts=("a question",), response_text="an answer")

    # Recall injects a "## Memories" block (into instructions), then retain runs.
    await provider.before_run(agent=None, session=None, context=ctx, state={})
    await provider.after_run(agent=None, session=None, context=ctx, state={})

    retained = client.aretain.await_args.kwargs["content"]
    assert "## Memories" not in retained
    assert "Secret recalled memory" not in retained


@pytest.mark.asyncio
async def test_retain_failure_is_silent():
    client = fake_client()
    client.aretain.side_effect = RuntimeError("server down")
    provider = _provider(client)
    ctx = session_context(response_text="a")
    # Must not raise.
    await provider.after_run(agent=None, session=None, context=ctx, state={})


# ── construction ─────────────────────────────────────────────────────────────


def test_is_a_real_context_provider():
    from agent_framework import ContextProvider

    provider = _provider(fake_client())
    assert isinstance(provider, ContextProvider)
    assert provider.source_id == "hindsight"
