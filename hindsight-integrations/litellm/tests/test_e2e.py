"""End-to-end tests for hindsight-litellm integration.

Exercises the full retain → recall/reflect roundtrip against a live
Hindsight instance and a live LLM provider. By default these tests are
skipped; set ``OPENAI_API_KEY`` and point ``HINDSIGHT_API_URL`` at a
reachable Hindsight server to enable them.

Run with::

    uv run pytest tests/test_e2e.py -v -s

Environment variables:
    HINDSIGHT_API_URL     URL of a reachable Hindsight server
                          (default: http://localhost:8888)
    HINDSIGHT_API_KEY     API key for the Hindsight server (optional)
    OPENAI_API_KEY        OpenAI API key for the LLM provider
    HINDSIGHT_LITELLM_E2E_MODEL
                          LLM model name to use (default: gpt-4o-mini)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest
import requests

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
LLM_MODEL = os.getenv("HINDSIGHT_LITELLM_E2E_MODEL", "gpt-4o-mini")

# Sleep time between retain and recall to allow fact extraction to complete.
RETAIN_SLEEP = 5


def _hindsight_available() -> bool:
    try:
        r = requests.get(f"{HINDSIGHT_API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _openai_key_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)
requires_openai = pytest.mark.skipif(
    not _openai_key_available(),
    reason="OPENAI_API_KEY not set",
)
requires_all = pytest.mark.skipif(
    not (_hindsight_available() and _openai_key_available()),
    reason=f"Requires reachable Hindsight at {HINDSIGHT_API_URL} and OPENAI_API_KEY",
)

# Every test in this file is a real-LLM / real-service test: it drives a live
# Hindsight server (server-side fact extraction) and/or makes real provider
# calls (OpenAI completions, etc.). Mark the whole module so it forms the
# "real LLM" bucket — excluded from the deterministic PR-CI bucket via
# `-m "not requires_real_llm"` and run on its own via `-m requires_real_llm`.
# The skipif guards above still apply at runtime, so a missing key/server skips
# gracefully within the bucket rather than failing.
pytestmark = pytest.mark.requires_real_llm


@pytest.fixture
def bank_id():
    return f"litellm-e2e-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def client():
    from hindsight_client import Hindsight

    c = Hindsight(base_url=HINDSIGHT_API_URL)
    try:
        yield c
    finally:
        # Close the client so aiohttp doesn't leak the ClientSession created
        # by the sync calls above (surfaced as "Unclosed client session").
        try:
            c.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def setup_and_teardown(request, bank_id, client):
    """Create bank before each test, clean up after.

    Only runs the bank lifecycle when Hindsight is actually reachable; if
    the test was already skipped via ``requires_hindsight`` we still get
    here for fixture resolution, so guard the network calls.
    """
    if not _hindsight_available():
        yield
        return

    from hindsight_litellm import cleanup

    client.create_bank(bank_id, name=f"LiteLLM E2E {bank_id}")
    yield
    cleanup()
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(client.adelete_bank(bank_id))
        # Close the async session in the same loop that created it.
        loop.run_until_complete(client.aclose())
        loop.close()
    except Exception:
        pass


def _recall_until_contains(query, needles, *, attempts=12, delay=1.0):
    """Poll recall() until any needle appears in the results, or timeout.

    retain → recall needs a moment for fact extraction + indexing; a single
    fixed sleep makes the assertion flaky, so poll instead of sleeping once.
    Returns the (lowercased) joined result text — empty/no-match on timeout, so
    the caller's assertion still fails with useful context.
    """
    from hindsight_litellm import recall

    joined = ""
    for _ in range(attempts):
        results = recall(query)
        joined = " ".join(r.text.lower() for r in results)
        if any(n in joined for n in needles):
            return joined
        time.sleep(delay)
    return joined


# ── Direct Memory APIs ────────────────────────────────────────────


@requires_hindsight
class TestDirectAPIs:
    """retain(), recall(), reflect() direct APIs without LLM calls."""

    def test_retain_and_recall_roundtrip(self, bank_id):
        from hindsight_litellm import configure, recall, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("The user's favorite programming language is Rust", sync=True)
        retain("The user works at Acme Corp as a backend engineer", sync=True)
        time.sleep(RETAIN_SLEEP)

        results = recall("What programming language does the user prefer?")
        assert results, "Expected at least one memory returned"
        texts = " ".join(r.text.lower() for r in results)
        assert "rust" in texts, f"Expected 'rust' in recall, got: {texts}"

    def test_recall_empty_bank(self, bank_id):
        from hindsight_litellm import configure, recall, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        results = recall("anything")
        assert len(results) == 0, "Empty bank should return no results"

    def test_retain_and_reflect_roundtrip(self, bank_id):
        from hindsight_litellm import configure, reflect, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("The user loves hiking in the mountains every weekend", sync=True)
        retain("The user owns a golden retriever named Max", sync=True)
        time.sleep(RETAIN_SLEEP)

        result = reflect("What are the user's hobbies and pets?")
        assert result.text, "Expected non-empty reflect response"
        assert len(result.text) > 20, f"Reflect response too short: {result.text}"

    def test_recall_with_tags(self, bank_id):
        from hindsight_litellm import configure, recall, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("User prefers dark mode", tags=["user:alice"], sync=True)
        retain("User prefers light mode", tags=["user:bob"], sync=True)
        time.sleep(RETAIN_SLEEP)

        alice_results = recall(
            "color theme preference",
            recall_tags=["user:alice"],
            recall_tags_match="any_strict",
        )
        assert alice_results, "Expected results for alice"
        texts = " ".join(r.text.lower() for r in alice_results)
        assert "dark" in texts, f"Expected 'dark' for alice, got: {texts}"

    def test_recall_include_entities(self, bank_id):
        from hindsight_litellm import configure, recall, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("Alice works with Bob on the Phoenix project at TechCorp", sync=True)
        time.sleep(RETAIN_SLEEP)

        results = recall("Phoenix project", include_entities=True)
        assert results, "Expected results with entities enabled"

    def test_reflect_with_tags(self, bank_id):
        from hindsight_litellm import configure, reflect, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("Session A: User discussed Python best practices", tags=["session:a"], sync=True)
        time.sleep(RETAIN_SLEEP)

        result = reflect(
            "What was discussed?",
            recall_tags=["session:a"],
            recall_tags_match="any_strict",
        )
        assert result.text, "Expected reflect response with tag filter"

    def test_async_retain_and_recall(self, bank_id):
        from hindsight_litellm import arecall, aretain, configure, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        async def run():
            await aretain("User is building a FastAPI app", bank_id=bank_id, sync=True)
            await asyncio.sleep(RETAIN_SLEEP)
            return await arecall("What is the user building?", bank_id=bank_id)

        results = asyncio.new_event_loop().run_until_complete(run())
        assert results, "Expected async recall results"
        texts = " ".join(r.text.lower() for r in results)
        assert "fastapi" in texts or "api" in texts, f"Expected FastAPI in recall, got: {texts}"


# ── enable() Monkeypatch ──────────────────────────────────────────


@requires_all
class TestEnableMonkeypatch:
    """enable() monkeypatch path: auto inject + auto store."""

    def test_completion_stores_and_injects(self, bank_id):
        import hindsight_litellm
        from hindsight_litellm import configure, disable, enable, recall, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL, sync_storage=True)
        set_defaults(bank_id=bank_id)
        enable()

        hindsight_litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "My name is Alice and I love Python."}],
        )
        time.sleep(RETAIN_SLEEP)

        response = hindsight_litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "What do you know about me?"}],
        )
        assert response.choices[0].message.content, "Expected LLM response"

        disable()
        results = recall("name and programming language")
        texts = " ".join(r.text.lower() for r in results)
        assert "alice" in texts or "python" in texts, f"Expected stored facts, got: {texts}"

    def test_completion_injection_uses_recalled_memories(self, bank_id):
        import hindsight_litellm
        from hindsight_litellm import configure, enable, retain, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)

        retain("The user's favorite color is indigo", sync=True)
        time.sleep(RETAIN_SLEEP)

        enable()
        response = hindsight_litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "What is my favorite color?"}],
        )
        content = response.choices[0].message.content.lower()
        assert "indigo" in content, f"Expected 'indigo' injected into response, got: {content}"

    def test_per_call_bank_id_override(self, bank_id):
        import hindsight_litellm
        from hindsight_client import Hindsight
        from hindsight_litellm import configure, enable, retain, set_defaults

        other_bank = f"other-{uuid.uuid4().hex[:8]}"
        client = Hindsight(base_url=HINDSIGHT_API_URL)
        client.create_bank(other_bank, name="Other bank")

        try:
            configure(hindsight_api_url=HINDSIGHT_API_URL)
            set_defaults(bank_id=bank_id)

            retain("Main bank fact: user likes cats", bank_id=bank_id, sync=True)
            retain("Other bank fact: user likes dogs", bank_id=other_bank, sync=True)
            time.sleep(RETAIN_SLEEP)

            enable()
            response = hindsight_litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "What animal does the user like?"}],
                hindsight_bank_id=other_bank,
            )
            content = response.choices[0].message.content.lower()
            assert "dog" in content, f"Expected 'dogs' from other bank, got: {content}"
        finally:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(client.adelete_bank(other_bank))
            except Exception:
                pass
            try:
                loop.run_until_complete(client.aclose())
            except Exception:
                pass
            loop.close()

    def test_streaming_stores_after_consumption(self, bank_id):
        import hindsight_litellm
        from hindsight_litellm import configure, disable, enable, recall, set_defaults

        configure(hindsight_api_url=HINDSIGHT_API_URL, sync_storage=True)
        set_defaults(bank_id=bank_id)
        enable()

        stream = hindsight_litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "My hobby is chess and I play every Sunday."}],
            stream=True,
        )
        content = ""
        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content += delta.content

        assert content, "Expected streamed content"
        time.sleep(RETAIN_SLEEP)

        disable()
        results = recall("hobby")
        texts = " ".join(r.text.lower() for r in results)
        assert "chess" in texts, f"Expected 'chess' in retained facts, got: {texts}"


# ── hindsight_memory() Context Manager ───────────────────────────


@requires_all
class TestContextManager:
    """hindsight_memory() context manager."""

    def test_basic_store_and_retrieve(self, bank_id):
        import hindsight_litellm
        from hindsight_litellm import (
            configure,
            hindsight_memory,
            is_enabled,
            set_defaults,
        )

        with hindsight_memory(
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            store_conversations=True,
            inject_memories=False,
        ):
            hindsight_litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "I prefer TypeScript over JavaScript."}],
            )

        assert not is_enabled(), "Should be disabled after context exit"

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(bank_id=bank_id)
        texts = _recall_until_contains("programming language preference", ["typescript", "javascript"])
        assert "typescript" in texts or "javascript" in texts, f"Expected stored fact, got: {texts}"

    def test_context_manager_with_session_id(self, bank_id):
        from hindsight_litellm import get_defaults, hindsight_memory

        with hindsight_memory(
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            session_id="conv-abc",
            inject_memories=False,
        ):
            d = get_defaults()
            assert d.session_id == "conv-abc"

    def test_context_manager_with_use_reflect(self, bank_id):
        import hindsight_litellm
        from hindsight_litellm import hindsight_memory, retain

        retain(
            "User is training for a marathon",
            bank_id=bank_id,
            hindsight_api_url=HINDSIGHT_API_URL,
            sync=True,
        )
        time.sleep(RETAIN_SLEEP)

        with hindsight_memory(
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            use_reflect=True,
            inject_memories=True,
            store_conversations=False,
        ):
            response = hindsight_litellm.completion(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "What fitness goals do I have?"}],
            )
        content = response.choices[0].message.content.lower()
        assert "marathon" in content or "running" in content or "training" in content, (
            f"Expected marathon context injected via reflect, got: {content}"
        )


# ── wrap_openai() Native Wrapper ─────────────────────────────────


@requires_all
class TestWrapOpenAI:
    """wrap_openai() native client wrapper."""

    def test_stores_and_injects(self, bank_id):
        from hindsight_litellm import configure, recall, set_defaults, wrap_openai
        from openai import OpenAI

        client = wrap_openai(
            OpenAI(),
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            store_conversations=True,
            inject_memories=True,
        )

        try:
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "My project is called Nebula and uses Go."}],
            )
            time.sleep(RETAIN_SLEEP)

            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "What project am I working on?"}],
            )
            content = response.choices[0].message.content.lower()
            assert "nebula" in content or "go" in content, (
                f"Expected Nebula/Go injected via wrap_openai, got: {content}"
            )
            # Quiet ruff F841 for the imports above used only for type-side-effects.
            configure(hindsight_api_url=HINDSIGHT_API_URL)
            set_defaults(bank_id=bank_id)
            _ = recall
        finally:
            client.close()

    def test_per_call_override(self, bank_id):
        from hindsight_litellm import wrap_openai
        from openai import OpenAI

        wrapped = wrap_openai(
            OpenAI(),
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
        )

        try:
            response = wrapped.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                hindsight_inject_memories=False,
                hindsight_store_conversations=False,
            )
            assert response.choices[0].message.content, "Expected response"
        finally:
            wrapped.close()

    def test_streaming(self, bank_id):
        from hindsight_litellm import configure, recall, set_defaults, wrap_openai
        from openai import OpenAI

        wrapped = wrap_openai(
            OpenAI(),
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            store_conversations=True,
            inject_memories=False,
        )

        try:
            stream = wrapped.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "My favorite sport is basketball."}],
                stream=True,
            )
            content = ""
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content

            assert content, "Expected streamed content"
            time.sleep(RETAIN_SLEEP)

            configure(hindsight_api_url=HINDSIGHT_API_URL)
            set_defaults(bank_id=bank_id)
            results = recall("sport")
            texts = " ".join(r.text.lower() for r in results)
            assert "basketball" in texts, f"Expected 'basketball' stored from stream, got: {texts}"
        finally:
            wrapped.close()


# ── Design Fix Verification ───────────────────────────────────────


@requires_hindsight
class TestDesignFixes:
    """E2E tests verifying PR #1711 design fixes behave as a user would expect."""

    def test_configure_without_bank_id_is_not_configured(self, client):
        from hindsight_litellm import cleanup, configure, is_configured, is_enabled

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        assert not is_enabled(), "Should not be enabled after configure() with no bank_id"
        assert is_configured() is False, "is_configured() should be False without explicit bank_id"
        cleanup()

    def test_enable_requires_bank_id(self, client):
        from hindsight_litellm import cleanup, configure, enable

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        with pytest.raises(RuntimeError, match="bank_id"):
            enable()
        cleanup()

    @requires_openai
    def test_injection_mode_prepend_user_injects_into_user_message(self, bank_id):
        """PREPEND_USER mode should put memories into the user message, not a system message."""
        import hindsight_litellm
        import litellm as _litellm
        from hindsight_litellm import (
            cleanup,
            configure,
            disable,
            enable,
            retain,
            set_defaults,
        )

        configure(hindsight_api_url=HINDSIGHT_API_URL)
        set_defaults(
            bank_id=bank_id,
            injection_mode="prepend_user",
            inject_memories=True,
            store_conversations=False,
        )

        retain("The user's preferred language is Haskell", bank_id=bank_id, sync=True)
        time.sleep(RETAIN_SLEEP)

        injected_messages = []
        original = _litellm.completion

        def spy_completion(*args, **kwargs):
            injected_messages.extend(kwargs.get("messages", []))
            raise RuntimeError("stop_before_llm_call")

        enable()
        _litellm.completion = spy_completion
        try:
            try:
                hindsight_litellm.completion(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": "What language do I prefer?"}],
                )
            except RuntimeError as e:
                if "stop_before_llm_call" not in str(e):
                    raise
        finally:
            _litellm.completion = original
            disable()
            cleanup()

        assert injected_messages, "Expected messages to be captured by spy"
        user_msgs = [m for m in injected_messages if m.get("role") == "user"]
        system_msgs = [m for m in injected_messages if m.get("role") == "system"]

        assert user_msgs, "Expected at least one user message"
        user_content = user_msgs[-1]["content"]
        assert "Haskell" in user_content, f"Expected 'Haskell' in user message, got: {user_content}"
        assert not any("Haskell" in m.get("content", "") for m in system_msgs), (
            "Memory should NOT be in system message for PREPEND_USER mode"
        )

    def test_context_manager_full_restore(self, bank_id):
        """hindsight_memory() must restore ALL settings (tags, recall_tags, sync_storage, etc.)."""
        from hindsight_litellm import (
            cleanup,
            configure,
            get_config,
            get_defaults,
            hindsight_memory,
            set_defaults,
        )

        configure(
            hindsight_api_url=HINDSIGHT_API_URL,
            sync_storage=True,
        )
        set_defaults(
            bank_id=bank_id,
            tags=["env:prod"],
            recall_tags=["env:prod"],
            recall_tags_match="any_strict",
            use_reflect=True,
            reflect_context="Be concise.",
        )

        with hindsight_memory(
            hindsight_api_url=HINDSIGHT_API_URL,
            bank_id=bank_id,
            tags=["env:test"],
            recall_tags=["env:test"],
            recall_tags_match="all",
            use_reflect=False,
        ):
            inner = get_defaults()
            assert inner.tags == ["env:test"]
            assert inner.recall_tags == ["env:test"]
            assert inner.recall_tags_match == "all"
            assert inner.use_reflect is False

        restored_config = get_config()
        restored_defaults = get_defaults()

        assert restored_config.sync_storage is True, "sync_storage not restored"
        assert restored_defaults.tags == ["env:prod"], f"tags not restored: {restored_defaults.tags}"
        assert restored_defaults.recall_tags == ["env:prod"], (
            f"recall_tags not restored: {restored_defaults.recall_tags}"
        )
        assert restored_defaults.recall_tags_match == "any_strict", (
            f"recall_tags_match not restored: {restored_defaults.recall_tags_match}"
        )
        assert restored_defaults.use_reflect is True, "use_reflect not restored"
        assert restored_defaults.reflect_context == "Be concise.", (
            f"reflect_context not restored: {restored_defaults.reflect_context}"
        )
        cleanup()

    def test_invalid_budget_raises_before_any_api_call(self, client):
        """configure(budget='extreme') raises ValueError immediately, without calling API."""
        from hindsight_litellm import cleanup, configure

        with pytest.raises(ValueError, match="budget"):
            configure(hindsight_api_url=HINDSIGHT_API_URL, budget="extreme")
        cleanup()

    def test_hindsight_error_not_value_error_on_missing_bank_id(self, bank_id):
        """Missing bank_id raises HindsightError (not ValueError) from _inject_memories."""
        from hindsight_litellm import HindsightError, _inject_memories, cleanup, configure

        configure(hindsight_api_url=HINDSIGHT_API_URL)

        with pytest.raises(HindsightError):
            _inject_memories([{"role": "user", "content": "Hello"}])
        cleanup()
