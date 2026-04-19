"""Tests for delta-mode mental model refresh.

Delta mode performs a surgical update on the existing mental model content:
- Unchanged sections are preserved byte-for-byte.
- Stale content is removed.
- New content from observations/facts is added, preferably by extending existing sections.

Fallback rules:
- If the mental model has no existing content, delta falls back to a full regeneration.
- If the source_query has changed since the last refresh, delta falls back to a full regeneration.

This file contains two kinds of tests:

1. TestDeltaRefreshPlumbing: fast, deterministic tests that monkey-patch reflect_async
   and the LLM call to verify branching logic (fallback conditions, provenance tracking).

2. TestDeltaRefreshGeminiEval: real-LLM behavioral evals against Gemini. These are
   gated on HINDSIGHT_RUN_GEMINI_EVALS=1 (plus a Gemini API key) because they cost
   money/time and require network access. They verify the actual quality of delta
   updates — format preservation, surgical edits, observation-grounding.
"""

import os
import uuid
from typing import Any

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.llm_wrapper import LLMConfig
from hindsight_api.engine.response_models import ReflectResult


def _canned_reflect_result(text: str, facts: list[dict] | None = None) -> ReflectResult:
    """Build a minimal ReflectResult for monkey-patching reflect_async."""
    return ReflectResult.model_validate(
        {
            "text": text,
            "based_on": {
                "observation": facts or [],
                "world": [],
                "experience": [],
                "mental-models": [],
                "directives": [],
            },
        }
    )


@pytest.fixture
def patch_reflect(monkeypatch):
    """Helper that patches memory.reflect_async to return a canned result and records the call.

    Usage:
        calls = patch_reflect(memory, text="hello", facts=[...])
        await memory.refresh_mental_model(...)
        assert len(calls) == 1
    """

    def _install(memory: MemoryEngine, *, text: str, facts: list[dict] | None = None):
        calls: list[dict] = []

        async def fake_reflect_async(**kwargs):
            calls.append(kwargs)
            return _canned_reflect_result(text, facts)

        monkeypatch.setattr(memory, "reflect_async", fake_reflect_async)
        return calls

    return _install


@pytest.fixture
def patch_llm_call(monkeypatch):
    """Patch the reflect LLM config's ``.call()`` used for the structured delta call.

    The structured-delta path passes ``response_format=DeltaOperationList``, so the
    LLM returns a Pydantic instance.  Each invocation of ``patch_llm_call`` installs
    a single canned response, in any of these shapes:

    - ``DeltaOperationList`` instance → returned as-is
    - ``[]`` (empty list) → no operations (this is the no-change case)
    - ``[{"op": "...", ...}, ...]`` → wrapped into ``{"operations": [...]}``
    - ``{"operations": [...]}`` → validated directly
    """
    from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

    def _to_op_list(resp: Any) -> DeltaOperationList:
        if isinstance(resp, DeltaOperationList):
            return resp
        if isinstance(resp, dict):
            if "operations" in resp:
                return DeltaOperationList.model_validate(resp)
            # Treat a bare op dict as a one-op list for ergonomics.
            return DeltaOperationList.model_validate({"operations": [resp]})
        if isinstance(resp, list):
            return DeltaOperationList.model_validate({"operations": resp})
        if isinstance(resp, str):
            # Tests that expect *no* call ever still install a sentinel; treat as no-op.
            return DeltaOperationList()
        raise TypeError(f"unsupported canned LLM response: {type(resp)!r}")

    def _install(memory: MemoryEngine, *, returns):
        calls: list[dict] = []
        canned = _to_op_list(returns)

        async def fake_call(*, messages, **kwargs):
            calls.append({"messages": messages, **kwargs})
            return canned

        monkeypatch.setattr(memory._reflect_llm_config, "call", fake_call)
        return calls

    return _install


class TestDeltaRefreshPlumbing:
    """Deterministic tests that verify the branching/plumbing of delta-mode refresh."""

    async def test_full_mode_does_not_call_delta_merge(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When trigger.mode='full', no second LLM call for delta merge occurs."""
        bank_id = f"test-delta-full-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nOriginal content.",
            trigger={"mode": "full"},
            request_context=request_context,
        )

        patch_reflect(memory, text="# Team\n\nRegenerated from scratch.")
        llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed is not None
        assert refreshed["content"] == "# Team\n\nRegenerated from scratch."
        assert len(llm_calls) == 0, "Delta merge LLM call must not happen in full mode"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_empty_content_falls_back_to_full(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When the mental model has no existing content there is nothing to anchor
        a surgical edit on, so delta falls back to full regeneration. The user's
        candidate from reflect_async is used verbatim.
        """
        bank_id = f"test-delta-empty-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="",  # no existing content
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        patch_reflect(memory, text="# Team\n\nFull fresh synthesis.")
        llm_calls = patch_llm_call(memory, returns=[])

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed["content"] == "# Team\n\nFull fresh synthesis."
        assert len(llm_calls) == 0  # delta path skipped entirely
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is not True

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_source_query_change_falls_back_to_full(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """If source_query changes after a refresh, the next delta run must do a full rewrite."""
        bank_id = f"test-delta-query-change-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nBaseline.",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh: establishes last_refreshed_source_query.
        patch_reflect(memory, text="# Team\n\nFirst pass.")
        patch_llm_call(memory, returns="unused-first")
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Now change the source_query — a genuine topic shift.
        await memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            source_query="Tell me about customers instead",
            request_context=request_context,
        )

        # Second refresh under the new query must do a FULL rewrite, not a delta merge.
        patch_reflect(memory, text="# Customers\n\nBrand new topic.")
        llm_calls = patch_llm_call(memory, returns="should-not-be-called")

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert refreshed["content"] == "# Customers\n\nBrand new topic."
        assert len(llm_calls) == 0, "Source-query change must bypass the delta merge"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_mode_applies_ops_when_query_stable(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """When content exists and source_query is stable, the delta LLM produces ops
        that are applied against the parsed structured doc. The unchanged section
        renders byte-identical, the new fact lands in a new block.
        """
        bank_id = f"test-delta-apply-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = (
            "# Team\n"
            "\n"
            "Alice is the lead.\n"
            "\n"
            "## Members\n"
            "\n"
            "- Alice — lead\n"
        )
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # First refresh: empty op list → structured doc unchanged → markdown is the
        # render of the parsed existing content. This also seeds the tracking column.
        patch_reflect(memory, text="ignored — full mode candidate")
        patch_llm_call(memory, returns=[])  # zero ops
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Second refresh: a new fact arrives; LLM returns one append_block op.
        candidate = "# Team\n\nAlice is the lead. Bob joined as junior engineer."
        patch_reflect(
            memory,
            text=candidate,
            facts=[
                {
                    "id": "obs-bob",
                    "text": "Bob joined the team as junior engineer",
                    "type": "observation",
                    "context": None,
                }
            ],
        )
        ops = [
            {
                "op": "append_block",
                "section_id": "members",
                "block": {
                    "type": "bullet_list",
                    "items": ["Bob — junior engineer"],
                },
            }
        ]
        llm_calls = patch_llm_call(memory, returns=ops)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert len(llm_calls) == 1, "Structured-delta LLM call must fire exactly once"
        system_msg = llm_calls[0]["messages"][0]["content"]
        user_msg = llm_calls[0]["messages"][1]["content"]
        # Prompt must include the structured doc + supporting facts + the system prompt.
        assert "minimal patch" in system_msg.lower()
        assert "operations" in system_msg.lower()
        assert "obs-bob" in user_msg
        assert "Bob joined" in user_msg
        # The structured JSON of the current doc must include the section id "members".
        assert '"id": "members"' in user_msg

        # New content includes the new bullet.
        assert "Bob — junior engineer" in refreshed["content"]
        # Unchanged section ("Alice is the lead.") still present.
        assert "Alice is the lead." in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is True
        applied = rr.get("delta_operations_applied") or []
        assert len(applied) == 1
        assert applied[0]["op"] == "append_block"
        assert applied[0]["section_id"] == "members"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_zero_ops_keeps_existing_content_byte_identical(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
    ):
        """Zero operations from the LLM must mean zero changes in the rendered output.

        This is the structural guarantee: any sections/blocks not mentioned by an
        op come through byte-identical. A no-op refresh therefore re-renders the
        same structured doc — which (after the first refresh has parsed and
        re-rendered it) is byte-stable.
        """
        bank_id = f"test-delta-noop-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = (
            "# Team\n"
            "\n"
            "Alice is the lead.\n"
            "\n"
            "## Members\n"
            "\n"
            "- Alice\n"
        )
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        # First refresh: parses + renders existing into structured form. The output
        # may not match `existing` byte-for-byte (whitespace normalised by renderer).
        patch_reflect(memory, text="ignored — full mode candidate")
        patch_llm_call(memory, returns=[])
        first = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        normalised = first["content"]

        # Second refresh: zero ops again → same bytes as first refresh.
        patch_reflect(memory, text="something completely different from existing")
        patch_llm_call(memory, returns=[])
        second = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        assert second["content"] == normalised
        rr = second.get("reflect_response") or {}
        assert rr.get("delta_applied") is True  # delta path ran; produced no changes
        assert rr.get("delta_operations_applied") == []

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_delta_llm_failure_falls_back_to_candidate(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        monkeypatch,
    ):
        """When the structured-delta LLM call raises, refresh falls back to the
        candidate markdown so the user still sees a fresh synthesis instead of
        an opaque failure.
        """
        bank_id = f"test-delta-llm-fail-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content="# Team\n\nExisting.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        # Seed tracking column with a successful zero-op refresh.
        patch_reflect(memory, text="ignored")

        async def ok_call(*, messages, **kwargs):
            from hindsight_api.engine.reflect.delta_ops import DeltaOperationList

            return DeltaOperationList()

        monkeypatch.setattr(memory._reflect_llm_config, "call", ok_call)
        await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Now the second refresh: LLM raises. Refresh must not crash; it should
        # store the candidate markdown.
        candidate = "# Team\n\nFallback candidate from reflect_async.\n"
        patch_reflect(memory, text=candidate)

        async def boom(*, messages, **kwargs):
            raise RuntimeError("simulated provider 500")

        monkeypatch.setattr(memory._reflect_llm_config, "call", boom)
        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        assert "Fallback candidate" in refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("delta_applied") is False

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_empty_reflect_answer_preserves_existing_content(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        patch_reflect,
        patch_llm_call,
        monkeypatch,
    ):
        """Regression: when the reflect agent returns an empty answer (small models
        sometimes hit this after exhausting tool-call retries), the refresh must
        NOT overwrite the existing content with an empty string.

        Previously this destroyed the working document on every transient upstream
        failure, and the next refresh saw current_content == "" and skipped the
        delta path entirely — a snowball that emptied valuable mental models.

        The scenario covered here is the realistic failure path: the structured
        delta call also fails (because the empty supporting facts produce empty
        / invalid JSON) so the fallback path kicks in. Without the guard, the
        fallback would write "" to the DB; with it, the existing content stays.
        """
        bank_id = f"test-empty-reflect-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        existing = (
            "# Team\n"
            "\n"
            "Alice is the lead.\n"
            "\n"
            "## Members\n"
            "\n"
            "- Alice\n"
        )
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Team Info",
            source_query="Tell me about the team",
            content=existing,
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        # Reflect returns "" — this is the upstream failure mode.
        patch_reflect(memory, text="")

        # Delta call also fails (mirrors the real groq behaviour where empty
        # supporting facts often produce empty / invalid JSON). Refresh then
        # falls back to the empty candidate, which the guard rejects.
        async def boom(*, messages, **kwargs):
            raise RuntimeError("simulated empty/invalid JSON from provider")

        monkeypatch.setattr(memory._reflect_llm_config, "call", boom)

        refreshed = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Existing content preserved exactly.
        assert refreshed["content"] == existing, (
            "Empty reflect answer overwrote existing content — guard regressed"
        )
        rr = refreshed.get("reflect_response") or {}
        assert rr.get("refresh_skipped") == "empty_candidate"

        await memory.delete_bank(bank_id, request_context=request_context)


# ---------------------------------------------------------------------------
# Real-Gemini evaluation tests
# ---------------------------------------------------------------------------

_GEMINI_API_KEY = (
    os.getenv("HINDSIGHT_GEMINI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_RUN_LLM_EVAL = os.getenv("HINDSIGHT_RUN_GEMINI_EVALS") == "1" and (
    bool(_GEMINI_API_KEY) or bool(_OPENAI_API_KEY)
)


pytestmark_gemini = pytest.mark.skipif(
    not _RUN_LLM_EVAL,
    reason=(
        "Real-LLM delta evals are gated. Set HINDSIGHT_RUN_GEMINI_EVALS=1 and provide "
        "GEMINI_API_KEY (preferred) or OPENAI_API_KEY to run."
    ),
)


@pytest.fixture
async def gemini_memory(memory_no_llm_verify: MemoryEngine):
    """MemoryEngine wired to a real LLM for reflect + structured delta.

    Prefers Gemini (the original target) but falls back to OpenAI when the
    Gemini key is unavailable — the structured-delta architecture works
    against either, and waiting on a single provider's key would block
    iteration. The chosen model is logged so test failures are unambiguous
    about which provider produced them.
    """
    if _GEMINI_API_KEY:
        provider = "gemini"
        model = os.getenv("HINDSIGHT_GEMINI_EVAL_MODEL", "gemini-2.0-flash")
        cfg = LLMConfig(provider=provider, api_key=_GEMINI_API_KEY, base_url="", model=model)
    else:
        provider = "openai"
        model = os.getenv("HINDSIGHT_OPENAI_EVAL_MODEL", "gpt-4o-mini")
        cfg = LLMConfig(provider=provider, api_key=_OPENAI_API_KEY or "", base_url="", model=model)
    print(f"\n[delta-eval] using provider={provider} model={model}")
    memory_no_llm_verify._reflect_llm_config = cfg
    memory_no_llm_verify._llm_config = cfg
    memory_no_llm_verify._retain_llm_config = cfg
    memory_no_llm_verify._consolidation_llm_config = cfg
    yield memory_no_llm_verify


_NEWS_FEED_SKILL_MARKDOWN = """## Purpose

Generate a concise, top-N personalized AI/ML news brief in response to user-triggered requests such as "ai news", "top 5 this week", or "what matters for builders today".

## Scope

- **In scope**: collecting, filtering, and summarizing AI/ML articles from user-preferred RSS feeds, applying user preferences stored in the AI News Feed Preferences mental model, and delivering the brief to the user.
- **Out of scope**: non-AI news, detailed article content, legal or privacy reviews beyond user preferences, and posting the brief to external platforms without explicit user approval.

## Rules

- **Always**:
  1. Use the AI News Feed Preferences mental model to retrieve user preferences; do not embed preferences in the skill file.
  2. Do not post the brief to any platform unless the user explicitly approves.
  3. Do not persist preferences locally; rely solely on the mental model.
  4. Refresh the feed after consolidation if the trigger-refresh-after-consolidation flag is true.
- **Prefer**:
  1. Provide a concise summary (about 2-3 sentences per article) for the top-N articles.
  2. Default to the top-5 articles unless the user specifies otherwise.
  3. Order articles chronologically or by relevance as per user preference.
  4. Highlight any user-specified topics or tags if present.

## Procedure

1. **Trigger detection** — identify a request containing keywords like "ai news", "top N", or "what matters".
2. **Preference retrieval** — call memory recall for the AI News Feed Preferences mental model to obtain RSS feed URLs and any filtering criteria.
3. **Feed consolidation** — fetch all feeds, de-duplicate entries, and apply any user-specified filters.
4. **Article selection** — choose the top-N articles based on date or user preference; if trigger-refresh-after-consolidation is true, re-fetch feeds before selection.
5. **Summarization** — generate a brief summary for each article, keeping it short and to the point.
6. **Approval check** — if the brief is to be posted externally, verify explicit user approval; otherwise, deliver it directly to the user.
7. **Memory retention** — store any new learnings or preferences observed during the task using memory retain.

## Inputs and Context

- **Source feeds**: user-specified RSS URLs stored in the mental model (e.g., https://aiagentmemory.org/index.xml).
- **Time window**: the latest update from each feed; typically the last 7 days for weekly briefs.
- **User preferences**: stored in the AI News Feed Preferences mental model; may include topics, tags, or language.

## Output Shape

- **Structure**: list of articles with title, publication date, source, and a 2-sentence summary.
- **Format**: plain text or markdown (as requested by the user).
- **Length**: concise — approximately 2-3 sentences per article; total brief about 200-300 words for top-5.
- **Voice/Tone**: neutral, informative, and concise; use bullet points for clarity.

## Stop Conditions

- If the mental model cannot be retrieved, refuse or request clarification.
- If the user has not provided any RSS feed URLs, ask for a preferred source.
- If the brief is requested for posting and explicit approval is missing, refuse.
- If the user explicitly requests to remove a skill or stop the briefing, comply immediately.

## Open Questions

- Desired brief length or word count?
- Preferred summary style (bullet vs paragraph).
- Whether the user wants to include non-AI but AI-related topics.
- Frequency or schedule for automated briefs (if any).
- Specific user-defined tags or topics to highlight.
"""


@pytestmark_gemini
class TestDeltaRefreshGeminiEval:
    """Real-LLM evals for the structured-delta refresh path.

    The structural guarantee these tests verify: sections and blocks not
    targeted by an LLM-emitted operation are byte-identical between the
    pre-refresh and post-refresh markdown render. This is what the
    structured-ops architecture buys us — the LLM cannot drift on text it
    never re-emits.

    Real Gemini is used (not a mock) because the failure mode we're guarding
    against is precisely "the LLM doesn't reliably do what the prompt says,
    even at temperature 0". Mocked output would prove the wiring works but
    not that the contract holds against an actual model.
    """

    async def _seed(
        self,
        memory: MemoryEngine,
        request_context: RequestContext,
        bank_id: str,
        existing_markdown: str,
        memories: list[str],
    ) -> dict[str, Any]:
        await memory.get_bank_profile(bank_id, request_context=request_context)
        mm = await memory.create_mental_model(
            bank_id=bank_id,
            name="Skill Doc",
            source_query="Document the news-feed skill: purpose, rules, procedure, stop conditions.",
            content=existing_markdown,
            trigger={"mode": "delta"},
            request_context=request_context,
        )
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": m} for m in memories],
            request_context=request_context,
        )
        await memory.wait_for_background_tasks()
        # First refresh: parses existing into structured form. With well-aligned
        # memories the LLM should emit zero ops, so the structured doc is just
        # the parsed existing content. The rendered markdown is canonicalised.
        first = await memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        return {"mm": mm, "first": first}

    async def test_no_change_when_observations_agree_with_existing(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """When observations only restate the existing doc, a second delta
        refresh produces output byte-identical to the first refresh's output.

        The first refresh canonicalises whitespace via the parser+renderer; we
        compare the *second* refresh against the *first* (not against the raw
        seed markdown), which is the actual repeat-refresh behaviour users
        will see in production.
        """
        bank_id = f"eval-delta-noop-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a concise top-N AI/ML news brief.",
                "Default brief size is top 5 unless the user specifies otherwise.",
                "Source feed: https://aiagentmemory.org/index.xml.",
                "The skill must not post externally without explicit approval.",
            ],
        )
        first_content = seeded["first"]["content"]

        second = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        second_content = second["content"]

        # Byte-identical render across refreshes when no new fact has arrived.
        assert second_content == first_content, (
            "Repeat delta refresh changed bytes when no new facts arrived.\n"
            f"--- diff sample (first 300 chars different) ---\n"
            f"first:  {first_content[:300]!r}\n"
            f"second: {second_content[:300]!r}"
        )
        rr = second.get("reflect_response") or {}
        # The LLM may emit zero ops (best case) or non-effective ops (still no
        # change to render); both are acceptable so long as the bytes match.
        assert rr.get("delta_applied") is True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_new_observation_is_merged_surgically(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """A new fact arrives; only the section relevant to it should change.

        Asserts the architectural guarantee at the section level: every
        section that the LLM did NOT name in an operation must render exactly
        the same bytes after the refresh as before. The new fact itself must
        appear somewhere in the output.
        """
        from hindsight_api.engine.reflect.structured_doc import (
            StructuredDocument,
            render_section,
        )

        bank_id = f"eval-delta-add-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a concise top-N AI/ML news brief.",
                "Default brief size is top 5.",
                "Source feed: https://aiagentmemory.org/index.xml.",
            ],
        )
        first_content = seeded["first"]["content"]
        first_struct = StructuredDocument.model_validate(
            seeded["first"]["reflect_response"]["delta_operations_applied"]
            and seeded["first"].get("structured_content")
            or {"version": 1, "sections": []}
        )
        # The first refresh's structured snapshot is what the second refresh
        # will operate on. Re-fetch via get_mental_model would also work.
        # For preservation comparison we re-parse first_content.
        from hindsight_api.engine.reflect.structured_doc import parse_markdown

        before = parse_markdown(first_content)

        # Introduce a brand-new fact that fits into "Inputs and Context" or
        # similar — but the model may pick any reasonable section.
        await gemini_memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {
                    "content": (
                        "The default time window for the news brief is the last 7 days, "
                        "matching the weekly cadence preferred by the user."
                    )
                },
            ],
            request_context=request_context,
        )
        await gemini_memory.wait_for_background_tasks()

        refreshed = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        content = refreshed["content"]
        rr = refreshed.get("reflect_response") or {}
        applied_ops = rr.get("delta_operations_applied") or []
        touched_section_ids = {op.get("section_id") for op in applied_ops if op.get("section_id")}

        # The fact must show up.
        assert "7 days" in content or "seven days" in content.lower(), (
            f"New fact about 7-day window missing from delta output: {content!r}"
        )

        # Every untouched section must render byte-identical to its pre-refresh form.
        after = parse_markdown(content)
        before_by_id = {s.id: s for s in before.sections}
        for section in after.sections:
            if section.id in touched_section_ids:
                continue
            orig = before_by_id.get(section.id)
            if orig is None:
                continue  # newly added section, no preservation contract
            assert render_section(orig) == render_section(section), (
                f"Untouched section {section.id!r} drifted between refreshes — the "
                f"structured-ops architecture's preservation guarantee was violated.\n"
                f"BEFORE:\n{render_section(orig)!r}\n"
                f"AFTER:\n{render_section(section)!r}"
            )

        assert rr.get("delta_applied") is True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_no_change_repeated_three_times_stays_byte_stable(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """Three consecutive no-change refreshes must produce three identical
        markdown outputs. This is the regression test for the original
        complaint where prose-merge delta drifted content across versions even
        when no observation changed.
        """
        bank_id = f"eval-delta-stable-{uuid.uuid4().hex[:8]}"
        seeded = await self._seed(
            gemini_memory,
            request_context,
            bank_id,
            existing_markdown=_NEWS_FEED_SKILL_MARKDOWN,
            memories=[
                "The news-feed skill produces a top-N AI brief on demand.",
                "It must not post without explicit user approval.",
            ],
        )
        c1 = seeded["first"]["content"]
        r2 = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        r3 = await gemini_memory.refresh_mental_model(
            bank_id=bank_id,
            mental_model_id=seeded["mm"]["id"],
            request_context=request_context,
        )
        assert r2["content"] == c1, "second refresh drifted vs first"
        assert r3["content"] == c1, "third refresh drifted vs first"

        await gemini_memory.delete_bank(bank_id, request_context=request_context)

    async def test_source_query_change_forces_full_rewrite(
        self, gemini_memory: MemoryEngine, request_context: RequestContext
    ):
        """Changing source_query must bypass delta and produce a full regeneration."""
        bank_id = f"eval-delta-query-change-{uuid.uuid4().hex[:8]}"
        await gemini_memory.get_bank_profile(bank_id, request_context=request_context)

        mm = await gemini_memory.create_mental_model(
            bank_id=bank_id,
            name="Subject",
            source_query="Summarize the team and how it operates.",
            content="# Team Overview\n\nAlice leads the team.\n",
            trigger={"mode": "delta"},
            request_context=request_context,
        )

        await gemini_memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice leads the team."},
                {"content": "The product is a memory system for AI agents."},
                {"content": "Customers include small SaaS startups and enterprise pilots."},
            ],
            request_context=request_context,
        )
        await gemini_memory.wait_for_background_tasks()

        # First refresh seeds tracking column under the team query.
        await gemini_memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )

        # Change the topic entirely.
        await gemini_memory.update_mental_model(
            bank_id=bank_id,
            mental_model_id=mm["id"],
            source_query="Summarize our customers and what we sell them.",
            request_context=request_context,
        )

        refreshed = await gemini_memory.refresh_mental_model(
            bank_id=bank_id, mental_model_id=mm["id"], request_context=request_context
        )
        content = refreshed["content"].lower()
        # Content should now be about customers/product, not (only) about Alice leading the team.
        assert "customer" in content or "product" in content, (
            f"Full rewrite should cover the new topic, got: {refreshed['content']!r}"
        )
        # delta_applied should be absent/False because we took the full path.
        assert (refreshed.get("reflect_response") or {}).get("delta_applied") is not True

        await gemini_memory.delete_bank(bank_id, request_context=request_context)
