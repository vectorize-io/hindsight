"""Integration test: delta mental model fuses generic SEO best practices with brand voice.

Scenario:
1. Create a bank with a delta-mode mental model ("editorial-preferences").
2. Ingest an SEO best practices document -> trigger mental model refresh.
3. Ingest a brand voice document -> trigger mental model refresh (delta).
4. Verify the delta fuses both documents organically.

Requires: HINDSIGHT_RUN_GEMINI_EVALS=1 + a Gemini/OpenAI API key.
"""

import os
import uuid
from collections import Counter

import pytest

from hindsight_api import MemoryEngine, RequestContext

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
_GEMINI_KEY = os.getenv("HINDSIGHT_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_RUN = os.getenv("HINDSIGHT_RUN_GEMINI_EVALS") == "1" and (bool(_GEMINI_KEY) or bool(_OPENAI_KEY))
pytestmark = [
    pytest.mark.skipif(not _RUN, reason="Set HINDSIGHT_RUN_GEMINI_EVALS=1 + LLM API key"),
    pytest.mark.hs_llm_core,
]

# ---------------------------------------------------------------------------
# Test documents — short but representative
# ---------------------------------------------------------------------------

SEO_BEST_PRACTICES = """\
# SEO Content Best Practices

## Content Structure
- Use clear H1/H2/H3 heading hierarchy for every article.
- Keep paragraphs under 3 sentences for scannability.
- Use bullet points and numbered lists to break up dense information.

## Tone and Voice
- Write in a professional, authoritative tone.
- Use industry-standard SEO terminology (e.g., "SERP", "CTR", "backlink").
- Address the reader in second person ("you").

## Keyword Strategy
- Place primary keyword in H1, first paragraph, and meta description.
- Target keyword density of 1-2% for primary terms.
- Include long-tail question keywords in H2/H3 subheadings.

## Technical Requirements
- Meta titles: 50-60 characters, primary keyword first.
- Meta descriptions: 150-160 characters, include CTA.
- Internal links: minimum 3 per article.
- Image alt text: descriptive, keyword-rich where natural.

## E-E-A-T Compliance
- Include author bios with credentials.
- Cite authoritative sources.
- Update content quarterly to maintain freshness.
"""

BRAND_VOICE = """\
# Plot Brand Voice Guide

## Who We Are
Plot is a finance app for freelancers. We handle invoicing, expense tracking,
and tax prep for people whose income is irregular.

## Voice Principles
- We talk like a smart friend who knows about money — not a bank, not a guru.
- Clarity always wins. If a 12-year-old can't understand it, rewrite it.
- We never lecture or moralize about financial decisions.

## Tone by Context
- Marketing: Confident, slightly wry. Example: "Built for income that doesn't show up on the same day every month."
- Support: Direct, human, accountable. Example: "That's our bug, not yours. We're fixing it now."
- Product UI: Quiet, precise. Example: "Income from Stripe — Mar 14."
- Errors: Calm, specific. Example: "We couldn't sync your bank. Try reconnecting."

## Writing Rules
- Always use contractions (it's, we're, you'll).
- Use Oxford comma.
- Always say "you", never "users" or "customers".
- Avoid jargon: never say "leverage", "empower", "solution", "holistic", "game-changing".
- No puns. Wit is fine — wordplay and wry asides, not dad jokes.

## What We Sound Like
- YES: "Here's what we found." / NO: "We are pleased to present our findings."
- YES: "Looks like this payment is late." / NO: "ALERT: Payment overdue! Action required!"
"""


class TestDeltaEditorialFusion:
    """Real-LLM test verifying delta mode correctly fuses two documents."""

    async def test_delta_fuses_seo_and_brand_voice(
        self,
        memory_real_llm: MemoryEngine,
        request_context: RequestContext,
    ):
        bank_id = f"test-editorial-{uuid.uuid4().hex[:8]}"
        memory = memory_real_llm
        await memory.get_bank_profile(bank_id, request_context=request_context)

        try:
            mm = await memory.create_mental_model(
                bank_id=bank_id,
                name="Editorial Preferences",
                source_query=(
                    "What are the editorial preferences and content guidelines? "
                    "Include tone, voice, formatting rules, and vocabulary rules."
                ),
                content="",
                trigger={
                    "mode": "delta",
                    "refresh_after_consolidation": False,
                    "fact_types": ["observation"],
                    "exclude_mental_models": True,
                },
                request_context=request_context,
            )
            mm_id = mm["id"]

            # Phase 1: Ingest SEO best practices
            await memory.retain_async(
                bank_id=bank_id,
                content=SEO_BEST_PRACTICES,
                document_id="seo-best-practices",
                request_context=request_context,
            )
            mm_after_seo = await memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm_id,
                request_context=request_context,
            )
            seo_content = mm_after_seo["content"]
            assert len(seo_content) > 100, f"First refresh produced too little content: {len(seo_content)} chars"

            # Phase 2: Ingest brand voice -> delta refresh
            await memory.retain_async(
                bank_id=bank_id,
                content=BRAND_VOICE,
                document_id="brand-voice",
                request_context=request_context,
            )
            mm_after_brand = await memory.refresh_mental_model(
                bank_id=bank_id,
                mental_model_id=mm_id,
                request_context=request_context,
            )
            fused = mm_after_brand["content"]
            rr = mm_after_brand.get("reflect_response") or {}
            fused_lower = fused.lower()

            # -- Verify fusion quality --

            # Brand voice concepts present (LLM may paraphrase, check synonyms)
            for concept, signals in {
                "contractions": ["contraction", "it's", "we're", "you'll"],
                "oxford comma": ["oxford comma"],
                "vocabulary rules": ["jargon", "leverage", "empower", "forbidden"],
            }.items():
                assert any(s in fused_lower for s in signals), (
                    f"Brand voice concept '{concept}' missing (looked for {signals}).\nFused content:\n{fused[:500]}"
                )

            # SEO concepts still present (not wiped by delta)
            for concept, signals in {
                "keywords": ["keyword"],
                "structure": ["heading", "h1", "h2", "structure"],
                "seo": ["meta", "e-e-a-t", "seo", "search"],
            }.items():
                assert any(s in fused_lower for s in signals), (
                    f"SEO concept '{concept}' missing (looked for {signals}).\nFused content:\n{fused[:500]}"
                )

            # Brand voice overrides generic tone
            assert any(t in fused_lower for t in ["friend", "wry", "plot", "witty"]), (
                f"Brand-specific tone missing from fused content.\nFused:\n{fused[:500]}"
            )

            # No duplicate paragraphs
            lines = [ln.strip() for ln in fused.split("\n") if ln.strip() and not ln.strip().startswith("#")]
            dupes = {line: cnt for line, cnt in Counter(lines).items() if cnt > 1}
            assert not dupes, "Duplicate paragraphs:\n" + "\n".join(f"  [{c}x] {t[:80]}" for t, c in dupes.items())

            # based_on accumulates from both docs
            obs_count = len(rr.get("based_on", {}).get("observation", []))
            assert obs_count > 5, f"Expected observations from both docs, got {obs_count}"

        finally:
            await memory.delete_bank(bank_id, request_context=request_context)
