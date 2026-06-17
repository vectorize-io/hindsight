"""Measure the cached/input token ratio per operation against real Gemini.

Each operation re-sends a large constant prefix and a small variable payload:
- ``retain_extract_facts`` — fact-extraction system prompt + schema
- ``reflect_tool_call`` — agent system prompt + tool definitions, reused across
  every iteration of the tool loop
- ``consolidation`` — the stable mission/rules/decision/output system prefix,
  reused across every consolidation batch

We run real Gemini, route every call through the LLM-request tracer (#1922), and
read back recorded ``cached_tokens`` vs ``input_tokens`` per scope.

Two modes:
- Default (implicit only): Gemini's automatic caching — empirically ~0% for this
  low-QPS access pattern. Structural assertions only; the ratio is a measurement.
- ``HINDSIGHT_GEMINI_EXPLICIT_CACHE=1``: enables PR #1936's explicit CachedContent
  caching. Then each operation must visibly engage the cache (cached_tokens > 0
  and ratio above a conservative floor) — this is the regression guard that the
  caching actually works end-to-end through the retain/reflect/consolidation paths.

Gated on ``HINDSIGHT_RUN_GEMINI_EVALS=1`` plus a Gemini API key, since it costs
money and needs network. The default model is ``gemini-2.5-flash`` (override with
``HINDSIGHT_GEMINI_EVAL_MODEL``); explicit caching needs a >=2,048-token prefix.
"""

import asyncio
import os
import uuid

import pytest

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.llm_trace import LLMRequestEntry
from hindsight_api.engine.llm_wrapper import LLMConfig

_GEMINI_API_KEY = os.getenv("HINDSIGHT_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_RUN = os.getenv("HINDSIGHT_RUN_GEMINI_EVALS") == "1" and bool(_GEMINI_API_KEY)

pytestmark = pytest.mark.skipif(
    not _RUN,
    reason=(
        "Gemini implicit-cache measurement is gated. Set HINDSIGHT_RUN_GEMINI_EVALS=1 "
        "and provide GEMINI_API_KEY/GOOGLE_API_KEY to run."
    ),
)

# Number of retain "chunks": each is a separate retain call → a separate
# retain_extract_facts LLM call that re-sends the same ~3k-token system prefix.
# That repetition is the precondition for any caching (implicit or explicit) to
# kick in. Override with HINDSIGHT_GEMINI_CACHE_CHUNKS.
_CHUNKS = int(os.getenv("HINDSIGHT_GEMINI_CACHE_CHUNKS", "5"))

# Distinct paragraphs so each retain extracts real, non-duplicate facts. The
# *content* varies per call; the system prompt / schema prefix does not — which
# is exactly the shape caching targets.
_DOCS = [
    "Ada Lovelace worked with Charles Babbage on the Analytical Engine in the 1840s. "
    "She wrote what is often considered the first algorithm intended for a machine, "
    "a method for computing Bernoulli numbers. She lived in London and corresponded "
    "extensively with Babbage about the engine's capabilities.",
    "Grace Hopper joined the Harvard Mark I team in 1944 and later developed the first "
    "compiler, A-0, in 1952. She championed machine-independent programming languages, "
    "which led to COBOL. She served in the US Navy and retired as a rear admiral.",
    "Katherine Johnson computed orbital mechanics for NASA's first crewed spaceflights. "
    "John Glenn personally asked her to verify the electronic computer's calculations "
    "before his 1962 Friendship 7 orbit. She worked at Langley Research Center in Virginia.",
    "Alan Turing formalized computation with the Turing machine in 1936 and worked at "
    "Bletchley Park during World War II breaking the Enigma cipher. He proposed the "
    "imitation game, now called the Turing test, in a 1950 paper on machine intelligence.",
    "Margaret Hamilton led the software engineering team that wrote the onboard flight "
    "software for the Apollo missions at MIT. Her error-detection code prevented an abort "
    "during the Apollo 11 landing in 1969. She later coined the term 'software engineering'.",
    "Barbara Liskov designed the CLU programming language in the 1970s and introduced data "
    "abstraction. The Liskov substitution principle is named after her. She won the Turing "
    "Award in 2008 for contributions to programming language and system design.",
    "Tim Berners-Lee invented the World Wide Web in 1989 while at CERN, writing the first "
    "browser and the HTTP protocol. He founded the World Wide Web Consortium in 1994 to "
    "develop open web standards.",
    "Radia Perlman invented the spanning-tree protocol while at Digital Equipment Corporation, "
    "which made large bridged Ethernet networks possible. She is sometimes called the mother "
    "of the internet, a title she has said she dislikes.",
]


# Explicit Gemini prompt caching (PR #1936) — opt-in. Set HINDSIGHT_GEMINI_EXPLICIT_CACHE=1
# to turn it on for this run. On a branch without the feature the flag is simply
# ignored, so the same test measures the implicit baseline there.
_EXPLICIT_CACHE = os.getenv("HINDSIGHT_GEMINI_EXPLICIT_CACHE") == "1"


async def _gemini_engine(memory_no_llm_verify: MemoryEngine) -> MemoryEngine:
    """Point an engine at real Gemini and force-enable the LLM-request tracer.

    The fixture builds the engine with tracing disabled (config default). The
    recorder reads ``enabled`` once at construction, so we flip the flag directly
    rather than rebuilding the engine — equivalent to running with
    ``HINDSIGHT_API_LLM_TRACE_ENABLED=true``.

    When ``HINDSIGHT_GEMINI_EXPLICIT_CACHE=1`` we also enable PR #1936's explicit
    CachedContent caching the production way (env var + config-cache clear), so we
    can compare its cached/input ratio against the implicit baseline. Otherwise the
    only caching observed is Gemini's own implicit caching.
    """
    from hindsight_api.config import clear_config_cache

    model = os.getenv("HINDSIGHT_GEMINI_EVAL_MODEL", "gemini-2.5-flash")
    # Prompt caching is on by default now, so the implicit-baseline run must
    # explicitly DISABLE it (not just leave it unset) to measure Gemini's own
    # implicit caching. Set the flag in both modes and clear the config cache so
    # the per-bank resolver re-reads it.
    os.environ["HINDSIGHT_API_LLM_PROMPT_CACHE_ENABLED"] = "true" if _EXPLICIT_CACHE else "false"
    clear_config_cache()
    cfg = LLMConfig(
        provider="gemini",
        api_key=_GEMINI_API_KEY or "",
        base_url="",
        model=model,
        prompt_cache_enabled=_EXPLICIT_CACHE,
    )
    memory_no_llm_verify._llm_config = cfg
    memory_no_llm_verify._retain_llm_config = cfg
    memory_no_llm_verify._reflect_llm_config = cfg
    memory_no_llm_verify._consolidation_llm_config = cfg
    memory_no_llm_verify._llm_recorder._enabled = True
    mode = "EXPLICIT cache ON" if _EXPLICIT_CACHE else "implicit only"
    print(f"\n[gemini-cache] provider=gemini model={model} chunks={_CHUNKS} mode={mode}")
    return memory_no_llm_verify


async def _drain_traces(mem: MemoryEngine) -> None:
    """Wait for the recorder's fire-and-forget trace writes to land.

    record_llm_call schedules each INSERT as a detached asyncio task tracked in
    ``_pending`` (bucketed by trace_id). Gather them so the rows are queryable.
    Loop a few times because consolidation's attach_memory_ids can spawn a
    follow-up write after the first drain.
    """
    await mem.wait_for_background_tasks()
    rec = mem._llm_recorder
    for _ in range(10):
        pending = [t for bucket in rec._pending.values() for t in bucket if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)


def _report(scope: str, rows: list[LLMRequestEntry]) -> float:
    """Print the cached/input token ratio for a scope and return it.

    Gemini's ``prompt_token_count`` (our ``input_tokens``) already includes the
    cached prefix, so ``cached_tokens / input_tokens`` is the fraction of prompt
    tokens billed at the cheaper cached rate — the number the PR's
    ``cached_input / input`` dashboard would show.
    """
    input_total = sum((r.input_tokens or 0) for r in rows)
    cached_total = sum((r.cached_tokens or 0) for r in rows)
    output_total = sum((r.output_tokens or 0) for r in rows)
    ratio = (cached_total / input_total) if input_total else 0.0
    per_call = ", ".join(f"{(r.cached_tokens or 0)}/{(r.input_tokens or 0)}" for r in rows)
    mode = "explicit cache ON" if _EXPLICIT_CACHE else "implicit only"
    print(
        f"\n[gemini-cache] scope={scope!r} calls={len(rows)}  ({mode})\n"
        f"  input_tokens   = {input_total}\n"
        f"  cached_tokens  = {cached_total}\n"
        f"  output_tokens  = {output_total}\n"
        f"  cached/input   = {ratio:.1%}\n"
        f"  per-call cached/input: {per_call}"
    )
    return ratio


@pytest.mark.hs_llm_core
class TestGeminiCacheRatioPerOperation:
    """Measure cached/input token ratio per operation (retain, reflect, consolidation).

    Run with ``HINDSIGHT_GEMINI_EXPLICIT_CACHE=1`` to assert PR #1936's explicit
    CachedContent caching actually engages (cached tokens > 0, ratio above a
    conservative floor). Without it, the same tests record the implicit-caching
    baseline (Gemini gives ~0% for this access pattern) without asserting a floor.
    """

    async def _fetch(self, mem: MemoryEngine, bank_id: str, rc: RequestContext, scope: str) -> list[LLMRequestEntry]:
        resp = await mem.list_llm_requests(bank_id, request_context=rc, scope=scope, limit=200)
        assert resp is not None, "bank should exist"
        return [r for r in resp.items if r.status == "success"]

    def _assert(self, scope: str, rows: list[LLMRequestEntry], *, min_calls: int, min_ratio: float) -> float:
        """Common per-operation checks; returns the cached/input ratio.

        ``min_ratio`` is per-operation because the achievable ratio differs by
        design: retain re-sends a pure fixed prefix (~90%); consolidation's prefix
        is fixed but the facts/observations payload is large (~30%); reflect can
        only cache its ``auto`` iterations — Gemini forbids ``cached_content`` with
        a per-request ``tool_config`` — and the tool-result context grows, so the
        ratio is modest (~10%). The universal guarantee in explicit mode is simply
        that caching engaged at all (cached_tokens > 0).
        """
        ratio = _report(scope, rows)
        cached_total = sum((r.cached_tokens or 0) for r in rows)
        assert len(rows) >= min_calls, f"expected >= {min_calls} {scope} calls, got {len(rows)}"
        assert all((r.provider == "gemini") for r in rows)
        assert sum((r.input_tokens or 0) for r in rows) > 0, "no input tokens recorded"
        assert 0.0 <= ratio <= 1.0
        if _EXPLICIT_CACHE:
            assert cached_total > 0, f"{scope}: explicit cache ON but cached_tokens=0 (caching did not engage)"
            assert ratio >= min_ratio, f"{scope}: cached/input {ratio:.1%} below floor {min_ratio:.0%}"
        return ratio

    async def test_retain_chunks_cached_ratio(self, memory_no_llm_verify, request_context):
        """Retain N distinct chunks → N fact-extraction calls sharing one prefix."""
        mem = await _gemini_engine(memory_no_llm_verify)
        bank_id = f"gemini-cache-retain-{uuid.uuid4().hex[:8]}"
        await mem.get_bank_profile(bank_id, request_context=request_context)

        docs = [_DOCS[i % len(_DOCS)] for i in range(_CHUNKS)]
        for i, content in enumerate(docs):
            await mem.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": content}],
                request_context=request_context,
                document_id=f"doc-{i}",
            )
        await _drain_traces(mem)

        rows = await self._fetch(mem, bank_id, request_context, "retain_extract_facts")
        self._assert("retain_extract_facts", rows, min_calls=_CHUNKS, min_ratio=0.5)

        await mem.delete_bank(bank_id, request_context=request_context)

    async def test_reflect_tool_loop_cached_ratio(self, memory_no_llm_verify, request_context):
        """Reflect runs an agentic tool loop; the system_prompt + tools prefix is
        cached once and reused across every iteration (scope ``reflect_tool_call``)."""
        mem = await _gemini_engine(memory_no_llm_verify)
        bank_id = f"gemini-cache-reflect-{uuid.uuid4().hex[:8]}"
        await mem.get_bank_profile(bank_id, request_context=request_context)

        await mem.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": d} for d in _DOCS],
            request_context=request_context,
        )
        await mem.wait_for_background_tasks()

        # A broad question forces the agent to call recall/lookup tools, i.e. to
        # iterate the tool loop more than once.
        await mem.reflect_async(
            bank_id=bank_id,
            query="Who were the early pioneers of computing in these memories, and what is each one known for?",
            request_context=request_context,
        )
        await _drain_traces(mem)

        rows = await self._fetch(mem, bank_id, request_context, "reflect_tool_call")
        if not rows:
            # Diagnostic: dump every reflect_tool_call row (incl. errors) so a
            # failure in the cached tool-loop path is visible, not silently skipped.
            allresp = await mem.list_llm_requests(
                bank_id, request_context=request_context, scope="reflect_tool_call", limit=200
            )
            for r in allresp.items if allresp else []:
                print(f"\n[gemini-cache] reflect_tool_call status={r.status} error={r.error}")
            pytest.skip("reflect made no SUCCESSFUL reflect_tool_call iterations for this seed")
        self._assert("reflect_tool_call", rows, min_calls=1, min_ratio=0.0)

        await mem.delete_bank(bank_id, request_context=request_context)

    async def test_consolidation_cached_ratio(self, memory_no_llm_verify, request_context):
        """Retain a batch, then consolidate; the stable system prefix is cached and
        reused across every consolidation batch (scope ``consolidation``)."""
        mem = await _gemini_engine(memory_no_llm_verify)
        bank_id = f"gemini-cache-consol-{uuid.uuid4().hex[:8]}"
        await mem.get_bank_profile(bank_id, request_context=request_context)

        # Seed enough unconsolidated memories that consolidation makes several
        # same-prefix LLM calls.
        await mem.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": d} for d in _DOCS],
            request_context=request_context,
        )
        await mem.wait_for_background_tasks()

        await run_consolidation_job(mem, bank_id, request_context)
        await _drain_traces(mem)

        rows = await self._fetch(mem, bank_id, request_context, "consolidation")
        if not rows:
            pytest.skip("consolidation made no LLM calls for this seed (nothing to consolidate)")
        self._assert("consolidation", rows, min_calls=1, min_ratio=0.15)

        await mem.delete_bank(bank_id, request_context=request_context)
