"""Diagnose a destructive delta: where does a false denial enter the page?

``kp_eval`` found a page that was RIGHT after wave 1 and WRONG after wave 2 —
"Release 0.9.3 was deployed to production on 18 March 2026" replaced by "there is
no record of any release being deployed to production on 18 March 2026". A
regression written into stored state, which a one-shot reflect eval cannot see.

A score cannot say where that came from, and the two candidates need opposite
fixes:

* the **reflect synthesis** inside the refresh already denies the fact, because
  the delta window genuinely does not contain it — a scoping problem; or
* the synthesis is fine and the **delta operations** delete or overwrite the good
  section anyway — a delta-application problem.

``dry_run_refresh_mental_model`` separates them: ``candidate_content`` is the raw
synthesis before any operation is applied, ``preview_content`` is the document
after. It runs the production pipeline unchanged and persists nothing, so it can
be repeated — which also answers whether the failure is stable or was one unlucky
sample.

Run with::

    cd hindsight-dev && uv run python -m benchmarks.prelude.kp_diagnose --repeats 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.models import RequestContext
from rich.console import Console

from benchmarks.prelude import hard_corpus
from benchmarks.prelude.fixture import _retain_many
from benchmarks.prelude.kp_eval import _PAGE_TRIGGER, _WAVES

console = Console()


class _CallRecorder:
    """Capture the real prompts and completions of a refresh's LLM calls.

    ``candidate_content`` and ``preview_content`` say WHAT the model concluded.
    They do not say what it was ASKED, which is the only way to tell a model that
    ignored an instruction from one that was never given it — and those need
    different fixes. So wrap the provider and keep the messages verbatim.
    """

    def __init__(self, provider):
        self.provider = provider
        self.calls: list[dict] = []
        self._call = provider.call
        self._call_with_tools = provider.call_with_tools
        provider.call = self._wrap(self._call, "call")
        provider.call_with_tools = self._wrap(self._call_with_tools, "call_with_tools")

    def _wrap(self, inner, kind):
        async def _recorded(*args, **kwargs):
            result = await inner(*args, **kwargs)
            content = getattr(result, "content", None)
            self.calls.append(
                {
                    "kind": kind,
                    "scope": kwargs.get("scope", "?"),
                    "messages": kwargs.get("messages") or (args[0] if args else []),
                    "output": content if isinstance(content, str) else repr(content),
                    "tool_calls": [tc.name for tc in getattr(result, "tool_calls", []) or []],
                }
            )
            return result

        return _recorded

    def restore(self) -> None:
        self.provider.call = self._call
        self.provider.call_with_tools = self._call_with_tools

    def by_scope(self, scope: str) -> list[dict]:
        return [c for c in self.calls if c["scope"] == scope]


def _excerpt(text: str, limit: int = 400) -> str:
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="hq-release-prod", help="question id to diagnose")
    parser.add_argument("--repeats", type=int, default=3, help="dry runs of the second wave")
    args = parser.parse_args()

    provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "")
    if not provider or provider == "mock":
        raise SystemExit("Needs a real model: set HINDSIGHT_API_LLM_PROVIDER/_MODEL/_API_KEY.")

    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        memory_llm_provider="mock",
        memory_llm_api_key="unused",
        memory_llm_model="mock",
        reflect_llm_provider=provider,
        reflect_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        reflect_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL"),
        reflect_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
        task_backend=SyncTaskBackend(),
    )
    await memory.initialize()

    ctx = RequestContext()
    corpus = hard_corpus.build()
    facts, questions = corpus.facts, corpus.questions
    question = next(q for q in questions if q.id == args.question)
    rows = [(f.id, f.text) for f in facts]
    waves = [rows[i::_WAVES] for i in range(_WAVES)]

    # Which wave carries the gold, and what does it say? The answer decides how
    # to read everything below: a denial is only a bug if the fact is already in
    # the page or in the window.
    gold_texts = {f.id: f.text for f in facts if f.id in set(question.gold)}
    gold_wave = {
        fid: next((i + 1 for i, w in enumerate(waves) if any(rid == fid for rid, _ in w)), None) for fid in gold_texts
    }

    bank_id = f"prelude-diag-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=ctx)
    await memory._config_resolver.update_bank_config(bank_id, {"enable_auto_consolidation": False}, ctx)

    page = await memory.create_knowledge_page(
        bank_id=bank_id,
        name=f"diag {question.id}",
        source_query=question.question,
        content="",
        trigger=_PAGE_TRIGGER,
        request_context=ctx,
    )
    if page is None:
        raise RuntimeError(f"creating the knowledge page for {question.id} returned nothing")
    mm_id = page["mental_model_id"]

    console.print(f"[bold]{question.id}[/bold]  {question.question}")
    for fid, text in gold_texts.items():
        console.print(f"  gold {fid} (wave {gold_wave[fid]}): {text}")
    console.print()

    # Wave 1 for real, so the page holds the correct claim going in.
    await _retain_many(memory, bank_id, ctx, waves[0], "world")
    await memory.refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=ctx)
    model = await memory.get_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=ctx)
    console.print(f"[bold]after wave 1[/bold] (persisted)\n  {_excerpt(model.get('content', ''))}\n")

    # Wave 2 ingested but NOT refreshed: every dry run below then previews the
    # same second refresh over the same window, which is what makes repeats
    # comparable.
    await _retain_many(memory, bank_id, ctx, waves[1], "world")

    expected = question.answer_criteria

    for attempt in range(1, args.repeats + 1):
        recorder = _CallRecorder(memory._reflect_llm_config)
        try:
            dry = await memory.dry_run_refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id, request_context=ctx)
        finally:
            recorder.restore()
        if dry is None:
            console.print(f"[red]dry run {attempt} returned None[/red]")
            continue
        ops = dry.delta_operations
        console.print(f"[bold]dry run {attempt}[/bold]  mode={dry.effective_mode}")
        console.print(f"  window created_after={dry.window.created_after}")
        console.print(f"  facts retrieved={dry.facts.retrieved} used={dry.facts.used}")
        console.print(f"  [cyan]candidate (raw synthesis)[/cyan]: {_excerpt(dry.candidate_content, 300)}")
        if ops:
            console.print(f"  ops applied={len(ops.applied)} skipped={len(ops.skipped)}")
            for op in ops.applied[:6]:
                console.print(f"    - {op.get('op', '?')}: {_excerpt(str(op), 160)}")
        console.print(f"  [magenta]preview (after ops)[/magenta]: {_excerpt(dry.preview_content, 300)}")

        # INPUT / EXPECTED / REAL for the two calls that decide the page.
        for scope in ("reflect", "mental_model_delta_ops"):
            for call in recorder.by_scope(scope):
                console.print(f"\n  [bold]== {scope} ({call['kind']}) ==[/bold]")
                for message in call["messages"]:
                    role = message.get("role", "?")
                    body = " ".join(str(message.get("content", "")).split())
                    # The delta-window warning is what this run is testing, so
                    # show whether it actually reached the model rather than
                    # assuming it did.
                    reached = "incomplete BY DESIGN" in body
                    marker = " [green](delta-window warning present)[/green]" if reached else ""
                    console.print(f"    [dim]{role}[/dim]{marker}: {_excerpt(body, 700)}")
                console.print(f"    [yellow]EXPECTED[/yellow]: {_excerpt(expected, 200)}")
                console.print(f"    [cyan]REAL[/cyan]: {_excerpt(call['output'], 400)}")
        console.print()

    pool = await memory._get_pool()
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
