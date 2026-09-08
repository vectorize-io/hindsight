"""Deterministic corpus loader for the prelude retrieval eval.

The corpus is authored, not extracted. Every stored row's text is byte-identical
to what ``corpus/*.yaml`` says, which is what makes gold labelling possible at
all: an eval whose corpus is produced by a real LLM has no stable ids to label
against and no way to tell a retrieval regression from an extraction one.

Three mechanisms carry the whole fixture:

* **Facts and observations** are retained through the normal pipeline with the
  ``mock`` LLM provider, scripted via ``set_response_callback`` so extraction
  returns exactly one fact carrying our text and our ``fact_type``. Observations
  are ordinary ``memory_units`` rows with ``fact_type='observation'`` (that is
  precisely what ``tool_search_observations`` queries), so they are seeded the
  same way rather than by running consolidation — consolidation would generate
  its own text and there would be nothing to label.
* **Mental models** are created with ``create_mental_model`` and its explicit
  content.
* **Staleness is ordering, not a field.** ``tool_search_mental_models`` derives
  ``is_stale`` from whether in-scope memories arrived since the model's last
  refresh, so a mental model marked ``stale: true`` is created BEFORE the facts
  are retained and a fresh one AFTER.

Embeddings come from the pinned local model, so a given corpus always produces
the same vectors and retrieval is reproducible end to end.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table
from hindsight_api.models import RequestContext

from benchmarks.prelude import hard_corpus
from benchmarks.prelude.ballast import generate as ballast_rows

CORPUS_DIR = Path(__file__).parent / "corpus"

# Batch size for the filler rows. One batch is one embedding round-trip, but the
# whole batch also lands in one transaction, so this trades round-trips against
# transaction size rather than being a pure "bigger is better".
_BALLAST_BATCH = 100


@dataclass(frozen=True)
class Question:
    """One labelled question from ``corpus/questions.yaml``."""

    id: str
    category: str
    question: str
    ideal_query: str
    gold: frozenset[str]
    short_circuit: bool
    # Answer-level labels. Retrieval metrics prove the evidence reached the
    # model; only these say whether the prose it wrote was right.
    answer_criteria: str = ""
    must_not_claim: str = ""


@dataclass
class Corpus:
    """The loaded corpus plus the mapping from authored id to stored row id."""

    questions: list[Question]
    # authored id (fact-deploy-001, obs-team, mm-billing) -> id of the row it became
    stored_ids: dict[str, str] = field(default_factory=dict)
    bank_id: str = ""
    # Stored ids of the filler rows, so contamination can be reported by name.
    ballast_ids: set[str] = field(default_factory=set)

    def gold_row_ids(self, question: Question) -> set[str]:
        """Gold ids translated into the ids retrieval will actually return."""
        return {self.stored_ids[a] for a in question.gold if a in self.stored_ids}


def load_questions() -> list[Question]:
    raw = yaml.safe_load((CORPUS_DIR / "questions.yaml").read_text(encoding="utf-8"))
    return [
        Question(
            id=q["id"],
            category=q["category"],
            question=q["question"],
            ideal_query=q["ideal_query"],
            gold=frozenset(q.get("gold") or []),
            short_circuit=bool(q.get("short_circuit", False)),
            answer_criteria=(q.get("answer_criteria") or "").strip(),
            must_not_claim=(q.get("must_not_claim") or "").strip(),
        )
        for q in raw["questions"]
    ]


def _scripted_extraction(text: str, fact_type: str):
    """A ``set_response_callback`` that makes retain store exactly ``text``.

    Retain's fact extraction is the only LLM call this fixture makes; returning a
    single fact keeps one document equal to one row, so the authored id maps to
    exactly one stored id.
    """

    def _callback(messages: list[dict], scope: str) -> Any:
        if scope == "retain_extract_facts":
            return {
                "facts": [
                    {
                        "what": text,
                        "when": "N/A",
                        "where": "N/A",
                        "who": "N/A",
                        "why": "N/A",
                        "fact_kind": "conversation",
                        "fact_type": fact_type,
                        "entities": [],
                    }
                ]
            }
        return None

    return _callback


def _batch_extraction(by_text: dict[str, str]):
    """A callback for a BATCH retain: one fact per content, text preserved.

    A batch fires one extraction call per content item, and the callback only
    sees the assembled prompt, not the bare content. The prompt embeds the
    content verbatim, so the row is identified by finding which of the batch's
    texts it contains — longest first, so a text that is a prefix of another
    cannot win. Ballast sentences are unique by construction (``ballast.generate``
    dedupes), so the match is unambiguous.
    """
    ordered = sorted(by_text, key=len, reverse=True)

    def _callback(messages: list[dict], scope: str) -> Any:
        if scope != "retain_extract_facts":
            return None
        prompt = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        text = next((t for t in ordered if t in prompt), None)
        if text is None:
            raise RuntimeError("Batch extraction saw a prompt carrying none of the batch's authored texts")
        return {
            "facts": [
                {
                    "what": text,
                    "when": "N/A",
                    "where": "N/A",
                    "who": "N/A",
                    "why": "N/A",
                    "fact_kind": "conversation",
                    "fact_type": by_text[text],
                    "entities": [],
                }
            ]
        }

    return _callback


async def _retain_many(
    memory: MemoryEngine, bank_id: str, ctx: RequestContext, rows: list[tuple[str, str]], fact_type: str
) -> dict[str, str]:
    """Retain many authored rows in ONE batch; return authored id -> stored id.

    Sequential ``retain_async`` costs one embedding round-trip per row, which is
    minutes at corpus scale. ``retain_batch_async`` embeds the whole batch in one
    call.
    """
    impl = _require_mock(memory)
    by_text = {text: fact_type for _, text in rows}
    impl.set_response_callback(_batch_extraction(by_text))
    try:
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": text} for _, text in rows],
            request_context=ctx,
        )
    finally:
        impl.clear_mock_calls()

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        found = await conn.fetch(
            f"SELECT id, text FROM {fq_table('memory_units')} WHERE bank_id = $1 AND text = ANY($2::text[])",
            bank_id,
            [text for _, text in rows],
        )
    by_stored = {r["text"]: str(r["id"]) for r in found}
    missing = [aid for aid, text in rows if text not in by_stored]
    if missing:
        raise RuntimeError(f"Batch retain stored {len(by_stored)}/{len(rows)} rows; missing {missing[:5]}")
    return {aid: by_stored[text] for aid, text in rows}


def _mock_impl(memory: MemoryEngine):
    """The MockLLM behind the retain LLM config, or None if not using the mock provider."""
    return getattr(memory._retain_llm_config, "_provider_impl", None)


def _require_mock(memory: MemoryEngine):
    """The MockLLM behind retain, or a clear error explaining why it is required."""
    impl = _mock_impl(memory)
    if impl is None or not hasattr(impl, "set_response_callback"):
        raise RuntimeError(
            "The prelude fixture requires the mock LLM provider for RETAIN "
            "(memory_llm_provider='mock'); corpus text must be authored, not extracted."
        )
    return impl


async def _retain_one(memory: MemoryEngine, bank_id: str, ctx: RequestContext, text: str, fact_type: str) -> str:
    """Retain one authored row and return the id it was stored under."""
    impl = _require_mock(memory)
    impl.set_response_callback(_scripted_extraction(text, fact_type))
    try:
        await memory.retain_async(bank_id=bank_id, content=text, request_context=ctx)
    finally:
        impl.clear_mock_calls()  # also drops the scripted callback

    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT id FROM {fq_table('memory_units')} WHERE bank_id = $1 AND text = $2 "
            "ORDER BY created_at DESC LIMIT 1",
            bank_id,
            text,
        )
    if row is None:
        raise RuntimeError(f"Retain did not store the authored row: {text!r}")
    return str(row["id"])


async def build_hard_corpus(memory: MemoryEngine, *, bank_id: str | None = None) -> Corpus:
    """Build the HARD corpus: dense same-topic near-misses, no inert filler.

    Every row here is in a question's own topic and vocabulary, so nothing is
    separable by similarity alone — unlike the ballast corpus, whose filler came
    from unrelated domains and turned out to be inert.
    """
    bank_id = bank_id or f"prelude-hard-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext()
    await memory.get_bank_profile(bank_id=bank_id, request_context=ctx)
    await memory._config_resolver.update_bank_config(bank_id, {"enable_auto_consolidation": False}, ctx)

    facts, questions = hard_corpus.build()
    corpus = Corpus(
        questions=[
            Question(
                id=q.id,
                category=q.category,
                question=q.question,
                ideal_query=q.ideal_query,
                gold=frozenset(q.gold),
                short_circuit=q.short_circuit,
                answer_criteria=q.answer_criteria,
                must_not_claim=q.must_not_claim,
            )
            for q in questions
        ],
        bank_id=bank_id,
    )

    rows = [(f.id, f.text) for f in facts]
    for start in range(0, len(rows), _BALLAST_BATCH):
        corpus.stored_ids.update(
            await _retain_many(memory, bank_id, ctx, rows[start : start + _BALLAST_BATCH], "world")
        )
    return corpus


async def build_corpus(memory: MemoryEngine, *, bank_id: str | None = None, ballast: int = 0) -> Corpus:
    """Build the fixture bank and return the corpus with its id mapping.

    Ordering is load-bearing: stale mental models must exist before the facts
    that make them stale, and fresh ones must be created after.
    """
    bank_id = bank_id or f"prelude-eval-{uuid.uuid4().hex[:8]}"
    ctx = RequestContext()
    await memory.get_bank_profile(bank_id=bank_id, request_context=ctx)
    # Consolidation would add rows this corpus never labelled, so keep it off.
    await memory._config_resolver.update_bank_config(bank_id, {"enable_auto_consolidation": False}, ctx)

    facts = yaml.safe_load((CORPUS_DIR / "facts.yaml").read_text(encoding="utf-8"))["facts"]
    layers = yaml.safe_load((CORPUS_DIR / "layers.yaml").read_text(encoding="utf-8"))
    corpus = Corpus(questions=load_questions(), bank_id=bank_id)

    # 1. Stale mental models first: everything retained below post-dates them.
    for mm in layers.get("mental_models") or []:
        if mm.get("stale"):
            created = await memory.create_mental_model(
                bank_id=bank_id,
                name=mm["name"],
                source_query=mm["source_query"],
                content=mm["content"].strip(),
                request_context=ctx,
            )
            corpus.stored_ids[mm["id"]] = str(created["id"])

    # 2. Raw facts, then observations.
    for entry in facts:
        corpus.stored_ids[entry["id"]] = await _retain_one(memory, bank_id, ctx, entry["text"], "world")
    for obs in layers.get("observations") or []:
        text = " ".join(obs["text"].split())
        corpus.stored_ids[obs["id"]] = await _retain_one(memory, bank_id, ctx, text, "observation")

    # 2b. Ballast. Irrelevant filler that makes the bank big enough for the recall
    # token budget to actually truncate — at ~30 rows every query returns
    # everything, so a bad query can only reorder evidence, never lose it.
    if ballast:
        rows = ballast_rows(ballast)
        for start in range(0, len(rows), _BALLAST_BATCH):
            chunk = rows[start : start + _BALLAST_BATCH]
            corpus.stored_ids.update(await _retain_many(memory, bank_id, ctx, chunk, "world"))
        corpus.ballast_ids = {corpus.stored_ids[aid] for aid, _ in rows}

    # 3. Fresh mental models last, so no in-scope memory post-dates them.
    #
    # Creating one is NOT enough to make it fresh. Staleness is resolved from the
    # refresh stamps (``last_refreshed_at`` / ``last_memory_seen_at``), and
    # compute_mental_models_are_stale reports a model with no stamp as stale
    # unconditionally — "a model no refresh has stamped". So a created-but-never-
    # refreshed model is always stale and can never short-circuit the forced
    # descent. The fixture stamps them directly rather than running a refresh,
    # which would regenerate the content this corpus depends on being exact.
    fresh_ids: list[str] = []
    for mm in layers.get("mental_models") or []:
        if not mm.get("stale"):
            created = await memory.create_mental_model(
                bank_id=bank_id,
                name=mm["name"],
                source_query=mm["source_query"],
                content=" ".join(mm["content"].split()),
                request_context=ctx,
            )
            corpus.stored_ids[mm["id"]] = str(created["id"])
            fresh_ids.append(str(created["id"]))

    if fresh_ids:
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE {fq_table('mental_models')} "
                "SET last_refreshed_at = NOW(), last_memory_seen_at = NOW() "
                "WHERE bank_id = $1 AND id = ANY($2::text[])",
                bank_id,
                fresh_ids,
            )

    return corpus
