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

CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass(frozen=True)
class Question:
    """One labelled question from ``corpus/questions.yaml``."""

    id: str
    category: str
    question: str
    ideal_query: str
    gold: frozenset[str]
    short_circuit: bool


@dataclass
class Corpus:
    """The loaded corpus plus the mapping from authored id to stored row id."""

    questions: list[Question]
    # authored id (fact-deploy-001, obs-team, mm-billing) -> id of the row it became
    stored_ids: dict[str, str] = field(default_factory=dict)
    bank_id: str = ""

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


def _mock_impl(memory: MemoryEngine):
    """The MockLLM behind the retain LLM config, or None if not using the mock provider."""
    return getattr(memory._retain_llm_config, "_provider_impl", None)


async def _retain_one(memory: MemoryEngine, bank_id: str, ctx: RequestContext, text: str, fact_type: str) -> str:
    """Retain one authored row and return the id it was stored under."""
    impl = _mock_impl(memory)
    if impl is None or not hasattr(impl, "set_response_callback"):
        raise RuntimeError(
            "The prelude fixture requires the mock LLM provider "
            "(HINDSIGHT_API_LLM_PROVIDER=mock); corpus text must be authored, not extracted."
        )
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


async def build_corpus(memory: MemoryEngine, *, bank_id: str | None = None) -> Corpus:
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
