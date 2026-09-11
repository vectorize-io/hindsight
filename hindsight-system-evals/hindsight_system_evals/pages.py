"""Build a knowledge page the way a caller does, and report what it converged to.

The scenario, end to end through the public API:

    create page (source_query = the question)
      -> retain wave 1 -> wait for the refresh -> snapshot
      -> retain wave 2 -> wait for the refresh -> snapshot
      -> grade the FINAL page

The waves are the point. A page is not written once; it accumulates, and each
delta refresh edits what is already stored. That is where a wrong answer stops
being a wrong answer and becomes a wrong *memory* — every later reflect reads it
back as fact. A single-wave build would never exercise a delta edit at all.

Waves never split a subject. Facts about one release travelling in different
waves make the later batch read as a correction of the earlier one, and a refresh
that supersedes on that basis is following its rules correctly — the corpus would
be the bug. ``split_into_waves`` keeps a subject together.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from hindsight_client import Hindsight
from hindsight_client_api.models.create_knowledge_page_response import CreateKnowledgePageResponse

from hindsight_system_evals import corpus as corpus_module
from hindsight_system_evals.corpus import HardFact, HardQuestion

#: Waits until a bank has no pending work — the ``settled`` fixture.
SettleFn = Callable[[str], Awaitable[None]]

#: Two is the minimum that exercises a delta at all: the first wave writes the
#: page, the second has to edit it.
WAVES = 2


@dataclass
class WaveSnapshot:
    wave: int
    facts_ingested: int
    content: str = ""

    @property
    def chars(self) -> int:
        return len(self.content)


@dataclass
class PageOutcome:
    """One page, built across every wave, with the evidence to explain a failure."""

    question_id: str
    category: str
    bank_id: str
    page_id: str = ""
    mental_model_id: str = ""
    waves: list[WaveSnapshot] = field(default_factory=list)
    final_content: str = ""
    correct: bool = False
    correct_reason: str = ""
    hit_trap: bool = False
    trap_reason: str = ""


def split_into_waves(facts: list[HardFact], count: int = WAVES) -> list[list[HardFact]]:
    """Split the corpus into waves without separating any subject's facts."""
    by_subject: dict[str, list[HardFact]] = {}
    for index, fact in enumerate(facts):
        # A fact with no subject cannot contradict anything, so it is its own group.
        key = f"{fact.cluster}:{fact.subject}" if fact.subject else f"_solo:{index}"
        by_subject.setdefault(key, []).append(fact)

    waves: list[list[HardFact]] = [[] for _ in range(count)]
    for position, key in enumerate(sorted(by_subject)):
        waves[position % count].extend(by_subject[key])
    return waves


def questions() -> list[HardQuestion]:
    return corpus_module.build().questions


def facts() -> list[HardFact]:
    return corpus_module.build().facts


async def prepare_bank(client: Hindsight, bank_id: str) -> None:
    """Configure the bank so seeding the corpus costs no model calls.

    Measured on the first blackbox run: of 269s wall time per page, 789s of LLM
    time went to fact extraction (one call per one-sentence fact) and 199s to
    consolidation, against 21s for the reflect and delta calls actually being
    evaluated. Neither is under test here:

    * ``chunks`` stores each item as written, with no extraction call. That also
      removes a confound: extraction may paraphrase a fact, and the gold labels
      point at the exact authored text.
    * consolidation and observations produce rows the page never reads — its
      trigger reads raw ``world``/``experience`` facts.

    This is ordinary per-bank configuration through the public config endpoint,
    so the eval is still blackbox.
    """
    await client.aupdate_bank_config(
        bank_id,
        retain_extraction_mode="chunks",
        enable_observations=False,
        enable_auto_consolidation=False,
    )


async def create_page(client: Hindsight, bank_id: str, question: HardQuestion) -> CreateKnowledgePageResponse:
    """Create the page for one question, reading raw facts and refreshed explicitly."""
    return await client.knowledge_base.create_knowledge_page(
        bank_id,
        {
            "name": f"eval {question.id}",
            "source_query": question.question,
            # Pages default to observation-only, which would make the eval depend
            # on consolidation writing usable observations first. Reading raw
            # facts keeps the corpus text exactly as authored, which is what the
            # gold labels point at.
            "trigger": {
                "mode": "delta",
                "fact_types": ["world", "experience"],
                "exclude_mental_models": True,
                # Refreshed explicitly below instead of after consolidation, because
                # consolidation is switched off for this bank — see prepare_bank.
                "refresh_after_consolidation": False,
            },
        },
    )


async def build_page(
    client: Hindsight,
    bank_id: str,
    question: HardQuestion,
    waves: list[list[HardFact]],
    settle: SettleFn,
) -> PageOutcome:
    """Create the page, feed it the corpus wave by wave, and snapshot each time."""
    outcome = PageOutcome(question_id=question.id, category=question.category, bank_id=bank_id)
    await prepare_bank(client, bank_id)

    created = await create_page(client, bank_id, question)
    outcome.page_id = created.page_id
    outcome.mental_model_id = created.mental_model_id
    await settle(bank_id)

    for index, wave in enumerate(waves, start=1):
        await client.aretain_batch(bank_id=bank_id, items=[{"content": fact.text} for fact in wave])
        await settle(bank_id)
        # The refresh is the thing under test, so it goes through the same public
        # endpoint and the same worker as any caller's — only the trigger is
        # explicit rather than consolidation-driven.
        await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=outcome.mental_model_id)
        await settle(bank_id)
        content = await read_page(client, bank_id, outcome.mental_model_id)
        outcome.waves.append(WaveSnapshot(wave=index, facts_ingested=len(wave), content=content))

    outcome.final_content = outcome.waves[-1].content if outcome.waves else ""
    return outcome


async def read_page(client: Hindsight, bank_id: str, mental_model_id: str) -> str:
    """The page's stored content, read through the public API.

    A page is a tree node backed by a mental model; the content lives on the
    model, so that is what gets read. Still blackbox — this is the same endpoint
    any caller uses.
    """
    model = await client.aget_mental_model(bank_id=bank_id, mental_model_id=mental_model_id)
    return getattr(model, "content", "") or ""
