"""The knowledge-bank retrieval cases: one category per kind of content, each on a dataset with
a public leaderboard, so a score sits next to the best published models.

| category | dataset | leaderboard metric |
|---|---|---|
| ``text`` | BEIR SciFact, FiQA-2018 | nDCG@10 (MTEB) |
| ``pdf`` | ViDoRe V3, 8 public tasks | nDCG@10 (ViDoRe / MTEB) |
| ``image`` | Flickr30k 1k and MS-COCO 5k test, text to image | recall@1/5/10 |
| ``video`` | MSR-VTT 1k-A and MSVD test, text to video | nDCG@10, recall@1 |
| ``research`` | BRIGHT-Pro (7 domains), BrowseComp-Plus | alpha-nDCG@25; nDCG@10, recall@100 |

``research`` ranks retrieval *systems* (rewriting, hybrid, reranking are fair), unlike the
leaderboards above, which rank bare embedding models.

A case writes its corpus into a knowledge bank (once per session), runs its queries through
search and returns its scores. A sixth category, agents answering from a knowledge bank, is
``evals/knowledge_agent`` (EnterpriseRAG-Bench, judged).

Sizes are small by default and full with ``--full``: the corpus is always whole (a smaller pool
makes every query easier), only the queries are cut. ``--full`` is the leaderboard number.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from hindsight_system_evals import knowledge_data as data
from hindsight_system_evals.knowledge import (
    RANKED_TOP_K,
    KnowledgeApi,
    SearchCall,
    Searched,
    failed_documents,
    ingest,
    latency,
    preloaded,
    ranked_documents,
    score_ranked,
    search_all,
)
from hindsight_system_evals.knowledge_data import FileItem, MediaType, TextItem
from hindsight_system_evals.report import KnowledgeIngest, KnowledgeResult

BANK_PREFIX = "syseval-kb-"


@dataclass(frozen=True)
class Sizes:
    vidore_queries_per_language: int | None
    beir_queries: int | None
    image_captions_per_image: int | None
    video_captions_per_video: int | None
    bright_pro_queries_per_domain: int | None
    browsecomp_queries: int | None


#: BRIGHT-Pro's 25 per domain is the sample its paper runs agents on.
DEFAULT = Sizes(17, 100, 1, 1, 25, 100)
FULL = Sizes(None, None, None, None, None, None)


@dataclass
class Bank:
    bank_id: str
    ingest: KnowledgeIngest


@dataclass
class Context:
    client: KnowledgeApi
    full: bool
    embedder: str
    #: Knowledge banks by corpus, filled on first use and shared by every case on that corpus.
    banks: dict[str, Bank]

    @property
    def sizes(self) -> Sizes:
        return FULL if self.full else DEFAULT

    async def bank(self, key: str, items: Sequence[FileItem | TextItem]) -> Bank:
        if key not in self.banks:
            bank_id = f"{BANK_PREFIX}{key.replace('_', '-')}-{uuid.uuid4().hex[:6]}"
            stats = await ingest(self.client, bank_id, [i.id for i in items], preloaded(items))
            bank = Bank(bank_id, stats)
            self.banks[key] = bank
            if stats.failed:
                raise AssertionError(
                    f"{bank_id}: {stats.failed} documents failed: {await failed_documents(self.client, bank_id)}"
                )
        return self.banks[key]


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    suite: Callable[[Sizes], data.RankedSuite]
    #: The media type the server's embedder must write, or the case is skipped.
    needs: MediaType


def _runs(searched: Sequence[Searched], page_ids: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    runs = {s.query_id: ranked_documents(s.results, page_ids) for s in searched}
    empty = [q for q, docs in runs.items() if not docs]
    assert not empty, f"{len(empty)} queries returned nothing, e.g. {empty[:5]}"
    return runs


async def run_case(ctx: Context, case: Case) -> KnowledgeResult:
    """Write the case's corpus (once per corpus), search every query, score."""
    suite = case.suite(ctx.sizes)
    bank = await ctx.bank(suite.name, suite.items)
    calls = [SearchCall(q.id, q.text, RANKED_TOP_K) for q in suite.queries]
    searched = await search_all(ctx.client, bank.bank_id, calls)
    return KnowledgeResult(
        category=case.category,
        case=case.id,
        embedder=ctx.embedder,
        corpus=len(suite.items),
        queries=len(suite.queries),
        metrics=score_ranked(suite.queries, _runs(searched, suite.page_ids)),
        ingest=bank.ingest,
        latency=latency(searched),
    )


def _beir(name: str) -> Callable[[Sizes], data.RankedSuite]:
    return lambda sizes: data.beir(name, queries=sizes.beir_queries)


def _vidore(task: str) -> Callable[[Sizes], data.RankedSuite]:
    """Each source PDF uploaded whole (the judged pages kept), every query language."""
    return lambda sizes: data.vidore_documents(task, queries_per_language=sizes.vidore_queries_per_language)


def _images(name: str) -> Callable[[Sizes], data.RankedSuite]:
    return lambda sizes: data.captioned_images(name, captions_per_image=sizes.image_captions_per_image)


def _videos(name: str) -> Callable[[Sizes], data.RankedSuite]:
    return lambda sizes: data.captioned_videos(name, captions_per_video=sizes.video_captions_per_video)


def _bright_pro(domain: str) -> Callable[[Sizes], data.RankedSuite]:
    return lambda sizes: data.bright_pro(domain, queries=sizes.bright_pro_queries_per_domain)


CASES: list[Case] = [
    Case("scifact", "text", _beir("scifact"), "text"),
    Case("fiqa", "text", _beir("fiqa"), "text"),
    *(Case(t, "pdf", _vidore(t), "document") for t in data.VIDORE_TASKS),
    *(Case(name.split("_")[0], "image", _images(name), "image") for name in data.IMAGE_SETS),
    *(Case(name.split("_")[0], "video", _videos(name), "video") for name in data.VIDEO_SETS),
    *(Case(f"brightpro_{d}", "research", _bright_pro(d), "text") for d in data.BRIGHT_PRO_DOMAINS),
    Case("browsecomp_plus", "research", lambda sizes: data.browsecomp_plus(queries=sizes.browsecomp_queries), "text"),
]

#: What a default run covers: one case per category.
MINIMUM = {"scifact", "hr", "flickr30k", "msrvtt", "brightpro_biology"}
