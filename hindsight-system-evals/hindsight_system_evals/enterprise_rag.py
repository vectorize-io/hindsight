"""EnterpriseRAG-Bench (Onyx): the dataset, the committed slice, and the corpus a run writes.

Upstream: https://github.com/onyx-dot-app/EnterpriseRAG-Bench (MIT), release v1.0.0 —
511,963 synthetic documents of one company ("Redwood Inference") across nine source systems,
and 500 questions in ten categories, each with a gold answer, the facts it must state and the
documents it comes from. Leaderboard:
https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard

**Why a slice.** Ingesting all 512k documents is hours of embedding; a run that needs that
cannot be iterated on. The slice keeps every gold document of the questions it covers and
adds two kinds of distractor, so a hit is still earned:

* *hard negatives*, per question: the BM25 top documents over the whole corpus for the
  question text and for each gold document's title. These are the near-duplicates (the
  benchmark deliberately plants updated or conflicting copies of documents) and the
  same-topic neighbours a real search has to beat;
* *random fill*: a seeded sample of the rest, so the bank is not all on-topic.

The hard negatives need BM25 over all 512k documents, which is minutes of work and GBs of
memory, so ``build`` computes them once into ``fixtures/enterprise-rag-slice.json`` (the
same arrangement as the BEIR slice). A run reads that file and the release zip only.

Questions are ordered so that every prefix is stratified by category: the first 50 hold the
benchmark's mix (17 basic, 13 semantic, 4 intra-document, ... 1 high-level), as do the first
100 and all 500. ``HINDSIGHT_EVAL_ERB_QUESTIONS`` picks the prefix.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import httpx
from pydantic import BaseModel

from hindsight_system_evals import knowledge
from hindsight_system_evals.knowledge import KnowledgeApi
from hindsight_system_evals.knowledge_data import CACHE, TextItem
from hindsight_system_evals.report import KnowledgeIngest

RELEASE = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0"
DATA_DIR = CACHE / "enterprise-rag"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "enterprise-rag-slice.json"
SLICE_SEED = 20261105
#: BM25 neighbours kept per question (question text) and per gold document (its title).
HARD_PER_QUESTION = 20
HARD_PER_GOLD = 10
INGEST_TIMEOUT = float(os.getenv("HINDSIGHT_EVAL_ERB_INGEST_TIMEOUT", str(12 * 3600)))

_DSID = re.compile(r"dsid_[0-9a-f]{32}")


class Question(BaseModel):
    question_id: str
    question_type: str
    source_types: list[str]
    question: str
    expected_doc_ids: list[str]
    gold_answer: str
    answer_facts: list[str]


class SliceQuestion(BaseModel):
    question_id: str
    #: BM25 hard negatives for this question (never its gold documents).
    hard_negatives: list[str]


class Slice(BaseModel):
    release: str
    seed: int
    #: Stratified order: any prefix keeps the benchmark's category mix.
    questions: list[SliceQuestion]


# --- download -----------------------------------------------------------------------------


def _download(name: str) -> Path:
    """A release asset, cached. The corpus zip is 1.26 GB; it is fetched once per machine."""
    path = DATA_DIR / name
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with httpx.stream("GET", f"{RELEASE}/{name}", follow_redirects=True, timeout=600) as response:
        response.raise_for_status()
        with partial.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                out.write(chunk)
    partial.rename(path)
    return path


def load_questions() -> list[Question]:
    with _download("questions.jsonl").open(encoding="utf-8") as f:
        return [Question.model_validate_json(line) for line in f if line.strip()]


# --- the corpus ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    #: Path inside the release, minus the id: ``slack/eng-platform/<slug>.txt``. Written as the
    #: knowledge-bank filename, so a search hit shows where the document lives.
    filename: str
    source: str
    text: str


@cache
def corpus_members() -> dict[str, str]:
    """dsid -> zip member, for every document in the release."""
    with zipfile.ZipFile(_download("all_documents.zip")) as z:
        out = {}
        for name in z.namelist():
            match = _DSID.search(name.rsplit("/", 1)[-1])
            if match and not name.endswith("/"):
                out[match.group(0)] = name
        return out


def _document(z: zipfile.ZipFile, document_id: str, member: str) -> CorpusDocument:
    directory, _, base = member.rpartition("/")
    slug = base.split("__", 1)[-1]
    return CorpusDocument(
        id=document_id,
        filename=f"{directory}/{slug}" if directory else slug,
        source=member.split("/", 1)[0],
        text=z.read(member).decode("utf-8", errors="replace"),
    )


def read_documents(document_ids: Iterable[str]) -> list[CorpusDocument]:
    members = corpus_members()
    with zipfile.ZipFile(_download("all_documents.zip")) as z:
        return [_document(z, d, members[d]) for d in document_ids]


def load_slice() -> Slice:
    return Slice.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Subset:
    questions: list[Question]
    #: Gold, then hard negatives, then random fill; no duplicates.
    document_ids: list[str]


def subset(n_questions: int, n_random: int) -> Subset:
    """The first ``n_questions`` of the slice, with every document they need and their
    distractors. ``n_random`` < 0 means the whole corpus."""
    by_id = {q.question_id: q for q in load_questions()}
    chosen = load_slice().questions[:n_questions]
    questions = [by_id[c.question_id] for c in chosen]
    gold = list(dict.fromkeys(d for q in questions for d in q.expected_doc_ids))
    hard = [d for c in chosen for d in c.hard_negatives]
    members = corpus_members()
    if n_random < 0:
        rest = sorted(members)
    else:
        # A seeded shuffle of every id: the same fill on every machine, and a bigger fill is a
        # superset of a smaller one, so two sizes differ only by what was added.
        pool = sorted(members)
        random.Random(SLICE_SEED).shuffle(pool)
        rest = pool[:n_random]
    return Subset(questions=questions, document_ids=list(dict.fromkeys([*gold, *hard, *rest])))


async def ingest(api: KnowledgeApi, bank_id: str, document_ids: Sequence[str]) -> KnowledgeIngest:
    """Write the documents as text, tagged with their source system, and wait until each is
    ready or failed (``knowledge.ingest``, documents read from the zip one batch at a time).

    Re-writing a document the bank already holds unchanged queues nothing, so pointing a run
    at a bank that already has the corpus (``HINDSIGHT_EVAL_ERB_BANK``) costs one upload pass,
    not a re-embed. The reported time is then the upload, and says so in ``passages_per_s``.
    """

    def load(ids: Sequence[str]) -> list[TextItem]:
        return [TextItem(id=d.id, text=d.text, filename=d.filename, tags=(d.source,)) for d in read_documents(ids)]

    return await knowledge.ingest(api, bank_id, document_ids, load, reuse_bank=True, timeout=INGEST_TIMEOUT)


# --- building the slice -------------------------------------------------------------------


def _stratified(questions: Sequence[Question], seed: int) -> list[Question]:
    """Each category shuffled, then merged by each question's position within its category
    (rank / size): every prefix then holds each category in proportion."""
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    rng = random.Random(seed)
    keyed = []
    for qtype in sorted(by_type):
        group = sorted(by_type[qtype], key=lambda q: q.question_id)
        rng.shuffle(group)
        keyed += [((rank + 0.5) / len(group), qtype, q) for rank, q in enumerate(group)]
    return [q for _, _, q in sorted(keyed, key=lambda k: (k[0], k[1]))]


def build() -> None:
    """Rank every question and gold title against all 512k documents with BM25 and write the
    slice. Minutes and ~10 GB of memory; run on purpose (`python -m ... build`)."""
    import bm25s
    import numpy as np

    questions = load_questions()
    members = corpus_members()
    ids = sorted(members)
    started = time.perf_counter()
    with zipfile.ZipFile(_download("all_documents.zip")) as z:
        texts = [z.read(members[d]).decode("utf-8", errors="replace") for d in ids]
    print(f"read {len(texts)} documents in {time.perf_counter() - started:.0f}s")
    titles = {d: t.split("\n", 1)[0] for d, t in zip(ids, texts, strict=True)}
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=True))
    del texts
    print(f"indexed in {time.perf_counter() - started:.0f}s")

    def top(queries: list[str], k: int) -> list[list[str]]:
        results, _ = retriever.retrieve(bm25s.tokenize(queries, stopwords="en", show_progress=False), k=k)
        return [[ids[int(i)] for i in np.asarray(row)] for row in results]

    per_question = top([q.question for q in questions], HARD_PER_QUESTION + 10)
    gold_titles = sorted({d for q in questions for d in q.expected_doc_ids})
    per_gold = dict(zip(gold_titles, top([titles[d] for d in gold_titles], HARD_PER_GOLD + 10), strict=True))
    rows = []
    by_id = {q.question_id: hits for q, hits in zip(questions, per_question, strict=True)}
    for q in _stratified(questions, SLICE_SEED):
        gold = set(q.expected_doc_ids)
        hard = [d for d in by_id[q.question_id] if d not in gold][:HARD_PER_QUESTION]
        for g in q.expected_doc_ids:
            hard += [d for d in per_gold[g] if d not in gold and d not in hard][:HARD_PER_GOLD]
        rows.append(SliceQuestion(question_id=q.question_id, hard_negatives=hard))
    FIXTURE.write_text(
        Slice(release="v1.0.0", seed=SLICE_SEED, questions=rows).model_dump_json(indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {FIXTURE} in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=["build", "download"])
    args = parser.parse_args()
    if args.command == "download":
        for name in ("questions.jsonl", "all_documents.zip"):
            print(_download(name))
    else:
        build()


if __name__ == "__main__":
    main()
