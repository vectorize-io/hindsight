"""A committed slice of a BEIR dataset, and the script that cuts it.

BEIR is the external anchor this repo had none of: real queries, real documents,
and relevance labels nobody here wrote. Everything else in `system-evals` grades
against gold facts we authored, which measures whether Hindsight agrees with us.

The slice is **committed**, not downloaded at test time, for the reasons the
refresh-cost fixture is frozen: a run must mean the same thing on every machine
and on a box with no internet, and a dataset that resized itself would make every
earlier number incomparable. `build` below re-cuts it on purpose.

Why a slice at all: SciFact is 5,183 abstracts, and this eval retains each one as
its own document with real extraction — one model call per document. The full
corpus is a bill, not a signal. The slice keeps every gold document for the
queries it covers and fills the rest with distractors, so a hit is still earned.

Full-corpus runs stay possible: point `HINDSIGHT_EVAL_BEIR_DIR` at an unpacked
BEIR dataset directory (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) and
nothing else changes. That is also how another dataset gets measured — NFCorpus
and TREC-COVID have graded qrels, which `ndcg_at_k` already handles.
"""

from __future__ import annotations

import json
import os
import random
from typing import TypeVar
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "beir-scifact-slice.json"

#: Upstream SciFact. Cutting the slice is the only thing that reads it.
SOURCE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

#: Slice size. The two numbers are chosen against different costs, which is why
#: they are not the same number:
#:
#: * **documents cost money** — one fact-extraction call each, at retain. 500 means
#:   a gold abstract has to beat ~475 distractors. It was 200 first, which turned
#:   out to be too easy to measure anything by: nDCG@10 read 0.933, well above
#:   SciFact's published ~0.70 over the full 5,183, and a benchmark that flatters
#:   is a floor that never trips.
#: * **queries cost nothing** — one `recall` each, no model call. So there is no
#:   reason to be stingy: at 25 queries a single one moves the macro-average by
#:   0.04, which is wider than any regression worth catching.
SLICE_QUERIES = 50
SLICE_DOCUMENTS = 500

#: Fixed, because the distractors have to be the same 175 abstracts every run —
#: a different filler set is a different benchmark.
SLICE_SEED = 20261001


@dataclass(frozen=True)
class BeirDocument:
    id: str
    title: str
    text: str

    @property
    def content(self) -> str:
        """What gets retained. Title first, as BEIR's own baselines concatenate it."""
        return f"{self.title}\n\n{self.text}".strip()


@dataclass(frozen=True)
class BeirQuery:
    id: str
    text: str
    #: document id -> relevance grade, straight from the qrels. Never empty: a
    #: query with no labelled document cannot be scored and is dropped at load.
    relevant: dict[str, int]


@dataclass(frozen=True)
class BeirSlice:
    name: str
    documents: tuple[BeirDocument, ...]
    queries: tuple[BeirQuery, ...]

    def describe(self) -> str:
        gold = len({d for q in self.queries for d in q.relevant})
        return f"{self.name}: {len(self.queries)} queries, {len(self.documents)} documents ({gold} gold)"


def load_beir_slice() -> BeirSlice:
    """The slice this eval scores: the committed fixture, or a full BEIR directory.

    `HINDSIGHT_EVAL_BEIR_QUERIES=n` keeps only the first n queries *and* drops
    nothing from the corpus — a smoke run must still face the same distractors,
    or it reports a number the real run cannot reproduce.
    """
    if directory := os.getenv("HINDSIGHT_EVAL_BEIR_DIR"):
        sliced = _read_beir_directory(Path(directory))
    else:
        sliced = _read_fixture(FIXTURE)

    if limit := os.getenv("HINDSIGHT_EVAL_BEIR_QUERIES"):
        sliced = BeirSlice(name=sliced.name, documents=sliced.documents, queries=sliced.queries[: int(limit)])
    return sliced


def _read_fixture(path: Path) -> BeirSlice:
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing. Cut it with: uv run python -m hindsight_system_evals.beir build"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BeirSlice(
        name=payload["name"],
        documents=tuple(BeirDocument(**doc) for doc in payload["documents"]),
        queries=tuple(
            BeirQuery(id=q["id"], text=q["text"], relevant={k: int(v) for k, v in q["relevant"].items()})
            for q in payload["queries"]
        ),
    )


#: Python 3.11 is the baseline interpreter, so no PEP 695 `def f[T: Base](...)`.
RowT = TypeVar("RowT", bound=BaseModel)


class _BeirRow(BaseModel):
    """Shared config for BEIR's jsonl rows.

    `_id` arrives as a string in some datasets and a bare int in others, and every
    lookup downstream is keyed on it — a silent `4983` vs `"4983"` mismatch would
    score every query at zero while looking like a retrieval collapse. Coercing it
    here rather than at each use site means there is one place it can be wrong.
    """

    model_config = ConfigDict(populate_by_name=True, coerce_numbers_to_str=True, extra="ignore")


class _CorpusRow(_BeirRow):
    """One line of BEIR's `corpus.jsonl`. `_id` is BEIR's spelling."""

    id: str = Field(alias="_id")
    title: str = ""
    text: str = ""


class _QueryRow(_BeirRow):
    """One line of BEIR's `queries.jsonl`. Its `metadata` is dataset-specific and unused."""

    id: str = Field(alias="_id")
    text: str


def _read_jsonl(path: Path, row: type[RowT]) -> list[RowT]:
    with path.open(encoding="utf-8") as handle:
        return [row.model_validate_json(line) for line in handle if line.strip()]


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """BEIR qrels are a TSV with a header row. A grade of 0 is an explicit non-match."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle, None)  # header: query-id, corpus-id, score
        for line in handle:
            if not line.strip():
                continue
            query_id, document_id, score = line.rstrip("\n").split("\t")[:3]
            if int(score) > 0:
                qrels.setdefault(query_id, {})[document_id] = int(score)
    return qrels


def _query_sort_key(query_id: str) -> tuple[int, int, str]:
    """Order query ids reproducibly whether they are numbers or not.

    SciFact numbers its queries ("1", "3", "5") and sorting those as strings puts
    "10" before "2", which would make `cut_slice` take a different 50 queries than
    anyone reading "the first 50" expects. NFCorpus names them ("PLAIN-3",
    "MED-118"), where `int()` simply raises — which it did, on the first non-SciFact
    dataset this loader ever saw.
    """
    return (0, int(query_id), "") if query_id.isdigit() else (1, 0, query_id)


def _read_beir_directory(directory: Path, *, split: str = "test") -> BeirSlice:
    """A whole unpacked BEIR dataset, unsliced — for a full-corpus run."""
    qrels = _read_qrels(directory / "qrels" / f"{split}.tsv")
    corpus = {row.id: row for row in _read_jsonl(directory / "corpus.jsonl", _CorpusRow)}
    queries = {row.id: row for row in _read_jsonl(directory / "queries.jsonl", _QueryRow)}
    return BeirSlice(
        name=directory.name,
        documents=tuple(
            BeirDocument(id=row.id, title=row.title, text=row.text) for _, row in sorted(corpus.items())
        ),
        queries=tuple(
            # Gold documents absent from the corpus are dropped: they would hold a
            # perfect run below 1.0 for a reason that is not Hindsight's. BEIR's own
            # corpora are complete, so this only bites a hand-trimmed directory.
            BeirQuery(
                id=query_id,
                text=queries[query_id].text,
                relevant={d: g for d, g in gold.items() if d in corpus},
            )
            for query_id, gold in sorted(qrels.items(), key=lambda item: _query_sort_key(item[0]))
            if query_id in queries and any(d in corpus for d in gold)
        ),
    )


def cut_slice(full: BeirSlice, *, queries: int = SLICE_QUERIES, documents: int = SLICE_DOCUMENTS) -> BeirSlice:
    """Take the first `queries` labelled queries, every gold document, then distractors.

    "First" by numeric query id rather than at random: the selection has to be
    reproducible from the dataset alone, so anyone can check the slice was not
    cherry-picked to flatter a score.
    """
    kept = full.queries[:queries]
    gold_ids = {document_id for query in kept for document_id in query.relevant}

    by_id = {doc.id: doc for doc in full.documents}
    missing = sorted(gold_ids - set(by_id))
    if missing:
        raise RuntimeError(f"gold documents absent from the corpus: {missing}")

    distractor_pool = sorted(set(by_id) - gold_ids)
    random.Random(SLICE_SEED).shuffle(distractor_pool)
    chosen = sorted(gold_ids) + distractor_pool[: max(0, documents - len(gold_ids))]
    return BeirSlice(
        name=full.name,
        documents=tuple(by_id[document_id] for document_id in sorted(chosen)),
        queries=kept,
    )


def _write(sliced: BeirSlice, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": sliced.name,
        "source": SOURCE_URL,
        "queries": [{"id": q.id, "text": q.text, "relevant": q.relevant} for q in sliced.queries],
        "documents": [{"id": d.id, "title": d.title, "text": d.text} for d in sliced.documents],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build(output: Path = FIXTURE) -> BeirSlice:
    """Download SciFact, cut the slice, write the fixture. Run on purpose only."""
    import io
    import tempfile
    import urllib.request
    import zipfile

    with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:  # noqa: S310 - fixed https URL above
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with tempfile.TemporaryDirectory() as scratch:
        archive.extractall(scratch)
        root = next(path for path in Path(scratch).iterdir() if (path / "corpus.jsonl").exists())
        sliced = cut_slice(_read_beir_directory(root))
    _write(sliced, output)
    return sliced


def main() -> None:
    import sys

    if sys.argv[1:2] != ["build"]:
        print("usage: python -m hindsight_system_evals.beir build")
        raise SystemExit(2)
    sliced = build()
    print(f"wrote {FIXTURE} — {sliced.describe()}")


if __name__ == "__main__":
    main()
