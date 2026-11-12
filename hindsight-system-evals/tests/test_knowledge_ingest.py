"""Writing a corpus into a knowledge bank, without a server: what the loaders hand the API and
how ingest waits, batches and retries."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from hindsight_system_evals import knowledge, knowledge_data
from hindsight_system_evals.knowledge import KnowledgeApi, ingest, preloaded
from hindsight_system_evals.knowledge_data import TextItem


def test_beir_leaves_out_documents_with_no_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "beir" / "fiqa"
    data = root / "fiqa"
    (data / "qrels").mkdir(parents=True)
    (root / ".complete").touch()
    rows = [{"_id": "1", "title": "", "text": "an answer"}, {"_id": "2", "title": "", "text": "  "}]
    (data / "corpus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data / "queries.jsonl").write_text(json.dumps({"_id": "q", "text": "question"}) + "\n")
    (data / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq\t1\t1\n")
    monkeypatch.setattr(knowledge_data, "CACHE", tmp_path)

    suite = knowledge_data.beir("fiqa", queries=None)

    assert [i.id for i in suite.items] == ["fiqa-1"]


async def test_ingest_batches_loads_and_writes_failed_documents_once_more(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knowledge.asyncio, "sleep", _no_sleep)
    uploads: list[list[str]] = []
    state = {"failed": {"d1"}}

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix(knowledge.PREFIX)
        if request.method == "POST" and path == "":
            return httpx.Response(409, json={"detail": "exists"})
        if path == "/b/ingest":
            ids = [i["document_id"] for i in json.loads(request.content)["items"]]
            uploads.append(ids)
            if ids == ["d1"]:  # the retry: d1 is ready this time
                state["failed"] = set()
            return httpx.Response(200, json={"document_ids": ids})
        if path == "/b":
            failed = len(state["failed"])
            return httpx.Response(200, json={"passage_count": 7, "ready_count": 450 - failed, "failed_count": failed})
        if path == "/b/documents":
            return httpx.Response(200, json={"documents": [{"document_id": d, "error": "x"} for d in state["failed"]]})
        return httpx.Response(404)

    api = KnowledgeApi("http://kb")
    api._http = httpx.AsyncClient(base_url="http://kb", transport=httpx.MockTransport(handle))
    items = [TextItem(id=f"d{n}", text="t", filename="f.txt") for n in range(450)]
    loaded: list[int] = []

    def load(ids: object) -> list[TextItem]:
        batch = preloaded(items)(ids)  # type: ignore[arg-type]
        loaded.append(len(batch))
        return list(batch)

    stats = await ingest(api, "b", [i.id for i in items], load, reuse_bank=True)
    await api.aclose()

    # 450 documents: two loads (400 + 50), texts uploaded 100 per request, then d1 again.
    assert loaded[:2] == [400, 50]
    assert [len(u) for u in uploads] == [100, 100, 100, 100, 50, 1] and uploads[-1] == ["d1"]
    assert stats.failed == 0 and stats.passages == 7


async def _no_sleep(_seconds: float) -> None:
    return None
