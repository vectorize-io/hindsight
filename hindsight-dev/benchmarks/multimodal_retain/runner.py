"""Run both arms of the multimodal-retain benchmark against a live server.

Two banks per article, from identical prose:

- **multimodal** — the article as callers can now send it, with each image inline
  in the position it occupies;
- **text-only** — the same article with the images simply absent. This is the
  honest pre-feature baseline: a caller who could not send images did not get
  captions instead, they got prose that pointed at pictures nobody saw.

Both arms are then asked the same questions, every one of which is answerable
only from an image. The comparison is the measurement.

The harness speaks HTTP, so the same code drives any build.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .corpus import Article
from .images import ImageSpec, render
from .judge import AnswerVerdict, FactVerdict, judge_all, judge_answer, judge_fact_present

ARMS = ("multimodal", "text-only")


class QuestionResult(BaseModel):
    question: str
    answer: str = ""
    verdict: AnswerVerdict | None = None
    error: str = ""


class ArmRun(BaseModel):
    """One arm's outcome for one article."""

    arm: str
    bank_id: str
    retained: bool = False
    retain_error: str = ""
    facts: list[str] = Field(default_factory=list)
    image_facts: list[FactVerdict] = Field(default_factory=list)
    questions: list[QuestionResult] = Field(default_factory=list)
    retain_ms: int = 0


class ArticleRun(BaseModel):
    article: str
    arms: list[ArmRun] = Field(default_factory=list)


class BenchmarkArtifact(BaseModel):
    """Everything a run produced, so metrics can be recomputed without re-running."""

    build: str
    api_url: str
    api_version: str = ""
    runs: list[ArticleRun] = Field(default_factory=list)


class _Client:
    def __init__(self, api_url: str, timeout: float = 600.0) -> None:
        self._base = api_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def _json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._http.request(method, f"{self._base}{path}", json=payload)
        response.raise_for_status()
        return response.json() if response.content else {}

    async def version(self) -> str:
        return (await self._json("GET", "/version")).get("api_version", "")

    async def retain(self, bank_id: str, content: str | list[dict], document_id: str) -> None:
        """Retain one article synchronously, so the bank is queryable on return."""
        await self._json(
            "POST",
            f"/v1/default/banks/{bank_id}/memories",
            {"items": [{"content": content, "document_id": document_id}], "async": False},
        )

    async def facts(self, bank_id: str) -> list[str]:
        """Every fact in the bank, so coverage is measured against what was stored."""
        response = await self._json("GET", f"/v1/default/banks/{bank_id}/memories/list?limit=200")
        return [item.get("text", "") for item in response.get("items", [])]

    async def reflect(self, bank_id: str, query: str) -> str:
        response = await self._json(
            "POST",
            f"/v1/default/banks/{bank_id}/reflect",
            {"query": query},
        )
        return response.get("text") or ""

    async def delete_bank(self, bank_id: str) -> None:
        try:
            await self._json("DELETE", f"/v1/default/banks/{bank_id}")
        except httpx.HTTPStatusError:
            # A bank that was never created is not an error worth failing a run for.
            pass


def _image_block(spec: ImageSpec) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(render(spec)).decode(),
        },
    }


def _multimodal_content(article: Article) -> list[dict]:
    return [
        {"type": "text", "text": block} if isinstance(block, str) else _image_block(block) for block in article.body
    ]


async def _run_arm(client: _Client, article: Article, arm: str, run_id: str) -> ArmRun:
    bank_id = f"mmbench-{run_id}-{article.name}-{arm}"
    result = ArmRun(arm=arm, bank_id=bank_id)

    content: str | list[dict] = _multimodal_content(article) if arm == "multimodal" else article.text_only_body()

    started = time.time()
    try:
        await client.retain(bank_id, content, document_id=article.name)
        result.retained = True
    except httpx.HTTPStatusError as exc:
        # A 422 here is a real finding, not a harness bug: it means the server's
        # retain LLM cannot read images, and the multimodal arm cannot be run at
        # all. Recorded rather than raised so the other arm still reports.
        result.retain_error = f"{exc.response.status_code}: {exc.response.text[:300]}"
        return result
    result.retain_ms = int((time.time() - started) * 1000)

    result.facts = await client.facts(bank_id)

    # Mechanism: did the image-only detail reach memory as facts at all?
    claims = [fact for spec in article.images for fact in spec.facts]
    result.image_facts = await judge_all([judge_fact_present(claim, result.facts) for claim in claims])

    # Outcome: can the bank answer questions that need those details?
    for question in article.questions:
        entry = QuestionResult(question=question.question)
        try:
            entry.answer = await client.reflect(bank_id, question.question)
        except httpx.HTTPStatusError as exc:
            entry.error = f"{exc.response.status_code}: {exc.response.text[:200]}"
        result.questions.append(entry)

    verdicts = await judge_all(
        [judge_answer(q.question, q.expected, entry.answer) for q, entry in zip(article.questions, result.questions)]
    )
    for entry, verdict in zip(result.questions, verdicts):
        entry.verdict = verdict

    return result


async def run_benchmark(
    api_url: str,
    articles: list[Article],
    *,
    build: str,
    run_id: str,
    keep_banks: bool = False,
    progress=None,
) -> BenchmarkArtifact:
    client = _Client(api_url)
    try:
        artifact = BenchmarkArtifact(build=build, api_url=api_url, api_version=await client.version())

        for article in articles:
            run = ArticleRun(article=article.name)
            for arm in ARMS:
                if progress is not None:
                    progress(f"{article.name} / {arm}")
                run.arms.append(await _run_arm(client, article, arm, run_id))
            artifact.runs.append(run)

        if not keep_banks:
            await asyncio.gather(*(client.delete_bank(arm.bank_id) for run in artifact.runs for arm in run.arms))
        return artifact
    finally:
        await client.close()
