"""An agent over a knowledge bank, scored on EnterpriseRAG-Bench the way its leaderboard is.

The knowledge bank does retrieval with no LLM; this puts an agent on top (Hermes, with the
``hindsight_knowledge`` plugin's search / read tools) and grades its answers with the
benchmark's own judge and metric (``enterprise_rag_scoring``), so the number lines up with
https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard.

Default: the first 50 questions of the committed slice (the benchmark's category mix), every
document they need, their BM25 hard negatives and 2,000 random distractors. ``--full``: all 500
questions and 20,000 random distractors. ``HINDSIGHT_EVAL_ERB_QUESTIONS`` /
``HINDSIGHT_EVAL_ERB_DISTRACTORS`` (-1 = the whole 512k corpus) override either.

Needs: the agent's model (``HINDSIGHT_EVAL_AGENT_BASE_URL`` / ``_MODEL``, an OpenAI-compatible
endpoint), Hermes' interpreter (``HINDSIGHT_EVAL_HERMES_PYTHON``), the judge's
``OPENAI_API_KEY``, and a server whose knowledge embedder is configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from hindsight_system_evals.enterprise_rag import Question, ingest, subset
from hindsight_system_evals.enterprise_rag_scoring import (
    JUDGE_EFFORT,
    JUDGE_MODEL,
    USAGE,
    QuestionScore,
    aggregate,
    score,
)
from hindsight_system_evals.knowledge import KnowledgeApi, embedder_name, percentile
from hindsight_system_evals.knowledge_agent import AgentAnswer, AgentQuestion, answer_all, hermes_python
from hindsight_system_evals.report import KNOWLEDGE, AgentLatency, KnowledgeResult
from hindsight_system_evals.target import Target

#: The leaderboard's overall score (completeness when correct, else 0), for the summary line.
LEADERBOARD = {"PipesHub": 93.24, "OpenAI File Search": 61.03, "Azure AI Search": 48.42}


class _StoredScore(BaseModel):
    """One line of ``scores.jsonl``: the score, and what it was a score of. A line is reused only when
    both still match: the same judge, and the same answer with the same cited documents."""

    judge: str
    answer_key: str
    score: QuestionScore


def _answer_key(answer: AgentAnswer) -> str:
    return hashlib.sha1("\x1f".join([answer.answer, *answer.document_ids]).encode()).hexdigest()


@dataclass(frozen=True)
class Scored:
    scores: list[QuestionScore]
    #: Question ids whose scoring raised (the judge unreachable after its retries).
    failed: list[str]


async def _score_all(questions: Sequence[Question], answers: Sequence[AgentAnswer], path: Path) -> Scored:
    """Score every answer, appending each score to ``path`` as it completes.

    Resumable like ``agent.jsonl``: an answer already scored by the same judge is not judged
    again, so a judge failure (or a killed run) costs only the questions it hit. A new answer, or
    another judge model or effort, is judged afresh. ``judge_calls`` and the judge tokens count
    this run's calls only. Delete the file to re-score from scratch.
    """
    judge = f"{JUDGE_MODEL}:{JUDGE_EFFORT}"
    keys = {q.question_id: _answer_key(a) for q, a in zip(questions, answers, strict=True)}
    done: dict[str, QuestionScore] = {}
    # Cut short by a killed run, a line can end mid-character: replace it rather than fail the resume.
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if text:
        for line in text.splitlines():
            try:
                stored = _StoredScore.model_validate_json(line)
            except ValidationError:
                continue  # blank, half-written by a killed run, or from before answers were keyed
            if stored.judge == judge and keys.get(stored.score.question_id) == stored.answer_key:
                done[stored.score.question_id] = stored.score
    failed: list[str] = []

    with path.open("a", encoding="utf-8") as out:
        if text and not text.endswith("\n"):
            out.write("\n")  # a killed run's last line, cut short: the next score must not join it

        async def one(q: Question, a: AgentAnswer) -> None:
            try:
                s = await score(q, a.answer, a.document_ids)
            except Exception as exc:  # one question's judge failure must not drop the others
                print(f"  scoring {q.question_id} failed: {exc!r}")
                failed.append(q.question_id)
                return
            done[q.question_id] = s
            line = _StoredScore(judge=judge, answer_key=keys[q.question_id], score=s)
            out.write(line.model_dump_json() + "\n")
            out.flush()

        await asyncio.gather(*(one(q, a) for q, a in zip(questions, answers, strict=True) if q.question_id not in done))
    return Scored(scores=[done[q.question_id] for q in questions if q.question_id in done], failed=failed)


# The package-wide 2400s timeout is for one eval case; 500 agent runs and ~4,000 judge calls
# take an hour or two.
@pytest.mark.timeout(8 * 3600)
async def test_knowledge_agent_benchmark_enterprise_rag(
    request: pytest.FixtureRequest, target: Target, tmp_path: Path
) -> None:
    if not hermes_python().exists():
        pytest.skip(f"Hermes interpreter not found at {hermes_python()} (set HINDSIGHT_EVAL_HERMES_PYTHON)")
    if not (os.getenv("HINDSIGHT_EVAL_ERB_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        pytest.skip("the benchmark's judge is GPT-5.4: set OPENAI_API_KEY")

    full = request.config.getoption("--full")
    n_questions = int(os.getenv("HINDSIGHT_EVAL_ERB_QUESTIONS") or (500 if full else 50))
    n_random = int(os.getenv("HINDSIGHT_EVAL_ERB_DISTRACTORS") or (20000 if full else 2000))
    chosen = subset(n_questions, n_random)
    # A fixed name per size, so a remote server that already holds the corpus (--keep-banks)
    # re-uploads it as unchanged documents instead of re-embedding.
    bank = os.getenv("HINDSIGHT_EVAL_ERB_BANK") or f"syseval-erb-{n_questions}q-{n_random}"
    work = Path(os.getenv("HINDSIGHT_EVAL_ERB_WORK_DIR") or tmp_path) / bank
    work.mkdir(parents=True, exist_ok=True)

    client = KnowledgeApi(target.url, target.api_key)
    try:
        # Iterating on the agent against a bank that already holds this corpus: skip even the
        # unchanged-document upload pass (minutes on 35k documents). The run then reports no ingest.
        ingested = None
        if not os.getenv("HINDSIGHT_EVAL_ERB_REUSE_BANK"):
            ingested = await ingest(client, bank, chosen.document_ids)
            print(f"\n{bank}: {ingested.documents} documents, {ingested.passages} passages in {ingested.wall_s:.0f}s")
        try:
            answers = await asyncio.to_thread(
                answer_all,
                [AgentQuestion(question_id=q.question_id, question=q.question) for q in chosen.questions],
                api_url=target.url,
                api_key=target.api_key,
                bank_id=bank,
                work_dir=work,
            )
        finally:
            if target.is_remote and not request.config.getoption("--keep-banks"):
                with contextlib.suppress(Exception):
                    await client.delete_bank(bank)
        embedder = embedder_name()
    finally:
        await client.aclose()

    # The upstream answers file, so a run can also be scored with Onyx's own script.
    (work / "answers.jsonl").write_text(
        "".join(
            json.dumps({"question_id": a.question_id, "answer": a.answer, "document_ids": a.document_ids}) + "\n"
            for a in answers
        ),
        encoding="utf-8",
    )
    scored = await _score_all(chosen.questions, answers, work / "scores.jsonl")
    scores, judge_failed = scored.scores, scored.failed

    metrics = aggregate(scores)
    seconds = [a.seconds for a in answers]
    metrics |= {
        "llm_calls_per_question": round(statistics.fmean(a.llm_calls for a in answers), 2),
        "prompt_tokens_per_question": round(statistics.fmean(a.prompt_tokens for a in answers)),
        "completion_tokens_per_question": round(statistics.fmean(a.completion_tokens for a in answers)),
        "tool_calls_per_question": round(statistics.fmean(sum(a.tool_calls.values()) for a in answers), 2),
        "agent_errors": float(sum(1 for a in answers if a.error)),
        "judge_errors": float(len(judge_failed)),
        "judge_calls": float(USAGE.calls),
        "judge_input_tokens": float(USAGE.input_tokens),
        "judge_output_tokens": float(USAGE.output_tokens),
    }
    KNOWLEDGE.append(
        KnowledgeResult(
            category="knowledge_agent",
            case=f"enterprise-rag-{n_questions}q",
            embedder=embedder,
            corpus=len(chosen.document_ids),
            queries=n_questions,
            metrics=metrics,
            ingest=ingested,
            agent_latency=AgentLatency(
                questions=len(answers),
                concurrency=int(os.getenv("HINDSIGHT_EVAL_AGENT_CONCURRENCY", "4")),
                answer_p50_ms=round(1000 * percentile(seconds, 50)),
                answer_p95_ms=round(1000 * percentile(seconds, 95)),
            ),
        )
    )
    board = ", ".join(f"{k} {v}" for k, v in LEADERBOARD.items())
    print(
        f"\nEnterpriseRAG-Bench ({n_questions} questions, {len(chosen.document_ids)} documents, judge {JUDGE_MODEL}/{JUDGE_EFFORT})"
    )
    for key, value in metrics.items():
        print(f"  {key:34} {value}")
    print(f"  leaderboard: {board}")
    print(f"  per-question detail: {work}")
    # A baseline, not a gate: the floor only catches a broken pipeline (no answers, no tools).
    assert metrics["agent_errors"] < len(answers) / 2, "most questions errored; see the agent log above"
    assert metrics["overall"] > 10, f"overall {metrics['overall']}: the agent is not finding anything"
    # After the result is recorded, so a judge outage still publishes what was scored; a rerun
    # with the same HINDSIGHT_EVAL_ERB_WORK_DIR judges only these.
    assert not judge_failed, f"{len(judge_failed)} questions could not be judged, e.g. {judge_failed[:5]}"
