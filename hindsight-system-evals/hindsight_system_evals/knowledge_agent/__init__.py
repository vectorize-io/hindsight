"""A Hermes agent answering questions over a knowledge bank, driven from the eval process.

The agent itself runs in Hermes' own venv (``runner.py``, started as a subprocess with
``HINDSIGHT_EVAL_HERMES_PYTHON``); its tools are the ``hindsight_knowledge`` plugin beside it.
This module only hands it the questions and reads the answers back.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runner.py"


def hermes_python() -> Path:
    return Path(
        os.getenv("HINDSIGHT_EVAL_HERMES_PYTHON") or Path.home() / "dev" / "hermes-agent" / ".venv" / "bin" / "python"
    )


class AgentQuestion(BaseModel):
    question_id: str
    question: str


class AgentAnswer(BaseModel):
    question_id: str
    answer: str
    document_ids: list[str]
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    tool_calls: dict[str, int]
    seconds: float
    error: str | None = None


def answer_all(
    questions: Sequence[AgentQuestion], *, api_url: str, api_key: str | None, bank_id: str, work_dir: Path
) -> list[AgentAnswer]:
    """Run every question through the agent; answers come back in question order.

    Resumable: answers already in ``work_dir/agent.jsonl`` are kept and their questions not
    asked again, so an interrupted 500-question run continues where it stopped.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "agent.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answer = AgentAnswer.model_validate_json(line)
                done[answer.question_id] = answer
    todo = [q for q in questions if q.question_id not in done]
    if todo:
        pending = work_dir / "questions.jsonl"
        pending.write_text("".join(q.model_dump_json() + "\n" for q in todo), encoding="utf-8")
        env = os.environ.copy()
        env |= {
            "HINDSIGHT_KNOWLEDGE_API_URL": api_url,
            "HINDSIGHT_KNOWLEDGE_BANK": bank_id,
        }
        if api_key:
            env["HINDSIGHT_KNOWLEDGE_API_KEY"] = api_key
        # stderr (one progress line per question) passes straight through to the pytest log.
        subprocess.run(
            [str(hermes_python()), str(RUNNER), "--questions", str(pending), "--out", str(out)], env=env, check=True
        )
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                answer = AgentAnswer.model_validate_json(line)
                done[answer.question_id] = answer
    return [done[q.question_id] for q in questions]
