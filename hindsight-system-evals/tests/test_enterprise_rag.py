"""The EnterpriseRAG-Bench pieces that can be wrong without a run noticing: the stratified
question order, upstream's gold-correction vote, the leaderboard's overall formula, and how the
agent plugin shows the server's results. No dataset download, no model."""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from hindsight_system_evals import enterprise_rag_scoring as scoring
from hindsight_system_evals.enterprise_rag import Question, _stratified
from hindsight_system_evals.enterprise_rag_scoring import QuestionScore, aggregate


def _question(qid: str, qtype: str) -> Question:
    return Question(
        question_id=qid,
        question_type=qtype,
        source_types=[],
        question="q",
        expected_doc_ids=[],
        gold_answer="",
        answer_facts=[],
    )


def test_every_prefix_keeps_the_category_mix() -> None:
    questions = [_question(f"b{i}", "basic") for i in range(175)] + [
        _question(f"h{i}", "high_level") for i in range(10)
    ]
    ordered = _stratified(questions, seed=1)
    first = collections.Counter(q.question_type for q in ordered[:37])  # 20% of 185
    assert first == {"basic": 35, "high_level": 2}
    assert sorted(q.question_id for q in ordered) == sorted(q.question_id for q in questions)


def _score(qtype: str, correct: bool, completeness: float, recall: float | None = 100.0) -> QuestionScore:
    return QuestionScore(
        question_id="x",
        question_type=qtype,
        correct=correct,
        correctness_reason="",
        completeness_pct=completeness,
        recall_pct=recall,
        invalid_extra_docs=0 if recall is not None else None,
        corrected=False,
    )


def test_overall_is_completeness_when_correct_else_zero() -> None:
    metrics = aggregate(
        [_score("basic", True, 50.0), _score("basic", False, 100.0), _score("info_not_found", True, 100.0, None)]
    )
    assert metrics["overall"] == 50.0  # (50 + 0 + 100) / 3
    assert metrics["correctness"] == pytest.approx(66.67)
    assert metrics["recall"] == 100.0  # questions with no gold documents are left out
    assert metrics["basic:overall"] == 25.0


async def _run_consensus(monkeypatch: pytest.MonkeyPatch, runs: list[dict[str, str]]) -> scoring.Correction | None:
    queue = list(runs)

    async def classify(*_args: object) -> dict[str, str]:
        return queue.pop(0)

    monkeypatch.setattr(scoring, "_classify", classify)
    monkeypatch.setattr(scoring, "read_documents", lambda ids: [])
    return await scoring._consensus("q", ["g"], ["c"])


async def test_a_run_agreeing_with_gold_ends_the_vote(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _run_consensus(monkeypatch, [{"g": "required", "c": "valid"}]) is None


async def test_a_candidate_needs_a_strict_majority(monkeypatch: pytest.MonkeyPatch) -> None:
    # c: required, required, invalid -> mean 4/3, not > 1.5 -> valid; the gold set stands.
    runs = [{"g": "required", "c": "required"}, {"g": "required", "c": "required"}, {"g": "valid", "c": "invalid"}]
    assert await _run_consensus(monkeypatch, runs) is None


async def test_gold_voted_invalid_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = [{"g": "invalid", "c": "required"}] * 3
    fix = await _run_consensus(monkeypatch, runs)
    assert fix is not None and fix.required == ["c"]


class _Reply:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self) -> dict:
        return self._body


def _plugin(monkeypatch: pytest.MonkeyPatch):
    path = (
        Path(__file__).resolve().parents[1] / "hindsight_system_evals/knowledge_agent/hindsight_knowledge/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("hk_test", path)
    plugin = importlib.util.module_from_spec(spec)
    sys.modules["hk_test"] = plugin
    spec.loader.exec_module(plugin)
    monkeypatch.setenv("HINDSIGHT_KNOWLEDGE_API_URL", "http://kb")
    monkeypatch.setenv("HINDSIGHT_KNOWLEDGE_BANK", "erb")
    return plugin


def test_plugin_search_heads_each_hit_with_its_document_id_not_a_number(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin(monkeypatch)
    hit = lambda d, text: {"document_id": d, "filename": f"{d}.txt", "text": text, "score": 0.5, "tags": []}  # noqa: E731
    monkeypatch.setattr(
        plugin, "_request", lambda *a, **k: _Reply(200, {"results": [hit("dsid_a", "x"), hit("dsid_b", "y")]})
    )
    lines = plugin.knowledge_search({"query": "q"}).splitlines()
    # A numbered list ("[1] dsid_a") gets cited as "Sources: [1]", which names no document.
    assert lines[0].startswith("dsid_a | dsid_a.txt")
    assert not any(line.startswith("[") for line in lines)


def test_plugin_grep_groups_the_servers_matches_by_document(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _plugin(monkeypatch)
    sent: list[dict] = []
    Reply = _Reply

    def request(method: str, url: str, **kwargs: object) -> Reply:
        body = kwargs["json"]
        assert url.endswith("/search")  # exact text is search with only a pattern
        sent.append(body)
        if body["regex"] and body["pattern"].startswith("("):
            return Reply(400, {"detail": "invalid regular expression: parentheses not balanced"})
        hit = lambda d, text, n: {"document_id": d, "filename": f"{d}.txt", "text": text, "score": n}  # noqa: E731
        return Reply(
            200,
            {
                "results": [
                    hit("d1", "intro\nWe added stream.timebox_finalized.", 2),
                    hit("d1", "timebox_finalized fires at the limit", 2),
                    hit("d2", "see timebox_finalized", 1),
                ]
            },
        )

    monkeypatch.setattr(plugin, "_request", request)
    out = plugin.knowledge_grep({"pattern": "timebox_finalized|timebox-finalized"})
    assert sent[-1] == {"pattern": "timebox_finalized|timebox-finalized", "regex": True, "top_k": 200, "neighbors": 0}
    lines = out.splitlines()
    assert lines[0].startswith("2 documents match")
    assert lines[2] == "d1 | d1.txt | 2 matching passages"
    assert lines[3:5] == ["    We added stream.timebox_finalized.", "    timebox_finalized fires at the limit"]
    assert lines[5] == "d2 | d2.txt | 1 matching passages"
    # A "regex" the server rejects is retried as plain text.
    plugin.knowledge_grep({"pattern": "(timebox"})
    assert [b["regex"] for b in sent[-2:]] == [True, False]
