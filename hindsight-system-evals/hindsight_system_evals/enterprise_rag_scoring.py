"""EnterpriseRAG-Bench's own scoring, ported so a number here is comparable to the leaderboard.

Ported from ``src/scripts/answer_evaluation/metrics_based_eval.py`` and ``src/utils/eval_utils.py``
of https://github.com/onyx-dot-app/EnterpriseRAG-Bench (MIT). The prompts are theirs, verbatim
(``src/prompts/answer_evaluation.py``, ``question_fact_extraction.py``); so are the judge, GPT-5.4
at medium reasoning effort through the Responses API, and the flow:

1. the answer's citations are stripped (an LLM call, then every ``dsid_…`` removed), so the
   judge is not swayed by them;
2. when the answer cites documents other than the gold set, three judge runs classify every
   gold and candidate document as required / valid / invalid (early exit on agreement with the
   gold set, ordinal majority vote with gold-biased ties); if the required set changes, the gold
   answer and its facts are regenerated from the new set (the leaderboard scores this way);
3. **correctness**: one holistic aligned-with-gold verdict; **completeness**: each gold fact
   validated on its own; **recall**: share of gold documents cited; **invalid extra
   documents**: cited documents that are neither gold nor judged valid;
4. **overall score** (the leaderboard's rank column): mean over questions of completeness when
   correct, else 0.

One deliberate difference: the documents shown to the judge in step 2 are the release's text
files (title line + body), where upstream dumps each document's JSON record. Same content, a
different wrapper; it only touches questions where the agent cites non-gold documents.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from hindsight_system_evals.enterprise_rag import Question, corpus_members, read_documents

JUDGE_MODEL = os.getenv("HINDSIGHT_EVAL_ERB_JUDGE_MODEL", "gpt-5.4")
JUDGE_EFFORT = os.getenv("HINDSIGHT_EVAL_ERB_JUDGE_EFFORT", "medium")
_JUDGE_CONCURRENCY = asyncio.Semaphore(int(os.getenv("HINDSIGHT_EVAL_ERB_JUDGE_CONCURRENCY", "16")))
_RETRIES = 3
_DSID = re.compile(r"dsid_[a-f0-9]{20,32}")

# --- the upstream prompts, verbatim ---------------------------------------------------------

ANSWER_CITATION_STRIPPING_PROMPT = """
Strip all citations and reference sections from the provided string without modifying any other content. Citations of all forms such as [number], [number][number], [number](link), etc. should all be removed. \
Anything that is clearly a citation, regardless of format should be removed. Also remove any reference footers or intermediate sections that are clearly references. \
Output the stripped string with no additional text or formatting changes.

String to strip:
```
{answer_string}
```

CRITICAL: Output only the stripped string and nothing else. Only citations should be removed, everything else should be left unchanged.
""".strip()

ANSWER_DOC_EVALUATION_PROMPT = """
You are a precise and detail-oriented document evaluation expert. Given a query and a set of documents, classify each document using one of three labels:

- "required": The document is essential to answering the query. Without it, the answer would be incomplete or incorrect. The information it provides is not already covered by other required documents.
- "valid": The document is relevant and contains related information, but is NOT necessary to fully answer the query. For example, it may corroborate or duplicate information already present in a required document, or it may provide supplementary context that is not directly asked for.
- "invalid": The document does not help answer the query, or is ruled out by qualifying details in the query even if the document seems superficially relevant.

Guidelines:
- Be extremely detail-oriented and strict with the query. There may be qualifying details that rule out documents even if the vast majority of the document seems relevant.
- A document that contains the same key facts as another required document should be classified as "valid", not "required". If it's a case where the additional document changes the answer or some metrics, then it is "required", if it wouldn't affect the answer, it's likely not required.
- Gold documents are very likely to be "required" but in rare cases a gold document may be "valid" or even "invalid" if a candidate document better answers the query. A candidate document may invalidate a gold document if it is a better source (newer or more authoritative) and disagrees with the gold document.
- Near-duplicate documents (same information, different source or version) should not both be "required" unless each contributes unique essential information. In cases where one of these is a gold document and another is a candidate, favor the gold document.

## Query
```
{query}
```

## Gold Documents
{gold_documents}

## Candidate Documents
{candidate_documents}


CRITICAL: Output only a JSON object with the following fields in the order shown below (make sure to include all gold and candidate documents):
{{
  "dsid_key": {{
    "reason": "reason for the classification",
    "classification": "required, valid, or invalid"
  }}
}}
""".strip()

ANSWER_UPDATOR_PROMPT = """
You are an expert dataset engineer whose job is to update a gold answer based on a query and a set of documents. The previous gold answer is provided for reference. \
The new answer should loosely have the same style, language, and length as the previous gold answer but be updated to be based on the set of documents provided. \
You must output only the new gold answer and nothing else. Keep the character set simple, do not output special characters like emojis, markdown, or other non-ASCII characters.

## Previous Gold Answer
```
{previous_gold_answer}
```

## Reference Documents
```
{reference_documents}
```

## Query
```
{query}
```

CRITICAL: Output only the new gold answer and nothing else.
""".strip()

ANSWER_WHOLISTIC_EVALUATION_PROMPT = """
You are a wholistic and detail-oriented answer evaluator. Given a query, a gold answer, and a candidate answer, evaluate if the candidate answer aligned with the gold answer.
Use the following metrics for evaluating the answer:
- The candidate answer must provide loosely the same information as the gold answer. The core aspects directly asked by the query must be addressed in the candidate answer and they must not conflict with the gold answer.
- If there are any specific quantities mentioned in both answers, they must match.
- The candidate answer is not required to contain all of the same details as the gold answer.
- The candidate answer must address the key parts of the query, if it is missing anything critical to the question, it is misaligned.
- The candidate answer may contain more details, richer information, or other helpful relevant information than the gold answer, this is ok.
- The candidate answer may offer up additional loosely related information that adds to the context of the answer, this is ok as long as it does not lead the user to an incorrect conclusion (compared to the gold answer).
- Do not penalize the candidate answer for stylistic differences. If the candidate answer offers follow up questions, asks additional clarifications to the user, or offers additional context, \
this is ok as long as it contains the necessary information to answer the question.

There is a separate check for answer completeness, this is not in scope for this evaluation. However, if there are core parts of the question being left out, this is misaligned.

## Query
```
{query}
```

## Gold Answer
```
{gold_answer}
```

## Candidate Answer
```
{candidate_answer}
```

## Output Format
Output a JSON with "reason" and "aligned" fields. The "reason" field should be a as concise as possible (max 1 sentence) explanation of why the candidate answer is aligned or misaligned with the gold answer. \
The "aligned" field should be a simple "yes" or "no", use only those two strings literally and nothing else.

CRITICAL: Output only a JSON object with the following fields in the order shown below (with no additional text or formatting):
{{
  "reason": "reason for the classification",
  "aligned": "yes or no"
}}
""".strip()

INDIVIDUAL_FACT_VALIDATOR_PROMPT = """
You are an answer validator. Given an answer and a statement, determine if the answer is consistent with and contains the information in the statement. \
The answer may contain more details or richer information than the statement but as long as it does not contradict the statement, this is valid. \
If there are negative statements such as "The answer must not say...", it is valid if the answer mentions the statement with caveats or qualifications. \
It is valid if additional context is shared for completeness however hallucinations are not allowed. \
Output a simple yes or no for if the answer is consistent with and contains the information in the statement.

## Answer
```
{answer}
```

## Statement
```
{statement}
```

CRITICAL: output only a simple yes if the answer is consistent with the statement or a no if the answer does not contain the information in the statement or contradicts the statement.
""".strip()

FACT_EXTRACTION_PROMPT = """
Given the following question and gold answer, extract a list of individual verifiable facts from the answer. Output your response as a list of strings. \
Format it as `["fact 1", "fact 2", "fact 3", ...]`. The facts are intended to verify the completeness and accuracy of a candidate answer. \
The gold answer may contain some facts that are not necessary to fully answer the question, you should not include these in your response. \
Keep the character set of the facts simple, do not output special characters like emojis, markdown, or other non-ASCII characters.

Note: do not overly break up facts. For example, "The auth issue might be caused by a bad API key" + "The auth issue might be caused by a bad auth header" \
is better as "The auth issue might be caused by a bad API key or a bad auth header".

# Question
```
{question}
```

# Gold Answer
```
{gold_answer}
```

CRITICAL: You must output your response as a list of strings. Do not include any other text or formatting.
""".strip()

ANTI_HALLUCINATION_FACT_VALIDATOR_PROMPT = """
You are given a list of facts which must be part of an expected answer. Some of them may be negation type statements which is to prevent hallucinations. \
If those exist in the list, you must output them in a list verbatim as they are in the fact list below. If there are no such anti-hallucination statements, you must output an empty list.

Fact List:
```
{fact_list}
```

Your output should be a list of strings like `["answer must not contain X", "the answer should not be Y", ...]`.
""".strip()


# --- the judge ------------------------------------------------------------------------------


@dataclass
class JudgeUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


USAGE = JudgeUsage()
_CLIENT: httpx.AsyncClient | None = None


async def _judge(prompt: str) -> str:
    """One GPT-5.4 call through the Responses API, as upstream's ``OpenAILLM`` makes it."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(timeout=600)
    key = os.environ.get("HINDSIGHT_EVAL_ERB_JUDGE_API_KEY") or os.environ["OPENAI_API_KEY"]
    body = {"model": JUDGE_MODEL, "input": [{"role": "user", "content": prompt}], "reasoning": {"effort": JUDGE_EFFORT}}
    async with _JUDGE_CONCURRENCY:
        for attempt in range(6):
            # A dropped connection or a timeout is as transient as a 429/5xx over ~4,000 calls a
            # run; retried the same way so one of them does not cost the question its score.
            try:
                response = await _CLIENT.post(
                    "https://api.openai.com/v1/responses", json=body, headers={"Authorization": f"Bearer {key}"}
                )
            except httpx.TransportError:
                if attempt == 5:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            if response.status_code in (429, 500, 502, 503) and attempt < 5:
                await asyncio.sleep(2**attempt)
                continue
            response.raise_for_status()
            break
    data = response.json()
    USAGE.calls += 1
    USAGE.input_tokens += data.get("usage", {}).get("input_tokens", 0)
    USAGE.output_tokens += data.get("usage", {}).get("output_tokens", 0)
    return "".join(
        part.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
    ).strip()


def _json(response: str) -> object:
    """Upstream's ``extract_json_from_response``: the outermost {...} or [...]."""
    starts = [i for i in (response.find("{"), response.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON in response")
    start = min(starts)
    end = response.rfind("}" if response[start] == "{" else "]")
    return json.loads(response[start : end + 1])


async def strip_citations(answer: str) -> str:
    try:
        stripped = await _judge(ANSWER_CITATION_STRIPPING_PROMPT.format(answer_string=answer)) or answer
    except httpx.HTTPError:
        stripped = answer
    return re.sub(r" {2,}", " ", _DSID.sub("", stripped)).strip()


@dataclass(frozen=True)
class Alignment:
    correct: bool
    reason: str


async def _correct(question: str, gold: str, candidate: str) -> Alignment:
    for _ in range(_RETRIES):
        try:
            parsed = _json(
                await _judge(
                    ANSWER_WHOLISTIC_EVALUATION_PROMPT.format(
                        query=question, gold_answer=gold, candidate_answer=candidate
                    )
                )
            )
        except (ValueError, httpx.HTTPError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("aligned"), str):
            return Alignment(re.search(r"\byes\b", parsed["aligned"], re.I) is not None, str(parsed.get("reason", "")))
    return Alignment(False, "judge failed")


async def _fact(answer: str, statement: str) -> bool:
    response = await _judge(INDIVIDUAL_FACT_VALIDATOR_PROMPT.format(answer=answer, statement=statement))
    first = response.strip().splitlines()[0].strip() if response.strip() else ""
    return re.search(r"\byes\b", first, re.I) is not None


# --- the gold-set correction flow -----------------------------------------------------------


def _doc_block(document_id: str, text: str) -> str:
    return f"Document ID: {document_id}\n```\n{text}\n```"


async def _classify(
    question: str, gold: list[str], candidates: list[str], texts: dict[str, str]
) -> dict[str, str] | None:
    shuffled = list(candidates)
    random.shuffle(shuffled)  # upstream shuffles candidates per run against positional bias
    prompt = ANSWER_DOC_EVALUATION_PROMPT.format(
        query=question,
        gold_documents="\n\n".join(_doc_block(d, texts[d]) for d in gold),
        candidate_documents="\n\n".join(_doc_block(d, texts[d]) for d in shuffled),
    )
    for _ in range(_RETRIES):
        try:
            parsed = _json(await _judge(prompt))
        except (ValueError, httpx.HTTPError):
            continue
        if not isinstance(parsed, dict):
            continue
        out = {}
        for d in gold + candidates:
            entry = parsed.get(d)
            if not isinstance(entry, dict) or entry.get("classification") not in ("required", "valid", "invalid"):
                break
            out[d] = entry["classification"]
        else:
            return out
    return None


@dataclass(frozen=True)
class Correction:
    required: list[str]
    valid: list[str]


async def _consensus(question: str, gold: list[str], candidates: list[str]) -> Correction | None:
    """Three runs, upstream's rules: any run agreeing with gold ends it; else ordinal majority
    (required 2, valid 1, invalid 0) with gold kept unless voted invalid and candidates needing a
    strict majority. ``None`` means the gold set stands."""
    texts = {d.id: d.text for d in read_documents(gold + candidates)}
    runs = []
    for _ in range(3):
        run = await _classify(question, gold, candidates, texts)
        if run is None:
            continue
        runs.append(run)
        if {d for d, c in run.items() if c == "required"} == set(gold):
            return None
    if not runs:
        return None
    score = {"required": 2, "valid": 1, "invalid": 0}
    required, valid = [], []
    for d in gold + candidates:
        avg = statistics.fmean(score[r[d]] for r in runs)
        cls = (
            ("required" if avg >= 0.5 else "invalid")
            if d in gold
            else ("required" if avg > 1.5 else "valid" if avg > 0.5 else "invalid")
        )
        (required if cls == "required" else valid if cls == "valid" else []).append(d)
    # Upstream keeps the gold set (and records no valid documents) when the vote confirms it or
    # marks nothing required; valid documents only count once the required set has changed.
    if set(required) == set(gold) or not required:
        return None
    return Correction(required=required, valid=valid)


async def _regenerate(question: Question, required: list[str]) -> Question:
    docs = read_documents(required)
    reference = "\n\n".join("\n".join(p for p in d.text.split("\n", 1) if p) for d in docs)
    # A judge that stays down after its retries leaves the gold answer as it was (only the
    # required set moves), the same fallback as an empty regenerated answer.
    try:
        new_answer = await _judge(
            ANSWER_UPDATOR_PROMPT.format(
                previous_gold_answer=question.gold_answer, reference_documents=reference, query=question.question
            )
        )
    except httpx.HTTPError:
        new_answer = ""
    if not new_answer:
        return question.model_copy(update={"expected_doc_ids": required})
    facts: list[str] = []
    guards: list[str] = []
    try:
        parsed = _json(await _judge(FACT_EXTRACTION_PROMPT.format(question=question.question, gold_answer=new_answer)))
        facts = [str(f) for f in parsed] if isinstance(parsed, list) else []
        parsed = _json(
            await _judge(ANTI_HALLUCINATION_FACT_VALIDATOR_PROMPT.format(fact_list=json.dumps(question.answer_facts)))
        )
        guards = [str(f) for f in parsed] if isinstance(parsed, list) else []
    except (ValueError, httpx.HTTPError):  # unparsable or unreachable: keep the original facts
        pass
    return question.model_copy(
        update={
            "expected_doc_ids": required,
            "gold_answer": new_answer,
            "answer_facts": (facts + [g for g in guards if g not in facts]) or question.answer_facts,
        }
    )


# --- one question -----------------------------------------------------------------------------


class QuestionScore(BaseModel):
    question_id: str
    question_type: str
    correct: bool
    correctness_reason: str
    completeness_pct: float
    #: ``None`` for the questions with no gold document (high-level, info-not-found).
    recall_pct: float | None
    invalid_extra_docs: int | None
    corrected: bool


async def score(
    question: Question, answer: str, document_ids: Sequence[str], *, correction: bool = True
) -> QuestionScore:
    cited = list(dict.fromkeys(document_ids))
    answer = await strip_citations(answer) if answer else ""
    effective, valid = question, []
    gold = list(question.expected_doc_ids)
    if correction and gold and cited and set(cited) != set(gold) and (set(cited) - set(gold)) & corpus_members().keys():
        # An id the agent mistyped names no document: it cannot be judged, and stays an invalid
        # extra. (Upstream refuses the whole answers file instead.)
        known = corpus_members()
        candidates = [d for d in cited if d not in gold and d in known]
        if len(candidates) > 20:
            candidates = random.sample(candidates, 20)
        fix = await _consensus(question.question, gold, candidates)
        if fix is not None:
            valid = fix.valid
            effective = await _regenerate(question, fix.required)

    expected = set(effective.expected_doc_ids)
    recall = len(set(cited) & expected) / len(expected) * 100 if expected else None
    extras = len(set(cited) - expected - set(valid)) if expected else None

    if not answer:
        correct, reason, completeness = False, "no answer", 0.0
    else:
        verdict, facts = await asyncio.gather(
            _correct(effective.question, effective.gold_answer, answer),
            asyncio.gather(*(_fact(answer, f) for f in effective.answer_facts), return_exceptions=True),
        )
        correct, reason = verdict.correct, verdict.reason
        if any(isinstance(f, BaseException) for f in facts):
            completeness = 0.0  # upstream: a failed fact check zeroes the question's completeness
        else:
            completeness = 100 * sum(bool(f) for f in facts) / len(facts) if facts else 100.0
    return QuestionScore(
        question_id=question.question_id,
        question_type=question.question_type,
        correct=correct,
        correctness_reason=reason,
        completeness_pct=round(completeness, 2),
        recall_pct=round(recall, 2) if recall is not None else None,
        invalid_extra_docs=extras,
        corrected=effective is not question,
    )


def aggregate(scores: Sequence[QuestionScore]) -> dict[str, float]:
    """The leaderboard's columns (overall, correctness, completeness, recall, invalid extras),
    then the overall score per category."""

    def block(rows: Sequence[QuestionScore]) -> dict[str, float]:
        recalls = [r.recall_pct for r in rows if r.recall_pct is not None]
        extras = [r.invalid_extra_docs for r in rows if r.invalid_extra_docs is not None]
        return {
            "overall": round(statistics.fmean(r.completeness_pct if r.correct else 0.0 for r in rows), 2),
            "correctness": round(100 * statistics.fmean(r.correct for r in rows), 2),
            "completeness": round(statistics.fmean(r.completeness_pct for r in rows), 2),
            "recall": round(statistics.fmean(recalls), 2) if recalls else 0.0,
            "invalid_extra_docs": round(statistics.fmean(extras), 2) if extras else 0.0,
        }

    out = block(scores)
    for qtype in sorted({s.question_type for s in scores}):
        out[f"{qtype}:overall"] = block([s for s in scores if s.question_type == qtype])["overall"]
    out["corrected_questions"] = float(sum(s.corrected for s in scores))
    return out
