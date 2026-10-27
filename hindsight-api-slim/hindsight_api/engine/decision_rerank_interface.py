"""What recall's decision reranker sends to a decision model, and how it reads the answers.

This file is a public interface, meant to be used outside hindsight-api too: a service
that reranks for Hindsight over the System One API can load it and send exactly the
questions recall's :class:`~hindsight_api.engine.search.decision_reranker.DecisionModelReranker`
sends, so a pool comes back in the order recall itself would give it. With pruning on
(:meth:`DecisionRerank.pruned`), it drops the same candidates recall would.

To keep it loadable on its own, by file path, without installing hindsight-api:

- it imports the standard library only, never the rest of ``hindsight_api``;
- the caller brings the tokenizer, as ``count`` and ``truncate`` functions, so each
  caller counts in the vocabulary of the model it talks to;
- it builds plain System One JSON; sending it is the caller's job;
- it has no ``from __future__ import annotations``: with string annotations, a dataclass
  looks its module up in ``sys.modules``, which a caller loading this file by path need
  not have registered.

Changing a name, a signature or the wording here changes what external callers send,
so treat it as an API: keep it backwards compatible, or say so in the release notes.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# The question asked of each candidate. Four wordings were tried on the same pool; this
# one is the best calibrated, lifting the wanted candidate to 0.74 against 0.59 for
# "contains information needed to answer" while holding the pool median at 0.08. Framing it
# as help-versus-distract asks the question a keep/drop decision actually turns on.
KEEP_INSTRUCTIONS = "Showing this memory to someone answering the question would help them, rather than distract them."

# Defensive ceilings on the query and the shared context, which every request carries:
# past the request ceiling, every request would fail. Each also stays within a quarter of
# that ceiling, so a low one still leaves half of each request for the questions.
MAX_QUERY_TOKENS = 2_000
MAX_SHARED_CONTEXT_TOKENS = 4_000
# The same for each candidate's text, which one question carries alone.
MAX_MEMORY_TOKENS = 4_000

# Most questions in one request. Measured on System One, ~750 single-candidate questions
# go through and 775 come back max_tokens_exceeded, so this is kept clear of the edge.
MAX_QUESTIONS = 700
# The default request ceiling, in tokens of the model's own vocabulary.
MAX_REQUEST_TOKENS = 26_000

# Pruning, opt-in: dropping candidates instead of only ordering them. A candidate is dropped
# as no help when its answer to the keep question falls below PRUNE_BELOW, or below
# PRUNE_BELOW_BEST of the best candidate's. The relative cut is needed because the answers
# move with the pool: asked alongside four memories answering at 0.9, an unrelated "staging
# cluster upgraded" came back 0.26, against 0.07 among weaker ones. It reuses the ranking
# answer, so pruning costs no extra request. Asking separately whether a memory "has nothing
# to do with the question" was worse than useless: the model answers on topic overlap and
# ignores the negation, so it dropped the needed facts and kept the junk.
#
# Repeats are not dropped. A pairwise "Memory B is a repeat of Memory A" question, both
# memories in the state, caught them, but only one pair per request: on a 250-candidate pool
# that is either a second round trip after the keep answers or hundreds of requests, and
# asked inside one request with the pool in the state it lost needed facts. With a model
# whose keep answers are sharp, the cut alone leaves a pool small enough that the repeats
# in it cost less than that check would.
PRUNE_BELOW = 0.2
PRUNE_BELOW_BEST = 0.4


@dataclass(frozen=True)
class RerankQuestion:
    """One candidate's question: would showing ``memory`` help someone answer, or distract?

    ``key`` names it in the request and in the answers; ``memory`` is the candidate's text,
    already capped.
    """

    key: str
    memory: str

    def system_one(self) -> dict[str, object]:
        """The question as System One takes it: a Noul with the task and the memory."""
        return {"type": "noul", "instructions": {"task": KEEP_INSTRUCTIONS, "memory": self.memory}}


@dataclass(frozen=True)
class DecisionRerank:
    """Builds one rerank's System One requests and turns the answers into scores.

    ``count(text)`` and ``truncate(text, max_tokens)`` are the tokenizer of the model the
    requests go to; ``max_request_tokens`` is the most tokens one request may hold.
    """

    count: Callable[[str], int]
    truncate: Callable[[str, int], str]
    max_request_tokens: int = MAX_REQUEST_TOKENS

    def capped(self, text: str, ceiling: int) -> str:
        """``text`` cut to ``ceiling`` tokens, and never past a quarter of a request."""
        ceiling = min(ceiling, self.max_request_tokens // 4)
        return self.truncate(text, ceiling) if self.count(text) > ceiling else text

    def state(self, query: str, shared_context: str | None = None) -> str:
        """The question's state: what is being asked, plus what every candidate shares.

        ``shared_context`` is what recall would otherwise paste onto every candidate (the
        current date, and the bank's own context when every candidate shares one). It goes
        here once instead: repeated per candidate, the model cannot say which candidate
        beats which, and it spends the window on the same text once per candidate.
        """
        query = self.capped(query, MAX_QUERY_TOKENS)
        shared = self.capped(shared_context or "", MAX_SHARED_CONTEXT_TOKENS)
        return f"{shared}\n\nQuestion: {query}" if shared else f"Question: {query}"

    def questions(self, docs: Sequence[str]) -> list[RerankQuestion]:
        """One question per candidate, keyed ``m0``, ``m1``, ... in the order of ``docs``.

        Each memory is capped like the query, so no single question can outgrow the window.
        """
        return [
            RerankQuestion(key=f"m{slot}", memory=self.capped(doc, MAX_MEMORY_TOKENS)) for slot, doc in enumerate(docs)
        ]

    def split(
        self, model: str, state: str | Mapping[str, object], questions: Mapping[str, Mapping[str, object]]
    ) -> list[list[str]]:
        """The keys of ``questions``, in groups whose requests each fit ``max_request_tokens``.

        ``questions`` maps each key to the question's System One JSON, so any question can
        be split, not only a :class:`RerankQuestion`. Counted on the System One body
        (``{"model", "state", "questions"}``), since that is what the endpoint counts against
        its window. At most ``MAX_QUESTIONS`` per group. A single question over the ceiling
        still goes out alone, and fails there.
        """
        envelope = self.count(json.dumps({"model": model, "state": state, "questions": {}}))
        groups: list[list[str]] = []
        current: list[str] = []
        used = envelope
        for key, question in questions.items():
            tokens = self.count(json.dumps({key: question}))
            if current and (len(current) >= MAX_QUESTIONS or used + tokens > self.max_request_tokens):
                groups.append(current)
                current, used = [], envelope
            current.append(key)
            used += tokens
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def scores(answers: Sequence[float | None]) -> list[float]:
        """Scores from each candidate's Noul answer, both aligned with ``docs``.

        The scores are positions, not confidences: the best gets 1.0 and each place below
        it 1/n less. An unanswered candidate (None) counts as 0.0. Equal answers keep the
        input order, so the caller's own order breaks ties.
        """
        n = len(answers)
        order = sorted(range(n), key=lambda slot: -(answers[slot] or 0.0))
        scores = [0.0] * n
        for position, slot in enumerate(order):
            scores[slot] = (n - position) / n
        return scores

    @staticmethod
    def pruned(answers: Sequence[float | None]) -> list[float]:
        """:meth:`scores`, with every candidate judged no help at exactly 0.0.

        No help is below ``PRUNE_BELOW``, or below ``PRUNE_BELOW_BEST`` of the best answer.
        The best candidate always stays: one weak memory beats an empty recall.
        """
        best = max((answer or 0.0 for answer in answers), default=0.0)
        floor = max(PRUNE_BELOW, PRUNE_BELOW_BEST * best)
        scores = DecisionRerank.scores(answers)
        return [score if score == 1.0 or (answer or 0.0) >= floor else 0.0 for answer, score in zip(answers, scores)]
