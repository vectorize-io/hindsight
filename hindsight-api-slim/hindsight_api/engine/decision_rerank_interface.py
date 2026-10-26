"""What recall's decision reranker sends to a decision model, and how it reads the answers.

This file is a public interface, meant to be used outside hindsight-api too: a service
that reranks for Hindsight over the System One API can load it and send exactly the
questions recall's :class:`~hindsight_api.engine.search.decision_reranker.DecisionModelReranker`
sends, so a pool comes back in the order recall itself would give it.

To keep it loadable on its own, by file path, without installing hindsight-api:

- it imports the standard library only, never the rest of ``hindsight_api``;
- the caller brings the tokenizer, as ``count`` and ``truncate`` functions, so each
  caller counts in the vocabulary of the model it talks to;
- it builds and reads plain System One JSON; sending it is the caller's job.

Changing a name, a signature or the wording here changes what external callers send,
so treat it as an API: keep it backwards compatible, or say so in the release notes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

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


class DecisionRerank:
    """Builds one rerank's System One requests and turns the answers into scores.

    ``count(text)`` and ``truncate(text, max_tokens)`` are the tokenizer of the model the
    requests go to; ``max_request_tokens`` is the most tokens one request may hold.
    """

    # A plain class, not a dataclass: a dataclass looks its module up in sys.modules,
    # which a caller loading this file by path need not have registered.
    def __init__(
        self,
        count: Callable[[str], int],
        truncate: Callable[[str, int], str],
        max_request_tokens: int = MAX_REQUEST_TOKENS,
    ) -> None:
        self.count = count
        self.truncate = truncate
        self.max_request_tokens = max_request_tokens

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

    def questions(self, docs: list[str]) -> dict[str, dict[str, Any]]:
        """One Noul per candidate, keyed ``m0``, ``m1``, ... in the order of ``docs``.

        Each memory is capped like the query, so no single question can outgrow the window.
        """
        return {
            f"m{slot}": {
                "type": "noul",
                "instructions": {"task": KEEP_INSTRUCTIONS, "memory": self.capped(doc, MAX_MEMORY_TOKENS)},
            }
            for slot, doc in enumerate(docs)
        }

    def split(
        self, model: str, state: str | dict[str, Any], questions: Mapping[str, dict[str, Any]]
    ) -> list[list[str]]:
        """The keys of ``questions``, in groups whose requests each fit ``max_request_tokens``.

        Counted on the System One body (``{"model", "state", "questions"}``), since that is
        what the endpoint counts against its window. At most ``MAX_QUESTIONS`` per group. A
        single question over the ceiling still goes out alone, and fails there.
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
    def scores(n: int, nouls: Mapping[str, float | None]) -> list[float]:
        """Scores for the ``n`` candidates, aligned with ``docs``, from each key's Noul answer.

        The scores are positions, not confidences: the best gets 1.0 and each place below
        it 1/n less. An unanswered candidate counts as 0.0. Equal answers keep the input
        order, so the caller's own order breaks ties.
        """

        def answer(slot: int) -> float:
            noul = nouls.get(f"m{slot}")
            return noul if noul is not None else 0.0

        order = sorted(range(n), key=lambda slot: -answer(slot))
        scores = [0.0] * n
        for position, slot in enumerate(order):
            scores[slot] = (n - position) / n
        return scores
