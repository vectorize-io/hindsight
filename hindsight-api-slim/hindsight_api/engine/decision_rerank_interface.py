"""What recall's decision reranker sends to a decision model, and how it reads the answers.

This file is a public interface, meant to be used outside hindsight-api too: a service
that reranks for Hindsight over the System One API can load it and send exactly the
questions recall's :class:`~hindsight_api.engine.search.decision_reranker.DecisionModelReranker`
sends, so a pool comes back in the order recall itself would give it. With pruning on
(:meth:`DecisionRerank.relevant`, :meth:`DecisionRerank.repeat_questions`,
:meth:`DecisionRerank.pruned`), it drops the same candidates recall would.

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
import re
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
# cluster upgraded" came back 0.26, against 0.07 among weaker ones. On a labelled bench of
# pools mixing needed facts, related context and unrelated facts, over 6 shuffles, these two
# cuts lost no needed fact and dropped 58/132 unrelated ones (0.2 alone: 42; half the best:
# lost two). Reusing the ranking answer costs no extra question. Asking separately whether a
# memory "has nothing to do with the question" was worse than useless: the model answers on
# topic overlap and ignores the negation, so it dropped the needed facts and kept the junk.
PRUNE_BELOW = 0.2
PRUNE_BELOW_BEST = 0.4
# A candidate that repeats one kept above it is dropped. Asked of the pair with both
# memories in the state: with both in the question's instructions the model answered
# ~0.9 for any pair, related or not, and asked of one candidate against the whole list
# above it, it dropped distinct customers in a "which customers" pool as repeats. Pairwise
# it dropped 72/72 repeats and no needed fact.
REPEAT_INSTRUCTIONS = "Memory B is a repeat of Memory A."
REPEAT_ABOVE = 0.5
# Pairs sharing fewer words than this (over the shorter memory) are not asked: a reworded
# repeat still shares its names and values. On the bench it asked a quarter of the pairs
# and still caught every repeat; 0.4 missed some.
REPEAT_MIN_OVERLAP = 0.3
# Only the best candidates are checked for repeats, bounding the pairs one rerank asks.
# ponytail: a repeat ranked below this survives; raise it if long pools show repeats.
REPEAT_CANDIDATES = 40
_DATE_PREFIX = re.compile(r"^\[Date: [^\]]*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.]*")


def _date(doc: str) -> str | None:
    match = _DATE_PREFIX.match(doc)
    return match.group(0) if match else None


def _words(doc: str) -> set[str]:
    # The sentence only: the date prefix and the trailing "| When: … | Involving: …" fields
    # are shared by unrelated memories about the same day or person.
    body = _DATE_PREFIX.sub("", doc).split(" | ")[0]
    return {word for word in _WORD.findall(body.lower()) if len(word) > 2}


def _may_repeat(a: str, b: str) -> bool:
    # Two memories dated differently are two occasions — "went climbing" on the 3rd and on
    # the 10th — however alike they read; the model calls those repeats.
    if _date(a) and _date(b) and _date(a) != _date(b):
        return False
    words_a, words_b = _words(a), _words(b)
    return len(words_a & words_b) >= REPEAT_MIN_OVERLAP * max(1, min(len(words_a), len(words_b)))


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
class RepeatQuestion:
    """Pruning's second question: does the candidate at ``slot`` repeat the one at ``above``?

    Unlike the keep questions it carries its own ``state`` (both memories), so each is one
    request of its own, with the single question keyed ``repeat``.
    """

    above: int
    slot: int
    state: str

    def system_one(self) -> dict[str, object]:
        """The question as System One takes it: a Noul on the pair in the state."""
        return {"type": "noul", "instructions": REPEAT_INSTRUCTIONS}


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
    def order(answers: Sequence[float | None]) -> list[int]:
        """The candidates' slots, best answer first. Equal answers keep the input order."""
        return sorted(range(len(answers)), key=lambda slot: -(answers[slot] or 0.0))

    @staticmethod
    def scores(answers: Sequence[float | None]) -> list[float]:
        """Scores from each candidate's Noul answer, both aligned with ``docs``.

        The scores are positions, not confidences: the best gets 1.0 and each place below
        it 1/n less. An unanswered candidate (None) counts as 0.0. Equal answers keep the
        input order, so the caller's own order breaks ties.
        """
        n = len(answers)
        scores = [0.0] * n
        for position, slot in enumerate(DecisionRerank.order(answers)):
            scores[slot] = (n - position) / n
        return scores

    @staticmethod
    def relevant(answers: Sequence[float | None]) -> list[int]:
        """Pruning, step one: the slots not judged no help, best first. The best always stays."""
        order = DecisionRerank.order(answers)
        floor = max(PRUNE_BELOW, PRUNE_BELOW_BEST * (answers[order[0]] or 0.0))
        return [slot for position, slot in enumerate(order) if position == 0 or (answers[slot] or 0.0) >= floor]

    def repeat_questions(
        self, docs: Sequence[str], relevant: Sequence[int], kinds: Sequence[str] | None = None
    ) -> list[RepeatQuestion]:
        """Pruning, step two: the pairs among the best ``relevant`` worth asking about.

        ``docs`` are the uncapped candidate texts; each is capped in the state as in its keep
        question. Pairs dated differently or sharing too few words are not asked, nor, when
        ``kinds`` gives each candidate's fact type, pairs of two different types: an
        observation is built from the very facts it repeats, and choosing between them is
        recall's ``prefer_observations``, which keeps the observation. Asked as a repeat, the
        observation ranked below its fact was the one dropped, so a recall preferring
        observations got the raw fact and no observation.
        """
        checked = relevant[:REPEAT_CANDIDATES]
        return [
            RepeatQuestion(
                above=above,
                slot=slot,
                state=f"Memory A: {self.capped(docs[above], MAX_MEMORY_TOKENS)}\n"
                f"Memory B: {self.capped(docs[slot], MAX_MEMORY_TOKENS)}",
            )
            for position, slot in enumerate(checked)
            for above in checked[:position]
            if (kinds is None or kinds[above] == kinds[slot]) and _may_repeat(docs[above], docs[slot])
        ]

    @staticmethod
    def pruned(
        answers: Sequence[float | None],
        relevant: Sequence[int],
        repeats: Sequence[RepeatQuestion],
        repeat_answers: Sequence[float | None],
    ) -> list[float]:
        """Pruning, the result: :meth:`scores`, with every dropped candidate at exactly 0.0.

        Dropped are the slots not in ``relevant``, and those that repeat a candidate kept
        above them (``repeat_answers`` aligned with ``repeats``). Greedy in rank order: a
        candidate goes only when it repeats one that stays, so of two repeats the
        better-ranked one is kept.
        """
        said_repeat = {
            (q.above, q.slot) for q, a in zip(repeats, repeat_answers, strict=True) if (a or 0.0) >= REPEAT_ABOVE
        }
        kept: list[int] = []
        for position, slot in enumerate(relevant):
            if position < REPEAT_CANDIDATES and any((above, slot) in said_repeat for above in kept):
                continue
            kept.append(slot)
        scores = DecisionRerank.scores(answers)
        keep = set(kept)
        return [score if slot in keep else 0.0 for slot, score in enumerate(scores)]
