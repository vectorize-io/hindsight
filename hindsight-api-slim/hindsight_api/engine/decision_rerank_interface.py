"""What recall's decision reranker sends to a decision model, and how it reads the answers.

This file is a public interface, meant to be used outside hindsight-api too: a service
that reranks for Hindsight over the System One API can load it and send exactly the
questions recall's :class:`~hindsight_api.engine.search.decision_reranker.DecisionModelReranker`
sends, so a pool comes back in the order recall itself would give it. With pruning on
(:meth:`DecisionRerank.pruned`), it drops the same candidates recall would, and with dedup
on (:meth:`DecisionRerank.repeat_questions`, :meth:`DecisionRerank.deduped`) the same
repeats.

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
from collections.abc import Callable, Collection, Mapping, Sequence
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
# Pruning drops no repeats: that is dedup, below, and it costs a second request.
PRUNE_BELOW = 0.2
PRUNE_BELOW_BEST = 0.4

# Dedup, opt-in: dropping a candidate that repeats one ranked above it. It needs both memories
# in one question, and which pairs matter is only known once the ranking answers are in, so it
# is a second request, after the ranking one. Among the candidates still in the pool, a pair is
# asked (both memories in the question, the better-ranked one as Memory A, one way round: the
# order is known by then) when the caller says the two are alike — recall passes the pairs
# whose stored embeddings have a cosine of ``ALIKE_FROM`` or more. Without that (the caller
# has no embeddings), every pair is asked. Either way at most ``MAX_REPEAT_PAIRS`` are asked,
# best-ranked first.
#
# ``ALIKE_FROM`` only narrows the pairs asked; the model decides. The lowest cosine of a true
# repeat in the labelled pools was 0.74 with the server's default local embedder and 0.69 with a
# multilingual one ("Rate limit on the public API: 100 req/min" vs "Clients may send at most 100
# requests a minute to the public API"), so 0.65 keeps a margin under both. It assumes a bank
# holds one language: a translated copy scores far lower (0.40), and is not looked for. An
# embedder that packs every text close together passes more pairs, never fewer repeats.
#
# ``MAX_REPEAT_PAIRS``: the pairs grow with the square of the candidates kept, and with a
# single-language embedder alike-but-distinct facts all sit above ``ALIKE_FROM`` ("Lena owns the
# billing service" vs "Mei owns the billing service" score 0.9+, true repeats 0.83-0.99), so the
# threshold barely narrows them. Uncapped, one recall kept 99 alike memories and asked 4,851
# pairs and 347k tokens. 100 pairs is about 7k tokens, one request at about 0.25s (p90 0.30s);
# in the earlier banks of 250 candidates a recall never asked more than 153 pairs even
# uncapped, median 6. Best-ranked first, because a repeat near the top costs the reader most.
# On a 1,147-memory English bank crowded with alike facts (22 recalls) with pruning on, the cap
# was hit in 11 recalls: best-ranked first still dropped 33/38 repeats with 1 fact wrongly
# dropped, where most-alike-first dropped 29/38 and wrongly dropped 7 (the most alike pairs
# are the near-identical templates the model most often calls repeats). Asking every alike
# pair dropped 38/38 but wrongly dropped 10. With pruning off, the cap reaches about the 15th
# candidate: every repeat in the top 20 dropped (35/35), 1 wrongly dropped.
#
# History: the pairs were first those joined by a semantic link retain stored. Retain keeps
# only a memory's 50 nearest above 0.7, so in a bank crowded with alike facts a repeat is often
# not linked: on a 1,109-memory bank, links caught 46/48 repeats where cosine caught 48/48.
# Ruled out: picking pairs before the ranking answers, to share the ranking request (26/40: a
# kept memory can rank as low as 199 by fusion, so only 76% of repeat pairs had both in the top
# 40); deciding by cosine alone (distinct pairs reach 0.96); character overlap; one choice over
# the whole pool, or the pool in the state (both lose needed facts); capping each candidate to
# its 3 or 5 nearest above it (missed 10/48 and 5/48: the nearest is often another copy that
# goes itself, so the copy is never asked against the one that stays).
DEDUP_INSTRUCTIONS = "Memory B is a repeat of Memory A."
REPEAT_ABOVE = 0.5
ALIKE_FROM = 0.65
MAX_REPEAT_PAIRS = 100
_DATE_PREFIX = re.compile(r"^\[Date: [^\]]*\]")


def _date(doc: str) -> str | None:
    match = _DATE_PREFIX.match(doc)
    return match.group(0) if match else None


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
    """Dedup's question: does the candidate at ``slot`` repeat the one at ``above``?

    ``above`` is ranked better, so it is Memory A. Both memories ride in the question, so
    every pair goes in one request under the ranking state.
    """

    key: str
    above: int
    slot: int
    memory_a: str
    memory_b: str

    def system_one(self) -> dict[str, object]:
        """The question as System One takes it: a Noul with the task and both memories."""
        return {
            "type": "noul",
            "instructions": {"task": DEDUP_INSTRUCTIONS, "memory_a": self.memory_a, "memory_b": self.memory_b},
        }


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

    def repeat_questions(
        self,
        docs: Sequence[str],
        scores: Sequence[float],
        kinds: Sequence[str] | None = None,
        alike: Collection[frozenset[int]] | None = None,
    ) -> list[RepeatQuestion]:
        """Dedup's questions: pairs of candidates still in the pool, the better-ranked as A.

        ``scores`` are :meth:`scores` or :meth:`pruned`; a candidate at 0.0 is already out.
        ``docs`` are the uncapped texts, capped here as in the ranking questions. ``alike``
        holds the pairs of slots the caller knows to be alike (recall: cosine of the stored
        embeddings at least ``ALIKE_FROM``); with it only those pairs are asked, and without it
        (``None``: the caller has no embeddings) every pair is. At most ``MAX_REPEAT_PAIRS``
        are asked, best-ranked first: a candidate is compared with those above it before the
        next one down is. Never asked: pairs dated differently — "went climbing" on the 3rd and
        on the 10th are two occasions, and the model calls those repeats — and, when ``kinds``
        gives each candidate's fact type, pairs of two types: an observation is built from the
        very facts it repeats, and choosing between them is recall's ``prefer_observations``.
        """
        checked = DecisionRerank._ranked(scores)
        # A generator: with pruning off, every pair of 300 candidates would be built only to be cut.
        pairs = (
            (above, slot)
            for position, slot in enumerate(checked)
            for above in checked[:position]
            if alike is None or frozenset((above, slot)) in alike
        )
        questions: list[RepeatQuestion] = []
        for above, slot in pairs:
            if len(questions) == MAX_REPEAT_PAIRS:
                break
            if _date(docs[above]) and _date(docs[slot]) and _date(docs[above]) != _date(docs[slot]):
                continue
            if kinds is not None and kinds[above] != kinds[slot]:
                continue
            questions.append(
                RepeatQuestion(
                    key=f"r{above}_{slot}",
                    above=above,
                    slot=slot,
                    memory_a=self.capped(docs[above], MAX_MEMORY_TOKENS),
                    memory_b=self.capped(docs[slot], MAX_MEMORY_TOKENS),
                )
            )
        return questions

    @staticmethod
    def deduped(
        scores: Sequence[float], repeats: Sequence[RepeatQuestion], answers: Mapping[str, float | None]
    ) -> list[float]:
        """``scores`` with every repeat at exactly 0.0; ``answers`` maps each question's key.

        Greedy in rank order: a candidate goes only when it repeats one that stays, so of two
        repeats the better-ranked one is kept, and a chain does not drop its own head.
        """
        said_repeat = {(q.above, q.slot) for q in repeats if (answers.get(q.key) or 0.0) >= REPEAT_ABOVE}
        kept: list[int] = []
        dropped: set[int] = set()
        for slot in DecisionRerank._ranked(scores):
            if any((above, slot) in said_repeat for above in kept):
                dropped.add(slot)
            else:
                kept.append(slot)
        return [0.0 if slot in dropped else score for slot, score in enumerate(scores)]

    @staticmethod
    def _ranked(scores: Sequence[float]) -> list[int]:
        # The slots still in the pool, best first.
        return sorted((slot for slot, score in enumerate(scores) if score > 0.0), key=lambda slot: -scores[slot])
