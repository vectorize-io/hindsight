"""Recall's reranker when a decision model is set up for recall.

It asks the decision model of each candidate separately whether showing it would help
someone answer the question, orders the pool by that, then drops the candidates a
confident verdict says are out of date. The order is final: recall applies none of the
recency, temporal or proof-count boosts it applies to a cross-encoder's scores.

This is not the ``typesafe`` reranker provider (``HINDSIGHT_API_RERANKER_PROVIDER``),
which is configured on its own and ranks differently. This one is configured by
``HindsightConfig.decision_model_config("recall")`` and talks to the endpoint through the
shared :class:`~hindsight_api.engine.decision_model.DecisionModel` client.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from dataclasses import dataclass

from ...config import DecisionModelConfig
from ..cross_encoder import CrossEncoderModel
from ..decision_model import DecisionModel, DecisionQuestion
from ..token_encoding import count_tokens, truncate_to_tokens

PROVIDER_NAME = "decision_model"

# The context every candidate in this rerank shares — the bank's own framing, which recall
# would otherwise paste onto each one. The decision model gets it in the question state
# once instead: repeated per candidate it cannot say which candidate beats which, it makes
# any overlap comparison between two of them read as near-identical, and it spends the
# window on the same text once per candidate. Set by CrossEncoderReranker.rerank().
rerank_shared_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hindsight_rerank_shared_context", default=None
)

# Defensive ceilings on the query and the shared context, which every request carries:
# past the request ceiling, every request would fail. Each also stays within a quarter of
# that ceiling, so a low one still leaves half of each request for the questions.
_MAX_QUERY_TOKENS = 2_000
_MAX_SHARED_CONTEXT_TOKENS = 4_000
_CONFLICT_OPTION_TOKENS = 300  # Per-candidate ceiling in a conflict question
# Recall prepends this per candidate, so it is held aside while the pool's shared
# preamble is found and put back afterwards.
_DATE_PREFIX = re.compile(r"^\[Date: [^\]]*\]\s*")

# The rank question, asked of each candidate. Four wordings were tried on the same pool;
# this one is the best calibrated, lifting the wanted candidate to 0.74 against 0.59 for
# "contains information needed to answer" while holding the pool median at 0.08. Framing it
# as help-versus-distract asks the question a keep/drop decision actually turns on.
_KEEP_INSTRUCTIONS = "Showing this memory to someone answering the question would help them, rather than distract them."

# Conflict resolution. Ranking candidates against the question is a different thing from
# noticing that two of them answer it differently. Where one candidate revises another,
# ranking both highly leaves the caller holding the stale value alongside the current one —
# and a generator reading them follows the majority, so a corrected fact ranked first but
# outnumbered by stale restatements does not change the answer. The question only ever asks
# one thing: shown candidates that disagree, which states the CURRENT answer. Judged
# together, because a candidate read on its own cannot be seen to revise anything.
#
# Strict wording only. Adding "when they are merely different details, pick the one that
# answers the question most directly" inverted the verdict on the measured pair: it chose
# the stale candidate, which echoes the question's own wording, at confidence 0.40-0.60
# against 0.78-0.80 for the strict form.
_CONFLICT_INSTRUCTIONS = (
    "These memories give different answers to the question. Which states the CURRENT "
    "answer? When they conflict, the most recently stated value supersedes the earlier one."
)

# Only questions whose answer IS a value can turn on a supersession.
_VALUE_QUESTION = re.compile(
    r"\b(when|what date|deadline|how many|how much|how long|which version|"
    r"what time|how often|total|count|number of)\b",
    re.I,
)
_MONTH = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b")
# Pronouns and framing words carry no subject, so they must not count as a shared term.
_QUESTION_STOPWORDS = frozenset(
    "what when which who how many much long does did your their that this with from have "
    "been will would about there here they them".split()
)


def _comparable(text: str) -> str:
    """A candidate stripped of the metadata recall prefixes, for comparing it to another.

    The date is not content: two candidates are not restatements of each other because they
    happened on the same day, and the date's own digits must not reach a value payload or
    "2024" counts as the number the question turns on. It stays on the text the model is
    asked about, where it is what lets a verdict prefer the more recent candidate.
    """
    return _DATE_PREFIX.sub("", text)


def _normalised_tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", "", _comparable(text).lower()).split())


def _token_overlap(a: set[str], b: set[str]) -> float:
    """Shared tokens over the SHORTER candidate, so a terse revision still matches."""
    return len(a & b) / max(1, min(len(a), len(b))) if a and b else 0.0


@dataclass(frozen=True)
class _ValuePayload:
    """The numbers and months a value question turns on."""

    numbers: frozenset[int]
    months: frozenset[str]

    def states(self, value: _ValuePayload) -> bool:
        """Whether this candidate states ``value``, possibly alongside more detail."""
        return value.numbers <= self.numbers and value.months <= self.months


def _value_payload(text: str) -> _ValuePayload:
    """Years are dropped, so "April 1, 2024" and "2024-04-01" compare on day and month
    rather than agreeing because they share 2024."""
    low = _comparable(text).lower()
    numbers = {int(n) for n in re.findall(r"\b(\d{1,4})\b", low) if int(n) < 1900}
    return _ValuePayload(frozenset(numbers), frozenset(_MONTH.findall(low)))


def _strip_shared_prefix(docs: list[str], indices: list[int]) -> dict[int, str]:
    """Each candidate's own text, with any preamble every candidate shares removed.

    Recall prefixes a candidate with its bank context, which for a single-conversation
    bank is the same block of profile text on every one of them. Text identical across
    the whole pool cannot discriminate between its members, and leaving it in breaks the
    comparisons here: candidates sharing 60 tokens of boilerplate clear any overlap
    threshold, and a value payload picks up numbers out of the preamble rather than the
    fact. It also costs the window the same boilerplate once per candidate, which on a
    300-candidate pool is most of it.

    Only whole lines are stripped, so a prefix is never cut mid-sentence, and only when
    every candidate in the pool agrees on them. The date recall prepends is taken off
    first and put back after: it is per-candidate, so leaving it in front would mean the
    pool shares no prefix at all and nothing would be stripped.
    """
    if len(indices) < 2:
        return {index: docs[index] for index in indices}
    marked = [_DATE_PREFIX.match(docs[index]) for index in indices]
    dates = [match.group(0) if match else "" for match in marked]
    bodies = [docs[index][len(date) :] for index, date in zip(indices, dates)]

    split = [body.split("\n") for body in bodies]
    shared = 0
    for lines in zip(*split):
        if len(set(lines)) > 1:
            break
        shared += 1
    if not shared:
        return {index: docs[index] for index in indices}
    stripped = {}
    for index, date, lines in zip(indices, dates, split):
        own = "\n".join(lines[shared:]).strip()
        stripped[index] = f"{date}{own}" if own else docs[index]
    return stripped


def _subject_terms(query: str) -> set[str]:
    """The query's own content words: a conflict counts only if both candidates are
    about what was ASKED, not merely about each other. Most candidates carry some
    incidental number, and without this the trigger fires almost everywhere."""
    return {w for w in re.findall(r"[a-z0-9']+", query.lower()) if len(w) > 3 and w not in _QUESTION_STOPWORDS}


class DecisionModelReranker(CrossEncoderModel):
    """Ranks a recall pool with a decision model, then resolves conflicts in it.

    **Rank — one Noul per candidate**, asking whether showing it would help rather than
    distract. Every candidate lands on one absolute scale, so the order holds all the way
    down to where the caller's budget cuts. Asking one comparative Choice over the whole
    pool instead ranks the head better (recall@1 0.94 against 0.87) but not the tail, and
    where the caller keeps dozens of memories for a generator it measured far worse end to
    end: 0.59 against 0.79 on a 20-question answer-quality set.

    **Conflicts — one Choice per disagreeing cluster**, asking which states the current
    answer. Only on questions whose answer is itself a value, among candidates that overlap
    heavily, differ on a number or date, and share the question's own subject terms. Losers
    are dropped rather than moved down, because a generator reading one correction among
    several stale restatements follows the majority.

    The scores handed back are positions, not confidences: descending from 1.0 in rank
    order, and exactly 0.0 for a superseded candidate, which :attr:`prunes_candidates`
    tells the caller to leave out.
    """

    prunes_candidates = True

    # There is deliberately no text-overlap de-duplication. Overlap is a guess about what a
    # candidate says, not a judgement that it is untrue, and it read a standing instruction
    # phrased like its neighbours as a restatement of them. Removing a candidate on that
    # basis cost 0.040 of 0.67 over 100 questions. Demoting instead does not help either:
    # the caller's budget cuts from the back, so moving a candidate there removes it just
    # the same. Supersession is the only thing that takes a candidate out, on a verdict.

    # Overlap at which two candidates are "the same thing with a different value". There is
    # deliberately no upper bound: the pairs that most need resolving are the ones that
    # overlap almost completely and differ on one value.
    CONFLICT_OVERLAP = 0.6
    # Conflicts are looked for around the top-ranked candidates only.
    CONFLICT_ANCHORS = 25
    # Both candidates must share this many of the question's own subject terms. Without it
    # the trigger fired on 94 of 100 questions and deleted 7.5 candidates on average —
    # indiscriminate pruning rather than conflict resolution.
    CONFLICT_MIN_SUBJECT_TERMS = 2
    # Most options in one conflict question. A crowded conflict is the case that matters
    # most: a stale value restated ten ways against one correction is exactly what makes a
    # generator answer with the stale one. So an oversized cluster is narrowed to its
    # best-ranked members rather than discarded.
    CONFLICT_MAX_CLUSTER = 25
    # Below this, the model is not resolving a conflict: wrong picks came back at 0.40-0.60
    # against about 0.80 for right ones, on a two-to-five option question where confidence
    # is not diluted by the option count.
    CONFLICT_MIN_CONFIDENCE = 0.7

    def __init__(self, client: DecisionModel):
        self._client = client

    @classmethod
    def from_config(cls, config: DecisionModelConfig) -> DecisionModelReranker:
        return cls(DecisionModel.from_config(config))

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    async def initialize(self) -> None:
        pass

    def _capped(self, text: str, ceiling: int) -> str:
        ceiling = min(ceiling, self._client.max_request_tokens // 4)
        return truncate_to_tokens(text, ceiling).text if count_tokens(text) > ceiling else text

    def _state(self, query: str) -> str:
        """The question's state: what is being asked, plus the framing every candidate shares."""
        query = self._capped(query, _MAX_QUERY_TOKENS)
        shared = self._capped(rerank_shared_context.get() or "", _MAX_SHARED_CONTEXT_TOKENS)
        return f"{shared}\n\nQuestion: {query}" if shared else f"Question: {query}"

    async def _rank(self, query: str, docs: list[str], indices: list[int]) -> list[int]:
        """``indices`` best first. A candidate whose request failed has no answer, scores
        0.0 and so falls behind every scored one, keeping its input order there: an
        unranked candidate is recoverable, a deleted one is not."""
        questions = {
            f"m{slot}": DecisionQuestion(type="noul", instructions={"task": _KEEP_INSTRUCTIONS, "memory": docs[index]})
            for slot, index in enumerate(indices)
        }
        answers = await self._client.ask_all(self._state(query), questions)

        def score(slot: int) -> float:
            answer = answers.get(f"m{slot}")
            return answer.noul if answer is not None and answer.noul is not None else 0.0

        # Stable on the caller's input order, so equal scores keep their RRF order.
        order = sorted(range(len(indices)), key=lambda slot: -score(slot))
        return [indices[slot] for slot in order]

    def _conflict_clusters(self, query: str, docs: list[str], order: list[int]) -> list[list[int]]:
        """Candidates that overlap heavily but disagree on a value, grouped around each anchor.

        Each cluster is one anchor plus its DIRECT peers, and nothing else. Grouping
        transitively does not work here: single-linkage at a 0.35 overlap floor collapsed a
        whole 80-candidate pool into one component, and no threshold separates cleanly — at
        0.60 the conflicting pair groups but drags 66 candidates in with it, at 0.70 and
        above the chaining stops but the pair itself splits apart. Anchoring on each
        top-ranked candidate avoids chaining entirely.
        """
        if not _VALUE_QUESTION.search(query):
            return []
        tokens = {index: _normalised_tokens(docs[index]) for index in order}
        payloads = {index: _value_payload(docs[index]) for index in order}
        subject = _subject_terms(query)

        def about_the_question(index: int) -> bool:
            return not subject or len(tokens[index] & subject) >= self.CONFLICT_MIN_SUBJECT_TERMS

        clusters: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for anchor in order[: self.CONFLICT_ANCHORS]:
            if not about_the_question(anchor):
                continue
            peers = [
                index
                for index in order
                if index != anchor
                and payloads[index] != payloads[anchor]
                and _token_overlap(tokens[anchor], tokens[index]) >= self.CONFLICT_OVERLAP
                and about_the_question(index)
            ]
            if not peers:
                continue
            # Keep the anchor and, when there are more peers than one question can hold,
            # the best-ranked of them: those are the ones a wrong verdict would cost most.
            if len(peers) + 1 > self.CONFLICT_MAX_CLUSTER:
                rank = {index: position for position, index in enumerate(order)}
                peers = sorted(peers, key=lambda index: rank[index])[: self.CONFLICT_MAX_CLUSTER - 1]
            cluster = tuple(sorted([anchor, *peers]))
            if cluster not in seen:
                seen.add(cluster)
                clusters.append(list(cluster))
        return clusters

    async def _superseded(self, query: str, docs: list[str], order: list[int]) -> set[int]:
        """The candidates a confidently-resolved conflict says are out of date."""
        clusters = self._conflict_clusters(query, docs, order)
        if not clusters:
            return set()

        questions = {
            f"conflict{number}": DecisionQuestion(
                type="choice",
                instructions=_CONFLICT_INSTRUCTIONS,
                criteria={
                    f"c{slot}": truncate_to_tokens(docs[index], _CONFLICT_OPTION_TOKENS).text
                    for slot, index in enumerate(cluster)
                },
            )
            for number, cluster in enumerate(clusters)
        }
        # A conflict whose request failed stays unresolved: both candidates are returned,
        # which is what recall did before. Failing the whole rerank would be worse.
        answers = await self._client.ask_all(self._state(query), questions)

        tokens = {index: _normalised_tokens(docs[index]) for index in order}
        payloads = {index: _value_payload(docs[index]) for index in order}
        subject = _subject_terms(query)
        # A verdict reaches beyond its own question, but only to candidates some cluster
        # already identified as conflicting peers. Reaching the whole pool would widen
        # deletion on the strength of one answer.
        conflicting = {index for cluster in clusters for index in cluster}

        superseded: set[int] = set()
        for number, cluster in enumerate(clusters):
            answer = answers.get(f"conflict{number}")
            if answer is None or not answer.choice or (answer.confidence or 0.0) < self.CONFLICT_MIN_CONFIDENCE:
                continue
            try:
                winner = cluster[int(answer.choice[1:])]
            except (ValueError, IndexError):
                continue
            # A confident verdict says which value is current, so everything that disagrees
            # with the winner on it is stale — not only the candidates in this question. The
            # same value is often restated many times and only some land in one cluster;
            # leaving the rest behind leaves the majority stale. The test is the one that
            # built the cluster, around the winner, except that a candidate restating the
            # winner's value with more detail around it agrees with it and is kept.
            superseded.update(
                index
                for index in conflicting
                if index != winner
                and not payloads[index].states(payloads[winner])
                and _token_overlap(tokens[winner], tokens[index]) >= self.CONFLICT_OVERLAP
                and (not subject or len(tokens[index] & subject) >= self.CONFLICT_MIN_SUBJECT_TERMS)
            )
        return superseded

    async def _rerank(self, query: str, docs: list[str], indices: list[int], scores: list[float]) -> None:
        # Judge each candidate on what is its own, not on the preamble they all carry.
        own = _strip_shared_prefix(docs, indices)
        docs = [own.get(index, doc) for index, doc in enumerate(docs)]

        order = await self._rank(query, docs, indices)
        superseded = await self._superseded(query, docs, order)
        # Never return nothing: one candidate the caller can dismiss beats silence.
        order = [index for index in order if index not in superseded] or order[:1]
        # Positions, descending from 1.0 so the caller keeps this order. The superseded keep
        # the 0.0 they started with, which prunes_candidates tells the caller to leave out.
        for position, index in enumerate(order):
            scores[index] = (len(order) - position) / len(order)

    async def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        docs = [doc for _, doc in pairs]
        by_query: dict[str, list[int]] = {}
        for index, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(index)
        scores = [0.0] * len(pairs)
        await asyncio.gather(*(self._rerank(query, docs, indices, scores) for query, indices in by_query.items()))
        return scores
