"""Observation tree: a bank's observations laid out as a bounded tree of folders and files.

No LLM. Observations are grouped by the entities they are about (rare entities weigh more,
ids and near-ubiquitous entities are ignored), split recursively with spherical k-means until a
group is small enough to be one file, and every folder is named after the entities that set it
apart from its siblings. Each topic is one markdown file listing every observation in it, with
dates and source counts, so a grep over the tree finds everything. Each folder holds a
``index.md``: what the folder is about, its topics, and its best-supported observations.

The tree is an Open Knowledge Format bundle (https://okf.md/spec/, v0.2): every topic file opens
with YAML frontmatter (``type``, ``title``, ``description``, ``tags``, ``timestamp``,
``generated``), folder overviews use the reserved ``index.md`` (frontmatter only on the root,
declaring ``okf_version``), and links are bundle-relative (``/gemma/vllm.md``).
"""

import json
import math
import posixpath
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from .. import __version__

# A group at or under this size becomes one leaf.
LEAF_MAX = 120
# Groups smaller than this are not a topic of their own: they go to the folder's other.
LEAF_MIN = 15
BRANCH = 8
MAX_DEPTH = 6
SUMMARY_TOP = 10
RELATED_TOP = 3
# An entity on more than this share of the bank's observations names nothing.
UBIQUITOUS_SHARE = 0.33
# Two seed entities sharing more than this share of their rows (Jaccard) are one subject.
SEED_OVERLAP = 0.5
# A row with no seed entity joins a group only if the group carries this share of its entity weight.
SHARE_TO_JOIN = 0.25
# What fits no topic in a folder; split by month, then ISO week, once it outgrows LEAF_MAX.
MISC_NAME = "misc"
# The one topic file of a bank too small to split.
LONE_NAME = "all"
INDEX = "index.md"
# OKF reserves index.md and log.md: no topic may take those names.
_RESERVED = frozenset({"index", "log"})
OKF_VERSION = "0.2"
OKF_TYPE = "Memory Topic"

# Speaker roles the extractor attaches to almost everything.
_ROLE_ENTITIES = {"user", "agent", "assistant"}
# PR/issue numbers, hashes, uuids, tenant/pod ids, and namespaced labels (knowledge:decision).
_ID_LIKE = re.compile(r"#\d|^(pr|issue|gh)[\s#-]*\d+$|^\d+$|(?=[0-9a-f]*\d)\b[0-9a-f]{7,}\b|:|\d+\.\d+\.\d+\.\d+")


@dataclass
class TreeObservation:
    id: str
    text: str
    proof_count: int
    entities: list[str]
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    updated_at: datetime | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TreeFile:
    path: str
    name: str
    # "index" (a folder's overview, OKF's reserved index.md) or "topic" (one topic, every
    # observation in it).
    kind: Literal["index", "topic"]
    title: str
    observation_count: int
    about: list[str]
    # Paths of related topic files elsewhere in the tree.
    related: list[str]
    # The observations the file lists, in the order it lists them.
    observations: list[TreeObservation]
    # The same file as markdown, for grep and for mounting as a filesystem.
    content: str


@dataclass
class TreeFolder:
    path: str
    name: str
    title: str
    observation_count: int
    folders: list["TreeFolder"] = field(default_factory=list)
    files: list[TreeFile] = field(default_factory=list)


@dataclass
class _Node:
    members: np.ndarray  # indices into the observation list
    centroid: np.ndarray | None
    children: list["_Node"] = field(default_factory=list)
    top_entities: list[str] = field(default_factory=list)
    slug: str = ""
    path: str = ""
    # The entity the group was seeded from: its path segment, since it moves only if the
    # group itself does (the most distinctive entities can swap order on a small change).
    seed: str = ""
    # The folder's ``other``: rows that fit none of its topics.
    rest: bool = False


def _useful(name: str) -> bool:
    low = name.strip().lower()
    return bool(low) and low not in _ROLE_ENTITIES and not _ID_LIKE.search(low)


def _slug(text: str) -> str:
    """Letters, digits and combining marks of any script, lowercased and joined by dashes
    (北京 and हिन्दी stay whole: an ASCII-only slug turned every non-Latin name into topic,
    topic-2, ..., and treating combining marks as separators split Devanagari apart)."""
    kept = "".join(c if unicodedata.category(c)[0] in "LNM" else "-" for c in text.lower())
    return re.sub(r"-+", "-", kept)[:40].strip("-") or "topic"


@dataclass
class _Features:
    """Sparse TF-IDF over entities, one row per observation, rows L2-normalised."""

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    vocab: list[str]
    n: int

    def subset(self, members: np.ndarray) -> "_Features":
        keep = np.isin(self.rows, members)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[members] = np.arange(len(members))
        return _Features(remap[self.rows[keep]], self.cols[keep], self.vals[keep], self.vocab, len(members))

    def reweighted(self) -> "_Features":
        """Re-weight by rarity *inside this group*, so what the whole group shares (the
        reason it is a group) stops gluing it together and the split follows what differs."""
        df = np.bincount(self.cols, minlength=len(self.vocab))
        vals = self.vals * np.log(self.n / np.maximum(df[self.cols], 1))
        norms = np.sqrt(np.bincount(self.rows, weights=vals**2, minlength=self.n))
        vals = vals / np.where(norms == 0, 1, norms)[self.rows]
        return _Features(self.rows, self.cols, vals, self.vocab, self.n)

    def sims(self, centroids: np.ndarray) -> np.ndarray:
        """Cosine similarity of every row to every centroid, shape (n, k)."""
        out = np.empty((self.n, len(centroids)))
        for c, centre in enumerate(centroids):
            out[:, c] = np.bincount(self.rows, weights=self.vals * centre[self.cols], minlength=self.n)
        return out

    def centroids(self, labels: np.ndarray, k: int) -> np.ndarray:
        cent = np.zeros((k, len(self.vocab)))
        np.add.at(cent, (labels[self.rows], self.cols), self.vals)
        norms = np.linalg.norm(cent, axis=1, keepdims=True)
        return cent / np.where(norms == 0, 1, norms)

    def dense_row(self, i: int) -> np.ndarray:
        row = np.zeros(len(self.vocab))
        mask = self.rows == i
        row[self.cols[mask]] = self.vals[mask]
        return row


@dataclass
class _BankFeatures:
    feats: _Features
    # Vocabulary indices of each observation's entities (the rows of ``feats``, unweighted).
    obs_cols: list[list[int]]


def _build_features(observations: list[TreeObservation]) -> _BankFeatures:
    n = len(observations)
    per_obs = [sorted({e for e in o.entities if _useful(e)}) for o in observations]
    df = Counter(e for ents in per_obs for e in ents)
    vocab = sorted(e for e, c in df.items() if c >= 2 and c <= UBIQUITOUS_SHARE * n)
    index = {e: i for i, e in enumerate(vocab)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    obs_cols: list[list[int]] = []
    for i, ents in enumerate(per_obs):
        cs = [index[e] for e in ents if e in index]
        obs_cols.append(cs)
        if not cs:
            continue
        weights = np.array([math.log(n / df[vocab[c]]) for c in cs])
        weights /= np.linalg.norm(weights)
        rows.extend([i] * len(cs))
        cols.extend(cs)
        vals.extend(weights.tolist())
    return _BankFeatures(
        _Features(np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(vals), vocab, n),
        obs_cols,
    )


def _seed_entities(feats: _Features, k: int, exclude: frozenset[int] = frozenset()) -> list[int]:
    """Up to ``k`` entities to start the groups from: the strongest entities in the group (by
    their summed weight, so an entity on nearly every row counts for little), skipping any that
    mostly co-occurs with one already chosen. Ties go to the name, so the choice depends only on
    counts, which a few observations more or less barely move. (Random k-means++ seeding re-rolled
    every group as soon as one row was added, removed or reordered: ~3% of a real bank's
    observations kept their file across a 1% change. Greedy coverage seeding stopped at one
    seed when one entity sat on most rows, leaving a 497-observation file.)"""
    mass = np.bincount(feats.cols, weights=feats.vals, minlength=len(feats.vocab))
    df = np.bincount(feats.cols[feats.vals > 0], minlength=len(feats.vocab))
    rows_of = {}
    chosen: list[int] = []
    candidates = [c for c in np.flatnonzero(df >= LEAF_MIN) if int(c) not in exclude]
    for c in sorted(candidates, key=lambda c: (-round(mass[c], 9), feats.vocab[c])):
        rows = set(feats.rows[(feats.cols == c) & (feats.vals > 0)].tolist())
        if any(len(rows & rows_of[o]) / len(rows | rows_of[o]) > SEED_OVERLAP for o in chosen):
            continue
        rows_of[int(c)] = rows
        chosen.append(int(c))
        if len(chosen) == k:
            break
    return chosen


def _assign(feats: _Features, seeds: list[int]) -> np.ndarray:
    """A label per row: ``c`` for the group grown from ``seeds[c]``, ``len(seeds)`` for the rest.

    A row goes to the seed entity it weighs most. A row carrying no seed joins the group that
    shares the largest part of its entity weight (counting entities at least two of the group's
    rows carry), if that part is at least ``SHARE_TO_JOIN``; otherwise it goes to the rest, the
    folder's ``other``. (Placing it by similarity to the group's mean put it next to the nearest
    group however far: median similarity 0.03, so a fact about deleting a document's request
    history landed in a topic about CLAUDE.md system stories.) No k-means rounds after this:
    each round let a small change in the bank ripple through every group."""
    k = len(seeds)
    cent = np.zeros((k, len(feats.vocab)))
    cent[np.arange(k), seeds] = 1.0
    sims = feats.sims(cent)
    labels = np.where(sims.max(axis=1) > 0, sims.argmax(axis=1), k)
    loose = np.flatnonzero(labels == k)
    if len(loose) and len(loose) < feats.n:
        # Entities each seeded group carries on at least two rows.
        counts = np.zeros((k, len(feats.vocab)))
        placed = labels[feats.rows] < k
        np.add.at(counts, (labels[feats.rows][placed], feats.cols[placed]), 1)
        carried = counts >= 2
        total = np.bincount(feats.rows, weights=feats.vals, minlength=feats.n)
        shared = np.zeros((feats.n, k))
        for g in range(k):
            shared[:, g] = np.bincount(feats.rows, weights=feats.vals * carried[g, feats.cols], minlength=feats.n)
        part = shared[loose] / np.where(total[loose] == 0, 1, total[loose])[:, None]
        labels[loose] = np.where(part.max(axis=1) >= SHARE_TO_JOIN, part.argmax(axis=1), k)
    return labels


def _split(feats: _Features, members: np.ndarray, depth: int, ancestors: frozenset[int] = frozenset()) -> _Node:
    sub = feats.subset(members)
    node = _Node(members=members, centroid=sub.centroids(np.zeros(sub.n, dtype=np.int64), 1)[0])
    if len(members) <= LEAF_MAX or depth >= MAX_DEPTH:
        return node
    # Ask for BRANCH groups and let the data decide how many survive: seeding stops when
    # every row already sits on a centre, and groups under LEAF_MIN fold into a neighbour.
    # (Sizing k as ceil(n / LEAF_MAX) asked a 160-observation bank with four unrelated
    # subjects for 2 groups, so files mixed subjects that share no entity.)
    local = sub.reweighted()
    # An ancestor's seed is what this whole group is about: seeding on it again only peels
    # the group into memlake/memlake/... folders.
    seeds = _seed_entities(local, BRANCH, ancestors)
    if len(seeds) < 2:
        return node
    labels = _assign(local, seeds)
    rest = len(seeds)
    # Rows that only carry what the whole group shares are about the folder, not one topic.
    blank = np.bincount(local.rows, weights=local.vals, minlength=sub.n) == 0
    labels[blank] = rest
    # A group too small to stand alone goes to the rest too, rather than into whichever group
    # is nearest (the rest itself is never folded away: it is the folder's ``other``).
    sizes = np.bincount(labels, minlength=rest + 1)
    labels[np.isin(labels, [c for c in range(rest) if sizes[c] < LEAF_MIN])] = rest
    kept = np.unique(labels)
    if len(kept) < 2:
        return node
    node.children = []
    for c in kept:
        own = frozenset([seeds[c]]) if c < rest else frozenset()
        child = _split(feats, members[labels == c], depth + 1, ancestors | own)
        child.seed = local.vocab[seeds[c]] if c < rest else ""
        child.rest = c == rest
        if child.rest and child.children:
            # The rest split into real topics: they sit beside the folder's other topics, and
            # only what still fits none stays in the folder's single other.md (a split rest
            # used to become an other/ folder holding its own other.md, other/other/other.md).
            node.children.extend(child.children)
        else:
            node.children.append(child)
    return node


def _name_children(
    node: _Node, obs_cols: list[list[int]], vocab: list[str], taken: frozenset[str] = frozenset()
) -> None:
    """Name each child after the entities that are common in it but rarer in its parent.

    ``taken`` are the names its ancestors already use, so a child never repeats its parent."""
    parent_df = Counter(c for i in node.members for c in obs_cols[i])
    parent_n = len(node.members)
    used: set[str] = set()
    for child in node.children:
        df = Counter(c for i in child.members for c in obs_cols[i])
        n = len(child.members)
        scored = sorted(
            ((df[c] / n) * math.log((df[c] / n) / (parent_df[c] / parent_n)), vocab[c])
            for c in df
            if df[c] >= 2 and vocab[c] not in taken
        )
        child.top_entities = [name for score, name in reversed(scored) if score > 0][:5]
        if not child.top_entities and not child.rest:
            # Nothing sets it apart from its parent (a bank small enough for one topic file):
            # title it after its most common entities (its path stays LONE_NAME).
            common = sorted(((-df[c], vocab[c]) for c in df if df[c] >= 2 and vocab[c] not in taken))
            child.top_entities = [name for _, name in common[:5]]
        if child.rest:
            slug = MISC_NAME
        elif child.seed:
            slug = _slug(child.seed)
        else:
            # Only a bank small enough for one topic file has an unseeded topic. A fixed name
            # keeps its path when a subject is added (naming it after its first entities
            # renamed the file, and moved every observation, each time one tied in).
            slug = LONE_NAME
        if not child.rest and (slug in _RESERVED or slug == MISC_NAME or slug.startswith(f"{MISC_NAME}-")):
            # A topic named index would overwrite the folder's index.md; misc-* would read as leftovers.
            slug = f"{slug}-topic"
        base, i = slug, 2
        while slug in used:
            slug, i = f"{base}-{i}", i + 1
        used.add(slug)
        child.slug = slug
        child.path = posixpath.join(node.path, slug)
        _name_children(child, obs_cols, vocab, taken | set(child.top_entities[:2]))


def _leaves(node: _Node) -> list[_Node]:
    return [node] if not node.children else [leaf for c in node.children for leaf in _leaves(c)]


def _stamp(d: datetime | None) -> str:
    """Date and time in UTC, the way observation notes write them. (A day alone was off by
    one: a fact "on the 18th" in UTC+2 is stored as 22:00 on the 17th, so the line said the
    17th while the note in the same text said 2026-09-18 07:46.)"""
    return d.astimezone(UTC).strftime("%Y-%m-%d %H:%M") if d else ""


def _shown(o: TreeObservation) -> datetime | None:
    return o.occurred_start or o.occurred_end or o.mentioned_at


def _when(o: TreeObservation) -> str:
    """When it happened if known, else when it was mentioned."""
    start, end = _stamp(o.occurred_start), _stamp(o.occurred_end)
    if start and end and start != end:
        return f"{start}..{end}"
    return start or end or _stamp(o.mentioned_at)


def _line(o: TreeObservation) -> str:
    text = " ".join(o.text.split())
    meta = [m for m in (_when(o), f"{o.proof_count} source{'' if o.proof_count == 1 else 's'}") if m]
    return f"- {text} ({' · '.join(meta)})"


def _file_path(node: _Node) -> str:
    # The root is always a folder, even with no topics (an empty bank), so its overview is index.md.
    return node.path + ".md" if node.path and not node.children else posixpath.join(node.path, INDEX)


def _title(node: _Node) -> str:
    return " · ".join(node.top_entities[:2]) or node.slug


def _ranked(node: _Node, observations: list[TreeObservation]) -> list[TreeObservation]:
    return [
        observations[i]
        for i in sorted(node.members, key=lambda i: (-observations[i].proof_count, observations[i].text))
    ]


def _link(node: _Node) -> str:
    """A bundle-relative link, as OKF recommends."""
    return "/" + _file_path(node)


def _latest(node: _Node, observations: list[TreeObservation]) -> datetime | None:
    stamps = [d for i in node.members if (d := observations[i].updated_at or observations[i].mentioned_at)]
    return max(stamps) if stamps else None


def _frontmatter(node: _Node, observations: list[TreeObservation], title: str) -> list[str]:
    """OKF frontmatter. Strings are JSON-quoted, which YAML reads as-is. The timestamps are the
    newest observation's, not the request's, so the same bank gives the same bundle."""
    about = node.top_entities
    description = f"{len(node.members)} observations" + (f" about {', '.join(about[:3])}." if about else ".")
    lines = [
        "---",
        f"type: {OKF_TYPE}",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        f"tags: {json.dumps(about, ensure_ascii=False)}",
    ]
    latest = _latest(node, observations)
    if latest:
        at = latest.astimezone(UTC).isoformat()
        lines += [f"timestamp: {at}", "generated:", f"  by: hindsight/{__version__}", f"  at: {at}"]
    return [*lines, "---", ""]


def _render_topic(node: _Node, observations: list[TreeObservation], related: list[_Node]) -> TreeFile:
    title = _title(node)
    ranked = _ranked(node, observations)
    lines = [*_frontmatter(node, observations, title), f"# {title}", "", f"{len(ranked)} observations. Times are UTC."]
    if node.top_entities:
        lines += ["", "**About:** " + ", ".join(node.top_entities)]
    if related:
        lines += ["", "**Related:** " + ", ".join(f"[{r.path}]({_link(r)})" for r in related)]
    lines += ["", "## Observations", ""] + [_line(o) for o in ranked]
    return TreeFile(
        path=_file_path(node),
        name=posixpath.basename(_file_path(node)),
        kind="topic",
        title=title,
        observation_count=len(ranked),
        about=node.top_entities,
        related=[_file_path(r) for r in related],
        observations=ranked,
        content="\n".join(lines) + "\n",
    )


def _render(
    node: _Node,
    observations: list[TreeObservation],
    related: dict[str, list[_Node]],
    title: str,
) -> TreeFolder:
    ranked = _ranked(node, observations)
    folder = TreeFolder(path=node.path, name=node.slug, title=title, observation_count=len(ranked))
    children = sorted(node.children, key=lambda c: -len(c.members))
    # OKF's index.md carries no frontmatter, except the root's okf_version.
    lines = ["---", f'okf_version: "{OKF_VERSION}"', "---", ""] if not node.path else []
    lines += [f"# {title}", "", f"{len(ranked)} observations in {len(children)} topics."]
    if node.top_entities:
        lines += ["", "**About:** " + ", ".join(node.top_entities)]
    lines += ["", "## Topics", ""]
    for c in children:
        about = f"; about {', '.join(c.top_entities[:3])}" if c.top_entities else ""
        lines.append(f"* [{_title(c)}]({_link(c)}) - {len(c.members)} observations{about}")
    lines += ["", "## Best supported", ""] + [_line(o) for o in ranked[:SUMMARY_TOP]]
    folder.files.append(
        TreeFile(
            path=_file_path(node),
            name=INDEX,
            kind="index",
            title=title,
            observation_count=len(ranked),
            about=node.top_entities,
            related=[],
            observations=ranked[:SUMMARY_TOP],
            content="\n".join(lines) + "\n",
        )
    )
    for c in children:
        if c.children:
            folder.folders.append(_render(c, observations, related, _title(c)))
        else:
            folder.files.append(_render_topic(c, observations, related.get(c.path, [])))
    return folder


def _split_misc(node: _Node, observations: list[TreeObservation]) -> None:
    """Split every misc file over LEAF_MAX into one file per month of its observations' dates
    (and a month over LEAF_MAX into ISO weeks). Calendar buckets, not misc-1/misc-2: an
    observation's month or week never changes, so it never moves file."""
    children: list[_Node] = []
    for child in node.children:
        if child.rest and not child.children and len(child.members) > LEAF_MAX:
            by_month: dict[str, list[int]] = {}
            for i in child.members:
                d = _shown(observations[i])
                by_month.setdefault(d.astimezone(UTC).strftime("%Y-%m") if d else "undated", []).append(int(i))
            # A month still over LEAF_MAX splits once more, by ISO week; no further.
            buckets: dict[str, list[int]] = {}
            for month, ids in by_month.items():
                if len(ids) <= LEAF_MAX or month == "undated":
                    buckets[month] = ids
                    continue
                for i in ids:
                    d = _shown(observations[i])
                    week = d.astimezone(UTC).strftime("%G-w%V") if d else "undated"
                    buckets.setdefault(week, []).append(i)
            for key in sorted(buckets):
                slug = f"{MISC_NAME}-{key}"
                children.append(
                    _Node(
                        members=np.array(buckets[key]),
                        centroid=None,
                        top_entities=child.top_entities,
                        slug=slug,
                        path=posixpath.join(node.path, slug),
                        rest=True,
                    )
                )
        else:
            _split_misc(child, observations)
            children.append(child)
    node.children = children


def build_observation_tree(observations: list[TreeObservation], bank_title: str = "Memories") -> TreeFolder:
    """Lay ``observations`` out as a tree. Deterministic: the same input gives the same tree.

    Rebuilt on every call (a stored tree would need one per tag scope), so stability comes from
    the build itself: no randomness, groups seeded from entity counts, paths named after the
    seed entity. A small change to the bank leaves most observations in their file.
    """
    # One canonical order, so the tree depends on which observations there are, not on the
    # order the store returned them in.
    observations = sorted(observations, key=lambda o: o.id)
    bank = _build_features(observations)
    feats, obs_cols = bank.feats, bank.obs_cols
    with_features = np.array([i for i, cs in enumerate(obs_cols) if cs], dtype=np.int64)
    without = [i for i, cs in enumerate(obs_cols) if not cs]
    root = _Node(members=np.arange(len(observations)), centroid=None)
    if len(with_features):
        clustered = _split(feats, with_features, 0)
        root.children = clustered.children or [clustered]
    _name_children(root, obs_cols, feats.vocab)
    if without:
        rest = next((c for c in root.children if c.rest and not c.children), None)
        if rest:  # one other.md at the root, not two
            rest.members = np.concatenate([rest.members, np.array(without)])
        else:
            root.children.append(
                _Node(members=np.array(without), centroid=None, slug=MISC_NAME, path=MISC_NAME, rest=True)
            )

    _split_misc(root, observations)
    leaves = [leaf for leaf in _leaves(root) if leaf.centroid is not None and leaf is not root]
    related: dict[str, list[_Node]] = {}
    if len(leaves) > 1:
        cent = np.array([leaf.centroid for leaf in leaves])
        sims = cent @ cent.T
        np.fill_diagonal(sims, -1)
        for i, leaf in enumerate(leaves):
            # Siblings are already listed in the parent's summary; link across branches instead.
            parent = posixpath.dirname(leaf.path)
            order = [j for j in np.argsort(-sims[i]) if sims[i, j] > 0 and posixpath.dirname(leaves[j].path) != parent][
                :RELATED_TOP
            ]
            related[leaf.path] = [leaves[j] for j in order]
    return _render(root, observations, related, bank_title)


def iter_files(folder: TreeFolder) -> Iterator[TreeFile]:
    yield from folder.files
    for sub in folder.folders:
        yield from iter_files(sub)
