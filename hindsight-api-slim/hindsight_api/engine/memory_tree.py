"""Memory tree: a bank's observations laid out as a bounded tree of folders and files.

No LLM. Observations are grouped by the entities they are about (rare entities weigh more,
ids and near-ubiquitous entities are ignored), split recursively with spherical k-means until a
group is small enough to be one file, and every folder is named after the entities that set it
apart from its siblings. Each topic is one markdown file listing every observation in it, with
dates and source counts, so a grep over the tree finds everything. Each folder holds a
``README.md``: what the folder is about, its topics, and its best-supported observations.
"""

import math
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# A group at or under this size becomes one leaf.
LEAF_MAX = 120
# Groups smaller than this are folded into their nearest sibling.
LEAF_MIN = 15
BRANCH = 8
MAX_DEPTH = 6
SUMMARY_TOP = 10
RELATED_TOP = 3
# An entity on more than this share of the bank's observations names nothing.
UBIQUITOUS_SHARE = 0.33
KMEANS_ITERS = 25
OTHER_NAME = "other"
README = "README.md"

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
    # "readme" (a folder's overview) or "topic" (one topic, every observation in it).
    kind: str
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


def _useful(name: str) -> bool:
    low = name.strip().lower()
    return bool(low) and low not in _ROLE_ENTITIES and not _ID_LIKE.search(low)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "topic"


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


def _kmeans(feats: _Features, k: int, rng: np.random.Generator) -> np.ndarray:
    """Spherical k-means with k-means++ seeding. Returns a label per row."""
    first = int(rng.integers(feats.n))
    centres = [feats.dense_row(first)]
    best = feats.sims(np.array(centres))[:, 0]
    for _ in range(1, k):
        dist = np.clip(1 - best, 0, None)
        if dist.sum() == 0:
            break
        nxt = int(rng.choice(feats.n, p=dist / dist.sum()))
        centres.append(feats.dense_row(nxt))
        best = np.maximum(best, feats.sims(np.array(centres[-1:]))[:, 0])
    cent = np.array(centres)
    labels = feats.sims(cent).argmax(axis=1)
    for _ in range(KMEANS_ITERS):
        cent = feats.centroids(labels, len(cent))
        new = feats.sims(cent).argmax(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _split(feats: _Features, members: np.ndarray, depth: int, rng: np.random.Generator) -> _Node:
    sub = feats.subset(members)
    node = _Node(members=members, centroid=sub.centroids(np.zeros(sub.n, dtype=np.int64), 1)[0])
    if len(members) <= LEAF_MAX or depth >= MAX_DEPTH:
        return node
    # Ask for BRANCH groups and let the data decide how many survive: seeding stops when
    # every row already sits on a centre, and groups under LEAF_MIN fold into a neighbour.
    # (Sizing k as ceil(n / LEAF_MAX) asked a 160-observation bank with four unrelated
    # subjects for 2 groups, so files mixed subjects that share no entity.)
    local = sub.reweighted()
    labels = _kmeans(local, BRANCH, rng)
    # Rows that only carry what the whole group shares have no local signal: place them by
    # their original features instead of letting them all fall into group 0.
    blank = np.bincount(local.rows, weights=local.vals, minlength=sub.n) == 0
    if blank.any():
        labels[blank] = sub.sims(sub.centroids(labels, labels.max() + 1))[blank].argmax(axis=1)
    # Fold groups too small to stand alone into the nearest remaining group.
    while True:
        sizes = np.bincount(labels)
        small = [c for c in range(len(sizes)) if 0 < sizes[c] < LEAF_MIN]
        alive = [c for c in range(len(sizes)) if sizes[c] >= LEAF_MIN]
        if not small or not alive:
            break
        cent = sub.centroids(labels, len(sizes))
        moved = np.isin(labels, small)
        sims = sub.sims(cent[alive])
        labels[moved] = np.array(alive)[sims[moved].argmax(axis=1)]
    groups = [members[labels == c] for c in np.unique(labels)]
    if len(groups) < 2:
        return node
    node.children = [_split(feats, g, depth + 1, rng) for g in groups]
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
        slug = "-".join(_slug(e) for e in child.top_entities[:2]) or "topic"
        base, i = slug, 2
        while slug in used:
            slug, i = f"{base}-{i}", i + 1
        used.add(slug)
        child.slug = slug
        child.path = posixpath.join(node.path, slug)
        _name_children(child, obs_cols, vocab, taken | set(child.top_entities[:2]))


def _leaves(node: _Node) -> list[_Node]:
    return [node] if not node.children else [leaf for c in node.children for leaf in _leaves(c)]


def _day(d: datetime | None) -> str:
    return d.date().isoformat() if d else ""


def _when(o: TreeObservation) -> str:
    """When it happened if known, else when it was mentioned."""
    start, end = _day(o.occurred_start), _day(o.occurred_end)
    if start and end and start != end:
        return f"{start}..{end}"
    return start or end or _day(o.mentioned_at)


def _line(o: TreeObservation) -> str:
    text = " ".join(o.text.split())
    meta = [m for m in (_when(o), f"{o.proof_count} sources") if m]
    return f"- {text} ({' · '.join(meta)})"


def _file_path(node: _Node) -> str:
    return node.path + ".md" if not node.children else posixpath.join(node.path, README)


def _title(node: _Node) -> str:
    return " · ".join(node.top_entities[:2]) or node.slug


def _ranked(node: _Node, observations: list[TreeObservation]) -> list[TreeObservation]:
    return [
        observations[i]
        for i in sorted(node.members, key=lambda i: (-observations[i].proof_count, observations[i].text))
    ]


def _render_topic(node: _Node, observations: list[TreeObservation], related: list[_Node]) -> TreeFile:
    title = _title(node)
    ranked = _ranked(node, observations)
    lines = [f"# {title}", "", f"{len(ranked)} observations."]
    if node.top_entities:
        lines += ["", "**About:** " + ", ".join(node.top_entities)]
    if related:
        here = posixpath.dirname(node.path) or "."
        lines += [
            "",
            "**Related:** " + ", ".join(f"[{r.path}]({posixpath.relpath(_file_path(r), here)})" for r in related),
        ]
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
    lines = [f"# {title}", "", f"{len(ranked)} observations in {len(children)} topics."]
    if node.top_entities:
        lines += ["", "**About:** " + ", ".join(node.top_entities)]
    lines += ["", "## Topics", ""]
    for c in children:
        rel = posixpath.relpath(_file_path(c), node.path or ".")
        lines.append(f"- [{_title(c)}]({rel}): {len(c.members)} observations")
    lines += ["", "## Best supported", ""] + [_line(o) for o in ranked[:SUMMARY_TOP]]
    folder.files.append(
        TreeFile(
            path=_file_path(node),
            name=README,
            kind="readme",
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


def build_memory_tree(observations: list[TreeObservation], bank_title: str = "Memories") -> TreeFolder:
    """Lay ``observations`` out as a tree. Deterministic: the same input gives the same tree.

    # ponytail: rebuilt from scratch on every call, so paths can move when observations change.
    # Upgrade: persist leaf centroids, route new observations to the nearest leaf, split a leaf
    # only when it outgrows LEAF_MAX.
    """
    bank = _build_features(observations)
    feats, obs_cols = bank.feats, bank.obs_cols
    with_features = np.array([i for i, cs in enumerate(obs_cols) if cs], dtype=np.int64)
    without = [i for i, cs in enumerate(obs_cols) if not cs]
    rng = np.random.default_rng(0)
    root = _Node(members=np.arange(len(observations)), centroid=None)
    if len(with_features):
        clustered = _split(feats, with_features, 0, rng)
        root.children = clustered.children or [clustered]
    _name_children(root, obs_cols, feats.vocab)
    if without:
        root.children.append(_Node(members=np.array(without), centroid=None, slug=OTHER_NAME, path=OTHER_NAME))

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


def iter_files(folder: TreeFolder):
    yield from folder.files
    for sub in folder.folders:
        yield from iter_files(sub)
