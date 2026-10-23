"""Markdown rendering for knowledge pages.

Knowledge pages render as *read-only* markdown documents over the existing mental
models: each mental model becomes a markdown body with a YAML frontmatter block
(``type`` required; ``title``/``description``/``tags``/``timestamp`` optional).

This module is intentionally pure: every function transforms the mental-model
dicts returned by ``MemoryEngine.list_mental_models`` / ``get_mental_model`` and
never touches the database. That keeps rendering unit-testable without a DB or
LLM and lets the HTTP layer stay a thin wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Every page carries exactly one ``type`` frontmatter field. We default to this
# when a page does not declare one via a ``type:<x>`` tag.
DEFAULT_PAGE_TYPE = "knowledge-page"

#: What a reader is told when a page has no body yet. A page with nothing under its
#: title is a state to act on, and blankness does not say so — an agent reads it as a
#: page that failed to render rather than one nobody has written, and writes a second
#: page for the same topic.
#:
#: Presentation only. It is produced when a page is READ and never stored, because the
#: stored body is embedded and BM25-indexed: a marker in the row would make every
#: unwritten page searchable as its own marker, which is the defect this replaced.
#: Plain words, not markdown emphasis — a snippet is printed rather than parsed, so
#: underscores would reach the reader as underscores.
EMPTY_PAGE_NOTICE = "No content yet."

# A page declares its ``type`` through a tag of the form ``type:runbook``.
# This keeps rendering schema-free (no new mental_models column): the type is
# lifted from the existing tags array.
TYPE_TAG_PREFIX = "type:"

INDEX_FILENAME = "index.md"


@dataclass(frozen=True)
class PageType:
    """A page's ``type`` and the tags that remain after the type tag is split off."""

    type: str
    display_tags: list[str]


def _scalar(value: Any) -> str:
    """Emit a YAML-safe double-quoted scalar.

    We always double-quote so arbitrary page names / source queries can't be
    misread as YAML special forms (``true``, ``2026-01-01``, ``- x``, etc.).
    """
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
    return f'"{escaped}"'


def page_type(tags: list[str] | None) -> PageType:
    """Split a ``type`` out of the tag list.

    The first ``type:<x>`` tag wins; all ``type:`` tags are removed from the
    returned ``display_tags`` so they don't leak into the page's displayed tags.
    Falls back to :data:`DEFAULT_PAGE_TYPE`.
    """
    resolved = DEFAULT_PAGE_TYPE
    display: list[str] = []
    for tag in tags or []:
        if tag.startswith(TYPE_TAG_PREFIX):
            suffix = tag[len(TYPE_TAG_PREFIX) :].strip()
            if suffix and resolved == DEFAULT_PAGE_TYPE:
                resolved = suffix
            continue
        display.append(tag)
    return PageType(type=resolved, display_tags=display)


def _timestamp(mm: dict[str, Any]) -> str | None:
    return mm.get("last_refreshed_at") or mm.get("created_at")


def frontmatter(mm: dict[str, Any]) -> dict[str, Any]:
    """Build the ordered frontmatter mapping for a mental model.

    ``None``/empty values are dropped by :func:`render_frontmatter`.
    """
    pt = page_type(mm.get("tags"))
    return {
        "id": mm.get("id"),
        "type": pt.type,
        "title": mm.get("name"),
        "description": mm.get("source_query"),
        "tags": pt.display_tags,
        "timestamp": _timestamp(mm),
    }


def render_frontmatter(fm: dict[str, Any]) -> str:
    """Render a frontmatter mapping into a ``---`` fenced YAML block."""
    lines = ["---"]
    for key, value in fm.items():
        if value is None:
            continue
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"{key}:")
            lines.extend(f"  - {_scalar(item)}" for item in value)
        else:
            lines.append(f"{key}: {_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def render_document(mm: dict[str, Any], *, notice_when_empty: bool = False) -> str:
    """Render a full markdown document: frontmatter block + markdown body.

    ``notice_when_empty`` puts :data:`EMPTY_PAGE_NOTICE` in place of a missing body,
    for the surfaces a person or an agent READS — where frontmatter and nothing else
    looks like a rendering failure rather than an unwritten page. It is off by default
    because the same renderer writes the export bundle, and a notice there would be
    indistinguishable from real content the next time the bundle is read back in.
    """
    body = (mm.get("content") or "").strip()
    if not body and notice_when_empty:
        body = EMPTY_PAGE_NOTICE
    return f"{render_frontmatter(frontmatter(mm))}\n\n{body}\n" if body else f"{render_frontmatter(frontmatter(mm))}\n"


def page_filename(page_id: str) -> str:
    """Bundle filename for a page id."""
    return f"{page_id}.md"


def log_filename(page_id: str) -> str:
    """Reserved per-page history filename."""
    return f"{page_id}.log.md"


def render_index(nodes: list[dict[str, Any]]) -> str:
    """Render the reserved ``index.md`` — nested markdown navigation over the tree.

    ``nodes`` is the flat folder/page list (each with ``id``, ``kind``, ``name``,
    ``parent_id``); folders nest their children, pages link to their ``.md``.
    """
    fm = render_frontmatter({"type": "index", "title": "Knowledge base"})
    lines = [fm, "", "# Knowledge base", ""]

    children: dict[Any, list[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(node.get("parent_id"), []).append(node)

    def walk(parent: Any, depth: int) -> None:
        ordered = sorted(children.get(parent, []), key=lambda n: (n.get("sort_order", 0), n.get("name") or ""))
        for node in ordered:
            indent = "  " * depth
            if node.get("kind") == "folder":
                lines.append(f"{indent}- **{node['name']}/**")
                walk(node["id"], depth + 1)
            else:
                description = node.get("source_query") or node.get("description")
                link = f"{indent}- [{node['name']}](./{page_filename(node['id'])})"
                lines.append(f"{link} — {description}" if description else link)

    walk(None, 0)
    if len(lines) == 4:
        lines.append("_No knowledge pages yet._")
    return "\n".join(lines) + "\n"


def render_log(mm: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """Render the reserved per-page ``log.md`` from refresh history.

    Each history entry is ``{previous_content, previous_reflect_response,
    changed_at}`` (newest first), capturing the content *before* a refresh.
    """
    name = mm.get("name") or mm.get("id")
    fm = render_frontmatter({"type": "log", "title": f"{name} — history"})
    lines = [fm, "", f"# {name} — history", ""]
    if not history:
        lines.append("_No refresh history._")
        return "\n".join(lines) + "\n"
    for entry in history:
        changed_at = entry.get("changed_at") or "unknown"
        previous = (entry.get("previous_content") or "").strip()
        lines.append(f"## {changed_at}")
        lines.append("")
        lines.append(previous if previous else "_(empty)_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
