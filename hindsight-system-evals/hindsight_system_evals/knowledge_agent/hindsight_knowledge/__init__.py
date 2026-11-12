"""Hermes plugin: search, read and grep a Hindsight knowledge bank.

Three tools, the loop PipesHub's agent runs on EnterpriseRAG-Bench (search, exact-match
lookup, full-document reading), all served by the knowledge-bank API:

* ``knowledge_search`` — the bank's search: meaning, with keyword matches (BM25 weights)
  fused in by the server, hits grouped per document;
* ``knowledge_read`` — a document's original text, by id, in pages;
* ``knowledge_grep`` — the bank's grep: a case-insensitive regular expression over every
  passage's text, documents with the most matches first. An exact identifier ("ENG-4512",
  "stream.timebox_finalized") is what meaning search ranks worst.

The plugin used to mirror the whole bank to disk and run its own BM25 and ripgrep over the
copy; the server does both now, and the plugin's BM25 blend found the gold documents no more
often than the server's search alone (EnterpriseRAG, 50 questions: 36 vs 35 in the top 4).

Configured from the environment, so the same file works as an installed plugin
(``$HERMES_HOME/plugins/hindsight_knowledge``) and when imported by the eval's runner:
``HINDSIGHT_KNOWLEDGE_API_URL``, ``HINDSIGHT_KNOWLEDGE_API_KEY``, ``HINDSIGHT_KNOWLEDGE_BANK``.

It runs inside Hermes' venv, so it uses only what that venv has (httpx) and nothing from
this repo.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

TOOLSET = "hindsight_knowledge"
#: One search hit's text in a result. The hit plus its neighbours can run to several
#: thousand characters; a model that wants more reads the document.
HIT_CHARS = int(os.getenv("HINDSIGHT_KNOWLEDGE_HIT_CHARS", "1200"))
READ_CHARS = int(os.getenv("HINDSIGHT_KNOWLEDGE_READ_CHARS", "12000"))
GREP_SNIPPET_CHARS = 240
#: Passages a grep asks the server for; grouped into documents here.
GREP_PASSAGES = 200


@dataclass(frozen=True)
class _Bank:
    base: str
    bank: str
    headers: dict[str, str]


def _config() -> _Bank:
    url = os.environ["HINDSIGHT_KNOWLEDGE_API_URL"].rstrip("/")
    bank = os.environ["HINDSIGHT_KNOWLEDGE_BANK"]
    key = os.getenv("HINDSIGHT_KNOWLEDGE_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return _Bank(f"{url}/v1/default/knowledge-banks/{quote(bank, safe='')}", bank, headers)


_CLIENT = httpx.Client(timeout=120)


#: Seconds between retries of a failed call: about four minutes in all. A remote embedder
#: behind a port-forward or a rolling restart is out for minutes, not seconds; a search that
#: gives up sooner hands the agent "search failed" and the question is lost.
_BACKOFF = (1, 2, 4, 8, 15, 30, 30, 30, 30, 30, 30)


def _request(method: str, url: str, **kwargs: object) -> httpx.Response:
    """One API call, retried on a 5xx or a dropped connection; a call that still fails is
    logged to stderr so a run can tell a degraded answer from a wrong one."""
    for wait in (*_BACKOFF, None):
        try:
            response = _CLIENT.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if wait is None:
                print(f"knowledge tool failed: {method} {url}: {exc}", file=sys.stderr)
                raise
        else:
            if response.status_code < 500 or wait is None:
                if response.status_code >= 500:
                    print(f"knowledge tool failed: {method} {url}: {response.status_code}", file=sys.stderr)
                return response
        time.sleep(wait)
    raise AssertionError("unreachable")


def knowledge_search(args: dict, **_: object) -> str:
    """Hits grouped by document, best first: a document that matches in three places is one
    candidate with snippets, not three of the ten slots."""
    cfg = _config()
    top_k = max(1, min(int(args.get("top_k") or 10), 50))
    body: dict[str, object] = {
        "query": args["query"],
        "top_k": min(200, top_k * 3),
        "neighbors": max(0, min(int(args.get("neighbors", 1)), 5)),
    }
    for key in ("tags", "media_types"):
        if args.get(key):
            body[key] = args[key]
    response = _request("POST", f"{cfg.base}/search", json=body, headers=cfg.headers)
    if response.status_code >= 400:
        return f"search failed ({response.status_code}): {response.text[:300]}"
    by_document: dict[str, list[dict]] = {}
    for hit in response.json()["results"]:
        by_document.setdefault(hit["document_id"], []).append(hit)
    if not by_document:
        return "No results."
    lines = []
    # Each hit is headed by its document id alone, never a list number: numbered hits ("[1] dsid_…")
    # got cited as "Sources: [1]" in 26 of 50 answers, which scores as citing nothing.
    for document_id, hits in list(by_document.items())[:top_k]:
        first = hits[0]
        tags = ",".join(first.get("tags") or [])
        lines.append(f"{document_id} | {first.get('filename') or ''} | tags={tags} | score={first['score']:.3f}")
        budget = HIT_CHARS
        for hit in hits[:2]:
            text = (hit.get("context") or hit.get("text") or "").strip()
            lines.append(text[:budget] + (" …" if len(text) > budget else ""))
            budget = HIT_CHARS // 2
        lines.append("")
    return "\n".join(lines)


def knowledge_read(args: dict, **_: object) -> str:
    cfg = _config()
    base, headers = cfg.base, cfg.headers
    document_id = args["document_id"].strip()
    response = _request("GET", f"{base}/documents/{quote(document_id, safe='')}/file", headers=headers)
    if response.status_code == 404:
        return f"No document {document_id!r}."
    if response.status_code >= 400:
        return f"read failed ({response.status_code}): {response.text[:300]}"
    text = response.text
    offset = max(0, int(args.get("offset") or 0))
    page = text[offset : offset + READ_CHARS]
    header = f"{document_id} — characters {offset}-{offset + len(page)} of {len(text)}"
    if offset + len(page) < len(text):
        header += f" (more: offset={offset + len(page)})"
    return f"{header}\n\n{page}"


def knowledge_grep(args: dict, **_: object) -> str:
    """The bank's exact-text search (search with only a pattern), grouped by document (most matches
    first), with
    each document's first matching lines."""
    cfg = _config()
    pattern = args["pattern"]
    limit = max(1, min(int(args.get("max_results") or 15), 50))
    body = {"pattern": pattern, "regex": True, "top_k": GREP_PASSAGES, "neighbors": 0}
    response = _request("POST", f"{cfg.base}/search", json=body, headers=cfg.headers)
    if response.status_code == 400 and "regular expression" in response.text:
        # A model's "regex" with an unbalanced bracket is meant literally.
        response = _request("POST", f"{cfg.base}/search", json={**body, "regex": False}, headers=cfg.headers)
    if response.status_code >= 400:
        return f"grep failed ({response.status_code}): {response.text[:300]}"
    by_document: dict[str, list[dict]] = {}
    for hit in response.json()["results"]:
        by_document.setdefault(hit["document_id"], []).append(hit)
    if not by_document:
        return f"No document matches {pattern!r}."
    try:
        matcher = re.compile(pattern, re.IGNORECASE)
    except re.error:
        matcher = re.compile(re.escape(pattern), re.IGNORECASE)
    lines = [f"{len(by_document)} documents match {pattern!r}; top {min(limit, len(by_document))} by matches:", ""]
    for document_id, hits in list(by_document.items())[:limit]:
        lines.append(f"{document_id} | {hits[0].get('filename') or ''} | {int(hits[0]['score'])} matching passages")
        shown = [line.strip() for hit in hits for line in (hit.get("text") or "").splitlines() if matcher.search(line)][
            :2
        ]
        for line in shown:
            lines.append("    " + (line[:GREP_SNIPPET_CHARS] + " …" if len(line) > GREP_SNIPPET_CHARS else line))
    return "\n".join(lines)


SCHEMAS = {
    "knowledge_search": {
        "name": "knowledge_search",
        "description": (
            "Search the company knowledge base by meaning. Returns the best-matching passages, each with "
            "its document id, filename (the source system and folder, e.g. slack/..., jira/..., "
            "confluence/...), tags and surrounding text. Phrase the query the way the answer would be "
            "written. Run several searches with different wordings for one question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "top_k": {"type": "integer", "description": "How many passages (1-50, default 10)."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only documents with one of these tags (the source system: slack, gmail, "
                    "linear, google_drive, hubspot, fireflies, github, jira, confluence).",
                },
                "neighbors": {
                    "type": "integer",
                    "description": "Passages of context on each side of a hit (0-5, default 1).",
                },
            },
            "required": ["query"],
        },
    },
    "knowledge_read": {
        "name": "knowledge_read",
        "description": (
            "Read a document's full original text by its document id (from knowledge_search or "
            "knowledge_grep). Long documents come in pages: pass the offset the header gives to continue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "offset": {"type": "integer", "description": "Character offset to start at (default 0)."},
            },
            "required": ["document_id"],
        },
    },
    "knowledge_grep": {
        "name": "knowledge_grep",
        "description": (
            "Find documents containing an exact string or regular expression (case-insensitive), e.g. an "
            "identifier, code name, ticket key, metric name, person's name or number. Returns matching "
            "document ids ranked by match count, with the matching lines. Use alternation for variants: "
            "'timebox|time-box'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression (case-insensitive)."},
                "max_results": {"type": "integer", "description": "Documents to return (1-50, default 15)."},
            },
            "required": ["pattern"],
        },
    },
}

HANDLERS = {"knowledge_search": knowledge_search, "knowledge_read": knowledge_read, "knowledge_grep": knowledge_grep}


def register(ctx) -> None:  # noqa: ANN001 - Hermes' PluginContext, not importable from here
    """Hermes plugin entry point."""
    for name, schema in SCHEMAS.items():
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=HANDLERS[name], emoji="📚")
