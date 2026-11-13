"""Hermes plugin: search and read a Hindsight knowledge bank.

Two tools, served by the knowledge-bank API:

* ``knowledge_search`` — the bank's search: meaning, with keyword matches (BM25 weights)
  fused in by the server, hits grouped per document, best first;
* ``knowledge_read`` — a document's original text, by id, in pages.

A third, ``knowledge_grep`` (the search's exact-text ``pattern``, as a regular expression), went
with that parameter: no retrieval benchmark used it.

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
import sys
import time
from dataclasses import dataclass
from urllib.parse import quote

import httpx

TOOLSET = "hindsight_knowledge"
#: A search's token budget (what its hits, and the text around them, may weigh), and its bounds.
SEARCH_TOKENS = int(os.getenv("HINDSIGHT_KNOWLEDGE_SEARCH_TOKENS", "4096"))
MIN_SEARCH_TOKENS, MAX_SEARCH_TOKENS = 500, 32_000
READ_CHARS = int(os.getenv("HINDSIGHT_KNOWLEDGE_READ_CHARS", "12000"))


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
    candidate with its passages, not three. The server fits them, and the text around each, in
    ``max_tokens``; nothing is cut here."""
    cfg = _config()
    body: dict[str, object] = {
        "query": args["query"],
        "max_tokens": max(MIN_SEARCH_TOKENS, min(int(args.get("max_tokens") or SEARCH_TOKENS), MAX_SEARCH_TOKENS)),
    }
    if args.get("tags"):
        body["tags"] = args["tags"]
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
    for document_id, hits in by_document.items():
        first = hits[0]
        tags = ",".join(first.get("tags") or [])
        lines.append(f"{document_id} | {first.get('filename') or ''} | tags={tags}")
        lines.extend((hit.get("text") or "").strip() for hit in hits)
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
                "max_tokens": {
                    "type": "integer",
                    "description": "How much text to get back in all, in tokens (500-32000, default 4096): the "
                    "best passages and the text around them, as much as fits.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only documents with one of these tags (the source system: slack, gmail, "
                    "linear, google_drive, hubspot, fireflies, github, jira, confluence).",
                },
            },
            "required": ["query"],
        },
    },
    "knowledge_read": {
        "name": "knowledge_read",
        "description": (
            "Read a document's full original text by its document id (from knowledge_search). Long documents come in pages: pass the offset the header gives to continue."
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
}

HANDLERS = {"knowledge_search": knowledge_search, "knowledge_read": knowledge_read}


def register(ctx) -> None:  # noqa: ANN001 - Hermes' PluginContext, not importable from here
    """Hermes plugin entry point."""
    for name, schema in SCHEMAS.items():
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=HANDLERS[name], emoji="📚")
