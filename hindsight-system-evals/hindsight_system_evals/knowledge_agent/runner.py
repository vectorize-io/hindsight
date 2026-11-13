"""Answer questions with a Hermes agent over a knowledge bank. Runs INSIDE Hermes' venv.

    <hermes-venv>/bin/python runner.py --questions in.jsonl --out out.jsonl

The eval (``evals/knowledge_agent``) starts this as a subprocess with Hermes' interpreter
(``HINDSIGHT_EVAL_HERMES_PYTHON``), because Hermes' pinned dependency set cannot share a venv
with this repo's packages. Nothing here imports from the repo: the two tools come from the
``hindsight_knowledge`` plugin next to this file, registered into Hermes' tool registry
exactly as its plugin loader would.

Each input line is ``{"question_id", "question"}``; each output line is the answer, the
document ids it cites, and what it cost (LLM calls, tokens, tool calls, seconds). Questions
run in parallel threads, one ``AIAgent`` each; the model is any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSID = re.compile(r"dsid_[0-9a-f]{32}")

SYSTEM_PROMPT = """\
You answer questions about the internal documents of Redwood Inference, a company that sells AI model \
inference. Every document is in a knowledge base: Slack messages, Gmail threads, Linear and Jira tickets, \
Google Drive files, HubSpot CRM records, Fireflies meeting transcripts, GitHub pull requests and Confluence \
pages. You have two tools:

- knowledge_search: finds passages by meaning. Each hit shows its document id, filename (the source system \
and folder) and the text around the match.
- knowledge_read: reads a whole document by id.

How to work (follow every step; skipping them is the most common way to get the answer wrong):
1. First call knowledge_search with the question exactly as asked.
2. Then search at least twice more with different wordings: the documents use internal jargon and project \
code names that the question paraphrases, so also try how the document itself would phrase it (its likely \
title, the team, the system, the customer).
3. Search for every specific term the question names: identifiers, code names, ticket keys, metric or \
config names, people, customers, numbers, dates.
4. Many documents look alike: drafts and updated copies, the same topic for another customer, team, region, \
version or date. knowledge_read the two to four most promising documents in full and pick the one that \
matches every qualifier in the question. A search snippet alone is never enough to answer.
5. Documents can disagree. Prefer the most recent and the most authoritative (an official doc, ticket or \
decision over a chat message or an early draft), and say what changed.
6. When the question asks for everything of a kind (all, every, list, which ones, end-to-end), collect it \
from every relevant document and keep searching until new searches return nothing new.
7. Answer only from what the documents say. Never fill a gap with general knowledge, a "standard" practice or \
a typical value. If a detail the question asks for is not in any document, say plainly that the documents \
do not state it (and give what they do say).

When you are done, write the final answer: direct, complete and specific (exact names, numbers, dates as the \
documents state them). End with one line `Sources: dsid_…, dsid_…` listing the full id of every document the \
answer uses, and only those."""

OUT_OF_BUDGET = (
    "You have used your research budget. Write the final answer now from what you found, ending with the "
    "`Sources:` line. If you did not find it, say so."
)


#: Sent once when an answer arrives after too little research (``--min-tool-calls``). A small
#: model tends to answer from the first plausible hit; the benchmark plants near-copies of
#: documents with changed facts, so the first hit is often the decoy.
VERIFY = (
    "Before this answer is final, check it. Search again with different wording, search for the specific "
    "names, numbers and identifiers in the question, and knowledge_read in full every document you rely on and "
    "any near-copy of it. Look for a document that matches the question's qualifiers more exactly or is more "
    "recent. Then write the complete final answer, ending with the `Sources:` line."
)


#: Sent after every answer with ``--check``: one self-review turn against the documents read.
CHECK = (
    "Review your answer against the documents before it is final. For every number, name, date, threshold and "
    "identifier in it, find the exact sentence in a document you read that states it; if none does, or a "
    "document you read says something different, fix it (read again if needed). Make sure the document you "
    "used matches every qualifier in the question (customer, team, product, version, date) and is the most "
    "recent or authoritative version. If the documents do not state something the question asks, say so. "
    "Then write the complete final answer again, ending with the `Sources:` line."
)


def _tool_counts(messages: list) -> Counter:
    return Counter(
        call["function"]["name"]
        for m in messages
        if m.get("role") == "assistant"
        for call in (m.get("tool_calls") or [])
    )


def _opening(plugin, question: str, preload: int) -> str:
    """The first user message. With ``--preload N`` it carries a search for the question and
    the full text of its top N documents, so the agent starts from the documents instead of
    from snippets (a small model tends to answer from the first snippet it sees)."""
    if not preload:
        return question
    found = plugin.knowledge_search({"query": question})
    # Each hit is headed by its bare document id (``dsid_… | filename | tags=…``): the plugin stopped
    # numbering hits ("[1] dsid_…"), and this used to match only the numbered form, so preload read nothing.
    ids = list(dict.fromkeys(re.findall(r"^(dsid_[0-9a-f]{32}) \|", found, re.M)))[:preload]
    documents = "\n\n".join(plugin.knowledge_read({"document_id": d}) for d in ids)
    return (
        f"Question: {question}\n\n"
        "A search for the question found these documents:\n\n"
        f"{found}\n\nThe full text of the top {len(ids)}:\n\n{documents}\n\n"
        "These are search results, not verified answers: similar-looking documents about another customer, team, "
        "version or date are common. Research further with the tools as the instructions say, then answer the "
        f"question: {question}"
    )


def _load_plugin():
    spec = importlib.util.spec_from_file_location("hindsight_knowledge", HERE / "hindsight_knowledge" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hindsight_knowledge"] = module
    spec.loader.exec_module(module)
    return module


def _register(plugin) -> list[str]:
    from tools.registry import registry

    names = []
    for name, schema in plugin.SCHEMAS.items():
        if os.getenv("HINDSIGHT_KNOWLEDGE_DISABLE_" + name.upper()):
            continue
        registry.register(name=name, toolset=plugin.TOOLSET, schema=schema, handler=plugin.HANDLERS[name], emoji="📚")
        names.append(name)
    return names


def _answer(args, plugin, row: dict) -> dict:
    from run_agent import AIAgent

    started = time.perf_counter()
    agent = AIAgent(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider,
        api_mode=args.api_mode,
        max_iterations=args.max_iterations,
        enabled_toolsets=[plugin.TOOLSET],
        ephemeral_system_prompt=SYSTEM_PROMPT,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        max_tokens=args.max_tokens,
        platform="batch",
    )
    error = None
    try:
        result = agent.run_conversation(_opening(plugin, row["question"], args.preload), task_id=row["question_id"])
        answer = (result.get("final_response") or "").strip()
        messages = result.get("messages") or []
        if not answer or not result.get("completed", True):
            # Out of iterations mid-research: one last turn with no budget, so a question the
            # agent half-answered is scored on what it found rather than as blank.
            follow = agent.run_conversation(OUT_OF_BUDGET, conversation_history=messages, task_id=row["question_id"])
            answer = (follow.get("final_response") or answer).strip()
            messages = follow.get("messages") or messages
        used = _tool_counts(messages)
        if answer and args.min_tool_calls and (sum(used.values()) < args.min_tool_calls or not used["knowledge_read"]):
            follow = agent.run_conversation(VERIFY, conversation_history=messages, task_id=row["question_id"])
            answer = (follow.get("final_response") or answer).strip()
            messages = follow.get("messages") or messages
        if answer and args.check:
            follow = agent.run_conversation(CHECK, conversation_history=messages, task_id=row["question_id"])
            answer = (follow.get("final_response") or answer).strip()
            messages = follow.get("messages") or messages
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        answer, messages, error = "", [], f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    tools = _tool_counts(messages)
    # The whole conversation, one file per question: which documents the agent searched, read and
    # trusted is what a wrong answer is diagnosed from, and the summary row keeps only counts.
    (args.traces / f"{row['question_id']}.json").write_text(json.dumps(messages, default=str), encoding="utf-8")
    sources = answer.rsplit("Sources:", 1)[-1] if "Sources:" in answer else answer
    return {
        "question_id": row["question_id"],
        "answer": answer,
        "document_ids": list(dict.fromkeys(DSID.findall(sources))),
        "llm_calls": int(
            getattr(agent, "session_api_calls", 0) or sum(1 for m in messages if m.get("role") == "assistant")
        ),
        "prompt_tokens": int(getattr(agent, "session_prompt_tokens", 0)),
        "completion_tokens": int(getattr(agent, "session_completion_tokens", 0)),
        "tool_calls": dict(tools),
        "seconds": round(time.perf_counter() - started, 1),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=os.getenv("HINDSIGHT_EVAL_AGENT_BASE_URL", "http://localhost:18800/v1"))
    parser.add_argument("--api-key", default=os.getenv("HINDSIGHT_EVAL_AGENT_API_KEY", "none"))
    parser.add_argument("--model", default=os.getenv("HINDSIGHT_EVAL_AGENT_MODEL", "gemma-4-26b-a4b"))
    parser.add_argument(
        "--provider",
        default=os.getenv("HINDSIGHT_EVAL_AGENT_PROVIDER", "custom"),
        help="custom: the OpenAI-compatible --base-url; or a Hermes provider with its own login (openai-codex)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_MAX_ITERATIONS", "24"))
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_MAX_TOKENS", "4096")))
    parser.add_argument(
        "--min-tool-calls", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_MIN_TOOL_CALLS", "5"))
    )
    parser.add_argument(
        "--check", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_CHECK", "1")), help="1: one self-review turn"
    )
    parser.add_argument("--preload", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_PRELOAD", "4")))
    parser.add_argument("--context-length", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_CONTEXT", "131072")))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("HINDSIGHT_EVAL_AGENT_CONCURRENCY", "4")))
    args = parser.parse_args()
    args.api_mode = None
    args.traces = Path(args.out).with_suffix(".traces")
    args.traces.mkdir(exist_ok=True)
    if args.provider != "custom":
        # Resolved before $HERMES_HOME moves to the scratch home below: the login lives in the user's.
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=args.provider, target_model=args.model)
        args.base_url, args.api_key, args.api_mode = runtime["base_url"], runtime["api_key"], runtime.get("api_mode")

    # Hermes reads and writes $HERMES_HOME (config, sessions, memory). A scratch one keeps a run
    # from reading the developer's own config or leaving sessions in it.
    if "HERMES_HOME" not in os.environ:
        home = Path(tempfile.mkdtemp(prefix="hermes-eval-"))
        # tool_search off: with it on, Hermes hides plugin tools behind tool_search/describe/call
        # once they pass 5% of the context, which cost the agent three extra calls per tool use.
        # context_length: the custom endpoint does not report one Hermes trusts.
        (home / "config.yaml").write_text(
            f"model:\n  context_length: {args.context_length}\ntools:\n  tool_search: false\n", encoding="utf-8"
        )
        os.environ["HERMES_HOME"] = str(home)
    plugin = _load_plugin()
    print(f"tools: {_register(plugin)}", file=sys.stderr)

    rows = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    lock = threading.Lock()
    with Path(args.out).open("a", encoding="utf-8") as out, ThreadPoolExecutor(args.concurrency) as pool:
        for result in pool.map(lambda r: _answer(args, plugin, r), rows):
            with lock:
                out.write(json.dumps(result) + "\n")
                out.flush()
            print(
                f"{result['question_id']}: {result['seconds']}s {result['tool_calls']} {result['error'] or ''}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
