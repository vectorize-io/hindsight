#!/usr/bin/env python3
"""Label issues by area, so the backlog can be filtered.

Gemini picks the area: its enum output mode can only answer with one of the
options below, so there is no free text to parse or validate.

    triage-labels.py <number>
    triage-labels.py backfill        # every open unlabelled issue, once

Needs the gh CLI (authenticated) and GEMINI_API_KEY.
"""

import json
import os
import subprocess
import sys
import urllib.request

CORE = "core"
INTEGRATION = "integration"
# Integrations that get their own label; every other one is just INTEGRATION.
OWN_LABEL = {
    "hermes": "integration:hermes",
    "openclaw": "integration:openclaw",
    "coding-agents": "integration:coding-agents",
}
ALL_LABELS = [CORE, INTEGRATION, *OWN_LABEL.values()]

# Label -> what it means to the model. The labels are the enum values, so its
# answer is already the label to apply.
# Claude Code and Codex are both coding-agent harnesses AND server LLM providers. The
# first descriptions only named them under coding-agents, so gemini-2.5-flash-lite filed
# 13 of 28 open coding-agents issues there by mistake (e.g. #5542, a claude-code provider
# bug). Core now names those providers and coding-agents says they are not its own.
ISSUE_OPTIONS = {
    CORE: "The core Hindsight service: API server, memory engine (retain, recall, reflect, "
    "consolidation, mental models, knowledge pages, import/export), database, tenant/auth extensions, "
    "LLM/embedding/reranker providers (including the claude-code, openai-codex and github-copilot "
    "providers, where the server uses that tool as its LLM), the server's MCP tools, SDK clients, CLI, "
    "the hindsight-embed local daemon/launcher, Docker images, deployment, control plane UI, docs",
    OWN_LABEL["hermes"]: "The Hermes Agent memory plugin/integration",
    OWN_LABEL["openclaw"]: "The OpenClaw (or NemoClaw) memory plugin/integration",
    OWN_LABEL["coding-agents"]: "The coding-agents plugin package (hindsight-integrations/coding-agents) that "
    "runs inside a coding agent harness (Claude Code, Codex, Cursor, OpenCode, Copilot CLI, Cline, Pi, ...): "
    "its installer, hooks, auto-recall/retain of sessions, bank selection per repo, knowledge tools, harness "
    "support. Not this label when the issue is about the server using Claude Code or Codex as its LLM "
    "provider: that is core",
    INTEGRATION: "Another framework integration (CrewAI, LangGraph, LiteLLM, Pydantic AI, n8n, Obsidian, "
    "MCP clients, ...)",
}
GEMINI_MODEL = "gemini-3.1-flash-lite"


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def issue_label(title: str, body: str) -> str:
    options = "\n".join(f"- {label}: {meaning}" for label, meaning in ISSUE_OPTIONS.items())
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
        data=json.dumps(
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": "Which part of the Hindsight project is this GitHub issue about?\n\n"
                                f"{options}\n\nGitHub issue title: {title}\n\n"
                                # ponytail: crude 20k-char cut to keep each call cheap; a longer issue
                                # is labelled from its opening only, which is where the area is named.
                                f"{(body or '')[:20_000]}"
                            }
                        ]
                    }
                ],
                # Enum mode: the answer can only be one of the labels, no free text to parse.
                "generationConfig": {
                    "responseMimeType": "text/x.enum",
                    "responseSchema": {"type": "STRING", "enum": list(ISSUE_OPTIONS)},
                },
            }
        ).encode(),
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["candidates"][0]["content"]["parts"][0]["text"].strip()


def label_issue(number: int) -> None:
    issue = json.loads(gh("issue", "view", str(number), "--json", "title,body"))
    label = issue_label(issue["title"], issue["body"])
    # REST, not `gh issue edit`: that one fails on the retired Projects (classic) field.
    gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/issues/{number}/labels", "-f", f"labels[]={label}")
    print(f"issue #{number}: {label}")


def backfill() -> None:
    for label in ALL_LABELS:
        gh("label", "create", label, "--force", "--color", "5319e7" if label == CORE else "0e8a16")
    issues = json.loads(gh("issue", "list", "--state", "open", "--limit", "1000", "--json", "number,labels"))
    for issue in issues:
        if not any(label["name"] in ALL_LABELS for label in issue["labels"]):
            try:
                label_issue(issue["number"])
            except Exception as error:  # one bad issue shouldn't stop the rest
                print(f"issue #{issue['number']}: FAILED {error}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1] == "backfill":
        backfill()
    else:
        label_issue(int(sys.argv[1]))
