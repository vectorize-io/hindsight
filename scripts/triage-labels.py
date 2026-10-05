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
ISSUE_OPTIONS = {
    CORE: "The core Hindsight service: API server, memory engine (retain, recall, reflect, "
    "consolidation, mental models), database, LLM/embedding/reranker providers, SDK clients, "
    "CLI, Docker images, deployment, control plane UI, docs",
    OWN_LABEL["hermes"]: "The Hermes Agent memory plugin/integration",
    OWN_LABEL["openclaw"]: "The OpenClaw (or NemoClaw) memory plugin/integration",
    OWN_LABEL["coding-agents"]: "The coding-agents integration: one package of hooks and memory for coding agent "
    "harnesses (Claude Code, Codex, Cursor, OpenCode, Copilot CLI, Cline, Pi, ...)",
    INTEGRATION: "Another framework integration (CrewAI, LangGraph, LiteLLM, Pydantic AI, n8n, Obsidian, "
    "MCP clients, ...)",
}
GEMINI_MODEL = "gemini-2.5-flash-lite"


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
                                "text": f"Which part of the Hindsight project is this GitHub issue about?\n\n{options}\n\n"
                                # ponytail: crude cut to keep the prompt small.
                                f"GitHub issue title: {title}\n\n{(body or '')[:20_000]}"
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
