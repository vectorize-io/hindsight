#!/usr/bin/env python3
"""Label PRs and issues by area, so the backlog can be filtered.

PRs are labelled from the paths they touch. Issues have no paths, so TypeSafe's
Jev model picks the area: a Choice question can only answer with one of the
options below, so there is no free text to parse or validate.

    triage-labels.py pr <number>
    triage-labels.py issue <number>
    triage-labels.py backfill        # every open PR and issue, once

Needs the gh CLI (authenticated) and, for issues, TYPESAFE_API_KEY.
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

# Label -> what it means to the model. The labels are the Jev option keys, so
# its answer is already the label to apply.
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
# An integration's docs page lives here, named after it.
INTEGRATION_DOC_DIRS = ("hindsight-docs/docs-integrations/", "skills/hindsight-docs/references/sdks/integrations/")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def pr_labels(number: int) -> set[str]:
    files = gh("api", "--paginate", f"repos/{{owner}}/{{repo}}/pulls/{number}/files", "--jq", ".[].filename")
    return {path_label(path) for path in files.split()}


def path_label(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "hindsight-integrations" and len(parts) > 2:
        return OWN_LABEL.get(parts[1], INTEGRATION)
    if path.startswith(INTEGRATION_DOC_DIRS):
        return OWN_LABEL.get(parts[-1].split(".")[0], INTEGRATION)
    return CORE


assert path_label("hindsight-integrations/hermes/a.py") == "integration:hermes"
assert path_label("hindsight-integrations/crewai/a.py") == INTEGRATION
assert path_label("hindsight-integrations/README.md") == CORE
assert path_label("hindsight-docs/docs-integrations/coding-agents.md") == "integration:coding-agents"
assert path_label("skills/hindsight-docs/references/sdks/integrations/crewai.md") == INTEGRATION
assert path_label("hindsight-api-slim/x.py") == CORE


def issue_label(title: str, body: str) -> str:
    request = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": "jev-latest",
                # ponytail: crude cut to stay well inside Jev's 32k-token context.
                "state": f"GitHub issue title: {title}\n\n{(body or '')[:20_000]}",
                "questions": {
                    "area": {
                        "type": "choice",
                        "instructions": "Which part of the Hindsight project is this GitHub issue about?",
                        "criteria": ISSUE_OPTIONS,
                    }
                },
            }
        ).encode(),
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["answers"]["area"]["choice"]


def add_labels(number: int, labels: set[str]) -> None:
    if not labels:  # a PR with no changed files
        return
    # REST, not `gh pr edit`: that one fails on the retired Projects (classic) field.
    label_args = [arg for label in sorted(labels) for arg in ("-f", f"labels[]={label}")]
    gh("api", "-X", "POST", f"repos/{{owner}}/{{repo}}/issues/{number}/labels", *label_args)


def label_pr(number: int) -> None:
    labels = pr_labels(number)
    add_labels(number, labels)
    print(f"PR #{number}: {', '.join(sorted(labels))}")


def label_issue(number: int) -> None:
    issue = json.loads(gh("issue", "view", str(number), "--json", "title,body"))
    label = issue_label(issue["title"], issue["body"])
    add_labels(number, {label})
    print(f"issue #{number}: {label}")


def backfill() -> None:
    for label in ALL_LABELS:
        gh("label", "create", label, "--force", "--color", "5319e7" if label == CORE else "0e8a16")
    for kind, label_one in (("pr", label_pr), ("issue", label_issue)):
        items = json.loads(gh(kind, "list", "--state", "open", "--limit", "1000", "--json", "number,labels"))
        for item in items:
            if not any(label["name"] in ALL_LABELS for label in item["labels"]):
                try:
                    label_one(item["number"])
                except Exception as error:  # one bad item shouldn't stop the rest
                    print(f"{kind} #{item['number']}: FAILED {error}", file=sys.stderr)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "backfill":
        backfill()
    else:
        {"pr": label_pr, "issue": label_issue}[command](int(sys.argv[2]))
