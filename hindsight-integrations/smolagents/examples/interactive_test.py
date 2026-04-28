"""Interactive test for the Hindsight SmolAgents integration.

Two modes:

1. **Tool mode (default)** — invoke retain/recall/reflect tools directly via REPL.
   No LLM required; verifies the tool plumbing and round-trips memories
   against a live Hindsight instance.

2. **Agent mode (requires OPENAI_API_KEY)** — spin up a real smolagents
   `CodeAgent` with the Hindsight tools attached, then chat with it. Verifies
   that an actual agent picks up the tools and uses them in context.

Usage:
    python examples/interactive_test.py --bank smol-demo
    python examples/interactive_test.py --bank smol-demo --agent     # agent mode

Tool mode commands:
  :retain <content>    Store a memory directly
  :recall <query>      Search memories
  :reflect <query>     LLM-backed answer using memories
  :memories            Dump the bank
  :script              Run a guided demo (retain → recall → reflect)
  :bank                Show current bank id
  :quit / :q           Exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hindsight_smolagents import (
    HindsightRecallTool,
    HindsightReflectTool,
    HindsightRetainTool,
    create_hindsight_tools,
)


BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(label: str, color: str = CYAN) -> None:
    print(f"{color}[{label}]{RESET}", end=" ")


def dump_memories(url: str, bank: str, api_key: str | None) -> None:
    req = urllib.request.Request(f"{url}/v1/default/banks/{bank}/memories/list")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        items = data.get("items", [])
        print(f"\n{YELLOW}=== Bank '{bank}' — {len(items)} memories ==={RESET}")
        for i, m in enumerate(items, 1):
            print(f"{i}. {m.get('text', '')[:200]}")
        if not items:
            print("(empty)")
        print()
    except Exception as e:
        print(f"{YELLOW}Could not list memories: {e}{RESET}")


def cmd_retain(retain: HindsightRetainTool, content: str) -> None:
    banner("RETAIN", CYAN)
    print(content[:200])
    try:
        result = retain.forward(content)
        banner("RESULT", GREEN)
        print(result)
    except Exception as e:
        banner("ERROR", YELLOW)
        print(e)


def cmd_recall(recall: HindsightRecallTool, query: str) -> None:
    banner("RECALL", CYAN)
    print(f"query: {query}")
    try:
        result = recall.forward(query)
        banner("RESULT", GREEN)
        print(result if result.strip() else "(no memories matched)")
    except Exception as e:
        banner("ERROR", YELLOW)
        print(e)


def cmd_reflect(reflect: HindsightReflectTool, query: str) -> None:
    banner("REFLECT", CYAN)
    print(f"query: {query}")
    try:
        result = reflect.forward(query)
        banner("RESULT", GREEN)
        print(result)
    except Exception as e:
        banner("ERROR", YELLOW)
        print(e)


def cmd_script(
    retain: HindsightRetainTool,
    recall: HindsightRecallTool,
    reflect: HindsightReflectTool,
    url: str,
    bank: str,
    api_key: str | None,
) -> None:
    """Run a guided demo: retain → wait → recall → reflect."""
    print(f"\n{GREEN}=== Scripted SmolAgents demo ==={RESET}\n")

    print(f"{DIM}>>> Step 1: Retain three facts about Ben{RESET}")
    facts = [
        "Ben is a developer at Vectorize building Hindsight, a memory system for AI agents.",
        "Ben prefers concise, direct communication over long-winded explanations.",
        "Ben works primarily in Python and TypeScript.",
    ]
    for fact in facts:
        cmd_retain(retain, fact)

    print(f"\n{DIM}>>> Step 2: Wait 8s for async fact extraction...{RESET}")
    import time
    time.sleep(8)
    dump_memories(url, bank, api_key)

    print(f"{DIM}>>> Step 3: Recall — should surface the stored facts{RESET}")
    cmd_recall(recall, "What do we know about Ben?")

    print(f"\n{DIM}>>> Step 4: Reflect — LLM-backed answer using the bank{RESET}")
    cmd_reflect(reflect, "Summarize Ben's role and communication preferences in one sentence.")

    print(f"\n{GREEN}=== Demo complete ==={RESET}\n")


def run_tool_mode(args: argparse.Namespace) -> None:
    tools = create_hindsight_tools(
        bank_id=args.bank,
        hindsight_api_url=args.hindsight_url,
        api_key=args.hindsight_api_key,
    )
    retain, recall, reflect = tools  # default order: retain, recall, reflect

    print(f"\n{GREEN}=== Hindsight SmolAgents — Tool Mode ==={RESET}")
    print(f"Bank:      {args.bank}")
    print(f"Hindsight: {args.hindsight_url}")
    print(f"\nCommands: {DIM}:retain <content>  :recall <query>  :reflect <query>  :script  :memories  :bank  :quit{RESET}\n")

    while True:
        try:
            line = input(f"{BLUE}smol> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            break
        if line == ":memories":
            dump_memories(args.hindsight_url, args.bank, args.hindsight_api_key)
            continue
        if line == ":bank":
            print(f"{YELLOW}bank: {args.bank}{RESET}")
            continue
        if line == ":script":
            cmd_script(retain, recall, reflect, args.hindsight_url, args.bank, args.hindsight_api_key)
            continue
        if line.startswith(":retain"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}Usage: :retain <content>{RESET}")
                continue
            cmd_retain(retain, parts[1])
            continue
        if line.startswith(":recall"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}Usage: :recall <query>{RESET}")
                continue
            cmd_recall(recall, parts[1])
            continue
        if line.startswith(":reflect"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}Usage: :reflect <query>{RESET}")
                continue
            cmd_reflect(reflect, parts[1])
            continue

        print(f"{YELLOW}Unknown command. Try :script for a guided demo.{RESET}")


def run_agent_mode(args: argparse.Namespace) -> None:
    """Run a real smolagents CodeAgent with Hindsight tools attached."""
    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{YELLOW}Agent mode requires OPENAI_API_KEY.{RESET}")
        sys.exit(1)

    try:
        from smolagents import CodeAgent, OpenAIServerModel
    except ImportError:
        print(f"{YELLOW}smolagents is not installed. Run `uv sync` from the integration dir.{RESET}")
        sys.exit(1)

    tools = create_hindsight_tools(
        bank_id=args.bank,
        hindsight_api_url=args.hindsight_url,
        api_key=args.hindsight_api_key,
    )

    model = OpenAIServerModel(
        model_id=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    agent = CodeAgent(tools=list(tools), model=model)

    print(f"\n{GREEN}=== Hindsight SmolAgents — Agent Mode ==={RESET}")
    print(f"Bank:      {args.bank}")
    print(f"Hindsight: {args.hindsight_url}")
    print(f"Model:     {model.model_id}")
    print(f"\nType a task, or :quit to exit.\n")

    while True:
        try:
            task = input(f"{BLUE}you> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task in (":quit", ":q", ":exit"):
            break
        if task == ":memories":
            dump_memories(args.hindsight_url, args.bank, args.hindsight_api_key)
            continue

        try:
            result = agent.run(task)
            banner("AGENT", GREEN)
            print(result)
        except Exception as e:
            banner("ERROR", YELLOW)
            print(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=f"smol-demo-{os.environ.get('USER', 'anon')}")
    parser.add_argument("--hindsight-url", default=os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888"))
    parser.add_argument("--hindsight-api-key", default=os.environ.get("HINDSIGHT_API_KEY"))
    parser.add_argument("--agent", action="store_true", help="Run a real CodeAgent (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.agent:
        run_agent_mode(args)
    else:
        run_tool_mode(args)


if __name__ == "__main__":
    main()
