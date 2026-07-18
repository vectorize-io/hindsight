"""Interactive text chat — test HindsightMemoryService by hand.

Simulates the Pipecat pipeline at text level so you can have a real conversation
and watch memory recall/retain happen in real time.

Usage:
    export OPENAI_API_KEY=sk-...           # optional; falls back to fake assistant
    python examples/interactive_chat.py --bank demo-$USER
    python examples/interactive_chat.py --bank demo-$USER --hindsight-url http://localhost:8888

Each turn shows:
  [RECALL]  — what Hindsight returned for the current user query
  [INJECT]  — the <hindsight_memories> system message added to the LLM context
  [LLM]     — the assistant's response (OpenAI or manual)
  [RETAIN]  — complete user+assistant turn sent to Hindsight

Commands:
  :quit / :q    — exit
  :memories     — dump all memories in the bank
  :reset        — clear context (keeps the Hindsight bank)
  :bank         — show current bank id
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from unittest.mock import MagicMock

# Ensure package is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hindsight_pipecat import HindsightMemoryService
from hindsight_pipecat.memory import _MEMORY_MARKER

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(label: str, color: str = CYAN) -> None:
    print(f"{color}[{label}]{RESET}", end=" ")


def _make_frame(messages: list[dict]) -> MagicMock:
    """Build a mock LLMContextFrame (Pipecat's text context carrier)."""
    from pipecat.frames.frames import LLMContextFrame

    frame = MagicMock(spec=LLMContextFrame)
    ctx = MagicMock()
    ctx.messages = messages
    frame.context = ctx
    return frame


async def _run_turn(service: HindsightMemoryService, messages: list[dict]) -> list[dict]:
    """Send messages through the memory service; return possibly-mutated list."""
    frame = _make_frame(messages)
    service.push_frame = MagicMock(return_value=asyncio.sleep(0))
    await service._handle_context_frame(frame)
    return frame.context.messages


def get_openai_response(messages: list[dict]) -> str | None:
    """Call OpenAI if available; otherwise return None (caller prompts manually)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,  # type: ignore[arg-type]
        max_tokens=200,
    )
    return resp.choices[0].message.content or ""


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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=f"demo-{os.environ.get('USER', 'anon')}")
    parser.add_argument("--hindsight-url", default=os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888"))
    parser.add_argument("--hindsight-api-key", default=os.environ.get("HINDSIGHT_API_KEY"))
    parser.add_argument(
        "--system-prompt",
        default="You are a friendly voice assistant with long-term memory. Use memories naturally; don't dump them verbatim. Keep responses concise.",
    )
    args = parser.parse_args()

    service = HindsightMemoryService(
        bank_id=args.bank,
        hindsight_api_url=args.hindsight_url,
        api_key=args.hindsight_api_key,
    )

    messages: list[dict] = [{"role": "system", "content": args.system_prompt}]

    use_openai = get_openai_response([{"role": "user", "content": "ping"}]) is not None
    llm_mode = "OpenAI" if use_openai else "manual (no OPENAI_API_KEY)"

    print(f"\n{GREEN}=== Pipecat + Hindsight Interactive Chat ==={RESET}")
    print(f"Bank:        {args.bank}")
    print(f"Hindsight:   {args.hindsight_url}")
    print(f"LLM mode:    {llm_mode}")
    print(f"System:      {args.system_prompt[:80]}...")
    print(f"\nType a message. Commands: {DIM}:quit  :memories  :reset  :bank{RESET}\n")

    turn_number = 0
    while True:
        try:
            user_input = input(f"{BLUE}you> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in (":quit", ":q", ":exit"):
            break
        if user_input == ":memories":
            dump_memories(args.hindsight_url, args.bank, args.hindsight_api_key)
            continue
        if user_input == ":reset":
            messages = [{"role": "system", "content": args.system_prompt}]
            service._last_retained_count = 0
            print(f"{YELLOW}Context reset (Hindsight bank untouched){RESET}")
            continue
        if user_input == ":bank":
            print(f"{YELLOW}bank: {args.bank}{RESET}")
            continue

        turn_number += 1
        print(f"\n{DIM}── Turn {turn_number} ──{RESET}")

        # Append the user message — the memory service will run retain/recall on this
        messages.append({"role": "user", "content": user_input})

        # Run through memory service (recall + inject + retain-any-prior-pair)
        messages = await _run_turn(service, messages)

        # Show what got recalled/injected
        memory_msg = next(
            (m for m in messages if m.get("role") == "system" and _MEMORY_MARKER in m.get("content", "")),
            None,
        )
        if memory_msg:
            banner("RECALL+INJECT", CYAN)
            content = memory_msg["content"]
            # Show just the memories, not the XML wrapper
            inner = content.replace(_MEMORY_MARKER, "").replace("</hindsight_memories>", "").strip()
            print(inner[:500] + ("..." if len(inner) > 500 else ""))
        else:
            banner("RECALL", DIM)
            print(f"{DIM}(no memories yet — bank is fresh or query didn't match){RESET}")

        # Get assistant response
        if use_openai:
            banner("LLM", GREEN)
            response = get_openai_response(messages) or "(empty)"
            print(response)
        else:
            banner("LLM (manual)", YELLOW)
            print(f"{DIM}Type what the assistant should say (one line):{RESET}")
            response = input("  ").strip()
            if not response:
                response = "(no response)"

        messages.append({"role": "assistant", "content": response})

        # Next turn, the service will retain this completed pair
        banner("RETAIN", GREEN)
        print(f"{DIM}scheduled — will fire on next turn's recall call{RESET}\n")

    # On exit, run one more turn to flush the last retain
    if turn_number > 0:
        print(f"{DIM}Flushing final turn...{RESET}")
        # A no-op user message to trigger retain of the last pair
        await _run_turn(service, messages)
        print(f"{DIM}Waiting 5s for async retain + fact extraction...{RESET}")
        await asyncio.sleep(5)
        dump_memories(args.hindsight_url, args.bank, args.hindsight_api_key)


if __name__ == "__main__":
    asyncio.run(main())
