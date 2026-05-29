"""Interactive Vapi webhook simulator — test HindsightVapiWebhook by hand.

Simulates Vapi's webhook events (assistant-request + end-of-call-report) so you
can watch memory retain/recall happen in real time without a real Vapi account
or phone number.

Usage:
    python examples/interactive_webhook.py --bank vapi-demo
    python examples/interactive_webhook.py --bank vapi-demo --hindsight-url http://localhost:8888

Commands:
  :call <caller-number>     Simulate assistant-request (incoming call)
  :end <transcript>         Simulate end-of-call-report (call ends with transcript)
  :script                   Run a scripted demo: 1 call ends → wait → next call recalls
  :memories                 Dump all memories in the bank
  :bank                     Show current bank id
  :quit / :q                Exit

Example:
  vapi> :end User: My name is Alex. Assistant: Hi Alex!  User: I prefer email.  Assistant: Got it.
  vapi> :memories
  vapi> :call +15551234567
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hindsight_vapi import HindsightVapiWebhook

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


def make_assistant_request(caller_number: str | None) -> dict:
    """Build a fake Vapi assistant-request webhook payload."""
    return {
        "message": {
            "type": "assistant-request",
            "call": {
                "customer": {"number": caller_number} if caller_number else {},
            },
        }
    }


def make_end_of_call(transcript: str) -> dict:
    """Build a fake Vapi end-of-call-report webhook payload."""
    return {
        "message": {
            "type": "end-of-call-report",
            "artifact": {"transcript": transcript},
        }
    }


async def cmd_end_call(webhook: HindsightVapiWebhook, transcript: str, wait_seconds: int = 8) -> None:
    banner("WEBHOOK: end-of-call-report", CYAN)
    print(f"transcript length: {len(transcript)} chars")
    event = make_end_of_call(transcript)
    response = await webhook.handle(event)
    banner("RESPONSE", GREEN)
    print(response if response else "None (HTTP 200, no body)")
    banner("RETAIN", GREEN)
    print(f"{DIM}fire-and-forget task scheduled{RESET}")

    # Give the async retain task a chance to actually fire and let Hindsight
    # extract facts. Without this wait the input() prompt blocks the event
    # loop before the task can make progress.
    print(f"{DIM}waiting {wait_seconds}s for retain + fact extraction...{RESET}")
    for _ in range(wait_seconds):
        await asyncio.sleep(1)
    print(f"{DIM}done — try :memories to see what was extracted{RESET}")


async def cmd_assistant_request(webhook: HindsightVapiWebhook, caller: str | None) -> None:
    banner("WEBHOOK: assistant-request", CYAN)
    print(f"caller: {caller or '(none)'}")
    event = make_assistant_request(caller)
    response = await webhook.handle(event)
    banner("RESPONSE", GREEN)
    if not response:
        print(f"{DIM}empty {{}} — no memories matched the recall query{RESET}")
        return

    # Pretty-print the assistantOverrides structure
    print(json.dumps(response, indent=2)[:800])
    overrides = response.get("assistantOverrides", {})
    msgs = overrides.get("model", {}).get("messages", [])
    if msgs:
        banner("INJECTED SYSTEM PROMPT", CYAN)
        content = msgs[0].get("content", "")
        print(content[:500] + ("..." if len(content) > 500 else ""))


async def cmd_script(webhook: HindsightVapiWebhook, url: str, bank: str, api_key: str | None) -> None:
    """Run a scripted demo that proves the full retain → recall cycle."""
    print(f"\n{GREEN}=== Scripted demo: Alex's first + second call ==={RESET}\n")

    # Call 1 ends with a transcript
    print(f"{DIM}>>> Step 1: Call 1 ends (end-of-call-report){RESET}")
    transcript = (
        "User: Hi, my name is Alex and I'm calling from New York. "
        "Assistant: Nice to meet you Alex! How can I help? "
        "User: I prefer email updates over phone calls, and my account number is A-12345. "
        "Assistant: Got it — email updates, account A-12345. Anything else? "
        "User: No that's all, thanks. "
        "Assistant: Have a great day!"
    )
    await cmd_end_call(webhook, transcript)

    # Wait for async extraction
    print(f"\n{DIM}>>> Step 2: Waiting 8s for async retain + fact extraction...{RESET}")
    await asyncio.sleep(8)
    dump_memories(url, bank, api_key)

    # Call 2 starts — should recall Alex's prefs
    print(f"{DIM}>>> Step 3: Call 2 starts (assistant-request) — should recall Alex{RESET}")
    await cmd_assistant_request(webhook, "+15551234567")
    print(f"\n{GREEN}=== Demo complete ==={RESET}\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=f"vapi-demo-{os.environ.get('USER', 'anon')}")
    parser.add_argument("--hindsight-url", default=os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888"))
    parser.add_argument("--hindsight-api-key", default=os.environ.get("HINDSIGHT_API_KEY"))
    args = parser.parse_args()

    webhook = HindsightVapiWebhook(
        bank_id=args.bank,
        hindsight_api_url=args.hindsight_url,
        api_key=args.hindsight_api_key,
    )

    print(f"\n{GREEN}=== Vapi + Hindsight Interactive Webhook Simulator ==={RESET}")
    print(f"Bank:      {args.bank}")
    print(f"Hindsight: {args.hindsight_url}")
    cmds = ":call <number>  :end <transcript>  :script  :memories  :bank  :quit"
    print(f"\nCommands: {DIM}{cmds}{RESET}\n")

    while True:
        try:
            line = input(f"{BLUE}vapi> {RESET}").strip()
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
            await cmd_script(webhook, args.hindsight_url, args.bank, args.hindsight_api_key)
            continue
        if line.startswith(":call"):
            parts = line.split(maxsplit=1)
            caller = parts[1].strip() if len(parts) > 1 else None
            await cmd_assistant_request(webhook, caller)
            continue
        if line.startswith(":end"):
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"{YELLOW}Usage: :end <transcript text>{RESET}")
                continue
            await cmd_end_call(webhook, parts[1])
            continue

        print(f"{YELLOW}Unknown command. Try :script for a guided demo, or :call/:end.{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
