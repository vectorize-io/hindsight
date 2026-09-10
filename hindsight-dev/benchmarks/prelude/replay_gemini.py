"""Replay a stored delta-ops request against Gemini, then interrogate the model.

The eval says WHAT went wrong. The stored ``llm_requests`` row says exactly what
was asked. Replaying that row verbatim answers whether the failure is in the
prompt or was one unlucky sample — and because the replay is an ordinary chat,
the same session can then ask the model why it chose what it chose and what it
would need to have been told instead.

That last part matters: a prompt fix guessed from the outside is a hypothesis,
while the model naming the ambiguity it acted on is evidence about the input.
Both still have to be verified by re-running the eval — the model's opinion about
its own reasoning is a lead, not a result.

Run with::

    cd hindsight-dev && uv run python -m benchmarks.prelude.replay_gemini --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def call_gemini(model: str, key: str, system: str, turns: list[dict], temperature: float = 0.0) -> str:
    """One Gemini call. ``turns`` is the running chat, so a reply can be appended."""
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": turns,
        "generationConfig": {"temperature": temperature},
    }
    request = urllib.request.Request(
        _ENDPOINT.format(model=model, key=key),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    candidates = payload.get("candidates") or []
    if not candidates:
        return f"[no candidates: {json.dumps(payload)[:300]}]"
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="messages JSON from the llm_requests row")
    parser.add_argument("--model", default=os.getenv("HINDSIGHT_API_LLM_MODEL", "gemini-3.7-flash"))
    parser.add_argument("--repeats", type=int, default=3, help="verbatim replays, to separate prompt from sampling")
    parser.add_argument("--interrogate", action="store_true", help="continue the chat: why, then how to fix")
    args = parser.parse_args()

    key = os.getenv("HINDSIGHT_API_LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if not key:
        raise SystemExit("Set HINDSIGHT_API_LLM_API_KEY or GEMINI_API_KEY")

    messages = json.loads(args.input.read_text(encoding="utf-8"))
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")

    print(f"=== REPLAY x{args.repeats} ({args.model}, temperature 0) ===\n")
    replies = []
    for attempt in range(1, args.repeats + 1):
        reply = call_gemini(args.model, key, system, [{"role": "user", "parts": [{"text": user}]}])
        replies.append(reply)
        ops = reply
        try:
            parsed = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            ops = ", ".join(
                f"{o.get('op')}({o.get('block_id') or o.get('section_id')})" for o in parsed.get("operations", [])
            )
        except Exception:
            pass
        print(f"replay {attempt}: {ops[:220]}")

    if not args.interrogate:
        return

    # Continue the SAME chat, so the model is reasoning about the answer it just
    # gave rather than a description of it.
    chat = [
        {"role": "user", "parts": [{"text": user}]},
        {"role": "model", "parts": [{"text": replies[-1]}]},
    ]

    why = (
        "The CURRENT DOCUMENT listed 3 customers (Denning, Barrow, Fairbank). The 4 supporting "
        "facts name 4 DIFFERENT customers (Aldridge, Gadsden, Ellery, Calloway). All 7 reported a "
        "billing error in Q2 2026, and none of the new facts contradicts any of the old ones. The "
        "correct result is a document listing all 7.\n\n"
        "Your operations replaced the 3 existing customers with the 4 new ones, so the document now "
        "says 4 and the first 3 are lost.\n\n"
        "Explain what in the prompt led you to replace rather than merge. Quote the specific lines "
        "you relied on. Be concrete and do not be agreeable for its own sake — if the prompt was "
        "unambiguous and you simply erred, say that instead."
    )
    print("\n=== WHY ===\n")
    answer = call_gemini(args.model, key, system, chat + [{"role": "user", "parts": [{"text": why}]}])
    print(answer.strip()[:2500])

    chat += [
        {"role": "user", "parts": [{"text": why}]},
        {"role": "model", "parts": [{"text": answer}]},
    ]
    fix = (
        "Now propose the minimal edit to the SYSTEM PROMPT that would have made you merge to 7.\n\n"
        "Constraints: it must not stop you superseding genuinely outdated content, must not turn "
        "into 'never replace anything', and must be a small number of lines. Give the exact text to "
        "add or change, and say which existing line it replaces if any."
    )
    print("\n=== PROPOSED PROMPT FIX ===\n")
    proposal = call_gemini(args.model, key, system, chat + [{"role": "user", "parts": [{"text": fix}]}])
    print(proposal.strip()[:2500])


if __name__ == "__main__":
    main()
