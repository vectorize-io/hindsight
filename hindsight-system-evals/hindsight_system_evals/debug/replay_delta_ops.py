"""Replay one captured delta-ops request, try prompt variants on it, ask the model why.

An end-to-end page eval takes minutes, far too slow to iterate on a prompt. This
takes ONE captured ``mental_model_delta_ops`` request (``diagnose_page --dump``)
and replays it N times per system-prompt variant, so a change is judged in
seconds.

Scoring is mechanical, not judged: apply the emitted operations to the captured
CURRENT DOCUMENT and check the claim under test is still in the result. It scores
the resulting DOCUMENT, never each operation — an op that rewrites "a total of 3"
to "a total of 7" need not repeat the customer names, and scoring ops one by one
failed exactly that correct merge.

Every variant runs N times because these calls are not reproducible even at
temperature 0: the same request produced a destructive edit in production and a
correct one on replay. A variant's score is a rate; one clean run proves nothing.

``--interrogate`` then continues the SAME chat, telling the model what the right
result was and asking which lines of the prompt led it astray. That answer is a
lead about the input, not a result — any fix still has to win the replay and then
the eval.

Run with::

    cd hindsight-system-evals
    uv run python -m hindsight_system_evals.debug.replay_delta_ops \\
        --input captures/hq-count-billing-mental_model_delta_ops-<n>.json \\
        --must-keep "Denning|Barrow|Fairbank" --repeats 5 --variants variants.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


@dataclass
class CapturedRequest:
    system: str
    user: str

    @property
    def document(self) -> str:
        """The CURRENT DOCUMENT section, so an untouched claim counts as surviving."""
        start = self.user.find("CURRENT DOCUMENT")
        end = self.user.find("## NEW INFORMATION")
        return self.user[start:end] if start != -1 and end != -1 else ""


@dataclass
class VariantScore:
    kept: int
    failures: list[str] = field(default_factory=list)


def load(path: Path) -> CapturedRequest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # A trace row stores the messages list; accept it bare or wrapped.
    messages = [Message.model_validate(m) for m in (raw["messages"] if isinstance(raw, dict) else raw)]
    return CapturedRequest(
        system=next(m.content for m in messages if m.role == "system"),
        user=next(m.content for m in messages if m.role == "user"),
    )


def survives(reply: str, must_keep: list[str], document: str) -> bool:
    """Whether every must-keep token is in the document the operations produce."""
    try:
        ops = json.loads(reply[reply.index("{") : reply.rindex("}") + 1]).get("operations", [])
    except ValueError:
        return False
    # Destructive ops drop what they target and we cannot resolve block ids from
    # the prompt text alone, so any of them discards the captured document; the
    # additive ops then contribute their own text. Conservative on purpose: a
    # variant only scores if the claim is provably still there.
    destructive = {"remove_block", "remove_section", "replace_block", "replace_section_blocks", "rename_section"}
    surviving = "" if any(op.get("op") in destructive for op in ops) else document
    added = " ".join(json.dumps(op.get("text") or op.get("blocks") or "", ensure_ascii=False) for op in ops)
    final = f"{surviving} {added}"
    return all(token in final for token in must_keep)


def _client() -> genai.Client:
    key = os.getenv("HINDSIGHT_EVAL_LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    if not key:
        raise SystemExit("Set HINDSIGHT_EVAL_LLM_API_KEY or GEMINI_API_KEY")
    return genai.Client(api_key=key)


def _turn(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=text)])


async def _generate(client: genai.Client, model: str, system: str, turns: list[types.Content]) -> str:
    reply = await client.aio.models.generate_content(
        model=model,
        contents=turns,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.0),
    )
    return reply.text or ""


async def score(
    client: genai.Client, model: str, system: str, captured: CapturedRequest, must_keep: list[str], repeats: int
) -> VariantScore:
    result = VariantScore(kept=0)
    for _ in range(repeats):
        reply = await _generate(client, model, system, [_turn("user", captured.user)])
        if survives(reply, must_keep, captured.document):
            result.kept += 1
        else:
            result.failures.append(reply[:200])
    return result


async def interrogate(client: genai.Client, model: str, captured: CapturedRequest, expected: str) -> None:
    chat = [_turn("user", captured.user)]
    first = await _generate(client, model, captured.system, chat)
    chat.append(_turn("model", first))

    why = (
        f"The correct result was: {expected}\n\n"
        "Compare that with the operations you just emitted. Explain what in the prompt led you to your "
        "choice, quoting the specific lines you relied on. Be concrete and do not be agreeable for its "
        "own sake — if the prompt was unambiguous and you simply erred, say so."
    )
    chat.append(_turn("user", why))
    answer = await _generate(client, model, captured.system, chat)
    print(f"\n=== WHY ===\n{answer.strip()[:2500]}")

    chat += [
        _turn("model", answer),
        _turn(
            "user",
            "Now propose the minimal edit to the SYSTEM PROMPT that would have produced the correct result. "
            "It must not stop you superseding genuinely outdated content and must be a few lines. Give the "
            "exact text, and which existing line it replaces if any.",
        ),
    ]
    proposal = await _generate(client, model, captured.system, chat)
    print(f"\n=== PROPOSED PROMPT FIX ===\n{proposal.strip()[:2500]}")


async def run(args: argparse.Namespace) -> None:
    captured = load(args.input)
    client = _client()
    must_keep = args.must_keep.split("|")

    candidates = {"captured (as the server sent it)": captured.system}
    if args.variants:
        candidates.update(json.loads(args.variants.read_text(encoding="utf-8")))

    print(f"claim that must survive: {must_keep} ({args.repeats} runs each, {args.model})\n")
    for name, system in candidates.items():
        result = await score(client, args.model, system, captured, must_keep, args.repeats)
        print(f"{result.kept}/{args.repeats}  {name}")
        if result.failures:
            print(f"        first failure: {result.failures[0][:160]}")

    if args.interrogate:
        await interrogate(client, args.model, captured, args.expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="a prompt dumped by diagnose_page --dump")
    parser.add_argument("--must-keep", required=True, help="pipe-separated tokens that must survive in the document")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variants", type=Path, help="JSON file of {name: system_prompt} to try")
    parser.add_argument("--model", default=os.getenv("HINDSIGHT_EVAL_LLM_MODEL", "gemini-3.7-flash"))
    parser.add_argument("--interrogate", action="store_true", help="continue the chat: why, then how to fix")
    parser.add_argument("--expected", default="", help="the correct result, for --interrogate")
    args = parser.parse_args()
    if args.interrogate and not args.expected:
        parser.error("--interrogate needs --expected")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
