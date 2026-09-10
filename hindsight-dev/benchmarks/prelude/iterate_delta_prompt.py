"""Fast loop for iterating on the delta-ops prompt against a captured failure.

The end-to-end page eval takes minutes per run, which is far too slow to try
prompt variants. This replays ONE captured ``mental_model_delta_ops`` request N
times per variant and scores the result directly, so a change can be judged in
seconds.

Scoring is deliberately mechanical, not judged: apply the emitted operations to
the captured CURRENT DOCUMENT and ask whether the claim that answers the topic
survived. That is the property under test, and a judge would only add latency and
noise to a question with a yes/no answer.

Every variant runs N times because these calls are not reproducible even at
temperature 0 -- the same request produced a destructive edit in production and a
correct one on replay. So a variant's score is a RATE, and a single clean run
proves nothing.

Run with::

    cd hindsight-dev && uv run python -m benchmarks.prelude.iterate_delta_prompt --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.reflect.prompts import STRUCTURED_DELTA_SYSTEM_PROMPT
from rich.console import Console

console = Console()


def survives(raw: str, must_keep: str, document: str) -> bool:
    """Whether the claim under test is still present after applying the ops.

    Scores the RESULTING DOCUMENT, not each operation in isolation. The first
    version checked every op's text for every token, which failed a correct
    two-op merge: the op that rewrites "A total of 3" to "A total of 7" does not
    itself mention the customer names, so a genuine fix scored as a regression.
    Judge the document the operations produce, never the operations.
    """
    try:
        ops = json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("operations", [])
    except Exception:
        return False

    # Start from the captured document text; destructive ops drop what they
    # target, additive ops contribute their own text.
    surviving = document
    for op in ops:
        kind = op.get("op")
        if kind in ("remove_block", "remove_section"):
            surviving = ""
        elif kind in ("replace_block", "replace_section_blocks", "rename_section"):
            surviving = ""
    added = " ".join(json.dumps(op.get("text") or op.get("blocks") or "") for op in ops)
    final = surviving + " " + added
    return all(token in final for token in must_keep.split("|"))


async def score(provider, system: str, user: str, must_keep: str, repeats: int, document: str) -> tuple[int, list[str]]:
    kept = 0
    samples: list[str] = []
    for _ in range(repeats):
        result = await provider.call(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            scope="delta_prompt_iteration",
        )
        raw = result.content if isinstance(result.content, str) else json.dumps(result.content)
        if survives(raw, must_keep, document):
            kept += 1
        else:
            samples.append(raw[:200])
    return kept, samples


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="captured delta_ops messages JSON")
    parser.add_argument(
        "--must-keep",
        default="0.9.3|production",
        help="pipe-separated tokens that must survive in the document",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variants", type=Path, help="JSON file of {name: system_prompt} to try")
    args = parser.parse_args()

    messages = json.loads(args.input.read_text(encoding="utf-8"))
    captured_system = next(m["content"] for m in messages if m["role"] == "system")
    user = next(m["content"] for m in messages if m["role"] == "user")
    # The captured CURRENT DOCUMENT, so an untouched claim counts as surviving.
    document = user[user.index("CURRENT DOCUMENT") : user.index("## NEW INFORMATION")]

    candidates: dict[str, str] = {
        "captured (as production sent it)": captured_system,
        "current tree": STRUCTURED_DELTA_SYSTEM_PROMPT,
    }
    if args.variants:
        candidates.update(json.loads(args.variants.read_text(encoding="utf-8")))

    provider = create_llm_provider(
        provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "gemini"),
        api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY", ""),
        base_url="",
        model=os.getenv("HINDSIGHT_API_LLM_MODEL", "gemini-3.7-flash"),
        reasoning_effort=None,
    )

    console.print(f"[bold]claim that must survive:[/bold] {args.must_keep}  ({args.repeats} runs each)\n")
    for name, system in candidates.items():
        kept, bad = await score(provider, system, user, args.must_keep, args.repeats, document)
        colour = "green" if kept == args.repeats else ("yellow" if kept else "red")
        console.print(f"[{colour}]{kept}/{args.repeats}[/{colour}]  {name}")
        if bad:
            console.print(f"        first failure: {bad[0][:160]}")


if __name__ == "__main__":
    asyncio.run(main())
