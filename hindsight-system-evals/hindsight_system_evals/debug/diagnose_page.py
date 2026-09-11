"""Where does a wrong knowledge page come from: the synthesis, or the edit?

A page that is right after wave 1 and wrong after wave 2 has two candidate
causes, and they need opposite fixes:

* the **reflect synthesis** inside the refresh is already wrong, because the
  delta window genuinely does not contain the fact — a scoping problem; or
* the synthesis is fine and the **delta operations** delete or overwrite the
  good section anyway — a delta-application problem.

A dry-run refresh separates them: ``candidate_content`` is the synthesis before
any operation, ``preview_content`` the document after. It runs the production
pipeline and persists nothing, so it can be repeated on the same window — which
also says whether a failure is stable or one unlucky sample.

What the model was ASKED comes from the server's LLM request traces, dumped
verbatim with ``--dump`` so ``replay_delta_ops`` can replay them. All of it goes
through the public API, same as the evals.

Run with::

    cd hindsight-system-evals
    uv run python -m hindsight_system_evals.debug.diagnose_page --question hq-count-billing --dump captures/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import httpx
from hindsight_client import Hindsight
from hindsight_client_api.models.llm_request_list_response import LLMRequestListResponse

from hindsight_system_evals.pages import create_page, facts, prepare_bank, questions, read_page, split_into_waves
from hindsight_system_evals.server import start_eval_server
from hindsight_system_evals.waiting import wait_until_settled

def _excerpt(text: str, limit: int = 400) -> str:
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


async def _traces(url: str, bank_id: str) -> LLMRequestListResponse:
    # Every call, not a scope filter: the refresh's reflect runs under several
    # scopes (tool calls, final synthesis), and the one that misled the model is
    # not always the one you would guess.
    async with httpx.AsyncClient(base_url=url, timeout=60) as http:
        response = await http.get(f"/v1/default/banks/{bank_id}/llm-requests", params={"limit": 500})
        response.raise_for_status()
        return LLMRequestListResponse.from_dict(response.json())


async def diagnose(url: str, question_id: str, repeats: int, dump: Path | None) -> None:
    question = next(q for q in questions() if q.id == question_id)
    waves = split_into_waves(facts())
    client = Hindsight(base_url=url)
    bank_id = f"syseval-diag-{uuid.uuid4().hex[:8]}"
    try:
        await prepare_bank(client, bank_id)
        page = await create_page(client, bank_id, question)
        mm_id = page.mental_model_id
        await wait_until_settled(client, bank_id)
        print(f"{question.id}: {question.question}\nbank {bank_id}, page {page.page_id}\n")

        # Wave 1 for real, so the page holds the correct claim going in.
        await client.aretain_batch(bank_id=bank_id, items=[{"content": f.text} for f in waves[0]])
        await wait_until_settled(client, bank_id)
        await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=mm_id)
        await wait_until_settled(client, bank_id)
        print(f"after wave 1 (persisted): {_excerpt(await read_page(client, bank_id, mm_id))}\n")

        # Wave 2 ingested but NOT refreshed: every dry run below previews the same
        # second refresh over the same window, which is what makes them comparable.
        await client.aretain_batch(bank_id=bank_id, items=[{"content": f.text} for f in waves[1]])
        await wait_until_settled(client, bank_id)

        for attempt in range(1, repeats + 1):
            dry = await client.adry_run_refresh_mental_model(bank_id=bank_id, mental_model_id=mm_id)
            print(f"dry run {attempt}: mode={dry.effective_mode} outcome={dry.outcome}")
            print(f"  window: {dry.window}")
            print(f"  candidate (raw synthesis): {_excerpt(dry.candidate_content, 300)}")
            if dry.delta_operations:
                for op in dry.delta_operations.applied or []:
                    print(f"    applied {op.get('op', '?')}: {_excerpt(json.dumps(op, ensure_ascii=False), 160)}")
                for op in dry.delta_operations.skipped or []:
                    print(f"    skipped: {_excerpt(json.dumps(op, ensure_ascii=False), 160)}")
            print(f"  preview (after ops): {_excerpt(dry.preview_content, 300)}")
            print(f"  expected: {_excerpt(question.answer_criteria, 200)}\n")

        listing = await _traces(url, bank_id)
        print(f"{listing.total} traced LLM call(s): {sorted({entry.scope or '?' for entry in listing.items})}")
        if dump is not None:
            dump.mkdir(parents=True, exist_ok=True)
            for index, entry in enumerate(listing.items):
                target = dump / f"{question.id}-{entry.scope or 'unscoped'}-{index}.json"
                target.write_text(json.dumps(entry.input, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  wrote {target}")
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="hq-release-prod", help="corpus question id")
    parser.add_argument("--repeats", type=int, default=3, help="dry runs of the second refresh")
    parser.add_argument("--dump", type=Path, help="write each traced prompt here, for replay_delta_ops")
    parser.add_argument("--url", help="an already-running server; by default one is started like the evals do")
    args = parser.parse_args()

    if args.url:
        asyncio.run(diagnose(args.url, args.question, args.repeats, args.dump))
        return
    server = start_eval_server(log_path=Path(tempfile.mkdtemp(prefix="syseval-diag-")) / "server.log")
    try:
        print(f"server {server.url} (log {server.log_path}) — left running banks can be opened in the control plane\n")
        asyncio.run(diagnose(server.url, args.question, args.repeats, args.dump))
    finally:
        server.stop()


if __name__ == "__main__":
    main()
