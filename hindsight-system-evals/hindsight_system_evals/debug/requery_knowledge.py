"""Score a knowledge case again on a bank that already holds its corpus: no upload, only the
queries. For trying search settings (a server restarted with other knobs, an index dropped)
without paying the ingest again.

    uv run --extra knowledge python -m hindsight_system_evals.debug.requery_knowledge \
        --api-url http://localhost:8888 --case hr --bank syseval-kb-vidore-hr-1a2b3c [--full]
"""

from __future__ import annotations

import argparse
import asyncio
import json

from hindsight_system_evals.knowledge import RANKED_TOP_K, KnowledgeApi, SearchCall, latency, score_ranked, search_all
from hindsight_system_evals.knowledge_cases import CASES, DEFAULT, FULL, _runs


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--top-k", type=int, default=RANKED_TOP_K, help="ranks scored (cut client-side)")
    args = parser.parse_args()
    case = next(c for c in CASES if c.id == args.case)
    suite = case.suite(FULL if args.full else DEFAULT)
    api = KnowledgeApi(args.api_url)
    try:
        searched = await search_all(api, args.bank, [SearchCall(q.id, q.text, args.top_k) for q in suite.queries])
    finally:
        await api.aclose()
    metrics = score_ranked(suite.queries, _runs(searched, suite.page_ids))
    keep = {k: v for k, v in metrics.items() if ":" not in k}
    print(
        json.dumps(
            {
                "case": case.id,
                "label": args.label,
                "top_k": args.top_k,
                **keep,
                "latency": latency(searched).model_dump(),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
