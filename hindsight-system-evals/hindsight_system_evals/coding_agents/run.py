"""Run the baseline: N real Claude Code sessions, and the search rate they produced.

    uv run python -m hindsight_system_evals.coding_agents --build

What happens, in order:

1. a Hindsight server starts on its own pg0 instance (or ``--api-url`` points at
   one you already run);
2. a temp HOME is built, the plugin installed into it from the checkout, and a
   small git repo written for the agent to work in;
3. a **warm-up session** runs one throwaway turn. Its only job is to trigger the
   plugin's SessionStart seeding and then wait for the bank to settle — a cold
   bank has no knowledge pages, so measuring against one would score an empty
   memory rather than the prompt;
4. the measured sessions run, each a fresh Claude session over the same scripted
   turns;
5. the plugin's own ``usage.jsonl`` is aggregated into the rate.

Sessions run one at a time on purpose: they share the sandbox HOME, and the
plugin's per-session usage cursors live in files keyed by session, so two
concurrent runs would interleave writes to the same log for no gain in signal.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

from hindsight_client import Hindsight

from hindsight_system_evals.coding_agents.metric import read_usage_stats, transcript_credits
from hindsight_system_evals.coding_agents.prefill import placeholder_pages, prefill_bank
from hindsight_system_evals.coding_agents.sandbox import (
    PLUGIN_DIR,
    REPO_ROOT,
    build_plugin,
    build_sandbox,
    refresh_credentials,
)
from hindsight_system_evals.coding_agents.scenario import BASELINE_PROMPTS, write_fixture_repo
from hindsight_system_evals.coding_agents.session import run_session
from hindsight_system_evals.server import start_eval_server
from hindsight_system_evals.waiting import wait_until_settled

WARMUP_PROMPT = "What files are in this repo? One line each."


def _load_env_file(path: Path) -> None:
    """Make `uv run` pick up the repo's provider settings without a shell export.

    Only fills what is unset: an explicit export always wins, which is how a run
    is pointed at a different model than the developer's own server uses.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coding-agents baseline")
    parser.add_argument("--sessions", type=int, default=5, help="measured sessions (default: 5)")
    parser.add_argument("--turns", type=int, default=8, help="turns per session, from the scripted list")
    parser.add_argument("--model", default="opus", help="Claude Code model (default: opus)")
    parser.add_argument("--api-url", default=None, help="an existing Hindsight server; one is started otherwise")
    parser.add_argument("--build", action="store_true", help="npm run build the plugin first")
    parser.add_argument("--no-warmup", action="store_true", help="skip the seeding warm-up (measures a cold bank)")
    parser.add_argument(
        "--no-prefill",
        action="store_true",
        help="skip the prior-session documents and the page refresh (measures an empty bank)",
    )
    parser.add_argument(
        "--auto-inject",
        choices=["reflect", "pages", "recall", "none"],
        default=None,
        help="plugin autoInject arm; 'none' removes the per-turn memory injection entirely",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override any plugin config field for this run, repeatable — e.g. "
            "--config pageRefreshEveryTurns=1. Values parse as JSON when they can, else as a string."
        ),
    )
    parser.add_argument("--out", default=None, help="write the run report as JSON here")
    parser.add_argument("--work-dir", default=None, help="keep the sandbox here instead of a temp dir")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"), help="env file for the server's provider")
    args = parser.parse_args(argv)

    _load_env_file(Path(args.env_file))
    prompts = BASELINE_PROMPTS[: args.turns]
    if args.turns > len(BASELINE_PROMPTS):
        parser.error(f"only {len(BASELINE_PROMPTS)} scripted prompts exist; add more in scenario.py")

    run_id = uuid.uuid4().hex[:8]
    # `.resolve()` is load-bearing, not tidiness. On macOS /tmp is a symlink to
    # /private/tmp, and the MCP server the installer registers only runs itself
    # when `process.argv[1]` equals its own `import.meta.url` — an unresolved
    # /tmp path fails that comparison, so the server exits 0 in silence, no tools
    # are exposed, and the run reports a 0% search rate that means nothing.
    root = (
        Path(args.work_dir) if args.work_dir else Path(tempfile.gettempdir()) / f"hs-coding-agents-eval-{run_id}"
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"sandbox: {root}", flush=True)

    if args.build:
        print(f"building {PLUGIN_DIR}", flush=True)
        build_plugin()

    server = None
    api_url = args.api_url
    if not api_url:
        # Its own database, not the suite's shared pg0 instance. Sharing one means
        # a new server inherits every earlier run's queued work: an arm was seen
        # spending its startup refreshing mental models belonging to banks from
        # two arms ago, which delays the run and puts the two arms' LLM traffic in
        # the same queue — the arms stop being comparable, which is the only thing
        # this harness is for.
        os.environ.setdefault("HINDSIGHT_EVAL_PG0_INSTANCE", f"hs-coding-agents-{run_id}")
        print("starting hindsight-api", flush=True)
        server = start_eval_server(log_path=root / "server.log")
        api_url = server.url
    print(f"server: {api_url}", flush=True)

    bank_id = f"coding-agents-eval-{run_id}"
    try:
        sandbox = build_sandbox(
            root,
            api_url=api_url,
            bank_id=bank_id,
            workdir=root / "repo",
            config_overrides=_config_overrides(args),
        )
        write_fixture_repo(sandbox.workdir)
        print(f"bank: {bank_id}", flush=True)

        if not args.no_warmup:
            print("warm-up session (seeds the bank)", flush=True)
            warm = run_session(sandbox, [WARMUP_PROMPT], model=args.model, log_path=root / "warmup.jsonl")
            if warm.error:
                print(f"  warm-up: {warm.error}", file=sys.stderr, flush=True)
            asyncio.run(_settle(api_url, bank_id))

        if not args.no_prefill:
            print("prefilling the bank and refreshing its pages", flush=True)
            still_empty = asyncio.run(_prefill(api_url, bank_id))
            # Loud, because the failure is otherwise invisible: a placeholder page
            # is still listed in the roster the model sees, so the run would
            # measure an agent being told its memory is empty.
            if still_empty:
                print(f"  WARNING: pages still unfilled after refresh: {still_empty}", file=sys.stderr, flush=True)
            else:
                print("  every page has real content", flush=True)

        session_ids: set[str] = set()
        # Credit is counted from the finished transcripts, not the plugin's live
        # flag: the Stop hook evaluates a turn before its final reply — the one
        # carrying the credit line — has been flushed, which understated
        # attribution 13-against-31 on the run that exposed it.
        credits: dict[str, set[int]] = {}
        for index in range(args.sessions):
            print(f"session {index + 1}/{args.sessions}", flush=True)
            refresh_credentials(sandbox)
            result = run_session(
                sandbox,
                prompts,
                model=args.model,
                log_path=root / f"session-{index + 1}.jsonl",
                debug_path=root / f"session-{index + 1}.debug.log",
            )
            session_ids.add(result.session_id)
            if result.transcript:
                credits[result.session_id] = transcript_credits(result.transcript)
            status = result.error or f"{result.completed_turns}/{len(prompts)} turns"
            print(f"  {status}", flush=True)

        stats = read_usage_stats(sandbox.usage_file, sessions=session_ids, credits=credits)
        report = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "suite": "coding-agents-search-rate",
            "model": args.model,
            "bank": bank_id,
            "sandbox": str(root),
            "prompts": prompts,
            "warmup": not args.no_warmup,
            "prefill": not args.no_prefill,
            "auto_inject": args.auto_inject or "(plugin default)",
            "config_overrides": _config_overrides(args),
            **stats.to_dict(),
        }
        _print_report(report, sandbox.usage_file)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}", flush=True)
        return 0
    finally:
        if server:
            server.stop()


def _config_overrides(args: argparse.Namespace) -> dict:
    """The plugin config this arm runs with — what makes a run an experiment.

    JSON-parsed so a number stays a number: `pageRefreshEveryTurns=1` as the
    string "1" is not the same field value, and the plugin would ignore it.
    """
    overrides: dict = {"autoInject": args.auto_inject} if args.auto_inject else {}
    for item in args.config:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--config expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


async def _settle(api_url: str, bank_id: str) -> None:
    client = Hindsight(base_url=api_url)
    try:
        await wait_until_settled(client, bank_id)
    finally:
        await client.aclose()


async def _prefill(api_url: str, bank_id: str) -> list[str]:
    client = Hindsight(base_url=api_url)
    try:
        await prefill_bank(client, bank_id, lambda bank: wait_until_settled(client, bank))
        return await placeholder_pages(client, bank_id)
    finally:
        await client.aclose()


def _print_report(report: dict, usage_file: Path) -> None:
    print("\n── search rate ──────────────────────────────")
    print(f"  turns                {report['turns']}  ({report['sessions']} sessions)")
    print(f"  searches             {report['searches']}")
    print(f"  searches / turn      {report['searches_per_turn']:.3f}")
    print(f"  turns that searched  {report['turns_with_search']}  ({report['search_turn_rate']:.1%})")
    print(f"  turns that retrieved {report['turns_with_retrieval']}  ({report['retrieval_turn_rate']:.1%})")
    print(f"  turns with any call  {report['turns_with_any_call']}")
    credit = report["credit_rate"]
    print(f"  credit rate          {'n/a (nothing retrieved)' if credit is None else f'{credit:.1%}'}")
    print("  calls by tool:")
    for tool, count in report["calls_by_tool"].items() or [("(none)", 0)]:
        print(f"    {count:4d}  {tool}")
    print(f"\nusage log: {usage_file}")


if __name__ == "__main__":
    raise SystemExit(main())
