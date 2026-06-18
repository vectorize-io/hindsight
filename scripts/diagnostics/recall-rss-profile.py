#!/usr/bin/env python3
"""Profile RSS while repeatedly calling the recall HTTP endpoint.

This is intentionally an external-process diagnostic: it can watch a running
Hindsight API/worker process with a real bank, model cache, and allocator state
instead of relying on an in-process microbenchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcStatus:
    rss_kb: int | None
    hwm_kb: int | None
    swap_kb: int | None
    threads: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call /memories/recall in a loop and sample /proc/<pid>/status. "
            "Use this to reproduce RSS retention around recall/reranking paths."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8888", help="Hindsight API base URL.")
    parser.add_argument("--bank-id", required=True, help="Memory bank to query.")
    parser.add_argument("--query", required=True, help="Recall query to execute.")
    parser.add_argument("--pid", type=int, required=True, help="PID of the API or worker process to sample.")
    parser.add_argument("--count", type=int, default=6, help="Number of recall calls to execute.")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between RSS samples.")
    parser.add_argument(
        "--post-idle-samples",
        type=int,
        default=12,
        help="RSS samples to collect after the final recall, to see whether memory returns to baseline.",
    )
    parser.add_argument("--budget", default="mid", choices=["low", "mid", "high"], help="Recall budget.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Recall max_tokens.")
    parser.add_argument(
        "--types",
        default=None,
        help="Comma-separated recall types, for example 'world,experience,observation'.",
    )
    parser.add_argument("--tags", default=None, help="Comma-separated tag filter.")
    parser.add_argument("--tags-match", default="any", help="Recall tags_match value.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--api-key", default=os.getenv("HINDSIGHT_API_KEY"), help="Bearer token, if required.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--malloc-trim",
        action="store_true",
        help="Send SIGUSR1 after each recall. Only use if the target process is configured to trim on that signal.",
    )
    return parser.parse_args()


def read_status(pid: int) -> ProcStatus:
    status_path = Path(f"/proc/{pid}/status")
    fields: dict[str, int] = {}
    try:
        with status_path.open() as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "VmSwap", "Threads"}:
                    fields[key] = int(value.strip().split()[0])
    except FileNotFoundError as exc:
        raise SystemExit(f"Process {pid} was not found at {status_path}") from exc
    return ProcStatus(
        rss_kb=fields.get("VmRSS"),
        hwm_kb=fields.get("VmHWM"),
        swap_kb=fields.get("VmSwap"),
        threads=fields.get("Threads"),
    )


def payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": args.query,
        "budget": args.budget,
        "max_tokens": args.max_tokens,
    }
    if args.types:
        payload["types"] = split_csv(args.types)
    if args.tags:
        payload["tags"] = split_csv(args.tags)
        payload["tags_match"] = args.tags_match
    return payload


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def call_recall(args: argparse.Namespace, payload: Mapping[str, Any]) -> tuple[int, float, int]:
    url = f"{args.base_url.rstrip('/')}/v1/default/banks/{args.bank_id}/memories/recall"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **auth_header(args.api_key)},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return response.status, time.monotonic() - start, len(data.get("results", []))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"recall failed with HTTP {exc.code}: {detail}") from exc


def auth_header(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sample_row(phase: str, iteration: int, status: ProcStatus, elapsed_s: float | None = None, results: int | None = None) -> dict[str, Any]:
    return {
        "ts": f"{time.time():.3f}",
        "phase": phase,
        "iteration": iteration,
        "elapsed_s": f"{elapsed_s:.3f}" if elapsed_s is not None else "",
        "results": results if results is not None else "",
        "rss_mb": kb_to_mb(status.rss_kb),
        "hwm_mb": kb_to_mb(status.hwm_kb),
        "swap_mb": kb_to_mb(status.swap_kb),
        "threads": status.threads if status.threads is not None else "",
    }


def kb_to_mb(value: int | None) -> str:
    if value is None:
        return ""
    return f"{value / 1024:.1f}"


def write_row(writer: csv.DictWriter[str], row: dict[str, Any]) -> None:
    writer.writerow(row)
    print(
        f"{row['phase']:>8} #{row['iteration']:<3} "
        f"rss={row['rss_mb']}MB hwm={row['hwm_mb']}MB swap={row['swap_mb']}MB "
        f"elapsed={row['elapsed_s'] or '-'} results={row['results'] or '-'}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.interval < 0:
        raise SystemExit("--interval must be >= 0")
    if args.post_idle_samples < 0:
        raise SystemExit("--post-idle-samples must be >= 0")

    payload = payload_from_args(args)
    rows = ["ts", "phase", "iteration", "elapsed_s", "results", "rss_mb", "hwm_mb", "swap_mb", "threads"]

    output_handle = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output_handle, fieldnames=rows)
        writer.writeheader()

        print(f"# git_sha={git_sha()} pid={args.pid} base_url={args.base_url} bank_id={args.bank_id}", file=sys.stderr)
        print(f"# payload={json.dumps(payload, sort_keys=True)}", file=sys.stderr)

        write_row(writer, sample_row("baseline", 0, read_status(args.pid)))
        for iteration in range(1, args.count + 1):
            status_code, elapsed_s, result_count = call_recall(args, payload)
            if status_code >= 400:
                raise RuntimeError(f"recall returned HTTP {status_code}")
            if args.malloc_trim:
                os.kill(args.pid, signal.SIGUSR1)
            write_row(writer, sample_row("recall", iteration, read_status(args.pid), elapsed_s, result_count))
            if args.interval:
                time.sleep(args.interval)

        for iteration in range(1, args.post_idle_samples + 1):
            if args.interval:
                time.sleep(args.interval)
            write_row(writer, sample_row("idle", iteration, read_status(args.pid)))
    finally:
        if args.output:
            output_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
