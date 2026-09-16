"""Run an AMB benchmark against a Hindsight server — local, or one already up.

[AMB](https://github.com/vectorize-io/agent-memory-benchmark) owns the datasets
(LoComo, LongMemEval, BEAM, PersonaMem, sde-bench…), the ingestion and retrieval
adapters, the answer model, the judge and the scoring. **This module owns none of
that**, on purpose: a second copy of LoComo living here is exactly how the two
drift until neither number means anything. If a result looks wrong, the fix is a
PR to AMB.

What is left for this side is small: resolve a target the way the pytest evals
do, point AMB's `hindsight-http` provider at it, and run AMB's own CLI at a
pinned ref.

    uv run run-amb --dataset locomo --split conv-26
    uv run run-amb --dataset locomo --split conv-26 --api-url https://api.dev.example
    uv run run-amb --dataset longmemeval --split single-session-user -- --query-limit 20

The ref is pinned in `AMB_REF` next to this package. Unpinned, a dashboard
movement is unattributable — benchmark drift and engine drift look identical.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from hindsight_system_evals.target import ENV_API_KEY, ENV_API_URL, eval_target

AMB_REPO = "https://github.com/vectorize-io/agent-memory-benchmark.git"
AMB_REF_FILE = Path(__file__).resolve().parents[1] / "AMB_REF"

#: Default checkout location. CI overrides it to somewhere cacheable.
DEFAULT_CHECKOUT = Path(os.getenv("AMB_CHECKOUT") or Path.home() / ".cache" / "hindsight" / "amb")

#: Normally nothing: AMB pins its own interpreter in `.python-version`, and uv
#: honours it for a `--project` run. The knob stays because the failure it guards
#: lands far from its cause — AMB's `requires-python` is only `>=3.11`, so an
#: unpinned checkout picks the newest interpreter on the machine, and on 3.14
#: `onnxruntime` (via cognee) has no wheel and the install dies before the
#: benchmark starts.
DEFAULT_PYTHON = os.getenv("AMB_PYTHON") or None


def _run(*cmd: str, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_checkout(checkout: Path, ref: str, *, with_submodules: bool) -> Path:
    """Clone (or update) AMB at exactly ``ref``.

    Blobless: AMB carries published result blobs and dataset caches, and a run
    needs its code, not its history.
    """
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", "--filter=blob:none", AMB_REPO, str(checkout))
    _run("git", "-C", str(checkout), "fetch", "--filter=blob:none", "origin", ref)
    _run("git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD")
    if with_submodules:
        # sde-bench's tasks are a submodule; the QA datasets do not need it.
        _run("git", "-C", str(checkout), "submodule", "update", "--init", "--recursive", "sdebench/datasets")
    return checkout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-amb",
        description="Run an AMB benchmark against a Hindsight server (local by default).",
        epilog="Anything after `--` is passed to `amb run` verbatim.",
    )
    parser.add_argument("--dataset", required=True, help="AMB dataset, e.g. locomo, longmemeval, sdebench")
    parser.add_argument("--split", default=None, help="dataset split; `amb splits --dataset X` lists them")
    parser.add_argument("--memory", default="hindsight-http", help="AMB memory provider (default: hindsight-http)")
    parser.add_argument(
        "--api-url",
        default=None,
        help=f"a server that is already up; ${ENV_API_URL} does the same. Omitted, one is started here.",
    )
    parser.add_argument("--amb-ref", default=None, help=f"override the ref pinned in {AMB_REF_FILE.name}")
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT, help="where AMB is cloned")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="override the interpreter AMB pins for itself")
    parser.add_argument("amb_args", nargs=argparse.REMAINDER, help="extra `amb run` arguments after `--`")
    args = parser.parse_args(argv)

    ref = args.amb_ref or os.getenv("AMB_REF") or AMB_REF_FILE.read_text().strip()
    checkout = ensure_checkout(args.checkout, ref, with_submodules=args.dataset == "sdebench")

    extra = [a for a in args.amb_args if a != "--"]
    cmd = ["uv", "run", "--project", str(checkout)]
    if args.python:
        cmd += ["--python", args.python]
    cmd += ["amb", "run", "--dataset", args.dataset, "--memory", args.memory]
    if args.split:
        cmd += ["--split", args.split]
    cmd += extra

    with eval_target(args.api_url) as target:
        env = os.environ | {
            # What `hindsight-http` reads…
            "HINDSIGHT_HTTP_URL": target.url,
            "HINDSIGHT_HTTP_KEY": target.api_key or "",
            # …and what the sde-bench coding provider reads. Same server either way.
            "SDE_HINDSIGHT_URL": target.url,
        }
        print(f"AMB {args.dataset} @ {ref[:8]} → {target.describe()}", flush=True)
        # cwd is the checkout so results land in its `outputs/`, where `amb view`
        # and `amb publish-results` expect them.
        completed = subprocess.run(cmd, cwd=checkout, env=env)

    if completed.returncode != 0 and not target.is_remote:
        print("The server this run started has stopped; re-run with --api-url to keep one alive.", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover - console script is the entry point
    raise SystemExit(main())
