"""Authoritative resolve scoring via the official SWE-bench evaluation harness.

We never hand-judge a patch. Each arm's ``preds.json`` (the mini-swe-agent format, a dict
keyed by instance_id) is fed to ``swebench.harness.run_evaluation`` in Docker, which applies
the patch and runs the repo's FAIL_TO_PASS / PASS_TO_PASS tests. We parse the resulting
``<model>.<run_id>.json`` report into ``{instance_id: resolved_bool}``.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def score_predictions(
    *,
    preds_path: Path,
    instance_ids: list[str],
    run_id: str,
    dataset_name: str,
    split: str,
    python_executable: str | None = None,
    max_workers: int = 2,
    timeout: int = 1800,
    namespace: str | None = "swebench",
) -> dict[str, bool]:
    """Run the official harness and return {instance_id: resolved}.

    Missing instances (harness errored / no patch) are reported as False.
    """
    py = python_executable or sys.executable
    preds_path = preds_path.resolve()
    workdir = preds_path.parent  # harness writes its report relative to CWD; pin it here
    cmd = [
        py,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        str(preds_path),  # absolute — CWD is workdir below
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout),
        "--cache_level",
        "env",
    ]
    if namespace is not None:
        cmd += ["--namespace", namespace]
    if instance_ids:
        cmd += ["--instance_ids", *instance_ids]

    env = dict(os.environ)
    # Reports are written relative to CWD by the harness; pin it to the arm's dir.
    subprocess.run(cmd, cwd=str(workdir), env=env, check=False)

    report = _find_report(workdir, run_id)
    resolved: dict[str, bool] = {iid: False for iid in instance_ids}
    if report is not None:
        for iid in report.get("resolved_ids", []):
            resolved[iid] = True
    return resolved


def _find_report(workdir: Path, run_id: str) -> dict | None:
    matches = sorted(glob.glob(str(workdir / f"*.{run_id}.json")))
    if not matches:
        return None
    try:
        return json.loads(Path(matches[-1]).read_text())
    except Exception:
        return None


# Lines in test_output.txt that echo applied patches (the harness applies the GOLD test patch
# and traces it). These must never reach the agent's memory — that would leak the answer.
_PATCH_ECHO_MARKERS = ("Applied patch", "Checking patch", "git apply", "patching file")
_DIFF_LINE_PREFIXES = ("+", "-", "diff --git", "index ", "@@", "--- ", "+++ ")


def extract_eval_feedback(
    *,
    workdir: Path,
    run_id: str,
    model_name: str,
    instance_id: str,
    max_output_chars: int = 3500,
) -> str:
    """CI-style feedback from one instance's official evaluation: which required tests still
    fail, which previously-passing tests the patch broke, and the failure output excerpts.

    This is exactly what a developer gets from a CI run — and nothing more. The gold patch and
    the gold *test* patch are excluded: only report.json statuses and the test-run output (with
    patch-echo/diff lines stripped) are used. Returns "" if no evaluation artifacts exist.
    """
    inst_dir = workdir / "logs" / "run_evaluation" / run_id / model_name / instance_id
    sections: list[str] = []
    has_failures = False

    report_path = inst_dir / "report.json"
    if report_path.exists():
        try:
            rep = json.loads(report_path.read_text()).get(instance_id, {})
        except Exception:
            rep = {}
        tests = rep.get("tests_status", {})
        f2p = tests.get("FAIL_TO_PASS", {})
        p2p = tests.get("PASS_TO_PASS", {})
        if rep.get("patch_is_None") or not rep.get("patch_exists", True):
            sections.append("The submitted patch was EMPTY — no change was applied.")
        if f2p.get("failure"):
            has_failures = True
            sections.append(
                "Tests written for the required fix that STILL FAIL with this patch:\n"
                + "\n".join(f"  - {t}" for t in f2p["failure"])
            )
        if f2p.get("success"):
            sections.append(
                "Required tests that PASS with this patch:\n" + "\n".join(f"  - {t}" for t in f2p["success"])
            )
        if p2p.get("failure"):
            has_failures = True
            sections.append(
                "REGRESSIONS — tests that passed before this patch and now FAIL:\n"
                + "\n".join(f"  - {t}" for t in p2p["failure"])
            )

    # The raw output tail only earns its tokens when there's a failure to explain — for a
    # fully-passing run it is just a wall of "... ok" lines.
    test_output_path = inst_dir / "test_output.txt"
    if has_failures and test_output_path.exists():
        try:
            excerpt = _failure_excerpt(test_output_path.read_text(errors="replace"), max_output_chars)
        except Exception:
            excerpt = ""
        if excerpt:
            sections.append(f"Test-run output (tail):\n{excerpt}")

    return "\n\n".join(sections)


def _failure_excerpt(text: str, max_chars: int) -> str:
    """The tail of the test-run output with patch-echo/diff lines stripped.

    Django (and most repos) print per-failure detail blocks and the FAILED summary at the end,
    so the tail carries the assertion/exception evidence a developer would read first.
    """
    kept = [
        line
        for line in text.splitlines()
        if not line.startswith(_DIFF_LINE_PREFIXES) and not any(m in line for m in _PATCH_ECHO_MARKERS)
    ]
    return "\n".join(kept)[-max_chars:]
