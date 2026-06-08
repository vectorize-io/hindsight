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
