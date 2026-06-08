#!/usr/bin/env python3
"""SWE-bench memory study orchestrator.

Runs a *consecutive* sequence of SWE-bench tasks from a single repository, twice:
  - control:   mini-swe-agent with no memory (each task cold)
  - treatment: the same agent + a persistent Hindsight bank (recall before, retain after)

Both arms run the identical, ordered task sequence; only the memory content differs. We
record tokens / steps / wall-clock / resolved per task, score patches with the official
SWE-bench harness, and emit a results JSON with a per-sequence warm-up curve.

Usage:
    python -m benchmarks.swebench.run_study --config config/smoke.yaml
    python -m benchmarks.swebench.run_study --config config/smoke.yaml --limit 2 --skip-score
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# Config & env
# --------------------------------------------------------------------------------------

# Appended to the stock instance template. Rendered for BOTH arms; control leaves
# `recalled_memories` empty so the block disappears and prompts are identical.
_MEMORY_BLOCK = """
{% if recalled_memories %}

<codebase_memory>
The following durable notes were learned by you (or another engineer) while solving EARLIER,
unrelated issues in THIS SAME repository. They are hints to help you navigate faster — verify
before relying on them, and do not assume they describe the current issue.
{{recalled_memories}}
</codebase_memory>
{% endif %}"""


def load_env_files() -> None:
    """Load .env from the study dir and the project root into os.environ (no overwrite)."""
    candidates = [HERE / ".env", HERE.parents[2] / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_agent_config(study: dict) -> dict:
    """Stock mini config (agent/model/environment) merged with study overrides + memory block.

    ``mini_base_config`` selects the stock template family:
      - ``swebench.yaml`` (default) — native tool-calling (best for frontier models).
      - ``swebench_backticks.yaml`` — text-based ```mswea_bash_command``` parsing, more robust
        for open models that emit malformed tool-call JSON. Pair with
        ``model.model_class: litellm_textbased``.
    """
    from minisweagent.config import builtin_config_dir

    base_name = study.get("mini_base_config", "swebench.yaml")
    stock_path = builtin_config_dir / "benchmarks" / base_name
    config = yaml.safe_load(stock_path.read_text())

    overrides = study.get("mini_overrides", {})
    config = deep_merge(config, overrides)

    # Append the memory injection block to the (possibly overridden) instance template.
    config["agent"]["instance_template"] = config["agent"]["instance_template"] + _MEMORY_BLOCK
    return config


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

DATASET_MAPPING = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


import re as _re

_DIFF_FILE_RE = _re.compile(r"^diff --git a/(\S+) b/", _re.MULTILINE)


def _patch_cluster_key(patch: str, depth: int) -> str | None:
    """The subsystem a gold patch belongs to: the most common dir prefix (at `depth`) of its
    touched files. Used only to GROUP related tasks — not exposed to the agent."""
    files = _DIFF_FILE_RE.findall(patch or "")
    if not files:
        return None
    from collections import Counter

    keys = ["/".join(f.split("/")[:depth]) for f in files if "/" in f]
    if not keys:
        return None
    return Counter(keys).most_common(1)[0][0]


def load_task_sequence(study: dict) -> list[dict]:
    from datasets import load_dataset

    ds = study["dataset"]
    subset = ds["subset"]
    split = ds.get("split", "test")
    repo_prefix = ds["repo_prefix"]  # e.g. "django__django-"
    limit = ds["max_tasks"]

    name = DATASET_MAPPING.get(subset, subset)
    rows = [dict(r) for r in load_dataset(name, split=split)]
    rows = [r for r in rows if r["instance_id"].startswith(repo_prefix)]

    def by_date(r: dict):
        return (str(r.get("created_at") or ""), r["instance_id"])

    # Cluster related tasks (same subsystem) so solving one transfers to the next — the key
    # lever for resolve-rate uplift. Off by default → chronological across all subsystems.
    if ds.get("cluster_by_patch"):
        depth = ds.get("cluster_depth", 3)
        groups: dict[str, list[dict]] = {}
        for r in rows:
            key = _patch_cluster_key(r.get("patch", ""), depth)
            if key:
                groups.setdefault(key, []).append(r)
        wanted = ds.get("cluster_key")
        if wanted and wanted in groups:
            chosen_key, chosen = wanted, groups[wanted]
        else:  # pick the densest cluster
            chosen_key, chosen = max(groups.items(), key=lambda kv: len(kv[1]))
        chosen.sort(key=by_date)
        print(
            f"Cluster '{chosen_key}': {len(chosen)} tasks available "
            f"(clusters: {sorted(((k, len(v)) for k, v in groups.items()), key=lambda x: -x[1])[:6]})"
        )
        return chosen[:limit]

    rows.sort(key=by_date)
    return rows[:limit]


# --------------------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------------------


def run_arm(
    *,
    arm: str,
    instances: list[dict],
    agent_config: dict,
    study: dict,
    out_dir: Path,
):
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    from .agent_hooks import MemoryAgent, MeteredAgent
    from .memory_glue import MemoryGlue
    from .metrics import TaskRecord

    enabled = arm == "treatment"
    repo = study["dataset"]["repo_label"]
    bank_id = f"{study['memory']['bank_prefix']}-{repo}-{arm}-s{study['seed']}"

    glue = MemoryGlue(
        base_url=os.environ["HINDSIGHT_API_URL"],
        api_token=os.environ.get("HINDSIGHT_API_TOKEN", ""),
        bank_id=bank_id,
        enabled=enabled,
        repo=repo,
        summary_model=study["memory"]["summary_model"],
        context_mode=study["memory"].get("context_mode", "recall"),
        recall_max_tokens=study["memory"].get("recall_max_tokens", 1024),
        recall_budget=study["memory"].get("recall_budget", "low"),
        recall_types=study["memory"].get("recall_types"),  # None = all types
        orientation_enabled=study["memory"].get("orientation_enabled", True),
        orientation_query=study["memory"].get("orientation_query"),
    )
    glue.reset_bank()

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    preds_path = arm_dir / "preds.json"
    preds: dict[str, dict] = {}

    records: list[TaskRecord] = []
    for seq, instance in enumerate(instances, start=1):
        iid = instance["instance_id"]
        task = instance["problem_statement"]
        print(f"\n=== [{arm}] task {seq}/{len(instances)}: {iid} ===", flush=True)

        model = get_model(config=copy.deepcopy(agent_config.get("model", {})))
        env = get_sb_environment(copy.deepcopy(agent_config), instance)
        agent_kwargs = dict(agent_config.get("agent", {}))
        agent_kwargs["output_path"] = arm_dir / iid / f"{iid}.traj.json"

        if enabled:
            agent = MemoryAgent(model, env, glue=glue, instance_id=iid, **agent_kwargs)
        else:
            agent = MeteredAgent(model, env, **agent_kwargs)

        t0 = time.time()
        exit_status, submission = "", ""
        try:
            info = agent.run(task)
            exit_status = info.get("exit_status", "")
            submission = info.get("submission", "") or ""
        except Exception as e:  # keep the session going; record the failure
            exit_status = type(e).__name__
            print(f"  !! {iid} raised {exit_status}: {e}", flush=True)
        wall = time.time() - t0

        rec = TaskRecord(
            arm=arm,
            seq=seq,
            instance_id=iid,
            input_tokens=getattr(agent, "input_tokens", 0),
            output_tokens=getattr(agent, "output_tokens", 0),
            n_steps=agent.n_calls,
            cost_usd=round(agent.cost, 4),
            wall_clock_s=round(wall, 2),
            exit_status=exit_status,
            recalled_chars=len(agent.extra_template_vars.get("recalled_memories", "")),
        )
        records.append(rec)
        preds[iid] = {
            "model_name_or_path": f"hindsight-{arm}",
            "instance_id": iid,
            "model_patch": submission,
        }
        preds_path.write_text(json.dumps(preds, indent=2))
        # Dump records incrementally so an interrupted run never loses completed-task data.
        (arm_dir / "records.json").write_text(json.dumps([r.as_dict() for r in records], indent=2))
        print(
            f"  -> steps={rec.n_steps} tokens={rec.total_tokens} "
            f"cost=${rec.cost_usd} wall={rec.wall_clock_s}s recalled_chars={rec.recalled_chars} "
            f"exit={exit_status}",
            flush=True,
        )

    return records, glue.stats.as_dict(), preds_path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="SWE-bench Hindsight memory study")
    ap.add_argument("--config", required=True, help="Path to study YAML (e.g. config/smoke.yaml)")
    ap.add_argument("--limit", type=int, default=None, help="Override dataset.max_tasks")
    ap.add_argument(
        "--step-limit", type=int, default=None, help="Override agent.step_limit (e.g. 80 for a fast directional run)"
    )
    ap.add_argument("--context-mode", choices=["recall", "reflect"], default=None, help="Override memory.context_mode")
    ap.add_argument(
        "--recall-types",
        default=None,
        help="Override memory.recall_types: 'all' or comma-separated (e.g. 'observation')",
    )
    ap.add_argument("--arms", default="control,treatment", help="Comma-separated arms to run")
    ap.add_argument(
        "--control-from",
        default=None,
        help="Reuse a prior run's control arm from its results.json (control is "
        "deterministic at temperature 0, so re-running it per config is wasted "
        "compute). Must be the SAME task-set, model, and step_limit.",
    )
    ap.add_argument("--skip-score", action="store_true", help="Skip official Docker scoring")
    args = ap.parse_args()

    load_env_files()
    os.environ.setdefault("HINDSIGHT_API_URL", "https://api.dev.hindsight.vectorize.io")

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = HERE / args.config  # resolve relative to the study dir
    study = yaml.safe_load(config_path.read_text())
    if args.limit is not None:
        study["dataset"]["max_tasks"] = args.limit
    if args.context_mode is not None:
        study["memory"]["context_mode"] = args.context_mode
    if args.recall_types is not None:
        study["memory"]["recall_types"] = (
            None
            if args.recall_types.lower() == "all"
            else [t.strip() for t in args.recall_types.split(",") if t.strip()]
        )

    agent_config = build_agent_config(study)
    if args.step_limit is not None:
        agent_config.setdefault("agent", {})["step_limit"] = args.step_limit
    instances = load_task_sequence(study)
    print(f"Loaded {len(instances)} tasks: {[r['instance_id'] for r in instances]}", flush=True)

    run_id = f"{study['run_id_prefix']}-s{study['seed']}"
    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_records: dict[str, list] = {}
    arm_mem: dict[str, dict] = {}
    arm_preds: dict[str, Path] = {}
    scored_arms: list[str] = []  # arms that still need official scoring

    # Reuse a deterministic control arm from a prior run instead of re-running + re-scoring it.
    reused_control = None
    if args.control_from and "control" in arms:
        from .metrics import TaskRecord

        prior = json.loads(Path(args.control_from).read_text())
        reused = [TaskRecord.from_dict(d) for d in prior["per_task"]["control"]]
        prior_ids = [r.instance_id for r in reused]
        if prior_ids != [r["instance_id"] for r in instances]:
            raise SystemExit(
                f"--control-from task-set mismatch: {args.control_from} has a different/ordered "
                f"instance list than the current run. Control reuse requires identical tasks."
            )
        reused_control = reused
        print(
            f"Reusing control arm from {args.control_from} "
            f"({sum(1 for r in reused if r.resolved)}/{len(reused)} resolved) — not re-running it.",
            flush=True,
        )

    for arm in arms:
        if arm == "control" and reused_control is not None:
            arm_records["control"] = reused_control  # already scored in the prior run
            arm_mem["control"] = {}
            continue
        recs, mem, preds_path = run_arm(
            arm=arm, instances=instances, agent_config=agent_config, study=study, out_dir=out_dir
        )
        arm_records[arm] = recs
        arm_mem[arm] = mem
        arm_preds[arm] = preds_path
        scored_arms.append(arm)

    # Official scoring per arm.
    if not args.skip_score:
        from .scoring import score_predictions

        ds = study["dataset"]
        instance_ids = [r["instance_id"] for r in instances]
        for arm in scored_arms:  # reused control is already scored — skip it
            print(f"\n=== scoring [{arm}] via official SWE-bench harness ===", flush=True)
            resolved = score_predictions(
                preds_path=arm_preds[arm],
                instance_ids=instance_ids,
                run_id=f"{run_id}-{arm}",
                dataset_name=DATASET_MAPPING.get(ds["subset"], ds["subset"]),
                split=ds.get("split", "test"),
                python_executable=os.environ.get("SWEBENCH_PYTHON"),
                max_workers=study.get("scoring", {}).get("max_workers", 2),
                timeout=study.get("scoring", {}).get("timeout", 1800),
            )
            for rec in arm_records[arm]:
                rec.resolved = resolved.get(rec.instance_id, False)

    # Build & write results.
    from .metrics import build_results

    results = build_results(
        config={
            "run_id": run_id,
            "seed": study["seed"],
            "dataset": study["dataset"],
            "model": agent_config.get("model", {}).get("model_name"),
            "memory": study["memory"],
            "scored": not args.skip_score,
            "arms": arms,
        },
        control=arm_records.get("control", []),
        treatment=arm_records.get("treatment", []),
        control_mem=arm_mem.get("control", {}),
        treatment_mem=arm_mem.get("treatment", {}),
    )
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {results_path}", flush=True)
    print(json.dumps(results.get("headline", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
