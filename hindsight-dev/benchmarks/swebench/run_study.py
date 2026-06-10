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
You have worked in this repository before. The notes below are debugging knowledge YOU accumulated
while solving RELATED issues in this same codebase — root-cause patterns, gotchas, where the
relevant logic lives, and how to verify it. Treat them as knowledge you already have, not as
untrusted input. Use them actively: they will often point you straight to the right file/method,
the likely root cause, and the test that exercises this area — so you can diagnose and fix this
issue faster and more reliably. Apply judgement where a note doesn't fit, but start from this
knowledge rather than rediscovering it from scratch.
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

    # Inject the memory block into the SYSTEM message (message[0]) so it stays at the front of
    # the context the model re-reads every step — rather than buried at the tail of the first
    # user message, which sinks to the bottom as the trajectory grows. Both arms render the same
    # system template; control leaves recalled_memories empty, so prompts match except for memory.
    # With step_reinject, MemoryAgent rewrites the system message in place each step, so skip the
    # static template block entirely.
    if not study.get("memory", {}).get("step_reinject", False):
        config["agent"]["system_template"] = config["agent"]["system_template"] + _MEMORY_BLOCK
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
    score_per_task: bool = False,
):
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    from .agent_hooks import MemoryAgent, MeteredAgent
    from .memory_glue import MemoryGlue, PlaceboGlue, ReplayGlue
    from .metrics import TaskRecord

    # "placebo" runs the same injection machinery as treatment but with task-irrelevant
    # notes of matched length and no retention — the content-vs-perturbation control.
    # "replay" re-injects a prior run's recorded blocks verbatim — the content-determinism test.
    enabled = arm in ("treatment", "placebo", "replay")
    repo = study["dataset"]["repo_label"]
    bank_id = f"{study['memory']['bank_prefix']}-{repo}-{arm}-s{study['seed']}"

    if arm == "placebo":
        glue: MemoryGlue | PlaceboGlue | ReplayGlue = PlaceboGlue(lengths=study.get("_placebo_lengths"))
    elif arm == "replay":
        glue = ReplayGlue(blocks=study.get("_replay_blocks") or {})
    else:
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
            include_chunks=study["memory"].get("include_chunks", False),
            max_chunk_tokens=study["memory"].get("max_chunk_tokens", 4096),
            orientation_enabled=study["memory"].get("orientation_enabled", True),
            orientation_query=study["memory"].get("orientation_query"),
            retain_style=study["memory"].get("retain_style", "insight"),
        )
    glue.reset_bank()

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    preds_path = arm_dir / "preds.json"
    preds: dict[str, dict] = {}

    # CI-feedback retry loop: a failed task is re-attempted by a FRESH agent up to
    # max_attempts times. Between attempts the official evaluation feedback (failing tests,
    # assertion output, regressions) is retained, so the treatment arm's next attempt starts
    # with verified knowledge of what went wrong — the control arm retries blind. Requires
    # per-task scoring (no outcome → no retry signal).
    max_attempts = max(1, int(study.get("attempts", 1))) if score_per_task else 1

    records: list[TaskRecord] = []
    memdbg: list[dict] = []  # per-attempt recalled/retained content for analysis
    for seq, instance in enumerate(instances, start=1):
        iid = instance["instance_id"]
        task = instance["problem_statement"]
        print(f"\n=== [{arm}] task {seq}/{len(instances)}: {iid} ===", flush=True)

        rec = TaskRecord(arm=arm, seq=seq, instance_id=iid)
        resolved_now: bool | None = None
        for attempt in range(1, max_attempts + 1):
            rec.n_attempts = attempt
            if attempt > 1:
                print(f"  .. attempt {attempt}/{max_attempts} (previous attempt failed the tests)", flush=True)
            if arm == "placebo":
                glue.current_key = (seq, attempt)  # length-match the placebo block to the real run
            elif arm == "replay":
                glue.current_key = (iid, attempt)  # re-inject the recorded block for this attempt

            model = get_model(config=copy.deepcopy(agent_config.get("model", {})))
            env = get_sb_environment(copy.deepcopy(agent_config), instance)
            agent_kwargs = dict(agent_config.get("agent", {}))
            suffix = f".attempt{attempt}" if max_attempts > 1 else ""
            agent_kwargs["output_path"] = arm_dir / iid / f"{iid}{suffix}.traj.json"

            if enabled:
                agent = MemoryAgent(
                    model,
                    env,
                    glue=glue,
                    instance_id=iid,
                    step_reinject=study["memory"].get("step_reinject", False),
                    reinject_every=study["memory"].get("reinject_every", 1),
                    defer_retain=score_per_task,
                    **agent_kwargs,
                )
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

            # Accumulate cost across attempts — the true price of reaching the outcome.
            rec.input_tokens += getattr(agent, "input_tokens", 0)
            rec.output_tokens += getattr(agent, "output_tokens", 0)
            rec.n_steps += agent.n_calls
            rec.cost_usd = round(rec.cost_usd + agent.cost, 4)
            rec.wall_clock_s = round(rec.wall_clock_s + wall, 2)
            rec.exit_status = exit_status
            rec.recalled_chars = len(agent.extra_template_vars.get("recalled_memories", ""))

            preds[iid] = {
                "model_name_or_path": f"hindsight-{arm}",
                "instance_id": iid,
                "model_patch": submission,
            }
            preds_path.write_text(json.dumps(preds, indent=2))

            # Outcome-aware retention: score THIS attempt now (official harness, single
            # instance) so retain knows whether it actually passed the tests — failed
            # approaches must not be stored as expertise, and the evaluation feedback is
            # the lesson material for the next attempt.
            eval_feedback = ""
            if score_per_task:
                from .scoring import extract_eval_feedback, score_predictions

                ds = study["dataset"]
                score_run_id = f"{out_dir.name}-{arm}-t{seq}a{attempt}"
                print(f"  .. scoring {iid} attempt {attempt}", flush=True)
                resolved_now = score_predictions(
                    preds_path=preds_path,
                    instance_ids=[iid],
                    run_id=score_run_id,
                    dataset_name=DATASET_MAPPING.get(ds["subset"], ds["subset"]),
                    split=ds.get("split", "test"),
                    python_executable=os.environ.get("SWEBENCH_PYTHON"),
                    max_workers=1,
                    timeout=study.get("scoring", {}).get("timeout", 1800),
                ).get(iid, False)
                if enabled:
                    eval_feedback = extract_eval_feedback(
                        workdir=preds_path.parent,
                        run_id=score_run_id,
                        model_name=f"hindsight-{arm}",
                        instance_id=iid,
                    )
                    glue.retain_after_task(
                        iid,
                        agent.transcript_text(),
                        resolved=resolved_now,
                        eval_feedback=eval_feedback or None,
                        attempt=attempt,
                    )
            if resolved_now:
                rec.resolved_at_attempt = attempt

            # Content analysis: exactly what was RECALLED, what the eval said, what was RETAINED.
            if enabled:
                memdbg.append(
                    {
                        "seq": seq,
                        "instance_id": iid,
                        "attempt": attempt,
                        "problem_statement": task[:1200],
                        "resolved": resolved_now,
                        "recalled_injected": agent.extra_template_vars.get("recalled_memories", ""),
                        "eval_feedback": eval_feedback,
                        "retained_summary": glue.last_retained_summary,
                    }
                )
                (arm_dir / "memory_debug.json").write_text(json.dumps(memdbg, indent=2))
            print(
                f"  -> attempt {attempt}: steps={agent.n_calls} "
                f"tokens={getattr(agent, 'input_tokens', 0) + getattr(agent, 'output_tokens', 0)} "
                f"recalled_chars={rec.recalled_chars} exit={exit_status}"
                + (f" resolved={resolved_now}" if resolved_now is not None else ""),
                flush=True,
            )
            if resolved_now:
                break

        rec.resolved = bool(resolved_now)
        records.append(rec)
        # Dump records incrementally so an interrupted run never loses completed-task data.
        (arm_dir / "records.json").write_text(json.dumps([r.as_dict() for r in records], indent=2))
        print(
            f"  => {iid}: resolved={rec.resolved} attempts={rec.n_attempts} "
            f"total_steps={rec.n_steps} total_tokens={rec.total_tokens}",
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
    ap.add_argument(
        "--agent-model", default=None, help="Override the agent model_name (e.g. vertex_ai/gemini-3.1-pro-preview)"
    )
    ap.add_argument("--context-mode", choices=["recall", "reflect"], default=None, help="Override memory.context_mode")
    ap.add_argument(
        "--recall-types",
        default=None,
        help="Override memory.recall_types: 'all' or comma-separated (e.g. 'observation')",
    )
    ap.add_argument(
        "--include-chunks",
        action="store_true",
        help="Recall the raw source chunks too, injected paired with each fact (coding precision)",
    )
    ap.add_argument(
        "--step-reinject",
        action="store_true",
        help="Re-recall against the agent's current focus each step and refresh memory in place",
    )
    ap.add_argument("--reinject-every", type=int, default=None, help="Refresh cadence for --step-reinject (steps)")
    ap.add_argument(
        "--retain-style",
        choices=["insight", "procedural"],
        default=None,
        help="Override memory.retain_style: what retain distils (procedural = outcome-aware "
        "working-practice lessons; failed tasks store only process traps)",
    )
    ap.add_argument(
        "--score-per-task",
        action="store_true",
        help="Score each treatment task right after it runs (official harness) and retain "
        "AFTER scoring, so retention is gated/framed by the real test outcome",
    )
    ap.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Max attempts per task (CI-feedback retry loop): a failed task is re-attempted "
        "by a fresh agent; treatment retains the official eval feedback between attempts, "
        "control retries blind. Implies per-task scoring for BOTH arms.",
    )
    ap.add_argument(
        "--arms",
        default="control,treatment",
        help="Comma-separated arms to run (control, treatment, placebo). 'placebo' injects "
        "task-IRRELEVANT notes of matched length instead of real memories — the "
        "content-vs-perturbation control.",
    )
    ap.add_argument(
        "--placebo-from",
        default=None,
        help="Path to a real run's treatment memory_debug.json; the placebo arm length-matches "
        "its injected blocks per (seq, attempt)",
    )
    ap.add_argument(
        "--replay-from",
        default=None,
        help="Path to a real run's treatment memory_debug.json; the replay arm re-injects its "
        "recorded blocks VERBATIM per (instance_id, attempt) — the content-determinism test",
    )
    ap.add_argument(
        "--only-instances",
        default=None,
        help="Comma-separated instance_ids: run only these tasks (in sequence order). Useful "
        "with --replay-from to re-test specific flips without re-running the whole set.",
    )
    ap.add_argument(
        "--control-from",
        default=None,
        help="Reuse a prior run's control arm from its results.json (control is "
        "deterministic at temperature 0, so re-running it per config is wasted "
        "compute). Must be the SAME task-set, model, and step_limit.",
    )
    ap.add_argument("--skip-score", action="store_true", help="Skip official Docker scoring")
    ap.add_argument(
        "--run-id",
        default=None,
        help="Override the run id / results dir name (default: {run_id_prefix}-s{seed}). Use "
        "when running a variant so it doesn't overwrite the run it's compared against.",
    )
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
    if args.include_chunks:
        study["memory"]["include_chunks"] = True
    if args.step_reinject:
        study["memory"]["step_reinject"] = True
    if args.reinject_every is not None:
        study["memory"]["reinject_every"] = args.reinject_every
    if args.retain_style is not None:
        study["memory"]["retain_style"] = args.retain_style
    if args.attempts is not None:
        study["attempts"] = args.attempts
    attempts = int(study.get("attempts", 1))
    if attempts > 1 and args.skip_score:
        raise SystemExit("--attempts > 1 requires scoring (the retry signal IS the test outcome)")
    if args.placebo_from:
        dbg = json.loads(Path(args.placebo_from).read_text())
        study["_placebo_lengths"] = {
            (e["seq"], e.get("attempt", 1)): len(e.get("recalled_injected") or "") for e in dbg
        }
    if args.replay_from:
        dbg = json.loads(Path(args.replay_from).read_text())
        study["_replay_blocks"] = {
            (e["instance_id"], e.get("attempt", 1)): (e.get("recalled_injected") or "") for e in dbg
        }
    # Per-task scoring is meaningless without scoring at all. Retries imply it for both arms.
    score_per_task = (args.score_per_task or attempts > 1) and not args.skip_score

    agent_config = build_agent_config(study)
    if args.agent_model is not None:
        agent_config.setdefault("model", {})["model_name"] = args.agent_model
    if args.step_limit is not None:
        agent_config.setdefault("agent", {})["step_limit"] = args.step_limit
    instances = load_task_sequence(study)
    if args.only_instances:
        only = {i.strip() for i in args.only_instances.split(",") if i.strip()}
        instances = [r for r in instances if r["instance_id"] in only]
        missing = only - {r["instance_id"] for r in instances}
        if missing:
            raise SystemExit(f"--only-instances not in the loaded task sequence: {sorted(missing)}")
    print(f"Loaded {len(instances)} tasks: {[r['instance_id'] for r in instances]}", flush=True)

    run_id = args.run_id or f"{study['run_id_prefix']}-s{study['seed']}"
    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_records: dict[str, list] = {}
    arm_mem: dict[str, dict] = {}
    arm_preds: dict[str, Path] = {}
    scored_arms: list[str] = []  # arms that still need official scoring

    # Reuse a control arm from a prior run instead of re-running + re-scoring it. Control is
    # only near-deterministic, and ONLY for an identical pipeline: any change to the rendered
    # templates (even whitespace) deterministically shifts every trajectory. (Learned the hard
    # way: a system-template tweak between runs turned a 137-step control into 72 steps and
    # silently inflated every treatment delta computed against the stale one.)
    import hashlib

    pipeline_fp = hashlib.sha256(
        json.dumps(
            {
                "system_template": agent_config.get("agent", {}).get("system_template", ""),
                "instance_template": agent_config.get("agent", {}).get("instance_template", ""),
                "model": agent_config.get("model", {}).get("model_name"),
                "step_limit": agent_config.get("agent", {}).get("step_limit"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

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
        prior_fp = prior.get("config", {}).get("pipeline_fingerprint")
        if prior_fp != pipeline_fp:
            raise SystemExit(
                f"--control-from pipeline mismatch: prior run fingerprint {prior_fp!r} != current "
                f"{pipeline_fp!r} (templates/model/step_limit differ, or the prior run predates "
                f"fingerprinting). A control from a different pipeline is NOT a valid baseline — "
                f"re-run the control arm."
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
        # In retry mode BOTH arms score per task (the retry signal is the outcome);
        # otherwise per-task scoring is a treatment-only concern (retention gating).
        arm_scored_per_task = score_per_task and (attempts > 1 or arm == "treatment")
        recs, mem, preds_path = run_arm(
            arm=arm,
            instances=instances,
            agent_config=agent_config,
            study=study,
            out_dir=out_dir,
            score_per_task=arm_scored_per_task,
        )
        arm_records[arm] = recs
        arm_mem[arm] = mem
        arm_preds[arm] = preds_path
        if not arm_scored_per_task:  # per-task-scored arms already carry resolved flags
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
            "attempts": attempts,
            "pipeline_fingerprint": pipeline_fp,
        },
        control=arm_records.get("control", []),
        treatment=arm_records.get("treatment", []),
        control_mem=arm_mem.get("control", {}),
        treatment_mem=arm_mem.get("treatment", {}),
    )
    # Extra arms (placebo, replay) sit outside the control/treatment pairing — attach explicitly.
    for extra in ("placebo", "replay"):
        if extra in arm_records:
            from .metrics import summarize_arm

            results["arms"][extra] = summarize_arm(arm_records[extra], arm_mem.get(extra, {}))
            results["per_task"][extra] = [r.as_dict() for r in arm_records[extra]]

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {results_path}", flush=True)
    print(json.dumps(results.get("headline", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
