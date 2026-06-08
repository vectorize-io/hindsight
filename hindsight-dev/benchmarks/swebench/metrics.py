"""Per-task records, aggregation, and the warm-up curve.

Headline metric is *efficiency at equal quality*: at equal-or-better resolve rate, a
memory-backed agent should spend fewer tokens / steps / seconds as the session warms up.
On emulated (non-native) Docker, treat ``wall_clock_s`` as indicative only — tokens and
steps are LLM-side and stay valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TaskRecord:
    arm: str
    seq: int  # position in the consecutive session (1-based)
    instance_id: str
    resolved: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    n_steps: int = 0
    cost_usd: float = 0.0
    wall_clock_s: float = 0.0
    exit_status: str = ""
    recalled_chars: int = 0  # size of the injected memory block (0 for control)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


def _safe_mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def summarize_arm(records: list[TaskRecord], memory_stats: dict | None = None) -> dict:
    n = len(records)
    resolved = [r for r in records if r.resolved]
    return {
        "n_tasks": n,
        "resolved": len(resolved),
        "resolve_rate": round(len(resolved) / n, 4) if n else 0.0,
        "total_input_tokens": sum(r.input_tokens for r in records),
        "total_output_tokens": sum(r.output_tokens for r in records),
        "total_tokens": sum(r.total_tokens for r in records),
        "total_steps": sum(r.n_steps for r in records),
        "total_cost_usd": round(sum(r.cost_usd for r in records), 4),
        "total_wall_clock_s": round(sum(r.wall_clock_s for r in records), 2),
        "mean_tokens_per_task": _safe_mean([r.total_tokens for r in records]),
        "mean_steps_per_task": _safe_mean([float(r.n_steps) for r in records]),
        "mean_wall_clock_s": _safe_mean([r.wall_clock_s for r in records]),
        "memory_layer": memory_stats or {},
    }


def warm_up_curve(control: list[TaskRecord], treatment: list[TaskRecord]) -> list[dict]:
    """Per-sequence-position comparison — the money chart.

    The gap should be ~0 at seq 1 (empty memory) and grow as the session warms up.
    Matched by ``seq`` so both arms share the identical task order.
    """
    by_seq_t = {r.seq: r for r in treatment}
    rows = []
    for c in sorted(control, key=lambda r: r.seq):
        t = by_seq_t.get(c.seq)
        if t is None:
            continue
        rows.append(
            {
                "seq": c.seq,
                "instance_id": c.instance_id,
                "control_tokens": c.total_tokens,
                "treatment_tokens": t.total_tokens,
                "token_delta": t.total_tokens - c.total_tokens,
                "token_delta_pct": _pct(c.total_tokens, t.total_tokens),
                "control_steps": c.n_steps,
                "treatment_steps": t.n_steps,
                "step_delta": t.n_steps - c.n_steps,
                "control_resolved": c.resolved,
                "treatment_resolved": t.resolved,
                "treatment_recalled_chars": t.recalled_chars,
            }
        )
    return rows


def _pct(base: int, new: int) -> float | None:
    if base == 0:
        return None
    return round((new - base) / base * 100.0, 2)


def _sign_test_p(wins: int, losses: int) -> float | None:
    """Two-sided exact binomial sign test p-value (no scipy dependency)."""
    n = wins + losses
    if n == 0:
        return None
    from math import comb

    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 4)


def paired_analysis(control: list[TaskRecord], treatment: list[TaskRecord]) -> dict:
    """Per-task paired comparison with distribution + significance — the evidence layer.

    A task counts only if BOTH arms ran it. Resolve flips/regressions and step/token
    win-loss-tie counts are reported with an exact sign-test p-value on the step deltas.
    """
    by_seq_c = {r.seq: r for r in control}
    pairs = [(by_seq_c[t.seq], t) for t in treatment if t.seq in by_seq_c]
    if not pairs:
        return {}

    step_wins = sum(1 for c, t in pairs if t.n_steps < c.n_steps)
    step_losses = sum(1 for c, t in pairs if t.n_steps > c.n_steps)
    step_ties = sum(1 for c, t in pairs if t.n_steps == c.n_steps)
    tok_wins = sum(1 for c, t in pairs if t.total_tokens < c.total_tokens)
    tok_losses = sum(1 for c, t in pairs if t.total_tokens > c.total_tokens)

    flips = [t.instance_id for c, t in pairs if t.resolved and not c.resolved]  # fail->pass
    regressions = [t.instance_id for c, t in pairs if c.resolved and not t.resolved]  # pass->fail
    step_deltas_pct = [_pct(c.n_steps, t.n_steps) for c, t in pairs if c.n_steps > 0]
    step_deltas_pct = [d for d in step_deltas_pct if d is not None]
    median_step_pct = round(sorted(step_deltas_pct)[len(step_deltas_pct) // 2], 1) if step_deltas_pct else None

    return {
        "n_paired_tasks": len(pairs),
        "resolve": {
            "control": sum(1 for c, _ in pairs if c.resolved),
            "treatment": sum(1 for _, t in pairs if t.resolved),
            "fail_to_pass_flips": flips,
            "pass_to_fail_regressions": regressions,
            "net_resolve_change": len(flips) - len(regressions),
        },
        "steps": {
            "improved": step_wins,
            "regressed": step_losses,
            "tied": step_ties,
            "sign_test_p": _sign_test_p(step_wins, step_losses),
            "median_delta_pct": median_step_pct,
        },
        "tokens": {
            "improved": tok_wins,
            "regressed": tok_losses,
            "sign_test_p": _sign_test_p(tok_wins, tok_losses),
        },
    }


def build_results(
    *,
    config: dict,
    control: list[TaskRecord],
    treatment: list[TaskRecord],
    control_mem: dict,
    treatment_mem: dict,
) -> dict:
    c_sum = summarize_arm(control, control_mem)
    t_sum = summarize_arm(treatment, treatment_mem)
    headline = {
        "resolve_rate_delta": round(t_sum["resolve_rate"] - c_sum["resolve_rate"], 4),
        "total_token_delta_pct": _pct(c_sum["total_tokens"], t_sum["total_tokens"]),
        "total_step_delta_pct": _pct(c_sum["total_steps"], t_sum["total_steps"]),
        "mean_tokens_per_task_delta_pct": _pct(c_sum["mean_tokens_per_task"], t_sum["mean_tokens_per_task"]),
    }
    return {
        "config": config,
        "headline": headline,
        "paired_analysis": paired_analysis(control, treatment),
        "arms": {"control": c_sum, "treatment": t_sum},
        "warm_up_curve": warm_up_curve(control, treatment),
        "per_task": {
            "control": [r.as_dict() for r in control],
            "treatment": [r.as_dict() for r in treatment],
        },
    }
