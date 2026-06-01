"""Streamlit UI for browsing mission-sandbox projects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Project path is passed as CLI arg by the `mission-sandbox ui` command
if len(sys.argv) > 1:
    PROJECT_PATH = Path(sys.argv[1])
else:
    PROJECT_PATH = None


def load_project(path: Path) -> dict:
    meta_file = path / "project.json"
    if not meta_file.exists():
        return {}
    return json.loads(meta_file.read_text())


def load_steps(path: Path) -> list[dict]:
    history_dir = path / "history"
    if not history_dir.exists():
        return []
    steps = []
    for f in sorted(history_dir.glob("*.json")):
        raw = json.loads(f.read_text())
        raw["_file"] = f.name
        steps.append(raw)
    return steps


def render_step_init(step: dict) -> None:
    docs = step.get("documents", [])
    obs = step.get("observations", [])
    st.write(f"**Documents ingested:** {len(docs)}")
    if docs:
        with st.expander(f"Documents ({len(docs)})"):
            for d in docs:
                st.code(d, language=None)
    st.write(f"**Observations produced:** {len(obs)}")
    if obs:
        with st.expander(f"Observations ({len(obs)})"):
            for o in obs:
                st.markdown(f"- {o['text']}")


def render_step_label(step: dict) -> None:
    st.write(f"**Model:** `{step.get('model', '?')}`")
    st.write("**Instructions:**")
    st.info(step.get("instructions", ""))
    obs = step.get("observations", [])
    good = [o for o in obs if o.get("label") == "good"]
    bad = [o for o in obs if o.get("label") == "bad"]
    st.write(f"**Results:** {len(good)} good, {len(bad)} bad out of {len(obs)} total")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Good")
        for o in good:
            with st.container(border=True):
                st.markdown(o["text"])
                st.caption(o.get("reason", ""))
    with col2:
        st.markdown("##### Bad")
        for o in bad:
            with st.container(border=True):
                st.markdown(o["text"])
                st.caption(o.get("reason", ""))


def render_step_optimize(step: dict) -> None:
    st.write(f"**Model:** `{step.get('model', '?')}`")
    st.write(f"**Input:** {step.get('good_count', 0)} good, {step.get('bad_count', 0)} bad examples")

    prev = step.get("previous_mission")
    proposed = step.get("proposed_mission", "")

    if prev:
        st.write("**Previous mission:**")
        st.warning(prev)
    else:
        st.write("**Previous mission:** _(none -- baseline)_")

    st.write("**Proposed mission:**")
    st.success(proposed)


def render_step_run(step: dict) -> None:
    mission = step.get("mission_applied", "")
    obs = step.get("observations", [])
    st.write("**Mission applied:**")
    st.info(mission)
    st.write(f"**Observations produced:** {len(obs)}")
    if obs:
        with st.expander(f"Observations ({len(obs)})"):
            for o in obs:
                st.markdown(f"- {o['text']}")


def render_step_promote(step: dict) -> None:
    st.write(f"**Target bank:** `{step.get('target_bank', '?')}`")
    st.write(f"**Backfill:** {'Yes' if step.get('backfill') else 'No'}")
    st.write("**Mission:**")
    st.success(step.get("mission", ""))


STEP_RENDERERS = {
    "init": render_step_init,
    "label": render_step_label,
    "optimize": render_step_optimize,
    "run": render_step_run,
    "promote": render_step_promote,
}

STEP_ICONS = {
    "init": ":inbox_tray:",
    "label": ":label:",
    "optimize": ":sparkles:",
    "run": ":arrows_counterclockwise:",
    "promote": ":rocket:",
}


def main() -> None:
    st.set_page_config(page_title="Mission Sandbox", page_icon=":microscope:", layout="wide")
    st.title(":microscope: Mission Sandbox")

    if PROJECT_PATH is None:
        st.error("No project path provided. Run: `mission-sandbox ui <project-dir>`")
        return

    project = load_project(PROJECT_PATH)
    if not project:
        st.error(f"No project found at `{PROJECT_PATH}`")
        return

    steps = load_steps(PROJECT_PATH)

    # -- Sidebar: project info -------------------------------------------------
    with st.sidebar:
        st.header("Project")
        st.write(f"**Path:** `{PROJECT_PATH}`")
        st.write(f"**Bank:** `{project.get('bank_id', '?')}`")
        st.write(f"**API:** `{project.get('api_url', '?')}`")
        st.write(f"**Created:** {project.get('created_at', '?')[:19]}")

        current_mission = project.get("mission")
        if current_mission:
            st.divider()
            st.subheader("Current Mission")
            st.info(current_mission)

        current_obs = project.get("observations", [])
        if current_obs:
            st.divider()
            st.subheader("Current Observations")
            good = sum(1 for o in current_obs if o.get("label") == "good")
            bad = sum(1 for o in current_obs if o.get("label") == "bad")
            unlabeled = sum(1 for o in current_obs if o.get("label") is None)
            st.write(f"Total: **{len(current_obs)}**")
            if good or bad:
                st.write(f":white_check_mark: {good} good  :x: {bad} bad  :grey_question: {unlabeled} unlabeled")

        st.divider()
        st.subheader("Steps")
        st.write(f"**{len(steps)}** history entries")

    # -- Main: history timeline ------------------------------------------------
    if not steps:
        st.info("No history yet. Run `mission-sandbox init` to get started.")
        return

    # Summary metrics across rounds
    label_rounds = [s for s in steps if s["type"] == "label"]
    optimize_rounds = [s for s in steps if s["type"] == "optimize"]
    if label_rounds:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Label rounds", len(label_rounds))
        with col2:
            st.metric("Optimize rounds", len(optimize_rounds))
        with col3:
            # Show improvement: good ratio in first vs last label round
            first_label = label_rounds[0]
            last_label = label_rounds[-1]
            first_obs = first_label.get("observations", [])
            last_obs = last_label.get("observations", [])
            first_good = sum(1 for o in first_obs if o.get("label") == "good") / max(len(first_obs), 1)
            last_good = sum(1 for o in last_obs if o.get("label") == "good") / max(len(last_obs), 1)
            st.metric(
                "Good ratio",
                f"{last_good:.0%}",
                delta=f"{last_good - first_good:+.0%}" if len(label_rounds) > 1 else None,
            )

    st.divider()

    # Render each step
    for i, step in enumerate(steps):
        step_type = step.get("type", "unknown")
        icon = STEP_ICONS.get(step_type, ":question:")
        timestamp = step.get("timestamp", "")[:19].replace("T", " ")
        filename = step.get("_file", "")

        with st.expander(
            f"{icon} **Step {i + 1}: {step_type.upper()}** — {timestamp}  `{filename}`", expanded=(i == len(steps) - 1)
        ):
            renderer = STEP_RENDERERS.get(step_type)
            if renderer:
                renderer(step)
            else:
                st.json(step)


if __name__ == "__main__":
    main()
