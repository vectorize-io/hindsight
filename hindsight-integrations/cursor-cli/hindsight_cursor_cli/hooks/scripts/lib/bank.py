"""Bank ID derivation and mission management.

Cursor CLI context dimensions:
  - agent   → configured name or "cursor-cli" (HINDSIGHT_AGENT_NAME)
  - project → derived from cwd / CURSOR_PROJECT_DIR / first workspace_roots
  - user    → from env var HINDSIGHT_USER_ID

Unlike Codex, the Cursor `stop` hook fires fire-and-forget without a
session-specific transcript path, so we don't include "session" as a
default granularity field. Users can still opt-in via configuration.

The channel dimension is omitted — Cursor CLI is a CLI tool without
multi-channel routing like Telegram/Discord agents.
"""

import os
import sys

from .state import read_state, write_state

DEFAULT_BANK_NAME = "cursor-cli"

# Valid granularity fields for the Cursor CLI integration.
# "session" is exposed but not part of the default granularity because the
# `stop` hook is fire-and-forget and doesn't carry a session_id by default.
VALID_FIELDS = {"agent", "project", "gitProject", "session", "user"}


def _resolve_project_name(hook_input):
    """Resolve the project name for bank-id derivation.

    Priority:
      1. CURSOR_PROJECT_DIR env var (Cursor sets this for every hook)
      2. hook_input.workspace_roots[0] (common field, when present)
      3. hook_input.cwd (set by some hooks)

    Returns "unknown" when no project source is available so bank IDs
    stay stable across test runners (avoids leaking the test runner's
    cwd into the bank name).
    """
    env_project = os.environ.get("CURSOR_PROJECT_DIR", "").strip()
    if env_project:
        return os.path.basename(env_project) or "unknown"

    if isinstance(hook_input, dict):
        roots = hook_input.get("workspace_roots")
        if isinstance(roots, list) and roots:
            first = roots[0]
            if isinstance(first, str) and first:
                return os.path.basename(first) or "unknown"

        cwd = hook_input.get("cwd", "")
        if cwd:
            return os.path.basename(cwd) or "unknown"

    return "unknown"


def _resolve_session_id(hook_input):
    """Resolve a stable session identifier from the hook payload."""
    if not isinstance(hook_input, dict):
        return "unknown"
    return hook_input.get("session_id") or hook_input.get("conversation_id") or "unknown"


def derive_bank_id(hook_input, config):
    """Derive a bank ID from hook context and config.

    When `dynamicBankId` is false, returns the static bank. When true,
    composes from granularity fields joined by '::'.
    """
    prefix = config.get("bankIdPrefix", "")

    if not config.get("dynamicBankId", False):
        base = config.get("bankId") or DEFAULT_BANK_NAME
        return f"{prefix}-{base}" if prefix else base

    fields = config.get("dynamicBankGranularity")
    if not fields or not isinstance(fields, list):
        fields = ["agent", "project"]

    for f in fields:
        if f not in VALID_FIELDS:
            print(
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" — '
                f"valid for cursor-cli: {', '.join(sorted(VALID_FIELDS))}",
                file=sys.stderr,
            )

    agent_name = config.get("agentName", "cursor-cli")
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")
    session_id = _resolve_session_id(hook_input)
    project_name = _resolve_project_name(hook_input)

    field_map = {
        "agent": agent_name,
        "project": project_name,
        "gitProject": project_name,  # alias for backwards-compat with codex configs
        "session": session_id,
        "user": user_id or "anonymous",
    }

    segments = [field_map.get(f, "unknown") for f in fields]
    base_bank_id = "::".join(segments)
    return f"{prefix}-{base_bank_id}" if prefix else base_bank_id


def ensure_bank_mission(client, bank_id, config, debug_fn=None):
    """Set bank mission on first use, skip if already set."""
    mission = config.get("bankMission", "")
    if not mission or not mission.strip():
        return

    missions_set = read_state("bank_missions.json", {})
    if bank_id in missions_set:
        return

    try:
        retain_mission = config.get("retainMission")
        client.set_bank_mission(bank_id, mission, retain_mission=retain_mission, timeout=10)
        missions_set[bank_id] = True
        if len(missions_set) > 10000:
            keys = sorted(missions_set.keys())
            for k in keys[: len(keys) // 2]:
                del missions_set[k]
        write_state("bank_missions.json", missions_set)
        if debug_fn:
            debug_fn(f"Set mission for bank: {bank_id}")
    except Exception as e:
        if debug_fn:
            debug_fn(f"Could not set bank mission for {bank_id}: {e}")
