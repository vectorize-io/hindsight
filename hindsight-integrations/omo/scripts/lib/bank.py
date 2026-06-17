"""Bank ID derivation and mission management for OMO integration."""

import os
import subprocess
import sys

from .state import read_state, write_state

DEFAULT_BANK_NAME = "omo"
VALID_FIELDS = {"agent", "project", "session", "channel", "user"}


def _resolve_project_name(cwd: str, config: dict) -> str:
    """Resolve the project name from the working directory.

    When resolveWorktrees is enabled, detects git worktrees and resolves
    to the main repository basename so all worktrees share the same bank.
    """
    if not cwd:
        return "unknown"

    if not config.get("resolveWorktrees", True):
        return os.path.basename(cwd)

    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_common_dir = result.stdout.strip()
            main_repo_path = os.path.dirname(git_common_dir)
            return os.path.basename(main_repo_path)
    except (OSError, subprocess.TimeoutExpired):
        pass

    return os.path.basename(cwd)


def derive_bank_id(hook_input: dict, config: dict) -> str:
    """Derive a bank ID from hook context and config.

    Resolution order:
      1. directoryBankMap — explicit directory→bank mapping
      2. Static mode (dynamicBankId=false) — single bank
      3. Dynamic mode (dynamicBankId=true) — composed from granularity fields
    """
    prefix = config.get("bankIdPrefix", "")

    cwd = hook_input.get("cwd", "")
    dir_map = config.get("directoryBankMap") or {}
    if cwd and dir_map:
        normalized_cwd = os.path.normpath(cwd)
        for dir_path, bank_id in dir_map.items():
            if os.path.normpath(dir_path) == normalized_cwd:
                return f"{prefix}-{bank_id}" if prefix else bank_id

    if not config.get("dynamicBankId", False):
        base = config.get("bankId") or DEFAULT_BANK_NAME
        return f"{prefix}-{base}" if prefix else base

    fields = config.get("dynamicBankGranularity")
    if not fields or not isinstance(fields, list):
        fields = ["agent", "project"]

    for f in fields:
        if f not in VALID_FIELDS:
            print(
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" — valid: {", ".join(sorted(VALID_FIELDS))}',
                file=sys.stderr,
            )

    session_id = hook_input.get("session_id", "")
    agent_name = config.get("agentName", "omo")
    channel_id = os.environ.get("HINDSIGHT_CHANNEL_ID", "")
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")

    field_map = {
        "agent": agent_name,
        "project": _resolve_project_name(cwd, config),
        "session": session_id or "unknown",
        "channel": channel_id or "default",
        "user": user_id or "anonymous",
    }

    segments = [field_map.get(f, "unknown") for f in fields]
    base_bank_id = "::".join(segments)

    return f"{prefix}-{base_bank_id}" if prefix else base_bank_id


def ensure_bank_mission(client, bank_id: str, config: dict, debug_fn=None):
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
