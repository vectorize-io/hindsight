"""Bank ID derivation and mission management.

Port of the Claude Code plugin's bank.py, adapted for Devin CLI's hook context.

Claude Code hooks receive `cwd` on stdin; Devin CLI hooks don't — instead the
CLI sets a `DEVIN_PROJECT_DIR` environment variable on every hook process. This
module prefers that env var and falls back to `hook_input["cwd"]` for forward
compatibility (and so this module's tests can share fixtures with the Claude
Code plugin's).

Dimensions:
  - agent   → configured name or "devin-cli" (HINDSIGHT_AGENT_NAME)
  - project → derived from DEVIN_PROJECT_DIR (working directory basename)
  - session → session_id from hook input
  - channel → from env var HINDSIGHT_CHANNEL_ID (for parity with other integrations)
  - user    → from env var HINDSIGHT_USER_ID (for multi-user agents)
"""

import os
import subprocess
import sys

from .state import read_state, write_state

DEFAULT_BANK_NAME = "devin-cli"

# Valid granularity fields
VALID_FIELDS = {"agent", "project", "session", "channel", "user"}


def _resolve_cwd(hook_input: dict) -> str:
    """Resolve the project working directory.

    Devin CLI sets DEVIN_PROJECT_DIR in every hook process's environment;
    prefer it over hook_input["cwd"] (which Devin CLI does not send, but which
    other Claude-Code-compatible hosts might).
    """
    return os.environ.get("DEVIN_PROJECT_DIR") or hook_input.get("cwd", "")


def _resolve_project_name(cwd: str, config: dict) -> str:
    """Resolve the project name from the working directory.

    When resolveWorktrees is enabled (default), detects git worktrees and
    resolves to the main repository basename so that all worktrees of the
    same repo share the same bank.
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
    # text=True decodes the child's stdout, so a path with invalid UTF-8 raises
    # UnicodeDecodeError — a ValueError, caught by neither of the others.
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError):
        pass

    return os.path.basename(cwd)


def derive_bank_id(hook_input: dict, config: dict) -> str:
    """Derive a bank ID from hook context and config.

    Resolution order:
      1. directoryBankMap — explicit directory→bank mapping (highest priority)
      2. Static mode (dynamicBankId=false) — single bank for everything
      3. Dynamic mode (dynamicBankId=true) — composed from granularity fields

    Args:
        hook_input: The hook's stdin JSON (has session_id; may have cwd).
        config: Plugin configuration dict.
    """
    prefix = config.get("bankIdPrefix", "")

    cwd = _resolve_cwd(hook_input)
    # `or {}` only catches falsy values, so a truthy non-mapping (a JSON list,
    # a string) still reached .items() and raised out of both hooks. Guarded
    # here as well as centrally in config.py because derive_bank_id() takes a
    # caller-supplied config and cannot assume it came from load_config().
    dir_map = config.get("directoryBankMap")
    if not isinstance(dir_map, dict):
        dir_map = {}
    if cwd and dir_map:
        # normcase is a no-op on POSIX; on Windows it normalizes drive-letter
        # case so mismatched launchers still match the configured map.
        normalized_cwd = os.path.normcase(os.path.realpath(cwd))
        for dir_path, bank_id in dir_map.items():
            # Values are type-checked, not just the map. A non-string value is
            # returned verbatim as the bank id — or, with a bankIdPrefix set,
            # f-string-formatted into one, so `["wrong"]` becomes the literal
            # bank `p-['wrong']`. Either way recall and retain silently address
            # a bank the user did not name. A bad entry falls through to the
            # next resolution branch instead, which lands on a real bank.
            if not isinstance(bank_id, str) or not bank_id:
                continue
            if os.path.normcase(os.path.realpath(dir_path)) == normalized_cwd:
                return f"{prefix}-{bank_id}" if prefix else bank_id

    if not config.get("dynamicBankId", False):
        base = config.get("bankId") or DEFAULT_BANK_NAME
        return f"{prefix}-{base}" if prefix else base

    fields = config.get("dynamicBankGranularity")
    if not fields or not isinstance(fields, list):
        fields = ["agent", "project"]

    # Elements are type-checked, not just the list. VALID_FIELDS and field_map
    # below are both hashed lookups, so an unhashable element — `["agent", []]`
    # — raises TypeError instead of reporting an unknown field, and takes both
    # hooks down with it. A non-string is reported and resolves to "unknown",
    # the same as any other unrecognised field.
    for f in fields:
        if not isinstance(f, str) or f not in VALID_FIELDS:
            print(
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" — '
                f"valid for Devin CLI: {', '.join(sorted(VALID_FIELDS))}",
                file=sys.stderr,
            )

    session_id = hook_input.get("session_id", "")
    agent_name = config.get("agentName", "devin-cli")

    channel_id = os.environ.get("HINDSIGHT_CHANNEL_ID", "")
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")

    field_map = {
        "agent": agent_name,
        "project": _resolve_project_name(cwd, config),
        "session": session_id or "unknown",
        "channel": channel_id or "default",
        "user": user_id or "anonymous",
    }

    segments = [field_map.get(f, "unknown") if isinstance(f, str) else "unknown" for f in fields]
    base_bank_id = "::".join(segments)

    return f"{prefix}-{base_bank_id}" if prefix else base_bank_id


def ensure_bank_mission(client, bank_id: str, config: dict, debug_fn=None):
    """Set bank mission on first use, skip if already set.

    Uses a state file to persist which banks have had their mission set
    across ephemeral hook invocations.
    """
    mission = config.get("bankMission", "")
    if not mission or not mission.strip():
        return

    missions_set = read_state("bank_missions.json", {})
    if bank_id in missions_set:
        return

    try:
        retain_mission = config.get("retainMission")
        client.set_bank_mission(bank_id, mission, retain_mission=retain_mission, timeout=10)
    except Exception as e:
        if debug_fn:
            debug_fn(f"Could not set bank mission for {bank_id}: {e}")
        return

    missions_set[bank_id] = True
    if len(missions_set) > 10000:
        keys = sorted(missions_set.keys())
        for k in keys[: len(keys) // 2]:
            del missions_set[k]

    # Split from the API call above so the two failures are not reported as
    # one. A write_state OSError used to surface as "Could not set bank
    # mission", when the mission had in fact been set and only the record of
    # it was lost. Still swallowed rather than raised: this runs on the
    # recall and retain hook paths, where an unwritable state directory must
    # not abort a hook that has already done its work. The cost is one
    # redundant set_bank_mission per later hook, not a wrong answer.
    try:
        write_state("bank_missions.json", missions_set)
    except OSError as e:
        if debug_fn:
            debug_fn(f"Bank mission set for {bank_id} but not recorded: {e}")
        return

    if debug_fn:
        debug_fn(f"Set mission for bank: {bank_id}")
