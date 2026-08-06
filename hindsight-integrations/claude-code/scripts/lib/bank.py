"""Bank ID derivation and mission management.

Port of Openclaw's deriveBankId() and banksWithMissionSet logic, adapted
for Claude Code's context model.

Openclaw derives bank IDs from: agent, channel, user, provider.
Claude Code equivalent dimensions:
  - agent   → configured name or "claude-code" (HINDSIGHT_AGENT_NAME)
  - project → derived from cwd (working directory basename)
  - session → session_id from hook input
  - channel → from env var HINDSIGHT_CHANNEL_ID (for Telegram/Discord agents)
  - user    → from env var HINDSIGHT_USER_ID (for multi-user agents)

The channel/user dimensions enable the same per-user/per-channel isolation
that Openclaw provides via its messageProvider/channelId/senderId context.
Telegram/Discord agents set HINDSIGHT_CHANNEL_ID and HINDSIGHT_USER_ID in
their environment to achieve equivalent behavior.
"""

import os
import subprocess
import sys

from .state import read_state, write_state

DEFAULT_BANK_NAME = "claude-code"

# Valid granularity fields for Claude Code
VALID_FIELDS = {"agent", "project", "session", "channel", "user"}


def _resolve_project_name(cwd: str, config: dict) -> str:
    """Resolve the project name from the working directory.

    When resolveWorktrees is enabled (default), detects git worktrees and
    resolves to the main repository basename so that all worktrees of the
    same repo share the same bank.

    For a regular repo at /home/user/myproject:
        git-common-dir → /home/user/myproject/.git → basename "myproject"

    For a worktree at /home/user/myproject-wt1 linked to /home/user/myproject:
        git-common-dir → /home/user/myproject/.git → basename "myproject"
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
            # git-common-dir returns the .git directory of the main repo
            # e.g. /home/user/myproject/.git → parent is /home/user/myproject
            main_repo_path = os.path.dirname(git_common_dir)
            return os.path.basename(main_repo_path)
    except (OSError, subprocess.TimeoutExpired):
        pass

    # Fallback: not a git repo or git not available
    return os.path.basename(cwd)


def derive_bank_id(hook_input: dict, config: dict) -> str:
    """Derive a bank ID from hook context and config.

    Port of: deriveBankId() in index.js

    Resolution order:
      1. directoryBankMap — explicit directory→bank mapping (highest priority)
      2. Static mode (dynamicBankId=false) — single bank for everything
      3. Dynamic mode (dynamicBankId=true) — composed from granularity fields

    Args:
        hook_input: The hook's stdin JSON (has session_id, cwd).
        config: Plugin configuration dict.
    """
    prefix = config.get("bankIdPrefix", "")

    # Check explicit directory-to-bank mapping first
    cwd = hook_input.get("cwd", "")
    dir_map = config.get("directoryBankMap") or {}
    if cwd and dir_map:
        # Normalize cwd for matching (trailing slashes, separators, and —
        # via normcase — drive-letter/path case on Windows, where launchers
        # disagree: PowerShell and git-bash hand children an UPPERCASE drive
        # while the VS Code extension spawn reports lowercase, so a
        # case-sensitive compare silently misses the map and falls through
        # to the default bank. normcase is a no-op on POSIX, preserving
        # case-sensitive matching there.
        normalized_cwd = os.path.normcase(os.path.realpath(cwd))
        for dir_path, bank_id in dir_map.items():
            if os.path.normcase(os.path.realpath(dir_path)) == normalized_cwd:
                return f"{prefix}-{bank_id}" if prefix else bank_id

    if not config.get("dynamicBankId", False):
        # Static mode — single bank for everything
        base = config.get("bankId") or DEFAULT_BANK_NAME
        return f"{prefix}-{base}" if prefix else base

    # Dynamic mode — compose from granularity fields
    fields = config.get("dynamicBankGranularity")
    if not fields or not isinstance(fields, list):
        fields = ["agent", "project"]

    # Warn on unknown fields (mirrors Openclaw's runtime check)
    for f in fields:
        if f not in VALID_FIELDS:
            print(
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" — '
                f"valid for Claude Code: {', '.join(sorted(VALID_FIELDS))}",
                file=sys.stderr,
            )

    # Build field values from hook context + env vars
    session_id = hook_input.get("session_id", "")
    agent_name = config.get("agentName", "claude-code")

    # Channel and user come from environment variables, set by the host agent
    # (e.g. Telegram bot sets HINDSIGHT_CHANNEL_ID=telegram-group-12345)
    channel_id = os.environ.get("HINDSIGHT_CHANNEL_ID", "")
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")

    field_map = {
        "agent": agent_name,
        "project": _resolve_project_name(cwd, config),
        "session": session_id or "unknown",
        "channel": channel_id or "default",
        "user": user_id or "anonymous",
    }

    # bank_id is stored as-is server-side; HTTP path encoding is the client layer's job.
    segments = [field_map.get(f, "unknown") for f in fields]
    base_bank_id = "::".join(segments)

    return f"{prefix}-{base_bank_id}" if prefix else base_bank_id


def _mark_bank_reconciled(bank_id: str) -> None:
    """Record that we've reconciled missions for a bank (local perf cache).

    This is only an optimization to avoid a GET on every hook invocation — it
    is NOT the correctness boundary. Correctness comes from checking the
    server's per-bank overrides before writing (see ensure_bank_mission).
    """
    missions_set = read_state("bank_missions.json", {})
    missions_set[bank_id] = True
    # Cap tracked banks (mirrors Openclaw's MAX_TRACKED_BANK_CLIENTS)
    if len(missions_set) > 10000:
        keys = sorted(missions_set.keys())
        for k in keys[: len(keys) // 2]:
            del missions_set[k]
    write_state("bank_missions.json", missions_set)


def ensure_bank_mission(client, bank_id: str, config: dict, debug_fn=None):
    """Seed configured missions onto a bank, filling only fields not already set.

    Intent: guarantee a bank has reflect/retain missions before its first use,
    using the plugin's configured defaults. But a bank shared by multiple
    parties may already have missions authored out-of-band (control plane UI or
    `PATCH /banks/{id}/config`). Those must win — the plugin only fills the
    gaps, it never overwrites an existing per-bank mission.

    To decide what is "already set" we inspect the bank's *overrides* (the
    bank-specific config), not the fully-resolved config, so a global/tenant
    default doesn't suppress seeding a brand-new bank.

    A local state file caches which banks we've reconciled to avoid a GET on
    every hook, but it is only an optimization: correctness is enforced by the
    server-side check, so a wiped state file (fresh machine, cleared cache)
    can no longer cause a clobber the way the old unconditional PATCH did.
    """
    reflect_mission = (config.get("bankMission") or "").strip()
    retain_mission = (config.get("retainMission") or "").strip()
    if not reflect_mission and not retain_mission:
        # Nothing configured → explicit opt-out; manage missions out-of-band.
        return

    # Fast path: already reconciled this bank in a previous hook invocation.
    if bank_id in read_state("bank_missions.json", {}):
        return

    try:
        existing = client.get_bank_config(bank_id, timeout=10)
        overrides = existing.get("overrides") or {}
        server_reflect = (overrides.get("reflect_mission") or "").strip()
        server_retain = (overrides.get("retain_mission") or "").strip()

        # Only send a mission field the bank doesn't already have set.
        reflect_update = reflect_mission if (reflect_mission and not server_reflect) else None
        retain_update = retain_mission if (retain_mission and not server_retain) else None

        if reflect_update is not None or retain_update is not None:
            client.set_bank_mission(
                bank_id,
                reflect_mission=reflect_update,
                retain_mission=retain_update,
                timeout=10,
            )
            if debug_fn:
                fields = [
                    name
                    for name, val in (("reflect", reflect_update), ("retain", retain_update))
                    if val is not None
                ]
                debug_fn(f"Seeded {', '.join(fields)} mission(s) for bank: {bank_id}")
        elif debug_fn:
            debug_fn(f"Bank {bank_id} already has missions set; leaving them unchanged")

        # Mark reconciled either way so we don't re-GET on every hook.
        _mark_bank_reconciled(bank_id)
    except Exception as e:
        # Don't fail if reconciliation fails — bank might not exist yet, or the
        # config API may be disabled. We deliberately do NOT mark the bank
        # reconciled here, so a transient failure is retried on the next hook.
        if debug_fn:
            debug_fn(f"Could not reconcile bank missions for {bank_id}: {e}")
