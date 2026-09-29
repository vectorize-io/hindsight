"""Bank ID derivation and mission management for Cursor plugin.

Supports static bank IDs ("cursor") or dynamic bank IDs derived from
the Cursor session context.
"""

import os
import sys
import urllib.parse

from .state import read_state, write_state

DEFAULT_BANK_NAME = "cursor"

VALID_FIELDS = {"agent", "project", "session", "channel", "user"}


def derive_bank_id(hook_input: dict, config: dict) -> str:
    """Derive a bank ID from hook context and config.

    When dynamicBankId is false, returns the static bank.
    When true, composes from granularity fields joined by '::'.
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
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" -- '
                f"valid for Cursor: {', '.join(sorted(VALID_FIELDS))}",
                file=sys.stderr,
            )

    cwd = hook_input.get("cwd", "")
    session_id = hook_input.get("conversation_id") or hook_input.get("session_id", "")
    agent_name = config.get("agentName", "cursor")
    channel_id = os.environ.get("HINDSIGHT_CHANNEL_ID", "")
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")

    field_map = {
        "agent": agent_name,
        "project": os.path.basename(cwd) if cwd else "unknown",
        "session": session_id or "unknown",
        "channel": channel_id or "default",
        "user": user_id or "anonymous",
    }

    segments = [urllib.parse.quote(field_map.get(f, "unknown"), safe="") for f in fields]
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
