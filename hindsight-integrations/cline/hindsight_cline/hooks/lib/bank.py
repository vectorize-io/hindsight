"""Bank ID derivation and mission management for Cline.

Cline context dimensions:
  - agent   → configured name or "cline"
  - project → basename of the first workspace root
  - session → taskId from the hook input
  - user    → from env var HINDSIGHT_USER_ID
"""

import os
import sys
from collections.abc import Callable
from typing import Optional

from .client import HindsightClient
from .cline_io import HookInput
from .config import HindsightClineConfig
from .state import read_state, write_state

DEFAULT_BANK_NAME = "cline"
VALID_FIELDS = {"agent", "project", "session", "user"}


def derive_bank_id(hook: HookInput, config: HindsightClineConfig) -> str:
    """Derive a bank ID from hook context and config.

    Static (default): the configured bankId. Dynamic: granularity fields
    joined by '::'.
    """
    prefix = config.bank_id_prefix

    if not config.dynamic_bank_id:
        base = config.bank_id or DEFAULT_BANK_NAME
        return f"{prefix}-{base}" if prefix else base

    fields = config.dynamic_bank_granularity
    if not fields or not isinstance(fields, list):
        fields = ["agent", "project"]

    for f in fields:
        if f not in VALID_FIELDS:
            print(
                f'[Hindsight] Unknown dynamicBankGranularity field "{f}" — '
                f"valid for Cline: {', '.join(sorted(VALID_FIELDS))}",
                file=sys.stderr,
            )

    workspace = hook.workspace_roots[0] if hook.workspace_roots else ""
    agent_name = config.agent_name or "cline"
    user_id = os.environ.get("HINDSIGHT_USER_ID", "")

    field_map = {
        "agent": agent_name,
        "project": os.path.basename(workspace.rstrip("/")) if workspace else "unknown",
        "session": hook.task_id or "unknown",
        "user": user_id or "anonymous",
    }

    segments = [field_map.get(f, "unknown") for f in fields]
    base_bank_id = "::".join(segments)
    return f"{prefix}-{base_bank_id}" if prefix else base_bank_id


def ensure_bank_mission(
    client: HindsightClient,
    bank_id: str,
    config: HindsightClineConfig,
    debug_fn: Optional[Callable[..., None]] = None,
) -> None:
    """Set the bank mission on first use; skip if already set (tracked in state)."""
    mission = config.bank_mission
    if not mission or not mission.strip():
        return

    missions_set = read_state("bank_missions.json", {})
    if bank_id in missions_set:
        return

    try:
        retain_mission = config.retain_mission
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
