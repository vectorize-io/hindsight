#!/usr/bin/env python3
"""PreToolUse hook: inject bank_id into agent_knowledge_* MCP tool calls.

Intercepts mcp__hindsight__agent_knowledge_* tool calls and injects
the resolved bank_id into tool_input, using the same derivation logic
as the recall/retain hooks (config chain + cwd context).

Exit codes:
  0 — always (allow the tool call to proceed)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.bank import derive_bank_id
from lib.config import debug_log, load_config


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    tool_input = hook_input.get("tool_input", {})

    # Skip if bank_id already provided (explicit override)
    if tool_input.get("bank_id"):
        return

    config = load_config()

    if not config.get("enableKnowledgeTools"):
        return

    bank_id = derive_bank_id(hook_input, config)
    debug_log(config, f"Injecting bank_id={bank_id} into {hook_input.get('tool_name')}")

    # Return updatedInput with bank_id injected
    updated = dict(tool_input)
    updated["bank_id"] = bank_id

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": updated,
        }
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Hindsight] inject_bank_id error: {e}", file=sys.stderr)
        sys.exit(0)
