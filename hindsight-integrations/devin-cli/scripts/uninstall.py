#!/usr/bin/env python3
"""Remove hindsight-memory's hook and MCP server entries from Devin CLI config.

Counterpart to install.py — only removes entries this plugin registered
(recognised by install.py's `_OURS_RE`, which matches entries written by any
version of the plugin, and by `_is_our_mcp_server()` for the "hindsight" MCP
entry); leaves any other configured hooks/MCP servers untouched.
"""

import sys

from install import (
    CONFIG_PATH,
    DROP_ENTRY,
    MCP_CONFIG_PATH,
    _is_our_mcp_server,
    _load_json,
    _strip_our_hooks,
    _write_json,
)


def uninstall_hooks() -> None:
    config = _load_json(CONFIG_PATH)
    hooks = config.get("hooks")
    # The same shapes install_hooks() guards against: a non-object "hooks", or
    # an event whose value is not a list. There is nothing for an uninstall to
    # remove from either, but reaching .keys() or iterating them still aborts
    # the script on a traceback.
    if not isinstance(hooks, dict):
        return
    changed = False
    for event in list(hooks.keys()):
        entries = hooks[event]
        if not isinstance(entries, list):
            continue
        # Per command, not per entry: a user is free to list this plugin's hook
        # alongside their own under one matcher, and uninstalling us is not
        # permission to delete theirs.
        kept = [stripped for stripped in map(_strip_our_hooks, entries) if stripped is not DROP_ENTRY]
        # Compared by value, not by length. An entry that merely *lost* one of
        # its commands keeps the list the same length, so a length check read
        # the removal as "nothing changed" and never wrote the file — the hook
        # stayed registered and kept running after an uninstall.
        if kept != entries:
            changed = True
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if changed:
        config["hooks"] = hooks
        _write_json(CONFIG_PATH, config)
        print(f"Removed hindsight-memory hooks from {CONFIG_PATH}")


def uninstall_mcp_server() -> None:
    mcp_config = _load_json(MCP_CONFIG_PATH)
    servers = mcp_config.get("mcpServers")
    if not isinstance(servers, dict) or "hindsight" not in servers:
        return
    # The key alone is not proof of ownership. "hindsight" is a name a user may
    # have configured themselves — install.py warns when it replaces such an
    # entry — and deleting a server this plugin never wrote is not an uninstall,
    # it is collateral damage.
    if not _is_our_mcp_server(servers["hindsight"]):
        print(
            f"Left the 'hindsight' MCP server in {MCP_CONFIG_PATH} alone: it does not point at this plugin's launcher.",
            file=sys.stderr,
        )
        return
    del servers["hindsight"]
    mcp_config["mcpServers"] = servers
    _write_json(MCP_CONFIG_PATH, mcp_config)
    print(f"Removed hindsight MCP server from {MCP_CONFIG_PATH}")


def main():
    uninstall_hooks()
    uninstall_mcp_server()
    print("hindsight-memory hooks and MCP server removed. Local state under")
    print("~/.hindsight/devin-cli/ was left in place; delete it manually if desired.")


if __name__ == "__main__":
    main()
