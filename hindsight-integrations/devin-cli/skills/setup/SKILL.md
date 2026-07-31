---
name: setup
description: Set up or verify Hindsight long-term memory for Devin CLI (registers hooks and the MCP server). Use when the user asks to set up, enable, install, or troubleshoot Hindsight memory for Devin CLI.
allowed-tools: Bash(python3 */scripts/install.py) Bash(python3 */scripts/uninstall.py)
---

# Set Up Hindsight Memory for Devin CLI

Devin CLI's plugin system does not yet template a bundled script's own install
path into hooks.json / mcp_config.json (no `${DEVIN_PLUGIN_ROOT}` equivalent —
see `scripts/install.py`'s docstring for the full explanation and how to
recheck it on a newer CLI). So installing this plugin via `devin plugins
install` alone gets you the skills, but **not** working hooks or the MCP
server — this skill runs the one-time setup step that wires those up.

## What to do when this skill is invoked

1. When the `skill` tool activated this skill, its output included a `Source:`
   or `Base directory:` path — something like
   `.../hindsight-memory/skills/setup/SKILL.md` or
   `.../hindsight-memory/skills/setup`. The plugin root is everything in that
   path *before* the `/skills/` segment — `.../hindsight-memory` in both
   examples above. Derive it that way rather than counting directory levels
   upward: the path may or may not end in the `SKILL.md` filename, so the same
   count lands somewhere different depending on which form you were given.
2. Run, via `exec`:
   ```
   python3 "<plugin_root>/scripts/install.py"
   ```
3. Report the script's output to the user verbatim (it prints the config
   files it wrote). If it errored because an existing config file has invalid
   JSON, tell the user which file and stop — don't try to fix their config
   file's JSON yourself without asking.
4. Tell the user to start a new Devin CLI session (or resume an existing one)
   for the hooks and MCP server to take effect — both are only loaded at
   session start.
5. Mention `~/.hindsight/devin-cli.json` as where to configure it further
   (API URL, bank ID, recall/retain tuning — see the plugin's README.md for
   the full settings reference), and that setting `OPENAI_API_KEY` or
   `ANTHROPIC_API_KEY` in the shell is enough to get a local daemon running
   with no other config.

## Uninstalling

If the user asks to remove/disable Hindsight memory, run:
```
python3 "<plugin_root>/scripts/uninstall.py"
```
This only removes the hook/MCP entries this plugin registered — it does not
touch other hooks or MCP servers the user has configured.

## Verifying it's working

- `/hooks` inside a Devin CLi session should list four hooks from this
  plugin's `scripts/` directory (SessionStart, UserPromptSubmit, Stop,
  SessionEnd).
- `/mcp` should list a `hindsight` server as Connected.
- With `debug: true` set in `~/.hindsight/devin-cli.json`, hook scripts log
  `[Hindsight] ...` lines to stderr — ask the user to check
  `~/.local/share/devin/cli/logs/` for the current session's log if something
  isn't working (or run a session with the CLI attached to a terminal you can
  watch directly).
