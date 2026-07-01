---
name: tools
description: Manage MCP tools with natural language commands to list, enable, and disable tools and tool groups
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **Read configuration files BEFORE making changes.**

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native file reading tools to inspect MCP and skill configurations.

---

## Step 1: Show Current Status

1. Read `.agents/mcp.json` (project configuration)
2. Read `~/.gemini/settings.json` if exists (Gemini CLI global settings); optional
3. Display status for each MCP server:
   - `available_tools: null` → "All enabled (no restrictions)"
   - `available_tools: [...]` → "N tools enabled" + list
4. If `toolGroups` is defined, display available group list

**Output example:**
```
Current MCP Tool Status

[serena]
- Status: All enabled (no restrictions)
- Available tools: 15

Available Tool Groups:
- memory: read_memory, write_memory, edit_memory, list_memories, delete_memory
- code-analysis: get_symbols_overview, find_symbol, find_referencing_symbols, search_for_pattern
- code-edit: replace_symbol_body, insert_after_symbol, insert_before_symbol, rename_symbol
- file-ops: list_dir, find_file
- all: All tools (no restrictions)

What would you like to change?
```

---

## Step 2: Parse User Command

Parse natural language commands:

| Command Pattern | Interpretation |
|-----------------|----------------|
| "current status", "list", "show" | Re-execute Step 1 |
| "memory tools only", "enable only {group}" | Set only that group's tools in `available_tools` |
| "disable {tool}", "turn off {tool}" | Remove that tool from `available_tools` |
| "enable all", "turn on all", "reset" | Set `available_tools: null` |
| "enable only {tool1}, {tool2}" | Set only specified tools in `available_tools` |
| "temporarily", "--temp" | Apply for session only (Step 3b) |

**Group combination support:**
- "memory + file tools" → Merge `memory` + `file-ops` groups
- "all except code analysis" → Exclude `code-analysis` from `all`

---

## Step 3: Update Configuration

### Step 3a: Permanent Modification (Default)

1. **Show before/after diff:**
   ```
   Pending mcp.json changes:

   Before:
   - serena.available_tools: null (all)

   After:
   - serena.available_tools: ["read_memory", "write_memory", "edit_memory", "list_memories", "delete_memory"]

   Apply changes? (Y/N)
   ```

2. **After user confirmation**, modify `.agents/mcp.json`

3. **Completion message:**
   ```
   Done! serena can now only use memory tools.

   Note: Changes will fully apply after IDE/CLI restart.
   Previous settings may continue to apply in current session.
   ```

### Step 3b: Temporary Application (`--temp` or "temporarily")

Temporary settings that apply only during the session:

1. Create `.serena/memories/tool-overrides.md` using `write_memory`:
   ```markdown
   # Tool Overrides (Temporary)

   ## Session
   Created: {ISO timestamp}
   Expires: Session end

   ## Overrides
   ```json
   {
     "serena": {
       "available_tools": ["read_memory", "write_memory"]
     }
   }
   ```

   ## Note
   This file contains temporary settings. It will be ignored in the next session.
   To apply permanently, run the `/tools` workflow without `--temp`.
   ```

2. **Completion message:**
   ```
   Temporarily applied!

   serena will only use memory tools for this session.
   To apply permanently, run `/tools enable memory only` (without --temp).
   ```

---

## Step 4: Handle Special Cases

### Unknown Tool Name
```
'{tool_name}' is an unknown tool.

Similar tools:
- read_memory
- write_memory

Please enter the exact tool name.
```

### Server Conflict
When multiple MCP servers are configured:
```
Multiple MCP servers detected:
- serena
- custom-memory

Which server's tools would you like to modify?
1. serena
2. custom-memory
3. All
```

### Empty Tool List
```
Setting available_tools to an empty array will disable all tools for that server.
Are you sure you want to continue? (Y/N)
```

---

## Quick Reference

| Command | Result |
|---------|--------|
| `/tools` | Show current status |
| `/tools memory only` | Enable only memory tools |
| `/tools code analysis + memory` | Enable both groups |
| `/tools all` | Enable all tools (reset) |
| `/tools read_memory, write_memory only` | Enable only specified tools |
| `/tools disable code edit` | Remove that group |
| `/tools memory only --temp` | Apply temporarily (this session only) |

---

## Runtime Override Protocol

How other workflows SHOULD check tool restrictions (advisory — no workflow currently automates this check; apply it manually when tool discipline matters):

1. **At workflow start:** Check `read_memory("tool-overrides.md")`
2. **If override exists:** Apply that setting with priority
3. **If not present or expired:** Use `.agents/mcp.json` settings

**Note:** If IDE/CLI doesn't directly support `available_tools`,
tool usage must be self-restricted at the workflow level.
