# Stitch MCP Integration (Optional)

## Overview

Google Stitch is an AI-powered UI design platform. Its MCP server enables
coding agents to interact with design projects programmatically.

Stitch MCP is **optional**; all 7 phases of the design workflow work
without it. It adds visual preview and design extraction capabilities.

## Prerequisites

### Option A: API Key (Recommended, simplest)
1. Go to [stitch.withgoogle.com](https://stitch.withgoogle.com/)
2. Profile → Stitch Settings → API Keys → Create Key
3. Set environment variable: `STITCH_API_KEY="your-key"`

### Option B: OAuth (Google Cloud)
```bash
npx @_davideast/stitch-mcp init
```
The wizard handles gcloud auth, API enablement, and MCP config.

### Option C: Manual gcloud
```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
gcloud beta services mcp enable stitch.googleapis.com --project=<PROJECT_ID>
```

## MCP Client Configuration

### Claude Code
```bash
claude mcp add stitch -e STITCH_API_KEY=YOUR_KEY -- npx @_davideast/stitch-mcp proxy
```

### Cursor / VS Code (.cursor/mcp.json or .vscode/mcp.json)
```json
{
  "mcpServers": {
    "stitch": {
      "command": "npx",
      "args": ["@_davideast/stitch-mcp", "proxy"],
      "env": { "STITCH_API_KEY": "YOUR_KEY" }
    }
  }
}
```

## Namespace Discovery

MCP tool names may be prefixed. Before calling, discover the prefix:
1. Call `list_tools` to enumerate available tools
2. Look for tools containing "list_projects", "get_screen", etc.
3. Use the discovered prefix (e.g., `mcp_stitch:list_projects`)

## Tool Mapping by Workflow Phase

### Phase 2: EXTRACT
| Step | Tool | Input |
|------|------|-------|
| Find projects | `list_projects` | filter: "view=owned" |
| Get design theme | `get_project` | resource name → designTheme object |
| List screens | `list_screens` | numeric project ID |
| Download HTML | `get_screen_code` | projectId, screenId |
| Download screenshot | `get_screen_image` | projectId, screenId (returns base64 PNG) |

### Phase 4: PROPOSE
| Step | Tool | Input |
|------|------|-------|
| Generate concept | `generate_screen_from_text` | text prompt + design context |
| Generate variants | `generate_variants` | existing screen ID |
| Preview | `get_screen_image` | for visual comparison |

### Phase 5: GENERATE
| Step | Tool | Input |
|------|------|-------|
| Build site | `build_site` | projectId + routes: [{screenId, route}] |
| Get page HTML | `get_screen_code` | per route |

## Available Tools

### Upstream (proxy + direct)
| Tool | Description |
|------|-------------|
| `list_projects` | List all accessible Stitch projects |
| `get_project` | Get project details (including designTheme) |
| `list_screens` | List screens within a project |
| `get_screen` | Get screen metadata and download URLs |
| `generate_screen_from_text` | Generate new screen from text prompt |
| `edit_screens` | Edit existing screens via text prompt |
| `generate_variants` | Generate design variants of a screen |

### Virtual (proxy only)
| Tool | Description |
|------|-------------|
| `build_site` | Map screens to routes, fetch HTML in parallel |
| `get_screen_code` | Retrieve screen + download HTML content |
| `get_screen_image` | Retrieve screen + download screenshot as base64 |

## Without Stitch

All workflow phases function without Stitch MCP:

| Phase | With Stitch | Without Stitch |
|-------|-------------|----------------|
| EXTRACT | API-based token extraction | Manual URL HTML/CSS analysis or skip |
| PROPOSE | Visual screen generation + screenshots | Text-based concept descriptions |
| GENERATE | build_site + get_screen_code | Direct code generation from DESIGN.md |

## Troubleshooting

```bash
# Verify configuration
npx @_davideast/stitch-mcp doctor --verbose

# Full reset
npx @_davideast/stitch-mcp logout --force --clear-config
npx @_davideast/stitch-mcp init
```
