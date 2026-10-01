---
title: "Screenpipe Context with Hindsight | MCP Recipe"
description: "Connect Screenpipe and Hindsight to one MCP client, retrieve selected desktop context, and retain reviewed memories with source references."
---

# Screenpipe

Use [Screenpipe](https://github.com/screenpipe/screenpipe) to retrieve recorded
work context and Hindsight to retain selected memories for future tasks. This
community recipe connects two existing MCP servers to one local agent. It does
not install a sync service or add a Hindsight ingestion adapter.

## Prerequisites

- A running Screenpipe desktop recorder on the agent's computer.
- Node.js and `npx`, plus the Screenpipe local API key in the bridge process's
  `SCREENPIPE_LOCAL_API_KEY` environment variable. Obtain the key with
  `screenpipe auth token` or the desktop app's configured connection.
- A working [Hindsight MCP server](./local-mcp.md) and a bank selected for this
  workflow. Set up the server and bank before adding the configuration below.
- A client that supports multiple Streamable HTTP MCP servers.

## Connect both servers

Start the Screenpipe bridge on loopback:

```bash
npx -y screenpipe-mcp@0.20.2 --http --port 3031
```

Add both entries to your client's existing `mcpServers` object:

```json
{
  "mcpServers": {
    "screenpipe": {
      "url": "http://127.0.0.1:3031/mcp"
    },
    "hindsight": {
      "url": "http://127.0.0.1:8888/mcp/screenpipe-work/"
    }
  }
}
```

Replace `screenpipe-work` with your chosen bank ID and preserve any existing
server entries. The Hindsight URL above assumes a local server without API-key
authentication; for a hosted or authenticated deployment use the URL and headers
from its [MCP setup](https://hindsight.vectorize.io/developer/mcp-server).
Loopback URLs are only reachable by a client on that computer.

Confirm tool discovery: Screenpipe HTTP provides `search_content`; Hindsight
provides `retain`, `recall` and `reflect` when those tools are enabled. A configured
URL alone does not prove the server is connected.

## Retrieve, review, retain

1. Search Screenpipe for one project or meeting with a narrow time range and a
   small result limit. Keep the returned timestamps and source identifiers.
2. Review the evidence and write a short note containing only the facts you want
   retained. Mark unknown outcomes and disagreements explicitly.
3. Use Hindsight's `retain` tool in the selected bank for that reviewed note.
   Include a source label such as `Screenpipe`, the device label, recording time
   and record identifiers in the content so later recall can identify the origin.
4. In a new session, use `recall` to retrieve the decision and compare it with the
   retained note. Use `reflect` when the task needs synthesis across memories.

For example, a meeting transcript can support a note that the team chose CSV for
a handoff. It does not prove the export was created or delivered. Preserve that
distinction when the agent summarizes the result.

## Data boundaries

Screenpipe search results enter the MCP client's model context. Retaining a note
also sends that selected content to the chosen Hindsight server and its configured
processing providers. Local storage alone does not make remote model processing
local. Review the destination before retaining sensitive material.

This recipe has no automatic deduplication or deletion reconciliation. Avoid
retaining the same episode repeatedly; keep source references so you can find and
correct earlier entries. Deleting a Screenpipe recording does not remove a note
already retained in Hindsight.

To disconnect, remove the Screenpipe entry from the client and stop the bridge
process started for this recipe. Existing Hindsight memories remain until you
remove them using Hindsight's memory-management tools.

See the [Screenpipe MCP reference](https://github.com/screenpipe/screenpipe/tree/main/packages/screenpipe-mcp)
for bridge authentication and transport details.
