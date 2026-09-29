---
title: "What's New in Hindsight Cloud: Native OAuth for MCP Clients"
authors: [hindsight]
date: 2026-03-27T12:00
tags: [hindsight-cloud, release, mcp, oauth, claude-code, cursor, windsurf]
description: "Hindsight Cloud now supports native OAuth 2.1 for MCP clients. Connect Claude Code, Cursor, Windsurf, and ChatGPT directly with no API keys required."
image: /img/blog/hindsight-cloud-mcp-oauth.png
hide_table_of_contents: true
---

![What's New in Hindsight Cloud: Native OAuth for MCP Clients](/img/blog/hindsight-cloud-mcp-oauth.png)

[Hindsight Cloud](https://ui.hindsight.vectorize.io) now supports native OAuth 2.1 for MCP clients. Tools like Claude Code, Claude Desktop, ChatGPT, Cursor, and Windsurf can connect directly to Hindsight's memory API with no API keys to generate, copy, or paste.

This post covers what changed, how to connect each supported client, and when OAuth is the right choice versus a traditional API key.

<!-- truncate -->

## TL;DR

- **Two-click connection** — point your MCP client at Hindsight Cloud, approve access, done
- **No API key management** — OAuth tokens are issued and rotated automatically
- **Scoped per org** — choose which organization the client can access during authorization
- **Revoke per client** — disconnect one tool without affecting others
- **Works today** — available in all Hindsight Cloud accounts

## What Changed

Until now, connecting an MCP client to Hindsight Cloud meant creating an API key in the dashboard, copying it, and pasting it into your client config. That works, but it creates friction — especially when onboarding a new machine, adding a team member, or rotating credentials.

With native OAuth 2.1, connecting is the same experience as any other OAuth-protected service. Your MCP client redirects you to Hindsight Cloud, you approve access, and you're done. No keys to manage, no config files to edit manually, no rotation reminders.

## How OAuth 2.1 + Dynamic Client Registration Works

Hindsight's MCP OAuth implementation follows two standards: OAuth 2.1 (the authorization flow) and RFC 7591 Dynamic Client Registration (how clients identify themselves to the server).

Here's what that means in practice:

**Without dynamic client registration**, you'd have to pre-register each MCP tool in the Hindsight dashboard before it could connect — an admin task that doesn't scale.

**With dynamic client registration**, MCP clients register themselves automatically on first connection. Claude Code, Cursor, or any compliant MCP client sends a registration request to Hindsight Cloud, receives a client ID, and then proceeds through the standard OAuth authorization flow. You approve it once in the browser, and the client is authorized.

The full flow:

1. You add the Hindsight MCP endpoint to your client config
2. Your MCP client sends a registration request to Hindsight Cloud
3. Hindsight issues a client ID automatically
4. Your client redirects you to the Hindsight Cloud authorization page
5. You log in (if needed) and approve access, selecting which org
6. Your client receives an access token and stores it locally
7. All future API calls use that token — no manual intervention

From your perspective, steps 2–6 happen in a browser window that opens automatically. You click approve, the window closes, and you're connected.

## Connecting Your MCP Client

### Claude Code

Add the Hindsight MCP server via the CLI:

```bash
claude mcp add hindsight https://mcp.hindsight.vectorize.io --transport http
```

Or edit `~/.claude/settings.json` directly:

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "https://mcp.hindsight.vectorize.io",
      "type": "http"
    }
  }
}
```

On first use, Claude Code will open a browser window for authorization. Log in with your Hindsight Cloud account, approve access, and you're connected. The three Hindsight memory tools (`recall_memory`, `retain_memory`, `reflect_on_memory`) will appear in Claude Code automatically.

### Claude Desktop

Edit your Claude Desktop config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "https://mcp.hindsight.vectorize.io"
    }
  }
}
```

Restart Claude Desktop. On the next launch it will prompt you to authorize Hindsight Cloud in your browser.

### Cursor

1. Open **Settings** (⌘, on macOS)
2. Navigate to **Features → MCP Servers**
3. Click **Add MCP Server**
4. Enter the server URL: `https://mcp.hindsight.vectorize.io`
5. Set the transport to **HTTP**
6. Save

Cursor will open a browser window for the OAuth authorization flow on first connection.

### Windsurf

1. Open the **Windsurf Settings** panel
2. Go to **Cascade → MCP Servers**
3. Click **Add Server**
4. Enter `https://mcp.hindsight.vectorize.io` as the server URL
5. Save and reload

The OAuth prompt will appear automatically on the next session start.

### ChatGPT

ChatGPT supports MCP via the Connectors panel in ChatGPT Plus and Team plans. Navigate to **Settings → Connectors**, click **Add Connector**, and paste `https://mcp.hindsight.vectorize.io`. The authorization flow runs in-browser.

## OAuth vs. API Key: When to Use Each

Both methods are supported. Neither is going away. Here's when each makes sense:

| | OAuth (MCP) | API Key |
|---|---|---|
| **Best for** | Interactive tools (Claude Code, Cursor, Windsurf) | Automated pipelines, CI/CD, server-to-server |
| **Setup** | Browser flow, no copy-paste | Generate in dashboard, paste into config |
| **Token rotation** | Automatic | Manual |
| **Revocation** | Per client, instant | Per key, in dashboard |
| **Team onboarding** | Each person authorizes their own client | Key must be shared or each person generates one |
| **Audit trail** | Per-client access log | Per-key access log |

**Use OAuth when** a developer is connecting their local coding tool to Hindsight and you want them to authorize using their own Hindsight Cloud credentials — no shared secrets, no key distribution.

**Use API keys when** you're running a script, a cloud function, or any non-interactive process that needs to call Hindsight without a browser. API keys also work well for quick local testing when you want a simpler setup.

## Scoped Access and Multi-Org Accounts

If your Hindsight Cloud account has access to multiple organizations, the OAuth authorization screen will ask you to select which org the MCP client can access. Each client is scoped to one org at a time.

This matters if you use Hindsight across multiple projects or teams: you can connect Cursor to your personal org and Claude Code to your team's org, and each tool only sees the memory banks it's authorized for.

## Troubleshooting

**The browser window doesn't open**

Some MCP clients require an explicit browser-launch permission. Check your client's MCP settings for an "allow OAuth redirects" or "open external browser" option. Alternatively, copy the authorization URL from the client's debug log and paste it manually into a browser.

**"Client registration failed"**

This usually means the MCP client doesn't support RFC 7591 dynamic client registration. Check the client's documentation for its OAuth support level. All clients listed in this post have been tested and confirmed working.

**Authorization succeeds but tools don't appear**

Restart or reload your MCP client after the OAuth flow completes. Some clients cache the tool list at startup and need a reload to pick up newly authorized servers.

**"Insufficient scope" errors when calling memory tools**

During authorization, confirm you approved access to the memory read/write scopes. If you approved with a reduced scope, remove the client connection and re-authorize.

## Permissions

Only **Owner** or **Admin** users can authorize an MCP client connection. This matches the same permission level required to create API keys. If you attempt to authorize and see a permissions error, ask your organization Owner or Admin to complete the authorization step.

## Get Started

Native MCP OAuth is available now in all [Hindsight Cloud](https://ui.hindsight.vectorize.io) accounts. Add `https://mcp.hindsight.vectorize.io` to your MCP client config and follow the authorization prompt.

For scripted or automated use cases, [API keys](https://hindsight.vectorize.io/developer/retrieval) remain available in the Hindsight Cloud dashboard. Both methods give full access to [recall](https://hindsight.vectorize.io/developer/retrieval), retain, and reflect.
