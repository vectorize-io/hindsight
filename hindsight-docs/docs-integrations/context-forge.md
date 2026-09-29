---
sidebar_position: 21
title: "ContextForge MCP Gateway with Hindsight | Integration"
description: "Register Hindsight as an MCP backend in IBM ContextForge gateway. Gives every connected AI tool — Dust, Claude Desktop, custom agents — access to long-term memory through a unified MCP hub."
---

# ContextForge

Register [Hindsight](https://vectorize.io/hindsight) as an MCP backend in [IBM ContextForge](https://github.com/IBM/mcp-context-forge), an open-source MCP gateway. Once registered, every AI tool connected to ContextForge — Dust, Claude Desktop, custom agents — gets access to Hindsight's retain, recall, and reflect tools through a single unified endpoint.

## Why ContextForge + Hindsight?

ContextForge acts as a central MCP hub that aggregates multiple MCP servers behind one authenticated endpoint. Adding Hindsight as a backend means:

- **One endpoint, many tools** — AI platforms connect to ContextForge once and get access to Hindsight memory alongside your other MCP servers (databases, workflows, metadata catalogs)
- **Centralized auth** — ContextForge handles SSO, RBAC, and JWT-based authentication; no need to expose Hindsight directly
- **Team-scoped memory** — Use ContextForge's team/visibility model to control which teams can access which memory banks
- **No code changes** — Hindsight's built-in MCP endpoint (`/mcp`) speaks standard Streamable HTTP; ContextForge connects to it natively

## Architecture

```
AI Tools (Dust, Claude Desktop, custom agents)
        │
        ▼
┌─────────────────────────────┐
│  ContextForge MCP Gateway   │  ← SSO, RBAC, session management
│  (unified MCP endpoint)     │
└──────────┬──────────────────┘
           │  Streamable HTTP
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
 Hindsight    PostgreSQL     OpenMetadata
 (memory)     MCP (SQL)      (metadata)
```

## Prerequisites

- A running Hindsight instance (Docker, Helm, or Cloud)
- A running ContextForge instance
- Network connectivity between ContextForge and Hindsight (same Kubernetes cluster, VPC, or public URL)

## Setup

### Option 1: Register via ContextForge Admin UI

1. Log in to ContextForge as an admin
2. Navigate to **Servers** → **Add Server**
3. Fill in:
   - **Name**: `hindsight`
   - **URL**: `http://<hindsight-host>:8888/mcp` (or your Hindsight API URL + `/mcp`)
   - **Transport**: Streamable HTTP
4. Click **Save**

### Option 2: Register via Admin API

```bash
# Authenticate
TOKEN=$(curl -s -X POST https://your-contextforge.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "your-password"}' \
  | jq -r '.access_token')

# Register Hindsight as an MCP server
curl -X POST https://your-contextforge.com/admin/servers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "hindsight",
    "url": "http://hindsight-api:8888/mcp",
    "transport": "streamable_http",
    "description": "Hindsight agent memory — retain, recall, and reflect"
  }'
```

### Option 3: Kubernetes (Helm)

If both services run in the same EKS/Kubernetes cluster, Hindsight is reachable at its in-cluster service DNS:

```
http://hindsight-api.hindsight.svc.cluster.local:8888/mcp
```

Register this URL in ContextForge using either the UI or API methods above.

For automated registration on every Helm deploy, add a post-install Job to your ContextForge chart that calls the Admin API (see the [Helm example](#helm-auto-registration) below).

## Verifying the Integration

Once registered, verify Hindsight tools are available through ContextForge:

```bash
# List tools via ContextForge's MCP endpoint
curl -s -X POST https://your-contextforge.com/mcp \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' \
  | jq '.result[] | select(.name | startswith("hindsight"))'
```

You should see Hindsight's tools: `retain`, `recall`, `reflect`, `list_banks`, etc.

## Bank Modes

Hindsight's MCP endpoint supports two modes. Choose based on your use case:

### Multi-bank mode (default)

Register the root MCP endpoint:

```
http://hindsight-api:8888/mcp
```

All tools include an optional `bank_id` parameter. Agents can create and switch between memory banks dynamically.

### Single-bank mode

Pin to a specific bank by including the bank ID in the URL:

```
http://hindsight-api:8888/mcp/my-team-bank/
```

Tools are scoped to that bank — no `bank_id` parameter needed. Useful when you want each ContextForge team to have its own isolated memory.

## Available Tools

Once registered, these Hindsight tools become available through ContextForge:

| Tool | Description |
|------|-------------|
| `retain` | Store information to long-term memory |
| `recall` | Search memories with natural language queries |
| `reflect` | Synthesize memories into a reasoned answer |
| `list_banks` | List all memory banks |
| `create_bank` | Create a new memory bank |
| `list_mental_models` | List pinned reflections |
| `create_mental_model` | Create a new mental model |
| `list_documents` | List ingested documents |

See the full tool list in the [MCP Server reference](https://hindsight.vectorize.io/developer/mcp-server#available-tools).

## Helm Auto-Registration

To automatically register Hindsight in ContextForge on every Helm deploy, add a post-install/post-upgrade Job to your ContextForge chart:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: register-hindsight-{{ .Release.Revision }}
  annotations:
    "helm.sh/hook": post-install,post-upgrade
    "helm.sh/hook-weight": "5"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: register
          image: curlimages/curl:latest
          command: ["/bin/sh", "-c"]
          args:
            - |
              # Wait for ContextForge
              for i in $(seq 1 12); do
                curl -sf http://context-forge:4444/health && break
                sleep 10
              done
              # Authenticate
              TOKEN=$(curl -s -X POST http://context-forge:4444/auth/login \
                -H "Content-Type: application/json" \
                -d "{\"email\": \"$ADMIN_USER\", \"password\": \"$ADMIN_PASS\"}" \
                | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
              # Register or update
              curl -X POST http://context-forge:4444/admin/servers \
                -H "Authorization: Bearer $TOKEN" \
                -H "Content-Type: application/json" \
                -d '{
                  "name": "hindsight",
                  "url": "http://hindsight-api.hindsight.svc.cluster.local:8888/mcp",
                  "transport": "streamable_http",
                  "description": "Hindsight agent memory"
                }'
          env:
            - name: ADMIN_USER
              valueFrom:
                secretKeyRef:
                  name: context-forge-credentials
                  key: ADMIN_USERNAME
            - name: ADMIN_PASS
              valueFrom:
                secretKeyRef:
                  name: context-forge-credentials
                  key: ADMIN_PASSWORD
```

## Connecting AI Tools via ContextForge

Once Hindsight is registered in ContextForge, AI tools connect to ContextForge's unified endpoint — not directly to Hindsight.

### Dust

1. In Dust → **Spaces → Tools → Add MCP Server**
2. URL: `https://your-contextforge.com/mcp`
3. Auth: Bearer token (ContextForge JWT or MCP client token)
4. Hindsight tools appear alongside all other registered MCP servers

### Claude Desktop

Add to your MCP config:

```json
{
  "mcpServers": {
    "context-forge": {
      "url": "https://your-contextforge.com/mcp",
      "headers": {
        "Authorization": "Bearer <your-contextforge-token>"
      }
    }
  }
}
```

## Troubleshooting

### Hindsight tools not appearing

Verify the server is registered and healthy:

```bash
# Check registered servers
curl -s https://your-contextforge.com/admin/servers \
  -H "Authorization: Bearer $TOKEN" | jq '.[] | select(.name == "hindsight")'

# Test Hindsight directly
curl -s http://hindsight-api:8888/health
```

### Connection refused from ContextForge

If ContextForge can't reach Hindsight:
- Verify network connectivity (same namespace/VPC, NetworkPolicy allows traffic)
- Check the registered URL matches the actual Hindsight service endpoint
- Ensure `SSRF_ALLOW_PRIVATE_NETWORKS=true` is set in ContextForge (required for in-cluster backends)

### Auth errors

ContextForge authenticates users at the gateway level. Hindsight's MCP endpoint doesn't require separate auth when accessed from within the cluster. If you've enabled `MCP_AUTH_TOKEN` on Hindsight, pass it via ContextForge's header passthrough:

```bash
# Ensure ContextForge forwards Authorization headers
ENABLE_HEADER_PASSTHROUGH=true
DEFAULT_PASSTHROUGH_HEADERS='["Authorization"]'
```
