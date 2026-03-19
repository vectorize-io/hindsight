---
slug: sandboxed-agent-persistent-memory-nemoclaw
title: "Give NemoClaw the Best Agent Memory Available In One Command"
description: Add persistent memory to a NemoClaw sandboxed AI agent without changing code. One command, one network policy, memories survive across sessions.
authors: [hindsight]
date: 2026-03-19
hide_table_of_contents: true
---

## TL;DR

- [NemoClaw](https://nemoclaw.ai) sandboxes isolate AI agents — controlled filesystem, processes, and network. That isolation makes persistent memory harder.
- We connected the `hindsight-openclaw` plugin to a live NemoClaw sandbox using [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup). No code changes — one command.
- External API mode is the natural fit: the plugin becomes a thin HTTP client, and the sandbox only needs one egress rule.
- Memories captured in one session are recalled in the next. The sandbox didn't interfere.
- The pattern generalizes: sandbox controls what the agent can *do*, memory controls what it *knows*. They compose cleanly.

## The Problem: Sandboxed Agents Have No Persistent Memory

AI agents running inside sandboxes present an interesting memory problem. The sandbox is designed to isolate the agent — it controls which files it can read, which processes it can spawn, and which network endpoints it can reach. That isolation is the point. But it creates a question: if every session starts in a clean, constrained environment, where does persistent memory live?

We set out to answer that with [NemoClaw](https://nemoclaw.ai), NVIDIA's sandboxed agent runtime built on OpenShell. The goal was simple: connect the `hindsight-openclaw` plugin to a live NemoClaw sandbox and verify that memories captured in one session are recalled in the next. No code changes allowed — if we needed to modify the plugin to make it work, we'd learned something important about the architecture.

We didn't need to change a line.

<!-- truncate -->

## The Approach: External API Mode for Sandbox Memory

[NemoClaw](https://nemoclaw.ai) runs [OpenClaw](https://openclaw.ai) inside an OpenShell sandbox. The sandbox enforces a filesystem policy (what paths the agent can read and write), a process policy (what it runs as), and a network egress policy (which outbound endpoints are permitted).

By default, the sandbox ships with policies for the services it needs: the LLM provider, GitHub, npm, the OpenClaw API. Everything else is blocked. That's a good default — an agent that can call arbitrary endpoints is harder to trust.

[Hindsight](https://hindsight.vectorize.io) operates as an external API. The plugin makes HTTPS calls to `api.hindsight.vectorize.io` to [retain and recall memories](https://hindsight.vectorize.io/blog/2026/03/04/mcp-agent-memory). From the sandbox's perspective, that's just another outbound endpoint — one that needs to be explicitly permitted.

The full stack looks like this:

```
┌─────────────────────────────────────────────┐
│  NemoClaw Sandbox (OpenShell)               │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  OpenClaw Gateway                    │   │
│  │  + hindsight-openclaw plugin         │   │
│  │    ↓ before_agent_start: recall      │   │
│  │    ↓ agent_end: retain               │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Network egress policy:                     │
│    ✓ api.anthropic.com                      │
│    ✓ integrate.api.nvidia.com               │
│    ✓ api.hindsight.vectorize.io  ← added    │
└─────────────────────────────────────────────┘
```

When the plugin retains a conversation, Hindsight doesn't just store raw text. It extracts structured facts, resolves entities, builds a [knowledge graph](https://hindsight.vectorize.io/blog/2026/03/12/spreading-activation-memory-graphs), and indexes everything for multi-strategy retrieval — semantic search, BM25 keyword matching, graph traversal, and temporal filtering with [cross-encoder reranking](https://hindsight.vectorize.io/blog/2026/03/04/mcp-agent-memory). That's what makes recall useful even when the agent's question doesn't match the exact wording of what was stored.

The plugin has two modes. In **local daemon mode**, it spawns a local `hindsight-embed` process and communicates with it over a local port. In **external API mode**, it skips the daemon entirely and makes HTTP calls directly to a Hindsight Cloud endpoint.

Inside a sandbox, local daemon mode is awkward. The sandbox controls which processes can be spawned, and a background daemon that launches `uvx` subprocesses is friction we don't need. External API mode is the natural fit: the plugin becomes a thin HTTP client, and the only infrastructure requirement is a network egress rule.

For background on the OpenClaw plugin itself — how it hooks into the gateway lifecycle, auto-injects memory into context, and prevents feedback loops — see [The Memory Upgrade Every OpenClaw User Needs](https://hindsight.vectorize.io/blog/2026/03/06/adding-memory-to-openclaw-with-hindsight).

## Implementation: One Command

The `hindsight-nemoclaw` package automates the entire setup — installing the plugin, configuring external API mode, reading your current sandbox policy, merging the Hindsight egress rule, and restarting the gateway:

```bash
npx @vectorize-io/hindsight-nemoclaw setup \
  --sandbox my-assistant \
  --api-url https://api.hindsight.vectorize.io \
  --api-token <your-api-key> \
  --bank-prefix my-sandbox
```

That's it. You'll see output like:

```
[0] Preflight checks...
  ✓ openshell found
  ✓ openclaw found

[1] Installing @vectorize-io/hindsight-openclaw plugin...
  ✓ Plugin installed

[2] Configuring plugin in ~/.openclaw/openclaw.json...
  ✓ Plugin config written (bank: my-sandbox-openclaw)

[3] Applying Hindsight network policy to sandbox "my-assistant"...
  ✓ Policy version 2 submitted
  ✓ Policy version 2 loaded (active version: 2)

[4] Restarting OpenClaw gateway...
  ✓ Gateway restarted

✓ Setup complete!
```

Use `--dry-run` to preview all changes before applying. Use `--skip-policy` if you manage sandbox policies manually.

## Verifying It Works

After setup, the gateway logs confirm the plugin is running:

```
[Hindsight] Plugin loaded successfully
[Hindsight] ✓ Using external API: https://api.hindsight.vectorize.io
[Hindsight] External API health: {"status":"healthy","database":"connected"}
[Hindsight] Default bank: my-sandbox-openclaw
[Hindsight] ✓ Ready (external API mode)
```

Send a message to the agent:

```bash
openclaw agent --agent main --session-id session-1 \
  -m "My name is Ben and I work on Hindsight. I prefer detailed commit messages."
```

The gateway logs show the hooks firing:

```
[Hindsight] before_agent_start - bank: my-sandbox-openclaw, channel: undefined/webchat
[Hindsight Hook] agent_end triggered - bank: my-sandbox-openclaw
[Hindsight] Retained 6 messages to bank my-sandbox-openclaw for session agent:main:...
```

Open a fresh session and ask what the agent remembers:

```bash
openclaw agent --agent main --session-id session-2 \
  -m "What do you remember about me?"
```

```
Right now I've just got the basics: your name is Ben, you're working on
Hindsight, and you like commit messages to be detailed. If there's anything
else you want me to keep in mind, let me know.
```

The memory survived the session boundary. The sandbox didn't interfere with it.

## What the Setup Command Does (Manual Alternative)

If you prefer to apply the steps yourself, here's what `hindsight-nemoclaw setup` does under the hood.

**Install the plugin:**

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
```

**Configure `~/.openclaw/openclaw.json`:**

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "<your-api-key>",
          "llmProvider": "claude-code",
          "dynamicBankId": false,
          "bankIdPrefix": "my-sandbox"
        }
      }
    }
  }
}
```

**Add the Hindsight block to your sandbox network policy** (note: `openshell policy set` replaces the full document — include all existing policies):

```yaml
network_policies:
  hindsight:
    name: hindsight
    endpoints:
      - host: api.hindsight.vectorize.io
        port: 443
        protocol: rest
        tls: terminate
        enforcement: enforce
        rules:
          - allow:
              method: GET
              path: /**
          - allow:
              method: POST
              path: /**
          - allow:
              method: PUT
              path: /**
    binaries:
      - path: /usr/local/bin/openclaw
```

```bash
openshell policy set my-sandbox --policy /path/to/full-policy.yaml --wait
openclaw gateway restart
```

## Pitfalls & Edge Cases

### 1. Policy replacement is full-document

`openshell policy set` replaces the entire policy document, not just the section you're adding. The `hindsight-nemoclaw setup` command handles this automatically — it reads the current policy, merges the Hindsight block, and re-applies the full document. If you're applying manually, make sure your YAML includes all existing network policies.

### 2. LaunchAgent can't follow symlinks on macOS

On macOS, the OpenClaw gateway runs as a LaunchAgent with a restricted security context that can't access `~/Documents` or other user directories. `openclaw plugins install --link` creates a symlink that the LaunchAgent can't follow — install as a copy instead:

```bash
# This works — copies files to ~/.openclaw/extensions/
openclaw plugins install @vectorize-io/hindsight-openclaw
```

If you see `EPERM: operation not permitted, scandir` in your gateway logs, this is what's happening.

### 3. Memory retention is asynchronous

When the plugin calls `retain` at the end of a session, [fact extraction and entity resolution](https://hindsight.vectorize.io/blog/2026/03/12/spreading-activation-memory-graphs) happen in the background on Hindsight's side. If you open a new session immediately, the most recent memories may not be indexed yet. In practice this is a few seconds — but it's worth knowing if you're testing back-to-back.

### 4. Binary-scoped egress is strict

The `binaries` field in the network policy means *only* the specified executable can reach the endpoint. If you update OpenClaw and the binary path changes, the egress rule silently stops working. Check your binary path after upgrades.

## Tradeoffs: External API vs. Local Daemon in a Sandbox

| | **External API mode** | **Local daemon mode** |
|---|---|---|
| **Setup** | One command | Process spawning permissions |
| **Dependencies** | HTTPS egress only | `uvx`, Python, local PostgreSQL |
| **Data location** | Hindsight Cloud | Local to sandbox |
| **Multi-sandbox sharing** | Same bank from anywhere | Per-sandbox only |
| **Sandbox compatibility** | Clean fit | Fights the process policy |

**Use external API mode** when you're in a sandbox, want shared memory across instances, or don't want to manage a local database.

**Use local daemon mode** when data must stay on the machine, network egress is completely locked down, or you're running outside a sandbox where process spawning is unrestricted.

For background on the local daemon approach, see [The Memory Upgrade Every OpenClaw User Needs](https://hindsight.vectorize.io/blog/2026/03/06/adding-memory-to-openclaw-with-hindsight).

## What This Pattern Means for Sandboxed Agent Memory

The pattern here is worth naming. A sandboxed agent isn't a limitation on persistent memory — it's a different trust boundary:

- **Sandbox** controls what the agent can *do* — filesystem access, process spawning, network calls.
- **Memory** controls what the agent *knows* — facts, entities, context from prior sessions.

Those are orthogonal concerns, and they compose cleanly.

By keeping memory in an external service and making the network policy explicit, you get both: an agent that's constrained in what it can affect, and one that builds durable knowledge across sessions. The policy file is a readable record of every external dependency the agent has. That transparency is useful.

There's also an interesting property of `dynamicBankId`:

- **Enabled** (`true`): each user gets an isolated memory bank. Memories from one user's sessions can't bleed into another's. Use this for multi-tenant deployments.
- **Disabled** (`false`): a shared bank accumulates context from all sessions. Use this for single-user sandboxes like a personal coding assistant.

> **Want to skip self-hosting?** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) is what we used in this walkthrough — no Docker, no infrastructure. Sign up, grab an API key, and run `npx @vectorize-io/hindsight-nemoclaw setup`.

## Recap

Persistent memory in a sandboxed AI agent is one command: `npx @vectorize-io/hindsight-nemoclaw setup`. It installs the plugin, applies the network egress rule, and configures external API mode — everything the sandbox needs to let Hindsight through.

The key insight: sandbox isolation and persistent memory are orthogonal concerns. The sandbox controls what the agent can affect; memory controls what the agent knows. One network policy rule bridges them without compromising either.

## Next Steps

- **Run the setup**: `npx @vectorize-io/hindsight-nemoclaw setup --help` to get started.
- **Try per-user memory banks**: Enable `dynamicBankId: true` to give each user isolated memory in multi-tenant deployments.
- **Explore the OpenClaw plugin in depth**: See [The Memory Upgrade Every OpenClaw User Needs](https://hindsight.vectorize.io/blog/2026/03/06/adding-memory-to-openclaw-with-hindsight) for how the plugin hooks into gateway lifecycle events.
- **Connect other agents to the same memory**: Hindsight works with [Hermes Agent](https://hindsight.vectorize.io/blog/2026/03/17/hermes-agent-memory), [Streamlit chatbots](https://hindsight.vectorize.io/blog/2026/03/17/python-chatbot-memory-streamlit), and [any MCP client](https://hindsight.vectorize.io/blog/2026/03/04/mcp-agent-memory).
- **Check out the docs**: Full API reference and SDK guides at [docs.hindsight.vectorize.io](https://docs.hindsight.vectorize.io/recall/).

---

**Resources:**
- [hindsight-nemoclaw on npm](https://www.npmjs.com/package/@vectorize-io/hindsight-nemoclaw)
- [hindsight-openclaw on npm](https://www.npmjs.com/package/@vectorize-io/hindsight-openclaw)
- [OpenClaw plugin documentation](https://vectorize.io/hindsight/sdks/integrations/openclaw)
- [Hindsight Cloud](https://ui.hindsight.vectorize.io)
