---
title: "Persistent memory inside a sandboxed AI agent"
description: How we added long-term memory to a NemoClaw sandbox without changing any code — just config and a network policy.
authors: [hindsight]
date: 2026-03-19
---

AI agents running inside sandboxes present an interesting memory problem. The sandbox is designed to isolate the agent — it controls which files it can read, which processes it can spawn, and which network endpoints it can reach. That isolation is the point. But it creates a question: if every session starts in a clean, constrained environment, where does memory live?

We set out to answer that with [NemoClaw](https://nemoclaw.ai), NVIDIA's sandboxed agent runtime built on OpenShell. The goal was simple: connect the `hindsight-openclaw` plugin to a live NemoClaw sandbox and verify that memories captured in one session are recalled in the next. No code changes allowed — if we needed to modify the plugin to make it work, we'd learned something important about the architecture.

We didn't need to change a line.

<!-- truncate -->

## The Architecture

NemoClaw runs [OpenClaw](https://openclaw.ai) inside an OpenShell sandbox. The sandbox enforces a filesystem policy (what paths the agent can read and write), a process policy (what it runs as), and a network egress policy (which outbound endpoints are permitted).

By default, the sandbox ships with policies for the services it needs: the LLM provider, GitHub, npm, the OpenClaw API. Everything else is blocked. That's a good default — an agent that can call arbitrary endpoints is harder to trust.

Hindsight operates as an external API. The plugin makes HTTPS calls to `api.hindsight.vectorize.io` to retain and recall memories. From the sandbox's perspective, that's just another outbound endpoint — one that needs to be explicitly permitted.

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

## Why External API Mode

The plugin has two modes. In **local daemon mode**, it spawns a local `hindsight-embed` process and communicates with it over a local port. In **external API mode**, it skips the daemon entirely and makes HTTP calls directly to a Hindsight Cloud endpoint.

Inside a sandbox, local daemon mode is awkward. The sandbox controls which processes can be spawned, and a background daemon that launches `uvx` subprocesses is friction we don't need. External API mode is the natural fit: the plugin becomes a thin HTTP client, and the only infrastructure requirement is a network egress rule.

To enable it, you set two config fields:

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

`hindsightApiUrl` + `hindsightApiToken` switches the plugin to HTTP mode. `llmProvider: "claude-code"` satisfies the LLM detection check using the Claude Code process already present in the sandbox — no additional API key needed. `dynamicBankId: false` writes all sessions to a single bank, which makes it easy to verify things are working.

## The Network Policy

OpenShell policy is a YAML document that gets applied to the sandbox. It defines exactly what the agent is allowed to do. Adding Hindsight means adding one block to `network_policies`:

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

The `binaries` field ties the network rule to a specific executable. Only the OpenClaw process can make calls to `api.hindsight.vectorize.io`. If a different process tried, the egress policy would block it.

One thing to be aware of: `openshell policy set` replaces the entire policy document, not just the section you're adding. Make sure your YAML includes all the existing network policies or they get removed.

```bash
openshell policy set my-sandbox --policy /path/to/full-policy.yaml --wait
# ✓ Policy version 2 submitted (hash: 3f3d742e7bc6)
# ✓ Policy version 2 loaded (active version: 2)
```

## One Wrinkle: The LaunchAgent

On macOS, the OpenClaw gateway runs as a LaunchAgent. LaunchAgents run under a restricted security context that can't access `~/Documents` or other user directories, even though your shell can. This matters because `openclaw plugins install --link` creates a symlink into the source directory — and the LaunchAgent can't follow it.

The fix is straightforward: install as a copy instead of a link.

```bash
# This will fail — LaunchAgent can't scan ~/Documents/...
openclaw plugins install --link /path/to/hindsight-integrations/openclaw

# This works — copies files to ~/.openclaw/extensions/
openclaw plugins install /path/to/hindsight-integrations/openclaw
```

If you see `EPERM: operation not permitted, scandir` in your gateway logs, this is what's happening.

## Watching It Work

After restarting the gateway, the logs confirm the plugin initialized correctly:

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

Now open a fresh session and ask what the agent remembers:

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

## What This Means for Sandboxed Agents

The pattern here is worth naming. A sandboxed agent isn't a limitation on memory — it's a different trust boundary. The sandbox controls what the agent can *do* (filesystem access, process spawning, network calls). Memory is about what the agent *knows*. Those are different concerns, and they compose cleanly.

By keeping memory in an external service and making the network policy explicit, you get both things: an agent that's constrained in what it can affect, and one that builds durable knowledge across sessions. The policy file is a readable record of every external dependency the agent has. That transparency is useful.

There's also an interesting property of `dynamicBankId`. When you enable it, each user gets an isolated memory bank — memories from one user's sessions can't bleed into another's. When you disable it, a shared bank accumulates context from all sessions. For a single-user sandbox like a personal coding assistant, the shared bank is fine. For multi-tenant deployments, per-user isolation is the right default.

## Getting Started

Full setup instructions are in [NEMOCLAW.md](https://github.com/vectorize-io/hindsight/blob/openclaw/hindsight-integrations/openclaw/NEMOCLAW.md) in the repository.

The short version:

1. Create a bank: `curl -X PUT https://api.hindsight.vectorize.io/v1/default/banks/<name> ...`
2. Install the plugin: `openclaw plugins install @vectorize-io/hindsight-openclaw`
3. Add the config block to `~/.openclaw/openclaw.json`
4. Add the `hindsight` network policy to your sandbox
5. Restart the gateway: `openclaw gateway restart`

No code changes required.

---

**Resources:**
- [hindsight-openclaw on npm](https://www.npmjs.com/package/@vectorize-io/hindsight-openclaw)
- [NemoClaw setup guide](https://github.com/vectorize-io/hindsight/blob/openclaw/hindsight-integrations/openclaw/NEMOCLAW.md)
- [OpenClaw plugin documentation](https://vectorize.io/hindsight/sdks/integrations/openclaw)
- [Hindsight Cloud](https://ui.hindsight.vectorize.io)
