---
title: "Guide: Add NemoClaw Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, nemoclaw, agents, memory]
description: "Add NemoClaw memory with Hindsight using the hindsight-nemoclaw setup command, so sandboxed OpenClaw agents recall past sessions and retain new conversations without code changes."
image: /img/guides/guide-nemoclaw-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add NemoClaw Memory with Hindsight](/img/guides/guide-nemoclaw-memory-with-hindsight.svg)

If you want **NemoClaw memory with Hindsight**, the cleanest setup is the `hindsight-nemoclaw` command. NemoClaw runs OpenClaw inside an OpenShell sandbox with strict network egress policies, and this package automates the full setup in one command: it installs the `hindsight-openclaw` plugin, configures external API mode, merges the Hindsight egress rule into your sandbox policy, and restarts the gateway. No code changes required.

This is a good fit for NemoClaw because the sandbox enforces strict network egress — every outbound endpoint must be explicitly permitted. The `hindsight-openclaw` plugin supports an external API mode, where it skips the local daemon and makes direct HTTPS calls to Hindsight Cloud. That turns the plugin into a thin HTTP client, so the only sandbox change needed is one egress rule for the Hindsight API.

This guide walks through running the setup command, understanding how the plugin hooks into the OpenClaw gateway lifecycle, choosing a bank strategy, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Sign up for [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and get an API key.
> 2. Run `npx @vectorize-io/hindsight-nemoclaw setup --sandbox <name> --api-token <key> --bank-prefix <prefix>`.
> 3. The command installs the `hindsight-openclaw` plugin, applies the egress policy, and restarts the gateway.
> 4. Memories go to a bank named `<prefix>-openclaw`.
> 5. Verify that a later session recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A NemoClaw sandbox with `openshell` and `openclaw` installed and working
- A Hindsight API token, from [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a self-hosted server
- Ability to update the sandbox network policy (or use `--skip-policy` if you manage policies manually)

## Step 1: Run the setup command

The one-command setup installs and configures everything:

```bash
npx @vectorize-io/hindsight-nemoclaw setup \
  --sandbox my-assistant \
  --api-token <your-api-key> \
  --bank-prefix my-sandbox
```

You'll see output like:

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

Use `--dry-run` first if you want to preview all changes before applying anything.

## Step 2: Understand the setup steps

The setup command performs five steps:

1. **Preflight** — verifies `openshell` and `openclaw` are installed
2. **Install plugin** — runs `openclaw plugins install @vectorize-io/hindsight-openclaw`
3. **Configure plugin** — writes external API mode config to `~/.openclaw/openclaw.json`
4. **Apply policy** — reads the current sandbox policy, merges the Hindsight egress block, and re-applies via `openshell policy set`
5. **Restart gateway** — runs `openclaw gateway restart`

The setup handles the fact that `openshell policy set` replaces the entire policy document, so it exports the current policy first and merges the Hindsight block in — existing rules aren't lost.

## Step 3: Choose a bank strategy

The plugin config controls how memory banks are named and isolated:

| Option | Type | Default | Description |
|---|---|---|---|
| `hindsightApiUrl` | string | — | Hindsight API base URL |
| `hindsightApiToken` | string | — | API token for authentication |
| `llmProvider` | string | auto-detect | LLM provider for memory extraction |
| `dynamicBankId` | boolean | `false` | Isolate memory per user (`true`) or share across sessions (`false`) |
| `bankIdPrefix` | string | `"nemoclaw"` | Prefix for the memory bank name |

When `dynamicBankId: false`, all sessions write to a single bank named `{bankIdPrefix}-openclaw`. When `dynamicBankId: true`, each user gets an isolated bank — useful for multi-tenant deployments.

Setting `llmProvider: "claude-code"` uses the Claude Code process already present in the sandbox, so no additional API key is needed for memory extraction.

## How the plugin uses memory

Once set up, the `hindsight-openclaw` plugin hooks into the OpenClaw gateway lifecycle at two points:

- **Recall (`before_agent_start`):** it recalls relevant memories from past sessions and injects them into context.
- **Retain (`agent_end`):** when the agent finishes, it retains the conversation to the Hindsight memory bank.

The sandbox doesn't interfere with either step — it sees the Hindsight calls as normal HTTPS egress to a permitted endpoint. For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

After setup, check the gateway logs:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight
```

On startup you should see:

```
[Hindsight] Plugin loaded successfully
[Hindsight] ✓ Using external API: https://api.hindsight.vectorize.io
[Hindsight] External API health: {"status":"healthy","database":"connected"}
[Hindsight] Default bank: my-sandbox-openclaw
[Hindsight] ✓ Ready (external API mode)
```

After a conversation:

```
[Hindsight] before_agent_start - bank: my-sandbox-openclaw, channel: undefined/webchat
[Hindsight Hook] agent_end triggered - bank: my-sandbox-openclaw
[Hindsight] Retained 6 messages to bank my-sandbox-openclaw for session agent:main:...
```

If the retain line appears after a session and a later session recalls that context, the setup is working.

## Common mistakes

### Expecting policy merge when applying manually

`openshell policy set` replaces the entire policy document. The setup command handles this automatically, but if you apply the policy yourself, export the current policy first so existing rules aren't lost.

### Symlinked plugin install on macOS

On macOS, the OpenClaw gateway runs as a LaunchAgent under a restricted security context. `openclaw plugins install --link` creates a symlink the LaunchAgent can't follow. If you see `EPERM: operation not permitted, scandir` in gateway logs, install as a copy instead — which is what the setup command does.

### Testing retain too early

Fact extraction and entity resolution happen in the background after retain. If you open a new session immediately after closing one, the most recent memories may not be indexed yet — typically a few seconds.

### Assuming egress survives binary upgrades

The `binaries` field in the network policy restricts the egress rule to a specific executable path. If OpenClaw updates and the binary path changes, the rule silently stops working. Check your binary path after upgrades.

## FAQ

### Do I need Hindsight Cloud?

No. You can point the plugin at a self-hosted Hindsight server with `--api-url` (or `hindsightApiUrl` in the config). Hindsight Cloud is the natural fit for sandboxes because it needs only a single HTTPS egress rule.

### Does this require code changes to my sandbox?

No. The setup command installs the plugin, writes the config, applies the egress policy, and restarts the gateway — no code changes required.

### How is memory scoped?

By bank. With `dynamicBankId: false` all sessions share a single `{bankIdPrefix}-openclaw` bank; with `dynamicBankId: true` each user gets an isolated bank.

### What if I manage sandbox policies manually?

Pass `--skip-policy` to skip the network policy update, then add the Hindsight egress block to your policy yourself.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [NemoClaw integration docs](https://hindsight.vectorize.io/docs/integrations/nemoclaw)
