---
title: "Guide: Add Continue Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, continue, coding-agents, memory]
description: "Add Continue memory with Hindsight using the hindsight-continue adapter, so you can recall project memory into chat with @hindsight and optionally recall and retain automatically in agent mode."
image: /img/guides/guide-continue-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Continue Memory with Hindsight](/img/guides/guide-continue-memory-with-hindsight.svg)

If you want **Continue memory with Hindsight**, the cleanest setup is the `hindsight-continue` adapter. It runs a small local server that implements Continue's HTTP context-provider contract on top of Hindsight recall. Type `@hindsight` in Continue chat and relevant project memory is pulled in and injected into the model's context at query time.

This is a good fit for Continue because Continue has no hook that runs before a message is sent, but it does support two native extension points the package uses. The HTTP context provider gives precise, on-demand recall through `@hindsight`. An optional MCP server plus a rules file gives automatic recall and retain in agent mode, subject to the agent following the rule.

This guide walks through installing the adapter, pointing it at your Hindsight backend, registering it in Continue's config, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-continue`.
> 2. Set `HINDSIGHT_API_KEY` (Cloud) or `HINDSIGHT_API_URL` (self-hosted), plus `HINDSIGHT_CONTINUE_BANK_ID`.
> 3. Run `hindsight-continue` — it serves on `127.0.0.1:8123`.
> 4. Register it as an `http` context provider in Continue's `config.yaml`.
> 5. Type `@hindsight` in chat and verify the recalled memory shows up in context.

## Prerequisites

Before you start, make sure you have:

- Continue installed (VS Code or JetBrains extension)
- Python 3.10+ for the adapter
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the adapter

Install the adapter with pip.

```bash
pip install hindsight-continue
```

`hindsight-continue` is a tiny local server. It receives Continue's context-provider requests, recalls from Hindsight, and returns context items shaped exactly the way Continue expects.

## Step 2: Point the adapter at Hindsight

Set the environment variables, then run the adapter.

For Hindsight Cloud:

```bash
export HINDSIGHT_API_KEY=hsk_...
export HINDSIGHT_CONTINUE_BANK_ID=my-project

hindsight-continue            # serves on 127.0.0.1:8123
```

For a self-hosted Hindsight server, point at it with `HINDSIGHT_API_URL` and omit the key:

```bash
export HINDSIGHT_API_URL=http://localhost:8888
export HINDSIGHT_CONTINUE_BANK_ID=my-project

hindsight-continue            # serves on 127.0.0.1:8123
```

The bank is set by `HINDSIGHT_CONTINUE_BANK_ID`. Use one bank per project so context from one codebase doesn't leak into another.

## Step 3: Register it in Continue

Add the adapter as an `http` context provider in Continue's `config.yaml` (`~/.continue/config.yaml` or a workspace `.continue` block):

```yaml
context:
  - provider: http
    params:
      url: "http://127.0.0.1:8123/"
      title: hindsight
      displayTitle: Hindsight
      description: Recall long-term memory from Hindsight
```

Now in Continue chat, type `@hindsight` — optionally followed by what you want to recall — and the matching memories are added to the model's context.

## How the adapter uses memory

Continue has no pre-prompt hook, so the adapter works at the extension points Continue does expose:

- **Recall (HTTP context provider):** type `@hindsight <query>` (or a bare `@hindsight`) and the adapter receives Continue's request (`{query, fullInput, options, workspacePath}`), recalls from Hindsight, and returns context items shaped `{name, description, content}`. Recall happens at query time, on demand.
- **Recall and retain (optional, MCP + rules):** point Continue's agent mode at the Hindsight MCP server for `retain`/`recall`/`reflect` tools, with a rules file that tells the agent to recall at the start of every task and retain durable facts.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Automatic recall in agent mode

The `@hindsight` provider is precise, on-demand recall. For hands-off recall and retain, wire the Hindsight MCP server into Continue's agent mode and add a "recall first" rule. Example assets are in the [`examples/.continue/`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/continue/examples/.continue) directory:

- an MCP server block that registers the Hindsight tools
- a rules file that tells the agent to recall automatically

Automatic recall is subject to the agent following the rule, so it is less deterministic than the `@hindsight` provider.

## Per-project memory banks

Memory is scoped per **bank**. The `HINDSIGHT_CONTINUE_BANK_ID` you set is the default bank the adapter recalls against.

A single request can target a different bank by sending `options.bankId` in the provider config, so you can override the default per provider block. Using one bank per project keeps each codebase's memory isolated and lets other Hindsight editor integrations working on the same project share the same memory.

## Verify that memory is working

A good test sequence is:

1. run `hindsight-continue` and register it in Continue
2. store a decision or convention in the project's bank
3. open Continue chat and type `@hindsight` with a related query
4. confirm the recalled memory appears in the model's context

For example:

- store the current auth conventions for the project
- ask about the auth conventions with `@hindsight auth conventions`

If the recalled context surfaces the earlier decision, the setup is working.

## Common mistakes

### Forgetting to set the bank

Recall runs against `HINDSIGHT_CONTINUE_BANK_ID`. If it is unset, the adapter has no default bank to recall from — set one per project.

### Expecting passive recall on every message

Continue has no pre-prompt hook, so the `@hindsight` provider only recalls when you invoke it. Type `@hindsight` when you want memory in context, or use the MCP + rules setup in agent mode.

### Mismatched URL between adapter and config

The adapter serves on `127.0.0.1:8123` by default. If you change the host or port, update the `url` in Continue's `config.yaml` to match.

### Assuming memory is global

Memory is scoped per bank. That is usually what you want, but do not expect one project's bank to recall another project's context.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set `HINDSIGHT_API_URL` and omit the key.

### Does @hindsight recall automatically?

No. The `@hindsight` provider recalls on demand when you invoke it. For automatic recall, use the optional MCP server plus a rules file in agent mode.

### How is memory scoped?

Per bank, set by `HINDSIGHT_CONTINUE_BANK_ID`. Use one bank per project, and override per request with `options.bankId` if needed.

### Can I change the adapter's host or port?

Yes. Set `HINDSIGHT_CONTINUE_HOST` and `HINDSIGHT_CONTINUE_PORT`, then update the `url` in Continue's config to match.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Continue integration docs](https://hindsight.vectorize.io/docs/integrations/continue)
