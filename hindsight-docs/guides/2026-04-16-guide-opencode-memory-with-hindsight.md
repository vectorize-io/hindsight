---
title: "Guide: Add OpenCode Memory with Hindsight"
authors: [benfrank241]
date: 2026-04-16
tags: [how-to, opencode, coding-agents, memory]
description: "Add OpenCode memory with Hindsight using the native plugin for auto-recall, auto-retain, and direct retain, recall, and reflect tools."
image: /img/blog/guide-opencode-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add OpenCode Memory with Hindsight](/img/blog/guide-opencode-memory-with-hindsight.png)

If you want **OpenCode memory with Hindsight**, the cleanest setup is to add the Hindsight plugin to OpenCode, point it at your Hindsight backend, and let it handle session recall, auto-retain, and direct memory tools for you. That gives OpenCode long-term memory across coding sessions instead of forcing each new session to rediscover the same context.

This is a good fit for coding workflows because OpenCode already has a plugin model that Hindsight can hook into. Once configured, the plugin can recall context when sessions begin, preserve memory when sessions go idle, and expose retain, recall, and reflect as explicit tools inside the agent workflow.

This guide walks through the plugin setup, the key config options, the difference between static and dynamic bank IDs, and a quick verification flow so you can confirm that memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Add `@vectorize-io/opencode-hindsight` to your OpenCode plugin config.
> 2. Point the plugin at your Hindsight backend.
> 3. Enable auto-recall and auto-retain.
> 4. Choose a bank ID strategy, static or dynamic.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- OpenCode installed and working
- A reachable Hindsight backend, local or [Hindsight Cloud](https://hindsight.vectorize.io)
- A decision about whether memory should be per project or shared across contexts

## Step 1: Add the plugin

Add the Hindsight plugin to your OpenCode config.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

OpenCode installs plugins from the array automatically at startup, so there is no separate `npm install` step for the plugin itself.

## Step 2: Point the plugin at Hindsight

For a local Hindsight server:

```bash
export HINDSIGHT_API_URL="http://localhost:8888"
opencode
```

For Hindsight Cloud:

```bash
export HINDSIGHT_API_URL="https://api.hindsight.vectorize.io"
export HINDSIGHT_API_TOKEN="your-api-key"
opencode
```

You can also configure plugin options inline in `opencode.json`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "hindsightApiUrl": "https://api.hindsight.vectorize.io",
      "hindsightApiToken": "your-api-key"
    }]
  ]
}
```

## What the plugin gives you

The plugin adds three direct tools:

- `hindsight_retain`
- `hindsight_recall`
- `hindsight_reflect`

It also supports:

- **auto-recall** when a session starts
- **auto-retain** when the session goes idle
- **compaction preservation**, so memory survives context window trimming

That means the integration works at two levels: the agent can call memory tools explicitly, and the plugin can also preserve memory automatically in the background.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Static vs dynamic bank IDs

### Static bank ID

A static bank is the simplest path. It works well when one OpenCode setup should always use one shared memory bank.

```bash
export HINDSIGHT_BANK_ID="my-project"
```

### Dynamic bank IDs

Dynamic bank IDs are better when one OpenCode environment moves across projects or users.

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
```

The plugin can derive the bank from fields such as agent, project, channel, or user. This is helpful when you want project isolation without manually rewriting the configuration every time.

## Recommended starter config

A practical starting point looks like this:

```json
{
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "hindsightApiUrl": "http://localhost:8888",
      "bankId": "my-project",
      "autoRecall": true,
      "autoRetain": true,
      "recallBudget": "mid",
      "retainEveryNTurns": 10,
      "debug": false
    }]
  ]
}
```

This gives you automatic memory behavior without making the config overly aggressive.

## Verify that memory is working

A good test sequence is:

1. start an OpenCode session
2. store a fact or project preference
3. close or idle the session
4. start a fresh session
5. ask about the saved fact

For example:

- session one stores that the repo prefers pnpm and strict TypeScript
- session two asks what the repo conventions are

If recall surfaces the stored preference, the setup is working.

You can also test explicit tool access by asking OpenCode to use retain, recall, or reflect directly.

## Common mistakes

### Using one static bank for unrelated projects

That is fine if you want shared memory, but risky if you expect project isolation.

### Turning on dynamic banks without understanding the naming inputs

If the derived IDs are not predictable, recall can look inconsistent.

### Forgetting that auto-retain happens on idle

If you test too early, the conversation may not have been stored yet.

### Assuming compaction means memory loss

The plugin explicitly retains before compaction and injects relevant memory back into context.

## FAQ

### Do I need Hindsight Cloud?

No. A local Hindsight server works too.

### Is the plugin enough by itself?

It handles the integration, but you still need a Hindsight backend.

### Should I use static or dynamic bank IDs?

Use static for one clear project bank. Use dynamic when the environment spans projects or users.

### Is this similar to Claude Code or Codex memory?

Yes in the broad sense, but OpenCode uses its own plugin and event model. For related workflows, compare [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) and [the Claude Code integration](https://hindsight.vectorize.io/docs/integrations/claude-code).

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare coding workflows in [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight)
