---
title: "OpenClaw and Claude Code Shared Memory Setup"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, openclaw, claude-code, memory, workflow]
description: "Set up shared memory between OpenClaw and Claude Code with Hindsight, reuse one bank across both tools, and verify that context moves cleanly between chat and coding work."
image: /img/blog/guide-openclaw-and-claude-code-shared-memory.png
hide_table_of_contents: true
---

If you want **OpenClaw and Claude Code shared memory**, the trick is not installing another memory system. It is making both tools point at the same Hindsight bank on purpose. Once that is true, context learned in chat can show up in coding sessions, and discoveries from coding sessions can come back into chat without anyone retyping the whole backstory.

This setup is useful when one workflow spans both tools. You might discuss product requirements with an OpenClaw assistant, switch to Claude Code to implement them, then return to OpenClaw for status updates. Without shared memory, each tool starts from zero. With a shared bank, they can reuse the same project facts, preferences, and working context.

This guide shows the safest way to wire that up, when to use a fixed shared `bankId`, how to keep the bank narrow enough to stay useful, and how to test that context really moves from one tool to the other. Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [Claude Code post](https://hindsight.vectorize.io/blog/claude-code-persistent-memory), and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open while you configure it.

<!-- truncate -->

> **Quick answer**
>
> 1. Pick a single shared `bankId` for the workflow, for example `team-product-alpha`.
> 2. Configure Claude Code and OpenClaw to use the same Hindsight backend and the same bank.
> 3. Use a focused retain mission so both tools store compatible, durable context.
> 4. Write one memory in OpenClaw, recall it in Claude Code, then test the reverse path.
> 5. Split the bank later if unrelated projects begin to pollute recall.

## Prerequisites

Before you share memory between tools, make sure:

- Hindsight is already running, local or cloud.
- OpenClaw is installed and the Hindsight plugin is healthy.
- Claude Code has the Hindsight plugin installed.
- You understand which project or team should own the shared bank.

If you still need the base setup, start with the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [docs home](https://hindsight.vectorize.io/docs), and the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain).

## Step by step

### 1. Choose one intentional shared bank ID

A shared bank should map to one real workflow, not your entire life. Good examples:

- `team-product-alpha`
- `customer-support-escalations`
- `launch-ops-q2`

Bad examples:

- `default`
- `shared`
- `everything`

The bank name should tell you who the memory is for and what work belongs in it. That one decision does more for recall quality than almost any token-setting tweak.

### 2. Point OpenClaw at the shared bank

Update the OpenClaw plugin config to use a fixed `bankId`:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
entries = config.setdefault('plugins', {}).setdefault('entries', {})
plugin = entries.setdefault('hindsight-openclaw', {'enabled': True, 'config': {}})
plugin['enabled'] = True
cfg = plugin.setdefault('config', {})
cfg['dynamicBankId'] = False
cfg['bankId'] = 'team-product-alpha'
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

This intentionally overrides the default dynamic bank behavior. You are telling OpenClaw, “for this agent, always write and recall from the same shared brain.”

### 3. Point Claude Code at the same bank

In the Claude Code Hindsight plugin config, use the same `bankId` and the same backend credentials:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "YOUR_TOKEN",
  "bankId": "team-product-alpha",
  "retainMission": "Extract durable project decisions, coding conventions, important constraints, and user preferences that should help across chat and coding sessions."
}
```

The exact location of your Claude Code plugin config can vary with install method, so use the plugin's own config surface, but the important thing is the shared `bankId`. If Claude Code points at `team-product-alpha` and OpenClaw points at `team-product-alpha`, they are reading and writing the same bank.

### 4. Align the retention mission across both tools

A shared bank only works if the stored memories make sense in both environments. That means both tools should prioritize:

- project decisions
- user preferences
- open tasks and constraints
- architecture context
- naming conventions and patterns

They should avoid storing:

- transient chat filler
- duplicate logs
- one-off operational noise

This is where the [team shared memory post](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) and [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) are helpful, because they show what a useful shared bank looks like in practice.

### 5. Test both directions

First, write in OpenClaw and recall in Claude Code.

1. In OpenClaw, tell the assistant: “Remember that the new launch checklist must include billing smoke tests and the team prefers concise summaries.”
2. Let the turn finish.
3. In Claude Code, ask it to continue launch work or summarize what it knows about the checklist.

Then test the reverse path.

1. In Claude Code, make a decision during implementation, for example: “We are using background jobs for webhook retries, not in-request processing.”
2. End the session so retention runs.
3. In OpenClaw, ask for a status summary.

If the shared bank is wired correctly, context should move both ways.

## Verifying it works

### Check the bank configuration

OpenClaw should show a fixed `bankId`:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path.home() / '.openclaw' / 'openclaw.json'
config = json.loads(path.read_text())
cfg = config['plugins']['entries']['hindsight-openclaw']['config']
print('dynamicBankId:', cfg.get('dynamicBankId'))
print('bankId:', cfg.get('bankId'))
PY
```

Claude Code should show the same bank in its Hindsight plugin config.

### Check cross-tool recall, not just local recall

The real success criterion is cross-tool continuity. If each tool recalls only what it retained itself, you still do not have shared memory.

## Troubleshooting / common errors

### OpenClaw and Claude Code still feel separate

One of them is almost always pointing at a different bank or a different backend URL.

### Recall quality is noisy

Your bank is too broad. Split by team, project, or customer workflow instead of using one bank for everything.

### One tool stores useful context, the other stores junk

Align the `retainMission`. Shared memory breaks down when one tool writes durable facts and the other writes conversation clutter.

## FAQ

### Should I use a fixed bank or dynamic banks here?

For true cross-tool sharing, use a fixed bank. Dynamic banks are great when you want automatic isolation, but a fixed bank is simpler when the whole point is sharing.

### Can multiple Claude Code users share the same bank too?

Yes, but only if that is intentional. A shared bank should match a real shared workflow, not a vague convenience setting.

### Is this better than copying notes manually between tools?

Yes, when the workflow is active and ongoing. Manual notes still matter, but shared memory reduces the repeated “catch-up” work.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want one backend shared between chat and coding tools.
- Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) and the [Claude Code persistent memory post](https://hindsight.vectorize.io/blog/claude-code-persistent-memory) open while you configure both sides.
- Use the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you still need a Hindsight server.
- Review the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) to tune how much context moves between tools.
- Read [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents) if you want to expand the pattern to a whole team.
