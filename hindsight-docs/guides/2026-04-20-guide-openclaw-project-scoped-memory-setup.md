---
title: "OpenClaw Project-Scoped Memory Setup Guide"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, openclaw, memory, projects, configuration]
description: "Set up OpenClaw project-scoped memory with Hindsight, keep one project’s context separate from another, and choose when to use fixed banks versus dynamic isolation."
image: /img/blog/guide-openclaw-project-scoped-memory-setup.png
hide_table_of_contents: true
---

If you want **OpenClaw project-scoped memory**, the goal is simple: when the agent is working on Project A, it should not recall facts from Project B. In practice, the cleanest way to get that behavior is to give each project its own bank, either with a fixed `bankId` per project agent or with a deliberate deployment pattern that maps one agent instance to one project.

This is different from per-user memory. Per-user memory answers “who is talking?” Project-scoped memory answers “which body of work should this conversation draw from?” If your OpenClaw setup supports several repos, products, or client environments, project scoping often matters more than user scoping.

This guide shows when project-scoped memory is the right fit, how to create one bank per project, and how to verify that cross-project bleed is gone without losing continuity inside the project itself. Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), the [docs home](https://hindsight.vectorize.io/docs), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), and the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) nearby while you configure it.

<!-- truncate -->

> **Quick answer**
>
> 1. Create one bank per project, for example `project-alpha` and `project-beta`.
> 2. Configure each OpenClaw agent or deployment to use the correct fixed `bankId`.
> 3. Keep project memory separate from personal memory unless you truly need both combined.
> 4. Test by teaching a fact in one project and confirming it does not appear in another.
> 5. Add a focused `retainMission` so each project bank stores durable project knowledge.

## Prerequisites

Before you scope memory by project, make sure:

- OpenClaw and the Hindsight plugin are already installed.
- You can run separate OpenClaw agents or config variants per project.
- You know which projects deserve isolated memory banks.

Reference material: [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw), [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Recall API reference](https://hindsight.vectorize.io/docs/api/recall), and [docs home](https://hindsight.vectorize.io/docs).

## Step by step

### 1. Decide what counts as a project boundary

A project boundary should map to real work, for example:

- one repo
- one customer deployment
- one product line
- one internal initiative

If the projects are unrelated enough that wrong recall would be distracting, they deserve separate banks.

### 2. Use a fixed bank per project

The simplest reliable pattern is one fixed bank per project agent:

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
cfg['bankId'] = 'project-alpha'
path.write_text(json.dumps(config, indent=2) + '\n')
print(f'Updated {path}')
PY
```

Then give the second project a different bank:

```json
{
  "bankId": "project-beta"
}
```

### 3. Keep the retain mission project-focused

A project bank works best when the agent retains:

- architecture decisions
- naming conventions
- deployment constraints
- recurring workflows
- known bugs and workarounds

A useful project mission looks like:

```text
Extract durable project decisions, architecture context, workflows, constraints, and recurring implementation details. Ignore casual chatter and temporary task noise.
```

### 4. Map one agent or one config variant to one project

This is the cleanest deployment pattern:

- Agent A -> `project-alpha`
- Agent B -> `project-beta`

That model is easier to reason about than trying to switch one live agent between several project banks ad hoc.

### 5. Test for cross-project isolation

1. In Project Alpha, tell the agent: “Remember that Alpha uses Postgres and deploys from GitHub Actions.”
2. Let retention finish.
3. Switch to Project Beta.
4. Ask what it remembers about the deployment stack.

Expected result: the Project Alpha fact should not appear in Project Beta.

## Verifying it works

### Print the active bank

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

### Test with a distinctive project fact

Pick a fact that would be obviously wrong in the other project, then confirm it stays isolated.

## Troubleshooting / common errors

### The agent still recalls the wrong project context

You are probably reusing the same bank across both projects, or the wrong config file is being loaded.

### Useful personal preferences stopped showing up

That is normal if you split project memory from user memory. If you need both, keep them separate conceptually and decide which one should win for the workflow.

### One project bank is too noisy

Tighten the retain mission so only durable project context is stored.

## FAQ

### Is project-scoped memory better than per-user memory?

Not always. They answer different questions. Project scope is about work context. Per-user scope is about person context.

### Can one user work across several project banks?

Yes. That is often the point. The same person can interact with different project agents that each use their own bank.

### Should every repo get its own bank?

Only if wrong recall across repos would hurt more than shared context would help.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want one managed backend for multiple project agents.
- Keep the [OpenClaw integration docs](https://hindsight.vectorize.io/docs/integrations/openclaw) open for the full plugin configuration surface.
- Use the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you still need to stand up Hindsight.
- Read the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) before tuning retrieval or storage.
- Compare adjacent collaboration patterns in [OpenClaw shared memory](https://hindsight.vectorize.io/blog/openclaw-shared-memory) and [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents).
