---
title: "Move Hermes Memory to Hindsight Cloud Guide"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, hermes, memory, cloud, migration]
description: "Move Hermes memory from local files to Hindsight Cloud, keep the setup clean, and verify that recall still works after the switch to a managed backend."
image: /img/blog/guide-move-hermes-memory-from-local-files-to-hindsight-cloud.png
hide_table_of_contents: true
---

If you want to **move Hermes memory from local files to Hindsight Cloud**, the migration is mostly about separating two kinds of memory that used to live together. Hermes can keep its local, bounded file memory for agent-local context, while Hindsight Cloud becomes the durable long-term memory backend that survives sessions and scales better across devices or shared workflows.

The important part is not copying every local artifact byte for byte. It is preserving the durable facts you actually care about, then pointing Hermes at the managed Hindsight backend so future retention and recall happen there. In practice, that means backing up your current Hermes memory files, switching the provider config to Cloud, and validating recall with a few known facts before you fully trust the new path.

This guide shows a safe migration path, how to configure Hermes for Hindsight Cloud, and how to test that your assistant still remembers what matters after the switch. Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain), and the [docs home](https://hindsight.vectorize.io/docs) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Back up your current Hermes local memory files first.
> 2. Decide which durable context is worth preserving.
> 3. Configure Hermes to use Hindsight Cloud with a stable `bank_id`.
> 4. Seed the new bank with a short retained summary if needed.
> 5. Verify recall before you remove or ignore the old local files.

## Prerequisites

Before you migrate, make sure:

- You have a Hindsight Cloud account and API token.
- Hermes already works locally, so you know what behavior you are trying to preserve.
- You can edit `~/.hermes/hindsight/config.json`.

Reference material: [Hindsight Cloud](https://hindsight.vectorize.io), [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain).

## Step by step

### 1. Back up the local Hermes memory files

Make a backup before you change anything:

```bash
mkdir -p ~/hermes-memory-backup
cp -R ~/.hermes ~/hermes-memory-backup/hermes-$(date +%Y%m%d-%H%M%S)
```

You may not need every file later, but you do want a clean rollback point.

### 2. Decide what should be preserved

Local file memory often contains a mix of durable context and short-lived session detail. For the move to Hindsight Cloud, preserve the durable layer:

- user preferences
- project context
- ongoing commitments
- key decisions
- stable workflows

If needed, create a compact summary you can retain into the new bank once Cloud is configured.

### 3. Configure Hermes for Hindsight Cloud

Update the native Hindsight provider config:

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get('HERMES_HOME', pathlib.Path.home() / '.hermes'))
path = base / 'hindsight' / 'config.json'
path.parent.mkdir(parents=True, exist_ok=True)

cfg = {
    'provider': 'hindsight',
    'hindsight_api_url': 'https://api.hindsight.vectorize.io',
    'api_key': 'YOUR_HINDSIGHT_TOKEN',
    'bank_id': 'hermes-primary',
    'memory_mode': 'hybrid',
    'prefetch_method': 'recall'
}

path.write_text(json.dumps(cfg, indent=2) + '\n')
print(f'Updated {path}')
PY
```

This does not delete your local Hermes files. It changes where long-term memory is read and written going forward.

### 4. Seed the new bank if you need continuity on day one

If there are a few critical facts you want available immediately, retain a compact summary into the new bank. For example:

```text
The user prefers concise answers, is launching Product Alpha in May, uses Postgres, and cares most about billing reliability and release checklists.
```

You do not need to seed everything. Seed only the durable context that would be frustrating to lose.

### 5. Verify recall in a fresh session

After switching to Cloud:

1. Launch Hermes.
2. Teach it one new durable fact.
3. End the session.
4. Start a new session.
5. Ask about the fact.

If recall works across the fresh session, the Cloud path is healthy.

## Verifying it works

### Print the active config

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get('HERMES_HOME', pathlib.Path.home() / '.hermes'))
path = base / 'hindsight' / 'config.json'
cfg = json.loads(path.read_text())
print('hindsight_api_url:', cfg.get('hindsight_api_url'))
print('bank_id:', cfg.get('bank_id'))
print('memory_mode:', cfg.get('memory_mode'))
PY
```

### Run a clean-session recall test

A successful migration is not “the file looks right.” It is “a new session still recalls the right durable context.”

## Troubleshooting / common errors

### Recall stopped working after the switch

Usually the API token or URL is wrong, or the new `bank_id` does not match the bank you expected.

### The assistant feels like it forgot everything

That usually means nothing was seeded and the old durable context only existed in local files. Retain a compact summary into the new bank to bridge the gap.

### I am seeing staging and production mix together

Split the bank by environment, for example `hermes-primary-prod` and `hermes-primary-staging`.

## FAQ

### Do I need to migrate every local memory file exactly?

No. Preserve the durable facts, not every artifact.

### Should I keep the same `bank_id` forever after moving to Cloud?

If you want continuity, yes. Change it only when you intentionally want a fresh bank.

### Is Hindsight Cloud better than local files for every Hermes use case?

Not automatically. Cloud is better when you want durability, sharing, or simpler operations. Local files can still be fine for narrow personal workflows.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you have not created the backend yet.
- Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) open for the native provider configuration surface.
- Use the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you want the self-hosted path instead of Cloud.
- Read the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) and [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) if you want to seed or tune the new bank more deliberately.
- Compare adjacent migration ideas in [Adding memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) and [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents).
