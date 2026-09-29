---
title: "Hermes Multi-User Memory Setup with Hindsight"
authors: [benfrank241]
date: 2026-04-20
tags: [how-to, hermes, memory, multi-user, configuration]
description: "Set up Hermes multi-user memory with Hindsight, map each user to the right bank, and keep recall isolated so one user never inherits another user’s context."
image: /img/blog/guide-hermes-multi-user-memory-with-hindsight.png
hide_table_of_contents: true
---

If you want **Hermes multi-user memory with Hindsight**, the design decision that matters most is not `memory_mode`. It is how you derive `bank_id` for each user. A single static bank is fine for a personal assistant. It is risky in a multi-user deployment, because recall gets better only when the bank contains the right person's history.

The safe rule is simple: one user, one bank identity. In practice that usually means deriving `bank_id` from a stable user identifier, and often a tenant identifier too, then passing that value into the Hermes Hindsight provider for each session. Once that is in place, recall stays relevant and memory isolation becomes much easier to reason about.

This guide shows how to pick a bank naming scheme, how to generate per-user config for Hermes, when to include a tenant prefix, and how to verify that one user's preferences never bleed into another user's session. Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [docs home](https://hindsight.vectorize.io/docs), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), and the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Derive a unique `bank_id` per user, and per tenant if your deployment is multi-tenant.
> 2. Pass that bank into Hermes before the session starts.
> 3. Keep `memory_mode` and `prefetch_method` separate from identity, they solve different problems.
> 4. Verify isolation by storing a fact for User A, then confirming User B cannot recall it.
> 5. Add an environment suffix if you run staging and production side by side.

## Prerequisites

Before you wire Hermes for multiple users, make sure:

- Hermes is already using the native Hindsight provider.
- You can control how Hermes is launched for each user session.
- Your app has a stable user ID, not a transient connection ID.
- You know whether users belong to separate tenants.

If you still need the base provider setup, start with the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes), the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart), the [Retain API reference](https://hindsight.vectorize.io/docs/api/retain), and the [docs home](https://hindsight.vectorize.io/docs).

## Step by step

### 1. Pick the right bank naming pattern

A reliable multi-user naming scheme usually looks like one of these:

```text
user:12345
tenant:acme:user:12345
tenant:acme:user:12345:env:prod
```

The point is not the exact string format. The point is that it should be:

- stable across sessions
- unique per user
- easy to inspect during debugging
- safe across tenants and environments

If users can exist in multiple tenants, include the tenant. If you run staging and production against the same backend, include the environment too.

### 2. Generate per-user Hermes config before launch

Hermes stores Hindsight provider config in `~/.hermes/hindsight/config.json` by default. In a multi-user app, you can generate a session-specific config that sets `bank_id` for the current user:

```bash
python - <<'PY'
import json, pathlib
user_id = '12345'
tenant_id = 'acme'
base = pathlib.Path.home() / '.hermes'
path = base / 'hindsight' / 'config.json'
path.parent.mkdir(parents=True, exist_ok=True)

cfg = {
    'provider': 'hindsight',
    'hindsight_api_url': 'https://api.hindsight.vectorize.io',
    'api_key': 'YOUR_HINDSIGHT_TOKEN',
    'bank_id': f'tenant:{tenant_id}:user:{user_id}:env:prod',
    'memory_mode': 'hybrid',
    'prefetch_method': 'recall'
}

path.write_text(json.dumps(cfg, indent=2) + '\n')
print(f'Wrote {path} with bank_id={cfg["bank_id"]}')
PY
```

This is the core move. Everything else, including memory mode, sits on top of the bank identity.

### 3. Separate isolation from recall behavior

Do not mix these two concerns:

- `bank_id` decides whose memory you are reading.
- `memory_mode` decides how Hermes uses memory during a turn.

A safe starting point for multi-user assistants is:

```json
{
  "bank_id": "tenant:acme:user:12345:env:prod",
  "memory_mode": "hybrid",
  "prefetch_method": "recall"
}
```

Use `hybrid` when you want automatic recall plus explicit tools. If you prefer a cleaner tool surface, switch to `context`. The [Hermes memory modes guide](https://hindsight.vectorize.io/guides/2026/04/14/guide-hermes-memory-modes-with-hindsight-hybrid-context-tools) is the best place to tune that part later.

### 4. Add tenant and environment boundaries early

Multi-user bugs often show up only after a second customer lands in the same environment. That is why I recommend adding tenant and environment markers before you think you need them.

A good production pattern is:

```text
tenant:{tenant_id}:user:{user_id}:env:{environment}
```

This makes it obvious which bank a session should hit, and it gives you a predictable rule for debugging unexpected recall.

### 5. Verify isolation with two real users

Do not stop at config inspection. Test the actual behavior.

1. Launch Hermes as User A.
2. Tell it something durable, for example: “Remember that I prefer weekly summaries on Fridays.”
3. End the session so retention completes.
4. Launch Hermes as User B with a different `bank_id`.
5. Ask what it remembers about summary preferences.

Expected result: User B should not inherit User A's fact.

Then launch Hermes again as User A and confirm the preference is still present.

## Verifying it works

### Print the effective bank ID

```bash
python - <<'PY'
import json, os, pathlib
base = pathlib.Path(os.environ.get('HERMES_HOME', pathlib.Path.home() / '.hermes'))
path = base / 'hindsight' / 'config.json'
cfg = json.loads(path.read_text())
print('bank_id:', cfg.get('bank_id'))
print('memory_mode:', cfg.get('memory_mode'))
print('prefetch_method:', cfg.get('prefetch_method'))
PY
```

### Run an isolation test, not just a health check

`hermes memory status` tells you the provider is configured. It does not prove isolation. Isolation is proven only when two users cannot recall each other's context.

## Troubleshooting / common errors

### Two users still share context

Usually they are still using the same `bank_id`, or a wrapper forgot to overwrite the default config before launch.

### Recall is empty for returning users

The provider may be switching bank IDs between sessions because the user identity source is unstable. Do not use a connection ID or a temporary session token.

### Staging polluted production memory

Add an explicit environment suffix to the bank naming scheme.

## FAQ

### Should I use one bank per user or one bank plus tags?

For most Hermes multi-user setups, one bank per user is the simpler and safer answer. Tags are useful when you have a strong reason to group users inside one retrieval space.

### Should I include the tenant even if user IDs are globally unique?

If you are certain they are globally unique, you can skip it. In practice, tenant prefixes make debugging easier and reduce future surprises.

### Do I need a different memory mode for multi-user setups?

No. Identity and recall mode are separate decisions. Start with `hybrid` unless you have a clear reason to prefer `context` or `tools`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a managed backend for multi-user Hermes.
- Keep the [Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes) open for the provider config surface.
- Use the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) if you still need a backend.
- Read the [Recall API reference](https://hindsight.vectorize.io/docs/api/recall) and [Retain API reference](https://hindsight.vectorize.io/docs/api/retain) to understand what gets surfaced and stored.
- Compare the bank-scoping tradeoffs in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents).
