---
title: "Shared Memory for OpenClaw: A Team Setup Guide"
authors: [benfrank241]
date: 2026-04-01T09:00
tags: [openclaw, memory, hindsight, teams, tutorial]
image: /img/blog/openclaw-shared-memory.png
description: "Every OpenClaw instance learns in isolation by default. Hindsight's shared memory bank connects them — one config change and your whole team's agents share one brain."
hide_table_of_contents: true
---

Every OpenClaw instance on your team is building up knowledge. Each one is learning your codebase, picking up your preferences, accumulating context from every conversation. But by default, that learning is isolated.

Dev A's instance figures out the right naming convention. Dev B's never hears about it. Dev C resolves a gnarly issue in the auth middleware; Dev D hits the same wall two weeks later. Each agent starts fresh in its own bank.

It doesn't have to work this way.

<!-- truncate -->

Hindsight supports shared memory banks — a single store that multiple OpenClaw instances read from and write to. Architecture decisions, known bugs, team conventions, institutional knowledge. One config change and every agent on your team draws from the same pool.

---

## What Default Memory Isolation Looks Like

The [hindsight-openclaw plugin](/sdks/integrations/openclaw) creates separate memory banks by default. Each unique combination of agent, channel, and user gets its own isolated store:

```
Alice's Telegram DMs   → bank: openclaw-telegram-alice
Bob's Telegram DMs     → bank: openclaw-telegram-bob
Alice's Slack DMs      → bank: openclaw-slack-alice
```

This is usually the right default. You don't want your Slack conversations bleeding into your Telegram ones, or Bob's personal context mixing with Alice's.

But there are situations where isolation is the wrong answer:

- **Same developer, two machines** — your laptop and your work desktop should share memory
- **Same developer, multiple channels** — your Telegram and Slack instances are the same person
- **Team shared knowledge** — conventions, architectural decisions, and known bugs that every agent should carry

These three cases need different configurations. The mechanism is the same in all of them: connect every instance to the same Hindsight server, and configure which fields are used to derive the bank ID.

---

## Pattern 1: Same Developer, Multiple Machines

If you run OpenClaw on a laptop and a desktop, the default config creates separate banks for each machine. They learn independently. Switch machines and your agent starts from scratch.

Fix this by connecting both machines to an external Hindsight server with identical config. In `~/.openclaw/openclaw.json` on **both** machines:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "hsk_your_token"
        }
      }
    }
  }
}
```

Same server, same bank derivation logic, same memories. The bank ID is still derived from agent + channel + user, so your Telegram DMs on your laptop and your Telegram DMs on your desktop resolve to the same bank.

---

## Pattern 2: Same Developer, All Channels

By default, your Telegram DMs and your Slack DMs are in different banks. Set `dynamicBankGranularity` to `["agent", "user"]` to merge them:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "hsk_your_token",
          "dynamicBankGranularity": ["agent", "user"]
        }
      }
    }
  }
}
```

Now everything you discuss across any channel writes to and reads from the same bank. Context from your Telegram conversations surfaces in Slack and vice versa.

---

## Pattern 3: Team Shared Memory

This is the highest-value use case. All team members' OpenClaw instances read from and write to a single shared bank. Conventions, known issues, architecture decisions — whatever one agent learns is available to every other agent on the team.

Connect every developer's instance to the same Hindsight server with a fixed bank ID. In `~/.openclaw/openclaw.json` on **every team member's machine**:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "hsk_your_team_token",
          "dynamicBankId": false,
          "bankId": "team-brain",
          "bankMission": "Shared memory bank for the engineering team. Stores architectural decisions, coding conventions, known bugs, and institutional knowledge.",
          "retainMission": "Extract and retain facts about the codebase architecture, coding standards, known bugs and their workarounds, team decisions and their rationale, and external system behaviors. Do not retain personal preferences, task-specific context, or anything that only applies to one developer's session."
        }
      }
    }
  }
}
```

The two key settings: `dynamicBankId: false` disables the per-context isolation, and `bankId: "team-brain"` sets the shared bank name. Every instance with this config writes to the same store.

---

## What Goes Into Team Memory

Not everything should go into a shared bank. One developer's debugging tangents and editor preferences will clutter every other agent's context.

Good candidates for team memory:

- **Conventions**: "We use snake_case for DB columns, camelCase in the API layer, never mix them."
- **Known bugs and gotchas**: "There's a race condition in the refresh token handler, see PR #441. Don't touch the token expiry logic without understanding that first."
- **Architecture decisions**: "We moved off SQLAlchemy to asyncpg in March, all new DB code uses asyncpg."
- **External system quirks**: "The Redis TTL in production is 15 minutes, not 30 as the README says."
- **Off-limits areas**: "The legacy billing service is being migrated by Q2. Don't add new features to it."

This is the `retainMission`'s job: telling Hindsight's extraction model what to capture and what to ignore. Without a focused mission, agents retain a mix of everything. With one, only team-relevant facts make it through.

Example missions for different team types:

**Product engineering team:**
```json
{
  "retainMission": "Extract technical decisions, API design choices, data model changes, and known issues. Retain the rationale behind decisions, not just the decision itself. Ignore one-off debugging sessions and personal editor preferences."
}
```

**Infrastructure team:**
```json
{
  "retainMission": "Extract facts about infrastructure configuration, deployment processes, service dependencies, known failure modes, and operational runbooks. Retain version constraints, environment-specific behavior, and incident learnings."
}
```

---

## What It Looks Like in Practice

Here's a concrete week on a team using shared memory.

**Monday morning.** Dev A starts on a new feature. Their OpenClaw instance already knows: the codebase uses asyncpg, not SQLAlchemy; the Redis TTL issue in production; and the naming convention for database columns. This came from the team bank — none of it required Dev A to ask anyone.

**Monday afternoon.** Dev A hits something unexpected: the JWT validation library silently ignores the `aud` claim when tokens have multiple audiences. They discuss it with their OpenClaw instance while working through the fix. The conversation ends. Hindsight processes the transcript and extracts it as a fact for the team bank: *"The JWT validation library does not validate the audience claim for tokens with multiple audiences. Manual validation required."*

**Tuesday morning.** Dev B starts working on a feature that touches authentication. Their instance recalls the JWT issue automatically, before Dev B has written a line of code. No Slack message. No code review comment. No one getting paged at 2am.

**Six weeks later.** A production incident surfaces: tokens with multiple audiences are being silently accepted. The on-call engineer's OpenClaw instance queries the team bank. The JWT issue is documented, with context. Resolution time: 12 minutes.

**Three months later.** A new engineer joins the team. On their first day, their OpenClaw instance has everything the team has learned. Not because anyone briefed them — because it's in the bank.

---

## Per-Team and Per-Project Namespacing

For organizations running multiple teams, use `bankIdPrefix` to namespace banks without needing separate servers:

```json
{
  "dynamicBankId": false,
  "bankId": "backend",
  "bankIdPrefix": "acme-",
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_team_token"
}
```

This produces bank ID `acme-backend`, isolated from an `acme-frontend` or `acme-infra` bank on the same server.

For multiple repositories with different conventions, use different `bankId` values and distribute the config as part of your team onboarding.

---

## Hosting Options

Three ways to run a shared Hindsight server:

| Option | Setup | Data control | Best for |
|--------|-------|-------------|----------|
| **Hindsight Cloud** | Zero setup, share an API token | Hosted by Vectorize | Most teams, including enterprise |
| **Self-hosted** | Deploy on your own infra via Docker | Fully yours | Regulated industries with on-prem requirements |
| **Local per-machine** | Each developer runs `hindsight-embed` independently | Local only | Individual use, not suitable for sharing |

For most teams, Hindsight Cloud is the right starting point. One API token shared across the team, one `bankId`, done. For teams with strict data residency requirements, the [self-hosted deployment](/developer/installation) gives you full control.

---

## Get Started

1. Create a [Hindsight Cloud account](https://ui.hindsight.vectorize.io/signup) and generate a team API token
2. Pick a `bankId` (e.g., `"team-brain"` or `"acme-backend"`)
3. Add the Pattern 3 config to `~/.openclaw/openclaw.json` on each developer's machine
4. Write a `retainMission` that focuses extraction on team-relevant facts
5. Have each developer run a session — the bank starts filling immediately

After a week of normal use, every agent on the team has meaningful shared context. After a month, it's the first thing developers notice: the agent already knows the codebase.

Your agents are learning. They might as well learn together.

---

*Set up the integrations: [OpenClaw](/sdks/integrations/openclaw) · [Memory banks reference](/developer/api/memory-banks)*
