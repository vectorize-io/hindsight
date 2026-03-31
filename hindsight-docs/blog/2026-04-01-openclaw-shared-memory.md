---
title: "Your OpenClaw Swarm Is Operating Without Shared Memory"
authors: [benfrank241]
date: 2026-04-01T09:00
tags: [openclaw, memory, hindsight, tutorial]
image: /img/blog/openclaw-shared-memory.png
description: "When you run multiple OpenClaw agents across Slack, Discord, and Telegram, they each start from zero — no shared context. Here's how to give the whole swarm a single memory bank."
hide_table_of_contents: true
---

You're running OpenClaw across multiple channels — Slack, Discord, Telegram, and more. Each instance is its own agent. Each is talking to your users, learning things, building up context. But by default, none of that knowledge is shared. Agent A has a conversation on Slack. Agent B starts fresh on Discord.

A swarm that can't share memory isn't really a swarm.

<!-- truncate -->

Hindsight solves this with shared memory banks — a single store that every instance in your swarm reads from and writes to. One agent learns something; every agent knows it. One config change.

---

## Why the Swarm Operates in Silos

The [hindsight-openclaw plugin](/sdks/integrations/openclaw) creates separate memory banks by default. Each unique combination of agent, channel, and user gets its own isolated store:

```
Agent on Slack    → bank: openclaw-agent-slack-user
Agent on Discord  → bank: openclaw-agent-discord-user
Agent on Telegram → bank: openclaw-agent-telegram-user
```

This default makes sense for strict isolation. But for a swarm — where multiple agents are serving the same team or user base — it means every agent is learning independently, with no way to propagate what it knows.

---

## The Setup: One Bank, Many Agents

Point every instance at the same external Hindsight endpoint and disable per-channel bank derivation. In `~/.openclaw/openclaw.json` on **every machine running an agent**:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "hsk_your_token",
          "dynamicBankId": false
        }
      }
    }
  }
}
```

`dynamicBankId: false` disables the default bank derivation. All agents write to and read from the same bank. What one learns, all know.

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  OpenClaw    │   │  OpenClaw    │   │  OpenClaw    │
│  (Slack)     │   │  (Discord)   │   │  (Telegram)  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Hindsight Memory    │
              │   (shared bank)       │
              └───────────────────────┘
```

---

## Pattern: Per-User Shared Memory

For some swarms, a single global bank is too broad. If your agents serve multiple users, you may want each user's context to be consistent across agents — but not bleed between users.

Set `dynamicBankGranularity` to `["user"]`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://api.hindsight.vectorize.io",
          "hindsightApiToken": "hsk_your_token",
          "dynamicBankGranularity": ["user"]
        }
      }
    }
  }
}
```

Now every agent in the swarm shares memory per user. User A's context follows them from Slack to Discord to Telegram. User B's context stays separate. The agents don't need to know which channel they're on — they just recall what's relevant for the user they're talking to.

---

## What Gets Shared

With a unified bank, knowledge accumulates across every agent in the swarm:

- **User preferences**: how they like responses structured, things to avoid, communication style
- **Ongoing projects**: what they're building, what stage it's at, who's involved
- **Recurring context**: schedules, key relationships, things being tracked
- **Decisions and history**: things users have mentioned, problems they're working through

Use `retainMission` to focus extraction on what actually travels:

```json
{
  "retainMission": "Extract user preferences, ongoing projects, recurring commitments, and important context. Retain facts that would be useful in any future conversation. Skip ephemeral task details and one-off requests."
}
```

Without a focused mission, the bank accumulates everything. With one, only the context that generalizes across conversations gets retained.

---

## What It Looks Like in Practice

A user tells your Slack agent they're launching a product next Friday and need to clear their calendar. The conversation ends; Hindsight extracts: *user has a product launch on [date]* and *user wants to protect their calendar this week*. Both facts land in the shared bank.

The next day, the same user opens Discord and asks your agent there to help prioritize their tasks. It already knows about the launch. The user didn't repeat themselves. They didn't paste in context. It just knew — because the Slack agent's learning is the Discord agent's starting point.

---

## A Note on Single-Instance Use

If you're running a single OpenClaw instance across many personal channels — Telegram, WhatsApp, iMessage — the same approach works. Each channel gets its own bank by default, but you can collapse them into one. Set `dynamicBankGranularity: ["agent", "user"]` to drop the channel dimension: all your channels on a given agent share the same memory store.

The swarm setup and the single-instance unified setup are the same mechanism — the difference is just how many machines are pointing at the bank.

---

## Hosting Options

Both patterns require an external Hindsight server so all agents can connect to the same store.

| Option | Setup | Data control | Best for |
|--------|-------|-------------|----------|
| **Hindsight Cloud** | Zero setup, one API token | Hosted by Vectorize | Most teams |
| **Self-hosted** | Deploy on your own infra via Docker | Fully yours | Privacy-sensitive setups |
| **Local per-machine** | Run `hindsight-embed` on each device | Local only | Single agent only, not shareable |

For most teams, Hindsight Cloud is the right starting point. Create an account, generate an API token, deploy the config across your agents. For setups requiring full data control, the [self-hosted deployment](/developer/installation) gives you that.

---

## Get Started

1. Create a [Hindsight Cloud account](https://ui.hindsight.vectorize.io/signup) and generate an API token
2. Deploy the config with `dynamicBankId: false` (or `dynamicBankGranularity: ["user"]`) on every agent instance
3. Add a `retainMission` focused on context that generalizes across conversations
4. Let the swarm run — the bank builds from the first conversation

The more your agents interact, the more the shared bank accumulates. Every agent in the swarm gets smarter from every other agent's conversations.

---

*Set up the integration: [OpenClaw](/sdks/integrations/openclaw) · [Memory banks reference](/developer/api/memory-banks)*
