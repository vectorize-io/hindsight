---
title: "Your OpenClaw Instances Are Learning in Isolation"
authors: [benfrank241]
date: 2026-04-01T09:00
tags: [openclaw, memory, hindsight, tutorial]
image: /img/blog/openclaw-shared-memory.png
description: "Every conversation builds context about you. Here's how to connect your Telegram, Slack, and desktop OpenClaw instances so they all learn from the same pool."
hide_table_of_contents: true
---

You use OpenClaw across multiple channels. Telegram on your phone. Slack at your desk. Maybe a few others. Each instance is learning about you — your preferences, your projects, how you like to communicate. But by default, what one learns stays with that one.

Switch channels and you're starting over. Your Slack OpenClaw has no idea what your Telegram instance learned last week. Same you, fragmented context.

It doesn't have to work this way.

<!-- truncate -->

Hindsight supports shared memory banks — a single store that multiple OpenClaw instances read from and write to. Everything your assistant learns about you, available everywhere you use it. One config change.

---

## Why Memory Gets Fragmented

The [hindsight-openclaw plugin](/sdks/integrations/openclaw) creates separate memory banks by default. Each unique combination of agent, channel, and user gets its own isolated store:

```
Your Telegram DMs   → bank: openclaw-telegram-you
Your Slack DMs      → bank: openclaw-slack-you
Your laptop         → bank: openclaw-telegram-you-laptop
Your phone          → bank: openclaw-telegram-you-phone
```

This default exists for good reasons — you don't want conversations from different contexts bleeding together in unpredictable ways. But it creates a problem: your assistant is learning about you in silos.

The fix depends on which silos you want to collapse:

- **Same channel, multiple machines** — your laptop and phone should share what they've learned
- **All your channels** — your Telegram and Slack instances are the same person; they should share context

---

## Pattern 1: Same Channel, Multiple Machines

If you use OpenClaw on both a laptop and a phone, each device maintains its own bank by default. Preferences you've shared on one device aren't known on the other.

Fix this by connecting both devices to an external Hindsight server. In `~/.openclaw/openclaw.json` on **both machines**:

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

The bank ID is still derived from agent + channel + user — so your Telegram DMs on your laptop and your Telegram DMs on your phone resolve to the same bank. The same context, available on both devices.

---

## Pattern 2: All Your Channels

This is the more valuable pattern. Your Telegram and Slack instances are the same person, but by default they're learning independently. Tell your Telegram OpenClaw about a project you're working on, then switch to Slack — and it has no idea.

Set `dynamicBankGranularity` to `["agent", "user"]` to merge all your channels into one bank:

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

This drops `channel` from the bank ID derivation. Every OpenClaw instance connected with your user account now reads from and writes to the same bank — regardless of whether you're on Telegram, Slack, WhatsApp, or anything else.

---

## What Gets Shared

With a unified bank, everything your assistant learns about you accumulates in one place:

- **Preferences**: how you like responses structured, your communication style, things to skip
- **Ongoing projects**: what you're building, what stage it's at, who's involved
- **Recurring context**: your weekly schedule, important relationships, things you're tracking
- **Things you've mentioned**: travel plans, decisions you've made, problems you're working through

The key is `retainMission` — a setting that tells Hindsight's extraction model what to focus on:

```json
{
  "retainMission": "Extract preferences, ongoing projects, recurring commitments, and important relationships. Retain context that would be useful in any future conversation with this person. Do not retain ephemeral task details or one-off requests."
}
```

Without a focused mission, the bank fills with a mix of everything. With one, only the context that travels well gets kept.

---

## What It Looks Like in Practice

You mention to your Telegram OpenClaw that you're planning a trip to Tokyo in June and need to wrap up a product launch before you go. The conversation ends; Hindsight processes it. Two facts land in your bank.

Two days later you open Slack and ask your OpenClaw to help you prioritize your week. It already knows about the launch deadline and the trip. You didn't repeat yourself. You didn't paste in context. It just knew.

Three weeks later you get a new phone. Fresh install, same config, same Hindsight token. Your first conversation picks up where you left off. The assistant knows your preferences, your projects, your context — because it's all in the bank.

---

## Hosting Options

Both patterns above require an external Hindsight server so all your instances can connect to the same store.

| Option | Setup | Data control | Best for |
|--------|-------|-------------|----------|
| **Hindsight Cloud** | Zero setup, one API token | Hosted by Vectorize | Most users |
| **Self-hosted** | Deploy on your own infra via Docker | Fully yours | Privacy-sensitive setups |
| **Local per-machine** | Run `hindsight-embed` on each device | Local only | Single device only, not shareable |

For most users, Hindsight Cloud is the right starting point. Create an account, generate an API token, drop it in your config. For setups where you need full data control, the [self-hosted deployment](/developer/installation) gives you that.

---

## Get Started

1. Create a [Hindsight Cloud account](https://ui.hindsight.vectorize.io/signup) and generate an API token
2. Add the Pattern 2 config to `~/.openclaw/openclaw.json` on each device and channel
3. Write a `retainMission` that focuses on context that travels across conversations
4. Start a session — the bank fills from the first conversation

The more you use it, the more your assistant knows. And now it knows consistently, everywhere you use it.

---

*Set up the integration: [OpenClaw](/sdks/integrations/openclaw) · [Memory banks reference](/developer/api/memory-banks)*
