---
title: "Hermes Agent on Windows: Set Up Persistent Memory with Hindsight"
authors: [benfrank241]
slug: "2026/06/01/hermes-hindsight-windows-setup"
date: 2026-06-01T18:00
tags: [hermes, windows, memory, agents, tutorial, hindsight]
description: "Nous Research just shipped Hermes Agent native on Windows. Here's how to give it persistent memory with Hindsight — one command, three modes, no Docker or WSL."
image: /img/blog/hermes-hindsight-windows-setup.png
hide_table_of_contents: true
---

![Hermes Agent on Windows with Hindsight](/img/blog/hermes-hindsight-windows-setup.png)

Earlier this week, Nous Research [announced that Hermes Agent is now natively supported on Windows](https://x.com/NousResearch/status/2061236625925886252). Here's how to give it persistent memory.

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a self-improving assistant with 40+ tools — chat, research, voice, gateway integrations into Telegram, Discord, Slack, and more. Out of the box, every conversation starts from zero: Hermes doesn't remember what you discussed yesterday, what project you were planning, or what preferences you've already shared. Hindsight closes that gap. Once configured, Hermes recalls relevant facts before every reply and retains the conversation afterward — across sessions, across platforms, across restarts.

This post is the platform-neutral Windows guide. If you specifically want a coding assistant with codebase memory, see the [Hermes Coding Assistant on Windows](https://hindsight.vectorize.io/blog/2026/06/01/hermes-hindsight-windows) companion.

<!-- truncate -->

---

## What You Get on Windows

The full Hermes + Hindsight feature set, native:

- **Three deployment modes** — pick Cloud, Local Embedded, or Local External in the setup wizard. No Docker, no WSL, no separate PostgreSQL install. Local Embedded ships with `pg0` (an embedded Postgres distribution); first boot unpacks the binaries and runs `initdb`, subsequent starts are fast.
- **Auto-recall and auto-retain** — every conversation starts with relevant memories injected into Hermes's context, and every conversation retains what you discussed for the next one.
- **Cross-platform identity** — if you run Hermes Gateway (Telegram, Discord, Slack), all platforms share the same memory bank. Tell Hermes something in Telegram, ask about it later from Discord, and it remembers.

The only Windows-specific gotcha is character encoding (more on that below).

---

## One-Command Setup

There's no Windows-specific install dance. From PowerShell:

```powershell
hermes memory setup
```

Pick **hindsight** from the provider list. The wizard then asks you to choose a mode:

```
Select mode
↑↓ navigate  ENTER/SPACE select  ESC cancel

→ (●) Cloud           Hindsight Cloud API (lightweight, just needs an API key)
  (○) Local Embedded  Run Hindsight locally (downloads ~200MB, needs LLM key)
  (○) Local External  Connect to an existing Hindsight instance
```

What each mode asks for:

- **Cloud** — your `hsk_...` token from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect). The default API URL (`https://api.hindsight.vectorize.io`) is correct — just accept it.
- **Local Embedded** — the wizard downloads and runs Hindsight locally (~200MB on first install, including the embedded PostgreSQL binary). You'll be prompted for an LLM provider (OpenAI, Anthropic, Gemini, Groq, Ollama, etc.), an API key for that provider, and an optional model override. Nothing else to install — `uvx` handles fetching `hindsight-embed`, and the daemon spawns automatically.
- **Local External** — point Hermes at a Hindsight instance you're already running (your own server, a teammate's, a self-hosted box). Provide the API URL and, if the instance requires it, an API key.

Confirm it's wired up:

```powershell
hermes memory status
```

You should see `provider: hindsight` and `status: ready`. On Local Embedded, first boot takes 60–90 seconds while pg0 unpacks Postgres and runs `initdb`.

---

## What Hermes Remembers

After a few conversations, the kind of facts that enter memory automatically:

- `"User prefers metric units and 24-hour time."`
- `"Planning a camping trip to Wyoming in August — group of six, mix of tent and cabin."`
- `"Weekly standup notes go to #team-eng on Slack at 5 PM ET."`
- `"User's writing voice is concise, avoids exclamation points, and prefers em-dashes."`

You don't tell Hermes any of this directly. The write pipeline extracts it from your conversations — the questions you ask, the decisions you make, the context you share along the way.

---

## Three Workflows Where Memory Pays Off

### Picking Up Where You Left Off

Without memory, every session starts with re-explaining. You spent yesterday planning a trip; today you open Hermes and say "let's keep going" — and it has no idea what you mean.

With Hindsight, the next session opens on the actual continuation:

> "Where did we land on the day-three itinerary for the Wyoming trip?"

Hermes already knows the trip, the group size, the constraints you've discussed, and the open questions. The first message is real work, not setup.

### Talking to Hermes Across Platforms

Hermes Gateway runs across Telegram, Discord, Slack, and other messaging platforms. Without memory, each platform feels like a different assistant — you tell Hermes a fact in Telegram on Monday, ask about it from Discord on Wednesday, and the Discord-side Hermes shrugs.

With Hindsight, the memory bank is shared. The plugin hooks every turn on every platform — whatever you say in one channel becomes recallable from any other. One assistant, many surfaces.

### Long-Running Research and Writing

Multi-week projects — a research dive, a book outline, a job hunt — accumulate hundreds of facts. Without memory, you fight the agent for context every session: re-pasting the brief, re-explaining who the audience is, re-stating constraints you set three weeks ago.

With memory, you query what Hermes already knows:

```
What did we decide about the audience for the agent-memory ebook?
```

```
What's still open on the job hunt — which roles am I waiting to hear back on?
```

The answers come from prior sessions — including the small decisions and side-notes that never made it into a doc.

---

## Which Mode Should You Pick?

A quick decision tree:

- **Solo, want zero ops:** Cloud. One API key, no daemon to think about, memory accessible from any machine you log into.
- **Solo, privacy-sensitive or offline-heavy:** Local Embedded. Memory bank stays on your machine, works on flights and air-gapped networks (you still need the network for the LLM call itself).
- **Team with a shared Hindsight server:** Local External. Point each user's Hermes at the same Hindsight instance and share a bank ID — institutional knowledge accumulates across the team.

The Hindsight v0.5.5 release added [native Windows support for embedded mode](/blog/2026/04/28/version-0-5-5), and every subsequent release has been verified on Windows. The [v0.4.21 native Windows post](/blog/2026/03/30/version-0-4-21) has the original walkthrough for running without Docker.

---

## Common Windows Gotchas

A few things to watch for, distilled from the Windows CI runs and user reports:

- **UTF-8 encoding** — Hermes and Hindsight log status with checkmarks and box-drawing characters (✓, ─, │) that crash Windows's default `cp1252` codec. If you see a `UnicodeEncodeError` on first run, set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` in your PowerShell profile. The Hindsight [Windows smoke test](https://github.com/vectorize-io/hindsight/blob/main/.github/workflows/windows-smoke.yml) sets both on every run for the same reason.
- **First-boot Postgres init time** (Local Embedded only) — 60–90 seconds on a cold Windows machine while `pg0` unpacks Postgres and runs `initdb`. Logs land at `~/.hindsight/profiles/<profile>.log`.
- **Long file paths** — if your home directory or workspace lives deep in the tree, enable `LongPathsEnabled` in the registry. Hindsight itself stays under the 260-char limit, but some Python wheel installs don't.
- **Antivirus on the daemon** — Windows Defender occasionally flags the embedded Postgres binary on first unpack. Add `~/.hindsight/` to the exclusion list if you hit slow startup or quarantine warnings.

---

## The Longer You Use It, the Less You Explain

Session one, Hermes knows nothing about you. Session five, it knows the units you prefer, the projects you're juggling, the people in your life. Session 30, it knows the recurring tasks, the writing voice, the small preferences that never get documented but always trip up a fresh assistant.

Hermes just shipped native Windows. Hindsight has been there. Plug them together and your assistant gets a memory on the platform you already work on.

Set it up with `hermes memory setup`, or start with the [Hindsight Windows installation guide](https://hindsight.vectorize.io/developer/installation#windows).

---

**Further reading:**
- [Hermes Coding Assistant on Windows](https://hindsight.vectorize.io/blog/2026/06/01/hermes-hindsight-windows) — the codebase-memory companion to this post
- [Building a Hermes Coding Assistant That Remembers Your Codebase](/blog/2026/05/25/hermes-coding-assistant-codebase-memory) — the platform-agnostic coding version
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/) — foundational concepts
- [Hindsight v0.4.21 — Native Windows Support](/blog/2026/03/30/version-0-4-21) — the original Windows release
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/) — full landscape comparison
