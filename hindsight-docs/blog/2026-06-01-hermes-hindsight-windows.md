---
title: "Hermes Agent on Windows: Add Persistent Codebase Memory with Hindsight"
authors: [benfrank241]
slug: "2026/06/01/hermes-hindsight-windows"
date: 2026-06-01T12:00
tags: [hermes, windows, coding, memory, agents, tutorial, hindsight]
description: "Nous Research just announced Hermes Agent native on Windows. Here's how to add persistent codebase memory with Hindsight — no Docker, no WSL, PowerShell start to finish."
image: /img/blog/hermes-hindsight-windows.png
hide_table_of_contents: true
---

![Hermes Agent on Windows with Hindsight](/img/blog/hermes-hindsight-windows.png)

Yesterday, Nous Research [announced that Hermes Agent is now natively supported on Windows](https://x.com/NousResearch/status/2061236625925886252). Here's how to give it persistent codebase memory.

If Windows is your daily driver, the gap has been real — WSL workarounds, Mac-only binaries, embedded databases that assume `bash`. Hermes just closed its side of that gap. Hindsight closes the memory side.

[Hermes Agent](https://github.com/NousResearch/hermes-agent) with Hindsight runs natively on Windows. No Docker. No WSL. PowerShell from start to finish. Embedded PostgreSQL handles the storage layer, the daemon spawns as a normal Windows process, and your codebase memory compounds across sessions the same way it does on every other platform.

This post is the Windows-specific companion to [Building a Hermes Coding Assistant That Remembers Your Codebase](/blog/2026/05/25/hermes-coding-assistant-codebase-memory). Same workflow, same compounding memory — different setup steps.

On the Hindsight side, we've been verifying the Windows path for weeks: a [daily Windows smoke test](https://github.com/vectorize-io/hindsight/blob/main/.github/workflows/windows-smoke.yml) runs the full API + embedded Postgres + Python client stack on `windows-latest` every morning at 06:00 UTC. Now that Hermes is native too, the pair is ready out of the gate.

<!-- truncate -->

---

## What You Get on Windows

The full Hermes + Hindsight feature set, native:

- **Embedded PostgreSQL** — Hindsight ships with `pg0`, an embedded Postgres distribution. No separate database install, no `winget install postgresql` step. First boot unpacks the binaries and runs `initdb`; subsequent starts are fast.
- **Auto-recall and auto-retain** — every coding session starts with relevant memories injected into Hermes's context, and every session retains what you discussed for the next one.
- **Local-only mode** — your memory bank lives in `~/.hindsight/` on your machine. Nothing leaves the laptop unless you point it at Hindsight Cloud.

The only Windows-specific gotcha is character encoding (more on that below).

---

## Two-Minute Setup

You need three things: Python 3.11+, an LLM API key (or a local provider), and Hermes installed.

### 1. Install Hindsight and Hermes

In PowerShell:

```powershell
# Hindsight API (includes embedded PostgreSQL)
pip install hindsight-api

# Hermes Agent
pip install hermes-agent
```

### 2. Set the UTF-8 environment variables

This is the one Windows-specific step. Hermes and Hindsight log status with checkmarks and box-drawing characters (✓, ─, │). Default Windows uses `cp1252`, which crashes on those glyphs. Force UTF-8 in your PowerShell profile:

```powershell
# One-time, for this session
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# Or permanent — add to $PROFILE
Add-Content $PROFILE "`$env:PYTHONUTF8 = '1'"
Add-Content $PROFILE "`$env:PYTHONIOENCODING = 'utf-8'"
```

This matches what the Hindsight Windows CI uses on every run. Without it, you'll see a `UnicodeEncodeError` the first time the API server tries to log a startup banner.

### 3. Run the Hermes setup wizard

```powershell
hermes memory setup
# Select "hindsight" when prompted
```

The wizard asks you to pick local or cloud mode. For local (everything on your Windows box):

- **Provider:** OpenAI, Anthropic, Gemini, Groq, Ollama — your pick
- **API key:** Pasted at the prompt (masked)

For cloud, paste your `hsk_...` token from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect).

### 4. Confirm it works

```powershell
hermes memory status
```

You should see `provider: hindsight` and `status: ready`. On first run with local mode, the embedded Postgres unpacks and initializes — give it 60–90 seconds on a cold machine.

---

## What Hermes Remembers on Your Windows Project

Same engine, same extraction. The kind of facts that enter memory automatically after a typical coding session:

- `"Project targets .NET 8, all new code under src/ uses file-scoped namespaces"`
- `"PowerShell deployment script fails on long file paths without the LongPathsEnabled registry key"`
- `"We switched from EF Core lazy loading to explicit Include() chains after the N+1 incident in March"`
- `"Convention: all async controllers wrap calls in ConfigureAwait(false) — except Razor pages, which don't need it"`

You don't tell Hermes any of this. The write pipeline extracts it from your conversation — the questions you ask, the bugs you describe, the decisions you make along the way.

---

## Three Workflows Where It Pays Off

### Starting a New Session

Without memory, every session on a complex Windows project starts with context-setting: the .NET version, the SDK, which projects depend on which, the deployment quirks. You burn 10 minutes before any actual work happens.

With Hermes + Hindsight, those facts are already injected. You open with the actual question:

> "Help me debug this timeout in the OrderService."

Hermes already knows the project's framework, the convention for async handling, and the deployment configuration. The first message is real work.

### Debugging Recurring Windows-Specific Issues

Some bugs only show up on Windows — path separators, line endings, file locking, antivirus interference with build outputs. Without memory, you re-diagnose each one independently.

With memory, when you describe a new failure that smells familiar, Hermes recalls the prior instance:

- `"MSBuild fails intermittently when Windows Defender real-time protection scans the obj/ directory mid-build — added an exclusion in 2026-03"`
- `"File lock errors on dotnet publish are usually visual-studio.exe still holding the previous output — closing VS clears it"`

These are the facts you'd never paste at the start of a debugging session. They're the ones that save the most time when recalled at the right moment.

### Onboarding a Teammate to Your Windows Project

If a teammate joins and you share a bank ID, they inherit your accumulated context immediately. They query Hermes for what it knows:

```
What do you know about the build pipeline?
```

```
What quirks have come up with the Authentication service on Windows?
```

The answers come from your past sessions — including the Windows-specific gotchas that never made it into the README.

---

## Local Mode: Why It's the Right Default on Windows

For most Windows developers, local mode is the right choice:

- **No external dependencies** — embedded Postgres, no separate install
- **No cloud round-trip** — your codebase facts never leave the machine
- **Offline-friendly** — works on flights and air-gapped networks (you only need network for the LLM provider)

If you want to share memory with a team or work across multiple Windows machines, cloud mode is a one-line config change. Same bank, accessed from anywhere.

The Hindsight v0.5.5 release added [native Windows support for embedded mode](/blog/2026/04/28/version-0-5-5), and every subsequent release has been verified on Windows. The [v0.4.21 native Windows post](/blog/2026/03/30/version-0-4-21) has the original walkthrough for running without Docker.

---

## Common Windows Gotchas

A few things to watch for, distilled from the Windows CI runs and user reports:

- **UTF-8 encoding** — covered above. Always set `PYTHONUTF8=1`.
- **First-boot Postgres init time** — 60–90 seconds on a cold Windows machine while `pg0` unpacks Postgres and runs `initdb`. Logs land at `~/.hindsight/profiles/<profile>.log`.
- **Long file paths** — if your projects live deep in the tree, enable `LongPathsEnabled` in the registry. Hindsight itself stays under the 260-char limit, but some Python wheel installs don't.
- **Antivirus on the daemon** — Windows Defender occasionally flags the embedded Postgres binary on first unpack. Add `~/.hindsight/` to the exclusion list if you hit slow startup or quarantine warnings.

---

## The Longer You Use It, the Less You Explain

Same payoff as on any other platform. Session one, Hermes knows nothing about your Windows project. Session five, it knows the framework, the conventions, the build pipeline. Session 30, it knows the Windows-specific quirks that never get documented but always trip up new contributors.

Hermes just shipped native Windows. Hindsight has been there. Plug them together and your codebase memory compounds on the platform you already work on.

Set it up with `hermes memory setup`, or start with the [Hindsight Windows installation guide](https://hindsight.vectorize.io/developer/installation#windows).

---

**Further reading:**
- [Building a Hermes Coding Assistant That Remembers Your Codebase](/blog/2026/05/25/hermes-coding-assistant-codebase-memory) — the platform-agnostic version of this post
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/) — foundational concepts
- [Hindsight v0.4.21 — Native Windows Support](/blog/2026/03/30/version-0-4-21) — the original Windows release
- [Hindsight v0.5.5 — Windows embedded-mode polish](/blog/2026/04/28/version-0-5-5) — local pg0 on Windows
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/) — full landscape comparison
