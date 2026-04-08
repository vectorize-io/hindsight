---
title: "One Memory for Every AI Tool I Use"
authors: [404sand808s]
date: 2026-04-07
tags: [mcp, memory, claude, claude-code, openai, hindsight, integration, self-hosting]
description: "How to wire Claude, ChatGPT, Claude Code, Codex, and OpenClaw to a single shared Hindsight memory bank using Cloudflare Workers as an OAuth 2.1 proxy."
image: /img/blog/one-memory-for-every-ai-tool.png
hide_table_of_contents: true
---

![One Memory for Every AI Tool I Use](/img/blog/one-memory-for-every-ai-tool.png)

Like many, I use different AI tools throughout the day depending on what I'm doing and where I am. Claude primarily, across the desktop and mobile apps, the Claude Code CLI, and the VS Code extension. Plus ChatGPT's mobile app and Codex on desktop.

<!-- truncate -->

A couple of weeks ago I added an OpenClaw agent to the mix, running on a Mac Mini and connected to Discord. People are split on it, but the hype around some of its capabilities clicked immediately: beyond the model itself, it has a much broader set of tools than conventional AI interfaces. But the most groundbreaking part was memory. Having OpenClaw remember things across conversations felt like a genuine shift, from isolated sessions to something persistent.

So why not do this with Claude or ChatGPT? Or better yet, why couldn't they all share one unified memory? That's the shared memory pattern I've been running for the past few weeks.

## The problem: no shared memory across AI tools

Every tool starts from zero. You explain your project to Claude, then explain it again to Codex, then again when you open a new Claude Code session. The context doesn't travel. Each tool treats you like a stranger because to it, you are one.

This isn't just inconvenient. It limits what these tools can do for you. The model that helped you design an architecture last week can't build on that work today, because it doesn't know it happened.

## The vision: one shared memory bank

Imagine: on the go, an idea strikes. You plug it into the Claude app and think through a few aspects. Later, you continue on desktop. Then you switch to Claude Code to start building, and it already has the context, the design decisions, and the reasoning behind them. Then you hand the project to OpenClaw. Each tool picks up where the last one left off, pulling from the same shared memory.

The effect is cumulative. After a week, you stop noticing the things you don't have to say anymore. Preferences, past decisions, project context, technical choices. It all carries over and makes each tool better at its job. The shared memory doesn't just save time, it makes the outputs better.

## How the shared memory stack is wired together

If you're using [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), none of the OAuth complexity below applies — it already speaks OAuth 2.1, so all cloud clients connect directly. The proxy setup is only needed if you're self-hosting.

My [Hindsight](/developer/api/quickstart) bank runs in Docker on an M4 Mac Mini (same as my OpenClaw). All clients connect to the same bank, but each one connects differently.

**OpenClaw** has the simplest setup. Since it's on the same machine, the [hindsight-openclaw plugin](/blog/2026/03/06/adding-memory-to-openclaw-with-hindsight) talks directly to Hindsight over `localhost` using the [standard Docker quick start](/developer/api/quickstart).

**Everything else** (the Claude desktop and mobile apps, Claude Code, and Codex) connects remotely via MCP. Since these tools aren't on the same machine, I needed to expose my local Hindsight instance through a public endpoint. The challenge is that cloud clients like the Claude apps require OAuth 2.1 with Dynamic Client Registration and PKCE to connect to an MCP server, while self-hosted Hindsight only speaks Bearer token auth. No OAuth endpoints, no discovery metadata. They can't talk directly. The stack needed to bridge this:

- **Hindsight** runs in Docker on the Mac Mini, bound to `localhost`
- **Cloudflare Tunnel** exposes it through a subdomain on a domain I own
- **Cloudflare Worker** (about 250 lines of code) sits in front, implementing the full OAuth 2.1 spec (discovery, DCR, PKCE S256, token exchange) and proxying authenticated requests through the tunnel to Hindsight
- **Cloud clients** (Claude apps, Claude Code, Codex) connect via MCP through the Worker

I built the Worker with `@cloudflare/workers-oauth-provider`, which handles most of the OAuth plumbing, and deployed it on Cloudflare with a KV namespace for token state. It serves a simple login page for the authorization flow and translates between the OAuth tokens that cloud clients send and the Bearer token that Hindsight expects. Once deployed, the Claude apps connected (just like authenticating any other MCP integration in their Customize section) and showed all Hindsight tools. Claude Code and Codex connect through the same Worker, since their remote MCP support uses the same OAuth flow. Hindsight's [multi-strategy retrieval architecture](/blog/2026/04/02/beam-sota) means every client gets high-quality recall regardless of how it connects.

## Getting them to store memories

Both OpenClaw and Claude Code's CLI already have native hook systems that Hindsight can latch onto. The [hindsight-openclaw plugin](/sdks/integrations/openclaw) auto-retains after each turn and auto-recalls before each response. The [hindsight-memory plugin for Claude Code](/sdks/integrations/claude-code) does the same thing using Claude Code's hook architecture: `SessionStart` for health checks, `UserPromptSubmit` for auto-recall, and `Stop` for auto-retain. No prompting needed. Memory just happens.

But for Claude outside of the Claude Code CLI, I can't benefit from those hooks directly. Instead, I use instructions. For Claude Code and the VS Code extension, a user-level `CLAUDE.md` file tells them how and when to use recall, retain, and reflect. For the Claude desktop and mobile apps and browser, the same instructions go into "user preferences" in the app settings.

In practice, this works well. The instructions tell the model to retain important context inline as conversations happen, and to do a final sweep when wrapping up in case anything was missed. The vast majority of sessions access Hindsight without me explicitly asking. On particularly important conversations, I'll double-check at the end.

Ideally, Claude's desktop and mobile apps would eventually expose the same kind of lifecycle hooks that Claude Code's CLI and OpenClaw already have. If the apps could fire events like `onConversationEnd` that an MCP server could subscribe to, memory capture wouldn't depend on prompt instructions at all. The building blocks exist. Hindsight already has hook-based plugins for both OpenClaw and Claude Code's CLI. The gap is just the app surfaces, which is where most people actually spend their time.

## What's next

I'm working with the Hindsight team to contribute the Worker as a reference OAuth proxy for self-hosters. The code is generic: swap the origin URL and credentials, deploy to your own Cloudflare account, and connect away.

The multi-client, single-bank pattern is where Hindsight has a real edge. Hindsight's architecture already supports sharing across all of them, with just a small gap to close on the OAuth front. One thing would further improve this pattern: MCP lifecycle hooks in the Claude apps. Claude Code's CLI already has them, and Hindsight's plugin uses them for automatic memory. Bringing the same capability to the desktop and mobile apps would close the last gap.

The easiest way to try this pattern is [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — no infrastructure, no OAuth proxy, connects directly to all cloud clients. Self-hosters can use the Worker reference once it's published.

- **Get started with Hindsight Cloud**: [ui.hindsight.vectorize.io/signup](https://ui.hindsight.vectorize.io/signup)
- **Claude Code integration**: [/sdks/integrations/claude-code](/sdks/integrations/claude-code)
- **OpenClaw integration**: [/sdks/integrations/openclaw](/sdks/integrations/openclaw)
- **Self-hosting quickstart**: [/developer/api/quickstart](/developer/api/quickstart)

If you've built something similar, or if any of this sparks ideas for how Hindsight could better support multi-client setups, I'd love to hear about it.
