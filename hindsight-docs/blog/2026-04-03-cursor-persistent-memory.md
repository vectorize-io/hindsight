---
title: "Giving Cursor a Long-Term Memory"
description: Cursor resets its memory every session. Learn how to add persistent cross-session memory using the Hindsight plugin for Cursor. Zero dependencies, automatic recall and retain.
authors: [DK09876]
date: 2026-04-03T12:00
tags: [cursor, integrations, memory, plugin, coding-agents]
image: /img/blog/cursor-persistent-memory.png
hide_table_of_contents: true
---

![Giving Cursor a Long-Term Memory](/img/blog/cursor-persistent-memory.png)

Cursor 3 introduced a plugin system with hooks, skills, and rules. It's a powerful architecture for extending what the agent can do. But one thing Cursor still doesn't have out of the box is persistent memory across sessions. Every new chat starts from scratch.

If you've spent three turns explaining your project's architecture, your team's naming conventions, or that you prefer functional patterns over classes, Cursor forgets all of it the moment the session ends.

The Hindsight plugin for Cursor fixes this. It hooks into the `beforeSubmitPrompt` event to recall relevant memories before every prompt, and the `stop` event to retain conversation transcripts after every task. No manual effort, no copy-pasting context, no dependencies to install.

<!-- truncate -->

## The Problem: Session-Scoped Memory

Cursor's built-in context system is designed around the current session. You get your codebase, your open files, and whatever you type. That's great for single-session tasks, but it falls apart for anything that spans multiple sessions:

- **Repeated explanations.** You tell Cursor your API uses snake_case, your frontend uses camelCase, and tests go in `__tests__/` directories. Next session? You explain it again.
- **Lost decisions.** You and Cursor agree on an architecture — event-driven with a message bus. Two days later, you start a new session and Cursor suggests REST endpoints.
- **No user model.** Cursor doesn't know that you're a senior engineer who doesn't need basic explanations, or that you prefer TypeScript over JavaScript, or that your team uses Vitest instead of Jest.

These aren't edge cases. They're the default experience for anyone who uses Cursor across more than a few sessions.

## How the Plugin Works

The Hindsight plugin uses two of Cursor's hook events to create a transparent memory loop:

### Auto-Recall (beforeSubmitPrompt)

Every time you send a prompt, the plugin fires before Cursor processes it:

1. Composes a query from your prompt (optionally including prior turns for context)
2. Calls Hindsight's recall API with multi-strategy retrieval (semantic search, BM25, graph traversal, temporal filtering)
3. Formats the results into a `<hindsight_memories>` block
4. Injects it as `additionalContext` — the agent sees it, you don't

The memories appear in the agent's context window but not in your chat transcript. Cursor uses them to inform its response without cluttering your conversation.

### Auto-Retain (stop)

After Cursor completes a task, the plugin fires again:

1. Reads the conversation transcript from Cursor's JSONL file
2. Strips any injected memory tags (preventing feedback loops)
3. Applies chunked or full-session retention based on your config
4. POSTs the transcript to Hindsight for fact extraction and storage

Hindsight extracts structured facts — preferences, decisions, project details — and builds a knowledge graph over time. The next session, those facts are available for recall.

### On-Demand Recall

The plugin also registers a `hindsight-recall` skill. If you want to explicitly search your memory mid-conversation, use `/hindsight-recall` with a query. This is useful when auto-recall didn't surface what you need, or when you want to search for something specific.

## Setup in 60 Seconds

**Step 1:** Install the plugin into your project:

```bash
cp -r hindsight-integrations/cursor /path/to/your-project/.cursor-plugin/hindsight-memory
```

**Step 2:** Set an LLM provider (for the local Hindsight daemon):

```bash
export OPENAI_API_KEY="sk-your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

Or connect to an existing Hindsight server:

```bash
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-server.com"}' > ~/.hindsight/cursor.json
```

**Step 3:** Open Cursor. The plugin activates automatically.

That's it. No pip install, no npm install, no build step. The plugin is pure Python stdlib.

## Per-Project Memory Isolation

By default, all Cursor sessions share a single memory bank (`cursor`). For teams or multi-project workflows, you can isolate memory per project:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

This derives the bank ID from the working directory. Your React project and your Go API get separate memories that don't cross-contaminate.

Available granularity fields: `agent`, `project`, `session`, `channel`, `user`. Combine them to match your workflow.

## Alternative: MCP Integration

If you'd rather use Cursor's native MCP support instead of the plugin system, Hindsight works there too. Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8888/mcp/"
    }
  }
}
```

This gives you explicit `retain`, `recall`, and `reflect` tools. The plugin approach is more automatic (hooks fire without you asking); the MCP approach gives you more control.

## What's Next

The Cursor plugin is open source and available in the [Hindsight repository](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor). Full configuration reference is in the [integration docs](/sdks/integrations/cursor).

Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or self-hosted via `hindsight-embed`.
