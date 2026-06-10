---
title: "Flowise Persistent Memory: Drop-In Tool Nodes for Any Chatflow"
authors: [benfrank241]
slug: "2026/06/10/flowise-persistent-memory"
date: 2026-06-10T12:00
tags: [flowise, memory, persistent-memory, hindsight, agents, langchain, tutorial]
description: "Add persistent long-term memory to any Flowise chatflow or agent with Hindsight. Three Tool nodes (Retain, Recall, Reflect) drop straight into a Conversational Agent and share one Hindsight API credential."
image: /img/blog/flowise-persistent-memory.png
hide_table_of_contents: true
---

![Flowise Persistent Memory with Hindsight](/img/blog/flowise-persistent-memory.png)

[Flowise](https://flowiseai.com) is the visual chatflow builder for LangChain. You drag chat models, tools, and agents onto a canvas, connect them, and ship a working assistant without writing the orchestration code. What you can't ship by default is an agent that remembers anything from one session to the next; Flowise chatflows are stateless across runs.

This post is a walkthrough of the new Hindsight integration for Flowise. It adds three Tool nodes (Retain, Recall, Reflect) that any chatflow can attach to an agent, sharing one credential and using the same memory bank you'd use from any other Hindsight integration.

## TL;DR

<!-- truncate -->

- Three new Tool nodes appear in Flowise under **Tools**: **Hindsight Retain**, **Hindsight Recall**, **Hindsight Reflect**.
- All three return a LangChain `DynamicStructuredTool`, so they slot into any Flowise Conversational Agent, Tool Agent, or custom flow that accepts tools.
- One shared **Hindsight API** credential covers all three nodes; configure once, reuse everywhere.
- **Hindsight Cloud is the recommended path.** No infrastructure to run; the credential's default URL points there.
- The memory bank is a regular Hindsight bank: the same one you'd write to from Claude Code, Cline, Codex, or a direct API call. Memory crosses tools.

## Why Flowise Needs Persistent Memory

Flowise gives non-developers (and developers in a hurry) a real path to a working agent without writing orchestration. Drag a `ChatOpenAI`, drop in a `Conversational Agent`, attach a `Calculator` tool, hit deploy. It works.

What it doesn't do is remember the user. The agent's working memory is the current conversation. Close the chat window and the next session starts from zero. The `BufferMemory` and `ConversationSummaryMemory` nodes in Flowise help within a session, but they don't survive across them and they don't extract or distill anything from the transcript.

For most production use cases this is the limit you hit first. A support agent that re-asks for the account number every visit. A coding assistant that re-relearns the project's conventions every reload. An onboarding flow that forgets where the user left off. You either build the memory layer yourself (Postgres, embeddings, retrieval logic, deduplication) or you don't ship the feature.

Hindsight's Flowise nodes give you the second path: drop in a tool, point it at a memory bank, done.

## How It Works

The integration adds three nodes under Flowise's **Tools** category:

| Node | What it does |
| --- | --- |
| **Hindsight Retain** | Store free-text content in a memory bank. Hindsight extracts structured facts asynchronously after the call returns. |
| **Hindsight Recall** | Search a bank for memories relevant to a natural-language query. Returns ranked results. |
| **Hindsight Reflect** | Get an LLM-synthesized answer over the bank ("what do we know about X?"). |

Each node, when initialized, returns a LangChain `DynamicStructuredTool` (from `@langchain/core/tools`) with a typed Zod schema. That's important: it means an agent can call the tool with the right shape, and Flowise can wire it into any agent socket that accepts a LangChain tool: Conversational Agent, Tool Agent, OpenAI Function Agent, all of them.

The tool names the LLM sees are `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`. The agent decides when to call them based on the tool's description and the user's request.

## The Three Tools In Detail

### Hindsight Retain

Stores a piece of content into a memory bank. Hindsight's extractor pulls structured facts out of it after the call returns, so the tool itself completes quickly.

**Tool input schema** (the agent fills these in):

```ts
{
  bankId: string,            // which bank to write to
  content: string,           // free-text to store
  tags?: string[],           // optional tags for filtering on recall
}
```

**Node configuration** (set in the Flowise UI):

| Field | Description |
| --- | --- |
| Default Bank ID | The bank to retain into when the agent doesn't pass one. Banks are created on first use. |

If neither the agent nor the node provides a `bankId`, the tool returns a clear error string rather than silently writing to the wrong place.

### Hindsight Recall

Searches a memory bank for content relevant to a query.

**Tool input schema:**

```ts
{
  bankId: string,
  query: string,                    // natural language
  budget?: "low" | "mid" | "high",  // search depth (default: mid)
  maxTokens?: number,               // cap on total recall size
  tags?: string[],                  // filter by tags
}
```

**Node configuration:**

| Field | Description |
| --- | --- |
| Default Bank ID | Searched when the agent doesn't pass one. |
| Default Budget | `low` / `mid` / `high`. Default `mid`. Higher values search more memories at higher cost. |

The result is a ranked set of memories serialized as JSON, which the agent reads back as context for its next response.

### Hindsight Reflect

Asks Hindsight to synthesize an answer from the bank using an LLM, rather than returning raw memories.

**Tool input schema:**

```ts
{
  bankId: string,
  query: string,
  budget?: "low" | "mid" | "high",  // default: mid
}
```

**Node configuration:**

| Field | Description |
| --- | --- |
| Default Bank ID | Reflected on when the agent doesn't pass one. |
| Default Budget | `low` / `mid` / `high`. Default `mid`. |

Reflect is the right tool when the agent needs a summary rather than raw facts. "What does this user prefer?" "What architectural decisions did we make on this project?" The bank does the legwork; the agent gets a synthesized paragraph instead of having to read and combine raw memories itself.

## Setup

You need a Hindsight account and an API key. The fastest path is **Hindsight Cloud**.

1. **Sign up** at [hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup). Free tier is enough to try the integration end to end.
2. **Create an API key** from the dashboard. It looks like `hsk_...`.
3. **In Flowise**, open **Credentials** → **Add Credential** → **Hindsight API**:
   - **API URL**: defaults to `https://api.hindsight.vectorize.io`. Leave as-is for Cloud; change for self-hosted (e.g. `http://localhost:8888`).
   - **API Key**: paste your `hsk_...` key. Leave blank if you're hitting an unauthenticated self-hosted instance.
4. Save the credential. All three tool nodes pick it up from the same `hindsightApi` name.

That's the whole connection step. From here the credential is reusable across every chatflow you build.

## Installation

A note on how the nodes get into your Flowise instance. Flowise distributes node packages from inside its own monorepo (`FlowiseAI/Flowise` under `packages/components/`), not from npm. The shortest path today is to run a local Flowise build with the Hindsight source files copied in:

```bash
git clone https://github.com/FlowiseAI/Flowise.git
cd Flowise

# Copy the three Hindsight tool nodes
cp -r /path/to/hindsight/hindsight-integrations/flowise/nodes/tools/Hindsight* \
  packages/components/nodes/tools/

# Copy the shared credential class
cp /path/to/hindsight/hindsight-integrations/flowise/credentials/HindsightApi.credential.ts \
  packages/components/credentials/

# Add the Hindsight client to the components package
cd packages/components && pnpm add @vectorize-io/hindsight-client
cd ../.. && pnpm install && pnpm build
pnpm start  # opens http://localhost:3000
```

Once Flowise starts, the three nodes show up in the **Tools** category and the **Hindsight API** credential type is available under **Credentials**.

## Example Chatflow: A Conversational Support Agent

The minimal flow that exercises all three nodes:

```
ChatOpenAI ─┐
            │
            ▼
  Conversational Agent ──► (output to chat)
    ├── tools: Hindsight Recall   (Default Bank ID: user-{{sessionId}})
    ├── tools: Hindsight Retain   (Default Bank ID: user-{{sessionId}})
    └── tools: Hindsight Reflect  (Default Bank ID: user-{{sessionId}})
```

1. Drop a **ChatOpenAI** (or any chat model) on the canvas.
2. Drop a **Conversational Agent** and connect the chat model.
3. Drop the three Hindsight tool nodes. Attach all three to the agent's `tools` input. Pick the **Hindsight API** credential on each one.
4. Set **Default Bank ID** to something session-scoped, e.g. `user-{{sessionId}}`. The agent can still pass a different `bankId` per call, but the node provides this fallback.
5. Deploy and send a message.

With this setup the agent learns, over the course of a few sessions, to:

- Call **Recall** before answering to ground the response in past conversations
- Call **Retain** after meaningful exchanges (decisions made, preferences revealed, problems solved)
- Call **Reflect** when the user asks open-ended questions like "what have we talked about?"

It doesn't take prompt engineering to get this behavior; the tool descriptions are explicit about when each one should fire, and the underlying agent prompts handle the routing. If you want tighter control, you can set the `Conversational Agent`'s system message to nudge the order explicitly ("Always check recall before answering").

## Tradeoffs

**Retain is asynchronous.** The retain call returns when the content lands in the bank, not when the extractor finishes. Facts become recallable within seconds, but a recall fired immediately after a retain may not yet see what you just stored. For chat-style flows this is fine; the next user turn happens after extraction completes. For automated scripts that retain-then-recall in the same run, add a short delay or use Reflect (which doesn't depend on extraction having finished).

**One bank ID per flow turn is the right default.** Multiple tools can use different banks within a single agent run, but most production flows do better with one bank per user (or per project, or per tenant) consistently. Treat `Default Bank ID` as the routing key for the conversation.

**Cloud vs self-hosted.** Cloud removes a lot of moving parts: extraction runs server-side, you don't have to thread an LLM key into the Flowise environment, and memory follows the bank rather than the machine. Self-hosting works the same way once you point the credential's API URL at your instance, but you become the one keeping `hindsight-api` running.

## Recap

| | Flowise default | With Hindsight Tool nodes |
| --- | --- | --- |
| Cross-session memory | None | Persistent, per bank |
| Memory setup | Buffer/summary nodes only | One credential + three tool nodes |
| Fact extraction | None | Async on every retain |
| Cross-bank synthesis | Not available | `Reflect` returns LLM-synthesized answers |
| Bank routing | n/a | Default per node, overridable per call |
| Cross-tool sharing | n/a | Same bank readable from Claude Code, Cline, MCP, API |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Flowise + Hindsight](/sdks/integrations/flowise)
- **Source:** [`vectorize-io/hindsight/hindsight-integrations/flowise`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/flowise)
- **Hindsight API reference:** [API quickstart](/developer/api/quickstart)
