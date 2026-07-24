
# AgentOS

The `@vectorize-io/hindsight-agentos` package gives [AgentOS](https://github.com/framerslab/agentos) agents long-term memory backed by [Hindsight](https://hindsight.vectorize.io).

`createHindsightMemory` returns an AgentOS `AgentMemoryProvider`. Attach it to an agent via `memoryProvider` and AgentOS auto-wires it on every call path (`generate`, `stream`, and `session.send` / `session.stream`):

- **`getContext`** runs *before* each model call — it recalls relevant memories from Hindsight and injects them into the system prompt.
- **`observe`** runs *after* each turn — it retains the exchange to Hindsight, where entity extraction and consolidation into world/experience facts and mental models happen server-side.

Both are enabled by default and fail safe — a memory-service hiccup never blocks your agent from responding, and retains are fire-and-forget so they never add turn latency.

## Installation

```bash
npm install @vectorize-io/hindsight-agentos @vectorize-io/hindsight-client
```

This package targets `@framers/agentos` `>=0.9.0` (declared as a peer dependency).

## Usage

```ts
import { agent } from "@framers/agentos";
import { createHindsightMemory } from "@vectorize-io/hindsight-agentos";
import { Hindsight } from "@vectorize-io/hindsight-client";

const memory = createHindsightMemory({
  client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }),
  bank: "ada",
  recall: { budget: "high", includeEntities: true },
  retain: { tags: ["source:agentos"] },
});

const ada = agent({ name: "Ada", memoryProvider: memory });

await ada.generate("What theme do I prefer?");
```

AgentOS memory-provider hooks receive only turn text (no per-message routing
context), so a provider instance maps to exactly one memory **bank**. Give each
agent (or user) its own bank for isolation — it defaults to `"default"`.

## Configuration

```ts
createHindsightMemory({
  client,

  // Which memory bank this agent reads/writes. Defaults to "default".
  bank: "ada",

  recall: {
    enabled: true,            // set false to disable recall
    budget: "mid",            // "low" | "mid" | "high" — latency vs. depth
    types: ["world", "experience"], // restrict to fact types
    maxTokens: 1000,          // cap recalled tokens (else uses AgentOS tokenBudget)
    includeEntities: false,   // include entity observations
    labelTypes: false,        // prefix each memory with its fact kind, e.g. [world]
    heading: "# Relevant long-term memories", // context-block heading
  },

  retain: {
    enabled: true,            // set false to disable retain
    async: true,              // fire-and-forget; never adds turn latency
    tags: ["source:agentos"], // tags on every retained memory
    metadata: { env: "prod" },
    includeAgentMessages: false, // also store the agent's own replies
  },
});
```

### Using only recall or only retain

Disable either side with `recall.enabled: false` or `retain.enabled: false`.

## How it works

| Hook | AgentOS seam | When it runs | What it does |
| --- | --- | --- | --- |
| `getContext` | Before generation | Before each model call | Calls Hindsight `recall` with the turn text and returns a context block AgentOS injects into the prompt |
| `observe` | After generation | After each turn | Calls Hindsight `retain` to persist the user turn (and optionally the agent's reply) |
