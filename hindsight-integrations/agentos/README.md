# @vectorize-io/hindsight-agentos

Long-term memory for [AgentOS](https://github.com/framerslab/agentos) agents, backed by [Hindsight](https://hindsight.vectorize.io).

`createHindsightMemory` returns an AgentOS [`AgentMemoryProvider`](https://agentos.sh). Attach it to an agent via `memoryProvider` and AgentOS auto-wires it on every call path (`generate`, `stream`, and `session.send` / `session.stream`):

- **`getContext`** runs _before_ each model call — it recalls relevant memories from Hindsight and injects them into the system prompt.
- **`observe`** runs _after_ each turn — it retains the exchange to Hindsight, where entity extraction and consolidation into world/experience facts and mental models happen server-side.

Both sides are enabled by default and fail safe — a Hindsight outage never blocks the agent from responding, and retains are fire-and-forget so they never add turn latency.

## Installation

```bash
npm install @vectorize-io/hindsight-agentos @vectorize-io/hindsight-client
```

Requires `@framers/agentos` `>=0.9.0` (peer dependency).

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

## Options

| Option                        | Description                                           | Default                         |
| ----------------------------- | ----------------------------------------------------- | ------------------------------- |
| `client`                      | A Hindsight client instance                           | required                        |
| `bank`                        | Memory bank this agent reads/writes                   | `"default"`                     |
| `recall.enabled`              | Enable memory recall / context injection              | `true`                          |
| `recall.budget`               | `"low" \| "mid" \| "high"` — latency vs. depth        | `"mid"`                         |
| `recall.types`                | Restrict to fact types                                | all                             |
| `recall.maxTokens`            | Cap recalled tokens (else uses AgentOS `tokenBudget`) | AgentOS budget                  |
| `recall.includeEntities`      | Include entity observations                           | `false`                         |
| `recall.labelTypes`           | Prefix each memory with its fact kind, e.g. `[world]` | `false`                         |
| `recall.heading`              | Heading above recalled memories                       | `# Relevant long-term memories` |
| `retain.enabled`              | Enable retaining turns                                | `true`                          |
| `retain.async`                | Fire-and-forget (no turn latency)                     | `true`                          |
| `retain.tags`                 | Tags on every retained memory                         | —                               |
| `retain.metadata`             | Metadata on every retained memory                     | —                               |
| `retain.includeAgentMessages` | Also store the agent's replies                        | `false`                         |

### Using only recall or only retain

Disable either side with `recall.enabled: false` or `retain.enabled: false`.

## How it works

| Hook         | AgentOS seam                             | When it runs           | What it does                                                                         |
| ------------ | ---------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------ |
| `getContext` | Before generation (`onBeforeGeneration`) | Before each model call | Calls Hindsight `recall` and returns a context block AgentOS injects into the prompt |
| `observe`    | After generation (`onAfterGeneration`)   | After each turn        | Calls Hindsight `retain` to persist the user turn (and optionally the agent's reply) |

## Development

```bash
npm install
npm test
npm run build
```

## License

MIT
