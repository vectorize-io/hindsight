# @vectorize-io/hindsight-eliza

Long-term memory for [elizaOS](https://github.com/elizaOS/eliza) agents, backed by [Hindsight](https://hindsight.vectorize.io).

The plugin registers two components on your agent:

- **`HINDSIGHT_MEMORY` provider** — recalls relevant memories and injects them into the prompt before each model call.
- **`HINDSIGHT_RETAIN` evaluator** — retains conversation messages to Hindsight after each turn.

Both are enabled by default, layer on top of elizaOS's existing memory, and fail safe — a Hindsight outage never blocks the agent from responding.

## Installation

```bash
npm install @vectorize-io/hindsight-eliza @vectorize-io/hindsight-client
```

Requires `@elizaos/core` `^1.7.2` (peer dependency).

## Usage

```ts
import { createHindsightPlugin } from "@vectorize-io/hindsight-eliza";
import { Hindsight } from "@vectorize-io/hindsight-client";

const hindsightPlugin = createHindsightPlugin({
  client: new Hindsight({ apiKey: process.env.HINDSIGHT_API_KEY }),
  recall: { budget: "high", includeEntities: true },
  retain: { tags: ["source:eliza"] },
});

export const character = {
  name: "Ada",
  plugins: [hindsightPlugin],
};
```

By default memories are stored per user (keyed by the message `entityId`). Pass a
`bank` string or `(message) => string` function to control this.

## Options

| Option                        | Description                                 | Default                         |
| ----------------------------- | ------------------------------------------- | ------------------------------- |
| `client`                      | A Hindsight client instance                 | required                        |
| `bank`                        | Fixed bank string, or `(message) => string` | `message.entityId`              |
| `recall.enabled`              | Enable the recall provider                  | `true`                          |
| `recall.budget`               | `"low" \| "mid" \| "high"`                  | `"mid"`                         |
| `recall.types`                | Restrict to fact types                      | all                             |
| `recall.maxTokens`            | Cap recalled tokens                         | API default                     |
| `recall.includeEntities`      | Include entity observations                 | `false`                         |
| `recall.heading`              | Heading above recalled memories             | `# Relevant long-term memories` |
| `retain.enabled`              | Enable the retain evaluator                 | `true`                          |
| `retain.async`                | Fire-and-forget (no turn latency)           | `true`                          |
| `retain.tags`                 | Tags on every retained memory               | —                               |
| `retain.metadata`             | Metadata on every retained memory           | —                               |
| `retain.includeAgentMessages` | Also store the agent's replies              | `false`                         |

You can also use `createHindsightProvider` and `createHindsightEvaluator`
directly if you want to assemble your own plugin.

## Development

```bash
npm install
npm test
npm run build
```

## License

MIT
