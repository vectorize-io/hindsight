# Hindsight for Flowise

Persistent long-term memory for [Flowise](https://flowiseai.com) chatflows and agents via [Hindsight](https://hindsight.vectorize.io).

Three Tool nodes — **Hindsight Retain**, **Hindsight Recall**, **Hindsight Reflect** — drop into any Flowise chatflow or agent flow alongside your other Tools.

- **Author**: Vectorize, Inc.
- **Repository**: <https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/flowise>
- **Contact**: support@vectorize.io · [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)

## Status

This package is the source-of-truth copy of the Hindsight Flowise nodes. Flowise distributes nodes only inside its main monorepo, so the user-facing distribution is an upstream PR to [`FlowiseAI/Flowise`](https://github.com/FlowiseAI/Flowise) under `packages/components/`.

## Installation (local Flowise dev fork)

While the upstream PR is in flight, you can use the nodes by copying them into a Flowise checkout:

```bash
git clone https://github.com/FlowiseAI/Flowise.git
cd Flowise
# Copy the three tool nodes
cp -r /path/to/hindsight/hindsight-integrations/flowise/nodes/tools/Hindsight* \
   packages/components/nodes/tools/
# Copy the credential class
cp /path/to/hindsight/hindsight-integrations/flowise/credentials/HindsightApi.credential.ts \
   packages/components/credentials/
# Add the client dep
cd packages/components && pnpm add @vectorize-io/hindsight-client
cd ../.. && pnpm install && pnpm build
pnpm start  # opens http://localhost:3000
```

## Setup

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip the self-hosting setup entirely.

1. **Sign up** at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier) or [self-host](https://hindsight.vectorize.io/developer/installation).
2. **Get an API key** from the Hindsight dashboard.
3. **In Flowise**, create a new credential of type **Hindsight API**:
   - **API URL**: `https://api.hindsight.vectorize.io` (Cloud) or your self-hosted URL
   - **API Key**: your `hsk_...` key (optional for self-hosted unauthenticated instances)

## Tools

### Hindsight Retain

Stores free-text content in a memory bank. Hindsight extracts facts asynchronously after the call returns.

| Field           | Description                                                |
| --------------- | ---------------------------------------------------------- |
| Default Bank ID | Memory bank to retain into when the agent doesn't pass one |

Tool input schema (the agent passes these): `bankId`, `content`, optional `tags`.

### Hindsight Recall

Searches a bank for memories relevant to a query. Returns ranked results.

| Field           | Description                                           |
| --------------- | ----------------------------------------------------- |
| Default Bank ID | Memory bank to search when the agent doesn't pass one |
| Default Budget  | `low` / `mid` / `high`                                |

Tool input schema: `bankId`, `query`, optional `budget`, `maxTokens`, `tags`.

### Hindsight Reflect

Returns an LLM-synthesized answer over a bank.

| Field           | Description                                               |
| --------------- | --------------------------------------------------------- |
| Default Bank ID | Memory bank to reflect on when the agent doesn't pass one |
| Default Budget  | `low` / `mid` / `high`                                    |

Tool input schema: `bankId`, `query`, optional `budget`.

## Example chatflow

A typical conversational agent recipe:

1. **ChatOpenAI** (or any chat LLM)
2. **Conversational Agent** with three tools attached: Hindsight Retain + Hindsight Recall + Hindsight Reflect
3. Set a default Bank ID like `user-${sessionId}` on each tool
4. The agent learns to call Recall before answering and Retain after meaningful exchanges

## Source

- GitHub: [`vectorize-io/hindsight`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/flowise)
- Docs: [hindsight.vectorize.io/sdks/integrations/flowise](https://hindsight.vectorize.io/sdks/integrations/flowise)
