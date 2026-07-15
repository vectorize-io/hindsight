
# Omnigent

Persistent long-term memory for every harness in [Omnigent](https://github.com/omnigent-ai/omnigent) — the meta-harness that wraps and coordinates multiple AI agents (Claude Code, Codex, Cursor, OpenCode, Hermes, Pi, and more).

Omnigent ships three built-in memory tools — `hindsight_recall`, `hindsight_retain`, and `hindsight_reflect` — that Omnigent intercepts and executes locally using `hindsight-client`. Because the tools live in the Omnigent runner rather than in each harness, every wrapped agent gets memory regardless of whether it has native Hindsight support.

## Installation

```bash
pip install "omnigent[memory]"
export HINDSIGHT_API_KEY=hsk_...
```

Get an API key from [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), or point at a self-hosted server with `HINDSIGHT_API_URL`.

## Setup

Add the three tools to your agent's YAML spec under `tools.builtins`:

```yaml
name: my-agent

tools:
  builtins:
    - name: hindsight_recall
      api_key: ${HINDSIGHT_API_KEY}
      bank_id: my-agent-memory      # optional; defaults to agent_id
      budget: mid                   # low / mid / high
      max_tokens: 4096

    - name: hindsight_retain
      api_key: ${HINDSIGHT_API_KEY}
      bank_id: my-agent-memory

    - name: hindsight_reflect
      api_key: ${HINDSIGHT_API_KEY}
      bank_id: my-agent-memory
```

Restart your agent — the three tools will appear in its tool list.

## How It Works

Omnigent intercepts `hindsight_*` calls at the runner level before they reach the wrapped harness, executes them locally via `hindsight-client`, and returns the result. This means:

- **Wrapped harnesses** (Claude Code, Codex, Cursor, Pi…) never need to handle the call; the runner does it.
- **Native harnesses** (when running agents in native mode) have the tools relayed to them as well.
- **One setup for all of them.** Many of these harnesses have their own native Hindsight integration, but through Omnigent you configure memory once and it applies to every harness you orchestrate, including custom ones with no native option.

The agent calls the tools **explicitly**; there are no automatic lifecycle hooks. Include instructions in your agent's system prompt:

```
- At the start of each task, call hindsight_recall with the user's request
  to load relevant decisions, preferences, and project context.
- When the user gives you a durable fact (a convention, a decision, a
  preference), call hindsight_retain to store it.
- Call hindsight_reflect to synthesize what you know about a topic across sessions.
```

## Bank Scoping

Omnigent resolves the memory bank in this order:

1. The explicit `bank_id` in your YAML config — most predictable, recommended.
2. The agent's `agent_id` — one bank per agent, shared across conversations.
3. The `conversation_id` — one bank per conversation; lost when the conversation ends.

To share memory between two agents, point them at the same `bank_id`.

## Configuration Reference

| Field | Description | Default |
|---|---|---|
| `api_key` | Hindsight API key | _(required for Cloud)_ |
| `api_url` | Hindsight API base URL | `https://api.hindsight.vectorize.io` |
| `bank_id` | Memory bank name | `agent_id` or `conversation_id` |
| `budget` | Recall token budget (`low` / `mid` / `high`) | `mid` |
| `max_tokens` | Maximum tokens returned by recall | `4096` |
| `tags` | CSV tags applied to retained memories | _(none)_ |
| `recall_tags` | CSV tags to filter recalled memories | _(none)_ |
| `recall_tags_match` | Tag match mode (`any` / `all` / `any_strict` / `all_strict`) | `any` |

## Self-Hosted

Point at your own server with `api_url`; no token needed for an open server:

```yaml
- name: hindsight_recall
  api_url: http://localhost:8888
  bank_id: local-memory
```

## Working Example

Omnigent ships a complete example at [`examples/remy/config.yaml`](https://github.com/omnigent-ai/omnigent/blob/main/examples/remy/config.yaml) — a conversational assistant with all three tools wired up and instructions on when to call each one:

```bash
HINDSIGHT_API_KEY=hsk_... omnigent run examples/remy
```

After a few conversations, ask it something it learned in a previous session.

## Which Harnesses Benefit

| Harness | Native Hindsight integration | Via Omnigent |
|---|---|---|
| Claude Code | Yes (official) | Yes |
| Cursor | Yes (official) | Yes |
| Codex | Yes (official) | Yes |
| OpenCode | Yes (official) | Yes |
| Pi | Yes (community, via epimetheus) | Yes |
| Custom harness | Usually none | Yes |

Most of these harnesses already have a native Hindsight integration, ideal when you run that tool standalone. Omnigent's value is one central memory setup that spans every harness you orchestrate at once, plus coverage for custom harnesses with no native option.

## Further Reading

- One memory for every AI tool: point multiple harnesses at the same bank.
- Inside retain(): what happens when the agent calls `hindsight_retain`.
- [Omnigent on GitHub](https://github.com/omnigent-ai/omnigent): the meta-harness source and examples.
