# hindsight-continue

[Continue.dev](https://continue.dev) integration for [Hindsight](https://github.com/vectorize-io/hindsight) — persistent long-term memory for your coding assistant.

Continue has no plugin hook that runs before a message is sent, but it does
support two native extension points that this integration uses:

1. **HTTP context provider** — type `@hindsight <query>` (or a bare `@hindsight`)
   in chat and relevant memory is recalled and injected into the model's context
   **at query time**. This package ships a tiny adapter server that implements
   Continue's HTTP context-provider contract on top of Hindsight recall.
2. **MCP server + rules** (optional) — wire the Hindsight MCP server into
   Continue's agent mode for `retain`/`recall`/`reflect` tools, with a rules file
   that tells the agent to recall automatically.

## Prerequisites

- A running Hindsight instance ([self-hosted via Docker](https://github.com/vectorize-io/hindsight#quick-start) or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup))
- Continue (VS Code or JetBrains extension)
- Python 3.10+

## Installation

```bash
pip install hindsight-continue
```

## Quick start (HTTP context provider)

**1. Run the adapter**, pointing it at your Hindsight instance and bank:

```bash
export HINDSIGHT_API_KEY=hsk_...            # omit for a local self-hosted server
export HINDSIGHT_API_URL=http://localhost:8888   # defaults to Hindsight Cloud
export HINDSIGHT_CONTINUE_BANK_ID=my-project

hindsight-continue            # serves on 127.0.0.1:8123
```

**2. Register it** in Continue's `config.yaml` (`~/.continue/config.yaml` or a
workspace `.continue` block):

```yaml
context:
  - provider: http
    params:
      url: "http://127.0.0.1:8123/"
      title: hindsight
      displayTitle: Hindsight
      description: Recall long-term memory from Hindsight
```

**3. Use it.** In Continue chat, type `@hindsight` (optionally followed by what
you want to recall) and the matching memories are added to the model's context.

The adapter receives Continue's request (`{query, fullInput, options, workspacePath}`),
recalls from Hindsight, and returns context items shaped `{name, description, content}` —
exactly what Continue expects.

### Per-request bank override

The configured `HINDSIGHT_CONTINUE_BANK_ID` is the default. A request may target
a different bank by sending `options.bankId`:

```yaml
context:
  - provider: http
    params:
      url: "http://127.0.0.1:8123/"
      options:
        bankId: another-bank
```

## Optional: automatic recall via MCP + rules

For hands-off recall/retain in **agent mode**, point Continue at the Hindsight
MCP server and add a rule telling the agent to call it. Example assets are in
[`examples/.continue/`](./examples/.continue):

- [`examples/.continue/mcpServers/hindsight.yaml`](./examples/.continue/mcpServers/hindsight.yaml) — the MCP server block
- [`examples/.continue/rules/hindsight.md`](./examples/.continue/rules/hindsight.md) — "always recall first" rule

## Configuration

| Env var | Description | Default |
| --- | --- | --- |
| `HINDSIGHT_API_KEY` | Hindsight API key | _(none; required for Cloud)_ |
| `HINDSIGHT_API_URL` | Hindsight API URL | `https://api.hindsight.vectorize.io` |
| `HINDSIGHT_CONTINUE_BANK_ID` | Default memory bank to recall against | _(none)_ |
| `HINDSIGHT_CONTINUE_HOST` | Adapter bind host | `127.0.0.1` |
| `HINDSIGHT_CONTINUE_PORT` | Adapter listen port | `8123` |

These can also be set programmatically via `configure(...)`. See `hindsight-continue --help`
for CLI flags.

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated live E2E (needs a Hindsight server)
```

## License

MIT
