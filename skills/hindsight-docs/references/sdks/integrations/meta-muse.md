
# Meta Muse

Bring the memory from your other AI tools into [Meta Muse](https://muse.ai), Meta's personal AI agent. Muse connects to your [Hindsight](https://vectorize.io/hindsight) memory banks over MCP. What you told Claude, ChatGPT, Claude Code, Hermes or OpenClaw is available in Muse, and what you tell Muse is available to them.

Muse gets all of your banks, not one. It keeps its own `muse` bank for what you and it do together, and reads your other banks when a question reaches into them.

> **💡 Hindsight Cloud (recommended)**
>
[Sign up free](https://ui.hindsight.vectorize.io/signup). Hindsight Cloud already supports the OAuth sign-in Muse uses, so there is nothing to run or configure.
## How It Works

Muse runs in Meta's cloud and has no plugin host, so Hindsight can't hook into its prompts or transcripts. Muse connects to Hindsight's remote MCP server as a custom connector and calls the memory tools itself:

| Tool | When Muse calls it |
|---|---|
| `recall` | Before tasks about people, projects, plans or preferences |
| `retain` | When you state a fact, decision or preference, or finish a task worth remembering |
| `reflect` | For questions about your history or habits |
| `get_mental_model` | Reads an "About me" profile at the start of a conversation |
| `list_banks`, `create_bank` | Learns which banks exist and creates its own `muse` bank on first connect |

Because Muse has no hooks, the connect prompt also asks Muse to create a nightly scheduled task that retains a summary of the day's conversations.

## Setup

1. Sign up at [vectorize.io/hindsight](https://vectorize.io/hindsight/).
2. Open Muse and paste the [connect prompt](https://github.com/vectorize-io/hindsight/blob/main/hindsight-integrations/meta-muse/connect-prompt.md).
3. Muse shows a **Connect** button for Hindsight. Click it and approve access on Hindsight's sign-in page.
4. Muse lists the banks it can see, creates its own `muse` bank plus the "About me" mental model and the nightly task, and tells you three things it found about you.

The prompt uses the root URL, `https://api.hindsight.vectorize.io/mcp` (see the [MCP server reference](../../developer/mcp-server.md) for the endpoint, its tools and its auth options), which reaches every bank in your account. That is the point: Muse writes to its own bank and reads the others when a question calls for it. Because the root URL also exposes bank management, the prompt forbids `delete_bank`, `clear_memories` and `invalidate_memory`. To hand Muse exactly one bank instead, connect it to `/mcp/<bank>/`: that endpoint takes the same OAuth sign-in and drops the `bank_id` argument from every tool.

## Self-Hosted Hindsight

Muse needs a public HTTPS endpoint with OAuth 2.1 discovery and dynamic client registration. Put the [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy) in front of your instance, then check the endpoint with the preflight tool from the [integration directory](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/meta-muse):

```bash
uvx --from ./hindsight-integrations/meta-muse hindsight-muse-preflight https://memory.example.com/mcp
```

It replays what Muse does: the unauthenticated 401 challenge, the OAuth metadata (registration endpoint, PKCE S256), and, with `HINDSIGHT_API_KEY` set, MCP `initialize` and `tools/list`.

## Verifying Setup

In Muse:

- "Which memory banks can you see?" → `list_banks`
- "What was I working on last week?" → `recall` across your work banks
- "Remember that I prefer window seats." → `retain` into `muse`
- "How do I usually plan trips?" → `reflect`

What Muse retains then shows up in your other tools connected to the same account.
