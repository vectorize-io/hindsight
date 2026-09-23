# Hindsight × Meta Muse

Bring the memory from your other AI tools into [Meta Muse](https://muse.ai), Meta's
personal AI agent. Muse connects to your Hindsight memory banks over MCP, so what you
told Claude, ChatGPT, Claude Code, Hermes or OpenClaw is available in Muse, and what
you tell Muse is available to them.

Muse gets all of your banks, not one. It keeps its own `muse` bank for what you and it
do together, and reads your other banks when a question reaches into them.

> **Recommended:** Use [Hindsight Cloud](https://vectorize.io/hindsight/). It already
> supports the OAuth sign-in Muse uses, so there is nothing to run or configure.

## How it works

Muse runs in Meta's cloud and has no plugin host, so Hindsight can't hook into its
prompts or transcripts. Muse connects to Hindsight's remote MCP server as a custom
connector and calls the memory tools itself:

| Tool | When Muse calls it |
|---|---|
| `recall` | Before tasks about people, projects, plans or preferences |
| `retain` | When you state a fact, decision or preference, or finish a task worth remembering |
| `reflect` | For questions about your history or habits |
| `get_mental_model` | Reads an "About me" profile at the start of a conversation |
| `list_banks`, `create_bank` | Learns which banks exist and creates its own `muse` bank on first connect |

Because Muse has no hooks, the connect prompt also asks Muse to create a nightly
scheduled task that retains a summary of the day's conversations.

```
Muse agent (Meta cloud)
        |
        | HTTPS + MCP (Streamable HTTP), OAuth 2.1
        v
Hindsight Cloud  api.hindsight.vectorize.io/mcp
        |
        +-- muse         <- what Muse learns about you
        +-- work         <- written by Claude, ChatGPT
        +-- coding       <- written by Claude Code, Hermes, OpenClaw
```

## Setup

1. Sign up at [vectorize.io/hindsight](https://vectorize.io/hindsight/).
2. Open Muse and paste the contents of [`connect-prompt.md`](./connect-prompt.md).
3. Muse shows a **Connect** button for Hindsight. Click it and approve access on
   Hindsight's sign-in page.
4. Muse lists the banks it can see, creates its own `muse` bank plus the "About me"
   mental model and the nightly task, and tells you three things it found about you.

The prompt connects Muse to the root URL (`/mcp`), which reaches every bank in your
account. That is the point: Muse writes to its own bank and reads the others when a
question calls for it. Because the root URL also exposes bank management, the prompt
forbids `delete_bank`, `clear_memories` and `invalidate_memory`. If you would rather
hand Muse exactly one bank, connect it to `/mcp/<bank>/` instead: that endpoint takes
the same OAuth sign-in and drops the `bank_id` argument from every tool.

## Self-hosted Hindsight

Muse needs a public HTTPS endpoint with OAuth 2.1 discovery and dynamic client
registration. Put the [`cloudflare-oauth-proxy`](../cloudflare-oauth-proxy/) in front of
your instance, then check the endpoint with the preflight tool:

```bash
uvx --from ./hindsight-integrations/meta-muse hindsight-muse-preflight https://memory.example.com/mcp

# Also check the MCP tools, with a token from the environment
HINDSIGHT_API_KEY=... uvx --from ./hindsight-integrations/meta-muse hindsight-muse-preflight https://memory.example.com/mcp
```

It replays what Muse does: the unauthenticated 401 challenge, the OAuth
protected-resource and authorization-server metadata (registration endpoint, PKCE S256),
then MCP `initialize` and `tools/list`.

## Verify it works

In Muse:

- "Which memory banks can you see?" → `list_banks`
- "What was I working on last week?" → `recall` across your work banks
- "Remember that I prefer window seats." → `retain` into `muse`
- "How do I usually plan trips?" → `reflect`

What Muse retains then shows up in your other tools connected to the same account.
