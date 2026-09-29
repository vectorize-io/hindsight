---
title: "Connect ChatGPT and Perplexity to Hindsight Memory"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [integrations, chatgpt, perplexity, guide]
description: "Connect ChatGPT and Perplexity to Hindsight with MCP, OAuth, and custom instructions so both tools can retain context instead of starting cold."
image: /img/guides/guide-connect-chatgpt-and-perplexity-to-hindsight.png
hide_table_of_contents: true
---

![Connect ChatGPT and Perplexity to Hindsight Memory](/img/guides/guide-connect-chatgpt-and-perplexity-to-hindsight.png)

If you want to **connect ChatGPT and Perplexity to Hindsight**, the path is now clearly documented for both tools. The new first party integration guides show the same core pattern in two surfaces: add a remote MCP connector, complete OAuth in the browser, then give the model explicit instructions to retain and recall useful context. Keep [the ChatGPT integration guide](https://hindsight.vectorize.io/sdks/integrations/chatgpt), [the Perplexity integration guide](https://hindsight.vectorize.io/sdks/integrations/perplexity), [the MCP server docs](https://hindsight.vectorize.io/sdks/developer/mcp-server), and [the docs home](https://hindsight.vectorize.io) open while you set it up.

<!-- truncate -->

## The quick answer

- Both integrations use the Hindsight MCP endpoint and browser based OAuth, so there are no API keys to paste into the model UI.
- ChatGPT and Perplexity both need explicit instructions if you want automatic retain and recall behavior instead of manual tool use.
- The easiest starting point is one bank per tool, then move to a shared bank only when you actually want cross tool memory.

## What changed in the docs

The big improvement is not a new transport. It is clarity. Hindsight now has dedicated setup pages for [ChatGPT](https://hindsight.vectorize.io/sdks/integrations/chatgpt) and [Perplexity](https://hindsight.vectorize.io/sdks/integrations/perplexity), with the actual MCP URL, the OAuth flow, and example custom instructions in one place.

That matters because these integrations are only half finished if the connector exists but the model never calls it. The new docs make that explicit: wire up the connector first, then teach the tool when to retain and when to recall.

## Connect ChatGPT first

In ChatGPT, go to **Settings**, then **Apps & Connectors**, then create a connector pointed at:

```text
https://api.hindsight.vectorize.io/mcp/default/
```

After you create it, the browser should open the Hindsight OAuth approval flow. Finish that step, then add custom instructions that tell ChatGPT to retain important facts, decisions, and constraints after each response and recall relevant memories before answering.

The setup page in ChatGPT is easy to verify. You should be able to open a chat, select the connector, and ask it to remember a concrete fact. If that round trip fails, stop there and fix the connector before you tune the prompt.

## Connect Perplexity second

Perplexity uses the same basic model, but there is one extra gating factor: remote MCP connectors require **Perplexity Pro**.

Use the same MCP server URL in **Settings**, then **Connectors**, then **Custom Connector**. Complete the OAuth approval flow, then add custom instructions that tell Perplexity to retain findings, sources, and search patterns after each search.

Perplexity is strongest when it combines current web search with what you already learned in earlier research sessions. That is the piece Hindsight adds.

## Decide whether you want separate or shared memory

There are two sane starting points:

1. **Separate banks** for ChatGPT and Perplexity, which is simpler and reduces accidental mixing.
2. **A shared bank** when the two tools are working on the same project and should build on the same research, decisions, and constraints.

If you are unsure, start separate. Then move to a shared setup once you have a concrete reason. The shared pattern is described more directly in [the MCP server docs](https://hindsight.vectorize.io/sdks/developer/mcp-server) and in the follow on workflow from [the Perplexity guide](https://hindsight.vectorize.io/sdks/integrations/perplexity).

## Troubleshooting the first connection

The most common misses are boring, but they are also fixable:

- The connector exists, but the model never calls it. This is usually a custom instructions problem.
- OAuth succeeds, but tool calls return empty results. This is usually because the bank has no useful retained content yet.
- One tool works and the other does not. That usually means you configured different URLs or approved a different Hindsight account in the browser.
- Perplexity hides the feature entirely. That usually means the account is not on a Pro plan.

Treat the setup as three separate checks: connector added, OAuth approved, tool behavior validated in a real conversation.

## FAQ

### Do I need API keys inside ChatGPT or Perplexity?

No. The documented integration path uses browser based OAuth against the Hindsight MCP server.

### Should I enable automatic retention immediately?

Usually yes, but keep the instructions focused on durable facts, decisions, and findings. Retaining every trivial exchange makes the bank noisier.

### Which tool should I wire up first?

Start with the one you use more often. ChatGPT is usually the better first setup for reasoning workflows, while Perplexity is a strong second step for research heavy work.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [the ChatGPT integration guide](https://hindsight.vectorize.io/sdks/integrations/chatgpt)
- [the Perplexity integration guide](https://hindsight.vectorize.io/sdks/integrations/perplexity)
- [the MCP server docs](https://hindsight.vectorize.io/sdks/developer/mcp-server)
- [the docs home](https://hindsight.vectorize.io)
