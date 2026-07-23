
# Perplexity

Add persistent, searchable memory to [Perplexity](https://www.perplexity.ai) using [Hindsight](https://vectorize.io/hindsight). Store research findings and automatically recall relevant context in future searches—all secured with OAuth.

## Overview

Perplexity excels at research and fact-checking, but you have to re-discover the same facts across sessions. Hindsight solves this by providing:

- **Research knowledge base** — Discoveries, findings, and sources persist across searches
- **Smart recall** — Hindsight automatically retrieves relevant research from your history
- **No API keys** — OAuth handles authentication securely
- **Custom instructions** — Aggressive auto-retain/recall via system prompts
- **Web search + memory** — Combine Perplexity's web research with your accumulated knowledge

## Quick Start

### 1. Create a Hindsight Cloud Account

[Sign up free](https://ui.hindsight.vectorize.io/signup) for Hindsight Cloud.

### 2. Add Hindsight as a Connector in Perplexity

Requires **Perplexity Pro** subscription. Remote MCP connectors are a Pro feature.

1. Go to [Perplexity Settings](https://www.perplexity.ai/settings)
2. Navigate to **Connectors → + Custom Connector**
3. Fill in:
   - **Name:** `Hindsight`
   - **MCP server URL:** `https://api.hindsight.vectorize.io/mcp/default/`
4. Click **Add** — a browser window opens for Hindsight Cloud login
5. Sign in to [Hindsight Cloud](https://ui.hindsight.vectorize.io) and approve access
6. Return to Perplexity; the connector is now active

OAuth auto-discovery handles authentication automatically. That's it!

### 3. Configure Custom Instructions for Automatic Retention

The connector is now active, but you need to tell Perplexity to actually use it. Add custom instructions to enable automatic memory capture and recall:

1. Go to **Settings → Personalization → Custom instructions**
2. Copy and paste this instruction:

```
After every search and response, automatically use the Hindsight tool to retain:
- Key research findings and sources
- Facts and data points we've discovered
- Your preferences or research patterns
- Methodologies or search strategies that worked well

Before each new search, automatically use Hindsight to recall relevant research and context from previous conversations. Use recalled memories to inform your search strategy and answer.

Retain and recall everything—Hindsight handles filtering and deduplication.
```

3. Save and close settings

Perplexity will now automatically retain research findings and recall them for future searches, building a persistent knowledge base from your research.

> **💡 Tip**
>
Feel free to experiment with the instructions to ensure proper behavior.
## Features

- **Auto-retain research** — Custom instructions tell Perplexity to store findings after every search
- **Intelligent recall** — Perplexity queries Hindsight before each search to include relevant research as context
- **OAuth-secured** — No API keys to copy-paste; sign in once via browser
- **Bank isolation** — Separate memory banks for different research projects or domains
- **Semantic search** — Hindsight understands context, not just keyword matching
- **Web + memory** — Combine Perplexity's web search with your accumulated knowledge

## What to Store

Store meaningful research data for best results:

- **Research findings** — Key discoveries, trends, statistics you've found
- **Source collection** — Useful articles, papers, resources you've discovered
- **Research patterns** — Topics you frequently research, methodologies that work
- **Preferences** — Information sources you prefer, reporting styles you like
- **Domain knowledge** — Industry facts, terminology, context you've learned
- **Decision context** — Why you chose one approach over another, constraints considered

**Example of what to store:**
```
"Q3 2026 AI benchmarks:
- Claude Opus 4.7: Best reasoning, ~$15/1M input tokens
- GPT-4o: Fastest inference, multimodal, ~$5/1M input
- Llama 3.1: Open source, good for on-device, varies by host
- Performance metrics: [include relevant benchmark links]
- Use cases I care about: [your priorities]"
```

Later, when you research *"What's the best LLM for my use case?"*, Hindsight recalls your preferences and benchmarks. Perplexity's answer becomes tailored to your situation, not generic.

## Best Practices

**Be intentional about what you store.** Not every search needs retention. Focus on storing insights that will be useful later: research findings, source collections, methodologies, and personal preferences. Storing trivial facts clutters your memory bank and makes retrieval less useful.

**Use consistent terminology.** If you call a topic "machine learning" in one search and "deep learning" in another, Hindsight's semantic search may miss the connection. Establish naming conventions and stick to them.

**Review and refine.** The Hindsight Cloud dashboard lets you browse your memory bank. Periodically review what you've stored. Delete outdated information (e.g., last year's benchmark data) and consolidate similar insights. A curated memory bank is more valuable than an exhaustive one.

**Structure multi-part findings.** When storing complex research (like technology comparisons), include context: what you were evaluating, the criteria, alternatives considered, and your conclusions. Future-you will thank present-you for the context.

**Cross-reference with other tools.** If using both ChatGPT and Perplexity on related tasks, store research context in a shared Hindsight bank (multi-bank mode) so both tools access the same knowledge base. This creates unified context across your workflow.

## Comparison with ChatGPT

| Aspect | ChatGPT | Perplexity |
|--------|---------|-----------|
| **Best for** | Deep conversations, reasoning with memory | Research with memory, fact-checking |
| **Hindsight integration** | Retain reasoning, insights, preferences | Retain research findings, sources |
| **Session continuity** | Good for multi-turn problem-solving | Good for iterative research |
| **Web integration** | Limited (beta) | Integrated; combines memory + web search |
| **Memory context limit** | Depends on conversation length | Depends on search result count |

**Recommended use:**
- **ChatGPT + Hindsight** — Build projects, learn complex topics, creative work
- **Perplexity + Hindsight** — Research, fact-checking, competitive analysis, news tracking

**Ideal setup:** Use both tools together. ChatGPT handles reasoning with context. Perplexity handles research with context. Let them share Hindsight banks for coordinated workflows.

## Architecture: Single-Bank vs. Multi-Bank

### Single-Bank Mode (Recommended)

Each connector accesses one memory bank. Simpler for most users.

- **URL:** `https://api.hindsight.vectorize.io/mcp/default/`
- **Setup:** Just enter the URL in the Connector settings
- **Best for:** Dedicated memory per tool (e.g., Perplexity uses a `research` bank)

### Multi-Bank Mode

Both tools access multiple banks via bank_id parameter.

- **URL:** `https://api.hindsight.vectorize.io/mcp`
- **Setup:** Requires additional configuration in Hindsight Cloud
- **Best for:** When ChatGPT and Perplexity collaborate on the same project

## Data Privacy and Security

- **OAuth-secured** — no API keys, no copy-paste secrets
- **Your account** — memories live in your Hindsight Cloud account
- **Encrypted in transit** — HTTPS + TLS for all connections
- **No vendor lock-in** — export your memories anytime
- **Scoped access** — the connector can only read/write to its assigned bank

When you approve OAuth in the browser, you're authorizing Perplexity to:
- **Read** your memory banks (to recall relevant research)
- **Write** to your memory banks (to store new findings)
- **Search** your memories (to find context)

You can revoke access anytime by removing the connector in Perplexity settings.

## Troubleshooting

**"Connector failed to load"**
- Verify you have Perplexity Pro (required for Remote MCP connectors)
- Verify the URL is correct (no typos in your bank ID)
- Check that your Hindsight Cloud account is active
- Try re-creating the connector

**"Authorization failed" or "Access denied"**
- Make sure you're signing in with the same Hindsight Cloud account where you want to store memories
- If using a team account, verify you have permission to access the bank
- Try logging out and back in

**"Memory tools appear but don't return results"**
- Give Hindsight a few seconds to index memories (processing is async)
- Make sure you've stored relevant memories using the `retain` operation
- Check the memory bank name matches your connector URL

**Memories aren't being stored**
- Use the Hindsight Cloud dashboard to verify memories are being created
- In Perplexity, explicitly ask Hindsight to store something: *"Hindsight, remember that Q3 2026 benchmarks show Claude Opus is best for reasoning"*
- Check that your bank isn't full (unlikely, but possible with very large memory sets)

## Next Steps

1. Verify you have Perplexity Pro
2. Create your Hindsight Cloud account
3. Add the Hindsight connector to Perplexity
4. Set up your custom instructions
5. Store your first research finding
6. Start a new search on a related topic — watch Hindsight retrieve your stored research
7. Build the habit of storing meaningful research insights over time

Over weeks and months, as your memory bank grows, Perplexity will give increasingly informed answers because it's combining current web research with your accumulated knowledge base instead of starting fresh every time.
