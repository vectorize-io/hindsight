<div align="center">

![Hindsight Banner](./hindsight-docs/static/img/hindsight-github-banner.png)

[Documentation](https://hindsight.vectorize.io) • [Integrations](https://hindsight.vectorize.io/integrations) • [Cookbook](https://hindsight.vectorize.io/cookbook) • [Benchmarks](https://benchmarks.hindsight.vectorize.io/) • [Paper](https://arxiv.org/abs/2512.12818) • [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)

[![Release](https://github.com/vectorize-io/hindsight/actions/workflows/release.yml/badge.svg)](https://github.com/vectorize-io/hindsight/actions/workflows/release.yml)
[![Version](https://img.shields.io/pypi/v/hindsight-api?logo=python&logoColor=white&label=version&color=blue)](https://pypi.org/project/hindsight-api/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/hindsight-client?logo=pypi&logoColor=white&label=PyPI&color=blue)](https://pypi.org/project/hindsight-client/)
[![NPM Downloads](https://img.shields.io/npm/dm/%40vectorize-io%2Fhindsight-client?logo=npm&logoColor=white&label=NPM&color=blue)](https://www.npmjs.com/package/@vectorize-io/hindsight-client)
[![Slack Community](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<br/>

<a href="https://trendshift.io/repositories/15603" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15603" alt="vectorize-io%2Fhindsight | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

---

## What is Hindsight?

Hindsight™ is an agent memory system built to create smarter agents that learn over time. Most agent memory systems focus on recalling conversation history. Hindsight is focused on making agents that learn, not just remember.

<video src="https://github.com/user-attachments/assets/923b798d-3581-4897-bb62-9cfa5a931682" controls></video>

It eliminates the shortcomings of alternative techniques such as RAG and knowledge graph and delivers state-of-the-art performance on long term memory tasks.

**Contents**

- [Memory Performance & Accuracy](#memory-performance--accuracy)
- [Quick Start](#quick-start) — [server](#1-start-a-server) · [clients](#2-connect-a-client) · [platforms](#supported-platforms) · [embedded](#python-embedded-no-server-required)
- [Adding Hindsight to Your Agent](#adding-hindsight-to-your-agent) — [LLM Wrapper](#llm-wrapper-2-lines-of-code) · [integrations](#integrations) · [coding agents](#coding-agents) · [MCP](#mcp-server)
- [Core Concepts](#core-concepts) — [memory types](#memory-types) · [retain / recall / reflect](#the-three-operations) · [observations](#observations) · [mental models & knowledge pages](#mental-models--knowledge-pages) · [banks](#memory-banks)
- [Use Cases](#use-cases)
- [Running in Production](#running-in-production)
- [Resources](#resources)

---

## Memory Performance & Accuracy

Hindsight is the most accurate agent memory system ever tested according to benchmark performance. It has achieved state-of-the-art performance on the LongMemEval benchmark, widely used to assess memory system performance across a variety of conversational AI scenarios. The current reported performance of Hindsight and other agent memory solutions as of January 2026 is shown here:

![Overview](./hindsight-docs/static/img/hindsight-benchmarks.png)

> Live, continuously updated results — including per-model accuracy, latency and cost — are published at [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io/).

The benchmark performance data for Hindsight has been independently reproduced by research collaborators at the Virginia Tech [Sanghani Center for Artificial Intelligence and Data Analytics](https://sanghani.cs.vt.edu/) and The Washington Post. Other scores are self-reported by software vendors.

Hindsight is being used in production at Fortune 500 enterprises and by a growing number of AI startups.

---

> 🤖 **Using a coding agent?** Install the Hindsight documentation skill for instant access to docs while you code:
> ```bash
> npx skills add https://github.com/vectorize-io/hindsight --skill hindsight-docs
> ```
> Works with Claude Code, Cursor, and other AI coding assistants.

---

## Quick Start

### 1. Start a server

#### Docker (recommended)

```bash
export OPENAI_API_KEY=sk-xxx

docker run -it --pull always --name hindsight --restart unless-stopped -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v hindsight-data:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

>API: http://localhost:8888
>UI: http://localhost:9999

Hindsight works with **25+ LLM providers** via `HINDSIGHT_API_LLM_PROVIDER` — hosted (`openai`, `anthropic`, `gemini`, `groq`, `bedrock`, `vertexai`, `minimax`, `deepseek`, `atlas`, …), fully local (`ollama`, `lmstudio`, `llamacpp`), any OpenAI-compatible endpoint, and gateways (`litellm`, `litellmrouter`) that reach the rest. Existing subscriptions work too: `openai-codex` (ChatGPT Plus/Pro), `claude-code` (Claude Pro/Max) and `github-copilot` (GitHub Copilot) need no API key. See [supported models](https://hindsight.vectorize.io/developer/models).

#### Docker (external PostgreSQL)

```bash
export OPENAI_API_KEY=sk-xxx
export HINDSIGHT_DB_PASSWORD=choose-a-password
cd docker/docker-compose
docker compose up
```

> Oracle AI Database is also supported for enterprise deployments with full feature parity. See the [storage documentation](https://hindsight.vectorize.io/developer/storage) for details.

#### Bare metal (pip)

```bash
pip install hindsight-api
export HINDSIGHT_API_LLM_API_KEY=sk-xxx

hindsight-api
```

#### Kubernetes (Helm)

```bash
helm install hindsight oci://ghcr.io/vectorize-io/charts/hindsight \
  --set api.llm.provider=openai \
  --set api.llm.apiKey=sk-xxx \
  --set postgresql.enabled=true
```

#### Managed (no server)

[Hindsight Cloud](https://vectorize.io/pricing) is the hosted option: managed infrastructure that scales automatically, plus a dashboard, backups, team collaboration and a 99.9% uptime SLA. Billing is usage-based with free credits to start — no fixed monthly or per-seat fee. Point any client at `https://api.hindsight.vectorize.io` with your API key and skip the deployment entirely.

[Compare self-hosted, Cloud and Enterprise →](https://vectorize.io/pricing) · [Sign up →](https://ui.hindsight.vectorize.io/signup)

All options, including Windows and air-gapped setups, are covered in the [installation guide](https://hindsight.vectorize.io/developer/installation).

### 2. Connect a client

```bash
pip install hindsight-client -U                                  # Python
npm install @vectorize-io/hindsight-client                        # Node.js / TypeScript
go get github.com/vectorize-io/hindsight/hindsight-clients/go     # Go
curl -fsSL https://hindsight.vectorize.io/get-cli | bash          # CLI
```

#### Python

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain: Store information
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")

# Recall: Search memories
client.recall(bank_id="my-bank", query="What does Alice do?")

# Reflect: Generate disposition-aware response
client.reflect(bank_id="my-bank", query="Tell me about Alice")
```

#### Node.js / TypeScript

```javascript
const { HindsightClient } = require('@vectorize-io/hindsight-client');

const main = async () => {
  const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

  await client.retain('my-bank', 'Alice loves hiking in Yosemite');

  const results = await client.recall('my-bank', 'What does Alice like?');
  console.log(results);
}

main();
```

Full reference: [Python](https://hindsight.vectorize.io/sdks/python) · [Node.js](https://hindsight.vectorize.io/sdks/nodejs) · [Go](https://hindsight.vectorize.io/sdks/go) · [CLI](https://hindsight.vectorize.io/sdks/cli) · [REST API](https://hindsight.vectorize.io/api-reference)

### Supported Platforms

| Platform | Docker | Bare Metal (pip) | Embedded DB (pg0) |
|----------|--------|------------------|--------------------|
| **Linux** (x86_64, ARM64) | ✅ | ✅ | ✅ |
| **macOS** (Apple Silicon / arm64) | ✅ | ✅ | ✅ |
| **macOS** (Intel / x86_64) | ✅ | ⚠️ | ✅ |
| **Windows** (x86_64) | ✅ | ✅ | ✅ |

⚠️ Intel Macs: use `hindsight-all-slim` — see the [installation guide](https://hindsight.vectorize.io/developer/installation#supported-platforms) for details.

### Python Embedded (no server required)

```bash
pip install hindsight-all -U
```

On Intel (x86_64) Macs, install `hindsight-all-slim` instead — see [Supported Platforms](#supported-platforms).

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(
    llm_provider="openai",
    llm_model="gpt-5-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"]
) as server:
    client = HindsightClient(base_url=server.url)
    client.retain(bank_id="my-bank", content="Alice works at Google")
    results = client.recall(bank_id="my-bank", query="Where does Alice work?")
```

A [Node.js equivalent](https://hindsight.vectorize.io/sdks/hindsight-all-npm) and a [daemon CLI](https://hindsight.vectorize.io/sdks/embed) are also available.

---

## Adding Hindsight to Your Agent

### LLM Wrapper (2 lines of code)

The easiest way to add memory to an existing agent is the LLM Wrapper. Swap your LLM client for a wrapped one — memories are then stored and retrieved automatically on every call, with no other changes to your code.

```bash
pip install hindsight-litellm
```

```python
from openai import OpenAI
from hindsight_litellm import wrap_openai

# Wrap your existing LLM client and you're done.
# Defaults to Hindsight Cloud; pass hindsight_api_url for a self-hosted server.
client = wrap_openai(
    OpenAI(),
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

# Hindsight recalls relevant memories before the call
# and retains the conversation after it.
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "What do you know about me?"}],
)
```

`wrap_anthropic()` does the same for the Anthropic SDK, and every setting — bank, recall budget, fact types, reflect instead of recall — can be overridden per call with `hindsight_*` kwargs. LiteLLM sits underneath, so the same integration covers **100+ models**. See the [LiteLLM integration](https://hindsight.vectorize.io/sdks/integrations/litellm).

If you need explicit control over *when* memories are stored and recalled, use the [SDKs or REST API](#2-connect-a-client) directly instead.

### Integrations

**60+ integrations** — most need no code changes.

| | |
|---|---|
| **Coding agents** | [Claude Code](https://hindsight.vectorize.io/sdks/integrations/claude-code) · [Codex](https://hindsight.vectorize.io/sdks/integrations/codex) · [Cursor](https://hindsight.vectorize.io/sdks/integrations/cursor) · [GitHub Copilot](https://hindsight.vectorize.io/sdks/integrations/github-copilot) · [opencode](https://hindsight.vectorize.io/sdks/integrations/opencode) · [Cline](https://hindsight.vectorize.io/sdks/integrations/cline) · [Aider](https://hindsight.vectorize.io/sdks/integrations/aider) · [Zed](https://hindsight.vectorize.io/sdks/integrations/zed) · [Continue](https://hindsight.vectorize.io/sdks/integrations/continue) · [Roo Code](https://hindsight.vectorize.io/sdks/integrations/roo-code) · [OpenHands](https://hindsight.vectorize.io/sdks/integrations/openhands) |
| **Agent frameworks** | [LangGraph / LangChain](https://hindsight.vectorize.io/sdks/integrations/langgraph) · [LlamaIndex](https://hindsight.vectorize.io/sdks/integrations/llamaindex) · [CrewAI](https://hindsight.vectorize.io/sdks/integrations/crewai) · [Pydantic AI](https://hindsight.vectorize.io/sdks/integrations/pydantic-ai) · [OpenAI Agents SDK](https://hindsight.vectorize.io/sdks/integrations/openai-agents) · [Google ADK](https://hindsight.vectorize.io/sdks/integrations/google-adk) · [Agno](https://hindsight.vectorize.io/sdks/integrations/agno) · [Strands](https://hindsight.vectorize.io/sdks/integrations/strands) · [AutoGen](https://hindsight.vectorize.io/sdks/integrations/autogen) · [Microsoft Agent Framework](https://hindsight.vectorize.io/sdks/integrations/agent-framework) · [Vercel AI SDK](https://hindsight.vectorize.io/sdks/integrations/ai-sdk) · [Haystack](https://hindsight.vectorize.io/sdks/integrations/haystack) |
| **No-code / low-code** | [n8n](https://hindsight.vectorize.io/sdks/integrations/n8n) · [Zapier](https://hindsight.vectorize.io/sdks/integrations/zapier) · [Dify](https://hindsight.vectorize.io/sdks/integrations/dify) · [Flowise](https://hindsight.vectorize.io/sdks/integrations/flowise) |
| **Apps & tools** | [ChatGPT](https://hindsight.vectorize.io/sdks/integrations/chatgpt) · [Perplexity](https://hindsight.vectorize.io/sdks/integrations/perplexity) · [Obsidian](https://hindsight.vectorize.io/sdks/integrations/obsidian) · [Pipecat](https://hindsight.vectorize.io/sdks/integrations/pipecat) · [Vapi](https://hindsight.vectorize.io/sdks/integrations/vapi) |

👉 [**Browse all integrations**](https://hindsight.vectorize.io/integrations)

### Coding Agents

One package gives CLI coding agents long-term project memory: a per-repo bank built automatically from git history and past sessions, injected into the agent as it starts working, plus curated knowledge pages covering architecture, conventions and in-flight work.

```bash
npx @vectorize-io/hindsight-coding-agents install all          # every detected agent, wired natively
npx @vectorize-io/hindsight-coding-agents install claude-code  # or just one
```

Supports Claude Code, Codex CLI, Cursor CLI, GitHub Copilot CLI, opencode, Kilo CLI, Cline CLI, Antigravity CLI, Devin CLI, Prime Agent, Grok Build and DeepSeek Harness. Ingestion is automatic — there is no setup command. See the [coding agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents).

### MCP Server

Every server ships a built-in [Model Context Protocol](https://modelcontextprotocol.io/) endpoint, one per bank, enabled by default:

```
http://localhost:8888/mcp/{bank_id}/
```

Point any MCP client at it to expose retain, recall and reflect as tools. See the [MCP server docs](https://hindsight.vectorize.io/developer/mcp-server).

---

## Core Concepts

![Overview](./hindsight-docs/static/img/hindsight-overview.webp)

### Memory Types

Most agent memory implementations rely on basic vector search or sometimes use a knowledge graph. Hindsight uses biomimetic data structures to organize agent memories in a way that is more like how human memory works:

- **World facts:** facts about the world ("The stove gets hot")
- **Experiences:** the agent's own experiences ("I touched the stove and it really hurt")
- **Observations:** consolidated, evidence-backed beliefs formed from many memories
- **Mental models:** learned understanding of the agent's world, synthesized from observations and facts

Memories live in **banks**. When memories are added, they are pushed into either the world facts or the experiences pathway, then represented as a combination of entities, relationships, and time series with sparse/dense vector representations to aid in later recall.

### The Three Operations

#### Retain

The `retain` operation is used to push new memories into Hindsight. It tells Hindsight to _retain_ the information you pass in as an input.

```python
client.retain(
    bank_id="my-bank",
    content="Alice got promoted to senior engineer",
    context="career update",
    timestamp="2025-06-15T10:00:00Z",
)
```

Behind the scenes, retain uses an LLM to extract key facts, temporal data, entities, and relationships. It passes these through a normalization process to transform extracted data into canonical entities, time series, and search indexes along with metadata. These representations create the pathways for accurate memory retrieval in the recall and reflect operations.

![Retain Operation](hindsight-docs/static/img/retain-operation.webp)

[Retain docs →](https://hindsight.vectorize.io/developer/retain)

#### Recall

The recall operation is used to retrieve memories. These memories can come from any of the memory types (world, experiences, etc.)

```python
client.recall(bank_id="my-bank", query="What does Alice do?")
client.recall(bank_id="my-bank", query="What happened in June?")   # temporal
```

Recall performs 4 retrieval strategies in parallel:
- Semantic: Vector similarity
- Keyword: BM25 exact matching
- Graph: Entity/temporal/causal links
- Temporal: Time range filtering

![Recall Operation](hindsight-docs/static/img/recall-operation.webp)

The individual results are merged, ordered by relevance using reciprocal rank fusion and a cross-encoder reranking model, then trimmed as needed to fit within the token limit.

[Recall docs →](https://hindsight.vectorize.io/developer/retrieval)

#### Reflect

The reflect operation performs a more thorough analysis of existing memories. This allows the agent to form new connections between memories and build a more thorough understanding of its world — or to answer a question that needs deep thinking rather than lookup.

```python
client.reflect(bank_id="my-bank", query="What should I know about Alice?")
```

For example, reflect supports use cases such as:

- An **AI Project Manager** reflecting on what risks need to be mitigated on a project.
- A **Sales Agent** reflecting on why certain outreach messages have gotten responses while others haven't.
- A **Support Agent** reflecting on opportunities where customers have questions not answered by current product documentation.

![Reflect Operation](hindsight-docs/static/img/reflect-operation.webp)

[Reflect docs →](https://hindsight.vectorize.io/developer/reflect)

### Observations

Retained facts don't stay a flat pile. In the background, Hindsight consolidates related facts into **observations** — deduplicated beliefs the bank has built up over time. Each observation keeps its supporting evidence with exact quotes and a proof count, and is *refined* rather than overwritten when new evidence arrives, so new information strengthens, weakens or extends an existing belief instead of silently replacing it.

[Observations docs →](https://hindsight.vectorize.io/developer/observations)

### Mental Models & Knowledge Pages

A **mental model** is a standing answer to a question about a bank ("What are this user's preferences?"). You define the question once; Hindsight writes the answer, stores it, and rewrites it in the background as the bank learns more. Reading one is a database read — no retrieval, no LLM call — so an agent can boot with a page of settled knowledge instead of rediscovering it every session.

**Knowledge pages** are mental models with the mechanics hidden: living documents a bank writes about itself, organized in folders like a wiki, searchable, and projectable onto disk as ordinary markdown. Supply a name and a question; every other decision is a default you can override.

[Mental models →](https://hindsight.vectorize.io/developer/mental-models) · [Knowledge pages →](https://hindsight.vectorize.io/developer/knowledge-pages)

### Memory Banks

A **bank** is an isolated memory store — one "brain" for one user, agent, or project. Isolation is strict: no cross-bank leakage. Banks carry background context and **disposition traits** (skepticism, literalism, empathy) that shape how reflect reasons over their memories, and can be created from declarative [bank templates](https://hindsight.vectorize.io/developer/api/bank-templates).

Two more things worth knowing:

- **Multilingual by default.** Input language is detected and preserved end to end — facts stay in their original language and entities keep their native script (张伟 stays 张伟, not "Zhang Wei"). [Docs →](https://hindsight.vectorize.io/developer/multilingual)
- **Memory Defense.** An opt-in, per-bank policy that scans every retain for secrets and PII against 45 patterns and either redacts the match (`[REDACTED:github_token]`) or blocks the item before it reaches storage. [Docs →](https://hindsight.vectorize.io/developer/memory-defense)

---

## Use Cases

Hindsight is built to support conversational AI agents as well as agents that are intended to perform tasks autonomously. The ideal use case for Hindsight are agents that require a blend of these features such as AI employees that need to handle open-ended tasks, change behavior based on user feedback, and learn to perform complex tasks to automate work at a level that approximates a human work. Hindsight can be used with simple AI workflows like those built with n8n and other similar tools, but may be overkill for such applications.

### Per-User Memories and Chat History

One of the simpler use cases you can use Hindsight for is to personalize AI chatbots and other conversational agents by storing and recalling memories associated with individual users.

The requirements for this use case usually look something like this:

![Per-User Memories](./hindsight-docs/static/img/per-user-memory-requirements.png)

<video src="https://github.com/user-attachments/assets/4805e8e1-e7d1-47c6-a4f8-2344a5ec8906" controls></video>

Satisfying these requirements in Hindsight is straightforward. When new user inputs and tool calls are ingested into Hindsight using the retain operation, custom metadata can be used to enrich the new memories. Metadata provides a convenient way to isolate memories that need to be restricted to a given user. Once these are fed into the retain operation, any raw memories and mental models that get created can be filtered when retrieving relevant memories.

![Per-User Memories](./hindsight-docs/static/img/per-user-memory-howto.png)

More patterns in the [Cookbook](https://hindsight.vectorize.io/cookbook) and [Best Practices](https://hindsight.vectorize.io/best-practices).

---

## Running in Production

| | |
|---|---|
| **Storage** | PostgreSQL + pgvector, or Oracle AI Database 23ai with full feature parity — [storage](https://hindsight.vectorize.io/developer/storage) |
| **Configuration** | Hierarchical: global env vars → per-tenant → per-bank — [configuration](https://hindsight.vectorize.io/developer/configuration) |
| **Monitoring** | Prometheus metrics and dashboards for LLM calls, tokens and latency — [monitoring](https://hindsight.vectorize.io/developer/monitoring) |
| **Operations** | Admin CLI for migrations, bank repair and stuck operations — [admin CLI](https://hindsight.vectorize.io/developer/admin-cli) |
| **Events** | Webhooks for retain, consolidation and refresh lifecycle events — [webhooks](https://hindsight.vectorize.io/developer/api/webhooks) |
| **Extensibility** | Tenant, auth and storage extension points — [extensions](https://hindsight.vectorize.io/developer/extensions) |
| **Managed** | Skip all of it with [Hindsight Cloud](https://vectorize.io/pricing) — managed, usage-based, 99.9% uptime SLA |

---

## Resources

**Documentation:**
- [Docs](https://hindsight.vectorize.io) · [FAQ](https://hindsight.vectorize.io/faq) · [Best Practices](https://hindsight.vectorize.io/best-practices) · [Cookbook](https://hindsight.vectorize.io/cookbook) · [Blog](https://hindsight.vectorize.io/blog)
- [Paper](https://arxiv.org/abs/2512.12818) · [Benchmarks](https://benchmarks.hindsight.vectorize.io/) · [RAG vs Memory](https://hindsight.vectorize.io/developer/rag-vs-hindsight)

**Clients:**
- [Python](https://hindsight.vectorize.io/sdks/python) · [Node.js](https://hindsight.vectorize.io/sdks/nodejs) · [Go](https://hindsight.vectorize.io/sdks/go) · [CLI](https://hindsight.vectorize.io/sdks/cli) · [REST API](https://hindsight.vectorize.io/api-reference)

**Community:**
- [Slack](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg)
- [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)

---

## Star History

[![Star history](https://raw.githubusercontent.com/vectorize-io/hindsight/main/.github/star-history/chart.svg)](https://github.com/vectorize-io/hindsight/stargazers)

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT — see [LICENSE](./LICENSE)

---

Built by [Vectorize.io](https://vectorize.io)

<img src="https://umami-pixel.chris-latimer.workers.dev/?id=a8b043e6-6964-454d-80df-69b69d3f0d50&host=github.com&url=/vectorize-io/hindsight" width="1" height="1" alt="" />
