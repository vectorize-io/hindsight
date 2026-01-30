<div align="center">

![Hindsight Banner](./hindsight-docs/static/img/hindsight-github-banner.png)

[Documentation](https://hindsight.vectorize.io) • [Paper](https://arxiv.org/abs/2512.12818) • [Cookbook](https://hindsight.vectorize.io/cookbook) • [Hindsight Cloud](https://vectorize.io/hindsight/cloud)

[![CI](https://github.com/vectorize-io/hindsight/actions/workflows/release.yml/badge.svg)](https://github.com/vectorize-io/hindsight/actions/workflows/release.yml)
[![Slack Community](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hindsight-api?label=PyPI)
![NPM Downloads](https://img.shields.io/npm/dm/%40vectorize-io%2Fhindsight-client?logoColor=orange&label=NPM&color=blue&link=https%3A%2F%2Fwww.npmjs.com%2Fpackage%2F%40vectorize-io%2Fhindsight-client)


</div>

---

## What is Hindsight?

Hindsight™ is an agent memory system built to create smarter agents that learn over time. Most agent memory systems focus on recalling conversation history. Hindsight is focused on making agents that learn, not just remember.


<video src="https://github.com/user-attachments/assets/923b798d-3581-4897-bb62-9cfa5a931682" controls></video>

It eliminates the shortcomings of alternative techniques such as RAG and knowledge graph and delivers state-of-the-art performance on long term memory tasks.

## Memory Performance & Accuracy

Hindsight is the most accurate agent memory system ever tested according to benchmark performance. It has achieved state-of-the-art performance on the LongMemEval benchmark, widely used to assess memory system performance across a variety of conversational AI scenarios. The current reported performance of Hindsight and other agent memory solutions as of January 2026 is shown here:

![Overview](./hindsight-docs/static/img/hindsight-bench.jpg)

The benchmark performance data for Hindsight has been independently reproduced by research collaborators at the Virginia Tech [Sanghani Center for Artificial Intelligence and Data Analytics](https://sanghani.cs.vt.edu/) and The Washington Post. Other scores are self-reported by software vendors.

Hindsight is being used in production at Fortune 500 enterprises and by a growing number of AI startups. 

## Adding Hindsight to Your AI Agents

The easiest way use Hindsight with an existing agent is with the LLM Wrapper. You can add memory to your agent with 2 lines of code. That will swap your current LLM client out with the Hindsight wrapper. After that, memories will be stored and retrieved automatically as you make LLM calls.

If you need more control over how and when your agent stores and recalls memories, there's also a simple API you can integrate with using the SDKs or directly via HTTP.

![Hindsight Banner](./hindsight-docs/static/img/migration-code.png)


## Quick Start

### Docker (recommended)

```bash
export OPENAI_API_KEY=your-key

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=o3-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

You can modify the LLM provider by setting `HINDSIGHT_API_LLM_PROVIDER`. Valid options are `openai`, `anthropic`, `gemini`, `groq`, `ollama`, and `lmstudio`. The documentation provides more details on [supported models](https://hindsight.vectorize.io/developer/models).

API: http://localhost:8888  
UI: http://localhost:9999

Install client:

```bash
pip install hindsight-client -U
# or
npm install @vectorize-io/hindsight-client
```

Python example:

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

### Python (embedded, no Docker)

```bash
pip install hindsight-all -U
```

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

### Node.js / TypeScript

```bash
npm install @vectorize-io/hindsight-client
```

```javascript
const { HindsightClient } = require('@vectorize-io/hindsight-client');

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

await client.retain('my-bank', 'Alice loves hiking in Yosemite');
await client.recall('my-bank', 'What does Alice like?');
```

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

---

## Architecture & Operations

![Overview](./hindsight-docs/static/img/hindsight-overview.webp)

Most agent memory implementation rely on basic vector search or sometimes use a knowledge graph. Hindsight uses biomimetic data structures to organize agent memories in a way that is more like how human memory works:

- **World:** Facts about the world ("The stove gets hot")
- **Experiences:** Agent's own experiences ("I touched the stove and it really hurt")
- **Mental Models:** Learned understanding of the agent's world formed by reflecting on raw memories and experiences.

Memories in Hindsight are stored in banks (i.e. memory banks). When memories are added to Hindsight, they are pushed into either the world facts or experiences memory pathway. They are then represented as a combination of entities, relationships, and time series with sparse/dense vector representations to aid in later recall.

Hindsight provides three simple methods to interact with the system:

- **Retain:** Provide information to Hindsight that you want it to remember
- **Recall:** Retrieve memories from Hindsight
- **Reflect:** Reflect on memories and experiences to generate new observations and insights from existing memories.

### Retain

The `retain` operation is used to push new memories into Hindsight. It tells Hindsight to _retain_ the information you pass in as an input.

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Simple
client.retain(
    bank_id="my-bank",
    content="Alice works at Google as a software engineer"
)

# With context and timestamp
client.retain(
    bank_id="my-bank",
    content="Alice got promoted to senior engineer",
    context="career update",
    timestamp="2025-06-15T10:00:00Z"
)
```

Behind the scenes, the retain operation uses an LLM to extract key facts, temporal data, entities, and relationships. It passes these through a normalization process to transform extracted data into canonical entities, time series, and search indexes along with metadata. These representations create the pathways for accurate memory retrieval in the recall and reflect operations. 

![Retain Operation](hindsight-docs/static/img/retain-operation.webp)

### Recall

The recall operation is used to retrieve memories. These memories can come from any of the memory types (world, experiences, etc.)

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Simple
client.recall(bank_id="my-bank", query="What does Alice do?")

# Temporal
client.recall(bank_id="my-bank", query="What happened in June?")
```

Recall performs 4 retrieval strategies in parallel:
- Semantic: Vector similarity
- Keyword: BM25 exact matching
- Graph: Entity/temporal/causal links
- Temporal: Time range filtering

![Retain Operation](hindsight-docs/static/img/recall-operation.webp)

The individual results from the retrievals are merged, then ordered by relevance using reciprocal rank fusion and a cross-encoder reranking model.

The final output is trimmed as needed to fit within the token limit.

### Reflect

The reflect operation is used to perform a more thorough analysis of existing memories. This allows the agent to form new connections between memories and build a more thorough understanding of its world.

For example, the `reflect` operation can be used to support use cases such as:

- An **AI Project Manager** reflecting on what risks need to be mitigated on a project.
- A **Sales Agent** reflecting on why certain outreach messages have gotten responses while others haven't.
- A **Support Agent** reflecting on opportunities where customers have questions not answered by current product documentation.

The `reflect` operation can also be used to handle on-demand question answering or analysis which require more deep thinking.

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

client.reflect(bank_id="my-bank", query="What should I know about Alice?")
```

![Retain Operation](hindsight-docs/static/img/reflect-operation.webp)

---

## Resources

**Documentation:** 
- [https://hindsight.vectorize.io](https://hindsight.vectorize.io)

**Clients:**
- [Python](http://hindsight.vectorize.io/sdks/python)
- [Node.js](http://hindsight.vectorize.io/sdks/nodejs)
- [REST API](https://hindsight.vectorize.io/api-reference)
- [CLI](https://hindsight.vectorize.io/sdks/cli)

**Community:**
- [Slack](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg)
- [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)

---
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vectorize-io/hindsight&type=date&legend=top-left)](https://www.star-history.com/#vectorize-io/hindsight&type=date&legend=top-left)
---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT — see [LICENSE](./LICENSE)

---

Built by [Vectorize.io](https://vectorize.io)
