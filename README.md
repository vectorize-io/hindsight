# Hindsight

**Long-term memory for AI agents.**

AI assistants forget everything between sessions. Hindsight fixes that with a memory system that handles temporal reasoning, entity connections, and personality-aware responses.

## Why Hindsight?

- **Temporal queries** — "What did Alice do last spring?" requires more than vector search
- **Entity connections** — Knowing "Alice works at Google" + "Google is in Mountain View" = "Alice works in Mountain View"
- **Agent opinions** — Agents form and recall beliefs with confidence scores
- **Personality** — Big Five traits influence how agents process and respond to information

## 60-seconds step


### 1. Install the Hindsight All package (client + API)

```bash
pip install hindsight-all
```

### 2. Import your OpenAI API key
```bash
export OPENAI_API_KEY=xx
```

### 3. Run embedded server and client

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(llm_provider="openai", llm_model="gpt-5.1-mini", llm_api_key=os.environ["OPENAI_API_KEY"]) as server:
    client = HindsightClient(base_url=server.url)

    client.put(agent_id="my-agent", content="Alice works at Google")
    client.put(agent_id="my-agent", content="Bob prefers Python over JavaScript")
    
    client.search(agent_id="my-agent", query="What does Alice do?")
    
    client.think(agent_id="my-agent", query="Tell me about Alice")
```



## Documentation

Full documentation: [hindsight-docs](./hindsight-docs)

- [Architecture](./hindsight-docs/docs/developer/architecture.md) — How ingestion, storage, and retrieval work
- [Python Client](./hindsight-docs/docs/sdks/python.md) — Full API reference
- [API Reference](./hindsight-docs/docs/api-reference/index.md) — REST API endpoints
- [Personality](./hindsight-docs/docs/developer/personality.md) — Big Five traits and opinion formation

## License

MIT
