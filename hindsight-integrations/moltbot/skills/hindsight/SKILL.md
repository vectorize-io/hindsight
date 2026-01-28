---
name: memory_search
description: Search your long-term memory for relevant facts, experiences, and context using semantic and graph-based retrieval
user-invocable: false
disable-model-invocation: false
---

# memory_search

Search your long-term memory for relevant information. This tool provides multi-strategy retrieval combining:
- Semantic search across facts and experiences
- BM25 keyword matching
- Entity graph traversal
- Temporal queries
- Cross-encoder reranking

## Usage

Call `memory_search` with a natural language query to find relevant memories:

```
memory_search "What does the user prefer for breakfast?"
memory_search "When did we discuss the project deadline?"
memory_search "Tell me about Paris"
```

## Returns

Returns a list of relevant memory fragments with:
- Content: The actual memory text
- Score: Relevance score (0-1)
- Metadata: Source document, creation date, entities

Use the results to inform your responses with context from past conversations.
