
# Memories

A **memory unit** is the atomic fact Hindsight extracts and stores. This page covers the endpoints for working with individual memory units — reading and listing them, inspecting how a derived observation evolved, and **curating** them (correcting, retiring, or restoring). Ingesting and querying memories is covered separately in [Retain](./retain.mdx) and [Recall](./recall.mdx).

{/* Import raw source files */}

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/v1/default/banks/{bank}/memories/list` | List/filter memory units in a bank |
| `GET` | `/v1/default/banks/{bank}/memories/{id}` | Fetch a single memory unit |
| `GET` | `/v1/default/banks/{bank}/memories/{id}/history` | Refresh history of a derived observation |
| `PATCH` | `/v1/default/banks/{bank}/memories/{id}` | Curate: edit / invalidate / restore |
| `DELETE` | `/v1/default/banks/{bank}/memories/{id}/observations` | Clear a memory's derived observations |

## List memory units

List the memory units in a bank. The response includes each unit's `fact_type` (`world` | `experience` | `observation`), `state` (`valid` | `invalidated`), metadata, entities, occurred dates, and — for facts a user has edited — an `edited_at` timestamp. Invalidated rows are **included by default** so curation stays auditable; filter with `state=`.

### Python

```python
# Section 'list-memories' not found in api/memories.py
```

### Node.js

```javascript
// List memory units in a bank. Invalidated rows are included by default.
const memories = await client.listMemories(BANK_ID);
for (const unit of memories.items) {
    console.log(`- [${unit.fact_type}] ${unit.text}`);
}

// Filter to only the invalidated facts (e.g. to review duplicates).
const invalidated = await client.listMemories(BANK_ID, { state: 'invalidated' });
console.log(`${invalidated.items.length} invalidated fact(s)`);
```

### CLI

```bash
# List memory units in a bank (invalidated rows are included by default)
hindsight memory list "$BANK_ID"

# Filter to only the invalidated facts (e.g. to review duplicates)
curl -s "$HINDSIGHT_URL/v1/default/banks/$BANK_ID/memories/list?state=invalidated"
```

### Go

```go
# Section 'list-memories' not found in api/memories.go
```

## Fetch a single memory unit

Fetch a memory unit by ID, including its content, metadata, entities, timestamps, tags, and curation state.

### Python

```python
# Section 'get-memory' not found in api/memories.py
```

### Node.js

```javascript
// Fetch a single memory unit (metadata, entities, dates, state).
const memory = await (
    await fetch(`${HINDSIGHT_URL}/v1/default/banks/${BANK_ID}/memories/${memoryId}`)
).json();
console.log(`Text: ${memory.text}`);
console.log(`Type: ${memory.type}  Entities: ${memory.entities}`);
```

### CLI

```bash
# Section 'get-memory' not found in api/memories.sh
```

### Go

```go
# Section 'get-memory' not found in api/memories.go
```

For a **derived observation**, the history endpoint returns how it was refreshed over time as new source facts arrived:

### Python

```python
# Section 'observation-history' not found in api/memories.py
```

### Node.js

```javascript
# Section 'observation-history' not found in api/memories.mjs
```

### CLI

```bash
# Section 'observation-history' not found in api/memories.sh
```

### Go

```go
# Section 'observation-history' not found in api/memories.go
```

## Curation: editing, invalidating & pruning

Memory is append-only by design — but sometimes a stored fact is **wrong**, has gone **stale**, or is a **duplicate**. Curation lets you correct or retire individual memories without losing the audit trail. Retired facts are moved out of the active set, so recall never returns them, while remaining fully recoverable.

### When to reach for what

Not every "bad memory" needs the same tool. Pick by *why* it's bad:

| The memory is… | Use | Why |
|---|---|---|
| **Wrong because the whole bank extracts badly** (e.g. consistently wrong subject) | Fix the bank's `retain_mission` / `observations_mission`, then **reprocess** the document | Systematic problems are best fixed at the source, then replayed — see [Retain](./retain.mdx) and [Observations](../observations.mdx). |
| **Wrong as a one-off** (a single misextracted fact) | **Edit** the memory | Corrects the fact and regenerates everything derived from it. |
| **No longer true, with nothing to replace it** (decommissioned server, a tool that was fixed, a role that changed) | **Invalidate** the memory | Nothing in the pipeline knows the world changed, so you tell it explicitly. |
| **A duplicate or superseded fact** | **Invalidate** the memory | Removes the noise from recall while keeping the audit trail. |
| **Superseded by a newer fact you're storing anyway** (e.g. "likes BMW" → "likes Toyota") | Just retain the new fact | Consolidation already reconciles in-stream contradictions into a single observation. |

The rule of thumb: **if Hindsight could have known, let consolidation handle it; if only you know, curate it.**

Only raw **world** and **experience** facts can be curated. Observations are *derived* — they regenerate from their sources, so you curate the underlying facts, not the observation. A `PATCH` on an observation returns `400`.

### Edit a memory

Correct what the LLM extracted. You can change the **text**, **context**, **occurred dates**, **fact type**, and **entities** — anything the extractor could have gotten wrong. Hindsight re-embeds the fact, drops the observations and links derived from the old version, and re-consolidates, so downstream knowledge reflects the correction. Edited facts are marked with an `edited_at` timestamp (surfaced as an **Edited** badge in the control plane).

You don't need to rebuild anything yourself: an edit **automatically recomputes the knowledge graph and links** in the background. The fact's entity associations are re-resolved from the new text/entities, its temporal and semantic links are re-derived, and consolidation re-runs — all triggered by the edit. The PATCH returns as soon as the change is committed; the graph/observation rebuild happens asynchronously right after.

### Python

```python
# Section 'edit-memory' not found in api/memories.py
```

### Node.js

```javascript
// Correct the fact's text. Re-embeds, drops derived observations/links,
// re-consolidates, and recomputes the graph automatically.
await patchMemory(memoryId, { text: 'The user visited Paris in 2023.', reason: 'wrong subject' });
```

### CLI

```bash
# Section 'edit-memory' not found in api/memories.sh
```

### Go

```go
# Section 'edit-memory' not found in api/memories.go
```

You can correct the dates, fact type, and entities the same way. For `context`, `occurred_start`, and `occurred_end`, an empty string `""` clears the field and omitting it leaves it unchanged. For `entities`, a list **replaces** the fact's entity set (names are resolved/find-or-created the same way retain does) and `[]` detaches them all; omitting it leaves them unchanged.

### Python

```python
# Section 'edit-memory-fields' not found in api/memories.py
```

### Node.js

```javascript
// Correct dates, fact type, and entities in one call. "" clears a field;
// entities replaces the set ([] detaches all); omit to leave unchanged.
await patchMemory(memoryId, {
    occurred_start: '2023-06-01',
    fact_type: 'experience',
    entities: ['Alice', 'Paris'],
});
```

### CLI

```bash
# Section 'edit-memory-fields' not found in api/memories.sh
```

### Go

```go
# Section 'edit-memory-fields' not found in api/memories.go
```

### Invalidate a memory (reversible)

Soft-retire a fact. An invalidated memory:

- **disappears from recall**, consolidation, and the knowledge graph,
- has its **links pruned** and its **derived observations re-computed** without it,
- **stays in the bank** for audit (visible via the memory and document views), and
- can be **restored** at any time.

### Python

```python
# Section 'invalidate-memory' not found in api/memories.py
```

### Node.js

```javascript
// Soft-retire a fact: removed from recall/consolidation/graph, links pruned,
// derived observations recomputed without it — but kept for audit.
await patchMemory(memoryId, { state: 'invalidated', reason: 'server decommissioned 2026-06-01' });
```

### CLI

```bash
# Section 'invalidate-memory' not found in api/memories.sh
```

### Go

```go
# Section 'invalidate-memory' not found in api/memories.go
```

Restoring moves the fact back into the active set and re-consolidates:

### Python

```python
# Section 'restore-memory' not found in api/memories.py
```

### Node.js

```javascript
// Restore a previously invalidated fact.
await patchMemory(memoryId, { state: 'valid' });
```

### CLI

```bash
# Section 'restore-memory' not found in api/memories.sh
```

### Go

```go
# Section 'restore-memory' not found in api/memories.go
```

Behind the scenes, invalidating **moves** the row out of the active `memory_units` table into a separate archive, so recall and consolidation never need a "skip invalidated" filter — the rows simply aren't there.

> **📝 Documents are the source of truth**
> 
A memory is extracted from a document. Editing or invalidating a memory does **not** change the document it came from — that's deliberate: the document stays as an accurate historical record. As a result, **reprocessing a document resets curation** of the facts it produced (extraction runs fresh from the original text). Fix systematic issues at the mission level and reprocess; use edit/invalidate for the residue.
### A pruning workflow

To clean up duplicates and reclaim noise: cluster duplicates from `memories/list`, then **invalidate** them — recall is clean immediately, and the audit trail is preserved.
