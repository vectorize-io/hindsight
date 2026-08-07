
# Knowledge Pages

Living markdown documents, organized in a folder tree, that rewrite themselves as the bank learns.

A page is backed by a [mental model](./mental-models) but is configured as a document: it is built from the bank's [observations](../observations) only, it never reads other pages, and it refreshes incrementally after each consolidation. See [Knowledge Pages](../knowledge-pages) for the concepts behind the API.

{/* Import raw source files */}

All endpoints below are relative to a bank:

```
/v1/default/banks/{bank_id}/knowledge-base
```

---

## Get the Tree

Returns the whole knowledge base as a nested tree of folders and pages. Page bodies are **not** included — fetch a page to read its content.

### Python

```python
# Fetch the whole knowledge base as a nested folder/page tree (no page bodies)
tree = client.get_knowledge_base_tree(BANK_ID)

for root in tree.roots:
    print(f"{root.kind}: {root.name}")
    for child in root.children:
        print(f"  {child.kind}: {child.name} (stale: {child.is_stale})")
```

### Node.js

```javascript
// Fetch the whole knowledge base as a nested folder/page tree (no page bodies)
const tree = await client.getKnowledgeBaseTree(BANK_ID);

for (const root of tree.roots) {
    console.log(`${root.kind}: ${root.name}`);
    for (const child of root.children ?? []) {
        console.log(`  ${child.kind}: ${child.name} (stale: ${child.is_stale})`);
    }
}
```

### CLI

```bash
# Show the folder/page tree (no page bodies)
hindsight knowledge-base tree "$BANK_ID"
```

### Go

```go
# Section 'get-tree' not found in api/knowledge-pages.go
```

```json
{
  "roots": [
    {
      "id": "kf-9f2c...",
      "kind": "folder",
      "name": "Operations",
      "parent_id": null,
      "mental_model_id": null,
      "managed": false,
      "description": null,
      "tags": [],
      "timestamp": "2026-08-01T11:04:02+00:00",
      "is_stale": null,
      "children": [
        {
          "id": "kp-2e85...",
          "kind": "page",
          "name": "Deploying the API",
          "parent_id": "kf-9f2c...",
          "mental_model_id": "mm-77ab...",
          "managed": false,
          "description": "How is the API deployed?",
          "tags": ["ops"],
          "timestamp": "2026-08-03T09:12:44+00:00",
          "is_stale": true,
          "children": []
        }
      ]
    }
  ]
}
```

| Field | Description |
|---|---|
| `kind` | `folder` or `page` |
| `mental_model_id` | The backing mental model (pages only) |
| `description` | The page's source query — the question that rebuilds it |
| `timestamp` | Last refresh for a page, last update for a folder |
| `is_stale` | Pages only: `false` when the page is up to date, `true` when it *may* need a refresh (see below) |
| `managed` | `true` when the node is flagged as system-owned rather than hand-authored |

### How `is_stale` is decided

The tree answers for every page at once, from a single bank-wide signal: the last time *any* memory was written, returned as `last_memory_write_at` by the bank stats endpoint. A page refreshed at or after that moment is up to date — nothing in the bank changed, so nothing in the page's scope did either. A page refreshed before it gets `is_stale: true`, which means *may* need a refresh: the write might well have been outside the page's tags.

Read the page's mental model when you need certainty — [`GET /mental-models/{id}`](./mental-models) evaluates the page's own tag and fact-type scope and returns an exact `is_stale`. It is the more expensive answer, which is why the tree does not compute it per page.

---

## Create a Page

Creating a page stores it with placeholder content and schedules the first build in the background. Poll the returned `operation_id` via the [operations API](./operations) to know when the content is ready.

### Python

```python
# Create a page — content is generated in the background
page = client.create_knowledge_page(
    BANK_ID,
    name="Deploying the API",
    source_query="How is the API deployed?",
    parent_id=folder.id,
    tags=["ops", "type:runbook"],
)

# Poll the operation to know when the first build has finished
print(f"Page ID: {page.page_id}, operation: {page.operation_id}")
```

### Node.js

```javascript
// Create a page — content is generated in the background
const page = await client.createKnowledgePage(
    BANK_ID,
    'Deploying the API',
    'How is the API deployed?',
    { parentId: folder.id, tags: ['ops', 'type:runbook'] },
);

// Poll the operation to know when the first build has finished
console.log(`Page ID: ${page.page_id}, operation: ${page.operation_id}`);
```

### CLI

```bash
# Create a page — content is generated in the background
hindsight knowledge-base create-page "$BANK_ID" \
  "Deploying the API" \
  "How is the API deployed?" \
  --parent-id "$FOLDER_ID" \
  --tags ops,type:runbook
```

### Go

```go
# Section 'create-page' not found in api/knowledge-pages.go
```

```json
{
  "page_id": "kp-2e85...",
  "mental_model_id": "mm-77ab...",
  "operation_id": "op-1d0f..."
}
```

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Page name. Must be unique within its folder, case-insensitively — a duplicate returns `409`. (Enforced on PostgreSQL only.) |
| `source_query` | string | Yes | The question the page answers, re-asked on every refresh. |
| `parent_id` | string | No | Folder to create the page in. `null` (or omitted) creates it at the root. |
| `tags` | list | No | Tags that scope which memories the page is built from. A `type:<x>` tag also sets the page's rendered type. |
| `max_tokens` | int | No | Content budget. Defaults to `4096` (a plain mental model defaults to `2048`). |
| `trigger` | object | No | Refresh configuration — see below. |

### Default Trigger

When `trigger` is omitted, the page is created with a document-oriented configuration:

```json
{
  "mode": "delta",
  "fact_types": ["observation"],
  "exclude_mental_models": true,
  "refresh_after_consolidation": true
}
```

This makes the page a living document built from consolidated observations only, refreshed incrementally whenever consolidation produces new knowledge in its scope, and never influenced by other pages.

| Setting | Why |
|---|---|
| `fact_types: ["observation"]` | The page reads consolidated beliefs, not the raw conversational noise underneath them. Observations are already deduplicated and evidence-backed, so a page reads as a settled document instead of a transcript. Enforced structurally — with only `observation` in scope, the refresh agent isn't given the raw-memory recall tool at all. |
| `exclude_mental_models: true` | A page never reflects on sibling pages. Without this, pages would cite each other and drift into a feedback loop where one wrong claim propagates across the knowledge base. |
| `mode: "delta"` | Each refresh edits the existing document with what is new since the last refresh instead of regenerating it, so hand-tuned structure and wording survive. See [Refresh Mode](./mental-models#refresh-mode). |
| `refresh_after_consolidation: true` | The page rewrites itself whenever consolidation produces new knowledge in its scope — gated by the same [staleness check](./mental-models#staleness-gating) as any mental model, so unrelated bank activity doesn't trigger rebuilds. |

### Page Lifecycle

1. **Create** — the page is stored with placeholder content and a background refresh is submitted; the call returns immediately with an `operation_id`.
2. **First build** — a full generation, since there is no prior document to edit.
3. **Consolidation** — new memories are retained and consolidated into observations.
4. **Staleness check** — pages whose trigger asks for it are checked against their own scope (tags and `fact_types` both apply).
5. **Delta refresh** — stale pages are rewritten by editing the existing document with the new observations only.

Observations are what a page is *built from*, but it can still inspect the evidence underneath them: the refresh agent can expand a memory to its original chunk or document (unless `store_document_text` is disabled for the bank), and if `HINDSIGHT_API_REFLECT_SOURCE_FACTS_MAX_TOKENS` is enabled — off by default — observation search also returns each observation's grounding facts.

> **🚨 Caution**
>
A supplied `trigger` **replaces** these defaults, it does not merge with them. Sending `{"trigger": {"mode": "full"}}` also resets `fact_types` to all types, `exclude_mental_models` to `false`, and `refresh_after_consolidation` to `false`. Repeat the fields you want to keep.
Every [mental model trigger setting](./mental-models#trigger-settings) is accepted here — including `refresh_cron` for scheduled rebuilds instead of consolidation-driven ones, and `tag_groups` for compound tag scoping.

---

## Create a Folder

### Python

```python
# Create a folder (omit parent_id, or pass None, to create it at the root)
folder = client.create_knowledge_folder(BANK_ID, name="Operations")

print(f"Folder ID: {folder.id}")
```

### Node.js

```javascript
// Create a folder (omit parentId, or pass null, to create it at the root)
const folder = await client.createKnowledgeFolder(BANK_ID, 'Operations');

console.log(`Folder ID: ${folder.id}`);
```

### CLI

```bash
# Create a folder (omit --parent-id to create it at the root)
hindsight knowledge-base create-folder "$BANK_ID" "Operations"
```

### Go

```go
# Section 'create-folder' not found in api/knowledge-pages.go
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Folder name |
| `parent_id` | string | No | Parent folder. Omit or pass `null` for the root. |

A `parent_id` that does not exist, or that points at a page rather than a folder, returns `400`.

---

## Read a Page

Returns the page rendered as a markdown document.

### Python

```python
# Read a page as a markdown document
document = client.get_knowledge_page(BANK_ID, page.page_id)

print(document.type)      # "runbook" — from the type:runbook tag
print(document.body)      # the synthesized markdown body
print(document.markdown)  # YAML frontmatter + body
```

### Node.js

```javascript
// Read a page as a markdown document
const document = await client.getKnowledgePage(BANK_ID, page.page_id);

console.log(document.type);      // "runbook" — from the type:runbook tag
console.log(document.body);      // the synthesized markdown body
console.log(document.markdown);  // YAML frontmatter + body
```

### CLI

```bash
# Read a page as a markdown document
hindsight knowledge-base get-page "$BANK_ID" "$PAGE_ID"
```

### Go

```go
# Section 'get-page' not found in api/knowledge-pages.go
```

```json
{
  "id": "kp-2e85...",
  "name": "Deploying the API",
  "type": "runbook",
  "description": "How is the API deployed?",
  "tags": ["ops"],
  "timestamp": "2026-08-03T09:12:44+00:00",
  "body": "# Deploying the API\n\n...",
  "markdown": "---\nid: \"kp-2e85...\"\ntype: \"runbook\"\n...\n---\n\n# Deploying the API\n\n..."
}
```

- `body` is the synthesized markdown body on its own.
- `markdown` is the full document: a YAML frontmatter block (`id`, `type`, `title`, `description`, `tags`, `timestamp`) followed by the body.
- `type` comes from a `type:<x>` tag and defaults to `knowledge-page`. The `type:` tag is removed from the returned `tags`.

---

## Search Pages

Document-level hybrid search: a full-text (BM25) match and a vector-similarity match, fused with Reciprocal Rank Fusion. There is no reranking step, which keeps it fast enough to be an agent's first call.

### Python

```python
# Hybrid search (full-text + vector) over whole pages
results = client.search_knowledge_base(BANK_ID, q="how do we deploy", limit=5)

for hit in results.results:
    print(f"{hit.score:.3f}  {hit.name}: {hit.snippet}")
```

### Node.js

```javascript
// Hybrid search (full-text + vector) over whole pages
const results = await client.searchKnowledgeBase(BANK_ID, 'how do we deploy', { limit: 5 });

for (const hit of results.results) {
    console.log(`${hit.score.toFixed(3)}  ${hit.name}: ${hit.snippet}`);
}
```

### CLI

```bash
# Hybrid search (full-text + vector) over whole pages
hindsight knowledge-base search "$BANK_ID" "how do we deploy" --limit 5
```

### Go

```go
# Section 'search-pages' not found in api/knowledge-pages.go
```

```json
{
  "results": [
    {
      "id": "kp-2e85...",
      "name": "Deploying the API",
      "mental_model_id": "mm-77ab...",
      "snippet": "The API is deployed via ...",
      "score": 0.032,
      "updated_at": "2026-08-03T09:12:44+00:00"
    }
  ],
  "total": 1
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q` | string | — | Required. Search query (min length 1). |
| `limit` | int | `10` | Maximum results, 1–50. |

This searches whole pages. To search individual memories, use [recall](./recall).

---

## Update or Move a Node

One `PATCH` renames a node, moves it, and/or updates a page's options. Each field applies only when present in the body.

### Python

```python
# Rename a node, move it, and/or update a page's options.
# Changing source_query rebuilds the page against the new question.
client.update_knowledge_node(
    BANK_ID,
    page.page_id,
    name="Deploying the API (v2)",
    tags=["ops", "type:runbook", "reviewed"],
)
```

### Node.js

```javascript
// Rename a node, move it, and/or update a page's options.
// Changing sourceQuery rebuilds the page against the new question.
await client.updateKnowledgeNode(BANK_ID, page.page_id, {
    name: 'Deploying the API (v2)',
    tags: ['ops', 'type:runbook', 'reviewed'],
});
```

### CLI

```bash
# Rename a node, move it, and/or update a page's options.
# Changing --source-query rebuilds the page against the new question.
hindsight knowledge-base update "$BANK_ID" "$PAGE_ID" \
  --name "Deploying the API (v2)" \
  --tags ops,type:runbook,reviewed
```

### Go

```go
# Section 'update-node' not found in api/knowledge-pages.go
```

| Parameter | Type | Applies to | Description |
|---|---|---|---|
| `name` | string | Both | New name |
| `parent_id` | string \| null | Both | New parent folder. Pass `null` explicitly to move to the root. |
| `source_query` | string | Pages | New question. Changing it schedules an async refresh so the page rebuilds against the new question. |
| `tags` | list | Pages | Replaces the page's tags. Pass `[]` to clear them. |
| `max_tokens` | int | Pages | New content budget |

Sending an empty body returns `400`; an unknown node returns `404`.

---

## Delete a Node

Deletes a folder or page **and its entire subtree**. The mental models backing the deleted pages are removed too.

### Python

```python
# Delete a folder or page — deleting a folder removes its whole subtree
client.delete_knowledge_node(BANK_ID, folder.id)
```

### Node.js

```javascript
// Delete a folder or page — deleting a folder removes its whole subtree
await client.deleteKnowledgeNode(BANK_ID, folder.id);
```

### CLI

```bash
# Delete a folder or page — deleting a folder removes its whole subtree
hindsight knowledge-base delete "$BANK_ID" "$FOLDER_ID" -y
```

### Go

```go
# Section 'delete-node' not found in api/knowledge-pages.go
```

```json
{"status": "deleted"}
```

---

## Export as a Markdown Bundle

Returns the whole knowledge base as a flat set of portable markdown files: a nested `index.md`, one `<page-id>.md` per page, and a `<page-id>.log.md` refresh history for pages that have been rebuilt.

### Python

```python
# Export the knowledge base as a portable markdown bundle
bundle = client.export_knowledge_base(BANK_ID)

for file in bundle.files:
    print(file.path)  # index.md, <page-id>.md, <page-id>.log.md
```

### Node.js

```javascript
// Export the knowledge base as a portable markdown bundle
const bundle = await client.exportKnowledgeBase(BANK_ID);

for (const file of bundle.files) {
    console.log(file.path);  // index.md, <page-id>.md, <page-id>.log.md
}
```

### CLI

```bash
# Export the knowledge base as a portable markdown bundle
hindsight knowledge-base export "$BANK_ID"
```

### Go

```go
# Section 'export' not found in api/knowledge-pages.go
```

```json
{
  "files": [
    {"path": "index.md", "content": "---\ntype: \"index\"\n..."},
    {"path": "kp-2e85....md", "content": "---\nid: \"kp-2e85...\"\n..."},
    {"path": "kp-2e85....log.md", "content": "---\ntype: \"log\"\n..."}
  ]
}
```

> **💡 Mirror it to disk**
>
`hindsight fs mount --bank my-bank` keeps a local folder in sync with this bundle via a background refresh loop, so `ls`, `grep`, `rg`, and your editor work against real files.
---

## Storage

| Table | Holds |
|---|---|
| `knowledge_pages` | The tree: folders and pages, their names, parents, and ordering. A page row references its backing mental model; a folder row has none. |
| `mental_models` | The content: the document body, its source query, tags, token budget, trigger, and refresh history. |

The page layer owns only tree structure — everything about the content lives on the backing mental model, which is why every [mental model](./mental-models) capability applies to pages unchanged.

---

## Endpoint Summary

| Method | Path | Description |
|---|---|---|
| `GET` | `/knowledge-base/tree` | Nested folder/page tree with staleness |
| `POST` | `/knowledge-base/folders` | Create a folder |
| `POST` | `/knowledge-base/pages` | Create a page (async first build) |
| `GET` | `/knowledge-base/pages/{page_id}` | Read a page as markdown |
| `GET` | `/knowledge-base/search` | Hybrid search over pages |
| `PATCH` | `/knowledge-base/nodes/{node_id}` | Rename, move, or reconfigure a node |
| `DELETE` | `/knowledge-base/nodes/{node_id}` | Delete a node and its subtree |
| `GET` | `/knowledge-base/export` | Export the whole base as markdown files |
