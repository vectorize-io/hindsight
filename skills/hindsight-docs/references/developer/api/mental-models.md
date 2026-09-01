
# Mental Models

User-curated summaries that provide high-quality, pre-computed answers for common queries.

See [Mental Models](../mental-models) for the concepts behind this API.

{/* Import raw source files */}

## What Are Mental Models?

Mental models are **saved reflect responses** that you curate for your memory bank. When you create a mental model, Hindsight runs a reflect operation with your source query and stores the result. During future reflect calls, these pre-computed summaries are checked first — providing faster, more consistent answers.

```mermaid
graph LR
    A[Create Mental Model] --> B[Run Reflect]
    B --> C[Store Result]
    C --> D[Future Queries]
    D --> E{Match Found?}
    E -->|Yes| F[Return Mental Model]
    E -->|No| G[Run Full Reflect]
```

### Why Use Mental Models?

| Benefit | Description |
|---------|-------------|
| **Consistency** | Same answer every time for common questions |
| **Speed** | Pre-computed responses are returned instantly |
| **Quality** | Manually curated summaries you've reviewed |
| **Control** | Define exactly how key topics should be answered |

### Hierarchical Retrieval

During reflect, the agent checks sources in priority order:

1. **Mental Models** — User-curated summaries (highest priority)
2. **Observations** — Consolidated knowledge
3. **Raw Facts** — Ground truth memories

Mental models are checked first because they represent your explicitly curated knowledge.

---

## Create a Mental Model

Creating a mental model runs a reflect operation in the background and saves the result:

### Python

```python
# Create a mental model (runs reflect in background)
result = client.create_mental_model(
    bank_id=BANK_ID,
    name="Team Communication Preferences",
    source_query="How does the team prefer to communicate?",
    tags=["team", "communication"]
)

# Returns an operation_id - check operations endpoint for completion
print(f"Operation ID: {result.operation_id}")
```

### Node.js

```javascript
// Create a mental model (runs reflect in background)
const result = await client.createMentalModel(
    BANK_ID,
    'Team Communication Preferences',
    'How does the team prefer to communicate?',
    { tags: ['team', 'communication'] },
);

// Returns an operation_id — check operations endpoint for completion
console.log(`Operation ID: ${result.operation_id}`);
```

### CLI

```bash
# Create a mental model (runs reflect in background)
hindsight mental-model create "$BANK_ID" \
  "Team Communication Preferences" \
  "How does the team prefer to communicate?"
```

### Go

```go
# Section 'create-mental-model' not found in api/mental-models.go
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Human-readable name for the mental model |
| `source_query` | string | Yes | The query to run to generate content |
| `id` | string | No | Custom ID for the mental model (alphanumeric lowercase with hyphens). Auto-generated if omitted. |
| `tags` | list | No | Tags that scope the model during reflect **and** filter source memories during refresh. Defaults to `all_strict` matching, so only memories carrying every listed tag are read. See [Tags and Visibility](#tags-and-visibility). |
| `max_tokens` | int | No | Maximum tokens for the mental model content |
| `trigger` | object | No | Trigger settings (see [Automatic Refresh](#automatic-refresh)) |

---

## Create with Custom ID

Assign a stable, human-readable ID to a mental model so you can retrieve or update it by name instead of relying on the auto-generated UUID:

### Python

```python
# Create a mental model with a specific custom ID
result_with_id = client.create_mental_model(
    bank_id=BANK_ID,
    name="Communication Policy",
    source_query="What are the team's communication guidelines?",
    id="communication-policy"
)

print(f"Created with custom ID: {result_with_id.operation_id}")
```

### Node.js

```javascript
// Create a mental model with a specific custom ID
const resultWithId = await client.createMentalModel(
    BANK_ID,
    'Communication Policy',
    "What are the team's communication guidelines?",
    { id: 'communication-policy' },
);

console.log(`Created with custom ID: ${resultWithId.operation_id}`);
```

### CLI

```bash
# Create a mental model with a specific custom ID
hindsight mental-model create "$BANK_ID" \
  "Communication Policy" \
  "What are the team's communication guidelines?" \
  --id communication-policy
```

### Go

```go
# Section 'create-mental-model-with-id' not found in api/mental-models.go
```

> **💡 Tip**
>
Custom IDs must be lowercase alphanumeric and may contain hyphens (e.g. `team-policies`, `q4-status`). If a mental model with that ID already exists, the request is rejected.
---

## Automatic Refresh

Mental models can be configured to **automatically refresh** when observations are updated. This keeps them in sync with the latest knowledge without manual intervention.

### Trigger Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `mode` | `"full"` \| `"delta"` | `"full"` | Refresh strategy. See [Refresh Mode](#refresh-mode) below. |
| `refresh_after_consolidation` | bool | false | Automatically refresh after observations consolidation |
| `refresh_cron` | string \| null | null | UTC 5-field cron expression for scheduled refreshes, such as `"0 3 * * *"` for daily at 03:00 UTC |
| `min_refresh_interval_seconds` | int \| null | null | Minimum seconds between two *automatic* refreshes of this model. See [Rate-limiting automatic refreshes](#rate-limiting-automatic-refreshes) below. `null` uses the bank/global default. |
| `tags_match` | string \| null | null | How the model's `tags` filter source memories during refresh: `any`, `all`, `any_strict`, `all_strict`, or `exact`. When `null`, a **tagged** model defaults to `all_strict` (a memory must carry every one of the model's tags). Set `"any"` to match memories carrying *any* of the tags — see [Tags and Visibility](#tags-and-visibility). |
| `tag_groups` | list \| null | null | Advanced boolean tag expressions that override flat `tags`/`tags_match` entirely. See the [Recall tags reference](./recall#tags). |
| `fact_types` | list \| null | null | Restrict which fact types the refresh reads: any of `world`, `experience`, `observation`. `null` means all three. Must not be an empty list. |
| `exclude_mental_models` | bool | false | Hide *all* other mental models from the refresh, so the model never synthesises from sibling models. |
| `exclude_mental_model_ids` | list \| null | null | Hide specific mental models by ID from the refresh. The model being refreshed is always excluded from itself. |
| `include_chunks` | bool \| null | null | Override whether the refresh's internal recall returns raw chunk text. `null` uses the bank/global `recall_include_chunks` default. |
| `recall_max_tokens` | int \| null | null | Override the token budget for facts retrieved during refresh. `null` uses the bank/global default. |
| `recall_chunks_max_tokens` | int \| null | null | Override the token budget for raw chunks retrieved during refresh. `null` uses the bank/global default. |
| `response_schema` | object \| null | null | JSON Schema for structured output. When set, each refresh also stores a `structured_output` alongside the markdown content. See [Structured Output](#structured-output) below. |
| `keep_trace` | bool | false | Record how each refresh reached its result under `reflect_response.trace`. See [Troubleshoot a Refresh](#troubleshoot-a-refresh). |

When `refresh_after_consolidation` is enabled, the mental model will be re-generated every time the bank's observations are consolidated — ensuring it always reflects the latest synthesized knowledge.

When `refresh_cron` is set, Hindsight checks the schedule on the server's mental-model refresh tick and refreshes the model only if memories in its scope have changed since the last refresh. `refresh_cron` and `refresh_after_consolidation` are mutually exclusive, so a model refreshes either after consolidation or on a fixed UTC schedule, not both.

### Rate-limiting automatic refreshes

A refresh is a full reflect run: retrieval plus an agentic LLM loop. With
`refresh_after_consolidation` on a bank that ingests continuously, every small retain can
therefore pay for a rebuild of every model whose scope it touched — and a handful of models
on a chatty bank adds up fast.

`min_refresh_interval_seconds` puts a floor on how often that may happen:

```json
{
  "trigger": {
    "refresh_after_consolidation": true,
    "min_refresh_interval_seconds": 1800
  }
}
```

A trigger that fires inside the window is **not dropped**. Its refresh is queued and parked
until the window closes, and every further trigger in the meantime folds into that same
queued refresh — so a burst of twenty retains costs one refresh, and that refresh still
reads everything the twenty retains added. The only thing you trade away is immediacy: the
model can lag its memories by up to the interval.

The floor applies to both automatic triggers (`refresh_after_consolidation` and
`refresh_cron`). It never applies to a refresh you ask for — the refresh endpoint, the MCP
tool, and the control plane's refresh button all run immediately, and additionally release a
parked refresh if one is waiting.

Set it per bank (or server-wide) with `mental_model_min_refresh_interval_seconds` /
`HINDSIGHT_API_MENTAL_MODEL_MIN_REFRESH_INTERVAL_SECONDS`; the per-model value wins when both
are set, so `min_refresh_interval_seconds: 0` exempts one model that does need to stay current
from a bank-wide floor. The default everywhere is `0` — no floor.

While a refresh is parked, its operation stays `pending` with `next_retry_at` set to the moment
the window closes, so `GET /operations` shows exactly what it is waiting for.

### Structured Output

A mental model's content is always markdown. Set `trigger.response_schema` to *also* attach a machine-readable view: a [JSON Schema](https://json-schema.org/) **object** with a non-empty `properties` map (nested objects and arrays are supported). Each refresh then stores a `structured_output` on the model's `reflect_response`, next to the markdown — you get both the prose and a typed object.

```json
{
  "trigger": {
    "response_schema": {
      "type": "object",
      "properties": {
        "risk_level": { "type": "string" },
        "open_questions": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["risk_level"]
    }
  }
}
```

Key behaviours:

- **Extracted from the final document.** The structured output is parsed from the content that is actually stored — so in `delta` mode it reflects the whole merged document, not just the facts that changed in that refresh.
- **Fails loudly.** If a `response_schema` is configured but the structured extraction cannot be produced, the refresh **fails** rather than silently persisting content with no structured view. The model's previous content and `structured_output` are preserved, and the refresh can be retried.
- **Invalid schemas are rejected** at request time: the schema must be an object with at least one property, and each property's `type` must be one of `string`, `number`, `integer`, `boolean`, `array`, or `object`.

See [Reflect → `response_schema`](./reflect#response_schema) for how the same schema works on the `reflect` endpoint. The `structured_output` value appears on the model's `reflect_response` (see [Response Fields](#response-fields)).

### Staleness Gating

Both automatic triggers run the same check before spending an LLM call: **is there a memory in this model's resolved scope newer than its last refresh?** The model's `tags`/`tags_match`, `tag_groups`, and `fact_types` all apply to that check, so activity elsewhere in the bank does not trigger a rebuild, and a cron tick over an unchanged scope is skipped entirely. Memories still waiting to be consolidated count — they are already stored, so a model whose scope reaches them is considered stale.

Every `is_stale` you can read is computed by that same rule: the flag on a single mental-model read, the one on the list, the one surfaced to the reflect agent, and the per-page flag on the [knowledge-base tree](./knowledge-pages). A model flagged stale is one a refresh would actually rewrite.

Listing does not cost one query per model — the whole page is answered together — so you do not have to approximate. If you are already holding `last_memory_write_at` from the bank stats endpoint, a model whose `last_memory_seen_at` is at or after it is provably up to date without asking at all: nothing in the bank changed, so nothing in that model's scope did.

**Deletions are invisible to it.** Staleness asks whether anything in scope has been *written* since the model last read the memories. Deleting an in-scope memory leaves no write behind, so it does not raise the flag, and a document that cites the deleted fact keeps reporting itself up to date until something in its scope is written or you refresh it yourself.

**Two timestamps answer two different questions.** `last_refreshed_at` is wall-clock: when a refresh last finished. It advances on every refresh that completes — including one that read the scope, found nothing new, and left the document alone — so a client driving refreshes itself can use it to tell "I already did this one" from "this one still needs a call". `last_memory_seen_at` is a position in the data: the newest in-scope memory the last refresh saw. It stands still while the model's scope is quiet, however many times you refresh, which is what makes it the right side of the staleness comparison. A model with a recent `last_refreshed_at` and an old `last_memory_seen_at` is not a contradiction — it means you refreshed a document that had nothing new to say.

### What a Refresh Reads

- **A point-in-time snapshot.** Each refresh is bounded by a database-time cutoff and only reads memories committed at or before it, so a write landing mid-refresh is never half-included. The watermark recorded afterwards is the newest in-scope memory the refresh actually saw — not `now()` — so a write that commits just after the snapshot stays "newer" and is picked up on the next round instead of being skipped.
- **Only what's new, in `delta` mode.** A delta refresh scopes its retrieval to memories created since the last refresh, so the LLM reads genuinely new information rather than re-reading the whole bank.
- **Cumulative grounding.** Even in delta mode, `reflect_response.based_on` accumulates: the facts from this refresh are merged with everything previous refreshes relied on, deduplicated by id. The document rests on all of them, not just the latest slice.
- **Nothing, if nothing is relevant.** A refresh that finds no topic-relevant memories leaves the content alone and only advances the watermark, so the same empty window stops re-triggering.
- **No LLM call at all, if there is nothing to read.** Before the agentic loop starts, a refresh checks whether anything is in reach under this model's own settings: an in-scope memory (`tags`/`tags_match`, `tag_groups` and `fact_types` all narrow it) inside the window it would read — the whole bank in `full` mode, everything since `last_memory_seen_at` in `delta` mode — or a sibling mental model, unless `exclude_mental_models` closed that door. When there is nothing, the refresh completes immediately, leaves the document untouched, and spends nothing. This is the common case on a bank that was just created: its pages are refreshed the moment they exist, before anything has been retained.

### Refresh Mode

Two strategies are available for how a refresh produces the new content:

- **`full`** *(default)* — every refresh regenerates the entire content from scratch. Simple and predictable: the LLM synthesises a fresh document from the retrieved memories. Best when the document is short, when you want every refresh to potentially restructure the output, or when you're not yet sure what the final shape should be.

- **`delta`** — refresh emits a list of typed *operations* (add a section, append a bullet, replace a block, remove a stale paragraph) against the document's existing structure, then renders the result. Sections that aren't targeted by any operation are copied through **byte-identical** — no paraphrasing, no whitespace drift, no list-style normalisation. Best for long-lived "playbook"–style mental models where you want stability across refreshes and only the genuinely changed parts to move.

#### How delta mode works

Hindsight keeps an authoritative **structured** representation of the document — an ordered list of sections, each holding a list of blocks, where a block is one markdown fragment (a paragraph, a list, a table, a code fence) stored exactly as written. The markdown you read is a deterministic render of that structure. Because a block is never re-interpreted, formatting a refresh didn't touch cannot be rewritten by one — a table stays a table. A delta refresh never asks the LLM to rewrite the document; it asks for operations against that structure:

| Operation | Effect |
|---|---|
| `add_section` / `remove_section` | Add or drop a whole section |
| `rename_section` | Change a section's heading |
| `replace_section_blocks` | Replace a section's contents wholesale |
| `append_block` / `insert_block` | Add a block at the end of, or after a named block in, a section |
| `replace_block` / `remove_block` | Change or drop one block |

Anything no operation mentions is copied through untouched, so unchanged prose is preserved rather than regenerated and checked. This matters because "preserve the unchanged content" is only a soft constraint on an LLM — generating the next token from a gestalt of the input is what it intrinsically does, so instructed-to-preserve prose drifts over many refreshes.

Sections and blocks are addressed by id, never by position, so an operation cannot land on the wrong one by miscounting. Failure modes are conservative by design: an operation referencing a section or block that doesn't exist — or a block that lives in a different section than the one it names — is **dropped** rather than guessed at, and the rest of the operations still apply. The refresh records which ones were dropped and why, so you can see that part of that round's new information didn't make it into the document.

Delta mode falls back to a full regeneration automatically in two cases:
1. The mental model has no existing content yet (nothing to anchor edits on).
2. The `source_query` has changed since the last refresh (the topic has shifted; the existing structure may no longer apply).

**A delta refresh never replaces the document with a partial one.** Because delta retrieval only reads memories newer than the last refresh, an answer written from that window covers just the recent slice of the topic — it is material for editing the document, not a replacement for it. So when the edits can't be made at all — the provider call fails, the response can't be read, or every single operation is rejected — the existing content stays exactly as it is and the refresh **fails** instead of completing. Nothing is lost, the refresh's time window is not advanced, and a retry sees the same memories again. The same holds for an empty answer: a populated document is never overwritten with an empty one.

| Use Case | Recommended Mode | Why |
|----------|-----------------|-----|
| Skill / playbook docs | `delta` | Sections live for many refreshes; only specific rules change |
| Onboarding summaries | `delta` | Adding new team members shouldn't restructure the doc |
| Real-time dashboards | `full` | Each refresh is a fresh snapshot |
| Short FAQ summaries | `full` | Whole-document regeneration is cheap and unambiguous |

### Python

```python
# Create a mental model with automatic refresh enabled
result = client.create_mental_model(
    bank_id=BANK_ID,
    name="Project Status",
    source_query="What is the current project status?",
    trigger={"refresh_cron": "0 3 * * *"}
)

# This mental model checks daily at 03:00 UTC and refreshes when scoped memories changed
print(f"Operation ID: {result.operation_id}")
```

### Node.js

```javascript
// Create a mental model with automatic refresh enabled
const result2 = await client.createMentalModel(
    BANK_ID,
    'Project Status',
    'What is the current project status?',
    { trigger: { refreshCron: '0 3 * * *' } },
);

// This mental model checks daily at 03:00 UTC and refreshes when scoped memories changed
console.log(`Operation ID: ${result2.operation_id}`);
```

### CLI

```bash
# Create a mental model and get its ID for subsequent operations
hindsight mental-model create "$BANK_ID" \
  "Project Status" \
  "What is the current project status?"
```

### Go

```go
# Section 'create-mental-model-with-trigger' not found in api/mental-models.go
```

### When to Use Automatic Refresh

| Use Case | Automatic Refresh | Why |
|----------|-------------------|-----|
| **Real-time dashboards** | ✅ Enabled | Status should always be current |
| **Policy summaries** | ❌ Disabled | Policies change infrequently, manual refresh preferred |
| **User preferences** | ✅ Enabled | Preferences evolve with new interactions |
| **FAQ answers** | ❌ Disabled | Answers are curated, should be reviewed before updating |

> **💡 Tip**
>
Enable automatic refresh for mental models that need to stay current. Disable it for curated content where you want to review changes before they go live.
---

## List Mental Models

### Python

```python
# List all mental models in a bank
mental_models = client.list_mental_models(bank_id=BANK_ID)

for mental_model in mental_models.items:
    print(f"- {mental_model.name}: {mental_model.source_query}")
```

### Node.js

```javascript
// List all mental models in a bank
const mentalModels = await client.listMentalModels(BANK_ID);

for (const mm of mentalModels.items) {
    console.log(`- ${mm.name}: ${mm.source_query}`);
}
```

### CLI

```bash
# List all mental models in a bank
hindsight mental-model list "$BANK_ID"
```

### Go

```go
# Section 'list-mental-models' not found in api/mental-models.go
```

---

## Get a Mental Model

### Python

```python
# Section 'get-mental-model' not found in api/mental-models.py
```

### Node.js

```javascript
// Get a specific mental model
const mentalModel = await client.getMentalModel(BANK_ID, mentalModelId);

console.log(`Name: ${mentalModel.name}`);
console.log(`Content: ${mentalModel.content}`);
console.log(`Last refreshed: ${mentalModel.last_refreshed_at}`);
```

### CLI

```bash
# Section 'get-mental-model' not found in api/mental-models.sh
```

### Go

```go
# Section 'get-mental-model' not found in api/mental-models.go
```

### Detail Levels

Both **List** and **Get** endpoints accept an optional `detail` query parameter that controls how much data is returned. This is useful for reducing response size, especially in agent boot flows or MCP clients where context budget is limited.

| Level | Fields Returned | Use Case |
|-------|----------------|----------|
| `metadata` | `id`, `bank_id`, `name`, `tags`, `last_refreshed_at`, `last_memory_seen_at`, `created_at` | Inventory — "what models exist?" |
| `content` | All metadata fields + `source_query`, `content`, `max_tokens`, `trigger` | Agent boot — "what do the models say?" |
| `full` (default) | All fields including `reflect_response` | Deep inspection — "what evidence backs this model?" |

```bash
# List only names and tags (smallest response)
curl "$BASE_URL/v1/default/banks/$BANK_ID/mental-models?detail=metadata"

# List with content but without provenance chains
curl "$BASE_URL/v1/default/banks/$BANK_ID/mental-models?detail=content"

# Get full detail (default behavior)
curl "$BASE_URL/v1/default/banks/$BANK_ID/mental-models/$MODEL_ID?detail=full"
```

The `detail` parameter is also available in the MCP tools:

```json
{"bank_id": "my-bank", "detail": "metadata"}
```

> **💡 Tip**
>
Use `detail=content` for agent orientation flows. It includes everything the agent needs to understand the models without the heavyweight `reflect_response` provenance chains, which can exceed 200KB for banks with many models.
### Response Fields

| Field | Type | Detail Level | Description |
|-------|------|-------------|-------------|
| `id` | string | metadata | Unique mental model ID |
| `bank_id` | string | metadata | Memory bank ID |
| `name` | string | metadata | Human-readable name |
| `tags` | list | metadata | Tags for filtering |
| `last_refreshed_at` | string | metadata | When a refresh last finished — wall-clock. Advances on every completed refresh, including one that found nothing new. Answers "have I already refreshed this?" |
| `last_memory_seen_at` | string | metadata | The newest in-scope memory the last refresh saw. Answers "is this behind the data?" — compare against `last_memory_write_at` from the stats endpoint |
| `created_at` | string | metadata | When the mental model was created |
| `source_query` | string | content | The query used to generate content |
| `content` | string | content | The generated mental model text |
| `max_tokens` | int | content | Maximum tokens for the mental model content |
| `trigger` | object | content | Trigger settings (see [Automatic Refresh](#automatic-refresh)) |
| `reflect_response` | object | full | Full reflect response including `based_on` provenance facts |

---

## Refresh a Mental Model

Re-run the source query to update the mental model with current knowledge:

### Python

```python
# Section 'refresh-mental-model' not found in api/mental-models.py
```

### Node.js

```javascript
// Refresh a mental model to update with current knowledge
const refreshResult = await client.refreshMentalModel(BANK_ID, mentalModelId);

console.log(`Refresh operation ID: ${refreshResult.operation_id}`);
```

### CLI

```bash
# Section 'refresh-mental-model' not found in api/mental-models.sh
```

### Go

```go
# Section 'refresh-mental-model' not found in api/mental-models.go
```

Refreshing is useful when:
- New memories have been retained that affect the topic
- Observations have been updated
- You want to ensure the mental model reflects current knowledge

**Refreshes coalesce.** A model has at most one refresh waiting at a time: if one is
already queued and has not started yet, this call returns *that* operation instead of
queueing an identical second one — the queued refresh reads the model as it stands when
it runs, so it already covers what you just asked for. Poll the returned `operation_id`
as usual. A refresh that is already *running* is not reused: it may have read the model
before your latest change, so a new operation is queued behind it.

---

## Troubleshoot a Refresh

When a refresh produces a document you didn't expect — nothing changed, the wrong
things changed, or delta edits didn't apply — the stored content alone doesn't say
why. Two tools report the reasoning behind a refresh.

### Dry run: preview a refresh before it happens

`POST /mental-models/{id}/dry-run-refresh` runs the real pipeline and reports what a
refresh *would* do. Nothing is written: not the content, structured document,
watermark, nor `last_refreshed_at`. Because nothing is persisted, a delta dry run
reads exactly the window the next real refresh will, and repeating it reads that same
window again.

```bash
curl -X POST "$BASE_URL/v1/default/banks/$BANK_ID/mental-models/$MODEL_ID/dry-run-refresh" \
  -H "Content-Type: application/json" -d '{}'
```

The response answers the questions the stored document can't:

| Field | What it tells you |
|-------|-------------------|
| `requested_mode` / `effective_mode` | Whether the refresh ran in the mode you configured |
| `mode_fallback_reason` | Why the delta edits weren't applied: `no_baseline_content`, `source_query_changed`, `structured_doc_unreadable`, `delta_ops_failed`, or `delta_ops_all_skipped` (every operation was rejected) |
| `scope` | The tags, match mode, and fact types that actually filtered memories — not the ones stored on the model |
| `window` | The `created_after`/`created_before` bounds read, and the watermark that would be persisted |
| `facts.retrieved` vs `facts.used` | How much retrieval returned versus how much the reflect agent judged relevant |
| `delta_operations` | The operations emitted, applied and skipped |
| `diff` | A unified diff from the stored content to the content it would write |
| `outcome` / `would_persist` | Whether a real refresh would write, keep the content, or fail |
| `warnings` | Conditions worth attention, in plain language |

The dry run takes no parameters, on purpose. It is the production refresh pipeline
with exactly two writes skipped — the content (and with it the structured document
and history entry) and the watermark that moves `last_refreshed_at`. Nothing about
how it runs can be configured, because a dry run you can configure stops predicting
the refresh it exists to predict. To preview a different configuration, change the
model with `PATCH` and run it again.

> **⚠️ Warning**
>
A dry run runs the same LLM calls a real refresh does, so it costs the same tokens
and takes the same time. It is validated and billed like a refresh.
### Keep traces: record every refresh as it happens

A dry run only explains a refresh you run yourself. Refreshes driven by
`refresh_after_consolidation` or `refresh_cron` happen with nobody watching, and by
the time you notice a bad document, the run that produced it is gone.

Setting `trigger.keep_trace` records the same reasoning on every refresh of that
model — scheduled ones included — under `reflect_response.trace`:

```bash
curl -X PATCH "$BASE_URL/v1/default/banks/$BANK_ID/mental-models/$MODEL_ID" \
  -H "Content-Type: application/json" \
  -d '{"trigger": {"mode": "delta", "keep_trace": true}}'
```

Only the latest refresh's trace is kept, and it is recorded even when a refresh
fails — which is when it matters most, since a failed refresh leaves nothing else
behind to inspect.

#### What a trace contains

The trace is deliberately shaped like a [reflect](./reflect) trace: the calls the
agent made, plus the decision specific to a refresh. It is stored on the mental
model row and re-read on every fetch, so it holds nothing that can be derived from
somewhere else.

| Field | Contents |
|-------|----------|
| `effective_mode` | Whether the run ended up `full` or `delta` |
| `mode_fallback_reason` | Why delta was requested but not applied — `no_baseline_content`, `source_query_changed`, `structured_doc_unreadable`, `delta_ops_failed`, `delta_ops_all_skipped` |
| `outcome` | `content_written`, `content_preserved_no_new_facts`, `refresh_failed_empty_candidate`, or `refresh_failed_delta_not_applied` (the edits didn't apply, so the document was kept and the refresh failed) |
| `tool_calls[]` | Per call: `tool`, the agent's `reason`, the full `input`, `result_count`, `duration_ms`, and the `iteration` it belongs to |
| `llm_calls[]` | Per call: `scope` (`agent_1`, `agent_2`, …, `final`) and `duration_ms` |
| `delta_operations` | The operations emitted in delta mode, `applied` and `skipped` |
| `usage` | Token counts across the run's LLM calls |
| `recorded_at`, `duration_ms`, `warnings[]` | When it ran, how long it took, and anything worth a human's attention |

#### What a trace deliberately omits

- **Tool outputs.** Only `result_count` is kept. Recall payloads are large and the
  trace is re-read on every model fetch, so storing them would grow the row without
  bound. A count is enough to see *that* a tool came back empty; when you need the
  raw prompts and responses, use [LLM request tracing](../monitoring), which stores
  them separately and expires them on its own retention schedule.
- **The evidence.** The facts a refresh grounded the document on already live in
  `reflect_response.based_on`, in the same shape reflect uses. The trace does not
  duplicate them.
- **The resolved scope and time window.** These describe what the model's current
  configuration reads rather than what a past run did, so they are returned by the
  dry run instead of being written on every refresh.

This keeps a stored trace small: a few kilobytes for a typical run, dominated by the
tool inputs.

---

## Clear a Mental Model

Clear a mental model's content so the next refresh performs a **full re-synthesis** from scratch, regardless of the model's trigger mode.

This is useful for delta-mode models that have accumulated drift over many incremental refreshes. Over time, small inaccuracies can compound as each delta refresh only sees new facts since the last. Clearing and then refreshing produces a clean baseline from all facts.

### Python

```python
# Section 'clear-mental-model' not found in api/mental-models.py
```

### Node.js

```javascript
// Clear a mental model's content, then refresh for a full re-synthesis
await client.clearMentalModel(BANK_ID, mentalModelId);

// Trigger a fresh full rebuild
const fullRefreshResult = await client.refreshMentalModel(BANK_ID, mentalModelId);

console.log(`Full refresh operation ID: ${fullRefreshResult.operation_id}`);
```

### CLI

```bash
# Section 'clear-mental-model' not found in api/mental-models.sh
```

### Go

```go
# Section 'clear-mental-model' not found in api/mental-models.go
```

The clear operation is synchronous and resets the content to an empty string. The model's configuration (name, source query, trigger settings) is preserved. Since the content is now empty, the next `/refresh` call will always perform a full regeneration — even if the model's trigger mode is set to `delta`.

> **💡 Tip**
>
For long-lived delta-mode mental models, consider scheduling a periodic clear + refresh (e.g. every 48 hours) to keep the content accurate while still benefiting from incremental delta updates in between.
---

## Update a Mental Model

Update the mental model's name:

### Python

```python
# Section 'update-mental-model' not found in api/mental-models.py
```

### Node.js

```javascript
// Update a mental model's metadata
const updated = await client.updateMentalModel(BANK_ID, mentalModelId, {
    name: 'Updated Team Communication Preferences',
    trigger: { refresh_after_consolidation: true },
});

console.log(`Updated name: ${updated.name}`);
```

### CLI

```bash
# Section 'update-mental-model' not found in api/mental-models.sh
```

### Go

```go
# Section 'update-mental-model' not found in api/mental-models.go
```

---

## Delete a Mental Model

### Python

```python
# Section 'delete-mental-model' not found in api/mental-models.py
```

### Node.js

```javascript
// Delete a mental model
await client.deleteMentalModel(BANK_ID, mentalModelId);
```

### CLI

```bash
# Section 'delete-mental-model' not found in api/mental-models.sh
```

### Go

```go
# Section 'delete-mental-model' not found in api/mental-models.go
```

---

## Tags and Visibility

Mental models support the same tag system as memories. When you assign tags to a mental model, those tags control both **which memories it reads** during refresh and **when it is surfaced** during reflect.

### How tags affect mental model refresh

> **⚠️ Warning**
>
Adding tags to a mental model narrows the pool of source memories its refresh can read from. If no memories carry those tags yet, refresh will return empty content (e.g. `"I cannot find any information…"`) even though direct `reflect` on the same query works. Backfill tags on the relevant memories first, or override the default via `trigger.tags_match` / `trigger.tag_groups`.
When a mental model is refreshed (manually or automatically), it runs an internal reflect call to regenerate its content. If the mental model has tags, that reflect call uses `all_strict` tag matching — meaning it will only read memories that carry **all** of the mental model's tags. Untagged memories are excluded.

```
Mental model tags: ["user:alice"]

During refresh, it reads:
  ✅ "Alice prefers async communication"     — has "user:alice"
  ✅ "Team uses Slack for announcements"      — has "user:alice" (plus other tags)
  ❌ "Company policy: no meetings on Fridays" — untagged, excluded
  ❌ "Bob dislikes long meetings"             — no "user:alice" tag
```

This means a mental model tagged `["user:alice"]` will also pick up memories tagged `["user:alice", "team"]` — extra tags on a memory don't disqualify it. Only the mental model's own tags are required to be present.

#### Overriding the default with `tags_match`

The `all_strict` default is the safest choice for single-tag models, but it filters out everything for a **multi-tag** model whose memories only carry one tag each. If a model tagged `["projects", "mental-model"]` reads from memories tagged narrowly (`["project:status"]`, `["tooling"]`, …), no single memory carries *all* the model's tags and the refresh comes back empty.

To match memories that carry **any** of the model's tags — the same default `recall` and `reflect` use — set `trigger.tags_match` to `"any"` at creation:

### Python

```python
# Override how the model's tags filter source memories on refresh.
# A tagged model defaults to "all_strict" (a memory must carry EVERY tag);
# use "any" when your memories are tagged narrowly (one topic each), so the
# refresh reads any memory carrying at least one of the model's tags.
result = client.create_mental_model(
    bank_id=BANK_ID,
    name="Current Projects",
    source_query="Which projects is the user currently working on?",
    tags=["projects", "mental-model"],
    trigger={"tags_match": "any"}
)

print(f"Operation ID: {result.operation_id}")
```

### Node.js

```javascript
// Override how the model's tags filter source memories on refresh.
// A tagged model defaults to 'all_strict' (a memory must carry EVERY tag);
// use 'any' when your memories are tagged narrowly (one topic each), so the
// refresh reads any memory carrying at least one of the model's tags.
const result3 = await client.createMentalModel(
    BANK_ID,
    'Current Projects',
    'Which projects is the user currently working on?',
    {
        tags: ['projects', 'mental-model'],
        trigger: { tagsMatch: 'any' },
    },
);

console.log(`Operation ID: ${result3.operation_id}`);
```

### CLI

```bash
# Override how the model's tags filter source memories on refresh.
# A tagged model defaults to all_strict (a memory must carry EVERY tag);
# pass --tags-match any when your memories are tagged narrowly (one topic
# each), so the refresh reads any memory carrying at least one of the tags.
hindsight mental-model create "$BANK_ID" \
  "Current Projects" \
  "Which projects is the user currently working on?" \
  --tags projects,mental-model \
  --tags-match any
```

### Go

```go
# Section 'create-mental-model-tags-match' not found in api/mental-models.go
```

The MCP `create_mental_model` tool exposes the same option as a top-level `tags_match` argument. Available modes are `any`, `all`, `any_strict`, `all_strict`, and `exact` — see the [Recall tags reference](./recall#tags) for their exact semantics.

### How tags affect mental model lookup during reflect

When you call `reflect` with tags, those same tags are used to filter which mental models the agent can see. A mental model is visible only if its tags overlap with the tags on the reflect request.

For more details on tag matching modes (`any`, `any_strict`, `all`, `all_strict`) and worked examples, see the [Recall tags reference](./recall#tags).

### Listing mental model tags

`GET /v1/default/banks/{bank_id}/tags` accepts a `source` query parameter that selects which tag space to enumerate:

- `source=memories` *(default)* — tags attached to memory units.
- `source=mental_models` — tags attached to mental models in this bank.

Use the `mental_models` source to populate autocomplete or filter UIs over mental-model tags, distinct from the (typically larger) memory tag set.

---

## History

Every time a mental model's content changes (via refresh or manual update), the previous version is saved with a timestamp. You can retrieve the full change log with the history endpoint:

### Python

```python
# Section 'get-mental-model-history' not found in api/mental-models.py
```

### Node.js

```javascript
// Get the change history of a mental model
const history = await client.getMentalModelHistory(BANK_ID, mentalModelId);

for (const entry of history) {
    console.log(`Changed at: ${entry.changed_at}`);
    console.log(`Previous content: ${entry.previous_content}`);
}
```

### CLI

```bash
# Section 'get-mental-model-history' not found in api/mental-models.sh
```

### Go

```go
# Section 'get-mental-model-history' not found in api/mental-models.go
```

### Response

The endpoint returns a list of history entries, most recent first:

| Field | Type | Description |
|-------|------|-------------|
| `previous_content` | string \| null | The content before this change (`null` if not available) |
| `changed_at` | string | ISO 8601 timestamp of when the change occurred |

Each entry captures the **content before the change** and when it happened. The current content is returned by the standard [Get a Mental Model](#get-a-mental-model) endpoint.

> **📝 Note**
>
History tracking is enabled by default. Set `HINDSIGHT_API_ENABLE_MENTAL_MODEL_HISTORY=false` to disable it, and `HINDSIGHT_API_MENTAL_MODEL_HISTORY_MAX_ENTRIES` to change how many versions are kept per model (default `50`).
---

## Use Cases

| Use Case | Example |
|----------|---------|
| **FAQ Answers** | Pre-compute answers to common customer questions |
| **Onboarding Summaries** | "What should new team members know?" |
| **Status Reports** | "What's the current project status?" refreshed weekly |
| **Policy Summaries** | "What are our security policies?" |

---

## Next Steps

- [**Reflect**](./reflect) — How the agentic loop uses mental models
- [**Observations**](../observations.md) — How knowledge is consolidated
- [**Operations**](./operations) — Track async mental model creation
