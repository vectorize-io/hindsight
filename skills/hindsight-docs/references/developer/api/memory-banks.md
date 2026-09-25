
# Memory Banks

Memory banks are isolated containers that store all memory-related data for a specific context or use case.

{/* Import raw source files */}

## What is a Memory Bank?

A memory bank is a complete, isolated storage unit containing:

- **Memories** — Facts and information retained from conversations
- **Documents** — Files and content indexed for retrieval
- **Entities** — People, places, concepts extracted from memories
- **Relationships** — Connections between entities in the knowledge graph
- **Directives** — Hard rules the agent must follow during reflect operations

Banks are completely isolated from each other — memories stored in one bank are not visible to another.

You don't need to pre-create a bank. Hindsight will automatically create it with default settings when you first use it.

Only *writes* create a bank — retaining, updating its profile, changing its config. **Reads of a
bank that does not exist return `404`**, so a typo'd, renamed or deleted `bank_id` is reported
rather than answered with empty results. That matters if you are monitoring a bank: `GET .../stats`
on a missing bank fails loudly instead of returning zeroed counters that look like a healthy,
empty bank.

> **💡 Prerequisites**
>
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
## Creating a Memory Bank

### Python

```python
client.create_bank(bank_id="my-bank")
```

### Node.js

```javascript
await client.createBank('my-bank');
```

### CLI

```bash
hindsight bank create my-bank
```

### Go

```go
client.BanksAPI.CreateOrUpdateBank(ctx, "my-bank").
	CreateBankRequest(hindsight.CreateBankRequest{}).Execute()
```

## Aliases {#aliases}

An **alias** is an extra id a bank answers to. Every endpoint that accepts a bank id accepts its aliases too, so an alias is a second way in to the same bank — nothing is copied, moved or duplicated.

This exists because a bank's id is the key on all of its data, so changing it means rewriting every row and cutting all clients over at once (that is [`rename-bank`](../admin-cli.md#rename-bank), and it needs downtime). An alias lets you do the same migration in phases: add the new id, move clients across a few at a time while both ids work, and stop when nothing calls the old one.

```
POST   /v1/default/banks/{bank_id}/aliases             {"alias": "new-id"}
GET    /v1/default/banks/{bank_id}/aliases
PATCH  /v1/default/banks/{bank_id}/aliases/{alias}     {"primary": true}
DELETE /v1/default/banks/{bank_id}/aliases/{alias}
```

### Showing a bank under an alias

Once callers have moved, the control plane can present the bank under the new id instead of the one it was created with. `PATCH` with `primary: true` promotes an alias; `primary: false` returns the bank to showing its own id without removing the alias.

This is **display only**. `bank_id` remains the bank's identity: authorisation, metering, exports and audit logs all continue to use it, and it is what every response reports. The control plane shows the real id alongside the alias so the two are never confused.

- **Optional, with no default.** A bank's own id is not an alias, so "not shown under an alias" is the normal state.
- **At most one.** Promoting an alias demotes whichever one was shown before, in a single transaction.
- **It cannot outlive the alias.** Deleting the promoted alias returns the bank to showing its own id.

**What to know:**

- **A bank can have several aliases**, so more than one old id can be retired at a time.
- **A name is unique across banks and aliases.** Adding one that already names a bank or another alias returns `409`, so an alias can never reach two banks.
- **An alias never replaces the bank's own id.** The real id keeps working and is what every response reports, including responses to requests made through an alias. Removing a bank's real id is not possible — that is what `rename-bank` is for.
- **Deleting an alias only closes that door.** The bank and its memories are untouched, and callers still using the removed id get exactly what they got before it existed.
- **Deleting a bank removes its aliases**, freeing those names for reuse.
- **Aliases are not carried by export, import or clone.** They describe which ids reach a bank on *this* deployment, so a copy does not inherit them — add them to the target deliberately.
- **A change takes a few seconds to reach every API replica.** See [`HINDSIGHT_API_BANK_ALIAS_CACHE_TTL_SECONDS`](../configuration.md#bank-alias-cache).

## Bank Configuration

Each memory bank can be configured independently per operation. Configuration can be set via the [bank config API](#updating-configuration), the Control Plane UI, or [server-wide environment variables](../configuration.md).

### retain_mission {#retain-configuration}

A plain-language description of what this bank should pay attention to during extraction. The mission is injected into the extraction prompt alongside the built-in rules — it steers focus without replacing the extraction logic.

```
e.g. Always include technical decisions, API design choices, and architectural trade-offs.
     Ignore meeting logistics, greetings, and social exchanges.
```

Works alongside any extraction mode. Leave blank for general-purpose extraction.

### retain_extraction_mode

Controls how aggressively facts are extracted:

| Mode | Description |
|------|-------------|
| `concise` *(default)* | Selective — only facts worth remembering long-term |
| `verbose` | Captures more detail per fact; slower and uses more tokens |
| `custom` | Write your own extraction rules via `retain_custom_instructions` |
| `verbatim` | Stores each chunk's original text as one memory; the LLM extracts metadata such as entities and dates |
| `chunks` | Stores each chunk as one memory without an LLM call; only caller-provided entities are available |

### retain_custom_instructions

Only active when `retain_extraction_mode` is `custom`. Replaces the built-in extraction rules entirely with your own instructions.

### retain_chunk_size

Maximum number of characters per chunk when splitting content for fact extraction. Larger chunks mean fewer LLM calls but may reduce extraction quality on long inputs; smaller chunks improve granularity at the cost of more calls.

Default: `3000`

### retain_structured_chunk_size

Maximum number of characters for a single JSONL line or conversation turn to keep whole when it exceeds `retain_chunk_size`. When unset, the limit is exactly `retain_chunk_size`; set a larger value for structured logs or chat transcripts where splitting a single record would lose useful context.

Default: unset, which uses `retain_chunk_size`

See [Retain configuration](../configuration.md#retain) for environment variable names and defaults.

### entity_labels {#entity-labels}

Defines a controlled vocabulary of `key:value` classification labels extracted at retain time and stored as entities. Because labels become entities, they automatically link memories in the knowledge graph (two memories with `pedagogy:scaffolding` are linked), improve semantic and BM25 retrieval, and optionally filter memories via the standard `tags`/`tags_match` API when `tag: true` is set on a group.

Each entry in `entity_labels` is a **label group** — one classification dimension:

```json
{
  "entity_labels": [
    {
      "key": "engagement",
      "description": "Student engagement level during the session",
      "type": "value",
      "optional": true,
      "values": [
        { "value": "active",  "description": "Student is actively participating" },
        { "value": "passive", "description": "Student is listening but not participating" }
      ]
    },
    {
      "key": "pedagogy",
      "description": "Teaching strategies used",
      "type": "multi-values",
      "values": [
        { "value": "scaffolding",          "description": "Breaking complex tasks into smaller steps" },
        { "value": "direct_instruction",   "description": "Explicit explanation by the teacher" },
        { "value": "socratic_questioning", "description": "Guiding through questions rather than answers" }
      ]
    }
  ]
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `key` | — | Label group identifier. Becomes the prefix in `key:value` entities (or `key:field:value` for `"map"`). |
| `description` | `""` | Shown to the LLM to guide label assignment. |
| `type` | `"value"` | `"value"` → pick one enum value; `"multi-values"` → pick multiple; `"text"` → free-form string; `"multi-text"` → any number of free-form strings; `"map"` → structured group with named fields. |
| `values` | `[]` | Allowed values for `"value"` and `"multi-values"` types. Ignored for `"text"`, `"multi-text"` and `"map"`. |
| `fields` | `{}` | Field definitions for `"map"` types. Each field is itself typed (`"text"`, `"multi-text"`, `"value"`, `"multi-values"`, or nested `"map"`). Ignored for non-map types. |
| `optional` | `true` | When `true` the LLM may skip the label if not applicable. When `false` the LLM must always assign a value. Has no effect on `"multi-values"` or `"multi-text"` groups (always optional). |
| `tag` | `false` | When `true`, extracted `key:value` labels are also written as tags on the memory unit, enabling filtering via `tags`/`tags_match` in recall/reflect. |

**Enum groups** (`type: "value"` or `type: "multi-values"`): the LLM picks from the predefined `values` list; anything outside the list is silently dropped. Vocabulary stays stable and graph links stay tight. Use `"multi-values"` when a fact can belong to several values at once.

**Free-text groups** (`type: "text"`): the LLM writes any string. Use the `description` field to provide examples and guidance. Graph clustering is less reliable than with enum groups because the model may phrase the same concept differently across sessions.

```json
{
  "key": "topic",
  "description": "Specific subject being discussed. Examples: algebra, quadratic equations, geometry.",
  "type": "text",
  "optional": true,
  "values": []
}
```

**Open-vocabulary groups** (`type: "multi-text"`): like `"text"`, but the LLM writes a *list* of strings instead of one — so a single fact can carry several values that nobody enumerated in the bank config. Use it when the interesting values only exist in the content: the names a thing is known by (canonical name plus abbreviations, acronyms and alternative spellings), ticket references a fact cites, product codes or SKUs. Each value becomes its own `key:value` entity, and with `tag: true` each is written as a tag, so a bank can derive a classification from content and then filter on it at recall without the caller supplying the vocabulary.

```json
{
  "key": "name",
  "description": "Every name the subject of this fact is known by — canonical name plus abbreviations, acronyms, short forms and alternative spellings.",
  "type": "multi-text",
  "tag": true
}
```

Retaining *"We deploy services to a Kubernetes cluster on EKS — the team usually just says k8s, or kube"* yields the entities and tags `name:kubernetes`, `name:k8s` and `name:kube`, so `{"tags": ["name:k8s"], "tags_match": "any_strict"}` finds the fact. A `"text"` group would keep only one of the three.

**Map groups** (`type: "map"`): defines a structured entity type with named fields. Each field is itself typed (`"text"`, `"multi-text"`, `"value"`, `"multi-values"`, or nested `"map"`) so you can describe rich entities like a person with name, role, and organization. Each extracted field is stored as a flat `key:field:value` entity string (e.g. `person:name:Alice`), reusing the existing entity storage with no schema changes — so map fields participate in the knowledge graph and retrieval the same way single-value labels do.

```json
{
  "key": "person",
  "description": "A person mentioned in the text",
  "type": "map",
  "fields": {
    "name":         { "type": "text", "description": "Full name of the person" },
    "role":         { "type": "text", "description": "Job title or role" },
    "organization": { "type": "text", "description": "Company or organization" }
  }
}
```

**How label entities resolve.** Regular entities resolve fuzzily so close name variants merge ("Alice" / "Alice Chen"). Label entities are different: their canonical names are user-defined, so two similar-looking values (`use:use-001` / `use:use-002`) must stay distinct. They therefore resolve by **exact match only** and are stored with `entity_kind = "label"`, which keeps them out of fuzzy name matching entirely — a free-text label group accumulating thousands of similar values doesn't slow down resolution of the bank's regular entities. The classification is fixed when the entity is first stored; removing a label group later doesn't reclassify its existing entities.

### entities_allow_free_form

By default, entity labels are extracted **alongside** regular named entities (people, places, concepts). Set to `false` to disable free-form extraction so only label entities are stored:

```json
{
  "entity_labels": [...],
  "entities_allow_free_form": false
}
```

### enable_observations {#observations-configuration}

Toggles observation consolidation on or off. When `false`, no consolidation runs for this bank — neither automatic nor manual. Defaults to `true` when the observations feature is enabled on the server.

### enable_auto_consolidation

Controls whether consolidation runs automatically after retain, delete, and update operations. When `false`, consolidation only runs when explicitly triggered via the [consolidate endpoint](../observations.md#trigger-consolidation). Defaults to `true`.

This is useful when you want full control over consolidation timing — for example, batching many retains before consolidating, or running [targeted consolidation](../observations.md#targeted-consolidation) for specific scopes only.

### observations_mission

Defines what this bank should synthesise into durable observations. Replaces the built-in consolidation rules entirely — leave blank to use the server default.

```
e.g. Observations are stable facts about people and projects.
     Always include preferences, skills, and recurring patterns.
     Ignore one-off events and ephemeral state.
```

### consolidation_llm_batch_size

Number of facts sent to the LLM in a single consolidation call. Higher values reduce LLM calls and improve throughput at the cost of larger prompts. Set to `1` to disable batching. Leave unset to use the server default (`8`).

### consolidation_source_facts_max_tokens

Total token budget for source facts included with observations in the consolidation prompt. Source facts give the LLM evidence to compare new facts against existing observations. `-1` = unlimited. Leave unset to use the server default (`-1`).

### consolidation_source_facts_max_tokens_per_observation

Per-observation token cap for source facts in the consolidation prompt. Each observation independently gets at most this many tokens of source facts, preventing a single observation with many source facts from consuming the entire budget. `-1` = unlimited. Leave unset to use the server default (`256`).

See [Observations configuration](../configuration.md#observations) for environment variable names and defaults.

### reflect_mission

A first-person narrative that provides identity and framing context for `reflect`. The agent uses this to ground its reasoning and apply a consistent perspective.

```
e.g. You are a senior engineering assistant.
     Always ground answers in documented decisions and rationale.
     Ignore speculation. Be direct and precise.
```

### disposition_skepticism

How skeptical vs trusting the bank is when evaluating claims during `reflect`. Scale 1–5.

### Python

```python
client.create_bank(bank_id="architect-bank")
client.update_bank_config(
    "architect-bank",
    reflect_mission="You're a senior software architect - keep track of system designs, "
            "technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
    disposition_skepticism=4,   # Questions new technologies
    disposition_literalism=4,   # Focuses on concrete specs
    disposition_empathy=2,      # Prioritizes technical facts
)
```

### Node.js

```javascript
await client.createBank('architect-bank');
await client.updateBankConfig('architect-bank', {
    reflectMission: "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns.",
    dispositionSkepticism: 4,   // Questions new technologies
    dispositionLiteralism: 4,   // Focuses on concrete specs
    dispositionEmpathy: 2,      // Prioritizes technical facts
});
```

### CLI

```bash
hindsight bank create architect-bank \
  --mission "You're a senior software architect - keep track of system designs, technology decisions, and architectural patterns. Prefer simplicity over cutting-edge." \
  --skepticism 4 \
  --literalism 4 \
  --empathy 2
```

### Go

```go
client.BanksAPI.CreateOrUpdateBank(ctx, "architect-bank").
	CreateBankRequest(hindsight.CreateBankRequest{
		ReflectMission: *hindsight.NewNullableString(hindsight.PtrString(
			"You're a senior software architect - keep track of system designs, " +
				"technology decisions, and architectural patterns. Prefer simplicity over cutting-edge.",
		)),
		DispositionSkepticism: *hindsight.NewNullableInt32(hindsight.PtrInt32(4)),
		DispositionLiteralism: *hindsight.NewNullableInt32(hindsight.PtrInt32(4)),
		DispositionEmpathy:    *hindsight.NewNullableInt32(hindsight.PtrInt32(2)),
	}).Execute()
```

| Value | Behaviour |
|-------|-----------|
| `1` | Trusting — accepts information at face value |
| `3` *(default)* | Balanced |
| `5` | Skeptical — questions and doubts claims |

### disposition_literalism

How literally to interpret information during `reflect`. Scale 1–5.

| Value | Behaviour |
|-------|-----------|
| `1` | Flexible — reads between the lines, considers context |
| `3` *(default)* | Balanced |
| `5` | Literal — takes things exactly as stated |

### disposition_empathy

How much to weight emotional context when reasoning during `reflect`. Scale 1–5.

| Value | Behaviour |
|-------|-----------|
| `1` | Detached — focuses on facts and logic |
| `3` *(default)* | Balanced |
| `5` | Empathetic — considers emotional context |

> **ℹ️ Info**
>
Disposition traits and `reflect_mission` only affect the `reflect` operation. `retain_mission` and `observations_mission` are separate per-operation settings.
### mcp_enabled_tools

An allowlist of MCP tool names that are enabled for this bank. When set, only the listed tools can be invoked; any tool not in the list returns an error (tools still appear in the MCP tools list for protocol compatibility). Set to `null` (or omit) to allow all tools.

```json
["recall", "reflect"]
```

Available tool names: `retain`, `recall`, `reflect`, `list_banks`, `create_bank`, `list_mental_models`, `get_mental_model`, `create_mental_model`, `update_mental_model`, `delete_mental_model`, `refresh_mental_model`, `list_directives`, `create_directive`, `delete_directive`, `list_memories`, `get_memory`, `list_documents`, `get_document`, `delete_document`, `list_operations`, `get_operation`, `cancel_operation`, `list_tags`, `get_bank`, `get_bank_stats`, `update_bank`, `delete_bank`, `clear_memories`, `get_knowledge_base_tree`, `search_knowledge_base`, `get_knowledge_page`, `create_knowledge_folder`, `create_knowledge_page`, `update_knowledge_node`, `delete_knowledge_node`.

### llm_gemini_safety_settings

Controls content filtering thresholds for Gemini and VertexAI providers. Accepts a list of safety setting objects in the [Google AI safety settings format](https://ai.google.dev/api/generate-content#v1beta.SafetySetting). When `null` (default), Gemini's built-in safety defaults are used.

```json
[
  {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}
]
```

Only applies when `HINDSIGHT_API_LLM_PROVIDER` is `gemini` or `vertexai`.

### recall_budget_function {#recall-budget-configuration}

Selects how the [`recall` request's `budget` parameter](./recall) (`low` / `mid` / `high`) maps to the internal `thinking_budget` integer used by every retrieval method (semantic, BM25, graph, temporal). Two functions are supported:

| Function | Behaviour |
|----------|-----------|
| `fixed` *(default)* | `thinking_budget = recall_budget_fixed_<level>` — independent of `max_tokens`. Preserves legacy behavior. |
| `adaptive` | `thinking_budget = round(max_tokens * recall_budget_adaptive_<level>)`, clamped to `[recall_budget_min, recall_budget_max]`. Retrieval breadth scales with the requested output size. |

```json
{
  "recall_budget_function": "adaptive",
  "recall_budget_adaptive_low": 0.05,
  "recall_budget_adaptive_mid": 0.1,
  "recall_budget_adaptive_high": 0.3,
  "recall_budget_min": 30,
  "recall_budget_max": 1500
}
```

### recall_budget_fixed_low / recall_budget_fixed_mid / recall_budget_fixed_high

When `recall_budget_function` is `fixed` (the default), these positive integers are used directly as the per-method retrieval limit for each `budget` level. Defaults: `100` / `300` / `1000` — exactly matching the legacy hardcoded mapping.

### recall_budget_adaptive_low / recall_budget_adaptive_mid / recall_budget_adaptive_high

When `recall_budget_function` is `adaptive`, these positive ratios multiply the request's `max_tokens` to derive the per-method retrieval limit. Defaults: `0.025` / `0.075` / `0.25` — chosen to roughly match the fixed defaults at `max_tokens = 4096`.

### recall_budget_min / recall_budget_max

Floor and ceiling applied to the result of the adaptive function (after the ratio multiplication). Both must be positive integers and `min ≤ max`. Defaults: `20` / `2000`.

See [Recall budget mapping](../configuration.md#recall-budget-mapping) for environment variable names and full defaults.

### memory_defense {#memory_defense}

Per-bank Memory Defense policy. Defaults to absent (Memory Defense disabled on this bank).

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Master switch. |
| `default_action` | `allow`\|`redact`\|`quarantine`\|`block` | `allow` | Fallback action when no rule matches. |
| `protected_tag_namespaces` | `list[str]` | `[]` | Writes with tags in these namespaces (`ns:*`) are subject to the `protected_key` detector. |
| `immutable_tag_namespaces` | `list[str]` | `[]` | Writes to these namespaces are blocked. |
| `rules` | `list[Rule]` | `[]` | Detector-to-action mappings (see below). |
| `detector_overrides` | `dict` | `{}` | Per-detector tuning (e.g. `size_anomaly.max_size`). |

`Rule` shape:

| Field | Required | Description |
|---|---|---|
| `on` | yes | Detector name (`prompt_injection`, `sensitive_data`, `protected_key`, `immutable_key`, `size_anomaly`) or `*` for any. |
| `action` | yes | One of `allow`, `redact`, `quarantine`, `block`. |
| `min_severity` | no | Minimum severity (`low`, `medium`, `high`, `critical`) for the rule to fire. Defaults to `low`. |

Invalid policies are rejected on PATCH with HTTP 422.

See the [Memory Defense guide](../memory-defense/index.md) for usage examples.

---

## Previewing Prompts

A mission only means something once you can see the prompt it lands in. `POST /v1/default/banks/{bank_id}/prompts/preview` renders the exact messages `retain`, `consolidation` or `reflect` would send for a bank — no LLM call, no reads, nothing stored.

The operation is the whole request. Everything that shapes the prompt is read from the bank — its resolved config, profile and directives — and the runtime data an operation would be given is a fixed bracketed placeholder:

### Python

```python
from hindsight_client_api.models import PromptPreviewRequest

preview = await client.banks.preview_prompt("my-bank", PromptPreviewRequest(operation="retain"))
for message in preview.messages:
    print(message.role, len(message.blocks))
```

### Node.js

```javascript
const { data: preview } = await sdk.previewPrompt({
    client: apiClient,
    path: { bank_id: 'my-bank' },
    body: { operation: 'retain' },
});
for (const message of preview.messages) console.log(message.role, message.blocks.length);
```

### CLI

```bash
curl --fail-with-body -X POST "$HINDSIGHT_URL/v1/default/banks/my-bank/prompts/preview" \
  -H "Content-Type: application/json" \
  -d '{"operation": "retain"}'
```

### Go

```go
preview, _, err := client.BanksAPI.PreviewPrompt(ctx, "my-bank").
	PromptPreviewRequest(hindsight.PromptPreviewRequest{Operation: hindsight.PtrString("retain")}).
	Execute()
for _, message := range preview.Messages {
	fmt.Println(message.Role, len(message.Blocks))
}
```

For `retain`, add `"strategy": "<name>"` to render under one of the bank's named retain strategies. Omit it and the bank's `retain_default_strategy` applies — exactly as it does for a retain that names none — so what you see is what retain would send. The response echoes the strategy that applied in `strategy`, and lists the bank's strategy names in `strategies` so a picker needs no second call.

There is deliberately nothing to override. A preview answers "what does this bank send"; letting a caller pass its own mission or sample text only moves that question somewhere the bank cannot answer it. To try a candidate value, save it and look again — the response says which settings are editable.

The response carries the messages in send order, each broken into the blocks it is built from:

```json
{
  "messages": [
    {
      "role": "system",
      "blocks": [
        {
          "text": "Extract SIGNIFICANT facts from text...",
          "source": "builtin",
          "field": "retain_extraction_mode",
          "section": "",
          "heading": "Selectivity",
          "active": true,
          "value": "concise",
          "kind": "choice",
          "choices": ["concise", "verbose", "verbatim", "chunks", "custom"],
          "editable": true
        },
        {
          "text": "",
          "source": "config",
          "field": "retain_custom_instructions",
          "section": "",
          "heading": "",
          "active": false,
          "kind": "text",
          "editable": true
        }
      ]
    },
    { "role": "user", "blocks": [] }
  ],
  "response_schema": { "type": "object", "properties": { "facts": {} } }
}
```

- **`messages`** — the request, system first, exactly as the model receives it.
- **`blocks`** — the **active** blocks of a message concatenate back to its text exactly: nothing dropped, duplicated or reordered. `source` says what produced each: `config` for a setting you can change, `builtin` for Hindsight's own wording. The runtime data an operation is given is not a block — it is a hole in the text, marked inline with `«…»`.
- **Inactive blocks** (`active: false`) have no text. They mark a setting that is switched off, at the point where it *would* land — so a mission you have not written yet is still visible where it would go.
- **Identifying a block** — whichever of these applies, in order: `field` is the config field behind it; `section` is a slug for a part no single field owns (`bank_identity`, `disposition`, `directives`); `heading` is the section heading the prompt text itself carries there. All three are machine values. The response carries **no display copy** — what a block is called, and what turning a switched-off one on would do, is for the client to say in the language it is running in.
- **`editable`** — false for server-level fields such as `llm_output_language`, which shape the prompt but cannot be overridden per bank.

**Both messages are returned on purpose.** For `retain` and `consolidation` the mission travels in the *user* message, not the system prompt: keeping the system prompt identical for every bank lets one provider-side prompt cache serve them all, which is a large cost saving on high-volume ingestion. Only `reflect` puts its mission in the system prompt.

### When there is no prompt

`chunks` extraction mode stores each chunk verbatim and never calls an LLM. There is no prompt to show, so `messages` is empty and `skipped_reason` explains why:

```json
{
  "operation": "retain",
  "messages": [],
  "skipped_reason": "Chunks mode stores each chunk verbatim as its own memory and never calls an LLM, ..."
}
```

In the Control Plane, each Mission field on the bank **Configuration** tab has a **Preview prompt** button, and each block's setting can be edited there and saved to the bank.

> **💡 Tip**
>
[Dry-run extraction](memories.md) is the paid counterpart: it spends a real LLM call to show what the same configuration actually *extracts*. It resolves its config the same way, strategy included, so the two agree. The Control Plane pairs them in one prompt tester.
## Updating Configuration

Bank configuration fields (retain mission, extraction mode, observations mission, etc.) are managed via a **separate config API**, not the `create_bank` call. This lets you change operational settings independently from the bank's identity and disposition.

### Setting Configuration Overrides

### Python

```python
client.update_bank_config(
    "my-bank",
    retain_mission="Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.",
    retain_extraction_mode="verbose",
    observations_mission="Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.",
    disposition_skepticism=4,
    disposition_literalism=4,
    disposition_empathy=2,
)
```

### Node.js

```javascript
await client.updateBankConfig('my-bank', {
    retainMission: 'Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges.',
    retainExtractionMode: 'verbose',
    observationsMission: 'Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events.',
    dispositionSkepticism: 4,
    dispositionLiteralism: 4,
    dispositionEmpathy: 2,
});
```

### CLI

```bash
hindsight bank set-config my-bank \
  --retain-mission "Always include technical decisions, API design choices, and architectural trade-offs. Ignore meeting logistics and social exchanges." \
  --retain-extraction-mode verbose \
  --observations-mission "Observations are stable facts about people and projects. Always include preferences, skills, and recurring patterns. Ignore one-off events." \
  --disposition-skepticism 4 \
  --disposition-literalism 4 \
  --disposition-empathy 2
```

### Go

```go
client.BanksAPI.UpdateBankConfig(ctx, "my-bank").
	BankConfigUpdate(hindsight.BankConfigUpdate{
		Updates: map[string]interface{}{
			"retain_mission": "Always include technical decisions, API design choices, and architectural trade-offs. " +
				"Ignore meeting logistics and social exchanges.",
			"retain_extraction_mode": "verbose",
			"observations_mission": "Observations are stable facts about people and projects. " +
				"Always include preferences, skills, and recurring patterns. Ignore one-off events.",
			"disposition_skepticism": 4,
			"disposition_literalism": 4,
			"disposition_empathy":    2,
		},
	}).Execute()
```

You can update any subset of fields — only the keys you provide are changed.

### Reading the Current Configuration

### Python

```python
# Returns resolved config (server defaults merged with bank overrides) and the raw overrides
data = client.get_bank_config("my-bank")
# data["config"]     — full resolved configuration
# data["overrides"]  — only fields overridden at the bank level
```

### Node.js

```javascript
// Returns resolved config (server defaults merged with bank overrides) and the raw overrides
const { config, overrides } = await client.getBankConfig('my-bank');
// config    — full resolved configuration
// overrides — only fields overridden at the bank level
```

### CLI

```bash
# Returns resolved config (server defaults merged with bank overrides)
hindsight bank config my-bank

# Show only bank-specific overrides
hindsight bank config my-bank --overrides-only
```

### Go

```go
// Returns resolved config (server defaults merged with bank overrides) and the raw overrides
result, _, _ := client.BanksAPI.GetBankConfig(ctx, "my-bank").Execute()
// result.Config     — full resolved configuration
// result.Overrides  — only fields overridden at the bank level
fmt.Println("Config keys:", len(result.GetConfig()))
```

The response distinguishes:
- **`config`** — the fully resolved configuration (server defaults merged with bank overrides)
- **`overrides`** — only the fields explicitly overridden for this bank

### Resetting to Defaults

### Python

```python
# Remove all bank-level overrides, reverting to server defaults
client.reset_bank_config("my-bank")
```

### Node.js

```javascript
// Remove all bank-level overrides, reverting to server defaults
await client.resetBankConfig('my-bank');
```

### CLI

```bash
# Remove all bank-level overrides, reverting to server defaults
hindsight bank reset-config my-bank -y
```

### Go

```go
// Remove all bank-level overrides, reverting to server defaults
client.BanksAPI.ResetBankConfig(ctx, "my-bank").Execute()
```

This removes all bank-level overrides. The bank reverts to server-wide defaults (set via environment variables).

You can also update configuration directly from the Control Plane UI — navigate to a bank and open the **Configuration** tab.

---

## Directives

Directives are hard rules that the agent must follow during [reflect](./reflect) operations. Unlike disposition traits which influence *how* the agent reasons, directives are explicit instructions that are enforced whenever they are in scope (see [Directive Scope and Tags](#directive-scope-and-tags)).

> **ℹ️ Info**
>
Directives only affect the `reflect` operation. They are injected into prompts and the agent is required to comply with them in all responses.
### When to Use Directives

Use directives for rules that must never be violated:

- **Language/style constraints**: "Always respond in formal English"
- **Privacy rules**: "Never share personal data with third parties"
- **Domain constraints**: "Prefer conservative investment recommendations"
- **Behavioral guardrails**: "Always cite sources when making claims"

### Directive Scope and Tags

Directives can carry `tags`, and those tags scope **when** a directive is applied during `reflect` — mirroring how tags scope memories:

- **Untagged directives always apply**, on every `reflect`.
- **Tagged directives apply only when the `reflect` request includes matching tags** (using the request's `tags_match` mode). A `reflect` call with no tags applies only the untagged directives.

To apply **every** active directive regardless of tags, set `apply_all_directives: true` on the `reflect` request. This ignores tag scope for directives (untagged and tagged alike are enforced) and is useful when an operator keeps tagged directives for organization but wants all of them enforced on an untagged reflection.

### Creating Directives

### Python

```python
# Create a directive (hard rule for reflect)
directive = client.create_directive(
    bank_id=BANK_ID,
    name="Formal Language",
    content="Always respond in formal English, avoiding slang and colloquialisms."
)

print(f"Created directive: {directive.id}")
```

### Node.js

```javascript
// Create a directive (hard rule for reflect)
const directive = await client.createDirective(
    BANK_ID,
    'Formal Language',
    'Always respond in formal English, avoiding slang and colloquialisms.'
);

console.log(`Created directive: ${directive.id}`);
```

### CLI

```bash
# Create a directive (hard rule for reflect)
hindsight directive create "$BANK_ID" \
  "Formal Language" \
  "Always respond in formal English, avoiding slang and colloquialisms."
```

### Go

```go
// Create a directive (hard rule for reflect)
directive, _, _ := client.DirectivesAPI.CreateDirective(ctx, bankID).
	CreateDirectiveRequest(hindsight.CreateDirectiveRequest{
		Name:    "Formal Language",
		Content: "Always respond in formal English, avoiding slang and colloquialisms.",
	}).Execute()

fmt.Printf("Created directive: %s\n", directive.GetId())
```

### Listing Directives

### Python

```python
# List all directives in a bank
directives = client.list_directives(bank_id=BANK_ID)

for d in directives.items:
    print(f"- {d.name}: {d.content[:50]}...")
```

### Node.js

```javascript
// List all directives in a bank
const directives = await client.listDirectives(BANK_ID);

for (const d of directives.items) {
    console.log(`- ${d.name}: ${d.content.slice(0, 50)}...`);
}
```

### CLI

```bash
# List all directives in a bank
hindsight directive list "$BANK_ID"
```

### Go

```go
// List all directives in a bank
directives, _, _ := client.DirectivesAPI.ListDirectives(ctx, bankID).Execute()

for _, d := range directives.GetItems() {
	content := d.GetContent()
	if len(content) > 50 {
		content = content[:50]
	}
	fmt.Printf("- %s: %s...\n", d.GetName(), content)
}
```

### Updating Directives

### Python

```python
# Update a directive (e.g., disable without deleting)
updated = client.update_directive(
    bank_id=BANK_ID,
    directive_id=directive_id,
    is_active=False
)

print(f"Directive active: {updated.is_active}")
```

### Node.js

```javascript
// Update a directive (e.g., disable without deleting)
const updated = await client.updateDirective(BANK_ID, directiveId, {
    isActive: false
});

console.log(`Directive active: ${updated.is_active}`);
```

### CLI

```bash
# Update a directive (e.g., disable without deleting)
hindsight directive update "$BANK_ID" "$DIRECTIVE_ID" --is-active false
```

### Go

```go
// Update a directive (e.g., disable without deleting)
isActiveFalse := false
updated, _, _ := client.DirectivesAPI.UpdateDirective(ctx, bankID, directiveID).
	UpdateDirectiveRequest(hindsight.UpdateDirectiveRequest{
		IsActive: *hindsight.NewNullableBool(&isActiveFalse),
	}).Execute()

fmt.Printf("Directive active: %v\n", updated.GetIsActive())
```

### Deleting Directives

### Python

```python
# Delete a directive
client.delete_directive(
    bank_id=BANK_ID,
    directive_id=directive_id
)
```

### Node.js

```javascript
// Delete a directive
await client.deleteDirective(BANK_ID, directiveId);
```

### CLI

```bash
# Delete a directive
hindsight directive delete "$BANK_ID" "$DIRECTIVE_ID" -y
```

### Go

```go
// Delete a directive
client.DirectivesAPI.DeleteDirective(ctx, bankID, directiveID).Execute()
```

### Directives vs Disposition

| Aspect | Directives | Disposition |
|--------|------------|-------------|
| **Nature** | Hard rules, must be followed | Soft influence on reasoning style |
| **Enforcement** | Strict — responses are rejected if violated | Flexible — shapes interpretation |
| **Use case** | Compliance, guardrails, constraints | Personality, character, tone |
| **Example** | "Never recommend specific stocks" | High skepticism: questions claims |

---

## Bank transfer (export & import)

Move a bank — or just its documents — between banks and instances **without re-running the LLM**. One archive format, one pair of endpoints, and three flags that decide what travels:

| Flag | Default | What it carries |
|------|---------|-----------------|
| `include_data` | `true` | Documents, raw chunks, extracted facts, consolidated observations, entities and links, attachments **and their bytes**, the curation archive of invalidated facts, the operations log, the maintenance queues — and what the bank synthesized from all of it: mental models, their refresh history, and the knowledge-page tree over them. |
| `include_bank_config` | `true` | How the bank is set up: its config overrides, directives, and webhooks. |
| `include_history` | `false` | `audit_log` and `llm_requests`. |

Mental models and knowledge pages count as **data**, not configuration. A mental model is a reading of the bank's facts and cites them by id in its `based_on` evidence, so carrying it without them would restore a synthesis whose grounding resolves to nothing. Their refresh history follows them for the same reason.

Embeddings and database ids are never carried: facts are re-embedded with the *target* bank's model and entities are re-resolved against it, so an archive moves cleanly to an instance configured with a different embedding model, vector extension, or text-search backend.

Webhooks travel with `include_bank_config`. When copying a bank whose webhooks point at a per-bank consumer, export with `include_bank_config=false`, or delete them on the copy.

### Export

`POST /v1/default/banks/{bank_id}/transfer/export` — runs as a **background operation** (a whole-bank export loads every unit and compresses a large archive). Returns `202` with an `operation_id`; poll the bank's operations endpoint, then download the archive from the `download_url` in `result_metadata`.

### Python

```python
# Whole bank, memories + config, no history.
# Submits the export, polls the operation, downloads the ZIP.
archive = await client.aexport_bank("transfer-py")

# Just the memories
memories_only = await client.aexport_bank("transfer-py", include_bank_config=False)

# Specific documents (a document subset carries no bank-level sections).
# The low-level call only submits; poll the returned operation yourself.
submission = await client.bank_transfer.export_bank_transfer(
    "transfer-py", document_id=["doc-1", "doc-2"], include_bank_config=False
)
```

### Node.js

```javascript
// Whole bank, memories + config, no history.
// Submits the export, polls the operation, downloads the ZIP.
const archive = await client.exportBank('transfer-js');

// Just the memories
const memoriesOnly = await client.exportBank('transfer-js', { includeBankConfig: false });

// Specific documents (a document subset carries no bank-level sections).
// The low-level call only submits; poll the returned operation yourself.
const { data: subset } = await sdk.exportBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js' },
    query: { document_id: ['doc-1', 'doc-2'], include_bank_config: false },
});
```

### CLI

```bash
# Whole bank, memories + config, no history
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export"

# Just the memories
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export?include_bank_config=false"

# Specific documents (a document subset carries no bank-level sections)
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/export?document_id=doc-1&document_id=doc-2&include_bank_config=false"
```

### Go

```go
// Whole bank, memories + config, no history
whole, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").Execute()

// Just the memories
memoriesOnly, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").
	IncludeBankConfig(false).Execute()

// Specific documents (a document subset carries no bank-level sections)
subset, _, err := client.BankTransferAPI.ExportBankTransfer(ctx, "transfer-go").
	DocumentId([]string{"doc-1", "doc-2"}).IncludeBankConfig(false).Execute()

// Each returns an operation id: poll it, then download result_metadata["storage_key"]
fmt.Println(whole.OperationId, memoriesOnly.OperationId, subset.OperationId)
```

A document subset must also pass `include_bank_config=false` (and leave `include_history` off). Bank-level sections cannot be scoped to documents, so the server rejects that combination with `400`.

### Import

`POST /v1/default/banks/{bank_id}/transfer/import` — multipart upload (`file` = the ZIP), also a background operation.

| Mode | Behaviour |
|------|-----------|
| `restore` (default) | Writes a whole bank into `target_bank_id`, which **must not already exist**. This is how a bank is moved to another instance, or copied under a new id. |
| `merge` | Folds the archive's documents into `{bank_id}`. `document_conflict` decides what happens to document ids that already exist: `skip` (default), `replace`, `new-id`. |

### Python

```python
# Restore a bank under a new id
operation_id = await client.aimport_bank("transfer-py", archive, target_bank_id="transfer-py-copy")
# The restore is recorded against the bank in the URL — poll it there
status = await client.operations.get_operation_status("transfer-py", operation_id)

# Merge an archive's documents into an existing bank
submission = await client.bank_transfer.import_bank_transfer(
    "transfer-py-other",
    ("transfer-py.zip", archive),
    mode="merge",
    document_conflict="replace",
)
```

### Node.js

```javascript
// Restore a bank under a new id
const restoreId = await client.importBank('transfer-js', new Blob([archive]), {
    targetBankId: 'transfer-js-copy',
});
// The restore is recorded against the bank in the URL — poll it there
const { data: restoreStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js', operation_id: restoreId },
});

// Merge an archive's documents into an existing bank
const { data: merge } = await sdk.importBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { mode: 'merge', document_conflict: 'replace' },
    body: { file: new Blob([archive]) },
});
```

### CLI

```bash
# Restore a bank under a new id
curl --fail-with-body -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/transfer/import?target_bank_id=transfer-bank-copy"

# Merge an archive's documents into an existing bank
curl --fail-with-body -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/transfer/import?mode=merge&document_conflict=replace"
```

### Go

```go
// Restore a bank under a new id
file, _ := os.Open(archivePath) // the ZIP downloaded from the export
restore, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go").
	File(file).TargetBankId("transfer-go-copy").Execute()
// The restore is recorded against the bank in the URL — poll it there
status, _, err := client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", restore.OperationId).Execute()

// Merge an archive's documents into an existing bank
file, _ = os.Open(archivePath)
merge, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go-other").
	File(file).Mode("merge").DocumentConflict("replace").Execute()
```

The include flags apply here too, and can only narrow: they restore a subset of what the archive holds, never more.

In `restore` mode the operation is recorded against `{bank_id}` — the bank in the URL — because the target bank does not exist yet. Poll that bank's operations endpoint for status and the per-component counts.

A restore carries the operations log as history, not as work: anything still in flight when the bank was exported is left behind, so a copied bank never re-runs the original's queued retains or re-fires its webhooks. In-flight work belongs to the bank that was exported — and a clone runs *inside* one such operation, so carrying them would put the clone's own unfinished record in the copy.

### Import facts extracted outside Hindsight

An archive doesn't have to come from an export. If you run your own extraction pipeline, build the archive yourself and import it with `mode=merge`: your chunks and facts are stored as given — no LLM call, no re-chunking — while Hindsight still re-embeds the facts, resolves their entities against the bank, builds the semantic, temporal, entity and causal links, and — like a retain — fires `retain.completed` webhooks and auto-consolidation when the bank has them enabled.

The archive is a ZIP with a `manifest.json` and one JSON file per document under `documents/`:

```text
import.zip
├── manifest.json            {"schema_version": 1, "source_bank_id": "external"}
└── documents/
    └── session-2026-09-22.json
```

A document file:

```json
{
  "id": "session-2026-09-22",
  "original_text": "Full original session text...",
  "tags": ["source:pi"],
  "chunks": [{ "chunk_index": 0, "chunk_text": "Caller-defined source region..." }],
  "facts": [
    {
      "text": "The user prefers lightweight local speech recognition models.",
      "fact_type": "experience",
      "chunk_index": 0,
      "context": "Discussion of local speech recognition",
      "mentioned_at": "2026-09-22T18:34:00Z",
      "occurred_start": "2026-09-22T18:34:00Z",
      "occurred_end": "2026-09-22T18:34:00Z",
      "entities": ["Parakeet"],
      "metadata": { "source_turn": "143" },
      "tags": ["source:pi"],
      "observation_scopes": "shared",
      "causal_relations": []
    }
  ]
}
```

| Fact field | Description |
|------------|-------------|
| `text` | Required. The memory, as it will be recalled. |
| `fact_type` | Required. `world` or `experience`. |
| `chunk_index` | Optional. The chunk in `chunks` this fact came from. |
| `context`, `metadata`, `tags`, `observation_scopes` | Optional. Same meaning as on a retain item, set per fact. |
| `mentioned_at`, `occurred_start`, `occurred_end` | Optional ISO 8601 dates. With none of them set, the fact is dated at import time. |
| `entities` | Optional entity names, resolved against the bank's existing entities. |
| `causal_relations` | Optional. `[{"relation_type": "caused_by", "target_fact_index": N}]`, where `N` is the position of another fact in the same document's `facts` list. |

Metadata and dates live on each fact; a document carries no metadata or timestamp of its own. Re-sending a document id follows `document_conflict` (`replace` to overwrite it).

### Python

```python
import io
import json
import zipfile

doc = {
    "id": "session-2026-09-22",
    "original_text": "Full original session text...",
    "chunks": [{"chunk_index": 0, "chunk_text": "Caller-defined source region..."}],
    "facts": [
        {
            "text": "The user prefers lightweight local speech recognition models.",
            "fact_type": "experience",
            "chunk_index": 0,
            "mentioned_at": "2026-09-22T18:34:00Z",
            "entities": ["Parakeet"],
        }
    ],
}
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("manifest.json", json.dumps({"schema_version": 1, "source_bank_id": "external"}))
    z.writestr(f"documents/{doc['id']}.json", json.dumps(doc))

submission = await client.bank_transfer.import_bank_transfer(
    "transfer-py-other",
    ("import.zip", buf.getvalue()),
    mode="merge",
    document_conflict="replace",
)
```

### Node.js

```javascript
import { crc32 } from 'node:zlib';

// Minimal uncompressed ZIP writer (Node has none built in); a library like jszip works too.
function zip(files) {
    const locals = [], centrals = [];
    let offset = 0;
    for (const [name, text] of Object.entries(files)) {
        const n = Buffer.from(name), d = Buffer.from(text), crc = crc32(d);
        const local = Buffer.alloc(30);
        local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x21, 12);
        local.writeUInt32LE(crc, 14); local.writeUInt32LE(d.length, 18); local.writeUInt32LE(d.length, 22);
        local.writeUInt16LE(n.length, 26);
        const central = Buffer.alloc(46);
        central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6);
        central.writeUInt16LE(0x21, 14); central.writeUInt32LE(crc, 16); central.writeUInt32LE(d.length, 20);
        central.writeUInt32LE(d.length, 24); central.writeUInt16LE(n.length, 28); central.writeUInt32LE(offset, 42);
        locals.push(local, n, d);
        centrals.push(central, n);
        offset += 30 + n.length + d.length;
    }
    const dir = Buffer.concat(centrals), end = Buffer.alloc(22);
    const count = Object.keys(files).length;
    end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(count, 8); end.writeUInt16LE(count, 10);
    end.writeUInt32LE(dir.length, 12); end.writeUInt32LE(offset, 16);
    return Buffer.concat([...locals, dir, end]);
}

const doc = {
    id: 'session-2026-09-22',
    original_text: 'Full original session text...',
    chunks: [{ chunk_index: 0, chunk_text: 'Caller-defined source region...' }],
    facts: [{
        text: 'The user prefers lightweight local speech recognition models.',
        fact_type: 'experience',
        chunk_index: 0,
        mentioned_at: '2026-09-22T18:34:00Z',
        entities: ['Parakeet'],
    }],
};
const archiveZip = zip({
    'manifest.json': JSON.stringify({ schema_version: 1, source_bank_id: 'external' }),
    [`documents/${doc.id}.json`]: JSON.stringify(doc),
});

const { data: external } = await sdk.importBankTransfer({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { mode: 'merge', document_conflict: 'replace' },
    body: { file: new Blob([archiveZip]) },
});
```

### CLI

```bash
mkdir -p external/documents
echo '{"schema_version": 1, "source_bank_id": "external"}' > external/manifest.json
cat > external/documents/session-2026-09-22.json <<'JSON'
{
  "id": "session-2026-09-22",
  "original_text": "Full original session text...",
  "chunks": [{"chunk_index": 0, "chunk_text": "Caller-defined source region..."}],
  "facts": [{
    "text": "The user prefers lightweight local speech recognition models.",
    "fact_type": "experience",
    "chunk_index": 0,
    "mentioned_at": "2026-09-22T18:34:00Z",
    "entities": ["Parakeet"]
  }]
}
JSON
(cd external && zip -qr ../import.zip manifest.json documents)

curl --fail-with-body -H "Authorization: Bearer $API_KEY" -F "file=@import.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/transfer/import?mode=merge&document_conflict=replace"
```

### Go

```go
doc := map[string]any{
	"id":            "session-2026-09-22",
	"original_text": "Full original session text...",
	"chunks":        []map[string]any{{"chunk_index": 0, "chunk_text": "Caller-defined source region..."}},
	"facts": []map[string]any{{
		"text":         "The user prefers lightweight local speech recognition models.",
		"fact_type":    "experience",
		"chunk_index":  0,
		"mentioned_at": "2026-09-22T18:34:00Z",
		"entities":     []string{"Parakeet"},
	}},
}
zipPath := filepath.Join(os.TempDir(), "import.zip")
out, _ := os.Create(zipPath)
zw := zip.NewWriter(out)
w, _ := zw.Create("manifest.json")
json.NewEncoder(w).Encode(map[string]any{"schema_version": 1, "source_bank_id": "external"})
w, _ = zw.Create("documents/session-2026-09-22.json")
json.NewEncoder(w).Encode(doc)
zw.Close()
out.Close()

file, _ = os.Open(zipPath)
external, _, err := client.BankTransferAPI.ImportBankTransfer(ctx, "transfer-go-other").
	File(file).Mode("merge").DocumentConflict("replace").Execute()
```

### Clone a bank

`POST /v1/default/banks/{bank_id}/clone` — copy a bank into a new one in a single call, without handling an archive yourself. This is the export and import above run back to back on this instance: nothing is re-extracted and no LLM is called, so the clone's facts are exactly the source's, re-embedded with the same model.

### Python

```python
operation_id = await client.aclone_bank("transfer-py", "transfer-py-clone")
# The operation is recorded against the source bank
status = await client.operations.get_operation_status("transfer-py", operation_id)
```

### Node.js

```javascript
const cloneId = await client.cloneBank('transfer-js', 'transfer-js-clone');
// The operation is recorded against the source bank
const { data: cloneStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js', operation_id: cloneId },
});
```

### CLI

```bash
curl --fail-with-body -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/clone?target_bank_id=transfer-bank-clone"
# -> {"operation_id": "…", "status": "pending"}
```

### Go

```go
clone, _, err := client.BankTransferAPI.CloneBank(ctx, "transfer-go").
	TargetBankId("transfer-go-clone").Execute()
// The operation is recorded against the source bank
status, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", clone.OperationId).Execute()
```

| Query param | Default | Description |
|-------------|---------|-------------|
| `target_bank_id` | — | Required. The bank to create; it must not already exist. |
| `include_data` | `true` | Copy the memories, everything backing them, and the mental models and knowledge pages synthesized from them. |
| `include_bank_config` | `true` | Copy the bank's config overrides, directives and webhooks. |
| `include_history` | `false` | Copy `audit_log` and `llm_requests`. |

The clone is independent from the moment it is made: later retains, consolidation and edits on either bank leave the other alone.

The operation is recorded against the **source** bank, because the target does not exist yet — poll the source's operations endpoint for status and the per-component counts.

> **⚠️ A clone inherits the source's webhooks**
>
Webhooks are part of a bank's configuration, so a clone made with the default flags will call the source's webhook endpoints. Pass `include_bank_config=false`, or delete them on the clone, when they point at a per-bank consumer.
### Document export & import (superseded)

The endpoints below still work exactly as documented and are unchanged; new integrations should use `/transfer/export` and `/transfer/import` above, which carry the same document archives plus the bank's own configuration.

Move documents — and the facts already extracted from them — between banks **without re-running the LLM**. Useful for testing a different embedding model, or copying data between banks/instances without paying for re-extraction. The archive carries documents, raw chunks, and extracted facts (entities by canonical name, causal links) — but **no embeddings or database ids**. On import, facts are re-embedded with the *target* bank's model and entities/links are recomputed against it, so imported documents are integrated with whatever already exists there.

### Export documents

`POST /v1/default/banks/{bank_id}/document-transfer/export` — runs as a **background operation** (a whole-bank export loads every unit and compresses a large archive, which on a big bank could exhaust memory and pin a connection). It returns `202` with an `operation_id`; poll the bank's operations endpoint, then download the archive from the `download_url` in `result_metadata`.

### Python

```python
# Submits the export (whole bank; pass document_ids=[...] to scope it),
# polls the operation until completed, downloads the archive.
archive = await client.aexport_documents("transfer-py")
with open("transfer-py-documents.zip", "wb") as f:
    f.write(archive)
```

### Node.js

```javascript
// Submits the export (whole bank; pass { documentIds: [...] } to scope it),
// polls the operation until completed, downloads the archive.
const docArchive = await client.exportDocuments('transfer-js');
await writeFile('transfer-js-documents.zip', docArchive);
```

### CLI

```bash
# 1. Submit the export (whole bank; add ?document_id=… to scope it)
OPERATION_ID=$(curl -sf -X POST -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/document-transfer/export" | jq -r .operation_id)
# -> {"operation_id": "…", "status": "pending"}

# 2. Poll until completed
until [ "$(curl -s -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations/$OPERATION_ID" | jq -r .status)" = completed ]; do
  sleep 1
done
DOWNLOAD_URL=$(curl -s -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-bank/operations/$OPERATION_ID" | jq -r .result_metadata.download_url)
# -> {"status":"completed","result_metadata":{
#      "download_url":"/v1/default/files/download/banks/transfer-bank/exports/…/transfer.zip",
#      "storage_key":"banks/transfer-bank/exports/…/transfer.zip","byte_size":12345,"filename":"transfer-bank-documents.zip"}}

# 3. Download the archive
curl -sf -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL$DOWNLOAD_URL" -o transfer-bank-documents.zip
```

### Go

```go
// 1. Submit the export (whole bank; add .DocumentId([]string{...}) to scope it)
export, _, err := client.DocumentTransferAPI.ExportDocuments(ctx, "transfer-go").Execute()

// 2. Poll until completed
var done *hindsight.OperationStatusResponse
for {
	done, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go", export.OperationId).Execute()
	if err != nil || done.Status == "completed" || done.Status == "failed" {
		break
	}
	time.Sleep(time.Second)
}

// 3. Download the archive (the client saves it to a temp file)
zipFile, _, err := client.DocumentTransferAPI.
	DownloadFile(ctx, done.ResultMetadata["storage_key"].(string)).Execute()
```

| Query param | Description |
|-------------|-------------|
| `document_id` | Repeatable. Export only these documents; omit for the whole bank. |
| `include_observations` | Also export consolidated observations (default `false`). Only valid for a **whole-bank** export — combining it with `document_id` returns `400`. |

> **📝 The synchronous `GET …/document-transfer` was removed**
>
It loaded the entire bank into memory and held a database connection for the full request, which could take down the shared API on large banks. It now returns `410` pointing here. Use the async flow above. The download route (`GET /v1/default/files/download/{key}`) authorizes the caller against the bank the archive belongs to.
The archive lives as long as its export **operation record** — indefinitely by default, or until the operation is pruned when `HINDSIGHT_API_OPERATION_RETENTION_DAYS` is set (the archive is deleted in step with the row). Deleting the operation removes the archive immediately.

### Import documents

`POST /v1/default/banks/{bank_id}/document-transfer` — multipart upload (`file` = the ZIP). Runs as a **background operation** (re-embedding + entity resolution can take a while), so it returns `202` with an `operation_id`; poll the bank's operations endpoint for status and the result counts in `result_metadata`.

### Python

```python
with open("transfer-py-documents.zip", "rb") as f:
    submission = await client.document_transfer.import_documents(
        "transfer-py-other", ("transfer-py-documents.zip", f.read()), on_conflict="replace"
    )

status = await client.operations.get_operation_status("transfer-py-other", submission.operation_id)
# status.result_metadata -> {"documents_imported": 3, "facts_imported": 42, "observations_imported": 5, ...}
```

### Node.js

```javascript
const { data: docImport } = await sdk.importDocuments({
    client: apiClient,
    path: { bank_id: 'transfer-js-other' },
    query: { on_conflict: 'replace' },
    body: { file: new Blob([await readFile('transfer-js-documents.zip')]) },
});

const { data: docImportStatus } = await sdk.getOperationStatus({
    client: apiClient,
    path: { bank_id: 'transfer-js-other', operation_id: docImport.operation_id },
});
// docImportStatus.result_metadata -> { documents_imported: 3, facts_imported: 42, observations_imported: 5, ... }
```

### CLI

```bash
OPERATION_ID=$(curl -sf -H "Authorization: Bearer $API_KEY" -F "file=@transfer-bank-documents.zip" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/document-transfer?on_conflict=replace" | jq -r .operation_id)
# -> {"operation_id": "…", "status": "pending"}

curl --fail-with-body -H "Authorization: Bearer $API_KEY" \
  "$HINDSIGHT_URL/v1/default/banks/transfer-other-bank/operations/$OPERATION_ID"
# -> {"status":"completed","result_metadata":{"documents_imported":3,"facts_imported":42,"observations_imported":5,...}}
```

### Go

```go
file, _ = os.Open(zipFile.Name())
imported, _, err := client.DocumentTransferAPI.ImportDocuments(ctx, "transfer-go-other").
	File(file).OnConflict("replace").Execute()

status, _, err = client.OperationsAPI.GetOperationStatus(ctx, "transfer-go-other", imported.OperationId).Execute()
// status.ResultMetadata -> {"documents_imported": 3, "facts_imported": 42, "observations_imported": 5, ...}
```

`on_conflict` controls what happens when a document id already exists in the target bank:

| Mode | Behavior |
|------|----------|
| `skip` (default) | Leave the existing document untouched. |
| `replace` | Delete the existing document's data and re-import. |
| `new-id` | Import a copy under a freshly generated id. |

### Observations

Consolidated observations are excluded by default — the target bank regenerates them from the imported facts during consolidation. Pass `include_observations=true` to carry them instead: they're restored with no LLM, their source references remapped to the imported facts (which are marked consolidated so the target won't re-consolidate them).

Because an observation can be derived from facts spanning several documents, `include_observations` is only supported on a **whole-bank export** (omit `document_id`); combining it with a document subset returns `400`.

> **⚠️ Imported observations are inserted as-is — no merge**
>
They are not merged or deduplicated against observations already in the target bank (consolidation merges related observations; import does not). Prefer importing observations into a fresh/empty bank, or omit `include_observations` and let the target consolidate the imported facts itself.
### Enabling / disabling

Both endpoints are gated by server-level flags (default `true`). A disabled endpoint returns `404`, and `/version` reports the state under `features.document_export_api` / `features.document_import_api` (the control plane hides the buttons accordingly).

| Variable | Gates |
|----------|-------|
| `HINDSIGHT_API_ENABLE_DOCUMENT_EXPORT_API` | `POST …/document-transfer/export` and `GET …/files/download/{key}` |
| `HINDSIGHT_API_ENABLE_DOCUMENT_IMPORT_API` | `POST …/document-transfer` |

## Migrating a bank to a new instance

To move a bank to an instance configured with a different **embedding model**, **vector extension**, or **text-search backend** — which can't be changed in place on a populated bank — export the whole bank and import it into the new instance, where every embedding and index is re-derived from the stored text with **no LLM re-extraction**. This carries documents, facts, observations, bank config, mental models, directives, and webhooks (never embeddings).

Use the `hindsight-admin export-bank` / `import-bank` commands and follow the blue-green runbook in **[Admin CLI → Migrating a bank to a new instance](../admin-cli.md#migrating-a-bank-to-a-new-instance)**.
