

# Knowledge Banks

Create knowledge banks, ingest files and text into them, and search their passages. For what a knowledge bank is and how ingest and search work, see [Knowledge Banks](../knowledge-banks.md).

All endpoints live under `/v1/default/knowledge-banks`. Each example shows the call with the Python, Node.js and Go clients, the [CLI](../../sdks/cli.md#knowledge-banks) (`hindsight kb`), and `curl`.

---

## Create a Knowledge Bank

### Python

```python
await kb.create_knowledge_bank(CreateKnowledgeBankRequest(bank_id="product-manuals"))
```

### Node.js

```javascript
await sdk.createKnowledgeBank({
    client: apiClient,
    body: { bank_id: 'product-manuals' },
    throwOnError: true,
});
```

### CLI

```bash
hindsight kb create product-manuals
```

### Go

```go
_, _, err = client.KnowledgeBanksAPI.CreateKnowledgeBank(ctx).
	CreateKnowledgeBankRequest(*hindsight.NewCreateKnowledgeBankRequest("product-manuals")).
	Execute()
```

### HTTP

```bash
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks" \
  -H "Content-Type: application/json" \
  -d '{"bank_id": "product-manuals"}'
```

The id shares one space with memory bank ids and bank aliases: creating a knowledge bank with one of those ids fails with `409`.

The response, like `GET /v1/default/knowledge-banks/{bank_id}`, describes the bank:

```json
{
  "bank_id": "product-manuals",
  "created_at": "2026-11-16T09:12:04Z",
  "document_count": 0,
  "passage_count": 0,
  "ready_count": 0,
  "pending_count": 0,
  "failed_count": 0,
  "documents_by_media_type": {},
  "last_document_at": null
}
```

List banks with `GET /v1/default/knowledge-banks`, with `limit` (default 100), `offset` and `q` (part of the bank id).

---

## Ingest Files

Upload one or more files (PDFs, Office documents, images, audio, video, text, Markdown, JSON, CSV, HTML) as `multipart/form-data`. Each file becomes one document. The call returns `202` right away and the files are processed in the background.

### Python

```python
result = await kb.ingest_knowledge_files(
    bank_id="product-manuals",
    files=[
        ("x200-manual.pdf", SAMPLE_PDF.read_bytes()),
        ("x200-quickstart.md", b"# X200 quick start\n\nHold the power button for two seconds.\n"),
    ],
    request=json.dumps({"files": [
        {"document_id": "x200-manual", "tags": ["x200"]},
        {"document_id": "x200-quickstart", "tags": ["x200", "quickstart"]},
    ]}),
)
print(result.document_ids, result.operation_ids)
```

### Node.js

```javascript
const { data: ingested } = await sdk.ingestKnowledgeFiles({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: {
        files: [
            new File([readFileSync(SAMPLE_PDF)], 'x200-manual.pdf'),
            new File(['# X200 quick start\n\nHold the power button for two seconds.\n'], 'x200-quickstart.md'),
        ],
        request: JSON.stringify({ files: [
            { document_id: 'x200-manual', tags: ['x200'] },
            { document_id: 'x200-quickstart', tags: ['x200', 'quickstart'] },
        ] }),
    },
    throwOnError: true,
});
console.log(ingested.document_ids, ingested.operation_ids);
```

### CLI

```bash
# Each file becomes one document; --wait returns once they are searchable.
# Without --document-id, a file's id is its path (here: x200-manual.pdf),
# and a folder uploads every file in it.
hindsight kb ingest product-manuals x200-manual.pdf \
  --document-id x200-manual --tags x200 --wait
hindsight kb ingest product-manuals x200-quickstart.md \
  --document-id x200-quickstart --tags x200,quickstart --wait
```

### Go

```go
manual, _ := os.Open("x200-manual.pdf")
quickstart, _ := os.Open("x200-quickstart.md")
options, _ := json.Marshal(map[string]any{"files": []map[string]any{
	{"document_id": "x200-manual", "tags": []string{"x200"}},
	{"document_id": "x200-quickstart", "tags": []string{"x200", "quickstart"}},
}})
ingested, _, err := client.KnowledgeBanksAPI.IngestKnowledgeFiles(ctx, "product-manuals").
	Files([]*os.File{manual, quickstart}).
	Request(string(options)).
	Execute()
```

### HTTP

```bash
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/files/ingest" \
  -F "files=@x200-manual.pdf" \
  -F "files=@x200-quickstart.md" \
  -F 'request={"files": [
        {"document_id": "x200-manual", "tags": ["x200"]},
        {"document_id": "x200-quickstart", "tags": ["x200", "quickstart"]}
      ]}'
```

The optional `request` field is JSON with one entry per file, in the same order:

| Field | Description |
|---|---|
| `document_id` | Stable id. Ingesting it again replaces the document. Defaults to a new random id. |
| `tags` | Tags a search can filter on. |
| `strategy` | [Ingestion strategy](../knowledge-banks.md#ingestion-strategies) for this file. Defaults to the server's. |

```json
{
  "document_ids": ["x200-manual", "x200-quickstart"],
  "operation_ids": ["6f1c...", "6f1c..."],
  "unchanged_document_ids": []
}
```

- `operation_ids` has the same id for every queued document: one background operation handles the whole upload.
- `unchanged_document_ids` lists documents identical to what is already stored. Nothing is queued for them.

**Knowing when it's done.** `hindsight kb ingest --wait` waits for you. Otherwise poll the [operation](#operations) with `GET /{bank_id}/operations/{operation_id}`, or read each document's `status`. A document goes `pending` → `processing` → `ready`, or `failed` with an `error`.

---

## Ingest Text

Send text as JSON, up to 100 items per call. By default the call returns once every document can be searched. If some items can't be ingested (an empty text, say), the others are still saved and the call fails with `400`, naming the ones that failed.

### Python

```python
await kb.ingest_knowledge_texts(
    "product-manuals",
    KnowledgeIngestRequest(items=[
        KnowledgeIngestTextItem(
            document_id="returns-policy",
            text="# Returns\n\nYou can return an X200 within 30 days...",
            tags=["policy"],
        )
    ]),
)
```

### Node.js

```javascript
await sdk.ingestKnowledgeTexts({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { items: [{
        document_id: 'returns-policy',
        text: '# Returns\n\nYou can return an X200 within 30 days...',
        tags: ['policy'],
    }] },
    throwOnError: true,
});
```

### CLI

```bash
hindsight kb ingest-text product-manuals \
  "# Returns

You can return an X200 within 30 days..." \
  --document-id returns-policy --tags policy
```

### Go

```go
returnsID := "returns-policy"
_, _, err = client.KnowledgeBanksAPI.IngestKnowledgeTexts(ctx, "product-manuals").
	KnowledgeIngestRequest(hindsight.KnowledgeIngestRequest{
		Items: []hindsight.KnowledgeIngestTextItem{{
			DocumentId: *hindsight.NewNullableString(&returnsID),
			Text:       "# Returns\n\nYou can return an X200 within 30 days...",
			Tags:       []string{"policy"},
		}},
	}).
	Execute()
```

### HTTP

```bash
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "document_id": "returns-policy",
        "text": "# Returns\n\nYou can return an X200 within 30 days...",
        "tags": ["policy"]
      }
    ]
  }'
```

| Field | Description |
|---|---|
| `items[].text` | The text, read as Markdown. |
| `items[].document_id` | Stable id. Ingesting it again replaces the document. |
| `items[].filename` | Defaults to `<document_id>.md`. |
| `items[].tags` | Tags a search can filter on. |
| `items[].strategy` | Ingestion strategy for this item. |
| `async` | `true` to queue the items and return `202` at once, like a file ingest. Default `false`. |

The response has the same shape as a file ingest. `operation_ids` is empty when the ingest ran before returning. With an [ingestion strategy](../knowledge-banks.md#ingestion-strategies), the documents it splits out are still processed in the background after the call returns.

---

## Search

### Python

```python
found = await kb.search_knowledge_bank(
    "product-manuals",
    KnowledgeSearchRequest(query="How long do I have to return an X200?", max_tokens=2048),
)
for hit in found.results:
    print(hit.document_id, hit.text)
```

### Node.js

```javascript
const { data: found } = await sdk.searchKnowledgeBank({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { query: 'How long do I have to return an X200?', max_tokens: 2048 },
    throwOnError: true,
});
for (const hit of found.results) console.log(hit.document_id, hit.text);
```

### CLI

```bash
hindsight kb search product-manuals "How long do I have to return an X200?" --max-tokens 2048
```

### Go

```go
query := "How long do I have to return an X200?"
maxTokens := int32(2048)
found, _, err := client.KnowledgeBanksAPI.SearchKnowledgeBank(ctx, "product-manuals").
	KnowledgeSearchRequest(hindsight.KnowledgeSearchRequest{
		Query:     *hindsight.NewNullableString(&query),
		MaxTokens: &maxTokens,
	}).
	Execute()
```

### HTTP

```bash
curl -X POST "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do I have to return an X200?", "max_tokens": 2048}'
```

### Parameters

#### query

The text to search with. It drives both the search by meaning (it is embedded) and the search by keywords (its rare words are matched as they are, in any language). Send `query`, `query_image`, `query_video`, or any mix of them: at least one is needed.

#### query_image

A base64 image to search with: a photo of a part, a screenshot, a page. Alone, it finds the pages, images and clips that look like it. With a `query`, the two are embedded together as one query. Needs an embedder that reads images: with one that doesn't, the search fails with `415`.

#### query_video

A base64 MP4 clip to search with, alone or with text and an image, embedded together as one query. Needs an embedder that reads video: with one that doesn't, the search fails with `415`.

#### budget

How deep the search ranks before `max_tokens` picks the results: `low`, `mid` (default) or `high`, keeping the best 50, 200 or 500 scored candidates. A higher budget reaches further down the list but takes longer, as in memory recall.

#### max_tokens

How much text the results can add up to, from 1 to 1,000,000. Defaults to `4096`. The best hits go in first, each text hit with the passages around it, so you get the paragraph rather than a cut sentence. Any budget left over widens the best hits a little more. An image, page or clip costs a small fixed amount plus its text. Set it to however much of your context window the documents should take.

#### tags

Only documents with these tags, matched as memory recall matches memory tags. See [`tags` in recall](./recall.md#tags) for how each mode treats untagged documents.

#### tags_match

How `tags` match: `any` (default: any of the tags, untagged documents included), `all` (every tag, untagged included), `any_strict` and `all_strict` (the same, untagged documents left out), or `exact` (exactly these tags). With `exact` and no tags, only untagged documents match.

#### tag_groups

And/or/not groups of tag filters, as memory recall takes them (see [`tag_groups` in recall](./recall.md#tag_groups)). The groups in the list must all match. A leaf with `resolve: "fuzzy"` matches its tags against the bank's own tags by similarity, so a typo still finds them.

#### document_ids

Only these documents. Useful to search inside one manual, or the files a user picked.

#### media_types

Only these kinds of passage: `document` (PDF pages), `image`, `audio`, `video` or `text`. Leave it out to search them all.

#### enable_reranking

`false` keeps the search's own order and skips the reranker. Leave it out (or `true`) to rerank the best hits when the server has a [reranker for knowledge banks](../knowledge-banks.md#searching); without one, it does nothing.

#### trace

`true` also returns how the search ranked, step by step: every candidate and its score, the keywords used, what the budget kept, and how long each step took. See [Trace](#trace) below.

### Response

```json
{
  "results": [
    {
      "document_id": "x200-manual",
      "filename": "x200-manual.pdf",
      "tags": ["x200"],
      "index": 30,
      "media_type": "document",
      "page": 31,
      "start_ms": null,
      "end_ms": null,
      "text": "Troubleshooting: pressure drop. If the pressure falls below...",
      "passage_url": "/v1/default/knowledge-banks/product-manuals/documents/x200-manual/passages/30/media",
      "document_url": "/v1/default/knowledge-banks/product-manuals/documents/x200-manual/file"
    },
    {
      "document_id": "x200-install",
      "filename": "install-walkthrough.mp4",
      "tags": ["x200", "video"],
      "index": 6,
      "media_type": "video",
      "page": null,
      "start_ms": 150000,
      "end_ms": 180000,
      "text": null,
      "passage_url": "/v1/default/knowledge-banks/product-manuals/documents/x200-install/passages/6/media",
      "document_url": "/v1/default/knowledge-banks/product-manuals/documents/x200-install/file"
    }
  ],
  "trace": null
}
```

- Results come **best first**. They carry no score: scores are only comparable within one search. With a reranker set for knowledge banks, the order is the [reranked](../knowledge-banks.md#searching) one, unless the search sets `enable_reranking: false`.
- A text hit's `text` holds the matched passage together with the passages around it that fit the budget. `index` is the matched one.
- `page` is set for PDF pages and images, `start_ms` and `end_ms` for audio and video.
- `passage_url` streams the passage's page image or clip. `document_url` streams the original file.

### Trace

With `"trace": true`, the response also has a `trace`:

| Field | What it shows |
|---|---|
| `candidate_searches` | Each candidate search (by meaning, by extra vectors, by keyword, and `document`: the text passages of the three best documents) and what it found, best first. |
| `hits` | Every scored candidate: its final `score`, its `similarity`, its `keyword` score, which searches `found_by` it, and its `outcome`: `returned`, `merged` (inside a better hit's text), `over_budget`, or `below_depth` (past what `budget` keeps). |
| `budget_steps` | How `max_tokens` was filled, step by step. |
| `keyword_terms` | The query words the keyword search used, with their weight. A word is left out when it is much more common than the query's rarest word, unless it is still rare in the bank (in fewer than about 1 in 8 passages). |
| `keyword_weight` | The keyword score's share of the final score. |
| `depth`, `candidates`, `tokens_used` | Candidates kept, candidates found, and tokens returned. |
| `timings_ms` | Milliseconds spent in each step: opening the bank, embedding the query, each database query, scoring, the rerank and building the text. |

---

## Tags

### Python

```python
tags = await kb.list_knowledge_bank_tags("product-manuals", q="x2*")
for item in tags.items:
    print(item.tag, item.count)
```

### Node.js

```javascript
const { data: tags } = await sdk.listKnowledgeBankTags({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    query: { q: 'x2*' },
    throwOnError: true,
});
for (const item of tags.items) console.log(item.tag, item.count);
```

### CLI

```bash
hindsight kb list-tags product-manuals -q 'x2*'
```

### Go

```go
tags, _, err := client.KnowledgeBanksAPI.ListKnowledgeBankTags(ctx, "product-manuals").
	Q("x2*").
	Execute()
```

### HTTP

```bash
curl "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/tags?q=x2*"
```

Lists the tags on a bank's documents, with how many documents carry each. `q` is a wildcard pattern (`*`), case-insensitive. Paging: `limit` (default 100, up to 1000), `offset`.

---

## Documents

| Call | What it does |
|---|---|
| `GET /{bank_id}/documents` | List documents. Filters: `q` (part of the id or filename), `status`, `media_type`, `tags`. Paging: `limit` (default 50, up to 500), `offset`. |
| `GET /{bank_id}/documents/{document_id}` | A document with all its passages. |
| `GET /{bank_id}/documents/{document_id}/file` | Download the original file. A document that failed for good no longer has it: `404`. |
| `GET /{bank_id}/documents/{document_id}/passages/{index}/media` | Stream a passage's page image, image, or audio or video window. |
| `DELETE /{bank_id}/documents/{document_id}` | Delete a document, its passages, its file, and the documents split out of it. |

A document looks like this:

```json
{
  "document_id": "x200-manual",
  "filename": "x200-manual.pdf",
  "content_type": "application/pdf",
  "media_type": "document",
  "size_bytes": 4821337,
  "status": "ready",
  "error": null,
  "passage_count": 40,
  "tags": ["x200"],
  "created_at": "2026-11-16T09:14:20Z",
  "updated_at": "2026-11-16T09:15:02Z",
  "document_url": "/v1/default/knowledge-banks/product-manuals/documents/x200-manual/file",
  "source_document_id": null
}
```

`source_document_id` is set when an [ingestion strategy](../knowledge-banks.md#ingestion-strategies) split this document out of another.

With the CLI:

```bash
hindsight kb list-documents product-manuals --status ready
hindsight kb get-document product-manuals x200-manual
hindsight kb download product-manuals x200-manual --out manual-copy.pdf
hindsight kb passage-media product-manuals x200-manual 0 --out page-1.png
```

```bash
hindsight kb delete-document product-manuals returns-policy
```

---

## Operations

Ingests run as background operations, under the knowledge bank's own path. They answer with the same shapes as a [memory bank's operations](./operations.md).

| Call | What it does |
|---|---|
| `GET /{bank_id}/operations` | List operations. Filters: `status`, `type`. Paging: `limit`, `offset`. |
| `GET /{bank_id}/operations/{operation_id}` | One operation's status. |
| `DELETE /{bank_id}/operations/{operation_id}` | Cancel a pending or running operation. |
| `POST /{bank_id}/operations/{operation_id}/retry` | Retry a failed operation. |
| `DELETE /{bank_id}/operations/{operation_id}/delete` | Delete a finished operation. |

With the CLI (`cancel`, `retry` and `delete` take the same arguments as `get`):

```bash
hindsight kb operation list product-manuals
hindsight kb operation get product-manuals "$OPERATION_ID"
```

---

## Settings

A bank can change how its files are split, and nothing else: the embedder, keys, search and server limits stay the server's. A change applies to files ingested after it: ingesting an unchanged file again splits it with the new settings.

### Python

```python
await kb.update_knowledge_bank_config(
    "product-manuals",
    KnowledgeBankConfigUpdate(updates={"text_passage_tokens": 256, "disabled_media_types": ["video"]}),
)
```

### Node.js

```javascript
await sdk.updateKnowledgeBankConfig({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    body: { updates: { text_passage_tokens: 256, disabled_media_types: ['video'] } },
    throwOnError: true,
});
```

### CLI

```bash
hindsight kb set-config product-manuals \
  --set text_passage_tokens=256 \
  --set 'disabled_media_types=["video"]'
```

### Go

```go
_, _, err = client.KnowledgeBanksAPI.UpdateKnowledgeBankConfig(ctx, "product-manuals").
	KnowledgeBankConfigUpdate(hindsight.KnowledgeBankConfigUpdate{
		Updates: map[string]interface{}{
			"text_passage_tokens":  256,
			"disabled_media_types": []string{"video"},
		},
	}).
	Execute()
```

### HTTP

```bash
curl -X PATCH "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"text_passage_tokens": 256, "disabled_media_types": ["video"]}}'
```

| Setting | What it changes | Range |
|---|---|---|
| `text_passage_tokens` | size of a text passage | 16–8192 |
| `pdf_page_pixels` | longest side of a rendered PDF page | 256–8192 |
| `pdf_bands` | also embed three horizontal strips of each PDF page: slightly better search, half the ingest speed | `true` / `false` |
| `image_tokens` | how much detail the embedder reads from each image | 70–1120 |
| `audio_window_seconds` | length of an audio window | 1–3600 |
| `video_window_seconds` | length of a video window | 1–3600 |
| `media_overlap_seconds` | overlap between windows | 0 to half the shorter window |
| `max_media_seconds` | longest audio or video file accepted | up to the server's limit |
| `max_pdf_pages` | longest PDF accepted | up to the server's limit |
| `disabled_media_types` | kinds of file this bank refuses | `document`, `image`, `audio`, `video`, `text` |

| Call | What it does |
|---|---|
| `GET /{bank_id}/config` | `config`: every setting as the bank uses it. `overrides`: only what the bank changes. |
| `PATCH /{bank_id}/config` | Change settings: `{"updates": {"text_passage_tokens": 256}}`. A `null` value goes back to the server's setting. |
| `DELETE /{bank_id}/config` | Go back to the server's settings. |

`PATCH` and `DELETE` need `HINDSIGHT_API_ENABLE_BANK_CONFIG_API` on.

With the CLI, to see the settings or undo changes:

```bash
# Every setting as the bank uses it, then only what the bank changes
hindsight kb config product-manuals
hindsight kb config product-manuals --overrides-only

# Drop one override, or all of them
hindsight kb set-config product-manuals --unset disabled_media_types
hindsight kb reset-config product-manuals -y
```

---

## Stats

`GET /{bank_id}/stats/timeseries` returns documents and passages added over time. `period` is `1d` (by hour), `7d`, `30d` or `90d` (by day; default `7d`). `media_type` limits it to one kind of file.

It feeds the control plane's chart and has no CLI command; `hindsight kb get` shows the bank's current counts.

---

## Delete a Knowledge Bank

### Python

```python
await kb.delete_knowledge_bank("product-manuals")
```

### Node.js

```javascript
await sdk.deleteKnowledgeBank({
    client: apiClient,
    path: { bank_id: 'product-manuals' },
    throwOnError: true,
});
```

### CLI

```bash
hindsight kb delete product-manuals -y
```

### Go

```go
_, _, err = client.KnowledgeBanksAPI.DeleteKnowledgeBank(ctx, "product-manuals").Execute()
```

### HTTP

```bash
curl -X DELETE "$HINDSIGHT_URL/v1/default/knowledge-banks/product-manuals"
```

Cancels the bank's ingests, then deletes every document, passage and file in it.

---

## Errors

| Status | When |
|---|---|
| `400` | A bad request: no query, a query image that isn't base64, an empty file, an unknown ingestion strategy, a `request` field that isn't valid JSON or doesn't have one entry per file, the same `document_id` twice, or a text ingest where some items failed. |
| `404` | No such bank or document, a deleted file, or knowledge banks are off. |
| `409` | The bank id is already a memory bank or a bank alias. |
| `413` | A file over `HINDSIGHT_API_KB_MAX_FILE_MB`, or over its type's limit in `HINDSIGHT_API_KB_MAX_FILE_MB_BY_MEDIA_TYPE`. |
| `415` | A file type the server, the bank or the embedder doesn't take, or a query the embedder can't embed (a video query on a model without video). |
| `422` | A `tag_groups` leaf with `resolve: "fuzzy"` that can't be resolved, or a bank with too many tags to match fuzzily. |
| `500` | An ingestion strategy's extension that fails to load. |
| `503` | The embedder or the database can't be reached, or timed out. Try again later. |
