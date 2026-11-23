

# Knowledge Banks

A **knowledge bank** stores your documents as they are and finds the right passage in them: a page of a PDF, a window of a video, a paragraph of a manual. You upload files, then search them with text, an image or a short clip.

There is no LLM anywhere in a knowledge bank. Nothing is summarised, extracted or rewritten. What you search is what you uploaded, and every result points back to its page or its second in the original file.

---

## Knowledge Banks and Memory Banks

A [memory bank](./api/memory-banks.md) learns from what it is told. Retain sends the content to an LLM, which pulls out facts, people and dates, and consolidation turns those facts into observations. What you get back is what the bank *learned*, not the text you sent.

A knowledge bank keeps the text. It is for content that is already right and must stay word for word: product manuals, policies, contracts, research papers, support articles, recorded calls, training videos.

**Figure: Memory Banks and Knowledge Banks.** An animated diagram on the docs site; its narration, step by step:

- **each on its own**
  1. On their own, each bank has its own job. A personal assistant remembers Sam. A help center searches its manuals.
  2. Retain sends the chat to an LLM. Ingest splits the files and embeds the pieces, with no LLM.
  3. The memory bank keeps what it learned about Sam. The knowledge bank keeps the files word for word.
  4. The assistant recalls what it knows about Sam. The help center searches its files.
  5. Recall returns facts the bank learned. Search returns the exact page and the second of the video.
- **together**
  1. A support agent needs both: one memory bank per customer, linked to one knowledge bank of policies.
  2. The memory bank keeps what the customer said, true or not. The knowledge bank keeps the policy as written.
  3. The customer asks. Reflect reads both banks.
  4. Memories bring the customer's context. The documents bring the rules.
  5. When they disagree, the document wins. The customer can't talk the agent out of the policy.

| | Memory bank | Knowledge bank |
|---|---|---|
| You send | conversations, notes, events | files and documents |
| It stores | facts and observations an LLM wrote | the original files, split into passages |
| You get back | facts, with their sources | passages, with their page or timestamp |
| Uses an LLM | yes, on every retain | never |
| Kinds of content | text, plus images through retain | PDF, image, audio, video, text, Office files |
| Operations | retain, recall, reflect | ingest, search |

The two work together: [link a knowledge bank to a memory bank](#linking-to-a-memory-bank) and reflect reads both. Knowledge banks and memory banks share one set of bank ids, so a knowledge bank cannot have the id of an existing memory bank.

---

## Ingesting Documents

Each file you ingest becomes one **document**, split into **passages**. Each passage is embedded on its own, and search returns passages.

**Figure: Ingesting into a Knowledge Bank.** An animated diagram on the docs site; its narration, step by step:

- **ingest**
  1. You upload files as they are. Each file becomes one document. Nothing is summarised and no LLM reads it.
  2. The call returns at once. One background operation picks up the whole upload.
  3. Each kind of file is split its own way: a PDF page by page, audio and video into short clips, text into passages that keep headings with their content.
  4. Every passage is embedded into the same space, so a text question can find a page image or a video clip.
  5. The passages swap in at once and the documents turn ready. The original files are kept, so search can point back to the page or the second.

Each kind of file is split its own way. None of them goes through OCR or an LLM: the embedder reads pages and clips as they are.

| File | Becomes | Types |
|---|---|---|
| PDF | one passage per page: the page rendered as an image, with its text layer kept and embedded too | `.pdf` |
| Image | one passage | `.png` `.jpg` `.webp` `.gif` `.bmp` `.tiff` `.heic` |
| Audio | windows of 60 seconds, 5 seconds of overlap | `.mp3` `.wav` `.m4a` `.aac` `.flac` `.ogg` `.opus` `.wma` |
| Video | windows of 30 seconds (picture and sound), 5 seconds of overlap | `.mp4` `.mov` `.mkv` `.webm` `.avi` `.m4v` `.mpeg` |
| Text | passages of about 512 tokens that follow the document's structure (see [how text is split](#how-text-is-split)) | `.txt` `.md` `.json` `.jsonl` `.csv` `.yaml` `.xml` `.rst` |
| Office and web | their text, then split as text | `.docx` `.pptx` `.xlsx` `.xls` `.html` `.epub` |

Audio and video are cut and decoded in-process (PyAV), with nothing to install on the server. Every size above can be changed for the whole server or [for one bank](#per-bank-settings).

**How an ingest runs:**

- **Files** are always ingested in the background: the call returns right away with the document ids and one operation id. Poll the operation, or each document's `status` (`pending`, `processing`, `ready`, `failed`), to know when it can be searched.
- **Text** sent as JSON is ingested before the call returns, unless you ask for `async`.
- **Same id, new content:** ingesting a document id again replaces that document. Its old passages stay searchable until the new ones are ready, then they swap in all at once.
- **Same id, same content:** nothing is done, as long as the tags, file name and bank settings are the same too and the old document is `ready`. The id comes back in `unchanged_document_ids`.
- **The original file is kept** in the server's [file storage](./configuration.md#file-storage) (PostgreSQL by default, or S3, GCS or Azure), so you can download it and stream the page or clip a result points to. A file that fails for good (it can't be read, or it's over a limit) is deleted.

Tags on a document let a search look at only part of a bank. See the [API reference](./api/knowledge-banks.md#ingest-files).

### How Text Is Split

Text, and the text of Office and web files, is split by its structure first:

- **A heading starts a new passage**, and waits for its content: a heading is never a passage on its own. A few words like "## Observations" alone used to match short questions better than any real answer.
- **Paragraphs are merged** until the passage reaches about 512 tokens (`text_passage_tokens`).
- **A paragraph bigger than that is cut by sentences.** If a heading was waiting, it leads the first piece.
- **No overlap** between passages: it didn't improve search.

Each passage is an exact slice of the original text, in order. It's embedded under its document's title: the title from its file name, or its first 40 words when the name gives none. So a clear file name helps search find the right document.

A small passage alone loses what its page says about it: a table's rows without the table's title, or the company named only in the footer. So each passage also gets a **window**: the consecutive passages around it, up to 2048 tokens, embedded once as one more vector shared by every passage in it. Search finds a passage by its own vector or its window's, whichever is closer, and still returns the small passage.

---

## Searching

A search takes text, an image, a short video, or any mix of them, embedded together as one query. It returns passages, best first.

**Figure: Searching a Knowledge Bank.** An animated diagram on the docs site; its narration, step by step:

- **search**
  1. A search takes text, an image, a short video, or several together. They become one query.
  2. Two searches run: one by meaning, one by keywords. Only the query's rarer words count as keywords, in any language.
  3. Each candidate is scored from all its vectors, with keyword matches mixed in. With a reranker set, the best go to it in one request: a text hit as its passage, a page as its picture and its text. Its order, blended with the search's order of documents, decides.
  4. The best hits come first, each with the text around it, until max_tokens is full.
  5. Results come back best first, each pointing to its page or its second in the original file.

**1. Find candidates.** Three searches run side by side:

- **by meaning:** the passages whose vectors are nearest to the query. A PDF page also has vectors for its text, in short sections and sentence by sentence, so a question can match a single line on a page. A text passage also has its [window](#how-text-is-split)'s vector.
- **by keywords:** only the query's rarer words take part (the rarest, and any other word that is rare in the bank), so "X200" counts and "the" doesn't. Words are matched as they are, with no stemming, so this works in any language, and a product code or a name finds its document whatever language the question is in.
- **by document:** every text passage of the three documents the meaning search ranks best is scored too, so the right passage of the right document is found even when it's worded unlike the question.

**2. Score them.** Each candidate is scored from all its vectors. On a PDF page, the text counts for 70% of the score and the page image for 30%. A text passage takes the better of its own score and its window's. Each score is then averaged with the best score in its document, so a passage of a document that matches well climbs. Keyword matches are then mixed in at 20%.

**3. Rerank them (optional).** With a reranker set (`HINDSIGHT_API_KB_RERANKER_URL` and `_MODEL`: any service that speaks Cohere's rerank API), the best candidates go to it in one request, in search order. Its order is then blended with the search's order of documents: the search is better at finding which document a question is about, the reranker at finding the passage inside it. The search's document order counts most when a few long documents hold all the candidates, and hardly at all when each candidate is its own document (a PDF page, an image or a clip counts as its own document here).

- **How deep it goes:** as many candidates as fit in `HINDSIGHT_API_KB_RERANKER_MAX_TOKENS` (120,000 tokens; at most 26,000 on a `low` budget), not a fixed count. The rest keep the search's order, and so do ties.
- **What the reranker reads:** a text hit as its passage, up to 512 tokens, under its document's title. Hits that would show it the same text (pieces of one short page) go once, as one candidate. An image or PDF page goes as its picture with its text; with `HINDSIGHT_API_KB_RERANKER_IMAGES=false`, or a picture under 280 or over 1568 pixels a side, it goes as its text. A video is not sent and keeps its place.
- **When it runs:** for a query with text, unless the search sets `enable_reranking: false`. A search by an image is reranked with the picture as the query too when `HINDSIGHT_API_KB_RERANKER_QUERY_IMAGES` is on (off by default: turn it on only for a reranker that takes an image query). A search by a video is reranked by its text alone. If the reranker can't be reached, the search fails with `503`.

Reflect's search of [linked knowledge banks](#linking-to-a-memory-bank) is reranked the same way.

**4. Fit the budget.** Two settings decide how much comes back:

- **`budget`** (`low`, `mid`, `high`) is how many scored candidates stay in the running: 50, 200 or 500. A higher budget reaches further down the list, but takes longer.
- **`max_tokens`** (default 4096) is how much text the results can add up to. The best hits go in first. Each text hit comes with the passages around it, so you get the paragraph, not a cut sentence. Any budget left over widens the best hits a little more. An image, page or clip costs a small fixed amount plus its text.

Results don't carry a score: scores are only comparable within one search, so the order is what to rely on. Pass `trace: true` to see everything: every candidate, its score at each step, the keywords used and their weight, what the budget kept, and how long each step took, rerank included. The control plane's search shows the same trace.

You can narrow a search by `tags`, with the same rules as memory recall (`tags_match`: `any`, `all`, `any_strict`, `all_strict`, `exact`; and `tag_groups` for and/or/not), to some `document_ids`, or to some `media_types` (`document`, `image`, `audio`, `video`, `text`). `GET /knowledge-banks/{bank_id}/tags` lists a bank's tags with how many documents carry each.

---

## Linking to a Memory Bank

A memory bank can read one or more knowledge banks. Add their ids to the memory bank's config:

### Python

```python
client.update_bank_config("support-agent", knowledge_banks=["product-manuals"])
```

### Node.js

```javascript
await client.updateBankConfig('support-agent', { knowledgeBanks: ['product-manuals'] });
```

### CLI

```bash
hindsight bank set-config support-agent --knowledge-banks product-manuals
```

### Go

```go
_, _, err = client.BanksAPI.UpdateBankConfig(ctx, "support-agent").
	BankConfigUpdate(hindsight.BankConfigUpdate{
		Updates: map[string]interface{}{"knowledge_banks": []string{"product-manuals"}},
	}).
	Execute()
```

### HTTP

```bash
curl -X PATCH "$HINDSIGHT_URL/v1/default/banks/support-agent/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"knowledge_banks": ["product-manuals"]}}'
```

From then on, when [reflect](./reflect.md) searches its memories, it also searches every linked knowledge bank. The passages reach the agent next to the memories, marked as documents, and filtered by the same tags as the reflect. **When a document and a memory disagree, the document wins**: the agent answers from the document and says the memory is out of date.

The link only reaches reflect. Recall doesn't return documents, and consolidation never reads them, so a manual is never folded into the bank's memories. How reflect searches linked banks can be tuned with the `HINDSIGHT_API_REFLECT_KB_SEARCH_*` settings in [Configuration](./configuration.md#reflect).

---

## Images, Audio and Video

A knowledge bank isn't limited to text. Every passage, whatever it came from, goes through the same embedding model, into one space:

- **Ask in text, find a picture or a clip.** A question can land on a chart in a PDF, a photo, or the 2:40 mark of a training video.
- **Search with an image or a clip.** Send a photo of a part, or a short video, and get back the pages and clips that look like it.
- **Results point to the moment.** A PDF hit points to its page and an audio or video hit to its seconds. The API streams just that page or clip.
- **Reflect sees the pictures.** When the reflect model can read images, a page or image hit from a linked knowledge bank reaches it as the picture, not only its text.

Which kinds of file a server takes depends on its embedder. The default, EmbeddingGemma 2, runs inside the server and reads text, images, audio and video. A text-only embedder takes text, and PDFs through their text layer. See [Configuration](./configuration.md#knowledge-banks) for the embedders and their settings.

---

## Over MCP

Connect an agent to `/mcp/kb/{bank_id}/` and it can search that knowledge bank, add text to it, and read its documents. See [Knowledge-Bank Tools](./mcp-server.md#knowledge-bank-tools).

---

## Per-Bank Settings

Each knowledge bank can change how its own files are split: the size of a text passage, the resolution of PDF pages, the length of audio and video windows, and which kinds of file it accepts. Everything else (the embedder, search, server limits) is the server's. See [Settings](./api/knowledge-banks.md#settings) in the API reference for every field.

---

## Ingestion Strategies

An ingestion strategy runs your own code on some file types before a knowledge bank splits them: unpack an archive, run OCR on scanned PDFs, or read a file format the bank doesn't know. Your code returns either more files, each ingested as its own document, or passages already cut. You set strategies up on the server (see [Configuration](./configuration.md#knowledge-banks)), and each ingest can pick one by name.
