

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
| Text | passages of about 256 tokens that follow the document's structure; a heading stays with the content under it | `.txt` `.md` `.json` `.jsonl` `.csv` `.yaml` `.xml` `.rst` |
| Office and web | their text, then split as text | `.docx` `.pptx` `.xlsx` `.xls` `.html` `.epub` |

Audio and video need the `ffmpeg` binary on the server's `PATH`. Every size above can be changed for the whole server or [for one bank](#per-bank-settings).

**How an ingest runs:**

- **Files** are always ingested in the background: the call returns right away with the document ids and one operation id. Poll the operation, or each document's `status` (`pending`, `processing`, `ready`, `failed`), to know when it can be searched.
- **Text** sent as JSON is ingested before the call returns, unless you ask for `async`.
- **Same id, new content:** ingesting a document id again replaces that document. Its old passages stay searchable until the new ones are ready, then they swap in all at once.
- **Same id, same content:** nothing is done. The id comes back in `unchanged_document_ids`.
- **The original file is kept** in the server's [file storage](./configuration.md#file-storage) (local disk, S3, GCS or Azure), so you can download it and stream the page or clip a result points to.

Tags on a document let a search look at only part of a bank. See the [API reference](./api/knowledge-banks.md#ingest-files).

---

## Searching

A search takes text, an image, a short video, or any mix of them, embedded together as one query. It returns passages, best first.

**Figure: Searching a Knowledge Bank.** An animated diagram on the docs site; its narration, step by step:

- **search**
  1. A search takes text, an image, a short video, or several together. They become one query.
  2. Two searches run: one by meaning, one by keywords. Only the query's rarer words count as keywords, in any language.
  3. Each candidate is scored from all its vectors, with keyword matches mixed in. The best hits come first, each with the text around it, until max_tokens is full.
  4. Results come back best first, each pointing to its page or its second in the original file.

**1. Find candidates.** Two searches run side by side:

- **by meaning:** the passages whose vectors are nearest to the query. A PDF page also has vectors for its text, in short sections and sentence by sentence, so a question can match a single line on a page.
- **by keywords:** only the query's rarer words take part, so "X200" counts and "the" doesn't. Words are matched as they are, with no stemming, so this works in any language, and a product code or a name finds its document whatever language the question is in.

**2. Score them.** Each candidate is scored from all its vectors. On a PDF page, the text counts for 70% of the score and the page image for 30%. Keyword matches are then mixed in at 20%.

**3. Fit the budget.** Two settings decide how much comes back:

- **`budget`** (`low`, `mid`, `high`) is how many scored candidates stay in the running: 50, 200 or 500. A higher budget reaches further down the list, but takes longer.
- **`max_tokens`** (default 4096) is how much text the results can add up to. The best hits go in first. Each text hit comes with the passages around it, so you get the paragraph, not a cut sentence. Any budget left over widens the best hits a little more. An image, page or clip costs a small fixed amount plus its text.

Results don't carry a score: scores are only comparable within one search, so the order is what to rely on. Pass `trace: true` to see everything: every candidate, its score at each step, the keywords used and their weight, what the budget kept, and how long each step took. The control plane's search shows the same trace.

You can narrow a search to some `tags` (any or all of them), some `document_ids`, or some `media_types` (`document`, `image`, `audio`, `video`, `text`).

---

## Linking to a Memory Bank

A memory bank can read one or more knowledge banks. Add their ids to the memory bank's config:

```bash
curl -X PATCH http://localhost:8888/v1/default/banks/support-agent/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"knowledge_banks": ["product-manuals", "return-policy"]}}'
```

From then on, when [reflect](./reflect.md) searches its memories, it also searches every linked knowledge bank at the same time, with the same search as above. Each bank's best hits are taken in turn, within the same `max_tokens`. The passages reach the agent next to the memories, marked as documents. **When a document and a memory disagree, the document wins**: the agent is told to answer from the document and say the memory is out of date.

If a linked bank can't be searched (it was deleted, or the caller may not read it), reflect skips it and goes on with the memories.

A link only reaches reflect:

- **Recall** over HTTP does not return documents.
- **Consolidation and mental model refreshes** never read them, so a manual is never folded into the bank's memories.
- **Exporting** a memory bank exports the link, not the knowledge bank's documents.

Permissions are checked for the caller of the reflect, not for a background job: a caller who can't read a linked knowledge bank doesn't get its passages.

---

## Choosing an Embedder

One embedder serves every knowledge bank on a server. What it can embed decides which kinds of files you can ingest.

| Provider | Model (default) | Text | Image | Audio | Video |
|---|---|:-:|:-:|:-:|:-:|
| `local` (default) | `google/embeddinggemma-2`, in-process, open weights | ✓ | ✓ | ✓ | ✓ |
| `gemini` | `gemini-embedding-2` | ✓ | ✓ | ✓ | ✓ |
| `vertex` | `gemini-embedding-2` on Vertex AI | ✓ | ✓ | ✓ | ✓ |
| `voyage` | `voyage-multimodal-3.5` | ✓ | ✓ | | ✓ |
| `cohere` | `embed-v4.0` | ✓ | ✓ | | |
| `openai` | any OpenAI-compatible `/v1/embeddings` server | set with `..._MODALITIES` (default `text,image`) | | | |
| `text` | any [memory bank embeddings provider](./models.md#embedding-model), text only | ✓ | | | |

- A file the embedder can't embed is refused: a text-only model refuses video. A PDF is embedded from its text layer alone when the model takes no images.
- `vertex` uses the memory banks' Vertex project, region and credentials. `text` uses the memory banks' own embedder, or its own one set with `HINDSIGHT_API_KNOWLEDGE_TEXT_EMBEDDINGS_*` (each takes the same values as the `HINDSIGHT_API_EMBEDDINGS_*` setting of the same name).
- **Pick the embedder before you ingest.** The first time it runs, the server fixes the vector size for every knowledge bank. To switch to a model with another size, delete the knowledge banks and ingest again: the server refuses to start knowledge banks with vectors of two sizes.

```bash
# Gemini Embedding 2 instead of the in-process model
export HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER=gemini
export HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_API_KEY=...
```

---

## Per-Bank Settings

A knowledge bank can change how its files are split, and nothing else: the embedder, keys, search weights and server limits stay the server's.

```bash
curl -X PATCH http://localhost:8888/v1/default/knowledge-banks/product-manuals/config \
  -H "Content-Type: application/json" \
  -d '{"updates": {"text_passage_tokens": 512, "disabled_media_types": ["video"]}}'
```

| Setting | What it changes | Range |
|---|---|---|
| `text_passage_tokens` | size of a text passage | 16–8192 |
| `pdf_page_pixels` | longest side of a rendered PDF page | 256–8192 |
| `pdf_bands` | also embed three horizontal strips of each PDF page: slightly better search, half the ingest speed | `true` / `false` |
| `image_tokens` | how much detail the embedder reads from each image | 70–1120 |
| `audio_window_seconds` | length of an audio window | 1–3600 |
| `video_window_seconds` | length of a video window | 1–3600 |
| `media_overlap_seconds` | overlap between windows | 0–3599 |
| `max_media_seconds` | longest audio or video file accepted | up to the server's limit |
| `max_pdf_pages` | longest PDF accepted | up to the server's limit |
| `disabled_media_types` | kinds of file this bank refuses | `document`, `image`, `audio`, `video`, `text` |

A `null` value goes back to the server's setting, and `DELETE .../config` resets them all. A change applies to files ingested after it: ingesting an unchanged file again splits it with the new settings. Writes need `HINDSIGHT_API_ENABLE_BANK_CONFIG_API` on, the same switch as memory bank config.

---

## Configuration

All server-wide; the split settings can also be changed [per bank](#per-bank-settings).

| Variable | Description | Default |
|---|---|---|
| `HINDSIGHT_API_ENABLE_KNOWLEDGE_BANKS` | Knowledge banks on or off. Off, their endpoints answer `404`, reflect ignores linked knowledge banks and the control plane hides them | `true` |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER` | Embedder: `local`, `gemini`, `vertex`, `voyage`, `cohere`, `openai`, `text` | `local` |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MODEL` | Model name | per provider, see [above](#choosing-an-embedder) |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_API_KEY` | API key for a remote provider (`voyage` also reads `VOYAGE_API_KEY`) | - |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_BASE_URL` | Base URL, for `openai` | - |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_DIMENSIONS` | Vector size to ask for, for models that support several | model's own |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MODALITIES` | What an `openai` server embeds, comma-separated | `text,image` |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_DEVICE` | Device for `local` (e.g. `cpu`, `cuda`, `mps`) | picked automatically |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_BATCH_SIZE` | Passages per embedder request | `32` |
| `HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_MAX_IN_FLIGHT` | Requests to a remote embedder at once, per process (`0`: no cap) | `16` |
| `HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES` | Kinds of file the server refuses, comma-separated (e.g. `video,audio`) | - |
| `HINDSIGHT_API_KNOWLEDGE_MAX_FILE_MB` | Largest file accepted | `500` |
| `HINDSIGHT_API_KNOWLEDGE_MAX_PDF_PAGES` | Longest PDF accepted | `2000` |
| `HINDSIGHT_API_KNOWLEDGE_MAX_MEDIA_SECONDS` | Longest audio or video file accepted | `7200` |
| `HINDSIGHT_API_KNOWLEDGE_TEXT_PASSAGE_TOKENS` | Size of a text passage | `256` |
| `HINDSIGHT_API_KNOWLEDGE_PDF_PAGE_PIXELS` | Longest side of a rendered PDF page | `1536` |
| `HINDSIGHT_API_KNOWLEDGE_PDF_BANDS` | Also embed three strips of each PDF page | `false` |
| `HINDSIGHT_API_KNOWLEDGE_PDF_TEXT_WEIGHT` | Share of a PDF page's score that comes from its text | `0.7` |
| `HINDSIGHT_API_KNOWLEDGE_KEYWORD_WEIGHT` | Share of the score that comes from keyword matches | `0.2` |
| `HINDSIGHT_API_KNOWLEDGE_IMAGE_TOKENS` | Detail the embedder reads from each image (70–1120) | `280` |
| `HINDSIGHT_API_KNOWLEDGE_AUDIO_WINDOW_SECONDS` | Length of an audio window | `60` |
| `HINDSIGHT_API_KNOWLEDGE_VIDEO_WINDOW_SECONDS` | Length of a video window | `30` |
| `HINDSIGHT_API_KNOWLEDGE_MEDIA_OVERLAP_SECONDS` | Overlap between windows | `5` |
| `HINDSIGHT_API_KNOWLEDGE_INGEST_CHILD_BYTES` | A background ingest is split into jobs of about this many bytes, so a big upload spreads across workers | `1048576` |
| `HINDSIGHT_API_KNOWLEDGE_INGEST_MEMORY_BUDGET_MB` | Memory one ingest job may hold before it writes what it has (`0`: no limit) | `128` |
| `HINDSIGHT_API_WORKER_KNOWLEDGE_INGEST_RESERVED_SLOTS` | Worker slots kept for knowledge ingests | `0` |
| `HINDSIGHT_API_INGEST_STRATEGIES` | [Ingestion strategies](#ingestion-strategies), as JSON | - |
| `HINDSIGHT_API_INGEST_DEFAULT_STRATEGY` | Strategy used when an ingest names none | - |
| `HINDSIGHT_API_KNOWLEDGE_STORE_EXTENSION` | [Your own store](#your-own-store), as `module.path:ClassName` | Postgres |

---

## Extending It

### Ingestion Strategies

An ingestion strategy runs your own code on some file types before the knowledge bank splits them: unpack an archive, run OCR on scanned PDFs, cut a file format the bank doesn't know. A strategy maps file types (an extension or a content type) to an extension class:

```bash
export HINDSIGHT_API_INGEST_STRATEGIES='{
  "game-assets": {
    ".unitypackage": {"extension": "acme.unity:Unpack"},
    ".pdf": {"extension": "acme.ocr:OcrPdf", "settings": {"dpi": 300}}
  }
}'
```

An ingest picks a strategy with `strategy` on each item, or gets `HINDSIGHT_API_INGEST_DEFAULT_STRATEGY`. A file whose type the strategy doesn't map is split as usual.

For each file, the extension returns either **files**, each ingested as its own document of the same bank (`<document_id>/<name>`, which records the document it came from), or **passages** already cut, which are embedded as they are. It runs on the worker with the ingest's retries: a `ValueError` fails the document for good, any other error is retried. Raise its `version` to make unchanged files go through it again.

### Your Own Store

Documents and passages live in Postgres by default, in their own tables with a pgvector index. To keep them somewhere else, implement `KnowledgeStore` (`hindsight_api/knowledge/store.py`) and set `HINDSIGHT_API_KNOWLEDGE_STORE_EXTENSION=module.path:ClassName`. The bank itself always stays in the `banks` table, so operations, audit and bank deletion work the same.

Access to every knowledge bank call goes through the [operation validator](./extensions.md) like the memory bank calls, with its own read and write operations.

---

## In the Control Plane

With knowledge banks on, the control plane lists them next to memory banks. Each knowledge bank has:

- a home page with its documents, passages and size over time;
- a documents list, where you can upload files or text, open a document to see its passages, and download the original;
- a search panel that takes text, an image or an MP4 clip, shows each result's page or clip, and the full trace;
- settings, with the bank's ingest operations and its [per-bank settings](#per-bank-settings).

A memory bank's settings show the knowledge banks it links, where you can add or remove them.

---

## Deleting

Deleting a document removes its passages, its stored file, and any documents an [ingestion strategy](#ingestion-strategies) split out of it. Deleting a knowledge bank first cancels its queued and running ingests, then removes its documents in batches, so a bank with a hundred thousand documents deletes without timing out. Memory banks that link a deleted knowledge bank keep working; reflect skips the missing bank.
