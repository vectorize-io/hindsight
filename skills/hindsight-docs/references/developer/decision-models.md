

# Decision Models

A **decision model** answers short, typed questions — *yes or no*, *which of these*, *how likely* — in well under a second. Hindsight hands it the small judgment calls hidden inside recall, consolidation, reflect and knowledge-page refresh, and keeps the LLM for the part that needs writing.

It is optional. Without one, every operation works exactly as it always has; with one, the same operations make fewer LLM calls. You set it up once and every operation that supports it starts using it.

> **💡 Get one from Hindsight Inference**
>
[Hindsight Inference](https://inference.hindsight.vectorize.io) serves `hs-decide-1-flash`, a decision model trained for the judgments Hindsight makes, at $0.02 per million input tokens. It works with every operation, reflect and knowledge pages included. Sign in once and pick the `hindsight` provider:

```bash
hindsight-admin cloud login     # signs in to Hindsight Cloud and saves a key
export HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
```

Instead of signing in, you can set `HINDSIGHT_API_DECISION_MODEL_API_KEY=hsk_...` with a key from Hindsight Cloud. Recall keeps its reranker; to rank recall with the decision model too, set `HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER=hindsight`.

The weights are open, so you can also [run it yourself](#run-it-yourself).
---

## Judging Is Not Writing

Look at what Hindsight asks an LLM, and much of it is not writing at all. It is judging:

- *Would this memory help answer the question?*
- *Does this new fact say anything an observation doesn't already say?*
- *Is the evidence gathered so far enough to answer?*
- *Could this new memory change that page?*

An LLM can answer these, but each answer is a full generation call: seconds of latency, a bill for every token of context, and free-form text that has to be parsed back into a decision. Most of those calls end in a one-word verdict.

A decision model is built for exactly this shape of question. It returns a typed answer — a choice, or a probability — instead of prose, so there is nothing to parse and nothing to go off-script. Hindsight asks it the judgment calls, settles everything it can on the spot, and only calls the LLM when something actually has to be written.

The result is the same memory, built and read the same way, at a fraction of the LLM calls.

---

## Ranking What Recall Returns

Recall normally reranks its candidates with a cross-encoder, which scores how similar each memory is to the query. With a decision model, recall asks a sharper question of each candidate on its own: *would showing this help someone answer the question?* It also tells the model the date the question is asked, so a memory that was true last year and isn't anymore can lose to the one that replaced it.

**Figure: Recall with a Decision Model.** An animated diagram on the docs site; its narration, step by step:

- **recall()**
  1. recall() gets a question. Nothing about the call changes when a decision model is set up.
  2. The four search strategies run as usual and their results are fused into one pool of candidates.
  3. Instead of a cross-encoder, the decision model is asked about each candidate on its own: would showing it help answer? It is told the date the question is asked.
  4. Its answers are the order. Recent and on-topic wins; that order is final, with no recency or proof boosts on top.
  5. With pruning on, the candidates judged no help are dropped too, at no extra request. The best one always stays.

The answers are the order, and that order is final: recall does not add its usual recency, temporal and proof-count boosts on top. Recall can also drop the candidates judged no help at all instead of just ranking them last — the best candidate always survives. See [Recall](./retrieval.md#decision-model-ranking).

---

## Settling Facts Before the LLM

Consolidation turns new facts into [observations](./observations.md). For each fact, it finds the observations the fact could touch, then normally asks the LLM what to do. Most of the time, the answer is obvious: the fact says something the bank already knows, or nothing worth keeping.

**Figure: Consolidation with a Decision Model.** An animated diagram on the docs site; its narration, step by step:

- **consolidation**
  1. Retain stored four new facts. Consolidation turns facts into observations in the background.
  2. For each fact it finds the observations it could touch, as it always has.
  3. The decision model is asked how each fact relates to those observations: does it restate one, add something new, change one, or is it chatter?
  4. Three are settled with no LLM call: the restatement becomes evidence, the new fact becomes an observation as stated, the chatter is skipped.
  5. Only the fact that changes what an observation says reaches the LLM, which rewrites it as before.
  6. Four facts, one LLM call. On the consolidation eval this cut LLM calls by about 40% at the same pass rate.

With a decision model, those obvious cases are settled without the LLM. A fact that **restates** an observation becomes its evidence, and **chatter** is skipped. Every other fact — one that **changes** what an observation says, or adds something **new** — reaches the LLM, which writes the observation text: the decision model never does. On the consolidation eval this cut LLM calls by about 40% at the same pass rate. See [Observations](./observations.md#decision-model-triage).

---

## Knowing When Reflect Has Enough

[Reflect](./reflect.md) is an agent: in each turn the LLM picks one tool, reads what comes back, and decides what to do next. Every layer it searches — mental models, observations, facts — costs another LLM turn, and the LLM decides on its own when it has seen enough.

**Figure: Reflect with a Decision Model.** An animated diagram on the docs site; its narration, step by step:

- **reflect()**
  1. reflect() gets a question that needs reasoning.
  2. One LLM call turns the request into a search query.
  3. Every layer runs at once with that query: mental models, observations and facts. In agent mode each one would cost its own LLM turn.
  4. The decision model keeps the evidence that matters and drops the rest.
  5. Then it judges whether that evidence already answers the request. If not, the LLM writes the next query and another round runs.
  6. It is enough, so the next LLM call writes the answer.
  7. Two LLM calls in all. If the decision model fails, the LLM decides when to stop, as in agent mode.

With a decision model, reflect changes shape. One LLM call writes a search query, and **every layer runs at once** with it. The decision model then keeps the evidence that matters, drops the rest, and judges whether what's left already answers the request. If it does, the next LLM call writes the answer; if not, the LLM writes a new query and another round runs. A question that took an agent several turns now takes a couple of LLM calls. See [Reflect](./reflect.md#decision-model-orchestration).

---

## Refreshing Only What Changed

A [knowledge page](./knowledge-pages.md) refreshes incrementally: it only has to consider the memories that arrived since its last refresh. Without a decision model, that refresh still runs reflect to gather them and an LLM to judge them.

**Figure: Knowledge Page Refresh with a Decision Model.** An animated diagram on the docs site; its narration, step by step:

- **refresh**
  1. Five memories arrived since this page was last refreshed. A page refreshes incrementally, so only these need a look.
  2. Instead of running reflect, the refresh asks the decision model about each memory, with the page in front of it: could this change the page?
  3. Two could change it. The other three are dropped and not offered again.
  4. Only those reach the LLM that edits the page. When none could change it, there is no LLM call at all.
  5. On the refresh cost eval, a 20-memory refresh went from 3 LLM calls to 1, with about 70% fewer input tokens.

With a decision model, the refresh skips reflect. It asks, for each new memory, with the page in front of it: *could this change the page?* Only the memories that could reach the LLM that edits the page; the rest are dropped and not offered again. When none could, the refresh makes no LLM call at all. On the refresh cost eval, a 20-memory refresh went from 3 LLM calls to 1, with about 70% fewer input tokens. See [Knowledge Pages](./knowledge-pages.md#decision-model-refresh).

---

## Safe to Lose

A decision model only ever saves LLM calls; it never becomes the only way an operation can finish. When it fails or times out:

- **Consolidation** sends the fact to the LLM, as before.
- **Reflect** lets the LLM decide when to stop, as in agent mode.
- **Knowledge pages** run reflect for the refresh, as before.
- **Recall** is the exception: the decision model *is* its ranker, so a recall fails if it cannot be reached after retries — the same as with any remote reranker.

Reflect and knowledge-page refresh currently need the `hindsight` or `typesafe` provider; with another provider they keep their LLM-only path and log a warning, while recall and consolidation still use it.

Every decision-model call is recorded in the bank's model call log next to the LLM, embedding and reranker calls, so you can see what it was asked and what it answered: open a bank in the control plane and go to **LLM Requests**.

## Run It Yourself {#run-it-yourself}

All three Hindsight models are open weights on [Hugging Face](https://huggingface.co/hindsight-ai), so you can run the whole set on your own hardware:

| Model | What it does | Size | License | Runs on |
|---|---|---|---|---|
| [`hindsight-ai/hs-decide-1-flash`](https://huggingface.co/hindsight-ai/hs-decide-1-flash) | Decision model | 0.8B | Apache 2.0 | vLLM, one GPU |
| [`hindsight-ai/hs-rerank-1`](https://huggingface.co/hindsight-ai/hs-rerank-1) | Reranker | 114M | MIT | In the API process, or Text Embeddings Inference |
| [`hindsight-ai/hs-embed-1`](https://huggingface.co/hindsight-ai/hs-embed-1) | Embeddings | 112M | MIT | In the API process, or Text Embeddings Inference |

**Decision model.** Serve it on [vLLM](https://docs.vllm.ai) with the plugin in the model's `serving/` folder, which adds the `/v1/systemone` endpoint Hindsight calls:

```bash
hf download hindsight-ai/hs-decide-1-flash --local-dir hs-decide-1-flash
pip install vllm==0.32.0 ./hs-decide-1-flash/serving
```

Then start it with the launch script from the [model card](https://huggingface.co/hindsight-ai/hs-decide-1-flash#vllm), adding `--api-key`, and point the `hindsight` provider at your server:

```bash
export HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
export HINDSIGHT_API_DECISION_MODEL_BASE_URL=http://localhost:8000
export HINDSIGHT_API_DECISION_MODEL_MODEL=hs-decide-1-flash
export HINDSIGHT_API_DECISION_MODEL_API_KEY=<the server's --api-key>
```

As with the hosted model, recall keeps its reranker unless you also set `HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER=hindsight`.

**Reranker and embeddings.** These are small enough to run inside the Hindsight API process, on CPU, with no extra server. They download from Hugging Face on first start:

```bash
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=hindsight-ai/hs-rerank-1
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=hindsight-ai/hs-embed-1
```

Memories already stored need re-embedding after you switch embedding models. To serve either one on Text Embeddings Inference instead, see [Models › Embeddings](./models.md#embeddings-local) and [Models › Reranker](./models.md#reranker-local).

---

## Next Steps

- [**Hindsight Inference**](https://inference.hindsight.vectorize.io) — Get a key for `hs-decide-1-flash`
- [**Hugging Face**](https://huggingface.co/hindsight-ai) — The open weights for all three models
- [**Configuration › Decision model**](./configuration.md#decision-model) — Set it up, turn it off, tune it
- [**Models › Decision Model**](./models.md#decision-model) — Providers and default models
- [**Recall**](./retrieval.md#decision-model-ranking) · [**Observations**](./observations.md#decision-model-triage) · [**Reflect**](./reflect.md#decision-model-orchestration) · [**Knowledge Pages**](./knowledge-pages.md#decision-model-refresh) — Each operation in detail
