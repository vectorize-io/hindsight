---
title: "The Fully Open Agent Memory Stack: Self-Hosting Hermes + Hindsight"
authors: [benfrank241]
slug: "2026/07/17/hermes-hindsight-open-stack"
date: 2026-07-17T12:00
tags: [hindsight, hermes, agent-memory, open-source, self-hosted, persistent-memory]
description: "Most agent-plus-memory stacks have a closed core: a frontier API model, a proprietary memory service, or both. Hermes and Hindsight are the exception. Every layer is open source and can run on your own hardware with no external calls."
image: /img/blog/hermes-hindsight-open-stack.png
hide_table_of_contents: true
---

![The fully open agent memory stack: an open-weights model, an open agent, and open memory, all self-hosted](/img/blog/hermes-hindsight-open-stack.png)

Most "AI agent with long-term memory" setups have at least one part you cannot host yourself. The model is a frontier API behind a paywall, or the memory is a proprietary service that stores your agent's history on someone else's servers, or both. You can move the wrapper code to your infrastructure, but the two things that actually see all your data, the model and the memory, stay in the cloud.

Hermes plus Hindsight is the exception. The model weights are open, the agent is open source, and the memory system is open source. Every layer can run on your own hardware, and in the default configuration the memory layer makes no external network calls at all. This post is the grounded, layer-by-layer version of that claim: what is actually open, under which license, and how the pieces wire together into a stack you fully own.

<!-- truncate -->

## TL;DR

- **The model:** Hermes is an open-weights family from Nous Research. You serve it locally with vLLM, llama.cpp, or Ollama.
- **The agent:** [Hermes Agent](https://github.com/NousResearch/hermes-agent) is MIT-licensed and model-agnostic. It talks to any OpenAI-compatible endpoint, so it drives your local model with no code changes.
- **The memory:** [Hindsight](https://vectorize.io/hindsight) is MIT-licensed. Self-host it with Docker or pip. Embeddings and reranking run locally by default, and the LLM it uses for extraction can be local too, so a self-hosted server needs no external calls and no API key.
- **The wiring:** `hermes config set memory.provider hindsight`, point it at your local Hindsight server, point Hermes at your local model, and the whole loop runs on your box.
- **The honest caveats:** local models need at least 64K context, automatic recall/retain needs a recent Hermes build, and the Hermes model licenses differ by version. Details below.

## Why "fully open" is rare

Agent memory is the part of the stack with the most sensitive data. It is a durable record of everything your agent has seen: conversations, decisions, code, customer details. That is exactly the data you least want to hand to a third-party service you cannot inspect or host.

The common stacks do hand it over. A closed model means every prompt, including the memories injected into it, goes to an external API. A proprietary memory service means your agent's entire history lives in someone else's database. "Open" often means only the glue code is open while the model and the memory, the two components that see everything, are not.

A fully open stack closes both gaps. Here are the three layers.

## Layer 1: the model (open weights)

Hermes is a family of open-weights models from Nous Research. The weights are published on Hugging Face and you can download and run them offline like any other open model.

Two precise points, because the details matter for anyone who cares about licensing:

- **Hermes 4** (14B, 70B, 405B) is a hybrid-reasoning family built on Llama 3.1 checkpoints. Because the base is Llama, these models carry the **Llama 3 Community License**, not a fully permissive one.
- **Hermes 4.3** (36B) is built on ByteDance's Seed-OSS-36B base and is released under **Apache 2.0**, which is the cleaner open-source story and a good fit for consumer or single-GPU self-hosting.

Either way, this is a standard open-weights model you can serve yourself. The Hugging Face cards give explicit instructions for vLLM and SGLang, and GGUF quantizations run in llama.cpp, Ollama, and LM Studio. That is the first layer: an actual model, running on your hardware, that never phones home.

## Layer 2: the agent (open source, model-agnostic)

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is Nous Research's agent framework, and it is MIT-licensed. Importantly, it is *not* tied to the Hermes model. It is model-agnostic: you point it at any OpenAI-compatible endpoint and switch models with `hermes model`, no code changes.

That is what lets the open model and the open agent meet. Serve a Hermes model locally and point the agent's `base_url` at it. In `~/.hermes/config.yaml`:

```yaml
model:
  default: NousResearch/Hermes-4.3-36B
  provider: custom
  base_url: http://localhost:8000/v1   # your local vLLM server
  context_length: 64000
```

The local backends all use the same pattern, just a different port: Ollama on `11434`, vLLM on `8000`, llama.cpp's server on `8080`, LM Studio on `1234`. Two practical notes from the setup docs worth knowing before you start:

- **Context length:** Hermes Agent requires a model with at least **64,000 tokens** of context and rejects smaller ones at startup. This is the single most common local-model snag.
- **Tool calling:** for tool use to work, vLLM needs `--enable-auto-tool-choice --tool-call-parser hermes`, and llama.cpp's server needs the `--jinja` flag. Memory operations are tool calls, so this matters.

At this point you have an open agent driving an open model, entirely on your machine. What is missing is memory: by default the agent is stateless across sessions, or it uses a built-in memory tool that just writes local markdown files. That is the third layer.

## Layer 3: the memory (open source, local by default)

Hindsight is MIT-licensed, and the whole thing self-hosts. The important part for a "fully open" claim is not just the license, it is that a self-hosted Hindsight server can run with **no external dependencies and no external calls**.

Start a server with Docker:

```bash
docker run -it --pull always --name hindsight --restart unless-stopped \
  -p 8888:8888 -p 9999:9999 \
  -v hindsight-data:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

Or with pip: `pip install hindsight-api` and run `hindsight-api`. The API comes up on `localhost:8888`, and a local server has **no authentication by default**, so it needs no API key.

Where does the data live, and what talks to the outside world? By default, nothing does:

- **Database:** Hindsight ships with an embedded PostgreSQL ("pg0"), so there is no separate database to run. Point it at your own PostgreSQL 14+ with a vector extension for production, or keep the embedded one for a local stack. On Windows, the embedded database means no Docker and no WSL.
- **Embeddings:** the default provider is `local` (`BAAI/bge-small-en-v1.5`), computed in-process. No embeddings API.
- **Reranking:** the default reranker is also `local` (a cross-encoder model). No rerank API.
- **The LLM:** Hindsight uses an LLM to extract facts on `retain` and to reason on `reflect`. This can be local too. Set `HINDSIGHT_API_LLM_PROVIDER=llamacpp` and it runs built-in inference with no external server (it auto-downloads a small GGUF on first run), or point it at your Ollama endpoint. `recall` itself, the search path, is just local embeddings and reranking, so retrieval never needs an LLM at all.

So the memory layer is genuinely local: local database, local embeddings, local reranking, and, if you want, a local extraction model. Nothing about storing or recalling a memory has to leave your network.

## Wiring it together

Hermes has a pluggable memory-provider system. The built-in provider writes markdown files; Hindsight is a selectable provider you turn on with one setting.

```bash
hermes config set memory.provider hindsight
```

Then tell Hermes where your self-hosted server is, in `~/.hermes/.env`:

```bash
HINDSIGHT_API_URL=http://localhost:8888
# No HINDSIGHT_API_KEY needed for a local server with auth disabled
```

Once the provider is active, Hermes uses Hindsight three ways:

- **Auto-recall:** on every turn, before the model is called, Hermes queries Hindsight for relevant memories and injects them into the system prompt (a `pre_llm_call` hook).
- **Auto-retain:** after each response, it retains the exchange to Hindsight (a `post_llm_call` hook).
- **Explicit tools:** `hindsight_recall`, `hindsight_retain`, and `hindsight_reflect` are available to the model for direct control.

You can pick the balance with the memory mode: `hybrid` (auto injection plus tools), `context` (auto injection only), or `tools` (tools only, no injection).

One version caveat to flag, because it is easy to miss: the automatic recall and retain hooks require a recent Hermes Agent build (the ones that added the `pre_llm_call` and `post_llm_call` hooks). On older versions the three explicit tools still register, but the automatic injection is silently skipped. If auto-memory does not seem to fire, update Hermes first.

There is also an even simpler path if you do not want to run a separate server: the Hindsight provider has a `local` mode that starts an embedded Hindsight with a built-in database automatically. It is the fastest way to try the stack. Running your own server on `localhost:8888` is the better choice when you want one memory store shared across several agents or machines, or full control over the extraction LLM.

## What the finished stack looks like

Put together, every box in the diagram is open and on your hardware:

| Layer | Component | License | Runs where |
|---|---|---|---|
| Model | Hermes 4 / 4.3 weights | Llama 3 / Apache 2.0 | vLLM, llama.cpp, Ollama, local |
| Agent | Hermes Agent | MIT | your machine |
| Memory | Hindsight | MIT | your machine (embedded DB, local embeddings) |

No frontier API model. No proprietary memory service. No required external network call in the memory path. The agent reasons with an open model, and it remembers with open memory, and both the reasoning and the remembering happen on infrastructure you control.

## When this actually matters (and when it does not)

Being able to self-host the whole stack is not free. You provide the GPU for the model, you run the server, and an open model at a size you can host may not match a frontier API model on the hardest tasks. So be honest about when the fully-open stack earns its keep:

- **Air-gapped or regulated environments** where data cannot leave your network. This is the case where a closed model or a memory SaaS is simply disqualified, and the open stack is the only option.
- **Data residency and privacy** requirements where "the vendor promises not to look" is not good enough, and you need the memory of your agent to physically stay on your systems.
- **No per-token cost and no lock-in.** Local inference has no metered API bill, and because both the agent and the model are swappable, you are not tied to any one provider.

If none of those apply, a hosted model and Hindsight Cloud will be less operational work, and that is a perfectly good choice. The point is not that everyone should self-host everything. The point is that with Hermes and Hindsight, you *can* self-host everything, including the two layers that see all your data, and very few agent-memory stacks let you do that.

## Further reading

- [Hermes Agent persistent memory with Hindsight](/sdks/integrations/hermes): the full integration reference, including memory modes.
- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): what the memory layer actually does with each exchange it stores.
- [One bank or many?](/blog/2026/07/16/bank-strategy-agent-memory): how to scope the memory once you have it running.
