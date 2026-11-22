---
title: "Claude Code token usage: why the bill grows each turn"
description: "One developer found about 70% of their Claude Code token cost, at API prices, was re-read context. Keep sessions short, and let recall return what helps."
authors: [chrislatimer]
date: 2026-11-21
slug: "2026/11/21/claude-code-token-usage-recall-prune-dedup"
tags: [claude-code, coding-agents, recall, agent-memory, hindsight]
image: /img/blog/claude-code-token-usage-recall-prune-dedup/featured.png
---

import ThemedImage from '@theme/ThemedImage';

You run Claude Code for hours each day. You want the agent to finish the work. You also want the token bill to stay under control.

Then you check what you pay for. One developer described it as “re-reading context.”

<!-- truncate -->

## The bill is context size times number of turns

One developer logged 25,247 Claude Code turns over 14 days. Cache reads accounted for about 70% of their estimated cost at API prices. They counted 5.4 billion cache-read tokens and 11.8 million output tokens. [Developer’s count on r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1x93701/i_counted_14_days_of_claude_code_tokens_and_about/)

A developer who builds an agent on the Claude Agent SDK [wrote on r/AI_Agents](https://www.reddit.com/r/AI_Agents/comments/1s6ywrh/how_do_you_manage_conversation_history_token/):

> "Each request re-sends the full conversation history to the API, including all previous tool calls and their results."

Another developer grouped 18,629 Claude Code requests by session age. The first hour held 9,339 requests and 35% of the tokens. Hours 1 to 5 held 36% of the requests and 45% of the tokens. [Developer’s session logs on r/ClaudeCode](https://www.reddit.com/r/ClaudeCode/comments/1xvr4ob/i_grouped_18629_claude_code_requests_by_session/)

And a Hermes Agent user [put it in one line](https://www.reddit.com/r/hermesagent/comments/1wco7ab/my_hermes_agent_sends_more_than_140k_tokens_of/):

> "Right now, one PROMPT can eat up, almost 20$ of API requests... it's making it basicly unusable."

Before compaction, each turn can send more context than the last. You pay to read earlier tokens again. That can make a strong model too costly for daily work.

<ThemedImage
  alt="Before compaction, each request sends earlier conversation context again with the new turn."
  sources={{light: "/img/blog/claude-code-token-usage-recall-prune-dedup/context-growth-light.svg", dark: "/img/blog/claude-code-token-usage-recall-prune-dedup/context-growth-dark.svg"}}
/>

## Why a shorter session is hard

The obvious fix is to start a new session more often. The 18,629-request count shows why that helps: in that developer's logs, a request in the first hour carried about 183k tokens, and a request in hours 1 to 5 carried about 326k.

But a new session knows nothing about your project. One developer [described every new session](https://www.reddit.com/r/ClaudeCode/comments/1x8aikj/how_do_you_keep_claude_code_from_forgetting_your/) like this:

> "I explained how the server restarts, which port the CMS is on, that blog posts have to stay drafts. Next day, same thing again."

Another [wrote](https://www.reddit.com/r/ClaudeCode/comments/1qn5tfc/how_do_you_handle_context_loss_between_claude/):

> "Claude has no idea what we did yesterday, what decisions we made, or why the code is structured a certain way."

So you are stuck with a trade. A long session keeps the context and grows the bill. A short session keeps the bill down, and you explain the project again.

## What Hindsight changes

:::note
Hindsight does not shrink Claude Code's own transcript. Claude Code still sends its conversation on each turn. No setting in Hindsight changes that.
:::

Hindsight keeps project memory across sessions. You can start a new session and use memory from earlier work.

**One memory bank per repository.** The [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) gives each repository one bank, `coding-agent::{gitProject}`, shared by every agent you wire. In the words of the docs: "a repo's git history and conversations flow into its memory bank in the background as you work."

### When Claude Code gets memory

The `SessionStart` hook prepares the page roster and tool guide. In default reflect mode, `UserPromptSubmit` first tries low-budget reflect on prompt 1. If the server confirms the bank is empty at session start, the first attempt waits for prompt 2. [Session start code](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/core/session-start.ts)

After a timeout or 5xx, the fallback searches pages first. It attempts recall only when the page search finds no match or fails. The hook can retry reflect once on the next prompt if the failure leaves no cached answer. A permanent 4xx stops retries. [Prompt hook code](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/core/hook.ts)

The hook adds a reflect or fallback answer at most once per session, on the prompt where it gets one. By default, each prompt gets the page roster and tool guide. The agent can call `hindsight_reflect` on demand.

By default, the `Stop` hook sends the session transcript to the memory bank. It drops compaction summaries. [Stop hook](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/claude-stop-hook.ts) · [Transcript handling](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/core/transcript.ts)

<ThemedImage
  alt="SessionStart prepares the guide. Default reflect starts on prompt 1, or prompt 2 if the server confirms an empty bank. A timeout or 5xx starts with page search. Each prompt gets the roster and tool guide by default. The agent can request reflect. By default, Stop sends the transcript without compaction summaries."
  sources={{light: "/img/blog/claude-code-token-usage-recall-prune-dedup/prompt-flow-light.svg", dark: "/img/blog/claude-code-token-usage-recall-prune-dedup/prompt-flow-dark.svg"}}
/>

**A token budget on recall.** Recall returns facts up to a `max_tokens` limit. The [Recall docs](https://hindsight.vectorize.io/developer/api/recall) say: "set `max_tokens` to however much of your context window you want to allocate to memories." By default, the hook asks reflect for its answer, not plain recall.

The hook uses plain recall in `autoInject: "recall"` mode. In default reflect mode, the fallback can also use plain recall after the page search described above.

When recall runs, the default recall body asks for at most 2,000 tokens of observations: `{"types": ["observation"], "budget": "low", "max_tokens": 2000, ...}`. The hook adds its answer at most once per session. In recall mode, an empty result can keep the attempt open for up to five prompts. A failure can keep it open for up to two prompts.

To use recall in place of reflect, add this setting to `~/.hindsight/coding-agent.json`.

```json
{
  "autoInject": "recall"
}
```

Or set this environment variable.

```bash
HINDSIGHT_AUTO_INJECT=recall
```

**Recall that drops what does not help.** A budget alone has a gap. The [Hindsight Models post](https://hindsight.vectorize.io/blog/2026/11/03/hindsight-models) says it plainly: "Recall used to fill its token budget whether the memories helped or not." Since Hindsight 1.2.0, recall can remove memories with a [decision model](https://hindsight.vectorize.io/developer/decision-models). From the Recall docs:

- `prune` "drops the results judged no help to the query. It reuses the ranking answers, so it costs no extra request, and the best result always stays."
- `dedup` "drops a result that repeats a better-ranked one, so the same fact retained from several documents comes back once. It costs one more decision-model request, after the ranking one."

**Where prune and dedup act in the integration.** When reflect searches the bank's facts and observations, it uses recall, and those recalls use the bank's `prune` and `dedup` settings. When a fresh mental model already answers, reflect can skip recall, and prune and dedup do not act.

Prune and dedup act only when 3 things hold: you turn them on, recall ranks with a decision model (step 2), and the bank keeps `enable_reranking` on. Then the memories that the first-prompt reflect and the `hindsight_reflect` tool recall come back with the unhelpful results and the repeats removed. The same is true of the `"recall"` mode above. No source measures how much this changes the size of the answer or your bill.

The release itself is in [What's new in Hindsight 1.2.0](https://hindsight.vectorize.io/blog/2026/11/03/version-1-2-0) and the [Hindsight Models post](https://hindsight.vectorize.io/blog/2026/11/03/hindsight-models). This post is about your bill, so it shows only the parts you need.

**What the numbers say.** The Coding Agents page reports AMB's sdebench: "a coding agent over 61 real repository tasks, three runs each, with and without memory."

| Measure | Count or result from the named source | Source |
| --- | --- | --- |
| API-price cost over 14 days for one developer | About 70% cache reads, 21% cache writes, under 8% output | [r/ClaudeAI count](https://www.reddit.com/r/ClaudeAI/comments/1x93701/i_counted_14_days_of_claude_code_tokens_and_about/) |
| Tokens per request by session age for one developer | About 183,000 in the first hour and 326,000 in hours 1 to 5 | [r/ClaudeCode logs](https://www.reddit.com/r/ClaudeCode/comments/1xvr4ob/i_grouped_18629_claude_code_requests_by_session/) |
| Claude Code mean cost per task | $0.445 without memory → $0.338 with Hindsight | [Coding Agents benchmark](https://hindsight.vectorize.io/sdks/integrations/coding-agents#does-it-work) |
| Claude Code corrections per task | 0.85 without memory → 0.36 with Hindsight | [Coding Agents benchmark](https://hindsight.vectorize.io/sdks/integrations/coding-agents#does-it-work) |
| Tasks solved per benchmark run | 60 to 61 of 61, with or without memory | [Coding Agents benchmark](https://hindsight.vectorize.io/sdks/integrations/coding-agents#does-it-work) |
| Hosted `hs-decide-1-flash` | $0.02 per million input tokens | [Decision Models](https://hindsight.vectorize.io/developer/decision-models) · [Hindsight Models](https://hindsight.vectorize.io/blog/2026/11/03/hindsight-models#hindsight-inference) |

This benchmark compares memory with no memory. It does not measure prune or dedup on their own, and no source measures a drop in Claude Code's cache reads from them.

## The plan: 3 steps

Step 2 uses server settings. The commands below are for a Hindsight server that you run: self-hosted, or the local daemon that the integration starts. A self-hosted server in the docs listens on `http://localhost:8888`. The local daemon listens on `127.0.0.1:9077`.

### Step 1: Give your repository a memory

Run the installer for Claude Code. [Coding Agents installation](https://hindsight.vectorize.io/sdks/integrations/coding-agents#install)

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

This adds 3 hooks to `~/.claude/settings.json`, an MCP server, and a companion skill. On a terminal, the installer asks where memory lives. The default answer is Hindsight Cloud.

For this plan, pick `daemon` or `self-hosted` at the prompt, because step 2 uses settings on a server you run. To script it, pass `--server`:

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code --server daemon
npx @vectorize-io/hindsight-coding-agents install claude-code --server self-hosted --api-url http://localhost:8888
```

The daemon needs `uv` on your PATH and an LLM for fact extraction. That LLM comes from an API key, or from the Claude Code CLI, which needs no key. If you also use Codex CLI or Cursor CLI on the same repository, `install all` wires every agent it detects to the same bank.

### Step 2: Let recall drop what does not help

Prune and dedup need a decision model, and recall must rank with it. The hosted model is `hs-decide-1-flash` on Hindsight Inference. Sign in once, and set the providers where your Hindsight server runs:

```bash
hindsight-admin cloud login     # signs in to Hindsight Cloud and saves a key
export HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
export HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER=hindsight
export HINDSIGHT_API_RECALL_DECISION_MODEL_PRUNE=true
export HINDSIGHT_API_RECALL_DECISION_MODEL_DEDUP=true
```

What each line does:

- With `HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight` and no `HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER` line, recall keeps its usual reranker, and `prune` and `dedup` are ignored.
- The `PRUNE` and `DEDUP` variables turn the two options on for every bank that has no setting of its own. Both are `false` unless you set them.
- Prune and dedup act only when recall runs its rerank step. A bank with `enable_reranking` set to `false`, or a server with `HINDSIGHT_API_ENABLE_RERANKING=false`, skips that step, and then prune and dedup do not act.
- In place of `hindsight-admin cloud login`, you can set `HINDSIGHT_API_DECISION_MODEL_API_KEY=hsk_...` with a key from Hindsight Cloud.
- For the local daemon, export the variables before the daemon starts. The integration forwards every `HINDSIGHT_API_*` variable to the daemon. It adopts a daemon that already runs and does not restart it.

With these settings, the recalls inside the first-prompt reflect use prune and dedup, unless `enable_reranking` is `false` for the bank, by its own setting or by the server default. So do the `hindsight_reflect` tool and the recalls your own agent sends when a request leaves `rerank` out.

Both pricing pages list $0.02 per million input tokens for `hs-decide-1-flash`. The Hindsight Models post lists $0.005 for embeddings and $5 in free credit for new accounts. The decision model weights use Apache 2.0. You can [run the model yourself](https://hindsight.vectorize.io/developer/decision-models) with vLLM on one GPU.

:::warning
For recall, the decision model is the ranker. If the server cannot reach it after retries, the recall fails.

To rank with the usual reranker in that case, set `HINDSIGHT_API_RECALL_DECISION_MODEL_FALLBACK=true`. It is off unless you set it. A recall that the fallback ranks does not use prune or dedup.
:::

To turn prune and dedup on for one bank only (Hindsight 1.2.1 or later), skip the two `PRUNE` and `DEDUP` exports and use the bank config API. Put your bank's name in place of `coding-agent::my-repo`:

```bash
curl -X PATCH "http://localhost:8888/v1/default/banks/coding-agent::my-repo/config" \
  -H "Content-Type: application/json" \
  -d '{"updates": {"recall_decision_model_prune": true, "recall_decision_model_dedup": true}}'
```

On the local daemon, use `http://127.0.0.1:9077` in place of `http://localhost:8888` in this command and the next one.

If you build your own agent, you can also ask for it on one recall request:

```bash
curl -X POST "http://localhost:8888/v1/default/banks/my-bank/memories/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do we run the tests?", "max_tokens": 2000, "rerank": {"prune": true, "dedup": true}}'
```

A request that leaves `rerank` out uses the bank's setting. A bank with no setting uses the server's environment variables.

### Step 3: End the session when the task changes

When one task is done, close Claude Code and start a new session for the next task. In default reflect mode, the hook tries reflect early in the new session. It adds an answer if it gets one. The new session does not carry the full transcript of the last task.

A shorter session limits how much transcript builds up before the next task. Earlier project decisions remain available in the memory bank. When recall ranks with the decision model and both options apply, it can drop unhelpful results and repeats.

## If you change nothing

Each turn of a long session keeps resending a context that grows, and you keep paying for it.

## What your work looks like after

You finish a task, close the session, and open a new one for the next task. An early memory answer can bring back how the tests run or which decisions the team made. With the settings above, if reflect recalls and the decision model ranks the results, prune and dedup can filter them.

You go from the developer who keeps one session open all day so the agent does not forget, to the developer who closes sessions without losing what they learned.

## Try it

Install the integration on the repository you work on most:

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

Then read the [rerank section of the Recall docs](https://hindsight.vectorize.io/developer/api/recall) and the [Decision Models](https://hindsight.vectorize.io/developer/decision-models) page before you turn on prune and dedup.
