---
title: "What's new in the Hindsight Hermes plugin 2.0.0"
description: Per-chat settings, page tools, dates on every recalled memory, a mental model in the system prompt, saves before compression, quieter cron and kanban runs, and a long round of fixes for the Hindsight memory provider in Hermes Agent
authors: [nicoloboschi]
date: 2026-11-26
hide_table_of_contents: true
tags: [release, hermes]
image: /img/blog/hermes-agent-memory.png
---

Version 2.0.0 of the Hindsight memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent) lets one profile use different memory settings in each chat, topic and platform. It also gives the agent tools for pages, dates every memory it recalls, and can put a mental model or page in the system prompt. It's a major version because a few defaults changed: cron runs stop auto-recalling, kanban workers stop writing, and cloud mode now needs an API key. Read [what changed by default](#what-changed-by-default) before you update.

<!-- truncate -->

<div style={{position: "relative", paddingBottom: "56.25%", height: 0, margin: "1rem 0 2rem", borderRadius: "8px", overflow: "hidden"}}>
  <iframe
    style={{position: "absolute", top: 0, left: 0, width: "100%", height: "100%"}}
    src="https://www.youtube.com/embed/6xqiECfcUUM"
    title="What's new in the Hindsight Hermes plugin 2.0.0"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  ></iframe>
</div>

- [**Per-chat settings**](#per-chat-settings): Change any setting for one platform, chat or forum topic.
- [**Page tools**](#page-tools): The agent can search, read and list pages, and create one when you ask.
- [**Dates on recalled memories**](#dates-on-recalled-memories): The model can tell an old fact from a fresh one.
- [**Background in the system prompt**](#background-in-the-system-prompt): A mental model or page in every session, from the first turn.
- [**More control over recall**](#more-control-over-recall): Strip boilerplate from the query, exclude tags, share one token budget across banks, and set a separate reflect budget.
- [**More control over what gets saved**](#more-control-over-what-gets-saved): Keep built-in memory notes out of the bank, label turns written by other agents, credit relayed pages and videos to their source, and store corrections.
- [**Saves before compression**](#saves-before-compression): Support for Hermes' strict pre-compress checkpoint.
- [**What changed by default**](#what-changed-by-default): Cron, kanban, cloud keys and removed settings.
- [**Fixes**](#fixes): Memory on the first message, saves kept across exit, failed saves and recalls no longer silent, and more.

## Per-chat settings

One Hermes profile often serves several places at once: Telegram forum topics, a Discord DM, a capture channel, the CLI. Until now every setting applied to all of them. The new `chats` map changes settings for one place. Each key is `platform`, `platform/chat_id` or `platform/chat_id/thread_id`, and any setting can go in an entry:

```json
{
  "bank_id": "hermes",
  "chats": {
    "pocket": { "auto_recall": false },
    "telegram/-1009999999999": { "recall_additional_banks": ["peptides"] },
    "telegram/-1001234567890": { "bank_id_template": "project-a-{thread}" },
    "telegram/-1001234567890/42": { "bank_id": "home-automation" }
  }
}
```

The most specific key that matches wins. To find a chat's key, send one message there and look for `Hindsight initialized: chat=…` in `~/.hermes/logs/agent.log`. The new `{chat}` and `{thread}` placeholders also work in `bank_id_template` and in the tag settings, so `"retain_tags": ["topic:{thread}"]` tags every memory with the forum topic it came from. The `{platform}` and `{user}` placeholders work there too, so one shared bank can keep each person's memories to themselves: `"retain_tags": ["user:{platform}:{user}"]` with `"recall_tags": ["user:{platform}:{user}", "shared"]` and `"recall_tags_match": "any_strict"`. See [per-chat settings](/sdks/integrations/hermes#per-chat-settings--chats) for the details.

## Page tools

Pages are documents Hindsight writes from a bank's memories and keeps up to date. The agent now gets `hindsight_search_pages`, `hindsight_read_page` and `hindsight_list_pages`, so it can answer "what's in my notes about X" from the page instead of from scattered facts. These are on by default (`page_tools`). `hindsight_create_page` is off by default (`page_create_tool`), because every page costs model calls each time it refreshes. When it's on, the agent creates a page only when you ask.

## Dates on recalled memories

Every recalled memory, in auto-recall and in the `hindsight_recall` tool, now carries its dates:

```
- DeepSeek V4.1 Flash is on Nous Portal (mentioned: 2026-10-18) [occurred: 2026-10-18]
```

`mentioned` is when the fact was stated, and `occurred` is when the event happened. Dates are shown in Hermes' own time zone. The model can now tell "I checked this last week" from "this is months old, check again" without searching everything again.

## Background in the system prompt

Recall depends on the message, so it can't stand in for stable background: a profile of the user, a project's conventions. The new `session_context` setting puts mental models or pages in the system prompt when a session starts:

```json
{ "session_context": ["mental_model:user-profile", "page:project-conventions"] }
```

Each is read once per session, with no model call, so the prompt stays the same for the whole session and prompt caching keeps working. A refreshed mental model shows up in the next session.

## More control over recall

- **`recall_query_strip_patterns`** removes lines that match a regex from the auto-recall query. Use it for a header that repeats on every message: it no longer fills the start of the query and drowns out the actual question.
- **`recall_exclude_tags`** keeps memories with a tag out of recall, while they stay in the bank. Until now you could only require tags.
- **`recall_max_tokens_mode: split`** divides `recall_max_tokens` across all the banks searched. Before, each extra bank got the full budget, so three banks could inject three times the setting.
- **`reflect_budget`** sets reflect's thoroughness on its own. Keep reflect at `low` on a large bank while recall stays at `mid`.

## More control over what gets saved

- **`mirror_builtin_memory: false`** stops notes the agent writes to Hermes' built-in memory (`MEMORY.md`, `USER.md`) from being copied into the bank. It's useful when you keep the bank curated.
- **Turns written by another agent** are saved under that agent's name instead of `User:`, so its words are no longer stored as yours. This needs a Hermes version that says who wrote each turn.
- **Summaries of a page or video are credited to their source.** When the agent summed up someone else's video, the creator's "I tested…" was saved as your own experience, or the agent's. Each save now lists the tools the agent called that turn (names and short arguments, never their output) and says that what a tool fetched belongs to its source.
- **Corrections.** When you tell the agent a memory is wrong or out of date, it now saves the correction: what memory said, what's true now, and why. The newer fact then replaces the old one.

## Saves before compression

With Hermes' `compression.checkpoint_required: true`, Hermes only compresses a conversation after the memory provider confirms it saved what is about to be dropped. The plugin never declared it could do that, so in that mode Hermes **never compressed** a conversation with Hindsight as the provider. Now it does. Every turn is already saved as it happens, so a normal compression loses nothing and nothing extra is saved. In strict mode, the plugin also saves the messages that are about to be compressed, and waits for the server to confirm. Only then does Hermes compress them. If that save fails, Hermes keeps the full conversation. The save goes into its own document, tagged `checkpoint:pre-compress`. Kanban workers without `retain_kanban` can't write, so strict mode never compresses their conversations.

## What changed by default

- **Cron runs no longer auto-recall.** Every cron prompt starts with a fixed notice longer than the recall query, so cron runs searched for that notice instead of the job. Set `recall_cron: true` to turn auto-recall back on. The recall and reflect tools still work in cron runs.
- **Kanban workers read memory but don't write to it.** Their reports used to duplicate what the parent session already saves. Set `retain_kanban: true` when the workers' runs are the only record of their work.
- **Cloud mode needs an API key.** A cloud config with a URL and no key used to report itself as available, then send every request unauthenticated. For a self-hosted server without a key, use `local_external`.
- **`bank_mission` and `bank_retain_mission` are removed.** The plugin read them but never sent them to the bank. Set missions with a [bank template](https://hindsight.vectorize.io/templates) or in the bank's config.
- **The page tools are offered by default.** Set `page_tools: false` to hide them.
- **Newer default models for the local server.** With `local_embedded` and no `llm_model` set, `openai` now uses `gpt-5-nano` instead of `gpt-4o-mini`, and `minimax` uses `MiniMax-M3`. `gpt-5-nano` thinks before it answers, so calls are slower and the thinking tokens are billed. Set `llm_model` to keep the old model.

## Fixes

- **The first message of a session gets memory.** Auto-recall runs in the background and lands on the next turn, so the first message of every session got none. That's often the one that matters most: on a new device, "any reason not to take amoxicillin?" was answered without the penicillin allergy the bank held. The first turn now recalls for its own message and waits up to 3 seconds. If the server is slower, the turn goes ahead without memory and the result lands on the next turn, so a slow server never adds latency.
- **Saves still unsent at exit are kept.** When Hermes exited, saves still queued after a 10-second wait were dropped, and so were saves waiting to be resent after a server outage. They are now written to `~/.hermes/hindsight/unsent-retains/` and sent when the next session starts. A hard crash can still lose saves that were only queued.
- **Failed saves are no longer silent.** The status line now says when saves are waiting to be retried, and when earlier saves were dropped, instead of always showing "saving to memory…".
- **A failed recall now says so.** The agent gets a short note that memory couldn't be checked this turn, so it doesn't conclude you never told it something.
- **A routed profile no longer reads another profile's shared config** in the Hermes gateway.
- **Recall is declared safe for read-only sub-agents**, so Hermes can give them recall without retain.
- **Tool results keep non-ASCII text** instead of escaping it, which uses about a third of the tokens for Chinese, Japanese and Korean.
- **Starter templates stop burning tokens.** The bank templates `hermes memory setup` offers rebuilt every mental model after every consolidation. On a busy bank that's a full model rebuild every few turns, and one report counted about 46M tokens in 10 hours. They now refresh hourly, and a refresh with nothing new to read is skipped. This applies to banks you set up from now on. To move an existing bank, set a `refresh_cron` on each of its mental models in the control plane.
- **The local server starts when uvx isn't on the PATH.** If Hermes ran without uvx on its PATH (a systemd service, Windows), `local_embedded` never started its server, and saves were dropped after their retries. The plugin now uses Hermes' own copy of uvx, also on Windows and with a custom `HERMES_HOME`. It no longer flashes a console window on Windows, and it needs `hindsight-embed` 0.10.3 or newer.
- **Tool errors show the server's message**, not the client's error text with its HTTP headers.
- **Long sessions no longer keep the ids of every finished save** when `prefetch_waits_for_retain` is off.
- **💭 replaces 👁️** in the status lines.

## Updating

Nothing updates the plugin on its own, and `hermes update` doesn't move it either. Pick the case that matches how you installed it.

**You follow the plugin catalog** (you ran `hermes plugins install hindsight`). The catalog pins a version Nous reviewed. Once that pin moves to 2.0.0, run:

```bash
hermes plugins update hindsight
```

Hermes re-reads the catalog at most every 6 hours, so `hermes plugins list` can take that long to show `update_available`.

**You want 2.0.0 now, before the catalog pin moves.** Install it pinned to the release's commit. `--ref` takes a full commit SHA, not a tag name, so look the SHA up first:

```bash
git ls-remote https://github.com/vectorize-io/hindsight 'refs/tags/integrations/hermes/v2.0.0^{}'
```

The first column is the SHA. Put it in place of `SHA`:

```bash
hermes plugins install vectorize-io/hindsight/hindsight-integrations/hermes --force --ref SHA
```

A pinned install stays on 2.0.0: `hermes plugins update hindsight` won't move it, even when the catalog moves. To change version, install again with a new `--ref`.

See [Updating](/sdks/integrations/hermes#updating) for every option, and the [Hermes integration reference](/sdks/integrations/hermes) for the full list of settings.

<details>
<summary>Every change in this release</summary>

| Change                                                                                         | Pull request                                                 | Issues                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-chat settings, `{chat}` and `{thread}` placeholders                                        | [#6900](https://github.com/vectorize-io/hindsight/pull/6900) | [#5490](https://github.com/vectorize-io/hindsight/issues/5490), [#6599](https://github.com/vectorize-io/hindsight/issues/6599), [#5723](https://github.com/vectorize-io/hindsight/issues/5723), [#5073](https://github.com/vectorize-io/hindsight/issues/5073), [#5984](https://github.com/vectorize-io/hindsight/issues/5984)                                                                 |
| First message of a session recalls its own memories                                            | [#6931](https://github.com/vectorize-io/hindsight/pull/6931) |                                                                                                                                                                                                                                                                                                                                                                                                |
| Unsent saves kept on disk at exit, sent next session                                           | [#6930](https://github.com/vectorize-io/hindsight/pull/6930) | [#6915](https://github.com/vectorize-io/hindsight/issues/6915)                                                                                                                                                                                                                                                                                                                                 |
| Relayed pages and videos credited to their source                                              | [#6913](https://github.com/vectorize-io/hindsight/pull/6913) | [#6818](https://github.com/vectorize-io/hindsight/issues/6818)                                                                                                                                                                                                                                                                                                                                 |
| Store corrections when a memory is wrong                                                       | [#6901](https://github.com/vectorize-io/hindsight/pull/6901) | [#5373](https://github.com/vectorize-io/hindsight/issues/5373) (in part)                                                                                                                                                                                                                                                                                                                       |
| Page tools                                                                                     | [#6897](https://github.com/vectorize-io/hindsight/pull/6897) | [#5289](https://github.com/vectorize-io/hindsight/issues/5289)                                                                                                                                                                                                                                                                                                                                 |
| Turns by other agents, `recall_exclude_tags`                                                   | [#6891](https://github.com/vectorize-io/hindsight/pull/6891) | [#5374](https://github.com/vectorize-io/hindsight/issues/5374), [#5375](https://github.com/vectorize-io/hindsight/issues/5375)                                                                                                                                                                                                                                                                 |
| Saves before compression                                                                       | [#6890](https://github.com/vectorize-io/hindsight/pull/6890) | [#6449](https://github.com/vectorize-io/hindsight/issues/6449)                                                                                                                                                                                                                                                                                                                                 |
| `mirror_builtin_memory`                                                                        | [#6889](https://github.com/vectorize-io/hindsight/pull/6889) | [#6725](https://github.com/vectorize-io/hindsight/issues/6725)                                                                                                                                                                                                                                                                                                                                 |
| Dates on memories, `recall_query_strip_patterns`, `session_context`                            | [#6886](https://github.com/vectorize-io/hindsight/pull/6886) | [#4697](https://github.com/vectorize-io/hindsight/issues/4697), [#5639](https://github.com/vectorize-io/hindsight/issues/5639), [#5285](https://github.com/vectorize-io/hindsight/issues/5285)                                                                                                                                                                                                 |
| `recall_max_tokens_mode: split`                                                                | [#6870](https://github.com/vectorize-io/hindsight/pull/6870) | [#5426](https://github.com/vectorize-io/hindsight/issues/5426)                                                                                                                                                                                                                                                                                                                                 |
| Save failures on the status line, and other fixes                                              | [#6869](https://github.com/vectorize-io/hindsight/pull/6869) | [#6819](https://github.com/vectorize-io/hindsight/issues/6819), [#6548](https://github.com/vectorize-io/hindsight/issues/6548), [#6601](https://github.com/vectorize-io/hindsight/issues/6601), [#6754](https://github.com/vectorize-io/hindsight/issues/6754), [#6064](https://github.com/vectorize-io/hindsight/issues/6064), [#6658](https://github.com/vectorize-io/hindsight/issues/6658) |
| `recall_cron`, read-only kanban workers, `retain_kanban`                                       | [#6703](https://github.com/vectorize-io/hindsight/pull/6703) | [#6145](https://github.com/vectorize-io/hindsight/issues/6145), [#6124](https://github.com/vectorize-io/hindsight/issues/6124), [#6419](https://github.com/vectorize-io/hindsight/issues/6419)                                                                                                                                                                                                 |
| `reflect_budget`                                                                               | [#6698](https://github.com/vectorize-io/hindsight/pull/6698) | [#6246](https://github.com/vectorize-io/hindsight/issues/6246)                                                                                                                                                                                                                                                                                                                                 |
| Failed-recall note, non-ASCII tool text, cloud key required, embed floor, removed mission keys | [#6693](https://github.com/vectorize-io/hindsight/pull/6693) | [#6482](https://github.com/vectorize-io/hindsight/issues/6482), [#6561](https://github.com/vectorize-io/hindsight/issues/6561), [#6438](https://github.com/vectorize-io/hindsight/issues/6438), [#6506](https://github.com/vectorize-io/hindsight/issues/6506), [#5601](https://github.com/vectorize-io/hindsight/issues/5601)                                                                 |
| Hourly template mental models, Hermes' uvx on Windows                                          | [#6681](https://github.com/vectorize-io/hindsight/pull/6681) | [#6451](https://github.com/vectorize-io/hindsight/issues/6451), [#6471](https://github.com/vectorize-io/hindsight/issues/6471), [#6524](https://github.com/vectorize-io/hindsight/issues/6524)                                                                                                                                                                                                 |
| Newer default models for the local server                                                      | [#6589](https://github.com/vectorize-io/hindsight/pull/6589) | [#6442](https://github.com/vectorize-io/hindsight/issues/6442)                                                                                                                                                                                                                                                                                                                                 |

</details>
