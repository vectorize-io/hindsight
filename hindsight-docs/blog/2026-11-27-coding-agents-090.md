---
title: "What's new in Hindsight Coding Agents 0.9.0"
description: Memory again after a compaction, pinned pages at session start, one bank for many repos, branch-aware memories, hooks that stay fast when the server is down, and a long round of saving fixes for the Hindsight memory plugin for coding agents
authors: [nicoloboschi]
date: 2026-11-27
hide_table_of_contents: true
tags: [release, coding-agents]
image: /img/blog/coding-agents-050-four-harnesses.png
draft: true
---

<!-- DRAFT. TODO before publishing:
  - new agents section once their PRs merge (Amp, Muse Code, Crush, Command Code, Qoder), and add them to the intro and the table
  - a 0.9.0 cover image (the image above is the 0.5.0 one)
  - video embed, like the Hermes 2.0.0 post
  - set the date to the release day and drop `draft: true`
-->

Version 0.9.0 of [Hindsight Coding Agents](/sdks/integrations/coding-agents) gives your agent its memory back after a compaction, lets you pin pages that every session starts with, and lets one bank serve many repositories. It also keeps the hooks fast when the Hindsight server is slow or down, and fixes a long list of ways a conversation could be saved twice or not at all. A few defaults changed: the hook time limits are lower, and Prime Agent 0.9 is no longer supported. Read [what changed by default](#what-changed-by-default) before you update.

<!-- truncate -->

- [**Memory after a compaction**](#memory-after-a-compaction): The agent gets its memory and pages again once the conversation is compacted.
- [**Pages in every session**](#pages-in-every-session): Pin pages and mental models that go into context when a session starts.
- [**One bank for many repos**](#one-bank-for-many-repos): Each repo's searches stay on its own memories, and every memory records its branch.
- [**Fast hooks when the server isn't**](#fast-hooks-when-the-server-isnt): A slow or unreachable server no longer stalls every prompt.
- [**Nothing lost when saving**](#nothing-lost-when-saving): Retries, mid-turn messages, forked sessions, replies sent through a tool.
- [**Shorter, more accurate injected text**](#shorter-more-accurate-injected-text): The tool guide once per session, source counts on memories, wording for banks that aren't a code repo.
- [**What changed by default**](#what-changed-by-default): Lower time limits, Prime Agent 0.10, two skills instead of one.
- [**Fixes**](#fixes): Antigravity, Codex, Kimi, DeepSeek Harness, Pi, opencode, pages outside git, and more.

## Memory after a compaction

When a long conversation fills the context window, the agent compacts it: it keeps a summary and drops the rest. That also dropped the memory and the list of pages Hindsight had shown it, so the second half of a long session ran without memory.

A compaction now counts as a new session start. On the next prompt the agent gets its memory and pages again, the same way a new session does. This works in Claude Code, Codex, Qwen Code, Factory Droid, Cursor, Kimi Code, opencode, opencode 2, Kilo and Pi, with nothing to turn on.

## Pages in every session

Recall depends on the prompt, so it can't stand in for stable background: the architecture, the team's conventions, the initiative you're in the middle of. The new `sessionContext` setting puts the full content of chosen pages and mental models into context when a session starts:

```json
{ "sessionContext": ["page:<id>", "mental_model:<id>"] }
```

A bare id is a mental model. For long sessions, `autoInjectEveryTurns` runs the automatic search again every N turns, with the latest prompt as the question, so later decisions still reach the agent.

## One bank for many repos

By default every repository gets its own bank. Some teams want one bank for a whole set of repositories. `retainTags` already recorded which repo a memory came from, but every search still read the whole bank. The new `recallTags` setting narrows the automatic recall and reflect, and the `hindsight_reflect` tool, to the same tag:

```json
{
  "bankId": "team-memory",
  "retainTags": ["project:{gitProject}"],
  "recallTags": ["project:{gitProject}"]
}
```

Every memory saved from a git repository now also records the branch it was learnt on. A fact from a branch that never merged no longer reads as the project's current state. The new `{branch}` placeholder works in the tag settings, so `"branch:{branch}"` can be a tag. See [recording where a memory came from](/sdks/integrations/coding-agents#recording-where-a-memory-came-from).

## Fast hooks when the server isn't

With the server unreachable, starting a session took 21 seconds and every prompt 16, even with low time limits set. And a slow answer could push the prompt hook past the 30 seconds agents allow, which throws away the whole turn. Now:

- A request waits at most its own time limit to connect, at most 5 seconds, so an unreachable server costs seconds instead of tens of seconds.
- The page list is fetched alongside the memory search instead of after it.
- Listing pages no longer asks the server to check every page's freshness just to show their names.

## Nothing lost when saving

At the end of a turn the plugin saves the conversation. Each of these closes a way it could be saved twice, or not at all:

- **A failed save is retried**, and when the agent quits first, a later session sends it. This works for the hook agents and for the plugin agents (opencode, Kilo, Pi, DeepSeek Harness).
- **Messages you type while the agent is working** are saved. Claude Code used to skip them.
- **Forked and continued sessions** no longer save their parent's history a second time.
- **A conversation that goes back to an earlier state** is saved. The server used to skip it as a repeat.
- **A transcript that gets shorter**, when opencode 2 compacts it, no longer stops saving for good.
- **Replies the agent sends through a tool** can be kept. `keepToolInputs` names tools whose input is really a message to you, like a chat bridge's reply tool (Claude Code Channels for Telegram, Discord and others). Messages that arrive through Channels are saved too.
- **Codex jobs started by another client** can be skipped with `skipOriginators`, so a job Claude Code hands to Codex isn't saved twice.

## Shorter, more accurate injected text

- **The full tool guide goes out once per session.** Later page refreshes are one line, and search results no longer repeat `toolGuideExtra`.
- **Recalled observations show how many sources back them**, as `(sources: 12)`, so the agent can tell one restated fact from something learnt many times. This needs a server that returns the count; older servers show the memory as before.
- **`writeTools: false`** removes the two tools that write to the bank, and every instruction to call them.
- **`injectQueryMaxChars`** caps how much of a long prompt the automatic search uses as its question. Reflect still gets the whole prompt.
- **`pageScope` and `toolGuideMoments`** let a bank that isn't a code repository drop the "this repository" wording from its pages and from the tool guide.
- **Hook output is plain text** unless the agent shows colors, so color codes no longer leak into notices.

## What changed by default

- **Lower hook time limits.** `reflectTimeoutMs` is now 15 seconds and `injectTimeoutMs` 5 seconds, so reflect plus its fallback fits well inside the 30 seconds agents allow. Values you set yourself still win.
- **Prime Agent 0.10 or newer.** Prime Agent 0.10, the Rust rewrite, dropped the JavaScript extensions the plugin ran in, so `install prime-agent` reported success and memory stayed off. The plugin now connects through Prime Agent's MCP settings. Prime Agent 0.9 is no longer supported.
- **Two companion skills instead of one.** The skill was 742 lines and the agent loaded all of it every time. `hindsight-coding-agent` now covers everyday use in about 70 lines, and `hindsight-coding-agent-setup` holds install, configuration and diagnostics.
- **A Hindsight Cloud config with no API key turns memory off**, instead of sending requests Cloud refuses.
- **The installer never touches an MCP server of your own called `hindsight`.** To keep both, register ours under another name with `install --mcp-name <name>`.

## Fixes

- **Antigravity CLI keeps its recall.** Recall was sent in a field Antigravity drops after the first tool call. A message sent while the agent is working now gets its own recall too.
- **Large Codex sessions no longer crash at the end of a turn.** The reader held tool output it was about to throw away.
- **Kimi Code finds sessions** through Kimi's own session index.
- **DeepSeek Harness imports past conversations** from its new compressed format, and its memory tools work when the app starts outside an opted-in project.
- **Pi shows Hindsight notices**, and **opencode reflects are no longer cut off at 300 seconds.**
- **Pages work in folders that aren't git repositories.**
- **Saving an initiative without its page id updates the page with the same title**, instead of creating a second one once the first was moved into a subfolder.
- **`--import-conversations` uses each agent's own settings.**
- **`apiHeaders`** sends extra headers with every request, for a server behind Cloudflare Access or a similar proxy.

## Updating

The plugin updates itself: once a day a session start checks npm and installs a newer version (`autoUpdate`, on by default). To update now:

```bash
npx @vectorize-io/hindsight-coding-agents@latest install
```

See [Install](/sdks/integrations/coding-agents#install) for every option, and the [configuration reference](/sdks/integrations/coding-agents#reference) for the full list of settings.

<details>
<summary>Every change in this release</summary>

| Change | Pull request | Issues |
| --- | --- | --- |
| A compaction counts as a new session start | [#6974](https://github.com/vectorize-io/hindsight/pull/6974) | [#6527](https://github.com/vectorize-io/hindsight/issues/6527) |
| `sessionContext` | [#6993](https://github.com/vectorize-io/hindsight/pull/6993) | [#5714](https://github.com/vectorize-io/hindsight/issues/5714) |
| `recallTags` | [#6976](https://github.com/vectorize-io/hindsight/pull/6976) | [#4226](https://github.com/vectorize-io/hindsight/issues/4226) |
| Branch on saved memories, `{branch}` placeholder | [#6973](https://github.com/vectorize-io/hindsight/pull/6973) | [#6456](https://github.com/vectorize-io/hindsight/issues/6456) |
| Hooks stay fast when the server is slow or unreachable | [#6971](https://github.com/vectorize-io/hindsight/pull/6971) | [#6167](https://github.com/vectorize-io/hindsight/issues/6167), [#6553](https://github.com/vectorize-io/hindsight/issues/6553) |
| List pages without the staleness scan | [#7000](https://github.com/vectorize-io/hindsight/pull/7000) | [#6258](https://github.com/vectorize-io/hindsight/issues/6258) |
| Retry a failed save, replay it from a later session | [#6935](https://github.com/vectorize-io/hindsight/pull/6935) | [#6097](https://github.com/vectorize-io/hindsight/issues/6097) |
| Replay a plugin session whose save failed before the host quit | [#6980](https://github.com/vectorize-io/hindsight/pull/6980) | [#5845](https://github.com/vectorize-io/hindsight/issues/5845), [#6559](https://github.com/vectorize-io/hindsight/issues/6559) |
| Messages sent while a turn is running (Claude Code) | [#6934](https://github.com/vectorize-io/hindsight/pull/6934) | [#6480](https://github.com/vectorize-io/hindsight/issues/6480) |
| Forked sessions don't re-save their parent's history | [#6937](https://github.com/vectorize-io/hindsight/pull/6937) | [#6642](https://github.com/vectorize-io/hindsight/issues/6642) |
| Continued sessions don't re-save their old history | [#6982](https://github.com/vectorize-io/hindsight/pull/6982) | [#6960](https://github.com/vectorize-io/hindsight/issues/6960) |
| A revisited conversation state is saved | [#6996](https://github.com/vectorize-io/hindsight/pull/6996) | [#6988](https://github.com/vectorize-io/hindsight/issues/6988) |
| `keepToolInputs`, inbound Channels messages | [#6964](https://github.com/vectorize-io/hindsight/pull/6964) | [#6481](https://github.com/vectorize-io/hindsight/issues/6481), [#6633](https://github.com/vectorize-io/hindsight/issues/6633) |
| `skipOriginators` (Codex) | [#6965](https://github.com/vectorize-io/hindsight/pull/6965) | [#5507](https://github.com/vectorize-io/hindsight/issues/5507) |
| Injected text follows the config, stops repeating, `autoInjectEveryTurns` | [#6943](https://github.com/vectorize-io/hindsight/pull/6943) | [#6487](https://github.com/vectorize-io/hindsight/issues/6487), [#6597](https://github.com/vectorize-io/hindsight/issues/6597), [#6320](https://github.com/vectorize-io/hindsight/issues/6320), [#6502](https://github.com/vectorize-io/hindsight/issues/6502), [#5713](https://github.com/vectorize-io/hindsight/issues/5713) |
| Source counts on recalled observations | [#6972](https://github.com/vectorize-io/hindsight/pull/6972) | [#5078](https://github.com/vectorize-io/hindsight/issues/5078), [#6966](https://github.com/vectorize-io/hindsight/issues/6966) |
| `injectQueryMaxChars` | [#6968](https://github.com/vectorize-io/hindsight/pull/6968) | [#5089](https://github.com/vectorize-io/hindsight/issues/5089) |
| `pageScope`, `toolGuideMoments` | [#6994](https://github.com/vectorize-io/hindsight/pull/6994) | [#6526](https://github.com/vectorize-io/hindsight/issues/6526) |
| Two companion skills | [#6981](https://github.com/vectorize-io/hindsight/pull/6981) | [#6749](https://github.com/vectorize-io/hindsight/issues/6749) |
| Prime Agent 0.10+, drop 0.9 | [#6945](https://github.com/vectorize-io/hindsight/pull/6945) | [#5976](https://github.com/vectorize-io/hindsight/issues/5976) |
| Antigravity recall as `systemMessage` | [#6942](https://github.com/vectorize-io/hindsight/pull/6942) | [#6632](https://github.com/vectorize-io/hindsight/issues/6632) |
| Antigravity recall for mid-turn messages | [#6983](https://github.com/vectorize-io/hindsight/pull/6983) | [#6944](https://github.com/vectorize-io/hindsight/issues/6944) |
| Bun hosts: saving after compaction, Pi notices, reflect past 300 s | [#6941](https://github.com/vectorize-io/hindsight/pull/6941) | [#6314](https://github.com/vectorize-io/hindsight/issues/6314), [#5557](https://github.com/vectorize-io/hindsight/issues/5557), [#6431](https://github.com/vectorize-io/hindsight/issues/6431) |
| Codex reader memory, Kimi session index, DeepSeek Harness import | [#6938](https://github.com/vectorize-io/hindsight/pull/6938) | [#6556](https://github.com/vectorize-io/hindsight/issues/6556), [#6457](https://github.com/vectorize-io/hindsight/issues/6457), [#5981](https://github.com/vectorize-io/hindsight/issues/5981) |
| Pages outside git, DeepSeek Harness tools outside an opted-in project | [#6932](https://github.com/vectorize-io/hindsight/pull/6932) | [#5982](https://github.com/vectorize-io/hindsight/issues/5982), [#5817](https://github.com/vectorize-io/hindsight/issues/5817) |
| Initiative capture without a page id updates the same-titled page | [#7003](https://github.com/vectorize-io/hindsight/pull/7003) | [#6919](https://github.com/vectorize-io/hindsight/issues/6919) |
| `--import-conversations` applies each agent's settings | [#6984](https://github.com/vectorize-io/hindsight/pull/6984) | [#6970](https://github.com/vectorize-io/hindsight/issues/6970) |
| `--mcp-name`, never touch a user's own `hindsight` server | [#6985](https://github.com/vectorize-io/hindsight/pull/6985) | [#6958](https://github.com/vectorize-io/hindsight/issues/6958) |
| `apiHeaders` | [#6969](https://github.com/vectorize-io/hindsight/pull/6969) | [#6651](https://github.com/vectorize-io/hindsight/issues/6651) |
| A keyless Hindsight Cloud config turns memory off | [#6882](https://github.com/vectorize-io/hindsight/pull/6882) | |

</details>
