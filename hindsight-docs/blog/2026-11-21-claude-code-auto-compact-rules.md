---
title: "Claude Code Auto Compact: Keep Rules in a Repo Memory Bank"
slug: "2026/11/21/claude-code-auto-compact-rules"
description: "How the Hindsight Coding Agents hooks keep Claude Code rules in a repo memory bank, and what comes back after auto compact."
authors: [chrislatimer]
date: 2026-11-21
tags: [claude-code, coding-agents, memory, deep-dive]
hide_table_of_contents: false
image: /img/blog/claude-code-auto-compact-rules/featured.png
---

import ThemedImage from '@theme/ThemedImage';

You run Claude Code on one repository for hours. Early in the session, you set the rules.

Skills live in this folder. Do not touch that file. Finish the whole feature without stopping to ask.

For an hour, the agent follows every rule. Then auto compact runs. The next answer breaks a rule that the agent followed one minute before.

You want to hand the agent a long task and leave it alone. After compaction, you cannot.

<!-- truncate -->

## You are not the only one who sees this

Developers report this problem in public, with logs.

On r/ClaudeAI, [u/apaas wrote](https://www.reddit.com/r/ClaudeAI/comments/1q1c063/comment/nx6radx/) in January 2026:

> "Claude losing its basic rule context after auto compaction is a _ridiculous_ bug which should be fixed immediately."

On the anthropics/claude-code issue tracker, users file the same report again and again:

- In [issue #19471](https://github.com/anthropics/claude-code/issues/19471), a user's CLAUDE.md said where the skill files live. "After context compaction, Claude searched from workspace root (`E:\workspace`) instead of project directory". The user asked if Claude read CLAUDE.md. The reply was: "I didn't read CLAUDE.md".
- In [issue #67500](https://github.com/anthropics/claude-code/issues/67500), a user writes: "After context compaction occurs mid-session, Claude reliably drops several critical behavioral rules that were established earlier in the conversation."
- [Issue #89733](https://github.com/anthropics/claude-code/issues/89733) has the title "CLAUDE.md and memory rules stop applying after compaction". The user logged 163 sessions.

These are reports by users. They are not statements by Anthropic.

The cost is your time and your attention. You watch the session. You repeat the rule.

You check each commit for the rule that slipped. A long task becomes a supervised task.

## What auto compact does

Anthropic's docs explain it in [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): "Claude Code manages context automatically as you approach the limit. It compacts the conversation into a summary." The same page names the risk: "The summary keeps your requests and key code snippets, but detailed instructions from early in the conversation may be lost."

[Auto compact runs when the conversation reaches the auto-compact window](https://code.claude.com/docs/en/costs). You can also run it by hand with `/compact`, and give it a focus, for example `/compact focus on the API changes`.

<ThemedImage
  alt="You set rules in the conversation. Auto compact summarizes it. SessionStart with source compact restores the Hindsight block. By default, Stop writes session turns to the repo memory bank outside the context window."
  sources={{light: "/img/blog/claude-code-auto-compact-rules/session-timeline-light.svg", dark: "/img/blog/claude-code-auto-compact-rules/session-timeline-dark.svg"}}
/>

## What survives compaction, and what does not

Anthropic's [memory docs](https://code.claude.com/docs/en/memory) say what comes back:

> Project-root CLAUDE.md survives compaction: after `/compact`, Claude re-reads it from disk and re-injects it into the session.

The same section says what can go missing: "If an instruction disappeared after compaction, it was given only in conversation, lives in a nested CLAUDE.md that hasn't reloaded yet, or is a path-scoped rule that hasn't matched a file since." Anthropic's advice is to "Add conversation-only instructions to CLAUDE.md to make them persist."

That advice is sound for the rules that apply to every task. But the reports above show two gaps:

1. **Rules that live in CLAUDE.md can still drop.** Issue #67500 adds a detail that is easy to miss. The compaction summary in that session held the user's rules: "Despite the summary containing the user's instructions, the assistant did not follow them."
2. **Most rules start in the conversation.** "Use the staging database." "We decided to keep the old API." You do not stop to edit CLAUDE.md each time. If every correction goes into CLAUDE.md, the file grows, and a long instruction file has its own problem. We wrote about that in [Stop Growing Your Always-On Context](/blog/2026/09/16/stop-growing-your-system-prompt).

:::note

No tool can make a model obey a rule. What you can control is where the rules live, and whether the agent gets a path back to them after compaction.

:::

## Keep the rules outside the transcript

The [Hindsight Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) keeps what your repository learns in a memory bank. The bank is on a Hindsight server, not in the Claude Code context window. Compaction does not change it.

The integration's README describes the round trip you know from long sessions: "A correction is a round trip: the agent guesses a project-specific decision wrong, you stop it, you explain, it redoes the work."

This is what the code does in version 0.8.8 of `@vectorize-io/hindsight-coding-agents`. The source is in [`hindsight-integrations/coding-agents`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/coding-agents).

### One bank for each repository

The default bank id is `coding-agent::{gitProject}`. The docs say the default "makes every agent share one bank per repo". With the default bank id, Claude Code, Codex CLI, Cursor CLI and the other supported agents use the same bank for the same repository.

### What goes into the bank

- **Git history.** By default, at session start, the integration can seed a cold bank from the repository's git history, in the background.
- **Your sessions.** By default, the Stop hook writes session turns back to the bank. The server extracts memories from what you said and what the agent did.
- **No recap of a recap.** The transcript reader drops Claude Code's compaction summaries (`isCompactSummary`). The bank gets what you said, not a summary of it.
- **Pages.** By default, the same background seed can also set up curated pages about the repository: architecture, conventions, past decisions and in-flight initiatives.

### What comes back, and when

The installer adds 3 hooks to `~/.claude/settings.json`. Each one runs on a Claude Code event.

**`SessionStart`, which includes the start after compaction.** Injects a `<hindsight_pages>` block. It tells the agent that the repository has a Hindsight memory, gives the page count or "No pages yet", and says which tools to call, and when.

**`UserPromptSubmit`, on the first prompt of a session.** Runs a low-budget reflect over the bank and injects the result at most once. This is the default, `autoInject: "reflect"`.

- When the server confirms that the bank is empty, the hook can wait for the second prompt.
- If the reflect times out or fails with a server error, the hook runs a fallback search in its place: the pages first, then recall when no page matches. If that search returns nothing, the hook tries reflect again on the next prompt.
- After a network error, an HTTP 408 or an HTTP 429, the hook skips the search and tries reflect again on the next prompt.
- The hook makes at most 2 reflect attempts in a session. After any other 4xx error, it does not try again.

**`UserPromptSubmit`, on every prompt.** Re-injects a `<hindsight_page_refresh>` reminder of the tools. By default, `pageRefreshEveryTurns` is 1. The block gives the full tool guide. It adds the page count when the last page list that the hook read in this session has pages.

**`Stop`.** By default, writes session turns back into the bank.

Claude Code sends `SessionStart` with the source `"compact"` after compaction. The installer registers the Hindsight `SessionStart` hook with no matcher. The [Claude Code hooks reference](https://code.claude.com/docs/en/hooks) says that when you omit the matcher, "the group activates on every occurrence of the event." With the integration enabled and a bank available, the hook can return the Hindsight block after either form of compaction.

<ThemedImage
  alt="SessionStart adds the Hindsight block. By default, the first prompt runs reflect, or can wait until prompt 2 for a bank the server confirms is empty. A timeout or server error triggers page search, then recall if needed. If the fallback returns nothing, or after a network error, 408 or 429, the hook retries reflect once on the next prompt. By default, each prompt adds the full tool guide, and can add the page count. By default, Stop writes the session to the bank."
  sources={{light: "/img/blog/claude-code-auto-compact-rules/memory-hooks-light.svg", dark: "/img/blog/claude-code-auto-compact-rules/memory-hooks-dark.svg"}}
/>

Source for the first-prompt and every-prompt flows: [hook code](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/core/hook.ts) and [page refresh code](https://github.com/vectorize-io/hindsight/blob/62a03bf/hindsight-integrations/coding-agents/src/core/page-injection.ts).

The agent also gets these MCP tools:

- `hindsight_search_pages(query)`: the first stop. It finds the pages that bear on the current turn.
- `hindsight_read_page(id)`: reads one page in full.
- `hindsight_reflect(query)`: reasons over the full memory of the repository for the decision behind a behavior. It is slower.
- `hindsight_ingest_document(title, content)`: saves notes that you want the bank to keep.

After compaction, the agent can call `hindsight_reflect` for a fresh answer from the bank.

### What Hindsight does not do

- It does not stop compaction or change what the Claude Code summary keeps.
- It does not paste all your rules into every prompt. It puts the tools, and the moment to use them, in front of the agent. The agent decides when to call them.
- A search returns the memories that match the query. A rule that has nothing to do with the current task may not come back.
- It does not force the model to follow a rule. It keeps the rules in a place that compaction does not touch, and it puts the path to them back after compaction and, by default, on every prompt.

### The numbers

The integration's docs report results on [AMB sdebench](https://agentmemorybenchmark.ai/dataset/sdebench): 61 real repository tasks, 3 runs each, with and without memory. Accuracy does not move. Every run solves 60 or 61 of the 61 tasks either way. What changes is how often you correct the agent, and what the tokens cost:

| Claude Code | No memory | With Hindsight |
| :- | :- | :- |
| Corrections per task | 0.85 | 0.36 |
| Cost per task | $0.445 | $0.338 |

<ThemedImage
  alt="On AMB sdebench, Claude Code averaged 0.85 corrections per task without memory and 0.36 with Hindsight. Cost per task was $0.445 without memory and $0.338 with Hindsight."
  sources={{light: "/img/blog/claude-code-auto-compact-rules/sdebench-light.svg", dark: "/img/blog/claude-code-auto-compact-rules/sdebench-dark.svg"}}
/>

Source: [Coding Agents benchmark results](https://hindsight.vectorize.io/sdks/integrations/coding-agents).

The values are the mean of 3 runs. The benchmark does not measure compaction. It measures the correction round trip above.

## The plan: 3 steps

### 1. Install the integration for Claude Code

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

On a terminal, the installer asks where memory lives. There are 3 modes:

| Mode | What runs | What it needs |
| :- | :- | :- |
| `cloud` | Hindsight Cloud (the default) | an API token |
| `self-hosted` | a Hindsight server that you run | its URL |
| `daemon` | a local `hindsight-embed` on your machine | `uv` on PATH and an LLM key for extraction |

To choose the mode in a script, pass `--server`:

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code --server daemon
```

The install adds 3 hooks in `~/.claude/settings.json`, registers the MCP server with `claude mcp add` (user scope), and adds a companion skill.

### 2. Get your rules into the bank

Work as you do now. When you set a rule in the conversation, the Stop hook writes session turns to the bank by default.

For rules you gave in past sessions, import your old Claude Code conversations. Run this from the repository folder:

```bash
cd /path/to/your/repo
npx @vectorize-io/hindsight-coding-agents install claude-code --import-conversations
```

The README says this "costs tokens roughly in proportion to the history imported, and it is safe to re-run".

For a rule that must hold across sessions, ask the agent to save it as a document:

```text
Save this rule to Hindsight memory with hindsight_ingest_document:
"Skills for this repo live in .claude/skills/ in the project folder, not in the workspace root."
```

When a memory is wrong or old, the Hindsight block at session start tells the agent to ingest a `Correction: <topic>` document. The document states what the memory said, what is true now, and the evidence. Newer facts supersede older ones.

Keep the rules that apply to every task in your project-root CLAUDE.md as well. Anthropic's docs say Claude Code re-reads that file after compaction.

### 3. Run a compaction and check what came back

On a new repository, the bank needs time to fill. Ask the agent to call `hindsight_sync_status`. The tool returns `"synced": true` only when 3 things hold:

- The git history seed is in the bank.
- The bank has at least one page, or the server has no pages API.
- The tool counts no server operation still running. If the server does not return a list of operations, the tool counts 0.

Then run `/compact` in the session. Ask a question that depends on one of your rules:

```text
Where do the skill files for this repo live?
```

The agent can answer from the pages with `hindsight_search_pages`. When an answer uses Hindsight memory, the tool guide tells the agent to mark it with a "From Hindsight memory" blockquote, so you can see where the answer came from.

To see what the hooks did, read the diagnostics file. It records reflect and page-fetch outcomes, one JSON line each:

```bash
tail -n 5 ~/.hindsight/coding-agents-logs/diag.jsonl
```

## If you change nothing

Your rules stay in CLAUDE.md and in the conversation. After the next auto compact, the agent can drop one of them. You find out in review, or later. Every long session still needs a person to watch it.

## What your sessions look like after

Auto compact runs in the middle of a long task. The SessionStart hook can add the Hindsight block and its tool guide again. The agent can search the pages or call `hindsight_reflect`. It can find the rule that you set two hours ago and mark where the rule came from.

You go from a developer who repeats rules after each compaction to a developer who sets a rule once and lets the repository keep it.

## Start here

Install the integration on the repository where you lose the most rules:

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

Not ready to install? Read the [Coding Agents docs](https://hindsight.vectorize.io/sdks/integrations/coding-agents). The configuration reference explains `autoInject`, `pageRefreshEveryTurns` and `toolGuideExtra`, the setting that adds your own memory guidance to the tool guide.
