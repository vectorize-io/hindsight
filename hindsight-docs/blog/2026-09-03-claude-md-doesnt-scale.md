---
title: "CLAUDE.md Doesn't Scale"
authors: [benfrank241]
slug: "2026/09/03/claude-md-doesnt-scale"
date: 2026-09-03T16:00
tags: [agent-memory, coding-agents, claude-code, agents-md, context, conventions]
description: "Every coding agent now reads a markdown project file. They rot silently, they fill up with the things that go stale fastest, and they cost context on every turn. Here's what they're good for, and what they can't be."
image: /img/blog/claude-md-doesnt-scale.png
hide_table_of_contents: true
---

![CLAUDE.md doesn't scale: hand-maintained agent context files rot silently while the codebase moves on](/img/blog/claude-md-doesnt-scale.png)

The Hindsight repository has a `CLAUDE.md`. It is **442 lines long**.

It also has an `AGENTS.md`. That one is **three lines**, and all it does is point at the other file.

We build agent memory for a living, and this is what our own hand-maintained agent context looks like. I'm not opening with that to be self-deprecating. I'm opening with it because if the people who think about this problem professionally end up here, the problem probably isn't discipline.

<!-- truncate -->

## First, the file is doing real work

It would be easy and wrong to argue that markdown project files are useless. `CLAUDE.md`, `AGENTS.md`, `QWEN.md`, Cursor's rules file — they caught on quickly because they solve something real.

They're **explicit**. You can read one and know exactly what the agent was told. Nothing is inferred.

They're **reviewable**. A change shows up in a diff, and someone can argue with it before it lands.

They're **version-controlled**, which means the instruction and the code it describes travel together.

And they're **intentional**. When you want an agent to always do a specific thing, writing it down is the most direct way to make that happen.

None of that goes away. The argument here isn't that you should delete the file. It's that a hand-maintained document cannot be the only thing your agent knows about your project, and the reasons are structural rather than a matter of trying harder.

## It rots, and the rot is invisible

Our `CLAUDE.md` has been touched in **35 commits out of 2,820**. That's roughly one commit in eighty. As I write this, the codebase has moved 42 commits since the file last changed.

That ratio isn't negligence. It's what happens to every document that isn't required by anything. Nothing breaks when it goes stale. No test fails. No build turns red.

And that's the actual problem: **a stale instruction doesn't error, it just gets followed.** The agent has no way to know that the file describes last quarter's architecture. It reads with total confidence, and confidently does the wrong thing, and the failure looks like a reasoning mistake rather than a documentation one.

A wrong `CLAUDE.md` is worse than no `CLAUDE.md`. An absent file makes an agent read the code. A stale file makes it trust a description of code that no longer exists.

## It fills up with the things that rot fastest

Look at what's actually in ours. Of 442 lines, **71 contain a file path, a command, or a directory reference.** Start the API server this way. Regenerate the clients with this script. Tests live here.

That is precisely the content with the shortest half-life. Paths move. Scripts get renamed. A command picks up a flag. Every one of those lines is a small hostage to refactoring.

It's also the content an agent least needs from you. A capable coding agent can read `package.json`, find the test runner, and work out how the project is laid out. It's good at that. Handing it a written description of things it could observe directly is the least valuable thing the file could contain, and the most likely to be wrong.

What an agent genuinely can't derive is the reasoning. Why the rounding rule is half-up. Why that endpoint isn't retried. Why the importer's tie-break looks backwards and is deliberate.

Almost none of that is in our file. It isn't in yours either, and not because you're careless — those decisions get made in a pull request thread or a conversation, at a moment when nobody is thinking about documentation. The knowledge that matters most is the knowledge least likely to get written down.

## Every line costs you on every turn

A markdown project file is unconditional. All 442 lines go into context whether you're fixing a typo or redesigning the retrieval pipeline.

That's a fixed tax on every single turn, and it only ever grows, because adding a line is easy and deleting one requires knowing it's obsolete — which is exactly what nobody knows.

Meanwhile the thing you actually need for *this* task is either in there diluted among four hundred other lines, or not in there at all.

There's a real tension here that a static file can't resolve. To be useful it has to be comprehensive. To be affordable it has to be short. Pick one.

## It multiplies with every agent you use

`CLAUDE.md` for Claude Code. `AGENTS.md` for Codex. `QWEN.md` for Qwen Code. A rules file for Cursor. Same project, same facts, four documents, four drift rates.

Our answer was the three-line `AGENTS.md` that redirects to `CLAUDE.md`, which is the sensible hack and also an admission. It works right up until an agent doesn't follow the pointer, or until someone edits one file and not the other.

The underlying issue is that these files belong to the *tool*, not to the project. Switch agents and you're maintaining another one. [Use several agents](/blog/2026/09/02/coding-agents-050-four-new-harnesses), which is increasingly normal, and you're maintaining a small set of documents that say almost the same thing and slowly stop agreeing.

## A line in a file has no provenance

Consider a line that says: *"We always use the repository pattern for data access."*

Who decided that? When? Was it a considered architectural choice, or something someone typed in month one that nobody revisited? Does it still apply to the module written last week? Is there a known exception?

The file can't tell you. It's an assertion with no date, no author, and no reasoning attached. You either trust it completely or you go digging — and the agent, which can't go digging through a decision that lives in a closed pull request, trusts it completely.

Git history has all of that context. So do the conversations where the decision was actually argued out. Neither one is in the file.

## So what should the file be?

Keep it. Make it smaller.

A markdown project file is at its best as a **standing instruction set**: the handful of things you actively want every agent to do, in every session, regardless of task. House style. A rule that isn't discoverable from the code. A warning about something genuinely dangerous.

That's a short document, and short documents don't rot as fast, because there's less surface to go stale and it's obvious when a line is wrong.

What shouldn't be in there is everything an agent can derive on its own, and everything that's really a record of what happened rather than an instruction about what to do.

| | A markdown project file | Derived memory |
|---|---|---|
| **Kept current by** | Someone remembering | What actually happened |
| **Goes stale** | Silently | Not really; it's a record |
| **Cost per turn** | Every line, every time | Only what the task needs |
| **Provenance** | None | When, who, and what changed |
| **Across agents** | One file each | One memory per project |
| **Best at** | Standing instructions | Decisions and their reasoning |

That second category is the one worth solving properly, because it's where the value is:

- **Decisions and their reasoning** belong in something derived from git history and past conversations, so they're captured whether or not anyone remembers to write them down.
- **Retrieval should be scoped to the task**, so a question about the importer surfaces the importer's history rather than four hundred lines about everything.
- **Facts should carry provenance** — when, who, and what changed since — so an agent can weigh a decision from last week differently from one from last year.
- **It should survive switching agents**, because the knowledge is about the project, not about the tool you happened to open.

That's the shape of the [knowledge pages](/blog/2026/08/13/knowledge-pages-coding-agents) idea: a curated layer that gets rebuilt in the background from what actually happened, rather than a file someone has to remember to edit.

## A test you can run right now

Open your project's agent file. Then run:

```bash
git log --oneline -- CLAUDE.md | wc -l
git rev-list --count HEAD
```

Compare the two numbers. Ours is 35 against 2,820.

Then ask a harder question: of the lines in that file, how many describe something an agent could have worked out by reading the code, and how many capture a decision it never could have?

If it's mostly the first, the file is carrying risk without carrying much value. And if it hasn't changed in forty commits, one of two things is true. Either the project stopped changing, or the file is already wrong and nothing has told you yet.

## Learn more

- [Four More Coding Agents That Remember Your Project](/blog/2026/09/02/coding-agents-050-four-new-harnesses) — the agents that share one project memory
- [Knowledge Pages for Coding Agents](/blog/2026/08/13/knowledge-pages-coding-agents) — a curated layer that maintains itself
- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — how to scope memory across repos and agents
