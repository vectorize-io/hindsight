---
title: "Your Paperclip Org Chart Has a Memory Boundary"
authors: [benfrank241]
slug: "2026/10/05/paperclip-agent-memory"
date: 2026-10-05T12:00
tags: [hindsight, paperclip, agent-memory, multi-agent, banks, integrations]
description: "Paperclip gives agents an org chart. Hindsight's bankGranularity decides what each agent is allowed to remember about the others."
image: /img/blog/paperclip-agent-memory.png
hide_table_of_contents: true
---

![Paperclip gives agents an org chart; bankGranularity decides where the memory lines are drawn](/img/blog/paperclip-agent-memory.png)

You have two agents working for the same company.

One has already investigated a customer integration, found the cause of a recurring failure, and left the decision in an issue comment. The other agent is now assigned a related issue. It has the same tools, the same company context, and access to the same work. But when it starts, it does not know what the first agent learned.

So it investigates the same failure again.

Now change one setting.

Instead of giving each agent its own memory bank, you give the company one shared bank. The second agent can recall the first agent's decision before it starts. The work compounds instead of restarting.

That sounds like a small configuration change. It is not.

Paperclip gives your agents an organisational structure: who reports to whom, who can receive work, and what each principal is allowed to do. Hindsight adds another boundary underneath that structure: what an agent can remember.

The question is not whether your agents should have memory. The question is where that memory boundary should sit.

And in the Paperclip integration, `bankGranularity` is the setting that answers it.

<!-- truncate -->

## TL;DR

- **The default is per company and per agent.** `bankGranularity` defaults to `["company", "agent"]`, which gives each agent its own Hindsight bank inside each Paperclip company.
- **That default is a decision, not a neutral setting.** It prevents agents from seeing each other's memories, but it also means an agent can repeat work another agent has already done.
- **Company granularity makes memory a collaboration surface.** `["company"]` gives every agent in a company the same bank while keeping companies separate.
- **Agent granularity changes the boundary in the other direction.** `["agent"]` lets an agent carry memory between companies, which can be useful but has obvious cross-company implications.
- **The choice lives in Paperclip's instance-level Hindsight plugin settings.** Install `@vectorize-io/hindsight-paperclip`, then configure it under Settings, Plugins, Hindsight Memory.
- In a multi-agent system, `bankGranularity` is the answer to "what is one agent allowed to remember about another's work", and the default already answers it for you.

## 1. What Paperclip already gives you

Paperclip models a company as an organisation of agents.

Agents have roles, titles, reporting lines, permissions and budgets. In Paperclip's own description, your agents have a boss, a title, and a job description, and delegation flows up and down that org chart. Issues are the unit of work, and a task can be assigned to an agent or a person.

Paperclip also governs who can do what. The org chart deliberately covers humans and agents alike. The access controls underneath are separate, though: human roles and permissions on one side, agent API keys and agent eligibility on the other.

One more property matters for everything that follows. A single Paperclip deployment can host several organisations, with company-scoped access checks keeping each organisation's work, agents and activity separate.

That gives you a useful model of the organisation:

```
                 CEO
                /   \
        Research     Engineering
          /             \
     Researcher       Coder
```

The org chart tells you who works for whom.

The task system tells you what each agent is working on.

The permission system tells you what each agent is allowed to do.

There is one thing those structures do not answer.

What is each agent allowed to remember?

That is where the memory boundary starts to matter.

## 2. The thing the org chart does not cover

Permissions govern actions.

Memory governs context.

Those are related, but they are not the same boundary.

Imagine a research agent discovering that a particular customer uses an unusual authentication flow. The agent records that decision in memory. Later, a coding agent receives a task involving the same customer.

Should the coding agent be able to recall that decision?

If the answer is yes, the two agents need a shared memory boundary.

If the answer is no, they need separate memory.

A Hindsight bank is one such boundary. It is the isolated memory store that recall and retain operate against. Memories in one bank are not automatically available to an agent recalling from another bank.

That makes bank selection an architectural decision.

The Paperclip integration exposes that decision as configuration instead of making you build it into each agent.

The important setting is `bankGranularity`.

The default is:

```json
["company", "agent"]
```

That means the research agent and coding agent in the same company do not share a bank.

The setting is doing exactly what it says. It is also making a decision for you.

## 3. What the plugin does, concretely

The Hindsight Paperclip plugin is installed at the Paperclip instance level. The current released package is `@vectorize-io/hindsight-paperclip` v0.4.0 and requires Paperclip 2026.720.0 or newer.

You install it with:

```bash
pnpm paperclipai plugin install @vectorize-io/hindsight-paperclip
```

Then configure it under Settings, Plugins, Hindsight Memory.

You do not modify each agent.

The plugin subscribes to Paperclip lifecycle events and connects those events to Hindsight.

At the start of a run, the plugin receives the issue associated with the run. It fetches the issue, combines its title and description into a recall query, and searches the bank for that agent.

The result is cached for the run.

During the run, the agent also has two tools:

```
hindsight_recall(query)
hindsight_retain(content)
```

`hindsight_recall` lets the agent perform a targeted lookup. If the query is the same as the run-start query, the cached result is reused. The comparison is trimmed string equality. A different query triggers a live recall.

`hindsight_retain` lets the agent store an important fact, decision, or outcome immediately.

There is also an automatic retention path.

When `issue.comment.created` fires, the plugin fetches the full comment body and retains it. The event payload carries only a 120-character snippet, so the plugin calls Paperclip's comments API for the full text. If that call fails, it falls back to the snippet rather than losing the comment entirely.

For an agent-authored comment, the comment's agent is used for bank attribution. If there is no agent attribution, the plugin falls back to the issue's assignee.

The flow is therefore:

```
Paperclip issue
      │
      ▼
agent.run.started
      │
      ├── fetch issue
      │
      ├── title + description
      │
      ▼
   Hindsight
      │
      ▼
 recall into run state
      │
      ▼
    Agent
      │
      ├── hindsight_recall()
      │
      └── hindsight_retain()
      │
      ▼
issue.comment.created
      │
      ▼
retain full comment
```

One detail is worth calling out because the v0.4.0 configuration currently makes it easy to misunderstand.

`agent.run.finished` is subscribed to, but the handler is a no-op. Paperclip's run-finished payload does not contain the agent's output, so the plugin cannot retain that output from the event.

In v0.4.0, `autoRetain` actually gates the automatic retention of comments. It does not make the `agent.run.finished` handler retain run output.

The manifest still describes the option as "Automatically retain agent run output when a run completes". That description does not match the released worker implementation. The code is the source of truth here.

## 4. Four ways to organise the memory

Once you see the bank as a memory boundary, `bankGranularity` becomes easier to reason about.

Do not start with the configuration string.

Start with the organisation you are trying to build.

### The default: each specialist keeps its own notebook

```json
["company", "agent"]
```

This produces:

```
paperclip::{companyId}::{agentId}
```

Suppose your company has three agents:

```
Researcher
Coder
Reviewer
```

Each gets a separate bank.

The researcher remembers its research. The coder remembers its coding work. The reviewer remembers its reviews.

Nothing one agent learns automatically becomes another agent's memory.

This is the safest default when agents have genuinely different responsibilities or when you are not yet sure what should be shared.

It is also quietly wasteful for collaborative teams.

If the researcher discovers an important fact and puts it in its own bank, the coder cannot recall it. The coder has to discover the same fact again, or receive it through some other explicit channel.

This is the key tradeoff.

Isolation prevents unwanted memory sharing.

It also prevents useful memory sharing.

### Company memory: the team keeps one notebook

```json
["company"]
```

This produces:

```
paperclip::{companyId}
```

Now every agent inside the company uses the same bank.

The researcher records a decision.

The coder can recall it.

The reviewer can recall what both of them learned.

This is the natural choice when the agents are different roles in the same collaborative team rather than independent specialists.

It also preserves the company boundary. Company A's agents do not automatically recall Company B's memories.

For many Paperclip deployments, this is the most interesting alternative to the default.

The org chart says the agents are part of one company.

The memory bank now agrees.

### Agent memory: the specialist carries its notebook between companies

```json
["agent"]
```

This produces:

```
paperclip::{agentId}
```

Notice what disappeared.

`companyId`.

The same agent identity therefore uses the same bank regardless of company.

That can be useful when the agent represents a durable specialist whose knowledge should follow it between company contexts.

It can also be exactly the wrong boundary.

If the same agent can work for multiple tenants, information learned in one company can become recallable in another.

That is not a subtle consequence of the configuration. It is what `["agent"]` means.

Use it only when cross-company memory is intentional.

### User memory: the notebook belongs to the person

Add `user` to the granularity:

```json
["company", "agent", "user"]
```

The resulting shape is:

```
paperclip::{companyId}::{agentId}::user::{userId}
```

Now memory is separated not only by company and agent, but by the human associated with the issue.

The important point is technical rather than regulatory: the plugin is creating a separate bank for each identified user within that company-agent boundary.

Do not read "useful for GDPR compliance" in the plugin configuration as a guarantee of compliance. A bank boundary can support an isolation strategy, but compliance depends on the rest of your system and your policies.

There is also a fallback worth understanding, because it is quieter than it looks.

If the plugin cannot identify a user, it omits the user segment rather than inventing an identity. The bank then becomes:

```
paperclip::{companyId}::{agentId}
```

which is the bank shared by every user of that agent.

Not inventing an identity is the right call. The consequence is the part to notice: under user granularity, memory from an issue with no identifiable user does not get its own isolated bank. It lands in the shared one. If per-user isolation is the point, check that your issues actually carry an identity.

### Static memory: deliberately remove the dynamic boundary

There is one more option:

```json
dynamicBankId: false
bankId: "shared-team-bank"
```

In static mode, the configured `bankId` is returned directly.

That means every agent using that configuration reads and writes the same bank.

The complete set of shapes is:

```
paperclip::{companyId}::{agentId}                  default
paperclip::{companyId}                             company
paperclip::{agentId}                               agent
paperclip::{companyId}::{agentId}::user::{userId}  user
{bankId}                                           static
```

The first four are derived from Paperclip identity. The last one is an explicit override.

That distinction matters.

With a static bank, you are no longer asking Paperclip to decide the boundary from the organisation. You are telling it exactly where the boundary is.

## 5. The user-identity detail that makes this concrete

The user granularity is not based on a random session identifier.

The plugin looks at the Paperclip issue.

If an issue has:

```
originId = "slack::alice@acme.com"
```

the plugin can extract:

```
alice@acme.com
```

If `creatorEmail` is available, that takes precedence.

Otherwise, the plugin splits `originId` on `::` and scans the segments from the end, taking the first one that contains an `@`.

That means a Slack-originated issue can become memory scoped to the person who created it.

For example:

```
paperclip::acme::support-agent::user::alice@acme.com
```

Alice's support context and Bob's support context do not have to become the same memory.

There is a useful design lesson here.

A memory boundary is only as good as the identity used to construct it.

The plugin is deliberately deriving that identity from the issue rather than from the run. That is why a run ID does not become a new memory bank every time an agent starts work.

## 6. What this costs you

There is no configuration that gives you perfect isolation and perfect sharing at the same time.

Shared memory has a failure mode.

If one agent records a wrong conclusion, that conclusion becomes available to other agents. A shared bank can make a bad decision travel faster.

Per-agent memory has a different failure mode.

If the researcher discovers the correct answer and the coder cannot see it, the system repeats work.

Neither problem is caused by Hindsight.

They are consequences of choosing where the memory boundary sits.

That is why the right question is not:

"Which setting gives my agents the best memory?"

Ask:

"If agent A learns something, should agent B be able to recall it?"

If yes, they belong on the same side of the boundary.

If no, they do not.

This is the same principle whether you are deciding between customers, teams, agents, or users.

## 7. How to choose

You can make the decision in a couple of minutes.

Start with the hardest boundary first.

**First, identify your tenancy boundary.**

If two companies must never share memory, keep `company` in the bank identity. Do not remove it just because shared context sounds useful.

**Then ask whether the agents collaborate on the same body of work.**

If they do, `["company"]` is usually the first configuration worth considering. It gives the company one shared memory bank while keeping companies separate.

**If the agents are independent specialists, keep the default.**

`["company", "agent"]` gives each specialist its own memory without crossing the company boundary.

**Only choose `["agent"]` when cross-company memory is intentional.**

Removing `company` is not a performance optimisation. It changes the information boundary.

**Add `user` when the human is part of the boundary.**

This is useful when the same agent serves multiple people and their memories should remain distinct.

**Use a static `bankId` when you want an explicit shared memory domain.**

This is the strongest form of "these agents are one team". It is also the easiest one to misconfigure across tenants, so make the shared scope deliberate.

There is one final check.

Ask whether the bank ID will still mean the same thing six months from now.

A stable identity makes a stable memory boundary.

## 8. If something looks wrong

**The plugin installed, but Paperclip asks for new capabilities**

v0.4.0 declares `issues.read` and `issue.comments.read` because it fetches the issue at run start and retrieves full comment bodies.

Paperclip may prompt for these on first install or upgrade. If it does, that is expected rather than a sign that something is wrong.

**You upgraded the plugin and the Hindsight API key stopped working**

Open the Hindsight Memory settings and select the API key again with Paperclip's secret picker.

Older plugin versions could save the secret's name. The current host expects a secret reference, so an upgrade can leave an older configuration pointing at the wrong representation.

**One agent never seems to get memory**

Check `enabledAgentIds`.

An empty list means all agents are enabled. Once the list contains IDs, only those agents are allowed through the plugin's recall and retain paths.

This can look like a bank problem when it is actually an enablement problem.

**A comment was never retained, and nothing looks broken**

Retention needs an agent to attribute the memory to.

The plugin uses the comment's own agent when there is one, and otherwise falls back to the issue's assigned agent. If the issue has no assigned agent either, there is nothing to attribute the memory to, so the plugin skips the retain and logs that no agent attribution was available.

A comment on an unassigned issue is therefore not stored. Assign the issue if you want its discussion to reach memory.

**You set `autoRetain` to false and comments stopped appearing in memory**

That is expected.

In v0.4.0, `autoRetain` gates the `issue.comment.created` retention handler. It does not control the no-op `agent.run.finished` subscription.

If you expected completed run output to be stored automatically, the current released worker does not do that from `agent.run.finished`.

**Recall is timing out on a self-hosted instance**

The default `requestTimeoutMs` is 15,000 milliseconds.

The plugin documentation specifically calls out longer recall times for self-hosted instances when issue descriptions are large. Increase the timeout if that is your bottleneck rather than treating a timeout as evidence that the memory bank is empty.

**You expected two agents to share memory, but they do not**

Check the actual bank boundary.

With the default:

```json
["company", "agent"]
```

these are different:

```
paperclip::acme::researcher
paperclip::acme::coder
```

Changing to:

```json
["company"]
```

makes both agents use:

```
paperclip::acme
```

A shared bank is a different memory boundary, not a different recall query.

## 9. FAQ

**Does changing `bankGranularity` migrate existing memory?**

No. Changing the granularity changes the bank ID that the plugin derives.

For example, moving from `["company", "agent"]` to `["company"]` changes `paperclip::acme::coder` into `paperclip::acme`. Those are different banks, so existing memories do not automatically move between them.

This is why the bank choice is easier to make before a large amount of memory accumulates.

**Can two agents share a bank while a third stays isolated?**

Not through a single `bankGranularity` setting in v0.4.0.

The configuration is instance-level, so `["company"]` applies the company bank to all enabled agents. `["company", "agent"]` gives all agents their own company-scoped banks.

`enabledAgentIds` controls which agents use the plugin. It does not give individual agents different bank strategies.

If you need a mixed topology, that needs to be designed outside this single global granularity setting.

**What happens when an agent is deleted?**

The plugin does not delete or migrate its Hindsight bank when an agent disappears.

The bank ID is derived from the agent ID, and the worker has no agent-deletion handler that removes the corresponding bank.

That means deleting an agent is not the same thing as deleting its memory. Treat bank lifecycle and Paperclip agent lifecycle as separate concerns.

**Does a run get a new bank?**

No.

The default bank contains the company and agent identity, not the Paperclip run ID.

A run therefore recalls from the same bank as previous runs for that agent in that company.

The run ID is used for temporary plugin state, such as the cached recall and user identity. It is not the default memory boundary.

**Can I use a static `bankId` and `bankGranularity` together?**

You can configure both, but the static bank wins unless `dynamicBankId` is explicitly true.

With:

```json
bankId: "shared-team-bank"
bankGranularity: ["company", "agent", "user"]
```

and no explicit `dynamicBankId: true`, the bank is:

```
shared-team-bank
```

Set `dynamicBankId: true` if you want the granularity-derived bank instead.

**Does the default mean agents never share knowledge?**

It means they do not share Hindsight bank memory by default.

Paperclip still has its own collaboration surfaces: issues, comments, delegation, and the organisation itself.

That distinction matters. The default does not prevent agents from communicating. It prevents one agent's Hindsight memory from automatically becoming another agent's Hindsight memory.

**Should every multi-agent Paperclip deployment use a shared bank?**

No.

Shared memory is useful when the agents are collaborating on the same domain and should benefit from each other's accumulated context.

It is the wrong boundary when different tenants, teams, or specialists need genuinely separate memory.

The point of `bankGranularity` is not to make sharing the default. It is to make the boundary explicit.

## 10. Learn more

If you are deciding how to structure this beyond Paperclip, these are the places to go next:

- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) for the underlying question: when two things should share a Hindsight bank, and when a new bank is a real isolation boundary.
- [One Bank, Two Audiences, Two Briefs](https://hindsight.vectorize.io/blog/2026/09/30/per-scope-consolidation-strategies) for how observation scopes let one bank serve different audiences with different consolidation rules.
- [Give Every Hermes Bot Its Own Memory](https://hindsight.vectorize.io/blog/2026/08/18/hermes-bot-mode-memory) for the closest prior multi-agent example, where independent bots can use private banks or deliberately share a room's memory.
- [Paperclip](https://github.com/paperclipai/paperclip) for the project's organisation, delegation, permissions, and plugin model, which is the other half of the boundary.

The useful mental model is simple.

Paperclip gives you the organisation.

Hindsight gives that organisation a memory.

`bankGranularity` decides where the memory lines are drawn.

Make those lines match the organisation you actually want.
