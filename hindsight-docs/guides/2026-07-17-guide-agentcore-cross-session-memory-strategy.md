---
title: "Guide: A Cross-Session Memory Strategy for Bedrock AgentCore"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, agentcore, bank-strategy, memory]
description: "A cross-session memory strategy for Amazon Bedrock AgentCore: scope the Hindsight bank to a stable user or agent identity so memory survives the ephemeral runtimeSessionId instead of evaporating between sessions."
image: /img/guides/guide-agentcore-cross-session-memory-strategy.svg
hide_table_of_contents: true
---

![Guide: A Cross-Session Memory Strategy for Bedrock AgentCore](/img/guides/guide-agentcore-cross-session-memory-strategy.svg)

If you want **cross-session memory** for Amazon Bedrock AgentCore, the decision that matters most is not which library you install — it is what identity your memory bank is keyed to. AgentCore Runtime sessions are intentionally short lived: they terminate on inactivity and reprovision fresh environments. Any memory tied to the `runtimeSessionId` evaporates the moment that session turns over, which is exactly why agents that demo well keep forgetting real users in production.

This is a strategy guide, not an install walkthrough. The `hindsight-agentcore` adapter already gives you a clean recall → execute → retain loop around each turn. What it cannot decide for you is the scope of your memory: whether "this user" means one person, one tenant, one agent persona, or some combination. That choice is what makes memory persist across sessions or fragment into per-session islands.

The rest of this guide covers the trade-offs: session (short-term) memory versus long-term memory, how to pick the stable identity that becomes your Hindsight bank, how to keep users isolated from each other, and how to handle multi-agent setups. Keep [the AgentCore integration docs](https://hindsight.vectorize.io/docs/integrations/agentcore) and [the docs home](https://hindsight.vectorize.io/docs) nearby while you decide.

<!-- truncate -->

> **Quick answer**
>
> 1. Never key the Hindsight bank to `runtimeSessionId` — it is ephemeral by design.
> 2. Key it to a stable identity: a validated user ID, usually with tenant and agent name.
> 3. Let session-scoped state stay in AgentCore; put durable memory in Hindsight.
> 4. Isolate users by putting the user ID in the bank so no two users share a bank.
> 5. Give each agent persona its own bank suffix so multi-agent memory stays coherent.

## Why AgentCore agents forget between sessions

AgentCore Runtime sessions are explicitly ephemeral. They end on inactivity and the next invocation gets a fresh environment. That is a feature — it keeps runtimes stateless and disposable — but it means anything you store "in the session" is gone by the next one.

The trap is that `runtimeSessionId` looks like a convenient key for memory. It is stable *within* a session and it is right there in the event payload. If you scope a memory bank to it, everything works in a single conversation and then silently resets when the session churns. You end up with a pile of orphaned single-session banks and an agent that reintroduces itself to the same user every time.

The fix is to move durable memory out of the session entirely and key it to something that outlives session churn.

## Session memory vs long-term memory

These are two different jobs, and conflating them is the root cause of the forgetting problem.

- **Session (short-term) memory** is the working context of the current conversation: the running dialogue, scratch state, the in-flight task. AgentCore Runtime already handles this within a session. It is fine for it to disappear when the session ends.
- **Long-term memory** is what should survive: user preferences, prior decisions, account details, learned patterns. This is what Hindsight stores, and it must not be tied to the session lifecycle.

The design rule that falls out of this: let AgentCore own session state, and let Hindsight own everything that needs to persist. The adapter recalls long-term memory at the start of a turn and enriches the prompt with it, then retains new durable facts after the turn — so each fresh session starts already knowing what earlier sessions learned.

## Choose a stable bank identity

The bank ID is your persistence boundary. Pick an identity that is the same across sessions for the same user, and that you can trust.

The default bank format the adapter uses is:

```
tenant:{tenant_id}:user:{user_id}:agent:{agent_name}
```

That is a good default because it encodes all three axes you usually care about — who the customer is (tenant), who the person is (user), and which agent is talking (agent name). None of those change between sessions, so the bank is stable.

Where the identity comes from matters as much as its shape. Prefer, in order:

1. A validated user ID from the AgentCore JWT/OAuth context (e.g. `jwt_claims["sub"]`).
2. The `X-Amzn-Bedrock-AgentCore-Runtime-User-Id` header.
3. An application-supplied user ID, but only in trusted server-side deployments.

The critical rule: never use a client-supplied identifier you have not validated. An untrusted ID lets one caller read another user's memory. If identity is missing, the safe move is to fail closed rather than fall back to a shared bank — the adapter's custom resolver path raises `BankResolutionError` for exactly this reason.

## Connect AgentCore to Hindsight

Installation and the concrete `configure(...)` / `HindsightRuntimeAdapter` wiring are already covered end to end in the setup guide. This guide is about strategy, so rather than repeat it, follow [Add AgentCore Runtime Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/05/04/guide-agentcore-memory-with-hindsight) to install `hindsight-agentcore`, point it at Hindsight Cloud or a self-hosted endpoint, and wire the recall → execute → retain loop into your handler. Come back here for the bank-scoping decisions.

## Multi-agent and per-user isolation

Two isolation questions come up once you have more than a trivial setup.

**Per-user isolation.** Because the user ID is part of the bank ID, two different users never share a bank. That isolation is only as strong as the identity you trust — which is why the user ID must come from validated auth, not the client. Get that right and cross-user leakage is structurally impossible: a different user resolves to a different bank.

**Multiple agents.** If several agent personas serve the same account, decide deliberately whether they should share memory or not:

- Keep the `agent:{agent_name}` suffix distinct per persona when each agent should have its own coherent memory — a support agent and a sales agent generally should not blur each other's context.
- Drop back to a shared suffix only when personas are genuinely meant to pool what they learn about a user.

If the default `tenant:user:agent` shape does not match your topology, you can override it with a custom `bank_resolver` — for example `f"acme:{context.user_id}:{context.agent_name}"`. Whatever shape you choose, the resolver should fail closed when identity is missing rather than silently return a shared bank.

## Verify that memory persists across sessions

The whole point of this strategy is survival across session churn, so test exactly that:

1. Run one turn for a test user and store a preference or account detail.
2. Let that Runtime session end (or start a genuinely new one) so you are not reusing session state.
3. Trigger a fresh session for the *same* user and ask for the earlier detail.
4. Confirm the adapter recalls it before the agent answers.
5. Repeat with a second user and confirm they cannot see the first user's memory.

If run two answers with details from run one, your bank identity is stable. If it cannot, the bank is almost certainly keyed to something ephemeral — turn on `verbose` logging and check the resolved bank ID against the two sessions.

## Common mistakes

### Keying the bank to runtimeSessionId

This is the mistake this whole guide exists to prevent. The session ID is ephemeral, so a bank scoped to it dies with the session and memory never accumulates.

### Trusting a client-supplied user ID

If the identity is not validated, a caller can point at another user's bank. Always derive the user ID from validated auth (JWT claims or the trusted runtime header).

### Confusing session state with long-term memory

Do not try to make AgentCore's session hold long-term memory, and do not push transient scratch state into Hindsight. Session state stays in AgentCore; durable facts go to Hindsight.

### Blurring multiple agents into one bank by accident

If distinct personas share a bank suffix without meaning to, their context bleeds together. Give each persona its own `agent_name` unless you explicitly want them to share.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `hindsight_api_url` at your own endpoint (for AWS, an in-VPC deployment on ECS/EKS with RDS PostgreSQL plus pgvector). Cloud is just the fastest way to start.

### Why not just use runtimeSessionId as the bank ID?

Because AgentCore Runtime sessions are ephemeral by design. A bank tied to that ID resets on every session turnover, which is the forgetting problem, not a fix for it.

### Should every agent get its own bank?

If each persona should keep coherent, separate memory — yes, keep the `agent_name` suffix distinct. Share a suffix only when personas are meant to pool memory about the same user.

### How is per-user isolation guaranteed?

The user ID is part of the bank ID, so different users resolve to different banks. That guarantee holds as long as the user ID comes from validated auth rather than the client.

## Next Steps

- Follow the [AgentCore setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-agentcore-memory-with-hindsight) for installation and wiring
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [AgentCore integration docs](https://hindsight.vectorize.io/docs/integrations/agentcore)
