---
title: "Per-User Memory for AI Products: Multi-Tenant Patterns"
description: "How to give every user isolated memory in a multi-tenant AI product: hard boundaries vs soft partitions, testing for cross-tenant leakage, GDPR deletion, and scaling."
authors: [benfrank241]
slug: "2026/08/04/per-user-multi-tenant-agent-memory"
date: 2026-08-04T12:00
tags: [hindsight, agent-memory, multi-tenancy, per-user, saas, architecture, deep-dive]
image: /img/blog/per-user-multi-tenant-agent-memory.png
hide_table_of_contents: true
---

![Per-user memory for AI products: hard isolation boundaries versus soft partitions in multi-tenant agent memory](/img/blog/per-user-multi-tenant-agent-memory.png)

The day your AI product recalls one customer's data inside another customer's session, you don't have a personalization bug. You have a breach. Per-user memory looks like a feature and behaves like a security boundary, and most "just add a `user_id`" advice quietly gets that backwards.

If you're building a multi-tenant AI product, memory is where the isolation guarantees you already enforce in your database meet a system that was designed to blur context together on purpose. This post is about how to keep them separate: the patterns, the one decision that matters, and how to prove it holds.

<!-- truncate -->

New to the category? Start with [what agent memory is](https://vectorize.io/what-is-agent-memory). This post assumes you're adding memory to a product with more than one customer and want their data to stay theirs.

## Why multi-tenant memory is a different problem

Single-user memory has one job: recall the right thing. Multi-tenant memory has two jobs, recall the right thing **and** never recall someone else's thing, and when those two goals conflict, isolation wins. A memory system tuned only for recall is happy to surface a semantically similar fact regardless of who it belongs to. That is exactly the failure you cannot ship.

This is the tension every per-user design has to resolve. Personalization wants broad recall across everything the product knows. Privacy wants each user's memory sealed off. The whole game is drawing that seal in a place the system enforces for you, instead of a place you have to remember to enforce on every call.

## The one decision that matters: hard boundary or soft partition?

Almost every "multi-tenant memory" guide reduces to "put a `user_id` on each record and filter by it." That's not wrong, but it hides the decision that actually determines whether you leak. There are two fundamentally different tools, and they are not interchangeable:

| | Hard isolation boundary | Soft partition |
|---|---|---|
| **What it is** | A separate store per tenant/user | A label/filter inside one shared store |
| **Enforced by** | The storage layer | Your query, at call time |
| **Use for** | Tenants, customers, per-user data where a leak is a bug | Slices you sometimes want to cross-reference (projects, topics) |
| **Failure mode** | Hard to leak — there's no cross-boundary query | Leaks the moment a filter is forgotten or wrong |

In Hindsight the hard boundary is a **bank**. A [bank is a recall boundary](/blog/2026/07/16/bank-strategy-agent-memory): `recall`, `retain`, and `reflect` each name exactly one `bank_id` and stay inside it, and there is no built-in query that spans banks. The soft partition is a **tag**: you attach tags when you retain and filter by them when you recall.

Here's the part people miss, and it's the whole reason this matters. **A filter you can forget to pass is not isolation.** Tag filtering in Hindsight defaults to a mode (`any`) that even *includes untagged memories*, because tags are meant as organizing hints, not walls. That default is the tell: if a memory absolutely must never surface in the wrong context, do not gate it behind a tag someone can omit. Put it behind a boundary the storage layer enforces.

So the rule for per-user products is short: **the user (or tenant) is a hard boundary, so it gets a bank. Everything else is a tag.**

## Per-user, per-org, and shared knowledge

Real products have more than two scopes. Usually three:

- **Private to the user** — this person's conversations, preferences, and history. Hard boundary.
- **Shared within an org/workspace** — team conventions, shared projects, account-level facts every seat should see. Hard boundary at the org level.
- **Global product knowledge** — docs, onboarding, defaults that everyone gets. A shared, read-mostly store.

The mapping is direct. Make the bank the thing that must never leak: `bank_id="user:{id}"` for a consumer product where each person's memory must never touch anyone else's, or `bank_id="org:{id}"` for B2B where seats in the same account should share context but two customers must not. Then use tags for the *kinds* of memory inside that boundary, project, source, sensitivity, so you can filter when you want and cross-reference when you want, without ever punching through the wall.

A concrete miss makes it stick. A support SaaS scoped its memory by a `customer_id` **tag** in one shared bank instead of a bank per customer. It worked in testing. Then a single retain call went out without the tag, one forgotten parameter, and because the default match mode includes untagged memories, Customer A's contract terms surfaced in Customer B's session. The fix wasn't a stricter filter. It was a real boundary: one bank per customer, so there was nothing to forget.

## Reading across boundaries without a cross-bank query

Hard boundaries raise an obvious question: if `recall` only ever names one bank, how does an agent use the user's private memory *and* the org's shared knowledge *and* the product's global docs in a single answer? You don't punch a hole in the boundary to do it. You compose at the application layer.

The pattern is a fan-out: recall from each bank the current caller is entitled to, then merge and rank the results before they reach the model. The entitlement check lives in your code, where it belongs, and each recall still stays inside its own boundary.

```python
def recall_for(user, query, k=8):
    scopes = [
        user_bank(user.id),          # private to this user
        f"org:{user.org_id}",        # shared within the org
        "product:global",            # read-mostly product knowledge
    ]
    hits = []
    for bank in scopes:
        hits += hs.recall(bank_id=bank, query=query, max_results=k)
    return rank(hits)[:k]            # merge + rerank in your app
```

Two properties fall out of this. A user only ever reads the banks your entitlement logic hands them, so there is still no way to address another user's bank. And the scopes compose by *addition*: you broaden an answer by adding a bank the caller is allowed to see, never by loosening a filter. Writes are the mirror image, they go to exactly one bank, the private one, unless a fact is genuinely org-level, in which case it's written to the org bank on purpose rather than by accident.

## Isolation you can prove

If tenant isolation is a security property, you test it like one. Don't assume the boundary holds because the happy path looks right. This is the same discipline as building [an eval you actually trust](/blog/2026/07/31/evaluate-agent-memory-system), applied to isolation instead of retrieval.

A minimal cross-tenant leakage test:

1. **Store as A, read as B.** Retain a distinctive fact in tenant A's bank. Recall it from tenant B's bank. Assert nothing returns.
2. **Fuzz with shared entities.** Give two tenants the same entity names ("the Acme account," "our staging server") and confirm [entity resolution](/blog/2026/06/29/entity-resolution-agent-memory) stays inside each boundary rather than merging across them.
3. **Attack the soft partition.** If you rely on tags anywhere, retain without a tag and confirm the untagged-default behavior doesn't surface it in a filtered query. This is where tag-only "isolation" fails.
4. **Delete and re-check.** Remove tenant A entirely, then confirm A's memories are gone and B's are untouched.

The core of it is three lines you can assert on, and everything else is variations:

```python
hs.retain(bank_id="tenant:A", content="A's secret: the launch date is March 3.")
leaked = hs.recall(bank_id="tenant:B", query="what is the launch date?")
assert leaked == [], f"cross-tenant leak: {leaked}"
```

Run it in CI, not once by hand. Boundaries drift when someone adds a new code path that constructs a `bank_id` from the wrong variable, and the only way you catch it is a test that stores as one tenant and reads as another.

## Per-user deletion and the GDPR question

Right-to-be-forgotten is where the boundary choice pays off, or bites you. If the user is the boundary, deletion is one operation: drop that user's bank and the memories, extracted entities, and the graph built from them go with it. Clean, complete, auditable.

Compare that to the soft-partition version. If every user's facts live in one shared index behind a `user_id` filter, "delete this user" becomes a surgical operation: find every row, every derived embedding, every graph edge that touches them, and remove it without disturbing anyone else, and then prove you got all of it. One of these you can hand to a compliance reviewer. The other you hope you got right. Isolation at the storage layer turns a deletion audit into a one-liner.

## Scaling to many tenants without an explosion

The obvious objection to "a bank per user" is cost: does ten thousand users mean ten thousand databases? It shouldn't, and the answer is an architecture question, not a modeling one.

Hindsight stores banks as scoped partitions of a single system rather than a separately provisioned database each, so the boundary is logical, enforced on every query, without the operational weight of standing up infrastructure per tenant. That's what lets the [system scale](/blog/2026/05/08/how-hindsight-scales) to many boundaries: you get the isolation of separate stores with the footprint of one. When you evaluate any memory system for multi-tenant use, this is the question to ask, because a system that needs a real database per tenant will make you choose between isolation and your infra bill, and that's a choice you shouldn't have to make.

A fintech team building multi-user AI [did exactly this from day one](/blog/2026/04/13/hindsight-financial-ai-memory-customer-story): per-tenant isolation as the boundary, tags for the softer slices within it, on a single self-hosted deployment.

## Putting it together

The pattern in code is small, which is the point. The boundary does the work.

```python
# Hard boundary: the user (or org) is the bank.
def user_bank(user_id: str) -> str:
    return f"user:{user_id}"

# Retain into the caller's bank only. Tags are for slices *within* the boundary.
hs.retain(
    bank_id=user_bank(current_user.id),
    content="Prefers dark mode and metric units. Works mostly on the billing service.",
    tags=["project:billing"],
)

# Recall never names another user's bank, so cross-user recall is not expressible.
hits = hs.recall(
    bank_id=user_bank(current_user.id),
    query="what are this user's preferences for the billing work?",
    tags=["project:billing"],   # soft filter inside the boundary
)
```

Two things to hold onto. First, `bank_id` is always derived from the authenticated caller, never from anything a request body can influence, that's the line that keeps tenant A from ever addressing tenant B's bank. Second, tags choose *which slice* inside the user's own memory, and their match modes (`any` includes untagged, `all` requires all tags, the `_strict` variants exclude untagged, `exact` matches the set) are about organization, not security. Reach for a new bank, not a stricter tag, whenever the answer to "should a memory stored by A be recallable by B?" is no.

## Three mistakes that cause leaks

- **Deriving `bank_id` from the request.** If any part of the bank id comes from a request body, a header, or a client-supplied field, a caller can address someone else's bank. Derive it from the authenticated session, only.
- **Using tags as the tenant wall.** Tags are organization, not isolation, and the default match mode includes untagged memories. A single retain without the tag leaks. Tenants get banks.
- **No leakage test in CI.** Boundaries drift the day someone adds a code path that builds the id from the wrong variable. The store-as-A, read-as-B test is the only thing that catches it before a customer does.

## Isolation first, personalization second

Three things to take with you. Multi-tenant memory optimizes for recall and isolation at once, and isolation wins ties. The decision that determines whether you leak isn't the schema, it's whether your tenant boundary is enforced by the storage layer or by a filter you have to remember, so make the tenant a hard boundary and make everything else a tag. And treat isolation as a testable property: store as one tenant, read as another, assert nothing comes back, in CI.

Get the boundary right and the rest gets easier, not harder. Deletion becomes a one-liner, personalization is free to be broad *inside* each user's own memory, and you stop hoping a filter was passed on every path through your code.

If you want a storage-enforced boundary without standing up a database per tenant, Hindsight gives you banks as isolated recall boundaries on a single deployment, self-hosted with one Docker command or managed on [Hindsight Cloud](https://hindsight.vectorize.io). Whatever you build on, pick the boundary before you pick the schema.

---

**Further reading:**
- [One Bank or Many? A Field Guide to Structuring Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — the banks-vs-tags model in depth
- [The 10 Things to Look For in an Agent-Memory System](/blog/2026/07/31/evaluate-agent-memory-system) — where tenant isolation fits in evaluation
- [How Hindsight Scales](/blog/2026/05/08/how-hindsight-scales) — many boundaries, one store
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems) — the broader landscape
