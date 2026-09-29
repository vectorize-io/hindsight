"""Give the bank something worth searching for, and make its pages real.

Two problems this fixes, both measured rather than guessed:

1. **The pages were empty.** The plugin seeds five knowledge pages at
   SessionStart and refreshes them on a ``H * * * *`` cron, so inside a session
   that never fires: the roster injected into the model listed pages whose
   content was the literal string ``Generating content...``. The first thing the
   model learned about this repo's memory was that it is empty.

2. **Nothing in memory changed an answer.** Seeded from a 7-commit repo, memory
   only ever repeated what the code already said, and an agent that reads the
   code loses nothing by skipping the search. The documents below are written so
   the opposite is true: each carries a decision that is *not* recoverable from
   the source — the free-shipping threshold is judged on the discounted total,
   `reserve` must reject quantity 0, the catalog stays uncached on purpose. A
   turn that skips the search gets these wrong.

**Why the placeholder pages are replaced rather than refreshed.** This
describes a server that still created pages with a placeholder body; a current
one creates them empty, which falls back to a full refresh instead of hitting
the guard below. The replacement is kept because ``--api-url`` can point this
harness at either. Refreshing a placeholder page fails on the server's own
guard — ``delta operations did not reach the document, and the reflect
candidate covers only memories newer than the last refresh, so writing it would
drop the rest of the document`` — because the page was created with a watermark
and a placeholder body, and every prefilled memory is newer than that. The
guard is right in general and unhelpful here, where the document it is
protecting is the placeholder. A page created *after* the documents are in has
no such history, so its first synthesis covers everything.

That replacement is also what an established repo looks like: pages with real
bodies, which is the steady state the cron reaches after a day of use and the
state this harness needs in order to measure anything about searching.

Everything goes through the published client — no engine access, same rule as
the rest of this package.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

from hindsight_client import Hindsight

from hindsight_system_evals.coding_agents.scenario import SettleFn

#: A never-refreshed page carries an empty body, and the emptiness checks below
#: catch that on their own. This string is what older servers wrote instead, kept
#: because ``--api-url`` can point this harness at any deployment.
PLACEHOLDER = "Generating content..."

#: How long to wait for the prefill's own background work. Shorter than the
#: suite's settle because nothing here waits on consolidation — but it is a real
#: synthesis per page, so not short.
SETTLE_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 2.0


@dataclass(frozen=True)
class PriorSession:
    """One past working session, as its write-back would have stored it."""

    context: str
    content: str


@dataclass(frozen=True)
class PageNode:
    """One page in the knowledge tree, flattened out of its folder structure."""

    id: str
    name: str
    mental_model_id: str | None


@dataclass(frozen=True)
class SeedPage:
    """A page to create once the documents are in, with the query it synthesises from."""

    name: str
    source_query: str


#: Written as a developer and an agent talking, not as documentation: that is the
#: shape the plugin's own session write-back produces, and a bank full of tidy
#: prose would be a bank this harness never actually sees in the field.
PRIOR_SESSIONS: list[PriorSession] = [
    PriorSession(
        context="pricing: discounts and the free shipping threshold (decided, not yet built)",
        content=(
            "User: we'll want percentage discounts on carts soon. before anyone writes it: does "
            "the free shipping threshold look at the original subtotal or the discounted one?\n"
            "Assistant: Nothing discounts today, so the code cannot tell you — it is a decision to "
            "make now.\n"
            "User: discounted. the threshold is about what the customer actually pays for goods. "
            "so a 55 EUR cart with 20% off is 44 EUR and pays shipping. write it down, because "
            "whoever implements it will read the current code, see the threshold on the raw "
            "subtotal, and assume that was the decision. it wasn't — there was no discount then.\n"
            "Assistant: Recorded as the decision for the future implementation: apply the discount "
            "to the subtotal FIRST, then compare that discounted amount against "
            "free_shipping_threshold, then tax."
        ),
    ),
    PriorSession(
        context="pricing: why Decimal everywhere",
        content=(
            "User: why is everything Decimal in pricing, it makes the code noisy\n"
            "Assistant: A float subtotal drifted by a cent on carts with three or more lines and "
            "the totals stopped matching the invoices finance sends out. We moved money to Decimal "
            "and quantize every output with ROUND_HALF_UP to two places. The rule is: money never "
            "touches float, not even for an intermediate.\n"
            "User: yeah that was a bad week. keep it."
        ),
    ),
    PriorSession(
        context="inventory: reserving zero or negative quantities",
        content=(
            "User: someone posted reserve with quantity 0 and got a 200 back, that's wrong\n"
            "Assistant: reserve() only checks available < quantity, so 0 passes and reserves "
            "nothing.\n"
            "User: a reservation of 0 is a client bug, not a no-op. reject anything <= 0 with the "
            "same error the pricing layer uses for a bad quantity — 400, not 409. 409 is only for "
            "genuinely out of stock."
        ),
    ),
    PriorSession(
        context="api: how errors are reported",
        content=(
            "User: where do we map exceptions to status codes\n"
            "Assistant: Only in orderbook/api.py. The domain modules raise their own exceptions "
            "(PricingError, OutOfStock, LookupError for an unknown sku) and never import the web "
            "framework; api.py is the single place that turns those into 400 / 409 / 404 with "
            "{'error': str(exc)}.\n"
            "User: right, and it stays that way. the day a domain module returns a status code we "
            "have two places to keep in sync."
        ),
    ),
    PriorSession(
        context="catalog: the cache we decided not to add",
        content=(
            "User: catalog lookups aren't cached, is that an oversight?\n"
            "Assistant: Deliberate. Prices change during flash sales and a stale price is a "
            "customer-visible error, while the lookup is a single indexed read. We tried a 60s TTL "
            "cache once and rolled it back the same day after a sale price took a minute to appear.\n"
            "User: yep. no cache until someone shows me the lookup is actually hot."
        ),
    ),
    PriorSession(
        context="inventory: why reservations are in memory",
        content=(
            "User: reservations are just a dict, what happens with two processes\n"
            "Assistant: They break — this service is deliberately single-process today. A "
            "threaded version was proposed and rejected: the reserve path is read-then-write with "
            "no lock, so it needs a real store rather than a mutex bolted on.\n"
            "User: correct. when we need to scale it, it moves to the database, not to threads."
        ),
    ),
    PriorSession(
        context="config: why pricing settings are a frozen dataclass",
        content=(
            "User: why is PricingConfig frozen, I wanted to tweak the tax rate at runtime\n"
            "Assistant: It is loaded once at startup and shared by every request. When it was "
            "mutable a test changed tax_rate and every later test in the same process priced "
            "differently — the failures looked random and cost an afternoon.\n"
            "User: right. a new rate is a restart, or a new config object passed into quote(). "
            "never mutate the shared one."
        ),
    ),
    PriorSession(
        context="api: the endpoints and what they return",
        content=(
            "User: what's the api surface right now\n"
            "Assistant: Two handlers in orderbook/api.py. post_quote takes {'items': [{sku, "
            "quantity}]} and returns the quote dict — currency, subtotal, tax, shipping, total — "
            "or 404 for an unknown sku and 400 for an unpriceable cart. post_reserve takes the "
            "same items and returns {'reserved': True}, or 409 when stock is short.\n"
            "User: keep the response a plain dict. no envelope, no status field inside the body — "
            "the status code is the status."
        ),
    ),
    PriorSession(
        context="structure: what each module owns",
        content=(
            "User: remind me what lives where\n"
            "Assistant: config.py holds the pricing knobs; pricing.py turns a cart into a quote; "
            "catalog.py resolves a sku to a unit price; inventory.py holds reservations; api.py is "
            "the only module that knows about HTTP.\n"
            "User: and that split is the point — pricing must stay callable from a script with no "
            "web server anywhere near it."
        ),
    ),
    PriorSession(
        context="tests: what a change is expected to ship with",
        content=(
            "User: do I need a test for this\n"
            "Assistant: Convention here: every pricing or inventory rule ships with a pytest case "
            "in tests/, asserting on the returned dict. No mocks for pure functions — they take "
            "plain arguments and return plain values, so a mock would only test itself.\n"
            "User: and name the test after the rule, not after the function."
        ),
    ),
]

#: Named the way the plugin names its own pages, so the roster the model sees
#: reads like an established repo rather than like a fixture.
#:
#: There are a dozen of them for a measured reason. With three pages the roster
#: injected at SessionStart *is* the index: the model reads the titles, picks the
#: obviously relevant one and calls hindsight_read_knowledge_page with its id —
#: measured at 5/5 on the one turn whose answer lived in memory, with zero
#: searches. Searching to find one of three titles you can already see would be a
#: wasted call, and the model is right not to make it. A roster the size of a
#: real repo's (the author's own has 29) is what makes search the cheap way in.
SEED_PAGES: list[SeedPage] = [
    SeedPage(
        name="Pricing decisions",
        source_query="What has been decided about pricing: discounts, the free shipping threshold, tax and rounding?",
    ),
    SeedPage(
        name="Money and numeric precision",
        source_query="Why is money Decimal here, what rounding is used, and what went wrong with floats?",
    ),
    SeedPage(
        name="Inventory and reservations",
        source_query=(
            "What has been decided about inventory reservations: invalid quantities, which "
            "status codes they return, and why reservations are held in memory?"
        ),
    ),
    SeedPage(
        name="Error handling and status codes",
        source_query="Where do domain exceptions become HTTP status codes, and which code means what?",
    ),
    SeedPage(
        name="Caching decisions",
        source_query="What is deliberately not cached in this project, and why was a cache rolled back?",
    ),
    SeedPage(
        name="Testing conventions",
        source_query="What does a change ship with here, where do tests live, and when are mocks used?",
    ),
    SeedPage(
        name="Concurrency and scaling",
        source_query="Why is this service single-process, and what was decided about threads versus a real store?",
    ),
    SeedPage(
        name="Configuration and startup",
        source_query="How is pricing configuration loaded and changed, and why is it a frozen dataclass?",
    ),
    SeedPage(
        name="Catalog and pricing data",
        source_query="How are unit prices looked up, and what has been decided about price freshness?",
    ),
    SeedPage(
        name="API surface",
        source_query="What endpoints exist, what shape do they return, and what does each status code mean?",
    ),
    SeedPage(
        name="Component map",
        source_query="What are the modules of this project and what is each responsible for?",
    ),
    SeedPage(
        name="Key decisions and rationale",
        source_query="What are the durable decisions in this repository and the reasoning behind each?",
    ),
]


async def prefill_bank(client: Hindsight, bank_id: str, settle: SettleFn) -> None:
    """Retain the prior sessions, then give the bank pages with real content."""
    await client.aretain_batch(
        bank_id=bank_id,
        items=[{"content": s.content, "context": s.context} for s in PRIOR_SESSIONS],
    )
    await settle(bank_id)
    await drop_placeholder_pages(client, bank_id)
    await create_seed_pages(client, bank_id)


async def drop_placeholder_pages(client: Hindsight, bank_id: str) -> list[str]:
    """Delete the pages the plugin seeded but never filled. Returns their names.

    Only the placeholders: a page that already has content is knowledge this run
    should keep, and deleting it would be the harness editing the thing it
    measures.
    """
    models = {m.id: m for m in (await client.alist_mental_models(bank_id=bank_id, detail="content")).items}
    dropped = []
    for node in await _page_nodes(client, bank_id):
        model = models.get(node.mental_model_id) if node.mental_model_id else None
        content = (model.content if model else "") or ""
        if content.strip() and PLACEHOLDER not in content:
            continue
        with contextlib.suppress(Exception):
            await client.adelete_knowledge_node(bank_id=bank_id, node_id=node.id)
            dropped.append(node.name)
    return dropped


async def create_seed_pages(client: Hindsight, bank_id: str) -> list[str]:
    """Create the pages and synthesise each one, in order.

    One at a time: each refresh is a full reflect, and firing three at one small
    server only makes all three slow. `refresh_after_consolidation` is off and
    there is no cron — this harness decides when a page is written, so a
    background refresh cannot change the roster mid-measurement.
    """
    created = []
    for page in SEED_PAGES:
        response = await client.acreate_knowledge_page(
            bank_id=bank_id,
            name=page.name,
            source_query=page.source_query,
            trigger={"refresh_after_consolidation": False, "refresh_cron": None},
        )
        await client.arefresh_mental_model(bank_id=bank_id, mental_model_id=response.mental_model_id)
        created.append(page.name)
    await _settle_ignoring_failures(client, bank_id)
    return created


async def placeholder_pages(client: Hindsight, bank_id: str) -> list[str]:
    """The pages whose content is still empty or the server's placeholder.

    Checked rather than assumed: a refresh that fails leaves the page in exactly
    the state this harness exists to avoid, and the failure is silent — the page
    is still listed in the roster the model sees.
    """
    models = await client.alist_mental_models(bank_id=bank_id, detail="content")
    return [m.name for m in models.items if not (m.content or "").strip() or PLACEHOLDER in (m.content or "")]


async def _page_nodes(client: Hindsight, bank_id: str) -> list[PageNode]:
    """Every page in the knowledge tree, flattened."""
    tree = await client.aget_knowledge_base_tree(bank_id=bank_id)
    out: list[PageNode] = []

    def walk(nodes) -> None:
        for node in nodes or []:
            if getattr(node, "kind", None) == "page":
                out.append(
                    PageNode(id=node.id, name=node.name, mental_model_id=getattr(node, "mental_model_id", None))
                )
            walk(getattr(node, "children", None))

    walk(getattr(tree, "roots", None))
    return out


async def _settle_ignoring_failures(client: Hindsight, bank_id: str) -> None:
    """Wait for background work, tolerating a failed operation.

    The suite's `wait_until_settled` raises on any failed operation in the bank,
    which is right for an eval asserting on a pipeline. Here a single page whose
    refresh the server declined must not abort a run: what matters is whether the
    pages ended up with content, and `placeholder_pages` answers that directly.
    """
    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    quiet = 0
    while time.monotonic() < deadline:
        busy = [
            op
            for status in ("pending", "processing")
            for op in (await client.operations.list_operations(bank_id, status=status, limit=100)).operations
        ]
        quiet = 0 if busy else quiet + 1
        if quiet >= 2:
            return
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
