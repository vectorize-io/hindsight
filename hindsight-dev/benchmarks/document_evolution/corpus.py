"""Seed documents and the fact streams that evolve them.

Each case is a document plus a sequence of rounds. A round contributes one new
fact and declares what should be true of the document afterwards, so coverage
can be judged without pinning wording:

- ``asserts`` — a claim the document must support once the round has landed.
- ``supersedes`` — a claim that must no longer be stated, when the fact replaces
  an earlier one. This is what catches a document that accretes contradictions.

``fragile`` marks the constructs a markdown round-trip used to destroy. The
corpus deliberately also carries a case with none of them (``playbook``): the
release question is not only "is the damage gone" but "is everything else at
least as good as before", and a document that never contained a table is where
that is measured.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Round(BaseModel):
    """One refresh: a new fact, and what the document must say afterwards."""

    fact: str
    asserts: str
    supersedes: str | None = None


class Case(BaseModel):
    """A seed document and the stream of facts that edits it."""

    name: str
    topic: str = Field(description="The mental model's source_query — what the document answers.")
    seed_memories: list[str] = Field(
        description="Retained before the first refresh, so a generated seed has something to say."
    )
    seed_document: str = Field(description="Authored seed, used when the run can write it directly.")
    rounds: list[Round]
    fragile: bool = Field(description="Whether the seed carries constructs a markdown round-trip can destroy.")


API_REFERENCE = Case(
    name="api-reference",
    topic="Document the memory API: its operations, latency budgets, failure handling and procedure.",
    fragile=True,
    seed_memories=[
        "The memory API exposes retain and recall operations.",
        "retain stores a memory and budgets 12ms; recall retrieves memories and budgets 40ms.",
        "Transient errors are retried; schema errors fail loudly.",
    ],
    seed_document="""## Purpose

Reference for the memory API: what each operation does, its latency budget, and how failures are handled.

## Operations

| Operation | Description | Budget |
| --- | --- | --- |
| `retain` | Store a memory | 12ms |
| `recall` | Retrieve memories | 40ms |

## Failure Handling

- Retry transient errors
  - Nested under retries
    - Deeper still, three levels
- Fail loudly on schema errors

## Example

```python
def handler(request):

    return {"ok": True}
```

## Constraints

Latency budget is 200ms  
measured at the p95.

> Never block the request path on consolidation.

## Procedure

5. Fifth step, numbering starts at five on purpose
6. Sixth step
""",
    rounds=[
        Round(
            fact="The recall endpoint gained a rerank stage that adds about 15ms, taking its budget to 55ms.",
            asserts="recall includes a rerank stage and its latency budget is 55ms.",
            supersedes="recall's budget is 40ms.",
        ),
        Round(
            fact="Schema errors are now reported with the offending field name.",
            asserts="A schema error names the offending field.",
        ),
        Round(
            fact="A new operation, reflect, answers questions by synthesizing stored memories, budgeting 80ms.",
            # Tests the fact, not one phrasing of it. The first version demanded the
            # document say reflect "answers questions"; documents that described it
            # as synthesising stored memories — the same operation, the wording the
            # source fact itself uses — were scored as missing it. A claim that a
            # correct document can fail measures the corpus, not the pipeline.
            asserts="There is a reflect operation over stored memories, with a budget of 80ms.",
        ),
        Round(
            fact="The p95 latency budget was raised from 200ms to 250ms.",
            asserts="The p95 latency budget is 250ms.",
            supersedes="The p95 latency budget is 200ms.",
        ),
        Round(
            fact="Consolidation runs on a background worker every five minutes.",
            asserts="Consolidation runs on a background worker every five minutes.",
        ),
    ],
)

PLAYBOOK = Case(
    name="onboarding-playbook",
    topic="Document how a new engineer is onboarded: who does what, in what order, and when they are done.",
    fragile=False,
    seed_memories=[
        "New engineers are onboarded by an assigned buddy over their first two weeks.",
        "Onboarding starts with laptop setup and repository access.",
        "An engineer is considered onboarded once they have shipped one change to production.",
    ],
    seed_document="""## Purpose

How a new engineer is brought up to speed in their first two weeks, and who is responsible for each part of it.

## Roles

Every new engineer is assigned a buddy for their first two weeks. The buddy answers questions, reviews their
first changes, and is the person they ask before asking anyone else.

The hiring manager owns the plan and checks in at the end of each week.

## Sequence

The first day is laptop setup and repository access. The rest of the first week is reading: the architecture
notes, the deployment runbook, and the last quarter of incident reviews.

In the second week the engineer picks up a small, well-scoped change and ships it with their buddy reviewing.

## Done

An engineer is onboarded once they have shipped one change to production and can describe how it got there.
""",
    rounds=[
        Round(
            fact="Buddies are now assigned by the team lead a week before the new engineer starts, not on day one.",
            asserts="The buddy is assigned by the team lead a week before the start date.",
            supersedes="The buddy is assigned on the engineer's first day.",
        ),
        Round(
            fact="Laptop setup is now handled by IT before the start date, so day one begins with repository access.",
            asserts="IT completes laptop setup before the start date and day one begins with repository access.",
            supersedes="The first day is spent on laptop setup.",
        ),
        Round(
            fact="The reading list gained the on-call handbook, which every new engineer must read in week one.",
            asserts="The week-one reading list includes the on-call handbook.",
        ),
        Round(
            fact="Engineers now pair with their buddy for the first production deploy rather than only being reviewed.",
            asserts="The engineer pairs with their buddy on the first production deploy.",
        ),
        Round(
            fact="The hiring manager check-in moved from weekly to a single conversation at the end of week two.",
            asserts="The hiring manager checks in once, at the end of the second week.",
            supersedes="The hiring manager checks in at the end of every week.",
        ),
    ],
)

# The shape #3361 was reported against, in the position it was reported in.
#
# Two details matter and the obvious version of this case has neither. First, the
# table has a row that does not carry both outer pipes — it renders as a table,
# people write it and models emit it, and a parser demanding both pipes on every
# row classified the whole block as prose. Second, it lives in a section the fact
# stream never concerns. A model editing a malformed table tends to rewrite it
# correctly, which repairs the damage before it can be measured; the reported
# failure was in "sections the operations never named", where nothing was
# supposed to change and so nothing could repair it.
MALFORMED_TABLE = Case(
    name="release-runbook",
    topic="Document the release runbook: the cadence, the checklist, and who signs off.",
    fragile=True,
    seed_memories=[
        "Releases are cut on Tuesdays and go to staging before production.",
        "A release needs sign-off from the on-call engineer.",
    ],
    seed_document="""## Cadence

Releases are cut on Tuesdays. A release that misses Tuesday waits for the next one rather than going out late.

## Sign-off

A release needs sign-off from the on-call engineer before it reaches production.

## Checklist

- Migrations applied
- Smoke tests green
- On-call engineer signed off

## Environment reference

Reference only — the release process does not change these.

| Environment | Region | Notes |
|---|---|---|
| staging | eu-west-1 | Mirrors production at a tenth of the size |
| production | eu-west-1 | Live traffic

Contact for access is the platform team  
via their shared channel.

> Environments are provisioned from infrastructure-as-code; do not edit them by hand.
""",
    rounds=[
        Round(
            fact="Releases moved from Tuesdays to Wednesdays.",
            asserts="Releases are cut on Wednesdays.",
            supersedes="Releases are cut on Tuesdays.",
        ),
        Round(
            fact="Sign-off now needs both the on-call engineer and the release manager.",
            asserts="Sign-off needs the on-call engineer and the release manager.",
            supersedes="Sign-off needs only the on-call engineer.",
        ),
        Round(
            fact="The checklist gained a rollback rehearsal, which happens before sign-off.",
            asserts="A rollback rehearsal happens before sign-off.",
        ),
        Round(
            fact="Smoke tests now run automatically in CI rather than manually.",
            asserts="Smoke tests run automatically in CI.",
            supersedes="Smoke tests are run manually.",
        ),
        Round(
            fact="A release that fails its rollback rehearsal is blocked until the rehearsal passes.",
            asserts="A failed rollback rehearsal blocks the release.",
        ),
    ],
)

CASES: list[Case] = [API_REFERENCE, PLAYBOOK, MALFORMED_TABLE]


def case_by_name(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown case {name!r}; have {[c.name for c in CASES]}")
