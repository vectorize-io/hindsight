"""Retain cases where the meaning of a message lives in another chunk (#6645).

Retain splits a document into chunks and extracts facts from each chunk on its own. A short
reply — "all of them except 5 and 7", "1) yes 2) discord general", "the cheaper one" — is only
a fact together with the message it answers. When that message is long, the reply opens the
next chunk and extraction sees it with nothing to refer to.

Each case is one JSON conversation, retained once, under a ``retain_chunk_size`` chosen so the
reply and what it refers to land in different chunks — and the case checks that they do,
through the public chunks API, so a chunker change cannot quietly turn it into a one-chunk case.
The referent is never in the last few words before the boundary: the options sit at the top of
a long message, or two messages back, or in the chunk after. Peeking at a sentence of the
neighbour does not recover it.

Every case has a ``-one-chunk`` control: the same input under a chunk size that keeps it whole.
The control passing while the case fails is the defect; both failing is the model.

The criteria ask for ONE fact that makes sense read alone. Recall returns facts one at a time,
and a judge shown the whole list will happily join "proceed with all items except 5 and 7" to
"dropping audit_2023 is item 5" — a join no reader of a single recalled fact can make.
"""

from __future__ import annotations

import json
from dataclasses import replace

from hindsight_system_evals.retain_capabilities import Case, Item

#: Small enough to split each conversation at its boundary, large enough for any one message.
SPLIT_CHUNK_SIZE = 1200
#: Keeps every conversation here in one chunk: the control.
WHOLE_CHUNK_SIZE = 20000


def _conversation(*messages: tuple[str, str]) -> str:
    return json.dumps([{"role": role, "content": content} for role, content in messages], ensure_ascii=False)


_MAINTENANCE_PLAN = (
    "Here is the maintenance plan for this week. Each point is independent, so you can pick any subset.\n"
    "1) Rotate the Stripe API keys. The current secret key has been in use since 2024 and is shared by the "
    "checkout service and the invoicing cron.\n"
    "2) Upgrade Postgres on db-main from 15 to 17. We need a 20-minute write freeze, best on Saturday night.\n"
    "3) Delete the legacy-exports S3 bucket. Nothing has read from it in eleven months and it costs about "
    "$300 a month.\n"
    "4) Turn on Dependabot for the backend and frontend repositories, with weekly grouped pull requests.\n"
    "5) Drop the audit_2023 table. It holds 180 GB and the compliance team said the data is archived elsewhere, "
    "but I could not confirm that myself.\n"
    "6) Move CI from GitHub-hosted runners to the two self-hosted machines in the office rack.\n"
    "7) Email every customer about last Tuesday's 40-minute outage, with the postmortem attached.\n"
    "Points 2 and 6 need a short maintenance window; the others can run any time. Tell me which ones to run "
    "and I will schedule them and report back on each when it is done."
)

_THREE_QUESTIONS = (
    "Before I set up the release automation I need three answers from you.\n"
    "(1) Authorisation: may I deploy api-gateway, billing-worker and search-indexer to production on my own, "
    "without asking you each time? These are the three services with full test coverage; everything else would "
    "still need your sign-off.\n"
    "(2) Reports: where should I post the weekly release report? The options are the #eng-reports channel on "
    "Slack, the #general channel on the team Discord, or an email to the leads list.\n"
    "(3) Logs: production logs older than 90 days take 2 TB on the log cluster. Should I delete them on a "
    "rolling basis, or keep them longer?\n"
    "The automation will run every night at 02:00 UTC once you answer. I will write a short summary of the "
    "setup into the runbook so the rest of the team knows what changed, who approved it and how to turn it off "
    "if something goes wrong during the first week. Nothing will be deployed until I have all three answers, "
    "and I will ping you again on Thursday if I still have not heard back."
)

_FLATS_DISCUSSION = (
    "Thanks, that helps. Some things to weigh between the two. The Via Po flat is five minutes on foot from "
    "your office, so you would never need the car for commuting, and the street has the market on Saturdays. "
    "The downside is the climb: with no lift, groceries and the bike go up four floors every time, and the "
    "building is old, so ask about the heating and when the windows were last replaced. The San Salvario one "
    "means a 25-minute tram ride on line 9, but the garden is a real plus if you want to work outside in summer, "
    "and ground floor flats there tend to be cooler. Ground floor also means you should check for damp in the "
    "bedroom walls and ask whether the shutters lock. In both cases ask the landlord for the last two years of "
    "condominium expenses, check whether the deposit is two or three months, and read the clause on early exit, "
    "since you said you may move to Milan in a couple of years. If you want I can draft the questions to send."
)

_MANAGER_RANT = (
    "Ugh, the week has been a lot. My new manager started on Monday and changed half of how we work already. "
    "She cancelled the team offsite we had planned for October in Lisbon, saying it was not a good use of the "
    "budget this quarter. She moved standups to 8:30 every day, which is brutal for the people who drop their "
    "kids at school, and she wants every pull request reviewed by her personally before it merges, at least "
    "for the next two months. She also asked each of us for a written list of what we worked on since January "
    "by Friday. To be fair she is clearly sharp, and she found a billing bug in her second hour that nobody "
    "had noticed for months. But the team is nervous, two people already asked me if I am looking around, and "
    "I honestly do not know yet what to tell them or how much of this is going to stick after the first month. "
    "My plan for now is to keep my head down, ship the search rewrite on time, and ask for a proper one-to-one "
    "next week so I understand what she actually expects from a senior engineer on this team. If it still feels this "
    "chaotic by the end of the quarter, I will talk to my old lead about moving back to the data platform group."
)

_MIGRATION_PROPOSAL = (
    "Here is the proposal for the billing migration, as you asked. We move from Chargebee to Stripe Billing "
    "during March. Tomás leads it, with one backend engineer full time and support from finance for the "
    "invoice templates. The budget is 40k euros: about 22k of engineering time, 8k for the Stripe migration "
    "service, and 10k for a two-week parallel run where both systems invoice and we compare every invoice "
    "before switching off Chargebee. The parallel run is the safest way to catch rounding and tax differences, "
    "but it is also the part that doubles the work for finance during those two weeks. Customers on annual "
    "plans keep their renewal dates; monthly customers move on their next cycle. The riskiest piece is the "
    "VAT handling for our EU customers, which Stripe Tax would take over. If you approve, Tomás starts on "
    "3 March and we go live by the 31st. Finance asked for a decision this week so they can plan the March close "
    "around it, and support wants two weeks' notice before any customer-facing invoice change. Do you approve, "
    "and do you want any changes to scope or budget?"
)

_CASES: tuple[Case, ...] = (
    Case(
        id="reply-to-a-numbered-plan",
        category="chunk_boundary",
        origin="#6645: 'all of them except 5 and 7' stored as 'an unspecified list except 5 and 7'",
        bank_config={"retain_chunk_size": SPLIT_CHUNK_SIZE},
        boundary=("Rotate the Stripe API keys", "except 5 and 7"),
        items=(
            Item(
                _conversation(
                    ("user", "Can you plan this week's infrastructure maintenance?"),
                    ("assistant", _MAINTENANCE_PLAN),
                    ("user", "Go ahead with all of them except 5 and 7."),
                    ("assistant", "Understood, scheduling them now."),
                ),
                timestamp="2026-11-16T09:00:00",
            ),
        ),
        must=(
            "One fact, read alone, records that the user excluded dropping the audit_2023 table or emailing "
            "customers about the outage, naming the excluded action — not only its point number.",
        ),
        traps=(
            "A fact says the user approved dropping the audit_2023 table or emailing customers about the outage.",
        ),
    ),
    Case(
        id="terse-answers-to-numbered-questions",
        category="chunk_boundary",
        origin="#6645: '1) yes 2) discord in general' stored as a permanent authorisation and a Discord preference",
        bank_config={"retain_chunk_size": SPLIT_CHUNK_SIZE},
        boundary=("api-gateway, billing-worker and search-indexer", "discord general"),
        items=(
            Item(
                _conversation(
                    ("user", "Set up nightly release automation for the platform team."),
                    ("assistant", _THREE_QUESTIONS),
                    ("user", "1) yes, those three only 2) discord general 3) no, keep them for a year"),
                    ("assistant", "Got it, setting it up."),
                ),
                timestamp="2026-11-16T09:00:00",
            ),
        ),
        must=(
            "One fact, read alone, records that the user authorised deploying api-gateway, billing-worker and "
            "search-indexer to production without asking each time.",
            "One fact, read alone, records that the weekly release report goes to the #general channel on Discord.",
            "One fact, read alone, records that the user does not want production logs older than 90 days "
            "deleted and wants them kept for a year.",
        ),
        traps=(
            "A fact presents Discord as a general preference of the user rather than where the weekly release "
            "report is posted.",
            "A fact says the user authorised deploying every service, or services other than api-gateway, "
            "billing-worker and search-indexer, without sign-off.",
        ),
    ),
    Case(
        id="choice-resolved-two-messages-back",
        category="chunk_boundary",
        origin="#6645: the referent is two messages back, so the previous message alone does not resolve it",
        bank_config={"retain_chunk_size": SPLIT_CHUNK_SIZE},
        boundary=("1,250", "the cheaper one"),
        items=(
            Item(
                _conversation(
                    (
                        "user",
                        "I am choosing between two flats in Turin. Flat A on Via Po: 1,450 euros a month, fourth "
                        "floor, no lift. Flat B in San Salvario: 1,250 euros a month, ground floor with a small "
                        "garden.",
                    ),
                    ("assistant", _FLATS_DISCUSSION),
                    ("user", "Ok, decided: I'll take the cheaper one and sign the lease on Friday."),
                    ("assistant", "Great choice, good luck with the signing."),
                ),
                timestamp="2026-11-16T09:00:00",
            ),
        ),
        must=(
            "One fact, read alone, records that the user decided to rent the San Salvario flat (Flat B) — "
            "naming it, not only as 'the cheaper one'.",
            "One fact, read alone, records that the user will sign the lease for the San Salvario flat on "
            "Friday 20 November 2026.",
        ),
        traps=("A fact says the user chose the Via Po flat.",),
    ),
    Case(
        id="name-given-in-the-next-chunk",
        category="chunk_boundary",
        origin="#6645, the other direction: who 'my new manager' is arrives in the chunk after",
        bank_config={"retain_chunk_size": SPLIT_CHUNK_SIZE},
        boundary=("cancelled the team offsite", "Lucía Ortega"),
        items=(
            Item(
                _conversation(
                    ("user", _MANAGER_RANT),
                    (
                        "user",
                        "Oh, I did not even say: her name is Lucía Ortega, she joined from Glovo where she ran "
                        "the payments team.",
                    ),
                    ("assistant", "That sounds like a lot of change at once."),
                ),
                timestamp="2026-11-16T09:00:00",
            ),
        ),
        must=(
            "One fact, read alone, names Lucía Ortega as the user's new manager.",
            "One fact, read alone, records that Lucía Ortega cancelled the team offsite in Lisbon, moved "
            "standups to 8:30, or reviews every pull request herself — naming her, not only 'the new manager'.",
        ),
    ),
    Case(
        id="approval-with-changes",
        category="chunk_boundary",
        origin="#6645: an approval that amends the proposal it answers",
        bank_config={"retain_chunk_size": SPLIT_CHUNK_SIZE},
        boundary=("Chargebee to Stripe Billing", "cap it at 30"),
        items=(
            Item(
                _conversation(
                    ("user", "Send me the billing migration proposal."),
                    ("assistant", _MIGRATION_PROPOSAL),
                    ("user", "Approved, but cap it at 30 and skip the parallel run."),
                    ("assistant", "Noted, I'll tell Tomás."),
                ),
                timestamp="2026-11-16T09:00:00",
            ),
        ),
        must=(
            "One fact, read alone, records that the user approved migrating billing from Chargebee to Stripe "
            "Billing.",
            "One fact, read alone, records that the user capped the billing migration budget at 30k euros "
            "(not a bare 'cap of 30').",
            "One fact, read alone, records that the user dropped the two-week parallel run of the billing "
            "migration.",
        ),
        traps=("A fact says the approved budget is 40k euros.",),
    ),
)


def _one_chunk(case: Case) -> Case:
    return replace(
        case,
        id=f"{case.id}-one-chunk",
        category="chunk_control",
        origin=f"control for {case.id}: the same input in one chunk",
        bank_config={**case.bank_config, "retain_chunk_size": WHOLE_CHUNK_SIZE},
        boundary_split=False,
    )


CHUNK_BOUNDARY_CASES: tuple[Case, ...] = _CASES + tuple(_one_chunk(case) for case in _CASES)

