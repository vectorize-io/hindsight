"""Consolidation must turn facts into observations that stay true as facts arrive.

An observation is the bank's belief about something, rewritten every time new
evidence comes in. Get one edit wrong and the bank does not hold a wrong
answer, it holds a wrong *memory* that recall and reflect serve as fact. Every
scenario here retains facts in waves, lets the real consolidation run after each
one, and grades what is left. Each scenario reproduces a failure this project has
actually shipped or been asked about, or a rule the prompt promises
(``hindsight_api/engine/consolidation/prompts.py``):

* restatements update one observation instead of growing siblings — #1566 (33
  observations of one event), #3249 (three split duplicates), #1759, rule 1;
* a newer fact supersedes the older belief, and the change reaches every
  observation it affects — #1151, #3248 (fnm -> nub), #3475 (retired facts stated
  as current), rules 4-5;
* recency is when a fact was *stated*, not when it was retained — #2550;
* a correction retires the false belief — #3509 (KeePass);
* one facet per observation, matched by entity, no arithmetic — #814 (the horse
  farm, ``tests/test_horse_observations.py``), rules 2, 3, 8;
* a list loses a member when that member leaves — rule 5;
* a vague reference gets resolved by a later fact — rule 6;
* distinct people and distinct numbers stay apart — rule 9, #5497 (batch size 8
  vs 4 scored as near-identical), #1725;
* observations keep the language of their facts — #3166/#3181, #4016;
* the mission wins: it can keep ephemeral data (#525) or forbid names (#4619);
  without one, chatter is dropped (decision guide);
* the same facts under different tags in a shared scope merge — #2188/#2202,
  #3564 (Traefik observed twice);
* in a large bank of near-identical services, and in a personal memory of people
  whose names come in lookalike pairs, an update still finds its twin and leaves
  the neighbours alone — #1907, #1977/#2000.

Deterministic checks run on every wave (see ``consolidation.py``): evidence
exists and matches ``proof_count`` (#4955), dates come from the evidence (#501,
#3477), no prompt metadata in the text (#2544), every durable fact is cited
(#5054), and a wave of restatements adds no observation.

Not covered here, on purpose: speaker attribution (#4572, #1680) is decided at
retain and belongs with the retain suites; failure handling, scopes mechanics
and strategy selection are deterministic and live in ``hindsight-system-tests``
(``test_20``-``test_29``).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

import pytest
from hindsight_client import Hindsight

from hindsight_system_evals.consolidation import (
    Fact,
    Scenario,
    Wave,
    noisy_personal_bank,
    noisy_service_bank,
    run_scenario,
)
from hindsight_system_evals.pages import SettleFn
from hindsight_system_evals.report import RECORDED, EvalRecord

log = logging.getLogger(__name__)


def _wave(*facts: Fact, no_new_observations: bool = False, tags: tuple[str, ...] = ()) -> Wave:
    return Wave(tuple(facts), tags=tags, no_new_observations=no_new_observations)


_SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in (
        Scenario(
            id="restatements-merge",
            origin="#3249: the same preference retained twice came back as three proof_count=1 duplicates; #1566",
            waves=(
                _wave(
                    Fact(
                        "The user values evidence-based decisions, acts quickly once the data is clear, and is "
                        "comfortable acting mid-task when the data justifies it.",
                        "2026-06-02",
                        ("user",),
                    ),
                ),
                _wave(
                    Fact("The user is comfortable acting mid-task if the data justifies it.", "2026-06-20", ("user",)),
                    Fact("The user acts quickly once the evidence is clear.", "2026-06-20", ("user",)),
                    Fact("The user values evidence-based decision-making over theory.", "2026-06-20", ("user",)),
                    no_new_observations=True,
                ),
                _wave(
                    Fact("When the numbers are in, the user decides fast.", "2026-07-08", ("user",)),
                    Fact("The user does not trust decisions made on theory alone.", "2026-07-08", ("user",)),
                    no_new_observations=True,
                ),
            ),
            must=(
                "The observations record that the user values evidence-based decisions and acts quickly once "
                "the data is clear.",
            ),
        ),
        Scenario(
            id="employer-supersedes",
            origin="#1151: many old facts support 'Alice works at Google'; one newer fact says she left",
            waves=(
                _wave(
                    Fact(
                        "Alice works at Google as a software engineer on the Maps team.",
                        "2025-03-04",
                        ("Alice", "Google"),
                    ),
                    Fact("Alice was promoted to senior engineer at Google.", "2025-06-11", ("Alice", "Google")),
                    Fact(
                        "Alice commutes to the Google office in Zurich three days a week.",
                        "2025-09-02",
                        ("Alice", "Google", "Zurich"),
                    ),
                    Fact(
                        "Alice gave an internal talk about Google's build tooling.", "2025-11-19", ("Alice", "Google")
                    ),
                    Fact("Alice has been at Google for four years.", "2026-01-15", ("Alice", "Google")),
                ),
                _wave(
                    Fact(
                        "Alice left Google in April 2026 and now works at Microsoft as a principal engineer.",
                        "2026-05-06",
                        ("Alice", "Google", "Microsoft"),
                    ),
                ),
            ),
            must=("An observation states that Alice now works at Microsoft (as a principal engineer).",),
            traps=(
                "Some observation presents Alice as CURRENTLY employed at Google, with nothing in that same "
                "observation saying she left. A past-tense record of her Google years is fine.",
            ),
        ),
        Scenario(
            id="retired-conventions",
            origin="#3248 (fnm -> nub, npx -> bun global, corepack removed), #3475 (autocommit=false reverted)",
            waves=(
                _wave(
                    Fact("The project uses fnm to manage Node.js versions.", "2026-07-16", ("fnm", "Node.js")),
                    Fact("The CLI is installed with `npx -y @acme/cli`.", "2026-07-16", ("@acme/cli", "npx")),
                    Fact("Corepack is enabled to pin the pnpm version.", "2026-07-16", ("Corepack", "pnpm")),
                    Fact(
                        "The database session is opened with autocommit disabled (autocommit=false).",
                        "2026-07-16",
                        ("autocommit",),
                    ),
                ),
                _wave(
                    Fact(
                        "The project switched from fnm to nub for Node.js version management.",
                        "2026-07-20",
                        ("fnm", "nub", "Node.js"),
                    ),
                    Fact(
                        "The CLI is now installed as a bun global (`bun add -g @acme/cli`) instead of via npx.",
                        "2026-07-20",
                        ("@acme/cli", "bun", "npx"),
                    ),
                    Fact("Corepack was removed from the project setup.", "2026-07-20", ("Corepack",)),
                    Fact(
                        "The autocommit=false setting was removed; the database session uses the driver's "
                        "default autocommit again.",
                        "2026-07-17",
                        ("autocommit",),
                    ),
                ),
            ),
            must=(
                "The observations say Node.js versions are managed with nub, and that the CLI is installed as a "
                "bun global.",
            ),
            traps=(
                "Some observation says the project currently manages Node.js versions with fnm, without saying "
                "it switched away from fnm.",
                "Some observation says the CLI is currently installed with npx, without saying that changed.",
                "Some observation says Corepack is currently enabled, without saying it was removed.",
                "Some observation says autocommit is currently disabled, without saying that setting was removed.",
            ),
        ),
        Scenario(
            id="recency-is-when-stated",
            origin="#2550: an older session backfilled after a newer one inverted 'used to like X, now Y'",
            waves=(
                _wave(
                    Fact("The user said Svelte is their favorite frontend framework.", "2026-07-10", ("user", "Svelte"))
                ),
                _wave(
                    Fact("The user said React is their favorite frontend framework.", "2026-07-05", ("user", "React"))
                ),
            ),
            must=("The observations convey that Svelte is the user's current favorite frontend framework.",),
            traps=(
                "Some observation presents React as the user's current favorite, or says the user moved from "
                "Svelte to React.",
            ),
        ),
        Scenario(
            id="correction-retires-false-belief",
            origin="#3509: five facts said Gramps works via KeePass; the user later confirmed KeePass is not installed",
            waves=(
                _wave(
                    Fact(
                        "Gramps reads its sync passwords from the KeePass database.",
                        "2026-08-01",
                        ("Gramps", "KeePass"),
                    ),
                    Fact(
                        "The Gramps nightly sync unlocks KeePass with a key file.", "2026-08-03", ("Gramps", "KeePass")
                    ),
                    Fact("The KeePass integration already works for Gramps.", "2026-08-05", ("Gramps", "KeePass")),
                ),
                _wave(
                    Fact(
                        "The user confirmed KeePass is not installed on the machine; the Gramps credentials are "
                        "stored in Bitwarden.",
                        "2026-08-12",
                        ("Gramps", "KeePass", "Bitwarden"),
                    ),
                ),
            ),
            must=("An observation states that the Gramps credentials are stored in Bitwarden.",),
            traps=(
                "Some observation presents Gramps as CURRENTLY reading its passwords from KeePass, or the KeePass "
                "integration as working, with nothing in that observation saying this was wrong.",
            ),
        ),
        Scenario(
            id="horse-farm",
            origin="#814 and tests/test_horse_observations.py: counts merged with names, naming bumped the count",
            waves=tuple(
                _wave(Fact(text, day, ents))
                for text, day, ents in (
                    ("The user has a farm.", "2026-01-05", ("user",)),
                    ("The user has 2 horses.", "2026-01-12", ("user",)),
                    ("The user has a horse named Daisy.", "2026-01-19", ("Daisy",)),
                    ("The user has a horse named Buttercup.", "2026-01-26", ("Buttercup",)),
                    ("The user sold Buttercup.", "2026-02-02", ("Buttercup",)),
                    ("The user now has 1 horse.", "2026-02-09", ("user",)),
                    ("The user has 5 horses on their farm.", "2026-02-16", ("user",)),
                    ("The user has a horse named Midnight.", "2026-02-23", ("Midnight",)),
                    ("The user's horses include Midnight and Shadow.", "2026-03-02", ("Midnight", "Shadow")),
                    ("The user's horses include Shadow and Twister.", "2026-03-09", ("Shadow", "Twister")),
                    ("The user reported that Shadow has died.", "2026-03-16", ("Shadow",)),
                )
            ),
            must=(
                "The observations record that the user sold Buttercup and that Shadow died.",
                "The observations give the user's most recently stated horse count as 5 (a note that Shadow died "
                "afterwards is fine; recomputing the number is not).",
            ),
            traps=(
                "Some observation states a horse count the user never stated (for example 3, 4 or 6), or "
                "decrements/increments a count on its own.",
                "Some observation says Daisy was sold or died.",
                "Some observation presents Buttercup as still owned by the user.",
                "Some observation presents Shadow as still alive.",
            ),
        ),
        Scenario(
            id="no-arithmetic",
            origin="prompt rule 8 ('I have 2 dogs' + 'a dog named Rex' must not become 3)",
            waves=(
                _wave(Fact("The user has 2 dogs.", "2026-04-01", ("user",))),
                _wave(Fact("The user has a dog named Rex.", "2026-04-08", ("Rex",))),
                _wave(Fact("The user adopted a puppy named Miso.", "2026-04-15", ("Miso",))),
            ),
            must=("The observations record a dog named Rex and a puppy named Miso.",),
            traps=("Some observation states the user has 3 dogs, 4 dogs, or any dog count other than 2.",),
        ),
        Scenario(
            id="cascade-to-group",
            origin="prompt rule 5 (a member leaving updates its own observation AND the list that contains it)",
            waves=(
                _wave(
                    Fact(
                        "The payments on-call rotation is Ana, Ben, Chen and Dmitri.",
                        "2026-08-20",
                        ("payments on-call rotation", "Ana", "Ben", "Chen", "Dmitri"),
                    ),
                    Fact(
                        "Ben is the payments on-call lead for September 2026.",
                        "2026-08-20",
                        ("Ben", "payments on-call rotation"),
                    ),
                ),
                _wave(
                    Fact(
                        "Ben left the payments on-call rotation on 3 September 2026 after moving to the data team.",
                        "2026-09-03",
                        ("Ben", "payments on-call rotation", "data team"),
                    ),
                ),
            ),
            must=(
                "The observations record that Ben left the payments on-call rotation, and still list Ana, Chen and "
                "Dmitri as members.",
            ),
            traps=(
                "Some observation presents Ben as a CURRENT member or the current lead of the payments on-call "
                "rotation, with nothing in that observation saying he left.",
                "Some observation drops Ana, Chen or Dmitri from the rotation.",
            ),
        ),
        Scenario(
            id="resolve-reference",
            origin="prompt rule 6 ('home country' -> 'Sweden')",
            waves=(
                _wave(
                    Fact("Lena plans to move back to her home country next spring.", "2026-05-02", ("Lena",)),
                    Fact("Lena's partner Tomas is Norwegian.", "2026-05-02", ("Lena", "Tomas")),
                ),
                _wave(Fact("Lena grew up in Gothenburg, Sweden.", "2026-05-20", ("Lena", "Gothenburg", "Sweden"))),
            ),
            must=("An observation states that Lena plans to move back to Sweden.",),
            traps=("Some observation says Lena is moving to Norway, or that Sweden is Tomas's home country.",),
        ),
        Scenario(
            id="distinct-entities-stay-apart",
            origin="prompt rule 9; #5497 ('batch size 8' vs 'batch size 4' look near-identical); #1725",
            waves=(
                _wave(
                    Fact("Marco is allergic to peanuts.", "2026-03-01", ("Marco",)),
                    Fact("Marta is allergic to shellfish.", "2026-03-01", ("Marta",)),
                    Fact("Marco is vegetarian.", "2026-03-01", ("Marco",)),
                    Fact("The ingest worker runs with a batch size of 8.", "2026-03-01", ("ingest worker",)),
                    Fact("The export worker runs with a batch size of 4.", "2026-03-01", ("export worker",)),
                ),
                _wave(Fact("The ingest worker's batch size was raised to 16.", "2026-03-15", ("ingest worker",))),
            ),
            must=(
                "The observations say the ingest worker's batch size is now 16 and the export worker's is 4.",
                "The observations say Marco is allergic to peanuts and Marta to shellfish.",
            ),
            traps=(
                "Some observation gives Marta a peanut allergy or Marco a shellfish allergy.",
                "Some observation says Marta is vegetarian.",
                "Some single observation states both Marco's and Marta's allergies.",
                "Some observation gives the export worker a batch size other than 4.",
                "Some observation presents 8 as the ingest worker's current batch size.",
            ),
        ),
        Scenario(
            id="language-chinese",
            origin="#3166/#3181: Chinese facts produced English observations ('User lives in Shenzhen.')",
            waves=(
                _wave(
                    Fact("张三住在深圳。", "2026-06-01", ("张三", "深圳")),
                    Fact("张三养了一只叫豆豆的猫。", "2026-06-01", ("张三", "豆豆")),
                    Fact("张三周末常带豆豆去深圳湾公园散步。", "2026-06-01", ("张三", "豆豆", "深圳湾公园")),
                    Fact("张三最近在准备晋升答辩。", "2026-06-01", ("张三",)),
                ),
            ),
            must=("The observations record that 张三 lives in 深圳 and has a cat named 豆豆.",),
            traps=("Some observation is written in a language other than Chinese (names aside).",),
        ),
        Scenario(
            id="language-mixed-update",
            origin="#3181: an English observation updated by a Chinese fact must be rewritten, not patched",
            waves=(
                _wave(Fact("Zhang Wei works at Tencent in Shenzhen.", "2026-05-01", ("Zhang Wei", "Tencent"))),
                _wave(
                    Fact("张伟上个月从腾讯离职，加入了字节跳动。", "2026-06-03", ("Zhang Wei", "腾讯", "字节跳动")),
                    Fact("张伟现在在字节跳动负责推荐系统。", "2026-06-03", ("Zhang Wei", "字节跳动")),
                ),
            ),
            must=("An observation records that Zhang Wei (张伟) now works at ByteDance (字节跳动).",),
            traps=(
                "Some single observation mixes English and Chinese sentences or clauses (an English sentence with "
                "Chinese details bolted on, or the reverse). Proper names alone do not count.",
                "Some observation presents Zhang Wei as currently working at Tencent.",
            ),
        ),
        Scenario(
            id="default-drops-chatter",
            origin="decision guide: purely ephemeral facts are omitted unless the mission targets them",
            waves=(
                _wave(
                    Fact(
                        "The user's company is migrating billing from Braintree to Stripe by Q4 2026.",
                        "2026-07-01",
                        ("Braintree", "Stripe"),
                    ),
                    Fact(
                        "The user said they were grabbing a coffee and would be back in five minutes.",
                        "2026-07-01",
                        durable=False,
                    ),
                    Fact(
                        "The user's screen currently shows the Grafana home dashboard.",
                        "2026-07-01",
                        ("Grafana",),
                        durable=False,
                    ),
                    Fact("The user said thanks for the help.", "2026-07-01", durable=False),
                ),
            ),
            must=("An observation records the billing migration from Braintree to Stripe by Q4 2026.",),
            traps=("Some observation records the coffee break, what is on the user's screen, or the thank-you.",),
        ),
        Scenario(
            id="mission-keeps-ephemeral",
            origin="#525: the built-in 'skip ephemeral state' rule discarded what the mission asked for",
            mission="Track every shell command the agent ran, with the time it ran and its exit status.",
            waves=(
                _wave(
                    Fact(
                        "At 14:02 the agent ran `pytest tests/` and it exited with status 1.", "2026-09-01", ("pytest",)
                    ),
                    Fact(
                        "At 14:05 the agent ran `ruff check --fix .` and it exited with status 0.",
                        "2026-09-01",
                        ("ruff",),
                    ),
                    Fact(
                        "At 14:07 the agent re-ran `pytest tests/` and it exited with status 0.",
                        "2026-09-01",
                        ("pytest",),
                    ),
                ),
            ),
            must=(
                "The observations record all three command runs: pytest at 14:02 exiting 1, ruff at 14:05 exiting "
                "0, and pytest again at 14:07 exiting 0.",
            ),
            traps=("Some observation gives a wrong exit status or time for one of the runs.",),
        ),
        Scenario(
            id="mission-forbids-names",
            origin="#4619 / system test 29: a company-wide scope told to name no company",
            mission="Record only generalized, industry-level trends. Name no specific company or person.",
            waves=(
                _wave(
                    Fact(
                        "Dana met the founders of Northwind, who are moving off frontier models to fine-tuned "
                        "open-weight models.",
                        "2026-08-10",
                        ("Dana", "Northwind"),
                    ),
                    Fact(
                        "Northwind cut its inference spend by 40% after switching to self-hosted models.",
                        "2026-08-10",
                        ("Northwind",),
                    ),
                    Fact(
                        "Helios Labs is also replacing hosted GPT-class APIs with self-hosted open-weight models.",
                        "2026-08-11",
                        ("Helios Labs",),
                    ),
                ),
            ),
            must=(
                "An observation captures the trend of companies moving from hosted frontier models to "
                "self-hosted or open-weight models.",
            ),
            traps=("Some observation names Northwind, Helios Labs, Dana, or any other specific company or person.",),
        ),
        Scenario(
            id="shared-scope-merges-across-tags",
            origin="#2188/#2202 (211 per-session duplicates), #3564 (Traefik facts observed once per harness tag)",
            observation_scopes="shared",
            waves=(
                _wave(
                    Fact(
                        "The project uses Traefik as its reverse proxy, configured through container labels.",
                        "2026-09-01",
                        ("Traefik",),
                    ),
                    Fact("TLS certificate renewal is automatic via ACME.", "2026-09-01", ("ACME",)),
                    tags=("harness:claude-code",),
                ),
                _wave(
                    Fact(
                        "Traefik is the project's reverse proxy; it is configured with container labels.",
                        "2026-09-04",
                        ("Traefik",),
                    ),
                    Fact("Certificates renew automatically through ACME.", "2026-09-04", ("ACME",)),
                    tags=("harness:codex",),
                    no_new_observations=True,
                ),
            ),
            must=("The observations record Traefik as the reverse proxy and automatic ACME certificate renewal.",),
            traps=("Two or more observations state that Traefik is the reverse proxy.",),
        ),
    )
}

#: The failures that shipped and had to be fixed. Everything else runs with ``--full``.
MINIMUM_ACCEPTANCE = (
    "restatements-merge",
    "employer-supersedes",
    "retired-conventions",
    "horse-farm",
    "language-chinese",
)


async def _run(client: Hindsight, bank_id: str, settled: SettleFn, scenario: Scenario) -> None:
    outcome = await run_scenario(client, bank_id, settled, scenario)
    log.info("%s", outcome.report())
    RECORDED.append(
        EvalRecord(
            kind="consolidation",
            question_id=scenario.id,
            category=scenario.origin.split(":")[0],
            correct=not (outcome.missed or outcome.problems),
            hit_trap=bool(outcome.traps_hit),
            bank_id=bank_id,
            reason="; ".join(outcome.traps_hit + outcome.missed + outcome.problems)[:2000],
            sizes=[len(s) for s in outcome.snapshots],
        )
    )
    if cost_path := os.getenv("HINDSIGHT_EVAL_CONSOLIDATION_COST"):
        # One line per scenario, so a run can be compared on cost AND outcome — a cheaper
        # configuration is only cheaper at the same quality.
        with open(cost_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "scenario": scenario.id,
                        "passed": outcome.passed,
                        **dataclasses.asdict(outcome.cost),
                    }
                )
                + "\n"
            )
    assert outcome.passed, outcome.report()


@pytest.mark.parametrize("scenario_id", MINIMUM_ACCEPTANCE)
async def test_consolidation_minimum_acceptance(
    client: Hindsight, bank_id: str, settled: SettleFn, scenario_id: str
) -> None:
    await _run(client, bank_id, settled, _SCENARIOS[scenario_id])


@pytest.mark.full
@pytest.mark.parametrize("scenario_id", [s for s in _SCENARIOS if s not in MINIMUM_ACCEPTANCE])
async def test_consolidation_full(client: Hindsight, bank_id: str, settled: SettleFn, scenario_id: str) -> None:
    await _run(client, bank_id, settled, _SCENARIOS[scenario_id])


@pytest.mark.full
async def test_consolidation_noisy_bank(client: Hindsight, bank_id: str, settled: SettleFn) -> None:
    """The same rules at scale: 30 near-identical services and chatter (see ``noisy_service_bank``)."""
    await _run(client, bank_id, settled, noisy_service_bank())


@pytest.mark.full
async def test_consolidation_noisy_personal_bank(client: Hindsight, bank_id: str, settled: SettleFn) -> None:
    """The same at scale on a different shape: 20 people whose names come in lookalike pairs."""
    await _run(client, bank_id, settled, noisy_personal_bank())


@pytest.mark.full
async def test_consolidation_noisy_bank_with_mission(client: Hindsight, bank_id: str, settled: SettleFn) -> None:
    """The service bank again under a bank's own mission, the common setup (#525, #4619 set one):
    the mission's wording must not change what lands, and chatter it does not ask for stays out."""
    scenario = noisy_service_bank()
    await _run(
        client,
        bank_id,
        settled,
        dataclasses.replace(
            scenario,
            id="noisy-service-catalogue-mission",
            mission=(
                "Keep a catalogue of the platform's services: which team owns each one, what language it is "
                "written in, the port it listens on and where its pages go. Leave out small talk."
            ),
            # The chatter's own words join the focus so the judge sees any observation it left.
            focus=(*scenario.focus, "coffee", "wifi", "thanks", "Grafana", "stand-up"),
            traps=(*scenario.traps, "Some observation records small talk: a coffee break, the wifi, a thank-you."),
        ),
    )
