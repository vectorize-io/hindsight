"""Deterministic filler memories that make the corpus big enough to discriminate.

At 28 rows every query returns the whole bank, so nothing is ever truncated and
"was the gold fact retrieved" is trivially yes. Only rank moves. A realistic bank
is large enough that the recall token budget cuts the tail off, which is where a
badly worded query actually loses evidence rather than just reordering it.

The labelled rows stay hand-authored — they carry the gold ids. Everything else
is ballast: it needs to be plausible, topically varied, and **irrelevant to every
question**. A filler row that happens to be about billing providers silently
becomes an unlabelled distractor and corrupts the gold, so `check_contamination`
verifies that none of them out-ranks the labelled evidence.

Generated from seeded templates rather than an LLM, for three reasons: the
retrieval tier stays runnable with no API key, the text is identical on every
machine, and a regenerated corpus would change the embeddings and make runs
incomparable. The cost is that templated prose is more uniform than real memories
— see the note in README.
"""

from __future__ import annotations

import random

# Deliberately disjoint from every question topic (releases, outages, subsystem
# ownership, answer-style preferences, billing). If you add a question, check its
# subject does not appear here.
_DOMAINS: dict[str, dict[str, list[str]]] = {
    "botany": {
        "subject": ["The greenhouse fern", "The rooftop lavender", "The windowsill orchid", "The courtyard maple"],
        "predicate": [
            "was repotted into a wider clay planter",
            "needed shade cloth through the driest weeks",
            "dropped leaves after the cold snap",
            "was propagated from a single cutting",
        ],
    },
    "cooking": {
        "subject": ["The sourdough starter", "The braised shoulder", "The stock pot", "The pickling brine"],
        "predicate": [
            "was left to rest overnight before shaping",
            "reduced by roughly half over a low flame",
            "took on more salt than the recipe called for",
            "was strained twice through muslin",
        ],
    },
    "cycling": {
        "subject": ["The lakeside route", "The gravel loop", "The morning commute", "The hill repeat session"],
        "predicate": [
            "was rerouted around the closed bridge",
            "felt slower into a persistent headwind",
            "ended earlier than planned because of rain",
            "was ridden on the heavier winter tyres",
        ],
    },
    "astronomy": {
        "subject": ["The meteor shower", "The lunar eclipse", "The comet's approach", "The observing session"],
        "predicate": [
            "peaked well after midnight",
            "was mostly lost behind high cloud",
            "was easier to see away from the town lights",
            "lasted about two hours from first contact",
        ],
    },
    "carpentry": {
        "subject": ["The walnut shelf", "The workbench top", "The bookcase joinery", "The oak stool"],
        "predicate": [
            "was finished with three coats of oil",
            "needed the mortises pared back by hand",
            "warped slightly in the damp workshop",
            "was clamped overnight while the glue cured",
        ],
    },
    "weather": {
        "subject": ["The February storm", "The dry spell", "The overnight frost", "The coastal fog"],
        "predicate": [
            "brought down branches along the lane",
            "left the water butt completely empty",
            "held on well past sunrise",
            "cleared by the middle of the afternoon",
        ],
    },
    "music": {
        "subject": ["The string quartet", "The second movement", "The recording session", "The upright piano"],
        "predicate": [
            "was rehearsed at a slower tempo",
            "needed retuning after the move",
            "ran long and finished after ten",
            "sounded thinner in the smaller room",
        ],
    },
    "gardening": {
        "subject": ["The vegetable beds", "The compost heap", "The espaliered pear", "The seed tray"],
        "predicate": [
            "were mulched before the first frost",
            "took most of a season to break down",
            "was pruned back hard in midwinter",
            "germinated unevenly on the cold windowsill",
        ],
    },
    "photography": {
        "subject": ["The evening shoot", "The wide lens", "The darkroom print", "The long exposure"],
        "predicate": [
            "was cut short by falling light",
            "showed noticeable softness at the edges",
            "came out flatter than the negative suggested",
            "picked up movement in the foliage",
        ],
    },
    "hiking": {
        "subject": ["The ridge path", "The valley crossing", "The summit approach", "The forest track"],
        "predicate": [
            "was boggy for most of its length",
            "took an hour longer than the guidebook said",
            "was closed for nesting season",
            "had been rerouted after the landslip",
        ],
    },
}

_QUALIFIERS = [
    "",
    " according to the log kept at the time",
    " which had not happened the previous year",
    " and the note was made the same evening",
    " though nobody wrote down why",
    " for the third time that season",
]


def generate(count: int, *, seed: int = 20260908) -> list[tuple[str, str]]:
    """Return ``count`` (id, text) filler memories, identical for a given seed.

    Ids are ``ballast-0001``… so a contaminating row is nameable in a report.
    """
    rng = random.Random(seed)
    domains = sorted(_DOMAINS)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    # Bounded so a template set too small for `count` fails loudly instead of
    # spinning: dedupe keeps the corpus from collapsing onto repeated sentences,
    # which would distort retrieval far more than the ballast being synthetic.
    for _ in range(count * 50):
        if len(out) == count:
            break
        domain = domains[len(out) % len(domains)]
        spec = _DOMAINS[domain]
        text = (
            f"{rng.choice(spec['subject'])} {rng.choice(spec['predicate'])}"
            f"{rng.choice(_QUALIFIERS)} in {rng.randint(2019, 2026)}."
        )
        if text in seen:
            continue
        seen.add(text)
        out.append((f"ballast-{len(out) + 1:04d}", text))
    if len(out) != count:
        raise RuntimeError(
            f"Ballast templates yield only {len(out)} unique sentences, need {count}. Add domains or predicates."
        )
    return out
