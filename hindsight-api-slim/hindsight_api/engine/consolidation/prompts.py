"""Prompts for the consolidation engine."""

from hindsight_api.engine.prompt_utils import (
    default_language_section,
    escape_for_prompt,
    output_language_directive,
)

# Default mission — tells the consolidator to track anything worth remembering.
# Banks override this via `observations_mission` to scope what gets retained.
# Consolidation behavior (what an observation is, how a fact changes one) lives in
# the rules below, not in the mission — but the mission takes priority over those
# rules when the two conflict.
_DEFAULT_MISSION = (
    "Track anything notable in the new facts — names, numbers, dates, places, "
    "events, decisions, claims, relationships, and recurring patterns."
)

# Consolidation's wording of the preserve-the-source-language rule; the selection
# between it and an explicit output language belongs to default_language_section(),
# which documents the invariant. Without this rule the whole prompt is English and
# multilingual models drift: Chinese source facts intermittently produce English
# observations. Retain's fact extraction carries the equivalent rule (see
# _DEFAULT_LANGUAGE_RULE there), making "preserve the source language" the
# pipeline-wide default.
_DEFAULT_LANGUAGE_RULE = """## LANGUAGE

Write every observation in the language of its own source facts — never translate them. Per observation, not per batch: when one merges facts of several languages, the majority wins. Proper nouns, identifiers, and units stay verbatim.

When an existing observation is written in a different language from the new facts updating it, do NOT edit its wording in place — that is what produces an English sentence with a Chinese detail bolted on. Discard the old phrasing and compose the merged observation from scratch in the new facts' language."""

_OBSERVATION_FIELDS = """- `id`: its identifier (`O1`, `O2`, …) — copy it into `observation_id`
- `text`: the observation content. A bracket after it, `(YYYY-MM-DD HH:MM: …)`, is an earlier state that the text before it replaced — not the current value. Keep the brackets as they are when you change the observation.
- `proof_count`: how many source facts this observation has already merged
- `occurred_start` / `occurred_end`: the span of the events behind the observation — earliest start and latest end across its source facts
- `mentioned_at`: the latest of the `mentioned_at` values of its source facts — the most recent point at which this observation was stated
- `source_memories`: the supporting facts behind this observation. May be partial or absent for large observations — the count above remains the true total. Each entry carries the same `text` and temporal fields as a new fact."""

# Per-batch data section for the cached split path — the stable format
# explanation above is omitted here (it lives in the cached prefix); only the
# variable facts/observations remain. Placeholders substituted at call time.
_SPLIT_INPUT_SECTION = """## INPUT

### New facts

{facts_text}

### Existing observations

{observations_text}"""


def build_mission_section(observations_mission: str | None) -> str:
    """The MISSION section, heading included, as the user message writes it.

    Shared with the prompt preview, which reports this section as the block the
    ``observations_mission`` setting produces. Rebuilding the heading there left two
    blocks both called "Mission" — the setting's, and the built-in gap that this
    ``## MISSION`` heading names too.
    """
    return f"## MISSION\n\n{escape_for_prompt(observations_mission or _DEFAULT_MISSION)}"


def build_consolidation_input(
    facts_text: str,
    observations_text: str,
    observations_mission: str | None = None,
    observation_capacity_note: str | None = None,
) -> str:
    """Per-batch user message: MISSION + INPUT data + any capacity constraint.

    The MISSION lives here (not in the cached system prefix) so the prefix stays
    bank-agnostic and one CachedContent serves every bank. The capacity note also
    lives here since it varies as observation slots fill.
    """
    mission_section = f"{build_mission_section(observations_mission)}\n\n"
    capacity_section = ""
    if observation_capacity_note:
        capacity_section = f"## CAPACITY CONSTRAINT\n\n{escape_for_prompt(observation_capacity_note)}\n\n"
    # _SPLIT_INPUT_SECTION omits the stable observation-format explanation (now in
    # the cached system prefix) — only the variable facts/observations remain.
    template = mission_section + capacity_section + _SPLIT_INPUT_SECTION
    return template.format(facts_text=facts_text, observations_text=observations_text)


# The consolidation prompt, written for small models (gemma-4). It replaced a prompt that asked
# for lists of creates, updates and deletes directly: gemma-4 scored 29/40 on the consolidation
# evals with that one and 36-38/40 with this one, with no code checks and no second call. Each answer is one entry per new fact, and inside it each
# of the fact's closest observations, plus any other it repeats, changes or ends, with a fixed
# verdict (`effect`). It first judged every observation about the subject: 99% of the entries
# were "unrelated", 2x the output tokens of the actions format on a large model. Listing only the
# touched ones lost the traps the closest ones are (a named dog bumped the count; 37 → 30/40), so
# the closest are still judged. Ids are short (F1, O1): a uuid is ~20 tokens. Asked for creates
# and updates directly, gemma-4 had closed the updates array after the
# first stale observation, merged facets into one sentence, and changed a lookalike's value. The
# code turns the entries into actions (consolidator._FactBatchResponse.to_actions) and does what
# gemma-4 got wrong both ways: it compares the times itself (`(stated after this fact)`).
_RULES = """## WHAT AN OBSERVATION IS

One short sentence about ONE subject and ONE facet of it: its owner, its port, its job, its city, a count, one named member. "Paolo is the user's cousin" and "Paolo works as a teacher" are two observations — never join facets into one sentence, never add a new facet to an existing observation.

## HOW TO WORK

Go through the new facts one at a time, in the order given, and write one entry in `facts` for each. Finish a fact completely before the next one.

For each fact:

1. `fact_id`: the fact's id, e.g. `F1`.
2. `about`: who or what the fact is about, copied exactly as the fact spells it ("ledger-sink", "Paola", "the on-call rotation").
3. `kind` — will this still be worth knowing next month?
   - `"momentary"`: no — it is only true or relevant for minutes. This is about TIME only, never about who or what the fact names. Greetings, thanks, small talk, a break, what someone is doing or looking at right now, what is on a screen. Nothing is recorded. (When the MISSION asks for such data, it is "lasting".) Never call a fact momentary because of the MISSION: a fact the MISSION wants recorded without some detail is "lasting" — record it without that detail. MISSION "name no company" + fact "Acme moved its API to Rust" → "lasting", new: "Companies are moving their APIs to Rust."
   - `"lasting"`: yes — names, numbers, owners, ports, channels, places, dates, jobs, decisions, preferences, rules, events, corrections.
4. `existing`: first EVERY one of the fact's `closest existing observations`, each judged — `"unrelated"` included. Then scan ALL the other observations and add only the ones this fact repeats, changes or ends; leave the other unrelated ones out. For each entry:
   - `observation_id`: its id, e.g. `O3`.
   - `observation_about`: who or what the observation's text is about, copied exactly as THAT TEXT spells it (not as the fact spells it). Compare it with `about`.
   - `effect` — what this fact does to that observation:
     - `"unrelated"`: a different subject (a name a letter or a word apart — "ledger-sync" / "ledger-sink", "Paolo" / "Paola" — is a DIFFERENT subject), or the same subject but another facet the fact does not touch. Nothing changes.
     - `"repeats"`: the observation already says what the fact says.
     - `"changes"`: the fact gives a new value for exactly the facet the observation states (a new owner for the owner observation, a new port for the port observation — never a language fact for an owner observation), or fills in what the observation left vague ("her home country" → "Sweden"). A count is changed only by a fact that states the new number itself: one member arriving, leaving or dying ("adopted a puppy", "Shadow died") is NOT a new count — the count observation is "unrelated" and keeps its number. Never calculate.
     - `"ends"`: the fact makes the observation no longer true as written — an item was sold, a tool retired, the observation is shown to be wrong, a member left a group, or someone left an employer: then EVERYTHING that was true only because of it ends too — their role there, their team there, their tenure there, their commute to its office.
   - `observation_text`: for repeats, changes or ends, the observation's text copied exactly. Leave it out for unrelated.
   - `text`: the observation after this fact.
     - unrelated or repeats → empty: `""`.
     - the observation is marked `(stated after this fact)` or `"newer_than_input_facts": true` → the fact is older news: effect `"changes"` with `text` empty. The observation stays the current value, and the fact is added to it as its earlier state for you.
     - changes → the new value; you may keep the old one as history ("The billing-gateway service is owned by team Nimbus; earlier by team Atlas."). A vague reference → the same sentence with the value filled in ("Lena plans to move back to Sweden next spring.").
     - ends, for a group or list → it goes on, present tense, naming the members who REMAIN, then who left ("The rotation is Ana, Ben and Dev; Cleo left it on 3 May 2025.").
     - ends, anything else → the past tense with the date, keeping what it said ("Alice commuted to the Google office in Zurich until she left Google in April 2026."). Shown to be wrong → say it was wrong and what is true.
5. `adds`: does the fact say anything that none of the observations you marked repeats, changes or ends already says? `"yes"` or `"no"`. A fact that only repeats an observation is "no".
6. `new`: when `adds` is "yes", one sentence per facet the fact states that no existing observation already says. Write what the fact says, as stated:
   - Never calculate. "The user has 2 dogs" + "the user has a dog named Rex" → new "The user has a dog named Rex"; the count observation is "unrelated" and stays 2.
   - Empty when the fact changes an observation marked `(stated after this fact)`: its content is that observation's earlier state.
7. `same_as_fact`: when this fact says the same as an EARLIER fact of this batch that already has a `new` observation, that fact's id — and leave `new` empty. Otherwise null.

A lasting fact must end up somewhere: an existing observation it repeats, changes or ends, or `new`. Otherwise it is lost.

`deletes` (rare): only when two EXISTING observations say exactly the same thing — delete one by its `observation_id`, with `merged_into` set to the other: `"deletes": [{{"observation_id": "O7", "merged_into": "O2"}}]`. Never delete because something changed or ended."""

# Facts are handed over in processing order, which is not the order things happened: imports and
# backfills deliver old conversations late. Small models read the batch as the latest news and let
# an older rule overwrite its newer amendment (#5938), and they compare dates badly, so the code
# makes the comparison and marks it; this section says what a mark means.
_TIME_RULE = """## TIME

Facts are NOT given in the order they happened: old conversations are imported late. What counts is when something was stated: `mentioned_at` (and `occurred_start` / `occurred_end` for when it happened). The most recently stated value is the current one — not the fact processed last, and not the one worded "now", "new", "amended" or "was updated": an older fact describing a change describes an EARLIER state.

The comparison is done for you: in a fact's `closest existing observations`, the ones stated after that fact are marked `(stated after this fact)`, and an observation stated after every new fact carries `"newer_than_input_facts": true` — both mean the fact is older news for it. Among a fact's closest observations, unmarked means the fact is the more recent.

The new facts are listed oldest first. When two new facts disagree about the same thing, the LATER one in the list is current: the earlier one's `new` sentence says it as the earlier state ("Before 21 October 2026, small overages were not waived."), never as the current rule. An older fact is still recorded, as the earlier state — never dropped."""

_OUTPUT_SECTION = """## OUTPUT FORMAT

Return one JSON object: `{{"facts": [...one entry per new fact, in order...], "deletes": []}}`.

### Example

New facts:
  [F1] ledger-sink now listens on port 8444. (mentioned_at=2025-05-10)
    closest existing observations: O1
  [F2] Thanks, that's really helpful! (mentioned_at=2025-05-10)
  [F3] Cleo left the on-call rotation on 3 May 2025. (mentioned_at=2025-05-10)
    closest existing observations: O2, O6
  [F4] The user has a dog named Rex. (mentioned_at=2025-05-10)
    closest existing observations: O4
  [F5] The user said their favorite framework is now React. (mentioned_at=2025-05-10)
    closest existing observations: O5 (stated after this fact)
  [F6] ledger-sink is listening on port 8444. (mentioned_at=2025-05-11)

Existing observations:
  {{"id": "O1", "text": "billing-sink listens on port 8336.", "mentioned_at": "2025-04-01"}}
  {{"id": "O2", "text": "The on-call rotation is Ana, Ben, Cleo and Dev.", "mentioned_at": "2025-04-01"}}
  {{"id": "O3", "text": "Cleo is on call on Mondays.", "mentioned_at": "2025-04-01"}}
  {{"id": "O4", "text": "The user has 2 dogs.", "mentioned_at": "2025-04-01"}}
  {{"id": "O5", "text": "The user's favorite framework is Svelte.", "mentioned_at": "2025-06-01", "newer_than_input_facts": true}}
  {{"id": "O6", "text": "Cleo lives in Lisbon.", "mentioned_at": "2025-04-01"}}

Output:

{{"facts": [
  {{"fact_id": "F1", "about": "ledger-sink", "kind": "lasting", "existing": [
      {{"observation_id": "O1", "observation_about": "billing-sink", "effect": "unrelated", "text": ""}}],
    "adds": "yes", "new": ["ledger-sink listens on port 8444."], "same_as_fact": null}},
  {{"fact_id": "F2", "about": "the conversation", "kind": "momentary", "existing": [], "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "F3", "about": "Cleo", "kind": "lasting", "existing": [
      {{"observation_id": "O2", "observation_about": "the on-call rotation", "effect": "ends", "observation_text": "The on-call rotation is Ana, Ben, Cleo and Dev.", "text": "The on-call rotation is Ana, Ben and Dev; Cleo left it on 3 May 2025."}},
      {{"observation_id": "O6", "observation_about": "Cleo", "effect": "unrelated", "text": ""}},
      {{"observation_id": "O3", "observation_about": "Cleo", "effect": "ends", "observation_text": "Cleo is on call on Mondays.", "text": "Cleo was on call on Mondays until she left the rotation on 3 May 2025."}}],
    "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "F4", "about": "the user's dog Rex", "kind": "lasting", "existing": [
      {{"observation_id": "O4", "observation_about": "the user's dogs", "effect": "unrelated", "text": ""}}],
    "adds": "yes", "new": ["The user has a dog named Rex."], "same_as_fact": null}},
  {{"fact_id": "F5", "about": "the user's favorite framework", "kind": "lasting", "existing": [
      {{"observation_id": "O5", "observation_about": "the user's favorite framework", "effect": "changes", "observation_text": "The user's favorite framework is Svelte.", "text": ""}}],
    "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "F6", "about": "ledger-sink", "kind": "lasting", "existing": [], "adds": "no", "new": [], "same_as_fact": "F1"}}
],
"deletes": []}}

Why: O1, F1's closest, is about billing-sink, not ledger-sink: unrelated. The rotation's membership (O2) and Cleo's Mondays (O3) end; O3 was not among the closest but mentions Cleo, so it is found by scanning. Cleo's city (O6), a closest one, is another facet: unrelated. The dog count (O4) is not touched by a named dog: unrelated. Svelte was stated after React, so F5 changes O5 with no text: Svelte stays the favorite and F5 is added as its earlier state. F6 has no closest observations and repeats F1.

### Text rules

- Clean prose. Never copy metadata into the text (`mentioned_at=...`, uuids, "Involving:" labels).
- Do NOT include `tags` — handled automatically."""


_INPUT_FORMAT = f"""## INPUT FORMAT

Each request provides new facts and existing observations. Every temporal field is optional and is omitted when unknown.

### New facts

Oldest first, one per line, formatted as `[id] fact text (temporal fields)`:
- `[id]`: the fact's identifier (`F1`, `F2`, …) — copy it into `fact_id`
- `occurred_start` / `occurred_end`: when the described event happened
- `mentioned_at`: when the fact was stated — its recency, NOT when it was added to memory
- `closest existing observations` (next line, when any): the ids of the existing observations most similar to this fact, best first, those stated after the fact marked `(stated after this fact)`. Ranked by similarity only: a lookalike name ranks close too.

### Existing observations

A JSON array pooled from recalls across the new facts. Each entry has:
{_OBSERVATION_FIELDS}"""


def build_consolidation_system_prompt(llm_output_language: str | None = None) -> str:
    """Bank-agnostic, cacheable system instruction for batch consolidation.

    The bank's MISSION, the capacity note and the batch itself ride in the user message
    (:func:`build_consolidation_input`), so this prefix is identical for every bank and one
    cached context serves them all. ``llm_output_language`` picks between keeping each
    observation in its source facts' language (unset) and one configured language.
    """
    language_section = default_language_section(_DEFAULT_LANGUAGE_RULE, llm_output_language)
    template = (
        "You keep a long-term memory. New facts arrive; you record what they say as observations — short "
        "sentences, one facet each — and keep existing observations true.\n\n"
        "If anything in the MISSION conflicts with the rules below, the MISSION takes priority.\n\n"
        f"{language_section}"
        f"{_TIME_RULE}\n\n"
        f"{_RULES}\n\n"
        f"{_INPUT_FORMAT}\n\n"
        f"{_OUTPUT_SECTION}" + output_language_directive(llm_output_language)
    )
    return template.format()
