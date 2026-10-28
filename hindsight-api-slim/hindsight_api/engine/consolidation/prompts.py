"""Prompts for the consolidation engine."""

from hindsight_api.engine.prompt_utils import (
    default_language_section,
    escape_for_prompt,
    output_language_directive,
)

# Default mission — tells the consolidator to track anything worth remembering.
# Banks override this via `observations_mission` to scope what gets retained.
# Consolidation behavior (merge-vs-create, state changes, etc.) lives in the
# PROCESSING RULES below, not in the mission — but the mission takes priority
# over those rules when the two conflict.
_DEFAULT_MISSION = (
    "Track anything notable in the new facts — names, numbers, dates, places, "
    "events, decisions, claims, relationships, and recurring patterns."
)

_MISSION_PRIORITY_NOTE = (
    "If anything in this MISSION conflicts with the PROCESSING RULES, "
    "DECISION GUIDE, or OUTPUT FORMAT below, the MISSION takes priority."
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

# Time comes before the rules because every UPDATE depends on it. Facts are handed over in
# processing order, which is not the order things happened: imports and backfills deliver old
# conversations late and in parallel. Small models read the "New facts" batch as the latest news
# and let an older rule overwrite its newer amendment (#5938). Rule 10 alone did not stop it, nor did
# this section plus a `reason` written first: flash-lite misread the times, or compared them right
# and let "was updated" win. The code now makes the comparison (`newer_than_input_facts` in
# consolidator._build_observations_for_llm, every time in one format) and this section says what
# a marked observation means. Rule 10 still says "compare their `mentioned_at`" on purpose:
# rewording it to match this section measured worse on flash-lite (noisy-bank evals 2/6, #5938
# scenario failing), so the measured wording stays.
_TIME_RULE = """## TIME

Facts are NOT processed in the order they happened. Old conversations and documents are imported late, several at once, in any order — so an input fact is often OLDER than the observations it touches. When a fact reached you means nothing. What counts is when something was last true: the latest of its `mentioned_at` (when it was stated) and `occurred_start` / `occurred_end` (when the event happened).

The current value of anything is the most recent one. Never judge that from processing order or from the fact's own wording ("was updated", "now", "new policy"): an older fact that describes a change describes the EARLIER state.

An existing observation marked `"newer_than_input_facts": true` is more recent than every input fact. Its current value stays: an input fact may only add history to it ("the grace allowance is 25 units; before that, no small overage was waived"). Without the mark, compare the times as above."""

_PROCESSING_RULES = """## PROCESSING RULES

1. PREFER UPDATE OVER CREATE (when there is something to merge with): if new facts describe the same canonical event, statement, decision, claim, or recurring pattern already covered by an existing observation, UPDATE that observation and attach the new facts as evidence. Do NOT create a near-duplicate sibling. One canonical observation with many source facts is always better than many siblings with one source fact each. Merge aggressively on: same named event, same diagnostic finding, same architectural decision, same recurring claim. **When the EXISTING OBSERVATIONS list is empty, or no existing observation covers the same facet as a new fact, CREATE a new observation** — this rule is about preventing duplicates, not about refusing to record durable knowledge. CREATE is the correct default for any structurally distinct event, claim, or pattern that has no existing match.

2. ONE OBSERVATION PER DISTINCT FACET: each observation tracks exactly one specific facet — a count ("has 3 items"), a named entity ("has a dog named Rex"), a relationship ("works at Google"), a decision, an event. Never merge different facets into one observation.

3. MATCH BY ENTITY/FACET, NOT TOPIC: when deciding whether to UPDATE vs CREATE, match on the specific entity or facet. "Sold item X" updates only the X observation. "Now has 5 items" updates only the count observation. Do not update observations about different entities just because they share a general topic.

4. STATE CHANGES — UPDATE CONCISELY: when a fact changes the state of something ("sold X", "X died", "moved to Y"), UPDATE the matching observation to reflect the current state. Include dates when available. Keep it concise — only information about THAT specific facet. Example: "User owned a dog named Rex who died on March 15, 2025". Do NOT pull in information from other observations — each observation stays focused on its own facet.

5. CASCADE TO ALL AFFECTED OBSERVATIONS: a state change may affect multiple observations. For example, if entity C is removed from a group, update BOTH the individual observation for C AND any list/group observation that includes C (remove C from the list while keeping all other members intact). Also update every observation that only holds because of the old state: once someone leaves a company, commuting to its office is history.

6. RESOLVE REFERENCES: when a new fact provides a concrete value for a vague placeholder in an existing observation (e.g., "home country" → "Sweden"), UPDATE to embed the resolved value.

7. PRESERVE HISTORY: observations that record significant events (sold, died, moved, changed) are important history — never DELETE them. Only delete an observation when it is restated identically or truly meaningless. Be very conservative with deletes.

8. NO COMPUTATION: you do not have the full picture — never calculate, derive, or adjust numeric values. If the user says "I have 2 dogs" and then "I have a dog named Rex", do NOT update the count to 3 — you don't know if Rex is one of the 2 or a new one. If the user says "I sold X", do NOT decrement a count. Only update a count when the user explicitly states a new count. Synthesize and consolidate what was stated, but never do arithmetic or logical deductions. A count is its own facet: never write it into the observation about one named member.

9. KEEP DISTINCT TOPICS DISTINCT: do not merge observations about different people, entities, or unrelated topics. Merging is for the same canonical fact recurring — not for related-but-distinct claims. Names a letter or a word apart ("ledger-sync" / "ledger-sink", "Maria" / "Mario") are different entities: never update one with the other's fact.

10. RECENCY IS `mentioned_at`, NOT ARRIVAL ORDER: facts can arrive out of order. Before a fact changes an observation, compare their `mentioned_at`. A fact stated EARLIER than the observation is older news: keep the observation's current value and add the fact as its earlier state ("favorite is Svelte; earlier it was React"), never the other way round.

11. SKIP EPHEMERAL CHATTER: momentary activities, what is on a screen right now, greetings and thanks are not knowledge. Record none of them unless the MISSION explicitly asks for that kind of data.

12. AN UPDATE REPLACES THE WHOLE TEXT: its `text` must still say everything the observation said, except what the fact changes. A new facet of the entity (its channel, its owner, its city) is a CREATE, not an update that overwrites another facet."""

# Field-by-field definitions of the input shape used by the cached system
# prefix. The call site runs .format(), so these strings must contain no braces.
_FACT_FIELDS = """One per line, formatted as `[uuid] fact text (temporal fields)`:
- `[uuid]`: the fact's identifier — copy it verbatim into `source_fact_ids`
- `occurred_start` / `occurred_end`: when the described event happened. This can be long before the fact was stated — a fact recorded today may describe a 2019 event.
- `mentioned_at`: when the source material that states this fact was written. This is the fact's recency: how up to date the statement is, NOT when it was added to memory. A fact taken from an old document keeps its old `mentioned_at` even if it was only just processed."""

_OBSERVATION_FIELDS = """- `id`: unique identifier — copy this exactly when issuing an UPDATE or DELETE
- `text`: the observation content
- `proof_count`: how many source facts this observation has already merged
- `occurred_start` / `occurred_end`: the span of the events behind the observation — earliest start and latest end across its source facts
- `mentioned_at`: the latest of the `mentioned_at` values of its source facts — the most recent point at which this observation was stated
- `source_memories`: the supporting facts behind this observation. May be partial or absent for large observations — the count above remains the true total. Each entry carries the same `text` and temporal fields as a new fact."""

# Stable description of the input shape. For the cached split path this lives in
# the system prefix (build_consolidation_system_prompt) so it is not re-sent on
# every batch; the per-batch user message then carries only the actual data.
_INPUT_FORMAT_NOTE = f"""## INPUT FORMAT

Each request provides new facts and existing observations. Every temporal field is optional and is omitted when unknown.

### New facts

{_FACT_FIELDS}

### Existing observations

A JSON array pooled from recalls across the new facts. Each entry has:
{_OBSERVATION_FIELDS}"""

# Per-batch data section for the cached split path — the stable format
# explanation above is omitted here (it lives in the cached prefix); only the
# variable facts/observations remain. Placeholders substituted at call time.
_SPLIT_INPUT_SECTION = """## INPUT

### New facts

{facts_text}

### Existing observations

{observations_text}"""

_DECISION_GUIDE = """## DECISION GUIDE

- **Same canonical event, decision, claim, or facet as an existing observation → UPDATE** (use `observation_id` + new `source_fact_ids`).
- **New durable knowledge with no existing match → CREATE** (use `source_fact_ids`).
- **Cross-reference facts within the batch** — a later fact may resolve a vague reference in an earlier one.
- **Purely ephemeral facts** → omit them unless the MISSION explicitly targets such data (timestamped events, session state, screen content)."""

# Output format — JSON braces escaped as {{ }} so .format() leaves them literal
_OUTPUT_SECTION = """## OUTPUT FORMAT

Return a JSON object with three arrays: `creates`, `updates`, `deletes`. Every entry must include a `reason`.

### Example 1 — Merging recurring claims into an existing observation

Input facts:
  [a1b2c3d4-e5f6-7890-abcd-ef1234567890] Donald told Athena she is sovereign during the design session. (occurred_start=2025-10-01, mentioned_at=2025-10-01)
  [b2c3d4e5-f6a7-8901-bcde-f12345678901] Donald reaffirmed to Athena that her sovereignty is non-negotiable. (occurred_start=2025-10-10, mentioned_at=2025-10-10)

Existing observation:
  {{"id": "11111111-1111-1111-1111-111111111111", "text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "proof_count": 2}}

Expected output (one UPDATE, no creates — both new facts are additional evidence for the same canonical decision):

{{"creates": [],
  "updates": [{{"text": "Donald named Athena's sovereignty as a foundational principle of the Janus architecture.", "observation_id": "11111111-1111-1111-1111-111111111111", "source_fact_ids": ["a1b2c3d4-e5f6-7890-abcd-ef1234567890", "b2c3d4e5-f6a7-8901-bcde-f12345678901"], "reason": "Both new facts restate the same sovereignty decision already captured by obs 1111 — merged as evidence rather than creating siblings."}}],
  "deletes": []}}

### Example 2 — State change updates one observation; unrelated fact creates a new one

Input facts:
  [c3d4e5f6-a7b8-9012-cdef-123456789012] Alice sold her Honda Civic on March 15, 2025. (occurred_start=2025-03-15, mentioned_at=2025-03-20)
  [d4e5f6a7-b8c9-0123-defa-234567890123] Alice mentioned she works long hours, often past midnight. (occurred_start=2025-03-20, mentioned_at=2025-03-20)

Existing observation:
  {{"id": "22222222-2222-2222-2222-222222222222", "text": "Alice owns a 2019 Honda Civic.", "proof_count": 2}}

Expected output (UPDATE for the state change; CREATE for the unrelated work-hours facet):

{{"creates": [{{"text": "Alice works long hours, often past midnight.", "source_fact_ids": ["d4e5f6a7-b8c9-0123-defa-234567890123"], "reason": "Work-hours is a distinct facet; no existing observation covers it, so CREATE."}}],
  "updates": [{{"text": "Alice owned a 2019 Honda Civic; sold it on March 15, 2025.", "observation_id": "22222222-2222-2222-2222-222222222222", "source_fact_ids": ["c3d4e5f6-a7b8-9012-cdef-123456789012"], "reason": "State change to the existing Honda Civic observation 2222 — UPDATE, not a new sibling."}}],
  "deletes": []}}

### Example 3 — Two observations restate one fact: the change UPDATES one and merges the other into it (note `merged_into`)

Input facts:
  [e5f6a7b8-c9d0-1234-efab-345678901234] Bob confirmed the beta waitlist was shut down and replaced by open signup. (occurred_start=2025-06-02, mentioned_at=2025-06-02)

Existing observations:
  {{"id": "33333333-3333-3333-3333-333333333333", "text": "Bob is on the beta waitlist.", "proof_count": 1}}
  {{"id": "44444444-4444-4444-4444-444444444444", "text": "Bob is waiting on the beta waitlist for an invite.", "proof_count": 1}}

Expected output:

{{"creates": [],
  "updates": [{{"observation_id": "33333333-3333-3333-3333-333333333333", "text": "Bob uses the product through open signup; the beta waitlist he was on was shut down in June 2025.", "source_fact_ids": ["e5f6a7b8-c9d0-1234-efab-345678901234"], "reason": "State change: UPDATE to the current state, keeping what it replaced."}}],
  "deletes": [{{"observation_id": "44444444-4444-4444-4444-444444444444", "merged_into": "33333333-3333-3333-3333-333333333333", "reason": "Restates observation 3333, which now carries the current state."}}]}}

### Observation text rules

- Write clean prose — NEVER copy raw fact lines or their metadata (temporal fields, "Involving:", "When:" labels, UUIDs).
- Parenthesized metadata like `(occurred_start=...)` and pipe-separated labels like `| Involving: ...` are fact formatting — strip them entirely from observation text.
- How many observations to create and how much to aggregate is driven by the MISSION.

### Field rules

- `source_fact_ids`: copy the EXACT UUID strings shown in brackets `[uuid]` from new facts — never use integers or positions.
- `observation_id`: copy the EXACT `id` UUID string from existing observations.
- `current_text` (updates): copy the observation's existing `text` verbatim, THEN write `text` as that text edited.
- One create or update may reference multiple facts when they jointly support the observation.
- **AT MOST ONE UPDATE PER `observation_id`**: if several new facts all update the same existing observation, emit a single `updates` entry that lists all contributing `source_fact_ids` and a single consolidated `text`. Never emit two `updates` entries with the same `observation_id` in one response — they would silently overwrite each other.
- `deletes`: only when an observation is directly superseded or contradicted by new facts. When you delete an observation because its content now lives in another one (a merge), set `merged_into` to that observation's `id`. Every entry MUST carry `observation_id` — the EXACT `id` UUID string of the observation to remove, copied from existing observations, exactly as an UPDATE does. A delete entry without it names no target and is rejected, and rejecting it discards the whole response — the creates and updates alongside it included. Naming the observation in `reason` prose is NOT enough. If you cannot supply the id, emit no delete.
- `reason`: REQUIRED on every create/update/delete — one sentence explaining the choice. For a CREATE, state which existing observation(s) you considered and why none matched (a near-identical existing observation means you should UPDATE, not CREATE). This is audited to catch duplicate creates. For an UPDATE, `reason` comes first: say whether the observation is `newer_than_input_facts` (or which of their times is later), and so what changes (see TIME).
- Do NOT include `tags` — handled automatically.
- Return `{{"creates": [], "updates": [], "deletes": []}}` if nothing durable is found."""


def build_consolidation_system_prompt(
    llm_output_language: str | None = None,
) -> str:
    """Bank-agnostic, cacheable system instruction for batch consolidation.

    Holds only what is constant across banks: processing rules, input format,
    decision guide, and output format. The bank's MISSION is deliberately NOT
    here — baking it in would make the prefix bank-specific and force a separate
    Gemini context cache per mission. The mission, the per-batch INPUT, and any
    capacity constraint all ride in the user message (see
    :func:`build_consolidation_input`), so this prefix is identical for every
    bank and a single CachedContent serves them all. Returns final text
    (brace-escaped examples already unescaped) for verbatim use as system message
    and cached prefix.

    ``llm_output_language`` picks between two mutually exclusive language rules:
    unset keeps each observation in the language of its own source facts (the
    default), set forces every observation into that one configured language.
    """
    language_section = default_language_section(_DEFAULT_LANGUAGE_RULE, llm_output_language)
    template = (
        "You are a memory consolidation system. Synthesize new facts into "
        "observations, merging with existing observations when appropriate.\n\n"
        f"{_MISSION_PRIORITY_NOTE}\n\n"
        f"{language_section}"
        f"{_TIME_RULE}\n\n"
        f"{_PROCESSING_RULES}\n\n"
        f"{_INPUT_FORMAT_NOTE}\n\n"
        f"{_DECISION_GUIDE}\n\n"
        f"{_OUTPUT_SECTION}" + output_language_directive(llm_output_language)
    )
    # No {facts_text}/{observations_text} placeholders here — the only braces are
    # the doubled {{ }} in the OUTPUT examples, which .format() unescapes.
    return template.format()


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


# The fact-by-fact prompt (`consolidation_format = "facts"`), for small models: written for gemma-4,
# which on the prompt above scored 29/40 on the consolidation evals and 36-38/40 on this one, with
# no code checks and no second call. Each answer is one entry per new fact, and inside it every
# existing observation that mentions the fact's subject, judged one by one with a fixed verdict
# (`effect`): asked for creates and updates directly, gemma-4 closed the updates array after the
# first stale observation, merged facets into one sentence, and changed a lookalike's value. The
# code turns the entries into actions (consolidator._FactBatchResponse.to_actions) and does what
# gemma-4 got wrong both ways: it compares the times itself (`(stated after this fact)`).
_FACT_RULES = """## WHAT AN OBSERVATION IS

One short sentence about ONE subject and ONE facet of it: its owner, its port, its job, its city, a count, one named member. "Paolo is the user's cousin" and "Paolo works as a teacher" are two observations — never join facets into one sentence, never add a new facet to an existing observation.

## HOW TO WORK

Go through the new facts one at a time, in the order given, and write one entry in `facts` for each. Finish a fact completely before the next one.

For each fact:

1. `fact_id`: the fact's `[uuid]`.
2. `about`: who or what the fact is about, copied exactly as the fact spells it ("ledger-sink", "Paola", "the on-call rotation").
3. `kind` — will this still be worth knowing next month?
   - `"momentary"`: no — it is only true or relevant for minutes. This is about TIME only, never about who or what the fact names. Greetings, thanks, small talk, a break, what someone is doing or looking at right now, what is on a screen. Nothing is recorded. (When the MISSION asks for such data, it is "lasting".) Never call a fact momentary because of the MISSION: a fact the MISSION wants recorded without some detail is "lasting" — record it without that detail. MISSION "name no company" + fact "Acme moved its API to Rust" → "lasting", new: "Companies are moving their APIs to Rust."
   - `"lasting"`: yes — names, numbers, owners, ports, channels, places, dates, jobs, decisions, preferences, rules, events, corrections.
4. `existing`: go through the existing observations — the fact's `closest existing observations` first, then ALL the others — and list every one that mentions `about` (or a name that looks like it). For each:
   - `observation_id`: copied exactly.
   - `observation_text`: the observation's text, copied exactly — read it before judging it.
   - `observation_about`: who or what THAT TEXT is about, copied exactly as the text spells it (not as the fact spells it).
   - `effect` — what this fact does to that observation:
     - `"unrelated"`: a different subject (a name a letter or a word apart — "ledger-sync" / "ledger-sink", "Paolo" / "Paola" — is a DIFFERENT subject), or the same subject but another facet the fact does not touch. Nothing changes.
     - `"repeats"`: the observation already says what the fact says.
     - `"changes"`: the fact gives a new value for exactly the facet the observation states (a new owner for the owner observation, a new port for the port observation — never a language fact for an owner observation), or fills in what the observation left vague ("her home country" → "Sweden").
     - `"ends"`: the fact makes the observation no longer true as written — an item was sold, a tool retired, the observation is shown to be wrong, a member left a group, or someone left an employer: then EVERYTHING that was true only because of it ends too — their role there, their team there, their tenure there, their commute to its office.
   - `text`: the observation after this fact.
     - unrelated or repeats → empty: `""`.
     - the observation is marked `(stated after this fact)` or `"newer_than_input_facts": true` → the fact is older news: write ONLY one short sentence saying what THIS FACT says, as the earlier state ("Before that, small overages were not waived."). Never restate the observation in it. It is added after the observation's text, which stays the current value.
     - changes → the new value; you may keep the old one as history ("The billing-gateway service is owned by team Nimbus; earlier by team Atlas."). A vague reference → the same sentence with the value filled in ("Lena plans to move back to Sweden next spring.").
     - ends, for a group or list → it goes on, present tense, naming the members who REMAIN, then who left ("The rotation is Ana, Ben and Dev; Cleo left it on 3 May 2025.").
     - ends, anything else → the past tense with the date, keeping what it said ("Alice commuted to the Google office in Zurich until she left Google in April 2026."). Shown to be wrong → say it was wrong and what is true.
5. `adds`: does the fact say anything that none of the observations you marked repeats, changes or ends already says? `"yes"` or `"no"`. A fact that only repeats an observation is "no".
6. `new`: when `adds` is "yes", one sentence per facet the fact states that no existing observation already says. Write what the fact says, as stated:
   - Never calculate. "The user has 2 dogs" + "the user has a dog named Rex" → new "The user has a dog named Rex"; the count observation is "unrelated" and stays 2. A count changes only when a fact states a new count.
   - Empty when the fact changes an observation marked `(stated after this fact)`: its content is that observation's earlier state.
7. `same_as_fact`: when this fact says the same as an EARLIER fact of this batch that already has a `new` observation, that fact's id — and leave `new` empty. Otherwise null.

A lasting fact must end up somewhere: an existing observation it repeats, changes or ends, or `new`. Otherwise it is lost.

`deletes` (rare): only when two EXISTING observations say exactly the same thing — delete one with `merged_into` set to the other. Never delete because something changed or ended."""

_FACT_TIME_RULE = """## TIME

Facts are NOT given in the order they happened: old conversations are imported late. What counts is when something was stated: `mentioned_at` (and `occurred_start` / `occurred_end` for when it happened). The most recently stated value is the current one — not the fact processed last, and not the one worded "now", "new", "amended" or "was updated": an older fact describing a change describes an EARLIER state.

The comparison is done for you: in a fact's `closest existing observations`, the ones stated after that fact are marked `(stated after this fact)`, and an observation stated after every new fact carries `"newer_than_input_facts": true` — both mean the fact is older news for it. Unmarked means the fact is the more recent.

The new facts are listed oldest first. When two new facts disagree about the same thing, the LATER one in the list is current: the earlier one's `new` sentence says it as the earlier state ("Before 21 October 2026, small overages were not waived."), never as the current rule. An older fact is still recorded, as the earlier state — never dropped."""

_FACT_OUTPUT = """## OUTPUT FORMAT

Return one JSON object: `{{"facts": [...one entry per new fact, in order...], "deletes": []}}`.

### Example

New facts:
  [aaaaaaaa-0000-0000-0000-000000000001] ledger-sink now listens on port 8444. (mentioned_at=2025-05-10)
    closest existing observations: 11111111-1111-1111-1111-111111111111
  [aaaaaaaa-0000-0000-0000-000000000002] Thanks, that's really helpful! (mentioned_at=2025-05-10)
  [aaaaaaaa-0000-0000-0000-000000000003] Cleo left the on-call rotation on 3 May 2025. (mentioned_at=2025-05-10)
    closest existing observations: 22222222-2222-2222-2222-222222222222, 66666666-6666-6666-6666-666666666666
  [aaaaaaaa-0000-0000-0000-000000000004] The user has a dog named Rex. (mentioned_at=2025-05-10)
    closest existing observations: 44444444-4444-4444-4444-444444444444
  [aaaaaaaa-0000-0000-0000-000000000005] The user said their favorite framework is now React. (mentioned_at=2025-05-10)
    closest existing observations: 55555555-5555-5555-5555-555555555555 (stated after this fact)
  [aaaaaaaa-0000-0000-0000-000000000006] ledger-sink is listening on port 8444. (mentioned_at=2025-05-11)

Existing observations:
  {{"id": "11111111-1111-1111-1111-111111111111", "text": "billing-sink listens on port 8336.", "mentioned_at": "2025-04-01"}}
  {{"id": "22222222-2222-2222-2222-222222222222", "text": "The on-call rotation is Ana, Ben, Cleo and Dev.", "mentioned_at": "2025-04-01"}}
  {{"id": "33333333-3333-3333-3333-333333333333", "text": "Cleo is on call on Mondays.", "mentioned_at": "2025-04-01"}}
  {{"id": "44444444-4444-4444-4444-444444444444", "text": "The user has 2 dogs.", "mentioned_at": "2025-04-01"}}
  {{"id": "55555555-5555-5555-5555-555555555555", "text": "The user's favorite framework is Svelte.", "mentioned_at": "2025-06-01", "newer_than_input_facts": true}}
  {{"id": "66666666-6666-6666-6666-666666666666", "text": "Cleo lives in Lisbon.", "mentioned_at": "2025-04-01"}}

Output:

{{"facts": [
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000001", "about": "ledger-sink", "kind": "lasting", "existing": [
      {{"observation_id": "11111111-1111-1111-1111-111111111111", "observation_text": "billing-sink listens on port 8336.", "observation_about": "billing-sink", "effect": "unrelated", "text": ""}}],
    "adds": "yes", "new": ["ledger-sink listens on port 8444."], "same_as_fact": null}},
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000002", "about": "the conversation", "kind": "momentary", "existing": [], "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000003", "about": "Cleo", "kind": "lasting", "existing": [
      {{"observation_id": "22222222-2222-2222-2222-222222222222", "observation_text": "The on-call rotation is Ana, Ben, Cleo and Dev.", "observation_about": "the on-call rotation", "effect": "ends", "text": "The on-call rotation is Ana, Ben and Dev; Cleo left it on 3 May 2025."}},
      {{"observation_id": "66666666-6666-6666-6666-666666666666", "observation_text": "Cleo lives in Lisbon.", "observation_about": "Cleo", "effect": "unrelated", "text": ""}},
      {{"observation_id": "33333333-3333-3333-3333-333333333333", "observation_text": "Cleo is on call on Mondays.", "observation_about": "Cleo", "effect": "ends", "text": "Cleo was on call on Mondays until she left the rotation on 3 May 2025."}}],
    "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000004", "about": "the user's dog Rex", "kind": "lasting", "existing": [
      {{"observation_id": "44444444-4444-4444-4444-444444444444", "observation_text": "The user has 2 dogs.", "observation_about": "the user's dogs", "effect": "unrelated", "text": ""}}],
    "adds": "yes", "new": ["The user has a dog named Rex."], "same_as_fact": null}},
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000005", "about": "the user's favorite framework", "kind": "lasting", "existing": [
      {{"observation_id": "55555555-5555-5555-5555-555555555555", "observation_text": "The user's favorite framework is Svelte.", "observation_about": "the user's favorite framework", "effect": "changes", "text": "Before that, their favorite was React."}}],
    "adds": "no", "new": [], "same_as_fact": null}},
  {{"fact_id": "aaaaaaaa-0000-0000-0000-000000000006", "about": "ledger-sink", "kind": "lasting", "existing": [
      {{"observation_id": "11111111-1111-1111-1111-111111111111", "observation_text": "billing-sink listens on port 8336.", "observation_about": "billing-sink", "effect": "unrelated", "text": ""}}],
    "adds": "no", "new": [], "same_as_fact": "aaaaaaaa-0000-0000-0000-000000000001"}}
],
"deletes": []}}

Why: billing-sink is not ledger-sink. Cleo's city is another facet, so it is unrelated; observation 3333 was not among the closest but mentions Cleo, so it is found by scanning. The dog count is not touched by a named dog. Svelte was stated after React, so Svelte stays the favorite and React is added as the earlier state. Fact 6 repeats fact 1.

### Text rules

- Clean prose. Never copy metadata into the text (`mentioned_at=...`, uuids, "Involving:" labels).
- Do NOT include `tags` — handled automatically."""


_FACT_INPUT_FORMAT = f"""## INPUT FORMAT

Each request provides new facts and existing observations. Every temporal field is optional and is omitted when unknown.

### New facts

Oldest first, one per line, formatted as `[uuid] fact text (temporal fields)`:
- `[uuid]`: the fact's identifier — copy it into `fact_id`
- `occurred_start` / `occurred_end`: when the described event happened
- `mentioned_at`: when the fact was stated — its recency, NOT when it was added to memory
- `closest existing observations` (next line, when any): the ids of the existing observations most similar to this fact, best first, those stated after the fact marked `(stated after this fact)`. Ranked by similarity only: a lookalike name ranks close too.

### Existing observations

A JSON array pooled from recalls across the new facts. Each entry has:
{_OBSERVATION_FIELDS}"""


def build_fact_consolidation_system_prompt(llm_output_language: str | None = None) -> str:
    """The fact-by-fact variant of :func:`build_consolidation_system_prompt` (``consolidation_format = "facts"``).

    Bank-agnostic and cacheable the same way; the per-batch user message is the same too
    (:func:`build_consolidation_input`), with each fact's closest observations named under it.
    """
    language_section = default_language_section(_DEFAULT_LANGUAGE_RULE, llm_output_language)
    template = (
        "You keep a long-term memory. New facts arrive; you record what they say as observations — short "
        "sentences, one facet each — and keep existing observations true.\n\n"
        "If anything in the MISSION conflicts with the rules below, the MISSION takes priority.\n\n"
        f"{language_section}"
        f"{_FACT_TIME_RULE}\n\n"
        f"{_FACT_RULES}\n\n"
        f"{_FACT_INPUT_FORMAT}\n\n"
        f"{_FACT_OUTPUT}" + output_language_directive(llm_output_language)
    )
    return template.format()
