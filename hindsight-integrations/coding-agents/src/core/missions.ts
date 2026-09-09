/**
 * Harness-agnostic Hindsight missions, retain strategies, and knowledge-page taxonomy.
 *
 * These describe HOW a coding project's memory is extracted and reasoned over. They are independent
 * of which agent harness produced the sessions, so they live in the shared core and are reused by
 * every harness adapter.
 */

import { createHash } from "node:crypto";

// ── retain missions (git vs chat need different extraction) ─────────────────────
export const GIT_MISSION =
  "You are ingesting a single git commit: its message and its full diff. Extract the concrete " +
  "technical DECISION and the CAUSE/INVARIANT it encodes, bound to the specific code entities " +
  "(functions, methods, files) and behaviors it changes. Preserve exact identifiers, paths, and " +
  "literal values verbatim. Preserve the 'REF-ID: <token>' marker verbatim in every fact. Capture " +
  "both WHAT changed and WHY. Issue/PR references (#123, GH-123, PROJ-123) are load-bearing: keep " +
  "them VERBATIM in the fact text and emit each as an ENTITY, so a later question about that issue " +
  "or PR retrieves this decision.";

export const GITLOG_MISSION =
  "You are ingesting an aggregated block of git commit MESSAGES ONLY (no diffs) — the project's " +
  "recent commit-message history, newest first. Extract the project's INITIATIVES, FEATURES, " +
  "ENHANCEMENTS, and notable changes or THEMES over time — what the project has been working on and " +
  "how it has evolved. Do NOT extract per-line code detail (there is no diff to draw it from). Group " +
  "related commits into a coherent initiative/theme where the messages make that clear; preserve exact " +
  "identifiers and literal values verbatim when quoting a subject line. Keep issue/PR references " +
  "(#123, GH-123) VERBATIM and emit each as an ENTITY — they are how future sessions will ask about " +
  "this work.";

export const CONVERSATION_MISSION =
  "You are ingesting a developer conversation as a JSONL transcript (one {role, content} turn per line): the " +
  "user's requests, the assistant's narration, and compact 'action' turns naming each tool use and " +
  'its target (e.g. "Edit boltons/strutils.py") with no arguments or outputs. It may be a SHORT ' +
  "decision chat or a LONG working session — scale the facts to the substance, never to the message " +
  "count. Extract the FEWEST facts that capture the OUTCOME: the settled DECISIONS and their exact " +
  "rules/values (quote literals VERBATIM); concrete CHANGES to specific code entities; problems and " +
  "how they were resolved; conventions or invariants established; at most one fact for a notable " +
  "REJECTED alternative ('initially proposed X, changed to Y because Z'). A short decision chat " +
  "usually yields 1-2 facts; a substantial working session several. CRITICAL: a conversation REVISES " +
  "itself — record ONLY the FINAL state as what is in effect; a superseded proposal appears ONLY " +
  "inside the rejected fact, NEVER as its own 'decided' fact; if the same setting changes several " +
  "times keep only the LAST, and make unmistakably clear which choice WON. Do NOT emit one fact per " +
  "message, per intermediate proposal, or per action turn. Keep issue/PR references (#123, GH-123) " +
  "VERBATIM and emit each as an ENTITY. Preserve the 'REF-ID: <token>' marker " +
  "verbatim in every fact. Do not invent; capture only what was actually settled.";

export const REFLECT_MISSION =
  "You are a debugging assistant with the project's past decisions in memory (git rationale and " +
  "developer chats). Given a bug's SYMPTOM, find the past decision whose rationale explains the ROOT " +
  "CAUSE — not one that merely shares vocabulary. Answer with the PRECISE fix: state the EXACT rule " +
  "and the LITERAL values, identifiers, strings, numbers, or set members that were decided — quote " +
  "them VERBATIM, never paraphrase, generalize, or omit them (give the actual decided value, not " +
  "'the project standard'). If memories CONFLICT on the same rule, the LATEST decision wins — " +
  "prefer facts that explicitly amend or supersede an earlier one, state the superseded rule as " +
  "no longer in effect, and never present it as the fix. Name the function/file to change and " +
  "cite the REF-ID(s). If NOTHING in memory genuinely explains THIS symptom, say exactly that in " +
  "one short sentence — a wrong-but-confident nearest match is worse than an honest miss; never " +
  "stretch an unrelated decision to fit.";

export const DOCUMENT_MISSION =
  "You are ingesting a standalone document (notes, docs, or structural findings). Extract the " +
  "concrete facts, concepts, and structure it describes.";

export const OBSERVATIONS_MISSION =
  "Consolidate durable knowledge about THIS codebase — recurring patterns, conventions, module " +
  "responsibilities, and how components relate — from the ingested commits and conversations. " +
  "Favor stable structural understanding over one-off details. When a new fact contradicts or " +
  "supersedes an existing observation, UPDATE that observation to reflect the current state rather " +
  "than creating a sibling alongside it; note that the rule was revised and when, so the superseded " +
  "version is visible as history rather than as a competing claim.";

export const RETAIN_STRATEGIES = {
  git: { retain_mission: GIT_MISSION, retain_extraction_mode: "verbose" },
  // ONE big aggregated document (last N commit messages, no diffs) -> a larger chunk size so it stays
  // in as few chunks as possible and the extractor sees the whole history arc at once.
  gitlog: {
    retain_mission: GITLOG_MISSION,
    retain_extraction_mode: "verbose",
    retain_chunk_size: 12000,
  },
  // ONE strategy for ALL developer conversations — backfilled decision chats and live working
  // sessions alike (they are the same content type in the same JSON transcript format; the mission
  // scales extraction to the substance, final-state-wins). Chunk big enough to hold a whole typical
  // conversation in ONE chunk so the extractor sees the full proposal→revision arc (the 3000
  // default SPLIT them into per-chunk fragments); very long sessions still split and fall back to
  // the consolidation layer.
  conversation: {
    retain_mission: CONVERSATION_MISSION,
    retain_extraction_mode: "verbose",
    retain_chunk_size: 12000,
  },
  // Structural documents (e.g. the codebase survey's ingested findings) aren't dialogue — the
  // chat strategy's "final decision vs rejected proposal" extraction doesn't apply. Verbose mode
  // with a bigger chunk size (documents can run long) captures the concrete facts/structure instead.
  document: {
    retain_mission: DOCUMENT_MISSION,
    retain_extraction_mode: "verbose",
    retain_chunk_size: 12000,
  },
  // Codebase-SURVEY lifecycle documents, ONE strategy with conditional rules: the survey's
  // internal status markers ("researching…"/"completed" baselines) must yield ZERO memories,
  // while any actual survey findings routed here extract as concrete structural facts.
  survey: {
    retain_extraction_mode: "custom",
    retain_custom_instructions:
      "This document belongs to the Hindsight codebase-survey lifecycle. Apply ONE of two rules: " +
      "(1) If the content is an internal status marker — it says it is an internal marker, or " +
      "merely announces that a survey started/completed at some commit — extract NOTHING: return " +
      "an empty list of facts. (2) Otherwise the content is survey FINDINGS about the codebase: " +
      "extract the concrete structural facts it states (components and their responsibilities, " +
      "key concepts, conventions, tech stack), preserving identifiers verbatim.",
    retain_chunk_size: 12000,
  },
} as const;

// ── passive tier tagging (entity_labels) ───────────────────────────────────────
// A single hierarchical bank-config group set by `configureBank` at seed time. `tag: true` makes the
// extractor copy each selected `knowledge:<value>` onto the fact's tags (via `_inject_label_tags`),
// giving every durable fact a knowledge-tier routing tag the server-side knowledge base (and any
// tag-filtered query) can select on. The vocabulary is FIXED (not per-feature) because tag matching
// is exact set-ops with no wildcards.
export interface EntityLabelValue {
  value: string;
  description: string;
}

export interface EntityLabelGroup {
  key: string;
  type: "multi-values";
  optional: boolean;
  tag: boolean;
  description: string;
  values: EntityLabelValue[];
}

export const KNOWLEDGE_LABELS: EntityLabelGroup = {
  key: "knowledge",
  type: "multi-values", // 0, 1, or several — empty is normal
  optional: true,
  tag: true, // emits knowledge:<value> onto the fact's tags
  description:
    "Routing labels for this project's Hindsight KNOWLEDGE PAGES — curated, human-readable summaries " +
    "of the repo's DURABLE engineering knowledge (architecture, key decisions, conventions, ongoing " +
    "initiatives), each page rebuilt automatically from the facts labeled for it. Mark a fact only when " +
    "it is durable, reusable knowledge a developer would still want surfaced in future sessions. " +
    "IMPORTANT: leave this EMPTY for routine, transient, or operational facts — a passing test, a " +
    "one-off command, a status update, a debugging dead-end. MOST facts should get no label here. " +
    "Assign more than one value only when the fact genuinely fits several.",
  values: [
    {
      value: "feature-work",
      description:
        "A new feature, initiative, or enhancement being planned or built — the capability being added " +
        "and the intent behind it. Not routine bug-fixes or chores.",
    },
    {
      value: "decision",
      description:
        "A technical decision that will constrain future work, with its rationale — why this approach " +
        "was chosen over alternatives, or a rule deliberately adopted.",
    },
    {
      value: "convention",
      description:
        "An established way this project does things — naming, structure, testing, error handling, or " +
        "another recurring pattern a contributor is expected to follow.",
    },
    {
      value: "component",
      description:
        "What a specific module, file, service, or subsystem is responsible for, or how components " +
        "depend on and connect to one another.",
    },
    {
      value: "concept",
      description:
        "A domain concept, key abstraction, or piece of project vocabulary a new contributor must " +
        "understand to work effectively.",
    },
  ],
};

// Knowledge PAGES (OKF pages = mental models) = a developer's durable mental model of the codebase,
// CONSOLIDATED from the ingested MEMORY (commit history + past conversations) — NOT mirrored from the
// current source (which would need constant re-sync). A universal 5-page taxonomy that generalizes to
// any repo; the curator populates each from history+chats and can spawn per-component sub-pages.
// A seeded page is a tag-scoped synthesis view: `tags` pins it to one `knowledge:<tier>` label so
// its synthesis draws from the facts the extractor routed to that tier (exact set-ops — see
// KNOWLEDGE_LABELS above; names/tiers mirror the label vocabulary). The tiers say what KIND of
// knowledge a fact is, never WHOSE — `pageScopeRule` below carries that half.
export interface KnowledgePage {
  name: string;
  source_query: string;
  tags: string[];
}

/**
 * The subject-scoping clause every page this plugin creates carries, naming the subject it is
 * about — the seeded taxonomy (through `pagesFor`) and each captured initiative alike.
 *
 * `project` is the repository when the bank is one repository's, and the BANK otherwise — a bank
 * several repos share has no repo to name, and naming whichever one seeded last made the sentence
 * flip on every session start (#4146). Either way it must be stable for the bank, because the
 * clause is PATCHed onto pages that outlive the session.
 *
 * A bank collects everything said IN a repository, which is NOT the same as everything said ABOUT
 * it: a repo that reads its dependency's source, drafts its upstream issues, or documents how it
 * configures a service files those facts here too — correctly, since that is where the work
 * happened. Nothing downstream can tell the two apart. Attribution tags (`project:`, `harness:`,
 * `workspace:`) record where a fact ARRIVED from, never what it is ABOUT, and by synthesis time the
 * source document is gone: the fact reads as a bare technical decision with no hint whose codebase
 * it belongs to. So the page builder answered "what are this project's key decisions?" over
 * everything the bank held and presented a dependency's decisions as the repo's own, upstream
 * commit SHAs and all (#3476).
 *
 * Naming the repo and stating the exclusion is what lets the synthesizer make that call while it
 * still has the fact's text in front of it. It rides on `source_query` rather than the bank's
 * `reflect_mission` because the mission is seeded ONCE and then belongs to whoever set it
 * (`codingBankManifest`, #2492) — a mission-only fix would never reach an existing bank, while a
 * reworded query re-syncs through `seedPages()`'s drift PATCH on the next run.
 */
export function pageScopeRule(project: string): string {
  return (
    ` Scope this page to ${project} ITSELF: the bank also holds facts about external tools, ` +
    `libraries and services that ${project} merely uses, configures, deploys or discusses, and ` +
    `those belong to somebody else's codebase. Include something only when its subject is ` +
    `${project}'s own code, configuration or process; when it is about a dependency, leave it out ` +
    `however well-evidenced it looks — including any commit SHA or identifier that belongs to that ` +
    `dependency's repository rather than this one.`
  );
}

/** The taxonomy before scoping — never seeded directly; `pagesFor` binds it to a repository. */
const PAGE_TAXONOMY: readonly KnowledgePage[] = [
  {
    name: "Component map",
    source_query:
      "From this project's commit history and past discussions, what are the main " +
      "components/modules/subsystems, what is each responsible for, and how do they relate to or " +
      "depend on one another? Describe the structure and responsibilities.",
    tags: ["knowledge:component"],
  },
  {
    name: "Core concepts",
    source_query:
      "What are the core concepts, domain abstractions, and key entities in this project — " +
      "the vocabulary a developer must understand? For each, explain what it represents and its role, " +
      "drawn from how they are introduced and discussed across the history and conversations.",
    tags: ["knowledge:concept"],
  },
  {
    name: "Conventions and patterns",
    source_query:
      "What conventions, idioms, and recurring patterns does this project follow — its " +
      "approach to testing, error handling, naming, structure, and how changes are typically made? " +
      "Describe how THIS project does things, as evidenced across its history and discussions.",
    tags: ["knowledge:convention"],
  },
  {
    name: "Key decisions and rationale",
    source_query:
      "What are the significant technical decisions made in this project and the rationale " +
      "behind them — the durable 'why we do it this way' a developer should know? Summarize the " +
      "decisions and their reasoning from the commit rationales and past conversations.",
    tags: ["knowledge:decision"],
  },
  {
    name: "Initiatives and enhancements",
    source_query:
      "Based on this repository's commit history, what are the major initiatives, features, and " +
      "enhancements the project has worked on? Summarize the themes and notable changes over time. " +
      "When a source memory's context carries a `[[page:<id>]]` link, repeat that link in the " +
      "summary of what it describes, so each initiative links to its detailed page.",
    tags: ["knowledge:feature-work"],
  },
];

/**
 * The seeded pages for one subject: the taxonomy above with `project` named in every query.
 *
 * A pure function of `project`, so the query text is STABLE for a given subject and `seedPages()`
 * PATCHes once (on the upgrade that introduces the clause) rather than on every deepen run — which
 * holds only while the caller's `project` is itself stable per bank (see `bankProjectName`).
 */
export function pagesFor(project: string): KnowledgePage[] {
  const scope = pageScopeRule(project);
  return PAGE_TAXONOMY.map((page) => ({ ...page, source_query: page.source_query + scope }));
}

// Refresh policy shared by every page this plugin creates — the seeded taxonomy above and the
// per-initiative pages `captureInitiative` adds.
export const PAGE_MAX_TOKENS = 4096;

/** A page's `trigger`, in the API's own shape (see MentalModelTrigger in api/http.py). */
export interface PageTrigger {
  fact_types: string[];
  /** How the page's own `tags` filter the memories a refresh reads. See `PAGE_TAGS_MATCH`. */
  tags_match: "any" | "all" | "any_strict" | "all_strict" | "exact";
  refresh_after_consolidation?: boolean;
  /** `null` CLEARS a schedule the page already has. The server drops an unstated counterpart only
   *  for a truthy field, so `{refresh_after_consolidation: false}` alone would leave a cron in
   *  place and the page would keep refreshing — see `pageTriggerPatch`. */
  refresh_cron?: string | null;
}

/**
 * `all` — AND over the page's tier tag, but INCLUDING untagged memories.
 *
 * The server defaults a tagged model to `all_strict`, which excludes untagged memories, and every
 * observation in these banks is untagged: `DEFAULT_OBSERVATION_SCOPES = "shared"` consolidates into
 * the single empty scope on purpose (#3564), so the consolidated tier carries no tags to match on.
 * Under `all_strict` that put `"observation"` in `PAGE_FACT_TYPES` and the consolidated beliefs it
 * asks for on opposite sides of the filter: pages declared the tier and could never retrieve a
 * single row of it, synthesizing from raw world/experience facts alone.
 *
 * `all` keeps the discrimination that matters — a `knowledge:decision` fact still cannot reach the
 * Component map, since it is tagged and lacks that page's tag — while letting the untagged shared
 * observations reach every page, where the page's `source_query` is what selects among them. That
 * is what one shared belief pool means.
 */
const PAGE_TAGS_MATCH = "all" as const;

/** A page synthesizes from all three tiers; the fact types are not a preference. */
export const PAGE_FACT_TYPES = ["world", "experience", "observation"];

/**
 * The default schedule: once an hour, each page on its own hashed minute (see `H` below).
 *
 * Hourly rather than daily because a knowledge page a coding agent reads at the start of a session
 * is worth little if it lags a day behind the repo; hourly rather than per-consolidation because a
 * refresh costs one LLM synthesis per page, and a repo under active work consolidates far more
 * often than once an hour. The server skips a tick that has nothing new to fold in, so an idle
 * repo pays nothing for the schedule.
 */
export const DEFAULT_PAGE_TRIGGER_CRON = "H * * * *";

/** The config fields that shape the trigger (a subset of Config — see core/config.ts). */
export interface PageTriggerConfig {
  pageTriggerType?: "auto-refresh" | "cron" | "manual";
  pageTriggerCron?: string;
}

// ── hashed cron fields (`H`) ───────────────────────────────────────────────────
/**
 * A cron field written `H` means "pick a value in this field's range by hashing the page", so
 * every page gets its OWN stable slot instead of the one the config literally names.
 *
 * One `pageTriggerCron` is shared by every page in every bank running this plugin — it ships as a
 * single documented example and is copied verbatim. A literal `"0 3 * * *"` therefore does not
 * schedule a refresh at 03:00; it schedules ALL of them at 03:00, on the worker pool that also
 * serves retain, so a session ingesting at 03:0x queues behind ~5 page syntheses per bank that
 * happened to share the one minute the docs suggested. Moving the hour moves the pile.
 *
 * `H` is Jenkins' syntax for exactly this problem, borrowed rather than invented because it is
 * already recognisable, and it composes with the rest of the expression instead of replacing it:
 *
 *   "H H * * *"      once a day, at this page's own minute and hour
 *   "H * * * *"      once an hour, at this page's own minute
 *   "H 3 * * *"      daily at 03:MM — spread within the hour the operator chose
 *   "H H(0-5) * * *" daily, spread across the night only
 *   "0 3 * * *"      unchanged: no `H`, no hashing, exactly what it says
 *
 * The alternative — one enum member per period (`daily-staggered`, then `hourly-staggered`, then
 * whatever is asked for next) — spells the schedule in the type name, so every new period is a new
 * config value, a new branch, and a new row of docs. Spreading is a property of the SCHEDULE, so it
 * belongs in the expression.
 *
 * `H` never leaves this package: `expandCronHash` resolves it to an ordinary 5-field expression
 * before the trigger is sent, because `refresh_cron` is parsed server-side as standard cron.
 */
const CRON_FIELD_RANGES: readonly (readonly [number, number])[] = [
  [0, 59], // minute
  [0, 23], // hour
  [1, 31], // day of month
  [1, 12], // month
  [0, 6], // day of week
];

const HASHED_FIELD = /^H(?:\((\d+)-(\d+)\))?$/;

/**
 * Does this expression ask for hashing at all? Plain crons take every path below unchanged.
 *
 * Any field STARTING with `H` counts, not just a well-formed one: no standard cron field begins
 * with `H` (values are digits, `*`, `,`, `-`, `/`, and the JAN-DEC/SUN-SAT names), so `"Hx"` is a
 * typo in this package's syntax rather than something the server was going to accept. Claiming it
 * here is what gets it reported as a malformed hashed field instead of an opaque cron parse error.
 */
export function isHashedCron(cron: string): boolean {
  return /(^|\s)H/.test(cron);
}

/**
 * The five fields of `cron` when every `H` in it is well-formed, else `undefined`.
 *
 * Only the `H` fields are checked. The rest are the server's to validate, as they already are —
 * this package does not own cron syntax, only the extension it adds to it.
 */
export function parseHashedCron(cron: string): string[] | undefined {
  const fields = cron.trim().split(/\s+/);
  if (fields.length !== CRON_FIELD_RANGES.length) return undefined;
  for (const [i, field] of fields.entries()) {
    if (!field.startsWith("H")) continue;
    const m = HASHED_FIELD.exec(field);
    if (!m) return undefined;
    if (m[1] === undefined) continue;
    const [lo, hi] = [Number(m[1]), Number(m[2])];
    const [min, max] = CRON_FIELD_RANGES[i];
    if (lo > hi || lo < min || hi > max) return undefined;
  }
  return fields;
}

/**
 * `seed`'s own value in `[lo, hi]` — stable across machines, processes and releases.
 *
 * The field index is hashed alongside the seed so `H H * * *` does not derive its minute and its
 * hour from one number: the two would move together across pages, collapsing the 1440 daily slots
 * the expression offers back towards 60.
 */
function hashedValue(seed: string, field: number, lo: number, hi: number): number {
  const digest = createHash("sha256").update(`${seed}\u0000${field}`).digest();
  return lo + (digest.readUInt32BE(0) % (hi - lo + 1));
}

/**
 * `cron` with each `H` replaced by `seed`'s own value for that field — an ordinary cron expression.
 *
 * Returns the input untouched when it holds no `H`, and when an `H` in it is malformed: a bad
 * expression is reported by the server that parses crons, not silently rewritten into a valid one
 * that runs at a time nobody asked for. `resolvePageTriggerType` rejects it before it gets here.
 */
export function expandCronHash(cron: string, seed: string): string {
  if (!isHashedCron(cron)) return cron;
  const fields = parseHashedCron(cron);
  if (!fields) return cron;
  return fields
    .map((field, i) => {
      const m = HASHED_FIELD.exec(field);
      if (!m) return field;
      const [lo, hi] =
        m[1] === undefined ? CRON_FIELD_RANGES[i] : ([Number(m[1]), Number(m[2])] as const);
      return String(hashedValue(seed, i, lo, hi));
    })
    .join(" ");
}

/**
 * `trigger` as it should be sent for ONE page, resolving any `H` against that page's identity.
 *
 * Applied where a page is created rather than where the trigger is built, because that is the only
 * place the identity exists: `buildPageTrigger` runs once per session for all of them.
 *
 * The seed is bank + page name — the pair that identifies a page across runs — so a page keeps its
 * slot for as long as it keeps its name, and two banks seeded from the same config land on
 * different ones. Hashing distributes; it does not partition, so two pages CAN still collide.
 */
export function pageTriggerFor(trigger: PageTrigger, bank: string, page: string): PageTrigger {
  const cron = trigger.refresh_cron;
  if (!cron || !isHashedCron(cron)) return trigger;
  return { ...trigger, refresh_cron: expandCronHash(cron, `${bank}\u0000${page}`) };
}

/**
 * How this project's pages keep themselves current.
 *
 * WHEN is the only part of this that is a preference. The default is `cron` on
 * `DEFAULT_PAGE_TRIGGER_CRON` — hourly, each page on its own hashed minute: current within the
 * hour, and bounded, since the server skips a tick when nothing changed. `auto-refresh`, which
 * every page used to ship with, rebuilds whenever consolidation produced new material — the most
 * current setting and by far the most expensive, since a busy repo consolidates constantly and
 * each pass is an LLM synthesis per page (#3506). `manual` refreshes only when something asks. A
 * page is a mental model like any other, so the scheduler picks it up either way
 * (`mental_models_with_cron()` filters on nothing but a non-empty `refresh_cron`).
 *
 * HOW a page refreshes is deliberately NOT stated here. `create_knowledge_page` owns that
 * (`KNOWLEDGE_PAGE_DEFAULT_TRIGGER`: delta refresh, no sibling pages in the reflect loop) and
 * merges a client's fields over it, so this sends only what it actually means and inherits the
 * rest. Restating the server's own defaults here would just freeze a copy of them that drifts the
 * next time they change.
 *
 * `fact_types` IS ours to state: the server's page default is observation-only, while these pages
 * are tag-scoped syntheses over the `knowledge:<tier>` labels the extractor puts on world and
 * experience facts. So is `tags_match`, for the reason `PAGE_TAGS_MATCH` gives.
 *
 * `refresh_after_consolidation` and `refresh_cron` are mutually exclusive server-side, so exactly
 * one of them is ever set here.
 */
export function buildPageTrigger(cfg: PageTriggerConfig = {}): PageTrigger {
  const base: PageTrigger = { fact_types: PAGE_FACT_TYPES, tags_match: PAGE_TAGS_MATCH };
  switch (cfg.pageTriggerType) {
    case "auto-refresh":
      return { ...base, refresh_after_consolidation: true };
    case "manual":
      return { ...base, refresh_after_consolidation: false };
    // "cron" and an unset type alike: the default schedule stands in for a missing expression, so
    // a trigger built from a partial config is never a cron trigger with nothing to fire on.
    default:
      return { ...base, refresh_cron: cfg.pageTriggerCron || DEFAULT_PAGE_TRIGGER_CRON };
  }
}

/** A page's refresh policy as the tree reports it — the EFFECTIVE one, defaults filled in. */
export interface CurrentPageTrigger {
  tags_match?: string;
  refresh_after_consolidation?: boolean;
  refresh_cron?: string | null;
}

/**
 * Has an existing page's refresh policy drifted from what this config asks for?
 *
 * Compared against the page's OWN resolved trigger (`pageTriggerFor`), not the shared one: under a
 * hashed cron every page has a different expression, and comparing the unresolved `H * * * *`
 * would report drift on every page on every session.
 *
 * Only the fields this plugin actually states are compared. Everything else on the trigger —
 * `mode`, sibling exclusion, `min_refresh_interval_seconds` — is the server's or the operator's,
 * and a re-sync must not have an opinion about it (#3506).
 */
export function pageTriggerDrifted(current: CurrentPageTrigger, desired: PageTrigger): boolean {
  return (
    current.tags_match !== desired.tags_match ||
    (current.refresh_cron ?? null) !== (desired.refresh_cron ?? null) ||
    Boolean(current.refresh_after_consolidation) !== Boolean(desired.refresh_after_consolidation)
  );
}

/**
 * The trigger to PATCH onto an existing page, given the one we would create it with.
 *
 * The server merges a trigger patch field by field and drops the unstated counterpart of a TRUTHY
 * refresh field, so a cron patch clears auto-refresh and vice versa. `manual` is the gap: its
 * `refresh_after_consolidation: false` is falsy, nothing is dropped, and a page that had a cron
 * would keep firing on it. Stating `refresh_cron: null` closes that.
 */
export function pageTriggerPatch(desired: PageTrigger): PageTrigger {
  if (desired.refresh_after_consolidation === false) return { ...desired, refresh_cron: null };
  return desired;
}

// ── the bank template ──────────────────────────────────────────────────────────
// The bank's CONFIG — missions, retain strategies, entity labels — as a single manifest for
// POST /banks/{id}/import. Idempotent (config fields apply as per-bank overrides), so the deepen
// engine can apply it every run. Replaces the old configureBank PUT+PATCH.
//
// The seeded pages are NOT part of this manifest: the template's `mental_models` key creates bare
// mental models with no knowledge-base node, which leaves them invisible to page search (it joins
// through `knowledge_pages`) and unreadable by node id. Pages are seeded separately, through the
// knowledge-base API, by `HindsightClient.seedPages()`.
export const CODING_BANK_TEMPLATE = {
  version: "1",
  bank: {
    reflect_mission: REFLECT_MISSION,
    enable_observations: true,
    observations_mission: OBSERVATIONS_MISSION,
    retain_mission: GIT_MISSION,
    retain_extraction_mode: "verbose",
    retain_default_strategy: "git",
    retain_strategies: RETAIN_STRATEGIES,
    entity_labels: [KNOWLEDGE_LABELS],
    entities_allow_free_form: true,
  },
} as const;

/** Bank-level missions the template seeds ONCE and then leaves alone (#2492). */
const MISSION_FIELDS = ["reflect_mission", "retain_mission", "observations_mission"] as const;

/** Bank-scoped config OVERRIDES exactly as `GET /banks/{id}/config` reports them. */
export type BankOverrides = Record<string, unknown>;

/** The manifest `configureBank` POSTs to `/banks/{id}/import`. */
export interface BankManifest {
  version: "1";
  bank: Record<string, unknown>;
}

/** A bank-config override the bank's owner actually made. Blank is not a choice; `false` is. */
function isSet(v: unknown): boolean {
  if (v === null || v === undefined) return false;
  if (typeof v === "string") return v.trim() !== "";
  return true;
}

/**
 * The template fields still missing from a bank whose overrides are `overrides` — or `undefined`
 * when it already carries them all and there is nothing to write.
 *
 * **This plugin only ever ADDS what is missing; it never overwrites what the bank already says.**
 *
 * Re-applying the whole template on every pass is how a plugin takes a bank over, and it has now
 * been fixed three times over the same shape: #1270 for OpenClaw's missions, #2492 for this
 * plugin's, and #3927 for everything those two left un-guarded. Each earlier fix protected only the
 * fields that had just been noticed, so the rest kept being stamped back on every session start.
 * Reading the current overrides and writing only where the bank is silent covers the whole surface
 * at once, including whatever gets added to the template next.
 *
 * The two container fields merge PER ENTRY, because the server stores each as ONE config value and
 * an import replaces it outright: re-sending the five strategies wholesale deleted any strategy the
 * user had defined, reverted their edits to the plugin's own (mission, extraction mode, chunk
 * size), and could leave `retain_default_strategy` naming a strategy that no longer existed.
 * Merging still lets a strategy ADDED by a newer plugin release reach an existing bank — the reason
 * the re-apply exists — without touching the entries already there.
 *
 * The consequence is deliberate: a release that REWORDS an existing strategy or label does not
 * reach a bank that already has it. Clearing that override on the bank takes the current default
 * back, since the next pass then finds the bank silent there.
 */
export function codingBankManifest(overrides: BankOverrides | undefined): BankManifest | undefined {
  // Unreadable overrides — the bank does not exist yet, or the deployment has the bank-config API
  // switched off. Nothing can have been customised through an API that is not there, and this same
  // POST is what CREATES the bank, so `{}` seeds the lot (every branch below fires, and the result
  // is CODING_BANK_TEMPLATE — asserted in missions.test.ts).
  const current = overrides ?? {};
  const template = CODING_BANK_TEMPLATE.bank;
  const bank: Record<string, unknown> = {};

  // The missions are seeded as a GROUP: the plugin writes all three together, so any one of them
  // being present means this bank has been seeded already, or its owner wrote their own (#2492).
  if (!MISSION_FIELDS.some((f) => isSet(current[f]))) {
    bank.reflect_mission = template.reflect_mission;
    bank.enable_observations = template.enable_observations;
    bank.observations_mission = template.observations_mission;
    bank.retain_mission = template.retain_mission;
    bank.retain_extraction_mode = template.retain_extraction_mode;
  }

  if (!isSet(current.retain_default_strategy))
    bank.retain_default_strategy = template.retain_default_strategy;
  if (!isSet(current.entities_allow_free_form))
    bank.entities_allow_free_form = template.entities_allow_free_form;

  const strategies =
    current.retain_strategies && typeof current.retain_strategies === "object"
      ? (current.retain_strategies as Record<string, unknown>)
      : {};
  const missing = Object.entries(template.retain_strategies).filter(([n]) => !(n in strategies));
  // The whole map is one config value, so the UNION has to be sent — not just the additions.
  if (missing.length > 0)
    bank.retain_strategies = { ...strategies, ...Object.fromEntries(missing) };

  const labels = Array.isArray(current.entity_labels) ? current.entity_labels : [];
  const [knowledgeGroup] = template.entity_labels;
  if (!labels.some((g) => (g as { key?: unknown } | null)?.key === knowledgeGroup.key))
    bank.entity_labels = [...labels, knowledgeGroup];

  return Object.keys(bank).length > 0 ? { version: "1", bank } : undefined;
}
