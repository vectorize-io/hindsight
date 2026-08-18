/**
 * Harness-agnostic Hindsight missions, retain strategies, and knowledge-page taxonomy.
 *
 * These describe HOW a coding project's memory is extracted and reasoned over. They are independent
 * of which agent harness produced the sessions, so they live in the shared core and are reused by
 * every harness adapter.
 */

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
 * The subject-scoping clause every seeded page's query carries, naming the repository it is about.
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
 * (CODING_BANK_STRUCTURE, #2492) — a mission-only fix would never reach an existing bank, while a
 * reworded query re-syncs through `seedPages()`'s drift PATCH on the next run.
 */
function pageScopeRule(project: string): string {
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
      "When a source memory carries a tag of the form `relatedPageId:<id>`, include a Markdown link " +
      "`[[page:<id>]]` to that page in the summary, so each initiative links to its detailed page.",
    tags: ["knowledge:feature-work"],
  },
];

/**
 * The seeded pages for one repository: the taxonomy above with `project` named in every query.
 *
 * A pure function of `project`, so the query text is STABLE for a given repo and `seedPages()`
 * PATCHes once (on the upgrade that introduces the clause) rather than on every deepen run.
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
  refresh_after_consolidation?: boolean;
  refresh_cron?: string;
}

/** A page synthesizes from all three tiers; the fact types are not a preference. */
export const PAGE_FACT_TYPES = ["world", "experience", "observation"];

/** The config fields that shape the trigger (a subset of Config — see core/config.ts). */
export interface PageTriggerConfig {
  pageTriggerType?: "auto-refresh" | "cron" | "manual";
  pageTriggerCron?: string;
}

/**
 * How this project's pages keep themselves current.
 *
 * WHEN is the only part of this that is a preference. `auto-refresh` — the default, and what every
 * page shipped with — keeps a living document, rebuilt whenever consolidation produced new
 * material: the most current setting and the most expensive, since a busy repo consolidates
 * constantly and each pass is an LLM synthesis per page (#3506). `cron` bounds that to a schedule
 * (the server skips a tick when nothing changed), `manual` refreshes only when something asks. A page is a mental model like any
 * other, so the scheduler picks it up either way (`mental_models_with_cron()` filters on nothing
 * but a non-empty `refresh_cron`).
 *
 * HOW a page refreshes is deliberately NOT stated here. `create_knowledge_page` owns that
 * (`KNOWLEDGE_PAGE_DEFAULT_TRIGGER`: delta refresh, no sibling pages in the reflect loop) and
 * merges a client's fields over it, so this sends only what it actually means and inherits the
 * rest. Restating the server's own defaults here would just freeze a copy of them that drifts the
 * next time they change.
 *
 * `fact_types` IS ours to state: the server's page default is observation-only, while these pages
 * are tag-scoped syntheses over the `knowledge:<tier>` labels the extractor puts on world and
 * experience facts.
 *
 * `refresh_after_consolidation` and `refresh_cron` are mutually exclusive server-side, so exactly
 * one of them is ever set here.
 */
export function buildPageTrigger(cfg: PageTriggerConfig = {}): PageTrigger {
  const base: PageTrigger = { fact_types: PAGE_FACT_TYPES };
  switch (cfg.pageTriggerType) {
    case "cron":
      return { ...base, refresh_cron: cfg.pageTriggerCron };
    case "manual":
      return { ...base, refresh_after_consolidation: false };
    default:
      return { ...base, refresh_after_consolidation: true };
  }
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

/**
 * The subset re-applied to a bank that is ALREADY configured — everything above minus the missions.
 *
 * The full template seeds a bank once. After that the missions are the user's: someone who rewrites
 * `reflect_mission` in the control plane means it, and re-importing the manifest on every seed pass
 * silently stamped the defaults back over it (#2492 — the same regression #1270 fixed for OpenClaw).
 *
 * The retain strategies and entity labels stay, because they are not preferences: this plugin writes
 * documents under `git` / `gitlog` / `conversation` / `document`, and a bank missing one of those
 * would reject the write. A newer plugin adding a strategy needs it to land on existing banks too.
 */
export const CODING_BANK_STRUCTURE = {
  version: "1",
  bank: {
    retain_strategies: RETAIN_STRATEGIES,
    entity_labels: [KNOWLEDGE_LABELS],
  },
} as const;
