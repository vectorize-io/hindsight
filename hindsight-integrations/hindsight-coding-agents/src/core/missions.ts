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
  "both WHAT changed and WHY.";

export const CHAT_MISSION =
  "You are ingesting a raw developer conversation (a JSON user/assistant transcript). Extract the "
  + "FEWEST facts that capture the OUTCOME — do NOT emit one fact per message, per intermediate "
  + "proposal, or per tool step; that fragments the decision and reads as contradictory out of order. "
  + "Prefer: (1) ONE consolidated fact stating the FINAL, settled decision and its exact rule/values "
  + "unambiguously; and (2) at most one fact for the key alternative that was REJECTED and why. "
  + "CRITICAL: a conversation usually REVISES its answer — an early proposal gets changed. Record ONLY "
  + "the FINAL state as the decision / what is in effect. A superseded proposal must appear ONLY inside "
  + "the rejected fact ('initially proposed X, changed to Y because…'), NEVER as its own 'decided' "
  + "fact. If the same setting changes several times, keep only the LAST. Make unmistakably clear which "
  + "choice WON. Quote literal values/identifiers verbatim. Preserve the 'REF-ID: <token>' marker in "
  + "each fact. Do not invent; capture only what was actually settled.";

export const REFLECT_MISSION =
  "You are a debugging assistant with the project's past decisions in memory (git rationale and " +
  "developer chats). Given a bug's SYMPTOM, find the past decision whose rationale explains the ROOT " +
  "CAUSE — not one that merely shares vocabulary. Answer with the PRECISE fix: state the EXACT rule " +
  "and the LITERAL values, identifiers, strings, numbers, or set members that were decided — quote " +
  "them VERBATIM, never paraphrase, generalize, or omit them (give the actual decided value, not " +
  "'the project standard'). Name the function/file to change and cite the REF-ID(s).";

export const OBSERVATIONS_MISSION =
  "Consolidate durable knowledge about THIS codebase — recurring patterns, conventions, module "
  + "responsibilities, and how components relate — from the ingested commits and conversations. "
  + "Favor stable structural understanding over one-off details.";

// CUSTOM extraction prompt for chats — replaces the default extractor's rules entirely, so we get a
// TINY number of coherent facts (final decision + optional rejection), not a fact per message.
export const CHAT_CUSTOM_INSTRUCTIONS =
  "You are reading ONE developer conversation (JSON user/assistant turns) about a coding decision. It "
  + "typically PROPOSES options and then REVISES them — only the LAST state is real.\n\n"
  + "Extract AT MOST 2 facts:\n"
  + "1. THE DECISION — a single fact stating the FINAL, in-effect rule and its EXACT values/identifiers, "
  + "unambiguously (e.g. \"the client pins API version v3 via the X-Api-Version header, not the URL path, "
  + "as settled after the gateway migration\"). Quote literals verbatim.\n"
  + "2. THE REJECTION (only if a notable alternative was tried) — one fact of the form \"initially "
  + "proposed X, but changed to Y because Z\".\n\n"
  + "HARD RULES:\n"
  + "- NEVER emit a separate fact per message, per intermediate proposal, or per tool step.\n"
  + "- A superseded proposal appears ONLY inside fact #2 — NEVER as its own 'decided' fact.\n"
  + "- If a setting changed several times, keep ONLY the last as the decision.\n"
  + "- Emit just 1 fact when there is no meaningful rejected alternative.\n"
  + "- Preserve the 'REF-ID: <token>' marker from the transcript in each fact. Do not invent.";

export const RETAIN_STRATEGIES = {
  git: { retain_mission: GIT_MISSION, retain_extraction_mode: "verbose" },
  // chunk big enough to hold a WHOLE typical chat in ONE chunk (these run ~2.5k tokens / ~10k chars;
  // the 3000 default was SPLITTING them -> per-chunk fragments). ~12k stays well under a 16k-context
  // model, so the custom "≤2 facts" prompt sees the full proposal→revision arc and emits the final
  // decision. (Very long chats would still split and fall back to the consolidation layer.)
  chat: { retain_extraction_mode: "custom", retain_custom_instructions: CHAT_CUSTOM_INSTRUCTIONS,
          retain_chunk_size: 12000 },
} as const;

// Knowledge PAGES (OKF pages = mental models) = a developer's durable mental model of the codebase,
// CONSOLIDATED from the ingested MEMORY (commit history + past conversations) — NOT mirrored from the
// current source (which would need constant re-sync). A universal 4-page taxonomy that generalizes to
// any repo; the curator populates each from history+chats and can spawn per-component sub-pages.
export const PAGES = [
  { name: "Component map",
    source_query: "From this project's commit history and past discussions, what are the main "
      + "components/modules/subsystems, what is each responsible for, and how do they relate to or "
      + "depend on one another? Describe the structure and responsibilities." },
  { name: "Core concepts",
    source_query: "What are the core concepts, domain abstractions, and key entities in this project — "
      + "the vocabulary a developer must understand? For each, explain what it represents and its role, "
      + "drawn from how they are introduced and discussed across the history and conversations." },
  { name: "Conventions and patterns",
    source_query: "What conventions, idioms, and recurring patterns does this project follow — its "
      + "approach to testing, error handling, naming, structure, and how changes are typically made? "
      + "Describe how THIS project does things, as evidenced across its history and discussions." },
  { name: "Key decisions and rationale",
    source_query: "What are the significant technical decisions made in this project and the rationale "
      + "behind them — the durable 'why we do it this way' a developer should know? Summarize the "
      + "decisions and their reasoning from the commit rationales and past conversations." },
];
