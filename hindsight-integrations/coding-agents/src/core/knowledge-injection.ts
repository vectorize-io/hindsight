export interface PageRef {
  id: string;
  title: string;
}

/** Defensive parse of HindsightClient.listPages() ({items:[{id,name}]}, flattened from the
 *  knowledge-base tree). The ids are knowledge-base node ids — the same id space the agent passes
 *  back to hindsight_read_knowledge_page. */
export function parsePageList(raw: unknown): PageRef[] {
  const items = (raw as { items?: unknown })?.items;
  if (!Array.isArray(items)) return [];
  const out: PageRef[] = [];
  for (const it of items) {
    const id = (it as { id?: unknown })?.id;
    const name = (it as { name?: unknown })?.name;
    if (typeof id === "string" && typeof name === "string") out.push({ id, title: name });
  }
  return out;
}

function roster(pages: PageRef[]): string {
  return pages.map((p) => `- ${p.title} (${p.id})`).join("\n");
}

/**
 * When-to-call guide for the FULL Hindsight tool suite. Shared by the SessionStart preamble and the
 * periodic refresh so the agent is told — repeatedly — not just that the tools exist but the moment
 * to reach for each one. Registering the tools isn't enough; the trigger for each has to be in
 * context. (Omits hindsight_diagnose — pure troubleshooting, no workflow trigger.)
 */
const TOOL_GUIDE =
  "- hindsight_search_knowledge_pages(query) — FIRST STOP for any question the project's accumulated " +
  "knowledge might answer (components, conventions, past decisions, initiatives): search the knowledge " +
  "pages and credit results visibly with a markdown blockquote so it renders as a callout, exactly: " +
  '"> 🧠 **From Hindsight memory (<page>)** — <the specific facts you drew on>".\n' +
  "- hindsight_list_knowledge_pages / hindsight_read_knowledge_page — BEFORE substantial work, list the pages and " +
  "read the relevant ones to ground yourself in this repo's architecture, conventions, and past decisions instead " +
  "of re-deriving them from the code; follow any [[page:<id>]] links you see.\n" +
  "- hindsight_reflect(query) — when pages are too shallow and you need the WHY: deep reasoning over the " +
  "repo's full memory for the past decision and exact values that explain a behavior or bug (slower — " +
  'use deliberately, and credit results with a blockquote header "> 🧠 **From Hindsight memory** — <summary>").\n' +
  "- hindsight_capture_initiative(title, summary) — right after the user approves a plan or finishes brainstorming a " +
  "new feature/capability and you are about to start implementing (BEFORE you write any code), call this to " +
  "record it as a tracked page; then call it AGAIN with relates_to_page_id set to that page whenever the goal, " +
  "scope, or rationale materially changes mid-work, so the page tracks the current plan and not the opening one. " +
  "Skip bug fixes, small tweaks, chores, and trivial course-corrections.\n" +
  "- hindsight_ingest_document(title, content) — save an external document or durable notes/findings you want " +
  "remembered (not the current conversation — that is captured automatically at session end).";

/**
 * autoReflect=false suppresses the injected first-prompt synthesis. Keep the pull trigger explicit,
 * but start with the curated pages: they are the fast path, while reflection is the slower fallback
 * when those pages do not contain enough depth for the new goal.
 */
const PAGES_FIRST_ON_GOALS =
  "- The user just set a NEW task or goal → search the knowledge pages FIRST with " +
  "hindsight_search_knowledge_pages. No synthesis is injected automatically in this configuration; " +
  "call hindsight_reflect only when those pages are too shallow and deeper reasoning is needed.\n";

export interface ToolGuideOpts {
  /** Add the new-goal pull trigger (tool-only reflect mode, cfg.autoReflect=false). It used to send
   *  the agent straight to hindsight_reflect; it now goes to the knowledge pages first and keeps
   *  reflect for what they don't cover. The field name is unchanged so call sites stay stable. */
  reflectOnNewGoals?: boolean;
}

function toolGuide(opts?: ToolGuideOpts): string {
  return (opts?.reflectOnNewGoals ? PAGES_FIRST_ON_GOALS : "") + TOOL_GUIDE;
}

/** SessionStart: teach the whole tool suite + when to use each, and list what pages exist. Empty-state aware. */
export function buildKnowledgePreamble(pages: PageRef[], opts?: ToolGuideOpts): string {
  const body = pages.length
    ? `Knowledge pages currently in this repository:\n${roster(pages)}`
    : "No knowledge pages yet — Hindsight is still learning this repo; they'll appear as it processes.";
  return (
    "<hindsight_knowledge>\n" +
    "This repository has a Hindsight memory + knowledge base (curated, continuously-updated pages plus the raw " +
    "memory behind them). The tools below are registered, but you must actually CALL them at the right moments:\n" +
    `${toolGuide(opts)}\n` +
    "ALSO your correction tool: when you verify a Hindsight memory is wrong or stale, ingest a " +
    '"Correction: <topic>" doc stating what memory claimed, what is true now, and the evidence — ' +
    "newer facts supersede older ones.\n" +
    `${body}\n` +
    "This tool guide and the page list are re-injected for you periodically as things change.\n" +
    "</hindsight_knowledge>"
  );
}

/**
 * Periodic UserPromptSubmit refresh. ALWAYS emits (never undefined) so the full tool guide keeps
 * re-appearing in context even on a fresh repo with no pages yet — precisely when the agent is
 * building its first features. The page roster is included only when pages exist; the reminder of
 * which tools exist and WHEN to call each is unconditional.
 */
export function buildRosterRefresh(pages: PageRef[], opts?: ToolGuideOpts): string {
  const rosterBlock = pages.length
    ? `Current Hindsight knowledge pages (may have changed):\n${roster(pages)}\n`
    : "";
  return (
    "<hindsight_knowledge_refresh>\n" +
    rosterBlock +
    "Reminder — this repo's Hindsight tools are available; call them at the right moments:\n" +
    `${toolGuide(opts)}\n` +
    "</hindsight_knowledge_refresh>"
  );
}
