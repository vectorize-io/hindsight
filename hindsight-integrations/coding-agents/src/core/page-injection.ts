import type { Config } from "./config";

export interface PageRef {
  id: string;
  title: string;
}

/** Defensive parse of HindsightClient.listPages() ({items:[{id,name}]}, flattened from the
 *  page tree). The ids are page node ids — the same id space the agent passes
 *  back to hindsight_read_page. */
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

const EMPTY_STATE =
  "No pages yet — Hindsight is still learning this repo; they'll appear as it processes.";

/**
 * What the agent is told EXISTS, without listing it.
 *
 * A full roster of titles and ids reads as an index, and an index makes search
 * pointless: measured over 40 real Claude Code turns against a prefilled bank, every single
 * retrieval was `hindsight_read_page` called with an id copied from this
 * block — 0 searches at 3 pages, and still 0 at 12. The model was right: searching to
 * locate one of a dozen titles already in its context buys nothing.
 *
 * It is the wrong trade anyway. A title says what a page is ABOUT; the search
 * returns the passage that bears on THIS turn, ranked, across pages whose titles
 * give no hint. Naming the count and the way in, rather than the contents, is what
 * makes the first call a search.
 */
function indexLine(pages: PageRef[]): string {
  const count = pages.length === 1 ? "1 page covers" : `${pages.length} pages cover`;
  return (
    `${count} this repository — architecture, conventions, past decisions and ` +
    "in-flight initiatives. They are deliberately NOT listed here: call " +
    "hindsight_search_pages(query) to find the ones that bear on the current turn, then " +
    "hindsight_read_page(<id>) on anything the results show is worth reading in full."
  );
}

/**
 * When-to-call guide for the FULL Hindsight tool suite. Shared by the SessionStart preamble and the
 * periodic refresh so the agent is told — repeatedly — not just that the tools exist but the moment
 * to reach for each one. Registering the tools isn't enough; the trigger for each has to be in
 * context. (Omits hindsight_diagnose — pure troubleshooting, no workflow trigger.)
 *
 * In three parts, because two of them do not always apply: the page tools are left out while the
 * bank has no pages (they would point at nothing), the write tools with `writeTools: false`.
 */
const PAGE_TOOLS_GUIDE =
  "- hindsight_search_pages(query) — FIRST STOP, and the way IN to everything below. The code " +
  "shows what is true today but not what was decided or why; memory shows what was decided or said " +
  "back then but not whether it still holds. Work built from either alone goes wrong: from code alone " +
  "it quietly re-litigates settled questions, from memory alone it acts on stale claims. Search " +
  "BEFORE you act whenever the turn is one of these — they are the ones that go wrong silently:\n" +
  "    • the user reports a bug or a wrong response (the intended behaviour, and the status code or " +
  "value it should return, may already have been decided);\n" +
  "    • you are about to write or change a test (what this project expects a change to ship with, and " +
  "how it asserts, is a convention, not a preference);\n" +
  "    • you are implementing something new, or two parts have to fit together;\n" +
  "    • the user asks why something is the way it is, or what is left to do;\n" +
  "    • you are about to commit, and need to know what the change was supposed to honour.\n" +
  "  It ranks the pages by relevance and shows the opening of each; every page can come back, so " +
  "the ranking, not the list, is the signal. Read the top page with hindsight_read_page before you " +
  "rely on it — the opening is not the passage that matched. What it returns is a past record, " +
  "not a live reading: a claim that something was fixed, " +
  "passes, or works is what someone said then — check it against the code before you rely on it, " +
  "and say so when the two disagree.\n" +
  "  CREDITING IS NOT OPTIONAL AND NOT A JUDGEMENT CALL. If you called this tool and anything it " +
  "returned reached your reply — quoted, paraphrased, or merely confirming what you were about to " +
  "say — open that part with a markdown blockquote, exactly: " +
  '"> 🧠 **From Hindsight memory (<page>)** — <the specific facts you drew on>". Rewriting a ' +
  "snippet in your own words does not make it yours. The one exception: a page whose snippet is " +
  '"No content yet." has nothing in it to credit. A search that turned up nothing useful needs ' +
  "no mention at all — just carry on.\n" +
  "- hindsight_list_pages / hindsight_read_page — BEFORE substantial work, list the pages and " +
  "read the relevant ones to ground yourself in this repo's architecture, conventions, and past decisions instead " +
  "of re-deriving them from the code; follow any [[page:<id>]] links you see.\n";

const REFLECT_GUIDE =
  "- hindsight_reflect(query) — when pages are too shallow and you need the WHY: deep reasoning over the " +
  "repo's full memory for the past decision and exact values that explain a behavior or bug (slower — " +
  'use deliberately, and credit results with a blockquote header "> 🧠 **From Hindsight memory** — <summary>").\n';

/** The two tools that WRITE to the bank: the host may deny them (#6487). */
const WRITE_TOOLS_GUIDE =
  "- hindsight_capture_initiative(title, summary) — right after the user approves a plan or finishes brainstorming a " +
  "new feature/capability and you are about to start implementing (BEFORE you write any code), call this to " +
  "record it as a tracked page; then call it AGAIN with relates_to_page_id set to that page whenever the goal, " +
  "scope, or rationale materially changes mid-work, so the page tracks the current plan and not the opening one. " +
  "Skip bug fixes, small tweaks, chores, and trivial course-corrections.\n" +
  "- hindsight_ingest_document(title, content) — save an external document or durable notes/findings you want " +
  "remembered (not the current conversation — that is captured automatically at session end).";

/**
 * Any autoInject other than "reflect" means no first-prompt synthesis. Keep the pull trigger explicit,
 * but start with the curated pages: they are the fast path, while reflection is the slower fallback
 * when those pages do not contain enough depth for the new goal.
 */
const PAGES_FIRST_ON_GOALS =
  "- The user just set a NEW task or goal → search the pages FIRST with " +
  "hindsight_search_pages. No synthesis is injected automatically in this configuration; " +
  "call hindsight_reflect only when those pages are too shallow and deeper reasoning is needed.\n";

export interface ToolGuideOpts {
  /** Add the new-goal pull trigger (no automatic synthesis: cfg.autoInject !== "reflect"). It used to send
   *  the agent straight to hindsight_reflect; it now goes to the pages first and keeps
   *  reflect for what they don't cover. The field name is unchanged so call sites stay stable. */
  reflectOnNewGoals?: boolean;
  /** cfg.toolGuideExtra: the team's own guidance, added after ours rather than replacing it. */
  extra?: string;
  /** cfg.writeTools: false leaves out every instruction to call a write tool (#6487). */
  writeTools?: boolean;
}

/** The guide options a config implies — one place, so every block a session sees agrees. */
export function toolGuideOpts(cfg: Config): ToolGuideOpts {
  return {
    reflectOnNewGoals: cfg.autoInject !== "reflect",
    extra: cfg.toolGuideExtra,
    writeTools: cfg.writeTools,
  };
}

function toolGuide(pages: PageRef[], opts?: ToolGuideOpts): string {
  const extra = opts?.extra?.trim();
  const has = pages.length > 0;
  const guide =
    (has && opts?.reflectOnNewGoals ? PAGES_FIRST_ON_GOALS : "") +
    (has ? PAGE_TOOLS_GUIDE : "") +
    REFLECT_GUIDE +
    (opts?.writeTools === false ? "" : WRITE_TOOLS_GUIDE);
  return guide.trimEnd() + (extra ? `\n${extra}` : "");
}

/** SessionStart: teach the whole tool suite + when to use each, and list what pages exist. Empty-state aware. */
export function buildPagesPreamble(pages: PageRef[], opts?: ToolGuideOpts): string {
  const body = pages.length ? indexLine(pages) : EMPTY_STATE;
  return (
    "<hindsight_pages>\n" +
    "This repository has a Hindsight memory + pages (curated, continuously-updated pages plus the raw " +
    "memory behind them). The tools below are registered, but you must actually CALL them at the right moments:\n" +
    `${toolGuide(pages, opts)}\n` +
    (opts?.writeTools === false
      ? ""
      : "ALSO your correction tool: when you verify a Hindsight memory is wrong or stale, ingest a " +
        '"Correction: <topic>" doc stating what memory claimed, what is true now, and the evidence — ' +
        "newer facts supersede older ones.\n") +
    `${body}\n` +
    "A short reminder of these tools is re-injected for you periodically as things change.\n" +
    "</hindsight_pages>"
  );
}

/**
 * Periodic UserPromptSubmit refresh. ALWAYS emits (never undefined), keeping the tools in front of
 * the agent even on a fresh repo with no pages yet. The full guide goes out once per session (the
 * caller decides when); every other refresh is `short`: one line, since the model already has the
 * guide and the tool descriptions carry it too — re-sending it every turn was ~5 KB, 90% of every
 * block (#6597). The tag stays the same so hosts that look for it still find it.
 */
export function buildRosterRefresh(
  pages: PageRef[],
  opts?: ToolGuideOpts & { short?: boolean }
): string {
  if (opts?.short) {
    const line = pages.length
      ? `${pages.length === 1 ? "1 Hindsight page covers" : `${pages.length} Hindsight pages cover`} ` +
        "this repo: search them (hindsight_search_pages) before bugs, tests, new work and commits, " +
        "and credit any hit you use."
      : "No Hindsight pages yet; hindsight_reflect answers from this repo's raw memory.";
    return `<hindsight_page_refresh>\n${line}\n</hindsight_page_refresh>`;
  }
  const rosterBlock = pages.length ? `${indexLine(pages)}\n` : "";
  return (
    "<hindsight_page_refresh>\n" +
    rosterBlock +
    "Reminder — this repo's Hindsight tools are available; call them at the right moments:\n" +
    `${toolGuide(pages, opts)}\n` +
    "</hindsight_page_refresh>"
  );
}
