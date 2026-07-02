#!/usr/bin/env node
/**
 * hindsight-coding-backfill — one-shot setup + ingest of a repo's history into a Hindsight bank,
 * for the reflect-only coding plugin.
 *
 * It (1) creates/updates the bank with the reflect mission, observations DISABLED, and two named
 * retain strategies — `git` and `chat` — then (2) ingests EVERY git commit (full message + full
 * diff, no pre-filtering) under the `git` strategy, and (3) ingests each developer conversation
 * RAW (never pre-summarized) under the `chat` strategy. Each item carries a REF-ID tracer so a
 * reflected fact can be traced back to its commit/session.
 *
 * Usage:
 *   hindsight-coding-backfill --repo <path> [--conversations <sessions.json>] --bank <id> \
 *       [--api-url http://localhost:8888] [--api-token X] [--limit N] [--reset] [--concurrency 8]
 *
 * conversations.json: [{ "id": "s1", "turns": [{"role":"user","text":"..."}, ...] }, ...]
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

// ── retain strategies (the two "different strategies" for git vs conversations) ─────────────
const GIT_MISSION =
  "You are ingesting a single git commit: its message and its full diff. Extract the concrete " +
  "technical DECISION and the CAUSE/INVARIANT it encodes, bound to the specific code entities " +
  "(functions, methods, files) and behaviors it changes. Preserve exact identifiers, paths, and " +
  "literal values verbatim. Preserve the 'REF-ID: <token>' marker verbatim in every fact. Capture " +
  "both WHAT changed and WHY.";
const CHAT_MISSION =
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
const REFLECT_MISSION =
  "You are a debugging assistant with the project's past decisions in memory (git rationale and " +
  "developer chats). Given a bug's SYMPTOM, find the past decision whose rationale explains the ROOT " +
  "CAUSE — not one that merely shares vocabulary. Answer with the PRECISE fix: state the EXACT rule " +
  "and the LITERAL values, identifiers, strings, numbers, or set members that were decided — quote " +
  "them VERBATIM, never paraphrase, generalize, or omit them (give the actual mapping/value/threshold, " +
  "e.g. the specific words a symbol maps to or the exact number, not 'the project standard'). Name " +
  "the function/file to change and cite the REF-ID(s).";

// CUSTOM extraction prompt for chats — replaces the default extractor's rules entirely, so we get a
// TINY number of coherent facts (final decision + optional rejection), not a fact per message.
const CHAT_CUSTOM_INSTRUCTIONS =
  "You are reading ONE developer conversation (JSON user/assistant turns) about a coding decision. It "
  + "typically PROPOSES options and then REVISES them — only the LAST state is real.\n\n"
  + "Extract AT MOST 2 facts:\n"
  + "1. THE DECISION — a single fact stating the FINAL, in-effect rule and its EXACT values/identifiers, "
  + "unambiguously (e.g. \"round_cents uses ROUND_HALF_DOWN so half-cents round toward zero, matching the "
  + "legacy ledger\"). Quote literals verbatim.\n"
  + "2. THE REJECTION (only if a notable alternative was tried) — one fact of the form \"initially "
  + "proposed X, but changed to Y because Z\".\n\n"
  + "HARD RULES:\n"
  + "- NEVER emit a separate fact per message, per intermediate proposal, or per tool step.\n"
  + "- A superseded proposal appears ONLY inside fact #2 — NEVER as its own 'decided' fact.\n"
  + "- If a setting changed several times, keep ONLY the last as the decision.\n"
  + "- Emit just 1 fact when there is no meaningful rejected alternative.\n"
  + "- Preserve the 'REF-ID: <token>' marker from the transcript in each fact. Do not invent.";

const RETAIN_STRATEGIES = {
  git: { retain_mission: GIT_MISSION, retain_extraction_mode: "verbose" },
  // chunk big enough to hold a WHOLE typical chat in ONE chunk (these run ~2.5k tokens / ~10k chars;
  // the 3000 default was SPLITTING them -> per-chunk fragments). ~12k stays well under a 16k-context
  // model, so the custom "≤2 facts" prompt sees the full proposal→revision arc and emits the final
  // decision. (Very long chats would still split and fall back to the consolidation layer.)
  chat: { retain_extraction_mode: "custom", retain_custom_instructions: CHAT_CUSTOM_INSTRUCTIONS,
          retain_chunk_size: 12000 },
};

// Knowledge PAGES (OKF pages = mental models) = a developer's durable mental model of the codebase,
// CONSOLIDATED from the ingested MEMORY (commit history + past conversations) — NOT mirrored from the
// current source (which would need constant re-sync). A universal 4-page taxonomy that generalizes to
// any repo; the curator populates each from history+chats and can spawn per-component sub-pages.
const PAGES = [
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

// ── args ────────────────────────────────────────────────────────────────────
function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return process.argv.includes(`--${name}`) ? "true" : def;
}
const REPO = arg("repo");
const CONV = arg("conversations");
const BANK = arg("bank");
const API_URL = (arg("api-url", "http://localhost:8888") as string).replace(/\/$/, "");
const API_TOKEN = arg("api-token");
const LIMIT = arg("limit") ? Number(arg("limit")) : undefined;
const RESET = process.argv.includes("--reset");
const NO_PAGES = process.argv.includes("--no-pages");
const CONCURRENCY = Number(arg("concurrency", "8"));

if (!REPO || !BANK) {
  console.error("usage: hindsight-coding-backfill --repo <path> --bank <id> [--conversations f.json] [--api-url U] [--limit N] [--reset] [--no-pages]");
  process.exit(1);
}

// ── Hindsight HTTP helpers (raw fetch — no client dep) ────────────────────────
const H = (): Record<string, string> => {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (API_TOKEN) h["Authorization"] = `Bearer ${API_TOKEN}`;
  return h;
};
const bankUrl = (suffix = "") => `${API_URL}/v1/default/banks/${encodeURIComponent(BANK!)}${suffix}`;

async function req(method: string, url: string, body?: unknown): Promise<Response> {
  const r = await fetch(url, { method, headers: H(), body: body ? JSON.stringify(body) : undefined });
  if (!r.ok && r.status !== 404) throw new Error(`${method} ${url} -> ${r.status} ${await r.text()}`);
  return r;
}

async function configureBank() {
  if (RESET) {
    await req("DELETE", bankUrl());
    console.log(`[bank] reset ${BANK}`);
  }
  // base config: reflect mission + observations ON (consolidated patterns enrich pages) + git default
  await req("PUT", bankUrl(), {
    name: BANK,
    reflect_mission: REFLECT_MISSION,
    enable_observations: true,
    observations_mission: "Consolidate durable knowledge about THIS codebase — recurring patterns, "
      + "conventions, module responsibilities, and how components relate — from the ingested commits "
      + "and conversations. Favor stable structural understanding over one-off details.",
    retain_mission: GIT_MISSION,
    retain_extraction_mode: "verbose",
  });
  // named strategies (git / chat) — per-item strategy overrides the default
  await req("PATCH", bankUrl("/config"), {
    updates: { retain_strategies: RETAIN_STRATEGIES, retain_default_strategy: "git" },
  });
  console.log(`[bank] configured ${BANK}: reflect mission set, observations OFF, strategies {git, chat}`);
}

const opIds: string[] = [];

async function retain(content: string, context: string, documentId: string, tags: string[],
                      strategy: string, opts: { timestamp?: string; metadata?: Record<string, string> } = {}) {
  // ASYNC: enqueue extraction server-side and return immediately. Sync retain blocks on the
  // extraction LLM and times out under parallel load / large diffs; async decouples them.
  const item: Record<string, unknown> = { content, context, document_id: documentId, tags, strategy };
  if (opts.timestamp) item.timestamp = opts.timestamp;         // when the content occurred (temporal ranking)
  if (opts.metadata) item.metadata = opts.metadata;            // source provenance (returned with recalls)
  const r = await req("POST", bankUrl("/memories"), { items: [item], async: true });
  try {
    const j = (await r.json()) as { operation_id?: string };
    if (j.operation_id) opIds.push(j.operation_id);
  } catch {
    /* ignore */
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let failures = 0;

async function pool<T>(items: T[], n: number, fn: (x: T, i: number) => Promise<void>) {
  let i = 0, done = 0;
  async function worker() {
    while (i < items.length) {
      const idx = i++;
      try {
        await fn(items[idx], idx);
      } catch (e) {
        failures++;
        console.warn(`  ! item ${idx} failed to enqueue: ${(e as Error).message?.slice(0, 120)}`);
      }
      if (++done % 25 === 0) console.log(`  ${done}/${items.length}`);
    }
  }
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, worker));
}

async function drain(ids: string[], label: string, maxMs = 60 * 60 * 1000) {
  // Poll each enqueued operation by id until terminal. The /operations LIST endpoint only shows
  // active ops (completed ones drop off), so per-id GET is the reliable "done" signal.
  if (!ids.length) return;
  console.log(`[wait] draining ${ids.length} ${label} operations …`);
  const start = Date.now();
  const TERMINAL = new Set(["completed", "failed", "cancelled", "error"]);
  const pending = new Set(ids);
  let failed = 0;
  while (pending.size && Date.now() - start < maxMs) {
    await Promise.all(
      [...pending].map(async (id) => {
        try {
          const r = await fetch(bankUrl(`/operations/${id}`), { headers: H() });
          if (!r.ok) return;
          const st = (((await r.json()) as { status?: string }).status || "").toLowerCase();
          if (TERMINAL.has(st)) {
            pending.delete(id);
            if (st !== "completed") failed++;
          }
        } catch {
          /* transient — retry next cycle */
        }
      }),
    );
    if (pending.size) {
      console.log(`  … ${pending.size}/${ids.length} ${label} ops pending`);
      await sleep(5000);
    }
  }
  console.log(`[wait] ${label} drained — ${ids.length - pending.size} done, ${failed} failed` +
              (pending.size ? `, ${pending.size} still pending at timeout` : ""));
}

async function createPages() {
  // Knowledge pages generate by reflecting over the EXTRACTED facts, so this must run AFTER the
  // git+chat extraction has drained. Each page becomes a mental model reflect consults first.
  console.log(`[pages] creating ${PAGES.length} knowledge pages …`);
  const pageOps: string[] = [];
  for (const p of PAGES) {
    try {
      // fact_types = ALL (world+experience+observation) so a page draws from raw facts AND
      // consolidated observations, not a narrow default; refresh after consolidation keeps it live.
      const body = { ...p, trigger: { fact_types: ["world", "experience", "observation"],
                                       refresh_after_consolidation: true } };
      const r = await req("POST", bankUrl("/knowledge-base/pages"), body);
      const j = (await r.json()) as { operation_id?: string; page_id?: string };
      if (j.operation_id) pageOps.push(j.operation_id);
      console.log(`  created page '${p.name}' -> ${j.page_id || "?"}`);
    } catch (e) {
      console.warn(`  ! page '${p.name}' failed: ${(e as Error).message?.slice(0, 140)}`);
    }
  }
  await drain(pageOps, "page-generation");
}

// ── git ───────────────────────────────────────────────────────────────────────
function git(...args: string[]): string {
  return execFileSync("git", ["-C", REPO!, ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
}

async function ingestGit() {
  // NEWEST-first: recent commits (the project's own decision commits) extract before the ancient
  // upstream noise, so the decisions that matter aren't starved at the tail of the extraction queue.
  let shas = git("rev-list", "HEAD").trim().split("\n").filter(Boolean);
  if (LIMIT) shas = shas.slice(0, LIMIT); // most recent N (validate the machine on a slice first)
  const repoName = REPO!.replace(/\/+$/, "").split("/").pop() || "repo";
  console.log(`[git] ingesting ${shas.length} commits (full message + full diff, no filter) …`);
  const US = "\x1f";
  await pool(shas, CONCURRENCY, async (sha) => {
    // one call for the commit header (everything git gives for free) + subject + body
    const [h, an, ae, aISO, cISO, subj, body] =
      (git("show", "-s", `--format=%H${US}%an${US}%ae${US}%aI${US}%cI${US}%s${US}%b`, sha)).split(US);
    const msg = (subj + (body?.trim() ? "\n\n" + body.trim() : "")).trim();
    const diff = git("show", "--format=", sha); // FULL diff, uncapped
    const content =
      `REF-ID: git:${sha.slice(0, 12)}\n` +
      `Git commit ${sha.slice(0, 12)} in the ${repoName} repository (${an}, ${aISO}).\n\n` +
      `Message:\n${msg}\n\nDiff:\n${diff}`;
    await retain(content, `git commit in ${repoName}`, `git:${sha}`, ["source:git"], "git", {
      timestamp: aISO,   // set the memory's timestamp to the commit's author date
      metadata: { source: "git", repo: repoName, commit: h, short_sha: sha.slice(0, 12),
                  author: an, author_email: ae, authored_at: aISO, committed_at: cISO, subject: subj },
    });
  });
  console.log(`[git] done: ${shas.length} commits ingested under strategy 'git'`);
}

// ── conversations ──────────────────────────────────────────────────────────────
async function ingestChats() {
  if (!CONV) {
    console.log("[chat] no --conversations file; skipping");
    return;
  }
  type Turn = { role: string; text: string; timestamp?: string };
  type Session = { id?: string; turns: Turn[] };
  const sessions = JSON.parse(readFileSync(CONV, "utf8")) as Session[];
  console.log(`[chat] ingesting ${sessions.length} chats (RAW, JSON user/assistant transcript) …`);
  const NOW = Date.now(); // anchor synthesized times to a real, ABSOLUTE clock (not a fabricated epoch)
  await pool(sessions, CONCURRENCY, async (s, i) => {
    const id = s.id || `s${i}`;
    // each turn gets an ABSOLUTE timestamp. Use the turn's own if provided; otherwise synthesize from
    // the real current time, staggered per session (1h back each) + 1 min/turn to preserve ordering.
    const sessBase = NOW - i * 3600000;
    const ts = (t: { timestamp?: string }, j: number) => t.timestamp || new Date(sessBase + j * 60000).toISOString();
    // JSON conversation format (Hindsight-preferred). Leading system turn carries the REF-ID tracer.
    const turns = [{ role: "system", content: `REF-ID: chat:${id}`, timestamp: new Date(sessBase).toISOString() },
                   ...(s.turns || []).map((t, j) => ({ role: t.role, content: t.text, timestamp: ts(t, j + 1) }))];
    await retain(JSON.stringify(turns), "developer chat", `chat:${id}`, ["source:chat"], "chat", {
      timestamp: new Date(sessBase).toISOString(),
      metadata: { source: "chat", chat: id, ref_id: `chat:${id}` },
    });
  });
  console.log(`[chat] done: ${sessions.length} chats ingested (JSON) under strategy 'chat'`);
}

async function main() {
  console.log(`hindsight-coding-backfill -> ${API_URL} bank=${BANK}`);
  await configureBank();
  // chats FIRST: they're few and carry the decisions that make memory necessary — ingesting them
  // before the (large) git flood keeps them from being starved in the server's extraction queue.
  await ingestChats();
  await ingestGit();
  await drain(opIds, "extraction");
  // knowledge pages are synthesized from the extracted facts, so create them AFTER the drain.
  if (NO_PAGES) console.log("[pages] skipped (--no-pages)");
  else await createPages();
  console.log(`\n✅ backfill complete${failures ? ` (${failures} items failed to enqueue)` : ""}. ` +
              "Point the plugin at this bank via HINDSIGHT_BANK_ID.");
}

main().catch((e) => {
  console.error("backfill failed:", e.message || e);
  process.exit(1);
});
