/**
 * Hindsight Wrapped: a year across every bank, from each bank's stats summary
 * (the server sums the year up; no memory is listed). Pure: the loading is in
 * the component.
 */
import type { StatsSummary } from "./api";
import type { MemoryMap } from "./memory-map";

/** One bank's year: its display name and its stats summary for the year. */
export interface BankYear {
  name: string;
  summary: StatsSummary;
}

export type Persona = "nightOwl" | "earlyBird" | "nineToFive" | "afternoon" | "evening";

export interface Ranked {
  name: string;
  count: number;
}

export interface WrappedStats {
  memories: number;
  /** Banks with at least one memory in the year. */
  banks: number;
  /** By memories in the year, most first. */
  topBanks: Ranked[];
  facts: number;
  observations: number;
  /** Null when no bank's store could count them. */
  documents: number | null;
  activeDays: number;
  longestStreak: number;
  busiestDay: Ranked | null;
  /** Memories per month, January first. */
  bars: number[];
  topEntities: Ranked[];
  topHarness: Ranked | null;
  persona: Persona | null;
  peakHour: number | null;
  /** Memories per local hour of the day, midnight first. */
  hourly: number[];
  /** The observation backed by the most facts. */
  strongestBelief: { text: string; proofs: number } | null;
  /** The earliest memory of the year. */
  firstMemory: { text: string; date: string } | null;
  /** The top banks' memories of the year as a constellation (dots only), when it could be read. */
  map?: MemoryMap | null;
}

/** Whether the server shows Wrapped at all: on unless `HINDSIGHT_CP_WRAPPED_ENABLED` is `false` or `0`. */
export function wrappedEnabled(value = process.env.HINDSIGHT_CP_WRAPPED_ENABLED): boolean {
  return !/^\s*(false|0)\s*$/i.test(value ?? "");
}

/** The button is shown through the January after the year, then goes by itself. */
export function wrappedInSeason(year: number, now = new Date()): boolean {
  return now < new Date(year + 1, 1, 1);
}

/** [start, end) of the year, in the viewer's local time. */
export function yearRange(year: number): { start: Date; end: Date } {
  return { start: new Date(year, 0, 1), end: new Date(year + 1, 0, 1) };
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function rank(counts: Map<string, number>, n: number): Ranked[] {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, n)
    .map(([name, count]) => ({ name, count }));
}

// The speakers of every conversation, in the languages banks are written in:
// always on top, never interesting.
const GENERIC_ENTITIES = new Set([
  "user",
  "agent",
  "assistant",
  "the user",
  "the agent",
  "the assistant",
  "benutzer",
  "nutzer",
  "assistent",
  "utilisateur",
  "usuario",
  "asistente",
  "utente",
  "assistente",
  "usuário",
  "用户",
  "用戶",
  "助手",
  "ユーザー",
  "アシスタント",
  "사용자",
  "어시스턴트",
]);
// A persona read off one or two days of activity is noise.
const PERSONA_MIN_DAYS = 3;

const QUOTE_MAX = 220;

/**
 * A memory as a slide can quote it: without the " | When: … | Involving: …"
 * suffix or a leading "REF-ID: … — " some writers add, cut at a word near
 * QUOTE_MAX characters.
 */
export function quote(text: string): string {
  const s = text
    .split(" | ")[0]
    .replace(/^REF-ID:\s*\S+\s*[—-]\s*/, "")
    .trim();
  if (s.length <= QUOTE_MAX) return s;
  const cut = s.slice(0, QUOTE_MAX);
  const space = cut.lastIndexOf(" ");
  return `${space > 0 ? cut.slice(0, space) : cut}…`;
}

function personaForHour(h: number): Persona {
  if (h >= 22 || h < 5) return "nightOwl";
  if (h < 9) return "earlyBird";
  if (h < 13) return "nineToFive";
  if (h < 18) return "afternoon";
  return "evening";
}

export function computeWrapped(year: number, banks: BankYear[]): WrappedStats {
  const { start, end } = yearRange(year);
  const bars = new Array(12).fill(0);
  const days = new Map<string, number>();
  const hours = new Array(24).fill(0);
  const entities = new Map<string, number>();
  const harnesses = new Map<string, number>();
  const bankTotals = new Map<string, number>();
  const add = (m: Map<string, number>, k: string, n: number) => m.set(k, (m.get(k) || 0) + n);
  const spelling = new Map<string, string>();
  let facts = 0;
  let observations = 0;
  let documents: number | null = null;
  let strongest: WrappedStats["strongestBelief"] = null;
  let first: { text: string; date: string } | null = null;

  for (const { name, summary } of banks) {
    if (summary.documents !== null) documents = (documents ?? 0) + summary.documents;
    // UTC hours, placed in the viewer's own days and hours.
    for (const h of summary.hours) {
      const d = new Date(h.time);
      if (d < start || d >= end) continue;
      const n = h.world + h.experience + h.observation;
      facts += h.world + h.experience;
      observations += h.observation;
      add(bankTotals, name, n);
      add(days, dayKey(d), n);
      hours[d.getHours()] += n;
      bars[d.getMonth()] += n;
    }
    const s = summary.strongest_observation;
    if (s && (!strongest || (s.proof_count ?? 1) > strongest.proofs))
      strongest = { text: quote(s.text), proofs: s.proof_count ?? 1 };
    const f = summary.first_memory;
    if (f && (!first || new Date(f.time) < new Date(first.date)))
      first = { text: quote(f.text), date: f.time };
    for (const e of summary.entities ?? []) {
      const n = e.name.trim();
      if (!n || n === "None" || GENERIC_ENTITIES.has(n.toLowerCase())) continue;
      // `key:value` entities are labels some integrations add (`knowledge:decision`), not names.
      if (n.indexOf(":") > 0) continue;
      // Banks write the same name in different cases; the first spelling seen is shown.
      const key = n.toLowerCase();
      if (!spelling.has(key)) spelling.set(key, n);
      add(entities, key, e.count);
    }
    for (const h of summary.harnesses ?? []) add(harnesses, h.name, h.count);
  }

  // Longest run of consecutive active days.
  let longestStreak = 0;
  let run = 0;
  for (const d = new Date(start); d < end; d.setDate(d.getDate() + 1)) {
    run = days.has(dayKey(d)) ? run + 1 : 0;
    longestStreak = Math.max(longestStreak, run);
  }

  const peak = hours.some((h) => h > 0) ? hours.indexOf(Math.max(...hours)) : null;

  return {
    memories: facts + observations,
    banks: bankTotals.size,
    topBanks: rank(bankTotals, 3),
    facts,
    observations,
    documents,
    activeDays: days.size,
    longestStreak,
    busiestDay: rank(days, 1)[0] ?? null,
    bars,
    topEntities: rank(entities, 5).map((e) => ({ ...e, name: spelling.get(e.name)! })),
    topHarness: rank(harnesses, 1)[0] ?? null,
    persona: peak === null || days.size < PERSONA_MIN_DAYS ? null : personaForHour(peak),
    peakHour: peak,
    hourly: hours,
    strongestBelief: strongest,
    firstMemory: first,
  };
}
