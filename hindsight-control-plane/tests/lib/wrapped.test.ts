import { describe, it, expect } from "vitest";
import type { StatsSummary } from "@/lib/api";
import {
  accountStamp,
  computeWrapped,
  readCache,
  writeCache,
  quote,
  wrappedEnabled,
  wrappedInSeason,
  type BankYear,
} from "@/lib/wrapped";

const bank = (name: string, over: Partial<StatsSummary>): BankYear => ({
  name,
  summary: {
    bank_id: name,
    start_date: "",
    end_date: "",
    hours: [],
    documents: 0,
    harnesses: [],
    entities: [],
    strongest_observation: null,
    first_memory: null,
    ...over,
  },
});

// A local-time hour as the UTC ISO the server sends, so the test is timezone-independent.
const hour = (m: number, d: number, h: number, world: number, observation = 0) => ({
  time: new Date(2026, m, d, h).toISOString(),
  world,
  experience: 0,
  observation,
});

describe("computeWrapped", () => {
  it("keeps the hours when a bank's store can't give the rest", () => {
    const s = computeWrapped(2026, [
      bank("a", { hours: [hour(2, 1, 9, 3)], documents: null, harnesses: null, entities: null }),
    ]);
    expect(s.memories).toBe(3);
    expect(s.peakHour).toBe(9);
    expect(s.documents).toBeNull();
    expect(s.topEntities).toEqual([]);
    expect(s.topHarness).toBeNull();
    expect(s.strongestBelief).toBeNull();
  });

  it("gives no documents total when only some banks could count them", () => {
    const s = computeWrapped(2026, [
      bank("a", { hours: [hour(2, 1, 9, 3)], documents: 5 }),
      bank("b", { hours: [hour(2, 1, 9, 1)], documents: null }),
    ]);
    expect(s.documents).toBeNull();
  });

  it("adds the banks' hours up in local time, and ranks banks, names, agents and beliefs", () => {
    const s = computeWrapped(2026, [
      bank("a", {
        documents: 3,
        hours: [
          { ...hour(0, 1, 0, 9), time: new Date(2025, 11, 31, 23).toISOString() }, // last year
          hour(2, 1, 23, 2),
          hour(2, 2, 23, 2, 1),
        ],
        strongest_observation: { text: "belief", time: "", proof_count: 7 },
        first_memory: {
          text: "first | When: 2026-03-01",
          time: new Date(2026, 2, 1, 23).toISOString(),
        },
        entities: [
          { name: "Alice", count: 4 },
          { name: "user", count: 99 },
          { name: "knowledge:decision", count: 50 },
        ],
        harnesses: [{ name: "claude-code", count: 5 }],
      }),
      bank("b", {
        documents: 1,
        hours: [hour(2, 3, 9, 1), hour(5, 9, 9, 0, 1)],
        strongest_observation: { text: "weaker", time: "", proof_count: 2 },
        first_memory: { text: "later", time: new Date(2026, 2, 3, 9).toISOString() },
        entities: [{ name: "alice", count: 1 }],
        harnesses: [{ name: "codex", count: 2 }],
      }),
      bank("idle", {}),
    ]);
    expect(s.memories).toBe(7);
    expect(s.facts).toBe(5);
    expect(s.observations).toBe(2);
    expect(s.documents).toBe(4);
    expect(s.banks).toBe(2);
    expect(s.topBanks).toEqual([
      { name: "a", count: 5 },
      { name: "b", count: 2 },
    ]);
    expect(s.activeDays).toBe(4);
    expect(s.longestStreak).toBe(3);
    expect(s.busiestDay).toEqual({ name: "2026-03-02", count: 3 });
    expect(s.bars[2]).toBe(6);
    expect(s.bars[5]).toBe(1);
    expect(s.peakHour).toBe(23);
    expect(s.hourly[23]).toBe(5);
    expect(s.hourly[9]).toBe(2);
    expect(s.persona).toBe("nightOwl");
    expect(s.topEntities).toEqual([{ name: "Alice", count: 5 }]);
    expect(s.topHarness).toEqual({ name: "claude-code", count: 5 });
    expect(s.strongestBelief).toEqual({ text: "belief", proofs: 7 });
    expect(s.firstMemory?.text).toBe("first");
  });

  it("an empty year has twelve empty months and no persona", () => {
    const s = computeWrapped(2026, []);
    expect(s.bars).toEqual(new Array(12).fill(0));
    expect(s.persona).toBeNull();
    expect(s.busiestDay).toBeNull();
    expect(s.banks).toBe(0);
  });

  it("needs three active days for a persona, and drops the speakers in any language", () => {
    const s = computeWrapped(2026, [
      bank("a", {
        hours: [hour(4, 1, 23, 1), hour(4, 2, 23, 1)],
        entities: ["Benutzer", "用户", "Assistent", "Polymarket"].map((name) => ({
          name,
          count: 2,
        })),
      }),
    ]);
    expect(s.persona).toBeNull();
    expect(s.peakHour).toBe(23);
    expect(s.topEntities).toEqual([{ name: "Polymarket", count: 2 }]);
  });

  it("quotes a memory without its suffixes and ref prefix, cut at a word", () => {
    expect(quote("REF-ID: conversation:abc — The deploy failed | When: 2026 | Involving: x")).toBe(
      "The deploy failed"
    );
    const long = quote("word ".repeat(100));
    expect(long.length).toBeLessThanOrEqual(221);
    expect(long.endsWith("word…")).toBe(true);
  });
});

describe("when Wrapped is shown", () => {
  it("is on unless the server turns it off", () => {
    expect(wrappedEnabled(undefined)).toBe(true);
    expect(wrappedEnabled("true")).toBe(true);
    expect(wrappedEnabled("false")).toBe(false);
    expect(wrappedEnabled(" FALSE ")).toBe(false);
    expect(wrappedEnabled("0")).toBe(false);
  });

  it("goes by itself after the January that follows the year", () => {
    expect(wrappedInSeason(2026, new Date(2026, 10, 27))).toBe(true);
    expect(wrappedInSeason(2026, new Date(2027, 0, 31, 23, 59))).toBe(true);
    expect(wrappedInSeason(2026, new Date(2027, 1, 1))).toBe(false);
  });
});

describe("the saved year", () => {
  const memoryStorage = () => {
    const m = new Map<string, string>();
    return {
      getItem: (k: string) => m.get(k) ?? null,
      setItem: (k: string, v: string) => void m.set(k, v),
    };
  };
  const year = computeWrapped(2026, [bank("a", { hours: [hour(2, 1, 9, 3)] })]);

  it("belongs to one account, whatever order its banks are listed in", () => {
    const a = [
      { bank_id: "new", created_at: "2026-05-01" },
      { bank_id: "old", created_at: "2025-01-01" },
    ];
    expect(accountStamp(a)).toBe(accountStamp([...a].reverse()));
    expect(accountStamp([{ bank_id: "other", created_at: "2025-01-01" }])).not.toBe(
      accountStamp(a)
    );
    expect(accountStamp([])).toBeNull();
  });

  it("is never read for another account", () => {
    const store = memoryStorage();
    const now = new Date(2026, 10, 27);
    writeCache(store, 2026, "a:1", year, now);
    expect(readCache(store, 2026, "a:1", now)).toEqual(year);
    expect(readCache(store, 2026, "b:1", now)).toBeNull();
    expect(readCache(store, 2026, null, now)).toBeNull();
  });

  it("is read again the next day while the year runs, and kept once it is over", () => {
    const store = memoryStorage();
    writeCache(store, 2026, "a:1", year, new Date(2026, 10, 27));
    expect(readCache(store, 2026, "a:1", new Date(2026, 10, 28))).toBeNull();
    writeCache(store, 2026, "a:1", year, new Date(2027, 0, 2));
    expect(readCache(store, 2026, "a:1", new Date(2027, 0, 20))).toEqual(year);
  });
});

describe("banks with the same name", () => {
  it("still count as two banks", () => {
    const s = computeWrapped(2026, [
      bank("Notes", { bank_id: "n1", hours: [hour(2, 1, 9, 3)] }),
      bank("Notes", { bank_id: "n2", hours: [hour(2, 1, 9, 1)] }),
    ]);
    expect(s.banks).toBe(2);
    expect(s.topBanks).toEqual([
      { name: "Notes", count: 3 },
      { name: "Notes", count: 1 },
    ]);
  });
});
