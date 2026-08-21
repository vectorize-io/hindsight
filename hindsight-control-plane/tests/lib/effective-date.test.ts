import { describe, expect, it } from "vitest";

import {
  effectiveDate,
  getTimelineGroupKey,
  groupTimelineItems,
  partitionByEffectiveDate,
} from "@/lib/effective-date";

const ISO = {
  occurred: "2024-01-15T10:30:00Z",
  mentioned: "2024-03-20T08:00:00Z",
  occurredEnd: "2024-02-10T12:00:00Z",
  created: "2024-04-01T00:00:00Z",
  legacyDate: "2024-05-05T00:00:00Z",
};

describe("effectiveDate", () => {
  it("prefers occurred_start over every later field (backend COALESCE order)", () => {
    const d = effectiveDate({
      occurred_start: ISO.occurred,
      mentioned_at: ISO.mentioned,
      occurred_end: ISO.occurredEnd,
      created_at: ISO.created,
      date: ISO.legacyDate,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.occurred));
  });

  it("falls back to mentioned_at when occurred_start is missing", () => {
    const d = effectiveDate({
      occurred_start: null,
      mentioned_at: ISO.mentioned,
      occurred_end: ISO.occurredEnd,
      created_at: ISO.created,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.mentioned));
  });

  it("falls back to occurred_end when occurred_start and mentioned_at are missing", () => {
    const d = effectiveDate({
      occurred_start: null,
      mentioned_at: null,
      occurred_end: ISO.occurredEnd,
      created_at: ISO.created,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.occurredEnd));
  });

  it("falls back to created_at when all occurred/mentioned fields are missing", () => {
    const d = effectiveDate({
      occurred_start: null,
      mentioned_at: null,
      occurred_end: null,
      created_at: ISO.created,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.created));
  });

  it("falls back to the legacy event_date field (list endpoint `date`)", () => {
    const d = effectiveDate({
      occurred_start: null,
      mentioned_at: null,
      occurred_end: null,
      created_at: null,
      date: ISO.legacyDate,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.legacyDate));
  });

  it("returns null when no field carries a date", () => {
    expect(effectiveDate({})).toBeNull();
    expect(
      effectiveDate({
        occurred_start: null,
        mentioned_at: null,
        occurred_end: null,
        created_at: null,
        date: null,
      })
    ).toBeNull();
  });

  it("skips empty strings and the graph endpoint's 'N/A' placeholder", () => {
    const d = effectiveDate({
      occurred_start: "",
      mentioned_at: "N/A",
      occurred_end: undefined,
      created_at: ISO.created,
      date: "",
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.created));
  });

  it("skips invalid strings and keeps walking the chain", () => {
    const d = effectiveDate({
      occurred_start: "not-a-date",
      mentioned_at: ISO.mentioned,
    });
    expect(d?.getTime()).toBe(Date.parse(ISO.mentioned));
  });

  it("parses the graph endpoint's space-separated `date` format as UTC", () => {
    const d = effectiveDate({ date: "2024-01-15 10:30" });
    // The space form is outside ECMA-262's grammar, so it must not depend on
    // `new Date` local-time behavior; interpret it as UTC deterministically.
    expect(d?.getTime()).toBe(Date.UTC(2024, 0, 15, 10, 30));
  });
});

interface TestRow {
  id: number;
  occurred_start?: string | null;
  mentioned_at?: string | null;
  occurred_end?: string | null;
  created_at?: string | null;
  date?: string | null;
}

describe("partitionByEffectiveDate", () => {
  it("includes rows that only carry mentioned_at and sorts by effective date", () => {
    const rows: TestRow[] = [
      { id: 1, mentioned_at: "2024-03-20T08:00:00Z" },
      { id: 2, occurred_start: "2024-01-15T10:30:00Z" },
      { id: 3, mentioned_at: "2024-02-10T12:00:00Z" },
    ];
    const { sortedItems, itemsWithoutDates } = partitionByEffectiveDate(rows);

    expect(itemsWithoutDates).toEqual([]);
    expect(sortedItems.map((e) => e.row.id)).toEqual([2, 3, 1]);
    expect(sortedItems[0].date.getTime()).toBe(Date.parse("2024-01-15T10:30:00Z"));
  });

  it("falls through an invalid preferred date instead of dropping the row", () => {
    const rows: TestRow[] = [
      { id: 1, occurred_start: "not-a-date", mentioned_at: "2024-03-20T08:00:00Z" },
    ];
    const { sortedItems, itemsWithoutDates } = partitionByEffectiveDate(rows);

    expect(itemsWithoutDates).toEqual([]);
    expect(sortedItems[0].row.id).toBe(1);
    expect(sortedItems[0].date.getTime()).toBe(Date.parse("2024-03-20T08:00:00Z"));
  });

  it("puts rows with no usable date aside, preserving their identity", () => {
    const rows: TestRow[] = [
      { id: 1, mentioned_at: "2024-03-20T08:00:00Z" },
      { id: 2, occurred_start: null, mentioned_at: null, occurred_end: null, date: "" },
      { id: 3 },
    ];
    const { sortedItems, itemsWithoutDates } = partitionByEffectiveDate(rows);

    expect(sortedItems.map((e) => e.row.id)).toEqual([1]);
    expect(itemsWithoutDates.map((r) => r.id)).toEqual([2, 3]);
  });

  it("keeps input order for equal effective dates (stable sort)", () => {
    const rows: TestRow[] = [
      { id: 1, mentioned_at: "2024-03-20T08:00:00Z" },
      { id: 2, occurred_start: "2024-03-20T08:00:00Z" },
      { id: 3, mentioned_at: "2024-03-20T08:00:00Z" },
    ];
    const { sortedItems } = partitionByEffectiveDate(rows);
    expect(sortedItems.map((e) => e.row.id)).toEqual([1, 2, 3]);
  });

  it("handles empty input", () => {
    expect(partitionByEffectiveDate([])).toEqual({ sortedItems: [], itemsWithoutDates: [] });
  });
});

describe("groupTimelineItems", () => {
  // Local-time constructors so the local date components below match the
  // intended values on any machine timezone.
  const entries = (dates: Date[]) =>
    dates.map((date, i) => ({ row: { id: i }, date }));

  it("buckets by month and orders groups ascending", () => {
    const sorted = entries([
      new Date(2024, 0, 15, 10, 30),
      new Date(2024, 0, 31, 10, 30),
      new Date(2024, 2, 5, 10, 30),
    ]);
    const groups = groupTimelineItems(sorted, "month");

    expect(groups.map((g) => g.key)).toEqual(["2024-01", "2024-03"]);
    expect(groups[0].items.map((e) => e.row.id)).toEqual([0, 1]);
    expect(groups[1].items.map((e) => e.row.id)).toEqual([2]);
    expect(groups[0].label).toBe(new Date(2024, 0, 15).toLocaleDateString("en-US", { year: "numeric", month: "short" }));
  });

  it("groups week buckets by Sunday-aligned start", () => {
    // 2024-01-10 is a Wednesday; its week starts Sunday 2024-01-07.
    const sorted = entries([
      new Date(2024, 0, 10, 10, 0),
      new Date(2024, 0, 12, 10, 0),
      new Date(2024, 0, 15, 10, 0), // next week (Monday, week start 2024-01-14)
    ]);
    const groups = groupTimelineItems(sorted, "week");

    expect(groups.map((g) => g.key)).toEqual(["2024-W01-01-07", "2024-W02-01-14"]);
    expect(groups[0].items.map((e) => e.row.id)).toEqual([0, 1]);
    expect(groups[1].items.map((e) => e.row.id)).toEqual([2]);
  });

  it("buckets by day at day granularity", () => {
    const sorted = entries([new Date(2024, 0, 15, 10, 0), new Date(2024, 0, 16, 10, 0)]);
    const groups = groupTimelineItems(sorted, "day");
    expect(groups.map((g) => g.key)).toEqual(["2024-01-15", "2024-01-16"]);
  });

  it("returns the full year as the key at year granularity", () => {
    const sorted = entries([new Date(2024, 5, 1), new Date(2023, 11, 31)]);
    const groups = groupTimelineItems(sorted, "year");
    expect(groups.map((g) => g.key)).toEqual(["2023", "2024"]);
  });

  it("returns no groups for empty input", () => {
    expect(groupTimelineItems([], "month")).toEqual([]);
  });

  it("derives deterministic keys for every granularity", () => {
    const date = new Date(2024, 0, 15, 10, 30);
    expect(getTimelineGroupKey(date, "year")).toBe("2024");
    expect(getTimelineGroupKey(date, "month")).toBe("2024-01");
    expect(getTimelineGroupKey(date, "day")).toBe("2024-01-15");
    expect(getTimelineGroupKey(date, "week")).toBe("2024-W02-01-14");
  });
});
