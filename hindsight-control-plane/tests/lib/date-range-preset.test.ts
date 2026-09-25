import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { resolveCustomRange, resolveDateRangePreset } from "@/lib/date-range-preset";

/**
 * The presets are what three list views send as `start_date`, so the two things
 * worth pinning are that each one subtracts the interval it names, and that
 * anything unrecognised means "no filter" rather than an empty window — a
 * preset that silently resolved to `now` would return nothing at all and look
 * like an empty bank.
 */
describe("resolveDateRangePreset", () => {
  const now = new Date("2026-03-15T12:00:00.000Z");

  it("sends no bounds for the all-time preset", () => {
    expect(resolveDateRangePreset("all", now)).toEqual({});
  });

  it.each([
    ["1h", "2026-03-15T11:00:00.000Z"],
    ["1d", "2026-03-14T12:00:00.000Z"],
    ["7d", "2026-03-08T12:00:00.000Z"],
    ["30d", "2026-02-13T12:00:00.000Z"],
  ])("subtracts the interval %s names", (preset, expected) => {
    expect(resolveDateRangePreset(preset, now).start_date).toBe(expected);
  });

  it("never sets an upper bound — the presets all mean 'since then'", () => {
    expect(resolveDateRangePreset("7d", now).end_date).toBeUndefined();
  });

  it("treats an unknown preset as no filter, not as an empty window", () => {
    expect(resolveDateRangePreset("since-tuesday", now)).toEqual({});
  });

  it("does not mutate the date it is given", () => {
    const before = now.toISOString();
    resolveDateRangePreset("30d", now);
    expect(now.toISOString()).toBe(before);
  });

  it("leaves the custom preset's bounds to its own inputs", () => {
    expect(resolveDateRangePreset("custom", now)).toEqual({});
  });
});

describe("resolveCustomRange", () => {
  it("sends nothing while both ends are blank", () => {
    expect(resolveCustomRange("", "")).toEqual({ bounds: {}, reversed: false });
  });

  it("accepts either end alone — the endpoints take one bound", () => {
    expect(resolveCustomRange("2026-03-01T00:00", "").bounds.end_date).toBeUndefined();
    expect(resolveCustomRange("2026-03-01T00:00", "").bounds.start_date).toBeDefined();
    expect(resolveCustomRange("", "2026-03-01T00:00").bounds.start_date).toBeUndefined();
    expect(resolveCustomRange("", "2026-03-01T00:00").bounds.end_date).toBeDefined();
  });

  it("reads the typed time as local, not as UTC", () => {
    // `datetime-local` carries no offset. Going through Date is what makes
    // "09:00" mean the user's 09:00; asserting against the same conversion
    // keeps the test true in any timezone CI runs in.
    const { bounds } = resolveCustomRange("2026-03-01T09:00", "");
    expect(bounds.start_date).toBe(new Date("2026-03-01T09:00").toISOString());
  });

  it("rejects a reversed range and sends no bounds", () => {
    const state = resolveCustomRange("2026-03-10T00:00", "2026-03-01T00:00");
    expect(state.reversed).toBe(true);
    expect(state.bounds).toEqual({});
  });

  // The input hands back a half-finished value while it is being typed. Some of
  // those parse (a bare "2027" is a valid ISO prefix) and some do not, and it was
  // the second kind that mattered: `new Date(x).toISOString()` raises RangeError
  // on an Invalid Date, which took the whole documents page down on the first
  // keystroke into the field.
  it.each(["2027-01-01T", "not-a-date", "99999-13-45T99:99"])(
    "treats the unparseable value %s as no bound instead of throwing",
    (partial) => {
      expect(() => resolveCustomRange(partial, "")).not.toThrow();
      expect(resolveCustomRange(partial, "").bounds).toEqual({});
    }
  );

  it("rejects a zero-length range — the end is exclusive, so it can match nothing", () => {
    const state = resolveCustomRange("2026-03-10T00:00", "2026-03-10T00:00");
    expect(state.reversed).toBe(true);
    expect(state.bounds).toEqual({});
  });

  describe("across a daylight-saving gap", () => {
    // Pinned, because this is the ONE case where comparing the typed text and
    // comparing the instants disagree — and CI runs in UTC, which has no gap. An
    // unpinned version of this test silently degenerates into "an ascending range
    // is not reversed" and stops guarding anything.
    const original = process.env.TZ;
    beforeAll(() => {
      process.env.TZ = "America/Los_Angeles";
    });
    afterAll(() => {
      process.env.TZ = original;
    });

    it("judges the range by the instants it sends, not by the typed text", () => {
      // 02:30 does not exist on this morning: it normalises forward to 10:30Z,
      // while 03:00 is 10:00Z. Ascending as text, half an hour backwards as time.
      const state = resolveCustomRange("2026-03-08T02:30", "2026-03-08T03:00");
      expect(new Date("2026-03-08T02:30").toISOString()).toBe("2026-03-08T10:30:00.000Z");
      expect(state.reversed).toBe(true);
      expect(state.bounds).toEqual({});
    });
  });
});
