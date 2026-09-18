import { describe, it, expect } from "vitest";

import { resolveDateRangePreset } from "@/lib/date-range-preset";

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
});
