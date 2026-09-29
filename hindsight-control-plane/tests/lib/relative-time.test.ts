import { afterEach, describe, expect, it, vi } from "vitest";

import { formatWatermark } from "@/lib/relative-time";

afterEach(() => {
  vi.useRealTimers();
});

/** Pin "now" so the today/this-year/earlier-year branches are decidable. */
function at(now: string) {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(now));
}

describe("formatWatermark", () => {
  it("shows only the clock time for today", () => {
    at("2026-08-19T18:00:00Z");
    // A watermark earlier the same day needs no date to be unambiguous.
    expect(formatWatermark("2026-08-19T12:38:00Z")).toMatch(/^\d{2}:\d{2}$/);
  });

  it("keeps the time when the day differs but the year does not", () => {
    at("2026-08-19T18:00:00Z");
    // Dropping the time here was the bug: "3 Aug" hid how far behind the model
    // actually was within that day.
    const out = formatWatermark("2026-08-18T08:38:00Z");
    expect(out).toMatch(/\d{2}:\d{2}$/);
    expect(out).not.toMatch(/^\d{2}:\d{2}$/);
    expect(out).not.toContain("2026");
  });

  it("carries the year once the watermark is from an earlier one", () => {
    at("2026-08-19T18:00:00Z");
    // Without this, a watermark from last August rendered identically to this
    // August — the same string for a year-old document and a fresh one.
    expect(formatWatermark("2025-08-19T12:38:00Z")).toContain("2025");
  });

  it("distinguishes two moments that read the same relatively", () => {
    at("2026-08-19T15:19:00Z");
    // The reason this is absolute at all: 11:19 and 09:18 both render as
    // "4 hours ago", which made "refreshed 4 hours ago · read to 4 hours ago"
    // look like the same fact printed twice.
    expect(formatWatermark("2026-08-19T11:19:00Z")).not.toBe(
      formatWatermark("2026-08-19T09:18:00Z")
    );
  });
});
