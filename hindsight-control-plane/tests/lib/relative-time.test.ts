import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { formatRelativeTime } from "@/lib/relative-time";

// Fixed reference point so "X ago" output is deterministic.
const NOW = new Date("2026-06-25T13:57:00.000Z");
const SEC = 1000;
const MIN = 60 * SEC;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

function ago(ms: number): string {
  return new Date(NOW.getTime() - ms).toISOString();
}

describe("formatRelativeTime", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders sub-minute durations as 'just now'", () => {
    expect(formatRelativeTime(ago(30 * SEC))).toBe("just now");
  });

  // Regression: every duration used to render one unit too small, so a
  // document created ~22 hours ago (i.e. yesterday) showed as "22 minutes ago".
  it("labels minutes, hours and days with the correct unit", () => {
    expect(formatRelativeTime(ago(22 * MIN))).toBe("22 minutes ago");
    expect(formatRelativeTime(ago(13 * HOUR))).toBe("13 hours ago");
    expect(formatRelativeTime(ago(22 * HOUR))).toBe("22 hours ago");
  });

  it("rolls 24h+ into days and uses 'yesterday' for one day", () => {
    expect(formatRelativeTime(ago(25 * HOUR))).toBe("yesterday");
    expect(formatRelativeTime(ago(2 * DAY))).toBe("2 days ago");
  });

  it("scales up to weeks, months and years", () => {
    expect(formatRelativeTime(ago(8 * DAY))).toBe("last week");
    expect(formatRelativeTime(ago(40 * DAY))).toBe("last month");
    expect(formatRelativeTime(ago(400 * DAY))).toBe("last year");
  });

  it("handles future timestamps", () => {
    expect(formatRelativeTime(new Date(NOW.getTime() + 3 * HOUR).toISOString())).toBe("in 3 hours");
  });
});
