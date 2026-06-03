import { describe, expect, it } from "vitest";
import { relativeTime, renderSyncStatus, type SyncStatus } from "../src/status-bar";

const base: SyncStatus = {
  configured: true,
  syncing: 0,
  pending: 0,
  lastSyncAt: null,
  error: false,
};
const NOW = 1_700_000_000_000;

describe("relativeTime", () => {
  it("reports recent times as 'just now'", () => {
    expect(relativeTime(NOW - 5_000, NOW)).toBe("just now");
  });

  it("rounds to minutes, hours, and days", () => {
    expect(relativeTime(NOW - 5 * 60_000, NOW)).toBe("5m ago");
    expect(relativeTime(NOW - 3 * 3_600_000, NOW)).toBe("3h ago");
    expect(relativeTime(NOW - 2 * 86_400_000, NOW)).toBe("2d ago");
  });
});

describe("renderSyncStatus", () => {
  it("prompts for setup when no API URL is configured", () => {
    const v = renderSyncStatus({ ...base, configured: false }, NOW);
    expect(v.text).toContain("set API URL");
  });

  it("shows a syncing state with pending count while in flight", () => {
    const v = renderSyncStatus({ ...base, syncing: 1, pending: 3 }, NOW);
    expect(v.text).toContain("syncing");
    expect(v.text).toContain("3");
  });

  it("surfaces an error state", () => {
    const v = renderSyncStatus({ ...base, error: true, lastSyncAt: NOW - 60_000 }, NOW);
    expect(v.text).toContain("sync failed");
  });

  it("shows the last-synced relative time when idle", () => {
    const v = renderSyncStatus({ ...base, lastSyncAt: NOW - 120_000 }, NOW);
    expect(v.text).toBe("Hindsight ✓ 2m ago");
    expect(v.tooltip).toContain("Last synced 2m ago");
  });

  it("notes pending edits in the idle tooltip", () => {
    const v = renderSyncStatus({ ...base, lastSyncAt: NOW - 1_000, pending: 2 }, NOW);
    expect(v.tooltip).toContain("2 pending");
  });

  it("indicates when nothing has synced yet", () => {
    const v = renderSyncStatus(base, NOW);
    expect(v.text).toContain("not synced yet");
  });

  it("prioritises in-flight syncing over a prior error", () => {
    const v = renderSyncStatus({ ...base, syncing: 1, error: true }, NOW);
    expect(v.text).toContain("syncing");
  });
});
