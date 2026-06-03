import { describe, expect, it } from "vitest";
import { relativeTime, renderSyncStatus, type SyncStatus } from "../src/status-bar";

const base: SyncStatus = {
  configured: true,
  syncing: 0,
  pending: 0,
  synced: 0,
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

  it("shows the synced note count and relative time when idle", () => {
    const v = renderSyncStatus({ ...base, synced: 412, lastSyncAt: NOW - 120_000 }, NOW);
    expect(v.text).toBe("Hindsight ✓ 412 notes · 2m ago");
    expect(v.tooltip).toContain("412 notes synced");
    expect(v.tooltip).toContain("last synced 2m ago");
  });

  it("singularises a one-note vault", () => {
    const v = renderSyncStatus({ ...base, synced: 1, lastSyncAt: NOW - 1_000 }, NOW);
    expect(v.text).toContain("1 note ");
    expect(v.text).not.toContain("1 notes");
  });

  it("surfaces pending edits in place of the timestamp when idle", () => {
    const v = renderSyncStatus({ ...base, synced: 10, lastSyncAt: NOW - 1_000, pending: 2 }, NOW);
    expect(v.text).toContain("2 pending");
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

  it("drops the brand prefix when asked (chat header reuse)", () => {
    const v = renderSyncStatus({ ...base, synced: 8, lastSyncAt: NOW - 120_000 }, NOW, "");
    expect(v.text).toBe("✓ 8 notes · 2m ago");
    expect(v.text).not.toContain("Hindsight");
  });
});
