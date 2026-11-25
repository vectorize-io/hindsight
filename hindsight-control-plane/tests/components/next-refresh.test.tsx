// @vitest-environment jsdom
/**
 * "Next refresh" for a refresh in flight (#6864): a scheduled time is a retry
 * only when an earlier attempt of that refresh failed.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: { time?: string }) =>
    values?.time ? `${key}:${values.time}` : key,
}));

import { NextRefresh } from "@/components/next-refresh";

afterEach(cleanup);

const future = () => new Date(Date.now() + 60 * 60 * 1000).toISOString();
const past = "2020-01-01T00:00:00Z";

describe("NextRefresh with a refresh in flight", () => {
  it.each([
    ["scheduled, never failed", { running: false, retry: false, nextAttemptAt: future() }, "nextRefreshQueued:in 1 hour"],
    ["waiting for a worker", { running: false, retry: false, nextAttemptAt: null }, "nextRefreshQueuedSoon"],
    ["past its time, never failed", { running: false, retry: false, nextAttemptAt: past }, "nextRefreshQueuedSoon"],
    ["first attempt running", { running: true, retry: false, nextAttemptAt: null }, "nextRefreshRunning"],
    ["waiting to retry", { running: false, retry: true, nextAttemptAt: future() }, "nextRefreshRetrying:in 1 hour"],
    ["retry due now", { running: false, retry: true, nextAttemptAt: null }, "nextRefreshRetryingSoon"],
    ["retry running", { running: true, retry: true, nextAttemptAt: null }, "nextRefreshRetryingNow"],
  ])("%s", (_name, attempt, text) => {
    render(<NextRefresh trigger={{ refresh_after_consolidation: true }} attempt={attempt} />);
    expect(screen.getByText(text)).toBeTruthy();
  });

  it("names the paused state only when nothing is in flight", () => {
    render(<NextRefresh refreshFailedAt="2026-09-23T08:02:02Z" />);
    expect(screen.getByText("nextRefreshPaused")).toBeTruthy();
  });
});
