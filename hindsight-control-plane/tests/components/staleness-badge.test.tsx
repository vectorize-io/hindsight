// @vitest-environment jsdom
/**
 * The paused state of the shared freshness indicator (#4532).
 *
 * Once a refresh fails, nothing runs that model again by itself, so the badge
 * must stop reporting staleness — "stale, refreshes itself" is a promise the
 * server is no longer keeping — and say paused instead, on every surface that
 * renders it.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import { StalenessBadge } from "@/components/staleness-badge";

afterEach(cleanup);

describe("StalenessBadge", () => {
  it("reports staleness when the last refresh worked", () => {
    render(<StalenessBadge isStale={true} trigger={{ refresh_after_consolidation: true }} />);
    expect(screen.getByText("stale")).toBeTruthy();
  });

  it("says paused, not stale, once a refresh has failed", () => {
    render(
      <StalenessBadge
        isStale={true}
        trigger={{ refresh_after_consolidation: true }}
        refreshFailedAt="2026-09-23T08:02:02Z"
      />
    );
    expect(screen.queryByText("stale")).toBeNull();
    expect(screen.getByText("paused")).toBeTruthy();
  });

  it("shows paused even on a surface that asked for no staleness", () => {
    // The tree passes is_stale only when it computed it; the pause is about the
    // refresh, not the data, so it must not be swallowed with the null.
    render(<StalenessBadge isStale={null} refreshFailedAt="2026-09-23T08:02:02Z" />);
    expect(screen.getByText("paused")).toBeTruthy();
  });

  it("says retrying, not paused, while an attempt is still queued", () => {
    // The failure is stamped on the first failed attempt, while the worker still
    // has retries left — calling that paused is wrong for those few minutes.
    render(<StalenessBadge isStale={true} refreshFailedAt="2026-09-23T08:02:02Z" retrying />);
    expect(screen.queryByText("paused")).toBeNull();
    expect(screen.getByText("retrying")).toBeTruthy();
  });

  it("renders nothing when staleness is unknown and nothing failed", () => {
    const { container } = render(<StalenessBadge isStale={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("marks the compact dot red when paused", () => {
    const { container } = render(
      <StalenessBadge isStale={false} refreshFailedAt="2026-09-23T08:02:02Z" variant="dot" />
    );
    expect((container.firstChild as HTMLElement).className).toContain("bg-red-500");
  });
});
