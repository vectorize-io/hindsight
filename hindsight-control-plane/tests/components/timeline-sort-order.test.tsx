// @vitest-environment jsdom
/**
 * The timeline's newest/oldest toggle (#5475).
 *
 * It used to sort oldest first with no way to flip it, so today's memories sat
 * on the last page. It now opens newest first, and one click reverses both the
 * groups and the memories inside them.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en",
}));

import { TimelineView } from "@/components/data-view";

afterEach(cleanup);

const rows = [
  { id: "a", text: "january memory", occurred_start: "2026-01-12T09:00:00Z" },
  { id: "c", text: "october memory", occurred_start: "2026-10-14T15:00:00Z" },
  { id: "b", text: "may memory", occurred_start: "2026-05-21T18:00:00Z" },
  { id: "b2", text: "late may memory", occurred_start: "2026-05-28T18:00:00Z" },
];

function shownOrder(): string[] {
  return screen.getAllByText(/memory$/).map((el) => el.textContent ?? "");
}

describe("TimelineView sort order", () => {
  it("opens newest first, then flips to oldest first", () => {
    render(<TimelineView data={null} filteredRows={rows} onMemoryClick={() => {}} />);

    expect(shownOrder()).toEqual([
      "october memory",
      "late may memory",
      "may memory",
      "january memory",
    ]);

    fireEvent.click(screen.getByRole("button", { name: /sortNewestFirst/ }));

    expect(shownOrder()).toEqual([
      "january memory",
      "may memory",
      "late may memory",
      "october memory",
    ]);
    expect(screen.getByRole("button", { name: /sortOldestFirst/ })).toBeTruthy();
  });
});
