// @vitest-environment jsdom
/**
 * The shared tag selector (documents, mental models, consolidation strategies).
 *
 * Suggestions arrive on a debounce, so for a moment after typing the dropdown
 * still shows — and highlights — suggestions for the *previous* input. Enter in
 * that window used to add the stale highlighted suggestion instead of what was
 * typed: typing "team:*" and pressing Enter added "company:acme".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import { TagFilterInput } from "@/components/tag-filter-input";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

async function renderWithSuggestions(onChange: (tags: string[]) => void) {
  vi.useFakeTimers();
  const fetchSuggestions = vi.fn(async (q: string) =>
    ["company:acme", "company:*", "team:*", "team:exec"].filter((tag) => tag.startsWith(q))
  );
  render(
    <TagFilterInput value={[]} onChange={onChange} fetchSuggestions={fetchSuggestions} placeholder="tag" />
  );
  const input = screen.getByPlaceholderText("tag");
  // Let the initial (empty-query) suggestions load, so the dropdown is populated.
  fireEvent.focus(input);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(200);
  });
  return input;
}

describe("TagFilterInput", () => {
  it("adds what was typed when Enter comes before the suggestions refresh", async () => {
    const onChange = vi.fn();
    const input = await renderWithSuggestions(onChange);

    // "company:acme" is highlighted from the empty query; type and press Enter
    // immediately, inside the debounce window.
    fireEvent.change(input, { target: { value: "team:*" } });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(onChange).toHaveBeenCalledWith(["team:*"]);
  });

  it("still adds the highlighted suggestion once it matches the input", async () => {
    const onChange = vi.fn();
    const input = await renderWithSuggestions(onChange);

    fireEvent.change(input, { target: { value: "team" } });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(onChange).toHaveBeenCalledWith(["team:*"]);
  });
});
