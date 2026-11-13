// @vitest-environment jsdom
/**
 * The Memory Bank section's knowledge-bank picker: a searchable multi-select that toggles
 * ids, and keeps a linked bank that no longer exists so it can still be unlinked.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
const bank = (bank_id: string) => ({ bank_id, document_count: 1, passage_count: 2 });
vi.mock("@/lib/api", () => ({
  client: { listAllKnowledgeBanks: () => Promise.resolve([bank("docs"), bank("faq")]) },
}));
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as unknown as typeof ResizeObserver;
Element.prototype.scrollIntoView ??= () => {};

const { LinkedKnowledgeBanksPicker } = await import("@/components/linked-knowledge-banks");

afterEach(cleanup);

describe("LinkedKnowledgeBanksPicker", () => {
  it("adds and removes banks, and lists a deleted one so it can be unlinked", async () => {
    const onChange = vi.fn();
    render(<LinkedKnowledgeBanksPicker value={["docs", "gone"]} onChange={onChange} />);
    expect(screen.getByRole("combobox").textContent).toContain("docs, gone");

    fireEvent.click(screen.getByRole("combobox"));
    fireEvent.click(await screen.findByText("faq"));
    expect(onChange).toHaveBeenLastCalledWith(["docs", "gone", "faq"]);

    expect(screen.getByText("missing")).toBeTruthy();
    fireEvent.click(screen.getByText("gone", { selector: "span" }));
    expect(onChange).toHaveBeenLastCalledWith(["docs"]);
  });
});
