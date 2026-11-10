// @vitest-environment jsdom
/**
 * The knowledge search panel's two modes, one search call: "by meaning" sends a query, "exact
 * text" sends the words as typed (codes, ids) as `pattern`, and no image.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));

const searchKnowledgeBank = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    searchKnowledgeBank: (...a: unknown[]) => searchKnowledgeBank(...a),
  },
}));

const { KnowledgeSearchPanel } = await import("@/components/knowledge-search-panel");

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("KnowledgeSearchPanel", () => {
  it("searches by meaning by default and by pattern in exact-text mode", async () => {
    const empty = { results: [], took_ms: 1 };
    searchKnowledgeBank.mockResolvedValue(empty);
    render(<KnowledgeSearchPanel bankId="docs" />);
    const input = screen.getByLabelText("placeholder");

    fireEvent.change(input, { target: { value: " key rotation " } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toMatchObject({ query: "key rotation", top_k: 10 });

    fireEvent.click(screen.getByText("modeExact"));
    expect(screen.getByText("addImage").closest("button")?.disabled).toBe(true);
    fireEvent.change(input, { target: { value: "ERR-4417" } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(2));
    const exact = searchKnowledgeBank.mock.calls[1][1];
    expect(exact).toMatchObject({ pattern: "ERR-4417", top_k: 10 });
    expect(exact.query).toBeUndefined();
  });
});
