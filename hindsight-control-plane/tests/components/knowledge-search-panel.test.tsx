// @vitest-environment jsdom
/**
 * The knowledge search panel is one search call: the text, an attached image or clip, and the
 * exact text (`pattern`) all go in the same request, and results keep the server's order.
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
  it("sends the text, the exact text and an attached clip in one search", async () => {
    searchKnowledgeBank.mockResolvedValue({ results: [], took_ms: 1 });
    globalThis.URL.createObjectURL = vi.fn(() => "blob:preview");
    globalThis.URL.revokeObjectURL = vi.fn();
    render(<KnowledgeSearchPanel bankId="docs" />);

    fireEvent.change(screen.getByLabelText("placeholder"), { target: { value: " key rotation " } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toEqual({ query: "key rotation", top_k: 10, neighbors: 1 });

    fireEvent.click(screen.getByText("options"));
    fireEvent.change(screen.getByLabelText("exactText"), { target: { value: "ERR-4417" } });
    const clip = new File([new Uint8Array([0, 0, 0, 24])], "clip.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("kb-search-media-input"), { target: { files: [clip] } });
    await screen.findByText("clip.mp4");
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(2));
    const body = searchKnowledgeBank.mock.calls[1][1];
    expect(body).toMatchObject({ query: "key rotation", pattern: "ERR-4417", top_k: 10 });
    expect(typeof body.query_video).toBe("string");
    expect(body.regex).toBeUndefined();
  });

  it("searches with exact text alone, and shows results in the server's order", async () => {
    const hit = (document_id: string, media_type: string) => ({
      document_id,
      index: 0,
      media_type,
      score: 0.5,
      tags: [],
      text: document_id,
      filename: `${document_id}.txt`,
    });
    searchKnowledgeBank.mockResolvedValue({
      results: [hit("first", "image"), hit("second", "text"), hit("third", "image")],
      took_ms: 1,
    });
    render(<KnowledgeSearchPanel bankId="docs" />);
    expect(screen.getByText("submit").closest("button")?.disabled).toBe(true);

    fireEvent.click(screen.getByText("options"));
    fireEvent.change(screen.getByLabelText("exactText"), { target: { value: "time-?box" } });
    fireEvent.click(screen.getByText("regex"));
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toMatchObject({ pattern: "time-?box", regex: true });
    expect(searchKnowledgeBank.mock.calls[0][1].query).toBeUndefined();
    const ranks = (await screen.findAllByText(/^#\d$/)).map((el) => el.textContent);
    expect(ranks).toEqual(["#1", "#2", "#3"]);
  });
});
