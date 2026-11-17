// @vitest-environment jsdom
/**
 * The knowledge search panel is one search call: the text and an attached image or clip go in the
 * same request, and results keep the server's order.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));

const searchKnowledgeBank = vi.fn();
const listKnowledgeBankTags = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    searchKnowledgeBank: (...a: unknown[]) => searchKnowledgeBank(...a),
    listKnowledgeBankTags: (...a: unknown[]) => listKnowledgeBankTags(...a),
  },
}));

const { KnowledgeSearchPanel } = await import("@/components/knowledge-search-panel");

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("KnowledgeSearchPanel", () => {
  it("sends the text and an attached clip in one search", async () => {
    searchKnowledgeBank.mockResolvedValue({ results: [] });
    globalThis.URL.createObjectURL = vi.fn(() => "blob:preview");
    globalThis.URL.revokeObjectURL = vi.fn();
    render(<KnowledgeSearchPanel bankId="docs" />);

    fireEvent.change(screen.getByLabelText("placeholder"), { target: { value: " key rotation " } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toEqual({
      query: "key rotation",
      max_tokens: 4096,
      trace: true,
    });

    const clip = new File([new Uint8Array([0, 0, 0, 24])], "clip.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("kb-search-media-input"), { target: { files: [clip] } });
    await screen.findByText("clip.mp4");
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(2));
    const body = searchKnowledgeBank.mock.calls[1][1];
    expect(body).toMatchObject({ query: "key rotation", max_tokens: 4096 });
    expect(typeof body.query_video).toBe("string");
  });

  it("suggests the bank's own tags and searches with the picked one", async () => {
    searchKnowledgeBank.mockResolvedValue({ results: [] });
    listKnowledgeBankTags.mockResolvedValue({
      items: [{ tag: "team:a", count: 3 }],
      total: 1,
      limit: 20,
      offset: 0,
    });
    render(<KnowledgeSearchPanel bankId="docs" />);
    fireEvent.change(screen.getByLabelText("placeholder"), { target: { value: "ship" } });
    fireEvent.click(screen.getByText("options"));

    const tagInput = screen.getByPlaceholderText("filterByTagPlaceholder");
    fireEvent.focus(tagInput);
    fireEvent.change(tagInput, { target: { value: "te" } });
    await waitFor(() => expect(listKnowledgeBankTags).toHaveBeenCalledWith("docs", "te*", 20));
    fireEvent.mouseDown(await screen.findByText("team:a"));
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toMatchObject({ tags: ["team:a"], tags_match: "any" });
  });

  it("sends tag groups as JSON and refuses ones that do not parse", async () => {
    searchKnowledgeBank.mockResolvedValue({ results: [] });
    render(<KnowledgeSearchPanel bankId="docs" />);
    fireEvent.change(screen.getByLabelText("placeholder"), { target: { value: "red" } });
    fireEvent.click(screen.getByText("options"));

    fireEvent.change(screen.getByLabelText("tagGroups"), { target: { value: "[{not json" } });
    fireEvent.click(screen.getByText("submit"));
    expect(await screen.findByText("tagGroupsInvalid")).toBeTruthy();
    expect(searchKnowledgeBank).not.toHaveBeenCalled();

    const groups = [{ not: { tags: ["oss"], match: "any_strict" } }];
    fireEvent.change(screen.getByLabelText("tagGroups"), { target: { value: JSON.stringify(groups) } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    expect(searchKnowledgeBank.mock.calls[0][1]).toMatchObject({ query: "red", tag_groups: groups });
  });

  it("shows results as one ranked list, in the server's order", async () => {
    const hit = (document_id: string, media_type: string) => ({
      document_id,
      index: 0,
      media_type,
      tags: [],
      text: document_id,
      filename: `${document_id}.txt`,
    });
    searchKnowledgeBank.mockResolvedValue({
      results: [hit("first", "image"), hit("second", "text"), hit("third", "image")],
    });
    render(<KnowledgeSearchPanel bankId="docs" />);
    expect(screen.getByText("submit").closest("button")?.disabled).toBe(true);

    fireEvent.change(screen.getByLabelText("placeholder"), { target: { value: "red car" } });
    fireEvent.click(screen.getByText("submit"));
    await waitFor(() => expect(searchKnowledgeBank).toHaveBeenCalledTimes(1));
    const ranks = (await screen.findAllByText(/^#\d$/)).map((el) => el.textContent);
    expect(ranks).toEqual(["#1", "#2", "#3"]);
  });
});
