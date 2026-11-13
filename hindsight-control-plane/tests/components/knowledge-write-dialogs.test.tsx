// @vitest-environment jsdom
/**
 * Adding a text to a knowledge bank posts `items` to /ingest (the API has no POST on
 * /documents), and a text identical to what is stored says "unchanged" rather than "added".
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
const toast = vi.hoisted(() => ({
  info: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
}));
vi.mock("sonner", () => ({ toast }));

const { ControlPlaneClient } = await import("@/lib/api");
const { client } = await import("@/lib/api");
const knowledgeBanks = await import("@/lib/knowledge-banks");
const { KnowledgeAddDocumentDialog } = await import("@/components/knowledge-write-dialogs");

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("writeKnowledgeTextDocument", () => {
  it("posts the text as one item to the ingest route", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        new Response(
          JSON.stringify({ document_ids: ["a"], operation_ids: [], unchanged_document_ids: [] })
        )
      );
    await new ControlPlaneClient().writeKnowledgeTextDocument("docs", { text: "hi", tags: ["t"] });
    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toMatch(/\/api\/knowledge-banks\/docs\/ingest$/);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ items: [{ text: "hi", tags: ["t"] }] });
  });
});

describe("KnowledgeAddDocumentDialog", () => {
  const renderText = () =>
    render(
      <KnowledgeAddDocumentDialog
        bankId="docs"
        open
        initialTab="text"
        initialFiles={[]}
        onOpenChange={() => {}}
        onWritten={() => {}}
      />
    );

  it("says added for a new text and unchanged for one already stored", async () => {
    const write = vi.spyOn(client, "writeKnowledgeTextDocument");
    write.mockResolvedValueOnce({
      document_ids: ["a"],
      operation_ids: [],
      unchanged_document_ids: [],
    });
    renderText();
    fireEvent.change(screen.getByLabelText("contentLabel"), { target: { value: "hello" } });
    fireEvent.click(screen.getByText("addDocumentSubmit"));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("written"));
    cleanup();

    write.mockResolvedValueOnce({
      document_ids: ["a"],
      operation_ids: [],
      unchanged_document_ids: ["a"],
    });
    renderText();
    fireEvent.change(screen.getByLabelText("contentLabel"), { target: { value: "hello" } });
    fireEvent.click(screen.getByText("addDocumentSubmit"));
    await waitFor(() => expect(toast.info).toHaveBeenCalledWith("writtenUnchanged"));
    expect(toast.success).toHaveBeenCalledTimes(1);
  });
});

describe("KnowledgeAddDocumentDialog upload", () => {
  it("sends the picked files in one request, so they become one operation", async () => {
    const upload = vi
      .spyOn(knowledgeBanks, "uploadKnowledgeFiles")
      .mockResolvedValue({ document_ids: ["a", "b", "c"], operation_ids: ["op"], unchanged_document_ids: [] });
    const files = ["a.pdf", "b.png", "c.txt"].map((n) => new File(["x"], n));
    render(
      <KnowledgeAddDocumentDialog
        bankId="docs"
        open
        initialTab="upload"
        initialFiles={files}
        onOpenChange={() => {}}
        onWritten={() => {}}
      />
    );
    fireEvent.click(screen.getByText("uploadFiles"));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("uploaded"));
    expect(upload).toHaveBeenCalledTimes(1);
    expect((upload.mock.calls[0][1] as File[]).map((f) => f.name)).toEqual(["a.pdf", "b.png", "c.txt"]);
  });
});
