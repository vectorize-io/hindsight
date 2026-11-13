// @vitest-environment jsdom
/**
 * A knowledge bank's Home and Documents: the documents views switch list first, a document's
 * status shows only as a spinner while it ingests (and a clear mark when it failed), the
 * Home chart filters by media type, and nothing reads the removed server config endpoint.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("sonner", () => ({ toast: { warning: vi.fn(), error: vi.fn(), success: vi.fn() } }));
vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: { href: string; children: React.ReactNode }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));
// The canvas graph itself is the memory banks' component; here only what it is given matters.
vi.mock("@/components/constellation", () => ({
  Constellation: ({ data }: { data: { nodes: unknown[]; links: unknown[] } }) => (
    <div data-testid="constellation">
      {data.nodes.length} nodes, {data.links.length} links
    </div>
  ),
}));

const api = vi.hoisted(() => ({
  listKnowledgeDocuments: vi.fn(),
  getKnowledgeDocument: vi.fn(),
  getKnowledgeTimeseries: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ client: api }));

const { KnowledgeDocumentsView } = await import("@/components/knowledge-documents-view");
const { KnowledgeBankHome } = await import("@/components/knowledge-bank-home");
const { buildKnowledgeGraph } = await import("@/components/knowledge-constellation");

beforeAll(() => {
  // Radix Select needs these, which jsdom lacks.
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.releasePointerCapture = () => {};
  Element.prototype.scrollIntoView = () => {};
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const doc = (document_id: string, status: string, media_type = "text") => ({
  document_id,
  filename: `${document_id}.md`,
  content_type: "text/markdown",
  media_type,
  size_bytes: 10,
  status,
  error: status === "failed" ? "could not parse" : null,
  passage_count: status === "ready" ? 2 : 0,
  tags: [],
  created_at: null,
  updated_at: "2026-11-01T00:00:00Z",
  document_url: `/v1/default/knowledge-banks/docs/documents/${document_id}/file`,
});

const passage = (index: number) => ({
  index,
  media_type: "text",
  text: `passage ${index}`,
  page: null,
  start_ms: null,
  end_ms: null,
  passage_url: null,
});

describe("KnowledgeDocumentsView", () => {
  const renderView = () =>
    render(<KnowledgeDocumentsView bankId="docs" onOpenDocument={() => {}} onChanged={() => {}} />);

  it("offers list, grid and constellation in that order, list selected", async () => {
    api.listKnowledgeDocuments.mockResolvedValue({ documents: [doc("a", "ready")], total: 1 });
    renderView();
    await screen.findByText("a.md");
    const group = screen.getByRole("group", { name: "layout" });
    const buttons = within(group).getAllByRole("button");
    expect(buttons.map((b) => b.getAttribute("aria-label"))).toEqual([
      "viewList",
      "viewGrid",
      "viewConstellation",
    ]);
    expect(buttons[0].getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByTestId("constellation")).toBeNull();

    api.getKnowledgeDocument.mockResolvedValue({ ...doc("a", "ready"), passages: [passage(0)] });
    fireEvent.click(buttons[2]);
    expect((await screen.findByTestId("constellation")).textContent).toBe("2 nodes, 1 links");
  });

  it("shows a spinner only while a document ingests, a mark when it failed, nothing when ready", async () => {
    api.listKnowledgeDocuments.mockResolvedValue({
      documents: [
        doc("ready-doc", "ready"),
        doc("pending-doc", "pending"),
        doc("busy-doc", "processing"),
        doc("bad-doc", "failed"),
      ],
      total: 4,
    });
    renderView();
    await screen.findByText("ready-doc.md");
    const row = (name: string) => screen.getByText(`${name}.md`).closest("tr")!;
    expect(within(row("ready-doc")).queryByTestId("status-spinner")).toBeNull();
    expect(within(row("ready-doc")).queryByText("ready")).toBeNull();
    expect(within(row("pending-doc")).getByTestId("status-spinner")).toBeTruthy();
    expect(within(row("busy-doc")).getByTestId("status-spinner")).toBeTruthy();
    expect(within(row("bad-doc")).queryByTestId("status-spinner")).toBeNull();
    expect(within(row("bad-doc")).getByTitle("could not parse").textContent).toContain("failed");
  });

  it("applies the filters and layout a link asked for", async () => {
    api.listKnowledgeDocuments.mockResolvedValue({ documents: [], total: 0 });
    render(
      <KnowledgeDocumentsView
        bankId="docs"
        onOpenDocument={() => {}}
        onChanged={() => {}}
        requestedKind="image"
        requestedStatus="failed"
      />
    );
    await waitFor(() =>
      expect(api.listKnowledgeDocuments).toHaveBeenLastCalledWith(
        "docs",
        expect.objectContaining({ media_type: "image", status: "failed" }),
        expect.anything()
      )
    );
  });
});

describe("KnowledgeBankHome", () => {
  const bank = {
    bank_id: "docs",
    created_at: null,
    document_count: 1,
    passage_count: 2,
    ready_count: 1,
    pending_count: 0,
    failed_count: 0,
    total_bytes: 10,
    documents_by_media_type: { text: 1 },
  };

  it("filters the activity chart by media type, and reads no server config", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    api.listKnowledgeDocuments.mockResolvedValue({ documents: [doc("a", "ready")], total: 1 });
    api.getKnowledgeDocument.mockResolvedValue({ ...doc("a", "ready"), passages: [passage(0)] });
    api.getKnowledgeTimeseries.mockResolvedValue({ buckets: [] });
    render(<KnowledgeBankHome bank={bank} onOpenDocument={() => {}} />);

    await waitFor(() =>
      expect(api.getKnowledgeTimeseries).toHaveBeenCalledWith("docs", "7d", undefined)
    );
    fireEvent.keyDown(screen.getByRole("combobox", { name: "kindFilter" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("option", { name: /image/ }));
    await waitFor(() =>
      expect(api.getKnowledgeTimeseries).toHaveBeenLastCalledWith("docs", "7d", "image")
    );
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it("links its cards to their pages", async () => {
    api.listKnowledgeDocuments.mockResolvedValue({ documents: [doc("a b", "ready")], total: 1 });
    api.getKnowledgeDocument.mockResolvedValue({ ...doc("a b", "ready"), passages: [] });
    api.getKnowledgeTimeseries.mockResolvedValue({ buckets: [] });
    render(<KnowledgeBankHome bank={{ ...bank, failed_count: 2 }} onOpenDocument={() => {}} />);

    const recent = await screen.findByText("a b.md");
    expect(recent.closest("a")?.getAttribute("href")).toBe(
      "/knowledge-banks/docs?view=home&doc=a%20b"
    );
    expect(screen.getByText("failedNotice").closest("a")?.getAttribute("href")).toBe(
      "/knowledge-banks/docs?view=documents&status=failed"
    );
    expect(screen.getByText("image").closest("a")?.getAttribute("href")).toBe(
      "/knowledge-banks/docs?view=documents&kind=image"
    );
  });
});

describe("buildKnowledgeGraph", () => {
  it("makes each document a hub with its passages linked to it", () => {
    const g = buildKnowledgeGraph(
      [
        { ...doc("a", "ready"), passages: [passage(0), { ...passage(1), text: null }] },
        { ...doc("b", "ready", "image"), passages: [passage(0)] },
      ] as never,
      { kind: (k) => k, passage: (p) => `#${p.index}` }
    );
    expect(g.nodes.map((n) => n.id)).toEqual(["doc:a", "doc:a#0", "doc:a#1", "doc:b", "doc:b#0"]);
    expect(g.nodes.map((n) => n.group)).toEqual(["a", "a", "a", "b", "b"]);
    expect(g.nodes[2].label).toBe("#1");
    expect(g.links).toEqual([
      { source: "doc:a#0", target: "doc:a", type: "passage" },
      { source: "doc:a#1", target: "doc:a", type: "passage" },
      { source: "doc:b#0", target: "doc:b", type: "passage" },
    ]);
  });
});
