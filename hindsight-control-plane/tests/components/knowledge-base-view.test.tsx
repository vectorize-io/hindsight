// @vitest-environment jsdom
/**
 * Regression tests for #3807: the Knowledge view must never request a page or a
 * mental model id under a bank that doesn't own it — neither while switching
 * banks (the previous bank's tree is still on screen) nor from a `?page=` deep
 * link aimed at another bank.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import type { KnowledgeNode } from "@/lib/api";

const searchParams = new URLSearchParams();
let currentBank: string | null = "bank-a";

vi.mock("next/navigation", () => ({
  useSearchParams: () => searchParams,
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/banks/bank-a",
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank }),
}));

const getKnowledgeTree = vi.fn();
const getKnowledgePage = vi.fn();
const getMentalModel = vi.fn();

vi.mock("@/lib/api", () => ({
  client: {
    getKnowledgeTree: (...args: unknown[]) => getKnowledgeTree(...args),
    getKnowledgePage: (...args: unknown[]) => getKnowledgePage(...args),
    getMentalModel: (...args: unknown[]) => getMentalModel(...args),
    searchKnowledgePages: vi.fn(),
  },
}));

// The page body renderer pulls in the markdown stack, which this test doesn't
// exercise; the modal is only mounted from a click.
vi.mock("@/components/compact-markdown", () => ({
  CompactMarkdown: ({ children }: { children?: string }) => <div>{children}</div>,
}));
vi.mock("@/components/mental-model-detail-modal", () => ({
  MentalModelDetailModal: () => null,
}));

const { KnowledgeBaseView } = await import("@/components/knowledge-base-view");

function pageNode(bank: string, suffix: string, name: string): KnowledgeNode {
  return {
    id: `kp-${bank}-${suffix}`,
    kind: "page",
    name,
    parent_id: null,
    mental_model_id: `mm-${bank}-${suffix}`,
    managed: false,
    description: null,
    tags: [],
    timestamp: null,
    is_stale: null,
    trigger: null,
    children: [],
  };
}

const TREES: Record<string, KnowledgeNode[]> = {
  "bank-a": [pageNode("bank-a", "1", "A first"), pageNode("bank-a", "2", "A second")],
  "bank-b": [pageNode("bank-b", "1", "B first")],
};

/**
 * Requests whose id belongs to a different bank than the one they were sent
 * under — the 404s reported in #3807. Ids embed their owning bank, so a plain
 * substring check is enough.
 */
function crossBankRequests(): string[] {
  const calls = [
    ...getKnowledgePage.mock.calls.map(([bank, id]) => ({ kind: "page", bank, id })),
    ...getMentalModel.mock.calls.map(([bank, id]) => ({ kind: "mental model", bank, id })),
  ];
  return calls
    .filter(({ bank, id }) => !String(id).includes(String(bank)))
    .map(({ kind, bank, id }) => `${kind} ${id} requested under ${bank}`);
}

beforeEach(() => {
  currentBank = "bank-a";
  searchParams.delete("page");
  getKnowledgeTree.mockReset();
  getKnowledgePage.mockReset();
  getMentalModel.mockReset();
  getKnowledgeTree.mockImplementation(async (bank: string) => ({ roots: TREES[bank] ?? [] }));
  getKnowledgePage.mockImplementation(async (bank: string, id: string) => {
    if (!id.includes(bank)) throw new Error("404 - page not found in this bank");
    return {
      id,
      name: id,
      type: "page",
      description: null,
      tags: [],
      timestamp: null,
      body: "body",
      markdown: "markdown",
    };
  });
  getMentalModel.mockImplementation(async (bank: string, id: string) => {
    if (!id.includes(bank)) throw new Error("404 - mental model not found in this bank");
    return { reflect_response: { based_on: { world: [], experience: [], observation: [] } } };
  });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("KnowledgeBaseView bank scoping", () => {
  it("auto-opens the first page of the bank and reads its mental model", async () => {
    render(<KnowledgeBaseView />);

    await waitFor(() => expect(getKnowledgePage).toHaveBeenCalledWith("bank-a", "kp-bank-a-1"));
    await waitFor(() => expect(getMentalModel).toHaveBeenCalledWith("bank-a", "mm-bank-a-1"));
    expect(crossBankRequests()).toEqual([]);
  });

  it("never pairs the new bank with the previous bank's page or mental model", async () => {
    const { rerender } = render(<KnowledgeBaseView />);
    await waitFor(() => expect(getMentalModel).toHaveBeenCalledWith("bank-a", "mm-bank-a-1"));

    currentBank = "bank-b";
    rerender(<KnowledgeBaseView />);

    await waitFor(() => {
      expect(crossBankRequests()).toEqual([]);
      expect(getKnowledgePage).toHaveBeenCalledWith("bank-b", "kp-bank-b-1");
      expect(getMentalModel).toHaveBeenCalledWith("bank-b", "mm-bank-b-1");
    });
  });

  it("ignores a ?page= id the bank doesn't own and falls back to its first page", async () => {
    searchParams.set("page", "kp-bank-a-1");
    currentBank = "bank-b";
    render(<KnowledgeBaseView />);

    await waitFor(() => {
      expect(crossBankRequests()).toEqual([]);
      expect(getKnowledgePage).toHaveBeenCalledWith("bank-b", "kp-bank-b-1");
    });
    // The content pane shows the fallback page rather than staying empty.
    await waitFor(() => expect(screen.getAllByText("kp-bank-b-1").length).toBeGreaterThan(0));
  });

  it("opens a ?page= id the bank does own, without also auto-selecting the first page", async () => {
    searchParams.set("page", "kp-bank-a-2");
    render(<KnowledgeBaseView />);

    await waitFor(() => expect(getKnowledgePage).toHaveBeenCalledWith("bank-a", "kp-bank-a-2"));
    expect(getKnowledgePage).not.toHaveBeenCalledWith("bank-a", "kp-bank-a-1");
  });
});
