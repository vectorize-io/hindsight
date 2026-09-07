// @vitest-environment jsdom
/**
 * The overview's bulk delete is the one destructive path in the control plane that
 * touches many banks at once, so it has two properties worth pinning:
 *
 *   1. Deletes run STRICTLY SEQUENTIALLY. `delete_bank` drops the bank's partial
 *      vector indexes with DROP INDEX CONCURRENTLY on the shared `memory_units`
 *      relation; concurrent drops deadlock, so a fan-out would fail banks at random.
 *   2. One failure does not abort the run, and the failed banks are reported and
 *      left selected rather than silently dropped.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

const push = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  usePathname: () => "/dashboard",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("next-intl", () => ({
  // Keys are echoed back; ICU arguments are appended so counts stay assertable.
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${Object.values(values).join(",")}` : key,
}));

const toastSuccess = vi.fn();
const toastWarning = vi.fn();
vi.mock("sonner", () => ({
  toast: { success: (...a: unknown[]) => toastSuccess(...a), warning: (...a: unknown[]) => toastWarning(...a) },
}));

const deleteBank = vi.fn();
const listBanks = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    deleteBank: (...args: unknown[]) => deleteBank(...args),
    listBanks: (...args: unknown[]) => listBanks(...args),
  },
}));

const setCurrentBank = vi.fn();
const loadBanks = vi.fn().mockResolvedValue(undefined);

function bank(id: string) {
  return {
    bank_id: id,
    name: null,
    mission: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: null,
    fact_count: 10,
    last_document_at: null,
    last_write_at: "2026-01-02T00:00:00Z",
  };
}

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank: null, setCurrentBank, loadBanks }),
}));

const { BanksOverview } = await import("@/components/banks-overview");

/** Selects every row, opens the confirm dialog and types the confirmation word. */
async function armBulkDelete() {
  render(<BanksOverview />);
  fireEvent.click(await screen.findByLabelText("selectAll"));
  fireEvent.click(screen.getByText("deleteSelected"));
  // The mocked `t` echoes keys, so the confirmation word is the key itself.
  const input = await screen.findByLabelText(/bulkDeleteConfirmLabel/);
  fireEvent.change(input, { target: { value: "bulkDeleteConfirmWord" } });
}

beforeEach(() => {
  vi.clearAllMocks();
  listBanks.mockResolvedValue({
    banks: [bank("alpha"), bank("beta"), bank("gamma")],
    total: 3,
    limit: 50,
    offset: 0,
  });
});

afterEach(() => cleanup());

describe("BanksOverview bulk delete", () => {
  it("deletes the selected banks one at a time, never concurrently", async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const resolvers: Array<() => void> = [];
    deleteBank.mockImplementation(() => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<void>((resolve) => {
        resolvers.push(() => {
          inFlight -= 1;
          resolve();
        });
      });
    });

    await armBulkDelete();
    fireEvent.click(screen.getAllByText("deleteSelected")[1]);

    // Drain the queue one request at a time; if the component fanned out, more than
    // one call would already be pending before the first resolves.
    for (let i = 0; i < 3; i++) {
      await waitFor(() => expect(deleteBank).toHaveBeenCalledTimes(i + 1));
      expect(maxInFlight).toBe(1);
      resolvers[i]();
    }

    await waitFor(() => expect(toastSuccess).toHaveBeenCalled());
    expect(deleteBank.mock.calls.map((c) => c[0])).toEqual(["alpha", "beta", "gamma"]);
    expect(loadBanks).toHaveBeenCalled();
  });

  it("keeps going after a failure and reports the banks that survived", async () => {
    deleteBank.mockImplementation((bankId: string) =>
      bankId === "beta" ? Promise.reject(new Error("bank is busy")) : Promise.resolve()
    );

    await armBulkDelete();
    fireEvent.click(screen.getAllByText("deleteSelected")[1]);

    await waitFor(() => expect(toastWarning).toHaveBeenCalledWith("bulkDeletePartial:2,1"));
    // All three were attempted — the rejection did not abort the loop.
    expect(deleteBank).toHaveBeenCalledTimes(3);
    expect(await screen.findByText("bulkDeleteFailures")).toBeTruthy();
    expect(screen.getByText(/bank is busy/)).toBeTruthy();
    expect(toastSuccess).not.toHaveBeenCalled();
  });

  it("suppresses the client's per-request error toast so failures are reported once", async () => {
    deleteBank.mockResolvedValue(undefined);

    await armBulkDelete();
    fireEvent.click(screen.getAllByText("deleteSelected")[1]);

    await waitFor(() => expect(deleteBank).toHaveBeenCalled());
    expect(deleteBank).toHaveBeenCalledWith("alpha", { suppressErrorToast: true });
  });

  it("fetches the next page by offset and drops the selection when the page changes", async () => {
    // 120 banks over a page size of 50 => three pages, so the pager renders.
    listBanks.mockImplementation(({ offset }: { offset: number }) =>
      Promise.resolve({
        banks: [bank(`row-${offset}-a`), bank(`row-${offset}-b`)],
        total: 120,
        limit: 50,
        offset,
      })
    );

    render(<BanksOverview />);
    fireEvent.click(await screen.findByLabelText("selectAll"));
    expect(screen.getByText("selectedCount:2")).toBeTruthy();

    const nextButton = document
      .querySelector(".lucide-chevron-right")
      ?.closest("button") as HTMLButtonElement;
    fireEvent.click(nextButton);

    await waitFor(() => expect(listBanks).toHaveBeenCalledWith({ q: undefined, limit: 50, offset: 50 }));
    // A selection made on page 1 must not survive onto page 2 — those rows are no
    // longer on screen and would be deleted unseen.
    await waitFor(() => expect(screen.queryByText(/^selectedCount/)).toBeNull());
  });

  it("greets instead of showing an empty table when the server has no banks", async () => {
    listBanks.mockResolvedValue({ banks: [], total: 0, limit: 50, offset: 0 });

    render(<BanksOverview />);

    expect(await screen.findByText("welcomeTitle")).toBeTruthy();
    // The search box and table are meaningless with nothing to search or list.
    expect(screen.queryByPlaceholderText("search")).toBeNull();
    expect(document.querySelector("table")).toBeNull();
  });

  it("keeps the search box when a query matches nothing, rather than greeting", async () => {
    render(<BanksOverview />);
    await screen.findByLabelText("selectAll");

    // An empty result for a query is not an empty server — the box must stay so the
    // query can be edited.
    listBanks.mockResolvedValue({ banks: [], total: 0, limit: 50, offset: 0 });
    fireEvent.change(screen.getByPlaceholderText("search"), { target: { value: "nope" } });

    expect(await screen.findByText("noSearchResults")).toBeTruthy();
    expect(screen.queryByText("welcomeTitle")).toBeNull();
    expect(screen.getByPlaceholderText("search")).toBeTruthy();
  });

  it("requires the confirmation word before the delete can run", async () => {
    render(<BanksOverview />);
    fireEvent.click(await screen.findByLabelText("selectAll"));
    fireEvent.click(screen.getByText("deleteSelected"));

    const action = screen.getAllByText("deleteSelected")[1].closest("button");
    expect(action?.disabled).toBe(true);

    fireEvent.change(await screen.findByLabelText(/bulkDeleteConfirmLabel/), {
      target: { value: "bulkDeleteConfirmWord" },
    });
    expect(action?.disabled).toBe(false);
  });
});
