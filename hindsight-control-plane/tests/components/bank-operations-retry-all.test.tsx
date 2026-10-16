// @vitest-environment jsdom
/**
 * #5498: "Retry all" on the Background Operations list. It only appears under
 * the Failed filter, retries just the failed rows on the current page, oldest
 * first, and keeps going when one retry call fails.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key} ${JSON.stringify(values)}` : key,
}));

vi.mock("@/lib/bank-context", () => ({
  useBank: () => ({ currentBank: "bank-a" }),
}));

const toastSuccess = vi.fn();
const toastError = vi.fn();
vi.mock("sonner", () => ({
  toast: { success: (...a: unknown[]) => toastSuccess(...a), error: (...a: unknown[]) => toastError(...a) },
}));

const listOperations = vi.fn();
const retryOperation = vi.fn();
vi.mock("@/lib/api", () => ({
  client: {
    listOperations: (...args: unknown[]) => listOperations(...args),
    retryOperation: (...args: unknown[]) => retryOperation(...args),
    cancelOperation: vi.fn(),
    deleteOperation: vi.fn(),
    getOperationStatus: vi.fn(),
  },
}));

vi.mock("@/components/document-chunk-modal", () => ({ DocumentChunkModal: () => null }));

const { BankOperationsView } = await import("@/components/bank-operations-view");

function op(id: string, createdAt: string) {
  return {
    id,
    task_type: "file_convert_retain",
    items_count: 1,
    document_id: null,
    created_at: createdAt,
    status: "failed",
    error_message: "boom",
  };
}

// Newest first, the way the API lists them.
const FAILED = [
  op("op-new", "2026-10-16T10:00:03Z"),
  op("op-mid", "2026-10-16T10:00:02Z"),
  op("op-old", "2026-10-16T10:00:01Z"),
];

beforeEach(() => {
  listOperations.mockImplementation(async (_bank: string, opts: { status?: string }) =>
    opts.status === "failed" ? { operations: FAILED, total: FAILED.length } : { operations: [], total: 0 }
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

async function openFailedFilter() {
  render(<BankOperationsView />);
  fireEvent.click(await screen.findByText("status.failed"));
  return screen.findByText(/action\.retryAll/);
}

describe("BankOperationsView retry all", () => {
  it("is hidden until the Failed filter is selected", async () => {
    render(<BankOperationsView />);
    await waitFor(() => expect(listOperations).toHaveBeenCalled());
    expect(screen.queryByText(/action\.retryAll/)).toBeNull();
  });

  it("retries the failed rows on the page, oldest first", async () => {
    retryOperation.mockResolvedValue({});
    const button = await openFailedFilter();
    expect(button.textContent).toContain('{"count":3}');

    fireEvent.click(button);

    await waitFor(() => expect(toastSuccess).toHaveBeenCalled());
    expect(retryOperation.mock.calls.map((c) => c[1])).toEqual(["op-old", "op-mid", "op-new"]);
    expect(toastSuccess).toHaveBeenCalledWith('retryAllDone {"done":3,"total":3}');
  });

  it("keeps going past a failed retry and reports the partial count", async () => {
    retryOperation.mockImplementation(async (_bank: string, id: string) => {
      if (id === "op-mid") throw new Error("409");
      return {};
    });
    fireEvent.click(await openFailedFilter());

    await waitFor(() => expect(toastError).toHaveBeenCalled());
    expect(retryOperation).toHaveBeenCalledTimes(3);
    expect(toastError).toHaveBeenCalledWith('retryAllDone {"done":2,"total":3}');
  });
});
