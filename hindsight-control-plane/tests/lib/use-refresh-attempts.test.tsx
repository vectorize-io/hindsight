// @vitest-environment jsdom
/**
 * The in-flight refreshes a view reads to tell "still retrying" from "paused" (#4532).
 *
 * The operations list carries every refresh the bank has run, so what matters is
 * which rows it keeps: only the ones still to happen, keyed by the model they
 * belong to, with the next attempt time the UI shows.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";

const listOperations = vi.fn();

vi.mock("@/lib/api", () => ({
  client: {
    listOperations: (...args: unknown[]) => listOperations(...args),
  },
}));

import { useRefreshAttempts } from "@/lib/use-refresh-attempts";

afterEach(() => {
  cleanup();
  listOperations.mockReset();
});

const op = (mentalModelId: string | null, status: string, nextRetryAt: string | null = null) => ({
  id: `op-${mentalModelId}-${status}`,
  task_type: "refresh_mental_model",
  items_count: 0,
  document_id: null,
  created_at: "2026-09-23T08:00:00Z",
  status,
  error_message: null,
  mental_model_id: mentalModelId,
  next_retry_at: nextRetryAt,
});

describe("useRefreshAttempts", () => {
  it("keeps only the refreshes that have still to run, with their next attempt", async () => {
    listOperations.mockResolvedValue({
      operations: [
        op("mm-waiting", "pending", "2026-09-23T08:05:00Z"),
        op("mm-running", "processing"),
        op("mm-done", "completed"),
        op("mm-broken", "failed"),
      ],
    });

    const { result } = renderHook(() => useRefreshAttempts("bank-a"));

    await waitFor(() => expect(result.current.size).toBe(2));
    expect(result.current.get("mm-waiting")).toEqual({ nextAttemptAt: "2026-09-23T08:05:00Z" });
    expect(result.current.get("mm-running")).toEqual({ nextAttemptAt: null });
    expect(result.current.has("mm-done")).toBe(false);
    expect(result.current.has("mm-broken")).toBe(false);
  });

  it("asks only for refresh operations, and only for the current bank", async () => {
    listOperations.mockResolvedValue({ operations: [] });

    renderHook(() => useRefreshAttempts("bank-a"));

    await waitFor(() => expect(listOperations).toHaveBeenCalled());
    expect(listOperations).toHaveBeenCalledWith("bank-a", {
      type: "refresh_mental_model",
      limit: 100,
    });
  });

  it("asks for nothing without a bank", async () => {
    const { result } = renderHook(() => useRefreshAttempts(null));

    await waitFor(() => expect(result.current.size).toBe(0));
    expect(listOperations).not.toHaveBeenCalled();
  });

  it("stays empty when the read fails, rather than breaking the view", async () => {
    listOperations.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => useRefreshAttempts("bank-a"));

    await waitFor(() => expect(listOperations).toHaveBeenCalled());
    expect(result.current.size).toBe(0);
  });
});
