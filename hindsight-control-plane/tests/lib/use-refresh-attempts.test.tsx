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

const op = (
  mentalModelId: string | null,
  status: string,
  nextRetryAt: string | null = null,
  retryCount = 0,
  errorMessage: string | null = null
) => ({
  id: `op-${mentalModelId}-${status}`,
  task_type: "refresh_mental_model",
  items_count: 0,
  document_id: null,
  created_at: "2026-09-23T08:00:00Z",
  status,
  error_message: errorMessage,
  retry_count: retryCount,
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
    expect(result.current.get("mm-waiting")).toEqual({
      running: false,
      retry: false,
      nextAttemptAt: "2026-09-23T08:05:00Z",
    });
    expect(result.current.get("mm-running")).toEqual({
      running: true,
      retry: false,
      nextAttemptAt: null,
    });
    expect(result.current.has("mm-done")).toBe(false);
    expect(result.current.has("mm-broken")).toBe(false);
  });

  it("calls a refresh a retry only once an attempt of it has failed (#6864)", async () => {
    listOperations.mockResolvedValue({
      operations: [
        // Held back by its minimum interval: a future time, but nothing failed.
        op("mm-scheduled", "pending", "2030-01-01T10:00:00Z"),
        op("mm-first-run", "processing"),
        op("mm-retry-wait", "pending", "2030-01-01T10:00:00Z", 1, "provider down"),
        op("mm-retry-run", "processing", "2026-09-23T08:05:00Z", 2, "provider down"),
        // Re-queued after a worker restart: counted, no error text.
        op("mm-recovered", "pending", null, 1),
      ],
    });

    const { result } = renderHook(() => useRefreshAttempts("bank-a"));

    await waitFor(() => expect(result.current.size).toBe(5));
    expect(result.current.get("mm-scheduled")?.retry).toBe(false);
    expect(result.current.get("mm-first-run")?.retry).toBe(false);
    expect(result.current.get("mm-retry-wait")?.retry).toBe(true);
    expect(result.current.get("mm-recovered")?.retry).toBe(true);
    // next_retry_at outlives the wait, so a running refresh has no next time.
    expect(result.current.get("mm-retry-run")).toEqual({
      running: true,
      retry: true,
      nextAttemptAt: null,
    });
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
