import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { scopeClient } from "./index.js";
const memory = {
  results: [{ id: "fixture", text: "A fixture observation", type: "observation" as const }],
};

describe("bank-scoped recall cancellation", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  it("aborts the transport at the deadline even if it ignores cancellation", async () => {
    const recall = vi.fn(
      (_bank: string, _query: string, _options?: { signal?: AbortSignal }) =>
        new Promise<never>(() => {})
    );
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    const pending = scoped.recall({ query: "question" }, 1000).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(1000);
    // The caller classifies with `error instanceof DOMException && name === "TimeoutError"`
    // (index.ts), so the class matters: a plain Error with the name reassigned would
    // satisfy toMatchObject and turn every auto-recall timeout into a logged error.
    expect(await pending).toBeInstanceOf(DOMException);
    expect(await pending).toMatchObject({ name: "TimeoutError" });
    expect(recall.mock.calls[0]?.[2]?.signal?.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("clears the deadline after success", async () => {
    const recall = vi.fn().mockResolvedValue(memory);
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    expect(await scoped.recall({ query: "question" }, 1000)).toEqual(memory);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("clears the deadline after failure", async () => {
    const error = new Error("transport failed");
    const recall = vi.fn().mockRejectedValue(error);
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    await expect(scoped.recall({ query: "question" }, 1000)).rejects.toBe(error);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("keeps recall without a deadline working", async () => {
    const recall = vi.fn().mockResolvedValue(memory);
    const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
    expect(await scoped.recall({ query: "question" })).toEqual(memory);
    expect(recall.mock.calls[0]?.[2]?.signal.aborted).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });
});
