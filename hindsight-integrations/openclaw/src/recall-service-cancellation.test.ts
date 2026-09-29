import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { HindsightClient } from "@vectorize-io/hindsight-client";
import { scopeClient } from "./index.js";

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

it("does not send with an already-stopped service signal", async () => {
  const controller = new AbortController();
  controller.abort();
  const recall = vi.fn();
  const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
  await expect(scoped.recall({ query: "question" }, 1000, controller.signal)).rejects.toBe(
    controller.signal.reason
  );
  expect(recall).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});

it("settles on stop even if transport never returns and cleans up", async () => {
  const controller = new AbortController();
  const recall = vi.fn(
    (_bank: string, _query: string, _options?: { signal?: AbortSignal }) =>
      new Promise<never>(() => {})
  );
  const scoped = scopeClient({ recall } as unknown as HindsightClient, "test-bank");
  const pending = scoped
    .recall({ query: "question" }, 1000, controller.signal)
    .catch((error: unknown) => error);
  const forwarded = recall.mock.calls[0]?.[2]?.signal;
  expect(forwarded).toBeDefined();
  const remove = vi.spyOn(forwarded!, "removeEventListener");
  controller.abort();
  expect(await pending).toBe(controller.signal.reason);
  expect(remove).toHaveBeenCalledWith("abort", expect.any(Function));
  expect(vi.getTimerCount()).toBe(0);
});
