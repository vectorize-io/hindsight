/**
 * Retry policy for idempotent calls when the server is at capacity.
 *
 * Mirrors tests/test_retry_on_capacity.py in the Python wrapper: the two
 * maintained wrappers are expected to expose the same behaviour.
 */
import { retryAfterMs, retryOnCapacity } from "../src/index";

function res(status: number, retryAfter?: string): { response: Response } {
  const headers = new Headers();
  if (retryAfter !== undefined) headers.set("retry-after", retryAfter);
  return { response: { status, headers } as unknown as Response };
}

describe("retryAfterMs", () => {
  it("reads delta-seconds", () => {
    expect(retryAfterMs(res(503, "7").response)).toBe(7000);
  });

  it("returns null when absent", () => {
    expect(retryAfterMs(res(503).response)).toBeNull();
  });

  it("falls back on the HTTP-date form", () => {
    expect(retryAfterMs(res(503, "Wed, 21 Oct 2026 07:28:00 GMT").response)).toBeNull();
  });

  it("clamps a negative value", () => {
    expect(retryAfterMs(res(503, "-5").response)).toBe(0);
  });
});

describe("retryOnCapacity", () => {
  it("does not retry a success", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return res(200);
    };
    await retryOnCapacity(send, 3, () => 0);
    expect(calls).toBe(1);
  });

  it("retries a 503 and returns the eventual success", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return calls < 3 ? res(503, "0") : res(200);
    };
    const out = await retryOnCapacity(send, 3, () => 0);
    expect(calls).toBe(3);
    expect(out.response?.status).toBe(200);
  });

  it("retries a 429 as well", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return calls < 2 ? res(429, "0") : res(200);
    };
    await retryOnCapacity(send, 3, () => 0);
    expect(calls).toBe(2);
  });

  it("gives up after maxAttempts and returns the last response", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return res(503, "0");
    };
    const out = await retryOnCapacity(send, 3, () => 0);
    expect(calls).toBe(3);
    expect(out.response?.status).toBe(503);
  });

  it("does not retry other failures", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return res(400);
    };
    await retryOnCapacity(send, 3, () => 0);
    expect(calls).toBe(1);
  });

  it("maxAttempts of 1 disables retrying", async () => {
    let calls = 0;
    const send = async () => {
      calls++;
      return res(503, "0");
    };
    await retryOnCapacity(send, 1, () => 0);
    expect(calls).toBe(1);
  });

  it("jitters the wait within Retry-After", async () => {
    // Two clients handed the same Retry-After must not wake together.
    const waits: number[] = [];
    const realSetTimeout = global.setTimeout;
    // @ts-expect-error test double
    global.setTimeout = (fn: () => void, ms: number) => {
      waits.push(ms);
      return realSetTimeout(fn, 0);
    };
    try {
      for (const r of [0.1, 0.5, 0.9]) {
        await retryOnCapacity(
          async () => res(503, "4"),
          2,
          () => r
        );
      }
    } finally {
      global.setTimeout = realSetTimeout;
    }
    expect(waits.every((w) => w >= 0 && w <= 4000)).toBe(true);
    expect(new Set(waits).size).toBeGreaterThan(1);
  });
});
