/** Small shared helpers (no harness or Hindsight coupling). */

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Bounded-concurrency map: run `fn` over `items`, at most `n` in flight. Never rejects on item error. */
export async function pool<T>(
  items: T[],
  n: number,
  fn: (x: T, i: number) => Promise<void>,
  onError?: (i: number, e: unknown) => void,
  onProgress?: (done: number, total: number) => void,
): Promise<void> {
  let i = 0, done = 0;
  async function worker() {
    while (i < items.length) {
      const idx = i++;
      try {
        await fn(items[idx], idx);
      } catch (e) {
        onError?.(idx, e);
      }
      onProgress?.(++done, items.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, worker));
}
