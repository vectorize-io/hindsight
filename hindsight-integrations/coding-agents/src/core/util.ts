/** Small shared helpers (no harness or Hindsight coupling). */
import { accessSync, constants } from "node:fs";
import { delimiter, join } from "node:path";

/**
 * Is `bin` runnable? A path (contains "/") -> exists + executable; a bare name -> found on PATH.
 *
 * Resolved by hand rather than by spawning: the callers use this to DECIDE whether to spawn, and
 * `which`/`where` is itself a process launch on a path where the answer is usually "no".
 */
export function binOnPath(bin: string): boolean {
  try {
    if (bin.includes("/")) {
      accessSync(bin, constants.X_OK);
      return true;
    }
    for (const dir of (process.env.PATH || "").split(delimiter)) {
      if (!dir) continue;
      try {
        accessSync(join(dir, bin), constants.X_OK);
        return true;
      } catch {
        /* keep scanning PATH */
      }
    }
    return false;
  } catch {
    return false;
  }
}

export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** `version >= min`, comparing numeric major/minor/patch. A missing or unparseable version is
 *  treated as OLDER than anything — capability probes must fail closed, not assume support.
 *  Pre-release/build suffixes ("0.9.0rc1", "0.9.0+dev") compare by their numeric part. */
export function semverGte(version: string | undefined, min: string): boolean {
  const parts = (v: string) => {
    const m = v.trim().match(/^(\d+)\.(\d+)(?:\.(\d+))?/);
    return m ? [Number(m[1]), Number(m[2]), Number(m[3] ?? 0)] : undefined;
  };
  const a = version ? parts(version) : undefined;
  const b = parts(min);
  if (!a || !b) return false;
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] > b[i];
  }
  return true;
}

/** Bounded-concurrency map: run `fn` over `items`, at most `n` in flight. Never rejects on item error. */
export async function pool<T>(
  items: T[],
  n: number,
  fn: (x: T, i: number) => Promise<void>,
  onError?: (i: number, e: unknown) => void,
  onProgress?: (done: number, total: number) => void
): Promise<void> {
  let i = 0,
    done = 0;
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
