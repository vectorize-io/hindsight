/** Small shared helpers (no harness coupling). */
import { accessSync, constants, statSync } from "node:fs";
import { delimiter, extname, isAbsolute, join, resolve } from "node:path";

/**
 * Where `bin` runs from, or undefined when it is not runnable. A path (contains a slash or
 * backslash) -> that file; a bare name -> the first match on PATH. Returns the full path so the
 * caller spawns exactly what was found.
 *
 * On Windows a bare name resolves through PATHEXT (`codex` -> `codex.exe`), never as itself: the
 * extensionless file npm drops next to its shims is a `#!/bin/sh` script spawn cannot run (#5710).
 * `exts` narrows the extensions tried — a caller that spawns without a shell passes [".exe"],
 * since `.cmd`/`.bat` only run through one. A directory is never a match (X_OK accepts those).
 *
 * Resolved by hand rather than by spawning: the callers use this to DECIDE whether to spawn, and
 * `which`/`where` is itself a process launch on a path where the answer is usually "no".
 */
export function binOnPath(
  bin: string,
  opts: { exts?: string[]; platform?: NodeJS.Platform; env?: NodeJS.ProcessEnv } = {}
): string | undefined {
  const platform = opts.platform ?? process.platform;
  const env = opts.env ?? process.env;
  const runnable = (file: string): boolean => {
    try {
      if (!statSync(file).isFile()) return false;
      if (platform !== "win32") accessSync(file, constants.X_OK); // Windows has no exec bit
      return true;
    } catch {
      return false;
    }
  };
  if (isAbsolute(bin) || /[\\/]/.test(bin)) return runnable(bin) ? resolve(bin) : undefined;
  const names =
    platform !== "win32" || extname(bin)
      ? [bin]
      : (opts.exts ?? (env.PATHEXT || ".COM;.EXE;.BAT;.CMD").split(";"))
          .filter(Boolean)
          .map((ext) => bin + ext.toLowerCase());
  for (const dir of (env.PATH || "").split(delimiter)) {
    if (!dir) continue;
    for (const name of names) {
      const file = join(dir, name);
      if (runnable(file)) return resolve(file);
    }
  }
  return undefined;
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

/** Name ownership alone is insufficient: users can legitimately register another server as
 * `hindsight`. Match the exact script shape this package has emitted under its source, npm, and
 * staged directory names; an incidental "coding-agents" string must not grant ownership. */
export function isOurMcpEntry(entry: unknown): boolean {
  if (!entry || typeof entry !== "object") return false;
  const candidate = entry as { command?: unknown; args?: unknown };
  if (candidate.command !== "node" || !Array.isArray(candidate.args)) return false;
  const script = candidate.args[0];
  if (typeof script !== "string") return false;
  const parts = script.replaceAll("\\", "/").split("/").filter(Boolean);
  return (
    parts.at(-1) === "mcp-server.js" &&
    parts.at(-2) === "dist" &&
    (parts.at(-3) === "coding-agents" || parts.at(-3) === "hindsight-coding-agents")
  );
}
