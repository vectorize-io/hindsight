/**
 * The refresh loop shared by foreground `mount` and the background daemon.
 * Runs an immediate sync, then repeats every `intervalSeconds` until aborted.
 */

import { runSync } from "./sync.js";
import type { MountConfig } from "./config.js";

export type LoopLogger = (message: string) => void;

export interface RunLoopOptions {
  signal?: AbortSignal;
  log?: LoopLogger;
}

const delay = (ms: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true }
    );
  });

export async function runLoop(config: MountConfig, opts: RunLoopOptions = {}): Promise<void> {
  const log = opts.log ?? (() => {});
  const signal = opts.signal;

  log(
    `mounting bank "${config.bankId}" at ${config.dir} (every ${config.intervalSeconds}s, ${config.apiUrl})`
  );

  while (!signal?.aborted) {
    try {
      const result = await runSync(config);
      const reverted = result.reverted > 0 ? `, ${result.reverted} reverted` : "";
      log(
        `synced ${result.total} pages / ${result.folders} folders — ${result.written} updated, ` +
          `${result.unchanged} unchanged, ${result.removed} removed${reverted}`
      );
    } catch (err) {
      log(`sync failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (signal?.aborted) break;
    await delay(config.intervalSeconds * 1000, signal);
  }

  log("mount stopped");
}
