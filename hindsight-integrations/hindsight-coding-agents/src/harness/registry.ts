/**
 * Harness registry. Add a coding agent by implementing HarnessAdapter (one file in this dir) and
 * registering it here — the backfill's --harness flag and the runtime's HINDSIGHT_HARNESS env both
 * resolve through getHarness().
 */
import type { HarnessAdapter } from "../core/types";
import { opencodeAdapter } from "./opencode";

export const HARNESSES: Record<string, HarnessAdapter> = {
  opencode: opencodeAdapter,
  // claude-code, cursor, … : add adapters here.
};

export const HARNESS_NAMES = Object.keys(HARNESSES);

export function getHarness(name: string): HarnessAdapter {
  const a = HARNESSES[name];
  if (!a) {
    throw new Error(`unknown harness '${name}'. available: ${HARNESS_NAMES.join(", ")}`);
  }
  return a;
}
