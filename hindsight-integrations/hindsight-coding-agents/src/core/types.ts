/**
 * Harness abstraction. A "harness" is a coding agent (opencode, claude-code, cursor, …). Each one
 * differs in exactly two places; everything else is shared core:
 *   1. how its past sessions are read for BACKFILL   -> ChatReader
 *   2. how its live runtime hooks bind reflect+inject -> HarnessAdapter.createRuntime
 * A new harness = one file implementing this interface, registered in harness/registry.ts.
 */
import type { RuntimeCore } from "./runtime";

/** A normalized past session (harness-independent), as produced by a ChatReader for backfill. */
export interface ChatSession {
  id?: string;
  turns: { role: string; text: string; timestamp?: string }[];
}

/** Reads a harness's past sessions into normalized ChatSessions for the backfill. */
export interface ChatReader {
  /** Locate + parse this harness's sessions. `conversations` is an optional pre-exported JSON file. */
  read(opts: { conversations?: string; repo?: string }): Promise<ChatSession[]>;
  /** One-line help shown in the CLI for this harness's chat source. */
  readonly describe: string;
}

/** Binds a specific agent's plugin API to the shared RuntimeCore. */
export interface HarnessAdapter {
  readonly name: string;
  /** How this harness's past sessions are ingested during backfill. */
  readonly chatReader: ChatReader;
  /**
   * Build the agent-native runtime entry from a RuntimeCore. The return type is harness-specific
   * (opencode: a Plugin hooks object), so it is intentionally `unknown` at the interface boundary —
   * the harness's own entrypoint casts it to that agent's expected shape.
   */
  createRuntime(core: RuntimeCore): unknown;
}
