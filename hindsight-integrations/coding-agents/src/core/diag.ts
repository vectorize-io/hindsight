/**
 * Shared diagnostics helper for hook-based harnesses (Claude Code, Cursor CLI, ...): appends a
 * JSON line to the diagnostic file so a silently memory-less session can't masquerade as one that
 * worked. Never throws — diagnostics must not break the agent.
 *
 * Deliberately neutral (no dependency on recall/reflect or retain specifics) so every lifecycle
 * hook (UserPromptSubmit, Stop, and future ones like SessionStart) can depend on it without
 * reaching into another hook's module.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { appendLogLine, log, logsDir } from "./log";

/** Where the JSONL trail lands. Exported so a failure notice can name the file to open. */
export function diagFilePath(): string {
  return process.env.HINDSIGHT_DIAG_FILE || join(logsDir(), "diag.jsonl");
}

export function diag(harness: string, event: string, extra: Record<string, unknown> = {}): void {
  log.debug(harness, `diag:${event}`, extra); // debug level shows the full story in ONE file
  try {
    appendLogLine(
      diagFilePath(),
      JSON.stringify({ ts: new Date().toISOString(), harness, event, ...extra }) + "\n"
    );
  } catch {
    /* diagnostics must not break the agent */
  }
}

/**
 * Parse a hook's stdin event. Null for no input, and for input that is not a JSON object — the
 * latter is recorded in the diag file, since the hook then exits 0 having done nothing.
 *
 * Leading byte-order marks are dropped first, including the form a UTF-8 BOM takes after a
 * Windows-1252 round trip ("ï»¿"): Cursor on Windows pipes the event through PowerShell, which
 * delivers it that way, and JSON.parse rejected it — so every Cursor hook silently did nothing
 * (#5366).
 */
export function parseHookEvent(harness: string, raw: string): Record<string, unknown> | null {
  const text = raw.replace(/^(?:﻿|ï»¿)+/, "");
  if (!text.trim()) return null;
  try {
    const ev: unknown = JSON.parse(text);
    if (ev && typeof ev === "object" && !Array.isArray(ev)) return ev as Record<string, unknown>;
    diag(harness, "hook_input_invalid", { error: "not a JSON object" });
  } catch (e) {
    diag(harness, "hook_input_invalid", { error: (e as Error).message });
  }
  return null;
}

/** Read and parse this hook invocation's event from stdin (see parseHookEvent). */
export function readHookEvent(harness: string): Record<string, unknown> | null {
  let raw: string;
  try {
    raw = readFileSync(0, "utf8");
  } catch {
    return null; // no stdin at all
  }
  return parseHookEvent(harness, raw);
}
