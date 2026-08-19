/**
 * Leveled plugin logging — ONE human-readable log file for debugging, next to (not replacing) the
 * structured diag JSONL contract (core/diag.ts, which benchmarks/harnesses parse).
 *
 *   file : $TMPDIR/hindsight-coding-agent/plugin.log   (override: HINDSIGHT_LOG_FILE)
 *   level: "info" default — config `logLevel`, or HINDSIGHT_LOG_LEVEL for ad-hoc debugging
 *          without touching config ("debug" | "info" | "warn" | "error")
 *
 * Line format: `<iso> LEVEL [scope] message {extra-json}` — greppable, tail-able. At "debug",
 * every diag event is mirrored here too, so one file tells the whole story. Never throws: logging
 * must not break the agent.
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

export type LogLevel = "debug" | "info" | "warn" | "error";
const WEIGHT: Record<LogLevel, number> = { debug: 10, info: 20, warn: 30, error: 40 };

let current: LogLevel =
  (["debug", "info", "warn", "error"] as const).find(
    (l) => l === process.env.HINDSIGHT_LOG_LEVEL
  ) ?? "info";

/** Entry points call this after loadConfig; the HINDSIGHT_LOG_LEVEL env override still wins. */
export function setLogLevel(level: LogLevel): void {
  if (!process.env.HINDSIGHT_LOG_LEVEL) current = level;
}

export function logFilePath(): string {
  return process.env.HINDSIGHT_LOG_FILE || join(tmpdir(), "hindsight-coding-agent", "plugin.log");
}

function write(level: LogLevel, scope: string, msg: string, extra?: Record<string, unknown>): void {
  if (WEIGHT[level] < WEIGHT[current]) return;
  try {
    const file = logFilePath();
    mkdirSync(dirname(file), { recursive: true });
    appendFileSync(
      file,
      `${new Date().toISOString()} ${level.toUpperCase().padEnd(5)} [${scope}] ${msg}` +
        (extra && Object.keys(extra).length ? ` ${JSON.stringify(extra)}` : "") +
        "\n"
    );
  } catch {
    /* logging must never break the agent */
  }
}

export const log = {
  debug: (scope: string, msg: string, extra?: Record<string, unknown>) =>
    write("debug", scope, msg, extra),
  info: (scope: string, msg: string, extra?: Record<string, unknown>) =>
    write("info", scope, msg, extra),
  warn: (scope: string, msg: string, extra?: Record<string, unknown>) =>
    write("warn", scope, msg, extra),
  error: (scope: string, msg: string, extra?: Record<string, unknown>) =>
    write("error", scope, msg, extra),
};

/**
 * Render a thrown value with its `cause` chain.
 *
 * Node's fetch reports EVERY transport failure as the bare string "fetch failed" and hides the real
 * reason (ECONNREFUSED, ENOTFOUND, a TLS error, a timeout) on `error.cause`. Logging only the
 * message turned a one-line diagnosis — "nothing is listening on the configured apiUrl" — into an
 * investigation, so every network failure we record goes through this instead.
 */
export function describeError(value: unknown, maxChars = 200): string {
  const parts: string[] = [];
  const seen = new Set<unknown>();
  let current: unknown = value;
  while (current && !seen.has(current)) {
    seen.add(current);
    const error = current as { message?: unknown; code?: unknown; cause?: unknown };
    const text = typeof error.message === "string" ? error.message : String(current);
    // A cause usually carries the code (ECONNREFUSED) the wrapper dropped; keep both, skip repeats.
    const withCode = error.code ? `${text} (${String(error.code)})` : text;
    if (withCode && parts.at(-1) !== withCode) parts.push(withCode);
    current = error.cause;
  }
  return (parts.join(": ") || String(value)).slice(0, maxChars);
}
