/**
 * Hindsight OpenClaw plugin logger.
 *
 * Routes output through OpenClaw's api.logger for consistent formatting
 * with other plugins (same colors/timestamps as mem0, etc.).
 *
 * Features:
 *  - Configurable log level: 'off' | 'error' | 'warning' | 'info' | 'debug'
 *  - Batched retain/recall summaries instead of per-event spam
 */

export type LogLevel = "off" | "error" | "warning" | "info" | "debug";

export interface LoggerConfig {
  /** Minimum severity to print. Default: 'info' */
  logLevel?: LogLevel;
  /** Interval in ms to print batched retain/recall summaries. 0 = print every event. Default: 300000 (5 min) */
  logSummaryIntervalMs?: number;
}

// Muted blue (38;5;103 = slate/dusty blue from 256-color palette)
const PREFIX = "\x1b[38;5;103mhindsight:\x1b[0m";

const LEVEL_RANK: Record<LogLevel, number> = {
  off: 0,
  error: 1,
  warning: 2,
  info: 3,
  debug: 4,
};

// Output backend — set via setApiLogger, falls back to console
let apiLogger: { info(msg: string): void; warn(msg: string): void; error(msg: string): void } = {
  info: (msg) => console.log(msg),
  warn: (msg) => console.warn(msg),
  error: (msg) => console.error(msg),
};

// Batched summary state
let retainCount = 0;
let retainMsgTotal = 0;
let recallCount = 0;
let recallMemoriesCount = 0;
const banksSeen = new Set<string>();
let lastSummaryTime = Date.now();
let summaryTimer: ReturnType<typeof setInterval> | null = null;

let currentLevel: LogLevel = "info";
let currentSummaryIntervalMs = 300_000; // 5 min

/** Bind to OpenClaw's api.logger for consistent output formatting */
export function setApiLogger(logger: {
  info(msg: string): void;
  warn(msg: string): void;
  error(msg: string): void;
}): void {
  apiLogger = logger;
}

export function configureLogger(cfg: LoggerConfig): void {
  currentLevel = cfg.logLevel ?? "info";
  currentSummaryIntervalMs = cfg.logSummaryIntervalMs ?? 300_000;

  // Restart summary timer
  if (summaryTimer) {
    clearInterval(summaryTimer);
    summaryTimer = null;
  }
  if (currentSummaryIntervalMs > 0 && LEVEL_RANK[currentLevel] >= LEVEL_RANK["info"]) {
    summaryTimer = setInterval(flushSummary, currentSummaryIntervalMs);
    summaryTimer.unref?.(); // don't keep process alive
  }
}

function allowed(level: LogLevel): boolean {
  return LEVEL_RANK[currentLevel] >= LEVEL_RANK[level];
}

/** Info-level log (requires 'info' or higher) */
export function info(msg: string): void {
  if (!allowed("info")) return;
  apiLogger.info(`${PREFIX} ${msg}`);
}

/** Debug log (requires 'debug') */
export function verbose(msg: string): void {
  if (!allowed("debug")) return;
  apiLogger.info(`${PREFIX} ${msg}`);
}

/** Warning (requires 'warning' or higher) */
export function warn(msg: string): void {
  if (!allowed("warning")) return;
  apiLogger.warn(`${PREFIX} ${msg}`);
}

/** Error (requires 'error' or higher) */
export function error(msg: string, err?: unknown): void {
  if (!allowed("error")) return;
  const detail = err instanceof Error ? err.message : err ? String(err) : "";
  apiLogger.error(`${PREFIX} ${detail ? `${msg}: ${detail}` : msg}`);
}

/** Track a retain event for batched summary */
export function trackRetain(bankId: string, messageCount: number): void {
  retainCount++;
  retainMsgTotal += messageCount;
  banksSeen.add(bankId);
  if (currentSummaryIntervalMs === 0 && allowed("info")) {
    apiLogger.info(`${PREFIX} auto-retained ${messageCount} messages (bank: ${bankId})`);
  }
}

/** Track a recall event for batched summary */
export function trackRecall(bankId: string, memoriesFound: number): void {
  recallCount++;
  recallMemoriesCount += memoriesFound;
  banksSeen.add(bankId);
  // per-event logging is handled by info() call at the injection site
}

/** Flush the batched summary to console */
export function flushSummary(): void {
  if (!allowed("info")) return;
  if (retainCount === 0 && recallCount === 0) return;

  const elapsed = Math.round((Date.now() - lastSummaryTime) / 1000);
  const parts: string[] = [];
  if (recallCount > 0)
    parts.push(`${recallCount} recalls (${recallMemoriesCount} memories injected)`);
  if (retainCount > 0) parts.push(`${retainCount} retains (${retainMsgTotal} messages captured)`);
  const bankList = [...banksSeen];
  const bankLabel = bankList.length === 1 ? "bank" : "banks";
  const banks = bankList.length > 0 ? ` (${bankLabel}: ${bankList.join(", ")})` : "";
  apiLogger.info(`${PREFIX} ${parts.join(", ")} in ${elapsed}s${banks}`);

  retainCount = 0;
  retainMsgTotal = 0;
  recallCount = 0;
  recallMemoriesCount = 0;
  banksSeen.clear();
  lastSummaryTime = Date.now();
}

/** Cleanup (call on plugin stop) */
export function stopLogger(): void {
  flushSummary();
  if (summaryTimer) {
    clearInterval(summaryTimer);
    summaryTimer = null;
  }
}
