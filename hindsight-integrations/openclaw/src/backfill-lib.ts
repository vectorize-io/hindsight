import { homedir } from "os";
import { dirname, join, resolve } from "path";
import { readFileSync, existsSync, mkdirSync, writeFileSync, readdirSync } from "fs";
import { deriveBankId, isOperationalSessionKey, prepareRetentionTranscript } from "./index.js";
import { compileSessionPatterns, matchesSessionPattern } from "./session-patterns.js";
import type { PluginConfig, PluginHookAgentContext } from "./types.js";
import { parseSessionFile } from "./session-file.js";
import type { ParsedSessionFile, SessionMessage } from "./session-file.js";

// Re-exported: these used to live here and callers import them from this module.
export { parseSessionFile };
export type { ParsedSessionFile, SessionMessage };

export interface BackfillCliOptions {
  openclawRoot: string;
  includeArchive: boolean;
  selectedAgents?: Set<string>;
  limit?: number;
  bankStrategy: "mirror-config" | "agent" | "fixed";
  fixedBank?: string;
}

export interface BackfillPlanEntry {
  filePath: string;
  agentId: string;
  sessionId: string;
  startedAt?: string;
  bankId: string;
  documentId: string;
  transcript: string;
  messageCount: number;
}

export interface BackfillCheckpointEntry {
  status: "enqueued" | "completed" | "failed";
  bankId: string;
  filePath: string;
  sessionId: string;
  updatedAt: string;
  error?: string;
}

export interface BackfillCheckpoint {
  version: 1;
  entries: Record<string, BackfillCheckpointEntry>;
}

interface RawBackfillCheckpointEntry extends Omit<BackfillCheckpointEntry, "status"> {
  status: BackfillCheckpointEntry["status"] | "queued";
}

interface RawBackfillCheckpoint {
  version: 1;
  entries: Record<string, RawBackfillCheckpointEntry>;
}

interface SessionDirectory {
  agentId: string;
  path: string;
}

const DEFAULT_PLUGIN_CONFIG: PluginConfig = {
  dynamicBankId: true,
  retainRoles: ["user", "assistant"],
};

export function defaultOpenClawRoot(): string {
  return resolve(join(homedir(), ".openclaw"));
}

export function defaultCheckpointPath(openclawRoot: string): string {
  return join(openclawRoot, "data", "hindsight-backfill-checkpoint.json");
}

export function loadPluginConfigFromOpenClawRoot(openclawRoot: string): PluginConfig {
  const configPath = join(openclawRoot, "openclaw.json");
  const raw = JSON.parse(readFileSync(configPath, "utf8")) as {
    plugins?: { entries?: Record<string, { config?: PluginConfig }> };
  };
  return {
    ...DEFAULT_PLUGIN_CONFIG,
    ...(raw.plugins?.entries?.["hindsight-openclaw"]?.config || {}),
  };
}

function sessionDirectories(openclawRoot: string, includeArchive: boolean): SessionDirectory[] {
  const agentsRoot = join(openclawRoot, "agents");
  if (!existsSync(agentsRoot)) {
    return [];
  }
  const result: SessionDirectory[] = [];
  for (const entry of readdirSync(agentsRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const agentId = entry.name;
    const sessionsDir = join(agentsRoot, agentId, "sessions");
    if (existsSync(sessionsDir)) {
      result.push({ agentId, path: sessionsDir });
    }
    if (includeArchive) {
      const archiveDir = join(agentsRoot, agentId, "sessions-archive-from-migration_backup");
      if (existsSync(archiveDir)) {
        result.push({ agentId, path: archiveDir });
      }
    }
  }
  return result.sort((a, b) => a.agentId.localeCompare(b.agentId) || a.path.localeCompare(b.path));
}

export function discoverSessionFiles(
  openclawRoot: string,
  includeArchive: boolean
): Array<{ agentId: string; filePath: string }> {
  const sessions: Array<{ agentId: string; filePath: string }> = [];
  for (const dir of sessionDirectories(openclawRoot, includeArchive)) {
    for (const entry of readdirSync(dir.path, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.endsWith(".jsonl")) continue;
      sessions.push({
        agentId: dir.agentId,
        filePath: join(dir.path, entry.name),
      });
    }
  }
  return sessions.sort(
    (a, b) => a.agentId.localeCompare(b.agentId) || a.filePath.localeCompare(b.filePath)
  );
}

function backfillContextForSession(session: ParsedSessionFile): PluginHookAgentContext {
  return {
    agentId: session.agentId,
    sessionKey: session.sessionKey,
  };
}

function deriveTargetBank(
  session: ParsedSessionFile,
  pluginConfig: PluginConfig,
  bankStrategy: BackfillCliOptions["bankStrategy"],
  fixedBank?: string
): string {
  if (bankStrategy === "agent") {
    return session.agentId;
  }
  if (bankStrategy === "fixed") {
    if (!fixedBank) {
      throw new Error("fixed bank strategy requires --fixed-bank");
    }
    return fixedBank;
  }
  return deriveBankId(backfillContextForSession(session), pluginConfig);
}

export function stableDocumentId(session: ParsedSessionFile, bankId: string): string {
  return `backfill::${bankId}::${session.agentId}::${session.sessionId}`;
}

/** Calls one OpenClaw Gateway method and returns its JSON result. */
export type GatewayCall = (method: string, params: Record<string, unknown>) => Promise<unknown>;

interface GatewaySessionRow {
  key?: string;
  agentId?: string;
  sessionId?: string;
  /** Epoch milliseconds. */
  startedAt?: number;
}

interface GatewayHistoryPage {
  messages?: Array<{ __openclaw?: { seq?: number } }>;
  hasMore?: boolean | null;
  nextOffset?: number | null;
}

// chat.history refuses a limit above 1000.
const HISTORY_PAGE = 1000;
const SESSIONS_PAGE = 200;

/**
 * Read every session through the Gateway (`sessions.list` + `chat.history`).
 *
 * OpenClaw 2026.10+ keeps transcripts in its own database (SQLite by default,
 * PostgreSQL when configured), so `sessions/*.jsonl` is empty there and the file
 * reader found nothing (#5569). These Gateway calls are OpenClaw's documented,
 * store-agnostic way to read them. History comes in the same message shape the
 * live `agent_end` hook gets, so it goes through the same transcript builder.
 */
export async function readGatewaySessions(
  call: GatewayCall,
  selectedAgents?: Set<string>
): Promise<ParsedSessionFile[]> {
  const rows: GatewaySessionRow[] = [];
  for (let offset = 0; ; ) {
    const page = (await call("sessions.list", { limit: SESSIONS_PAGE, offset })) as {
      sessions?: GatewaySessionRow[];
      hasMore?: boolean;
      nextOffset?: number | null;
    };
    rows.push(...(page.sessions ?? []));
    if (!page.hasMore || typeof page.nextOffset !== "number") break;
    offset = page.nextOffset;
  }

  const wanted = rows.filter(
    (row): row is GatewaySessionRow & { key: string; agentId: string } =>
      !!row.key && !!row.agentId && (!selectedAgents || selectedAgents.has(row.agentId))
  );
  const readOne = async (row: { key: string; agentId: string } & GatewaySessionRow) => {
    // Pages run newest first; seq puts the whole history back in order.
    const messages: Array<{ __openclaw?: { seq?: number } }> = [];
    for (let offset = 0; ; ) {
      const page = (await call("chat.history", {
        sessionKey: row.key,
        limit: HISTORY_PAGE,
        offset,
      })) as GatewayHistoryPage;
      messages.push(...(page.messages ?? []));
      if (!page.hasMore || typeof page.nextOffset !== "number") break;
      offset = page.nextOffset;
    }
    messages.sort((a, b) => (a.__openclaw?.seq ?? 0) - (b.__openclaw?.seq ?? 0));
    return {
      filePath: `gateway:${row.key}`,
      agentId: row.agentId,
      sessionId: row.sessionId || row.key,
      sessionKey: row.key,
      startedAt:
        typeof row.startedAt === "number" ? new Date(row.startedAt).toISOString() : undefined,
      messages: messages as unknown as SessionMessage[],
    };
  };
  // Each call spawns the openclaw CLI (seconds each), so read a few at once.
  // ponytail: fixed batches of 4; a worker pool only if big gateways need it.
  const sessions: ParsedSessionFile[] = [];
  for (let i = 0; i < wanted.length; i += 4) {
    sessions.push(...(await Promise.all(wanted.slice(i, i + 4).map(readOne))));
  }
  return sessions;
}

/**
 * Build the import plan. `sessions` are the ones read from the Gateway; without
 * them, the `*.jsonl` files under the OpenClaw root are read instead.
 */
export function buildBackfillPlan(
  pluginConfig: PluginConfig,
  opts: BackfillCliOptions,
  sessions?: ParsedSessionFile[]
): {
  entries: BackfillPlanEntry[];
  discoveredSessions: number;
  skippedEmpty: number;
  skippedFiltered: number;
} {
  const entries: BackfillPlanEntry[] = [];
  let discoveredSessions = 0;
  let skippedEmpty = 0;
  let skippedFiltered = 0;
  // The live hooks skip these sessions, so backfill does too: without this it
  // imported every cron, heartbeat and subagent run (#5569).
  const skipOperational = pluginConfig.skipOperationalSessions !== false;
  const ignorePatterns = compileSessionPatterns(pluginConfig.ignoreSessionPatterns ?? []);
  const candidates =
    sessions ??
    discoverSessionFiles(opts.openclawRoot, opts.includeArchive)
      .filter((c) => !opts.selectedAgents || opts.selectedAgents.has(c.agentId))
      .map((c) => parseSessionFile(c.filePath, c.agentId));

  for (const parsed of candidates) {
    discoveredSessions += 1;
    const key = parsed.sessionKey;
    if (
      key &&
      ((skipOperational && isOperationalSessionKey(key)) ||
        matchesSessionPattern(key, ignorePatterns))
    ) {
      skippedFiltered += 1;
      continue;
    }
    const retention = prepareRetentionTranscript(parsed.messages, pluginConfig, true);
    if (!retention) {
      skippedEmpty += 1;
      continue;
    }
    const bankId = deriveTargetBank(parsed, pluginConfig, opts.bankStrategy, opts.fixedBank);
    entries.push({
      filePath: parsed.filePath,
      agentId: parsed.agentId,
      sessionId: parsed.sessionId,
      startedAt: parsed.startedAt,
      bankId,
      documentId: stableDocumentId(parsed, bankId),
      transcript: retention.transcript,
      messageCount: retention.messageCount,
    });
    if (opts.limit && entries.length >= opts.limit) {
      break;
    }
  }

  return { entries, discoveredSessions, skippedEmpty, skippedFiltered };
}

export function loadCheckpoint(checkpointPath: string): BackfillCheckpoint {
  if (!existsSync(checkpointPath)) {
    return { version: 1, entries: {} };
  }
  const raw = JSON.parse(readFileSync(checkpointPath, "utf8")) as RawBackfillCheckpoint;
  if (raw.version !== 1 || !raw.entries || typeof raw.entries !== "object") {
    return { version: 1, entries: {} };
  }
  return {
    version: 1,
    entries: Object.fromEntries(
      Object.entries(raw.entries).map(([key, entry]) => [
        key,
        {
          ...entry,
          status: entry.status === "queued" ? "enqueued" : entry.status,
        },
      ])
    ) as Record<string, BackfillCheckpointEntry>,
  };
}

export function saveCheckpoint(checkpointPath: string, checkpoint: BackfillCheckpoint): void {
  mkdirSync(dirname(checkpointPath), { recursive: true });
  writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2) + "\n", "utf8");
}

export function checkpointKey(entry: Pick<BackfillPlanEntry, "bankId" | "documentId">): string {
  return `${entry.bankId}::${entry.documentId}`;
}
